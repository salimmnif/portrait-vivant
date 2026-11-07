"""Tests de l'interface, sans vraie webcam. Lancement : python -m pytest tests

Un vrai serveur tourne sur cet ordinateur, et une fausse caméra remplace la webcam.
"""
import json
import threading
import time
from http.client import HTTPConnection

import cv2
import numpy as np
import pytest

import interface
from simulation import CameraFactice, portraits_de_test


class CameraDeTest(CameraFactice):
    def release(self):
        pass


def attendre(condition, delai_s=5.0):
    fin = time.perf_counter() + delai_s
    while time.perf_counter() < fin:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def serveur(monkeypatch):
    image = portraits_de_test()[0].image
    monkeypatch.setattr(interface, "ouvrir_camera", lambda: CameraDeTest([image]))
    moteur = interface.Moteur({f"p{i}": p for i, p in enumerate(portraits_de_test()[:2])})
    moteur.demarrer()
    http = interface.creer_serveur(moteur, port=0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    yield moteur, http.server_address[1]
    http.shutdown()
    http.server_close()
    moteur.arreter()


def requete(port, methode, chemin, corps=None, entetes=None):
    connexion = HTTPConnection("127.0.0.1", port, timeout=10)
    if isinstance(corps, dict):
        corps = json.dumps(corps).encode()
    connexion.request(methode, chemin, corps, entetes or {})
    reponse = connexion.getresponse()
    contenu = reponse.read()
    connexion.close()
    return reponse.status, contenu


def etat_de(port):
    return json.loads(requete(port, "GET", "/etat")[1])


def jpeg_de(image):
    return cv2.imencode(".jpg", image)[1].tobytes()


def test_la_page_est_servie(serveur):
    _, port = serveur
    code, contenu = requete(port, "GET", "/")
    assert code == 200 and "Portrait vivant".encode() in contenu


def test_etat_initial(serveur):
    _, port = serveur
    etat = etat_de(port)
    assert etat["camera_active"] is False and etat["neutre"] is False
    assert etat["portraits"] == ["p0", "p1"] and etat["intensite"] == 1.0


def test_miniature(serveur):
    _, port = serveur
    code, contenu = requete(port, "GET", "/miniatures/1")
    assert code == 200 and contenu[:2] == b"\xff\xd8"
    assert requete(port, "GET", "/miniatures/99")[0] == 404
    assert requete(port, "GET", "/miniatures/abc")[0] == 404


def test_le_flux_portrait_envoie_du_jpeg_sans_camera(serveur):
    _, port = serveur
    connexion = HTTPConnection("127.0.0.1", port, timeout=10)
    connexion.request("GET", "/flux/portrait")
    reponse = connexion.getresponse()
    assert "multipart/x-mixed-replace" in reponse.getheader("Content-Type")
    debut = reponse.read(4000)
    connexion.close()
    assert b"image/jpeg" in debut and b"\xff\xd8" in debut


def test_parcours_camera_neutre_puis_arret(serveur):
    moteur, port = serveur
    assert requete(port, "POST", "/neutre")[0] == 409  # pas encore de visage

    assert requete(port, "POST", "/camera", {"active": True})[0] == 200
    assert attendre(lambda: etat_de(port)["camera_active"] and etat_de(port)["visage"])
    assert attendre(lambda: moteur.images["webcam"] is not None)  # publiée juste après
    assert moteur.images["webcam"][:2] == b"\xff\xd8"

    assert requete(port, "POST", "/neutre")[0] == 200
    assert etat_de(port)["neutre"] is True
    assert attendre(lambda: etat_de(port)["fps"] > 0)

    assert requete(port, "POST", "/camera", {"active": False})[0] == 200
    assert attendre(lambda: not etat_de(port)["camera_active"])
    etat = etat_de(port)
    assert etat["neutre"] is False and etat["fps"] == 0
    assert attendre(lambda: moteur.images["webcam"] is None)


def test_le_flux_webcam_s_arrete_quand_la_camera_est_eteinte(serveur):
    _, port = serveur
    connexion = HTTPConnection("127.0.0.1", port, timeout=10)
    connexion.request("GET", "/flux/webcam")
    assert connexion.getresponse().read() == b"--image-suivante\r\n"
    connexion.close()


def test_camera_introuvable_previent_la_page(serveur, monkeypatch):
    moteur, port = serveur
    monkeypatch.setattr(interface, "ouvrir_camera", lambda: None)
    requete(port, "POST", "/camera", {"active": True})
    assert attendre(lambda: moteur.erreur is not None)
    etat = etat_de(port)
    assert "Webcam introuvable" in etat["erreur"] and etat["camera_demandee"] is False
    assert etat_de(port)["erreur"] is None  # le message n'est donné qu'une fois


class CameraLente(CameraDeTest):
    """Comme une vraie webcam : une image toutes les 30 ms."""

    def read(self):
        time.sleep(0.03)
        return super().read()


def test_la_page_reste_reactive_quand_la_camera_tourne(serveur, monkeypatch):
    moteur, port = serveur
    image = portraits_de_test()[0].image
    monkeypatch.setattr(interface, "ouvrir_camera", lambda: CameraLente([image]))
    requete(port, "POST", "/camera", {"active": True})
    assert attendre(lambda: etat_de(port)["visage"])
    requete(port, "POST", "/neutre")
    durees = []
    for chemin in ["/etat", "/miniatures/0", "/miniatures/1"] * 8:
        debut = time.perf_counter()
        assert requete(port, "GET", chemin)[0] == 200
        durees.append(time.perf_counter() - debut)
    assert np.median(durees) < 0.1 and max(durees) < 0.5


def test_changer_de_portrait_quand_la_camera_tourne(serveur):
    moteur, port = serveur
    requete(port, "POST", "/camera", {"active": True})
    assert attendre(lambda: etat_de(port)["visage"])
    requete(port, "POST", "/neutre")
    assert requete(port, "POST", "/portraits/1")[0] == 200
    time.sleep(0.2)
    assert moteur.etat.rendu.shape == moteur.etat.portraits[1].image.shape
    assert etat_de(port)["indice"] == 1


def test_flux_avec_parametre_de_relance(serveur):
    _, port = serveur
    connexion = HTTPConnection("127.0.0.1", port, timeout=10)
    connexion.request("GET", "/flux/portrait?t=123")
    reponse = connexion.getresponse()
    assert reponse.status == 200 and b"image/jpeg" in reponse.read(2000)
    connexion.close()


def test_camera_relancee_apres_un_arret(serveur):
    _, port = serveur
    for _ in range(2):
        requete(port, "POST", "/camera", {"active": True})
        assert attendre(lambda: etat_de(port)["camera_active"])
        requete(port, "POST", "/camera", {"active": False})
        assert attendre(lambda: not etat_de(port)["camera_active"])


def test_ajouter_une_image_avec_un_visage(serveur):
    _, port = serveur
    code, _ = requete(
        port, "POST", "/portraits", jpeg_de(portraits_de_test()[0].image),
        {"X-Nom-Fichier": "ma%20photo%20%C3%A9t%C3%A9.jpg"},
    )
    assert code == 200
    etat = etat_de(port)
    assert etat["portraits"][-1] == "ma photo été"
    assert etat["indice"] == len(etat["portraits"]) - 1


def test_le_nom_de_fichier_ne_peut_pas_sortir_de_son_nom(serveur):
    _, port = serveur
    requete(
        port, "POST", "/portraits", jpeg_de(portraits_de_test()[0].image),
        {"X-Nom-Fichier": "..%2F..%2Fsecret.jpg"},
    )
    assert etat_de(port)["portraits"][-1] == "secret"


@pytest.mark.parametrize("corps", [
    jpeg_de(np.full((300, 300, 3), 120, np.uint8)),  # pas de visage
    b"ceci n'est pas une image",
    b"",
])
def test_image_refusee_sans_changer_le_portrait(serveur, corps):
    _, port = serveur
    code, contenu = requete(port, "POST", "/portraits", corps)
    assert code == 422 and "Aucun visage" in json.loads(contenu)["message"]
    assert etat_de(port)["indice"] == 0 and len(etat_de(port)["portraits"]) == 2


def test_image_trop_lourde_refusee(serveur):
    _, port = serveur
    connexion = HTTPConnection("127.0.0.1", port, timeout=10)
    connexion.putrequest("POST", "/portraits")
    connexion.putheader("Content-Length", str(interface.TAILLE_IMAGE_MAX + 1))
    connexion.endheaders()
    reponse = connexion.getresponse()
    assert reponse.status == 413
    connexion.close()


def test_choisir_un_portrait(serveur):
    _, port = serveur
    assert requete(port, "POST", "/portraits/1")[0] == 200
    assert etat_de(port)["indice"] == 1
    assert requete(port, "POST", "/portraits/9")[0] == 404
    assert requete(port, "POST", "/portraits/abc")[0] == 404


@pytest.mark.parametrize("valeur, attendue", [(1.5, 1.5), (9.0, 2.0), (-3.0, 0.0)])
def test_intensite_bornee(serveur, valeur, attendue):
    _, port = serveur
    assert requete(port, "POST", "/intensite", {"valeur": valeur})[0] == 200
    assert etat_de(port)["intensite"] == attendue


def test_debug(serveur):
    _, port = serveur
    requete(port, "POST", "/debug", {"actif": True})
    assert etat_de(port)["debug"] is True


@pytest.mark.parametrize("chemin, corps", [
    ("/intensite", {"valeur": "beaucoup"}),
    ("/intensite", {}),
    ("/camera", {}),
    ("/camera", b"pas du json"),
])
def test_reglage_invalide_refuse(serveur, chemin, corps):
    _, port = serveur
    assert requete(port, "POST", chemin, corps)[0] == 400


def test_chemin_inconnu(serveur):
    _, port = serveur
    assert requete(port, "GET", "/secret")[0] == 404
    assert requete(port, "POST", "/secret")[0] == 404


@pytest.mark.parametrize("entetes", [
    {"Host": "site-malveillant.example"},
    {"Origin": "https://site-malveillant.example"},
])
def test_un_autre_site_ne_peut_pas_piloter_la_webcam(serveur, entetes):
    moteur, port = serveur
    code, _ = requete(port, "POST", "/camera", {"active": True}, entetes)
    assert code == 403 and moteur.camera_demandee is False
    assert requete(port, "GET", "/etat", None, entetes)[0] == 403
    assert requete(port, "GET", "/flux/webcam", None, entetes)[0] == 403


def test_le_serveur_n_ecoute_que_sur_cet_ordinateur():
    assert interface.ADRESSE == "127.0.0.1"
