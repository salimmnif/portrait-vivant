"""Tests des mesures, sans webcam ni fenêtre. Lancement : python -m pytest tests"""
import cv2
import numpy as np
import pytest

import mesures
from reperes import NOMBRE_REPERES, DetecteurReperes
from simulation import CameraFactice, portraits_de_test
from test_recalage import tourner_la_tete_3d


@pytest.fixture(scope="module")
def visage():
    with DetecteurReperes() as detecteur:
        return detecteur.detecter_3d(portraits_de_test()[0].image)


def serie_bruitee(visage, bruit=1.0, nb_images=300):
    aleatoire = np.random.default_rng(0)
    return (visage + aleatoire.normal(0, bruit, (nb_images, NOMBRE_REPERES, 3))).astype(np.float32)


def test_tremblement_d_un_bruit_connu(visage):
    # La différence de deux bruits indépendants a un écart-type de √2 fois le bruit.
    assert mesures.tremblement(serie_bruitee(visage, 2.0)) == pytest.approx(2.0 * np.sqrt(2), rel=0.05)


def test_tremblement_nul_si_rien_ne_bouge(visage):
    assert mesures.tremblement(np.repeat(visage[np.newaxis], 20, axis=0)) == pytest.approx(0)


def test_le_lissage_reduit_le_tremblement_de_plus_en_plus(visage):
    lignes = mesures.comparer_lissages(serie_bruitee(visage))
    valeurs = [ligne["tremblement_px"] for ligne in lignes]
    assert [ligne["coefficient"] for ligne in lignes] == sorted(
        (ligne["coefficient"] for ligne in lignes), reverse=True
    )
    assert all(apres < avant for avant, apres in zip(valeurs, valeurs[1:]))


def test_coefficient_1_laisse_la_serie_intacte(visage):
    serie = serie_bruitee(visage, nb_images=10)
    assert np.allclose(mesures.lisser_serie(serie, 1.0), serie)


def test_recalages_comparees_sur_une_tete_qui_tourne(visage):
    serie = np.array([tourner_la_tete_3d(visage, lacet=a) for a in (5.0, 15.0, 25.0)])
    lignes = mesures.comparer_recalages(serie, visage)
    assert lignes[2]["lacet_deg"] == pytest.approx(25, abs=0.5)
    for ligne in lignes:
        assert ligne["recalage_3d"] < ligne["recalage_2d"] < ligne["sans"]


def ligne_de_mesure(sans, en_2d, en_3d, lacet=0.0, tangage=0.0, roulis=0.0):
    return {
        "sans": sans, "recalage_2d": en_2d, "recalage_3d": en_3d,
        "lacet_deg": lacet, "tangage_deg": tangage, "roulis_deg": roulis,
    }


@pytest.mark.parametrize("angles, attendu", [
    ({"lacet": 2.0, "tangage": -3.0, "roulis": 4.0}, "De face (moins de 5°)"),
    ({"lacet": -20.0, "tangage": 6.0, "roulis": 3.0}, "Tournée"),
    ({"lacet": 2.0, "tangage": -30.0, "roulis": 5.0}, "Levée ou baissée"),
    ({"lacet": 6.0, "tangage": 3.0, "roulis": -15.0}, "Penchée"),
])
def test_classement_selon_la_rotation_dominante(angles, attendu):
    assert mesures.classer_mouvement(ligne_de_mesure(1, 1, 1, **angles)) == attendu


def test_resume_par_mouvement_avec_medianes():
    lignes = [
        ligne_de_mesure(9.0, 2.0, 1.0, lacet=20.0),
        ligne_de_mesure(4.0, 1.0, np.nan, lacet=25.0),
        ligne_de_mesure(6.0, 500.0, 3.0, lacet=30.0),  # une image décroche
        ligne_de_mesure(1.0, 1.0, 1.0),
    ]
    resume = mesures.resumer_recalages(lignes)
    assert [ligne["mouvement"] for ligne in resume] == ["De face (moins de 5°)", "Tournée"]
    tournee = resume[1]
    assert tournee["nb_images"] == 3
    assert tournee["sans"] == 6.0
    assert tournee["recalage_2d"] == 2.0  # la médiane ignore le pic à 500
    assert tournee["recalage_3d"] == 2.0  # et les valeurs manquantes


def test_resume_des_temps():
    resume = mesures.resumer_temps({"Détection": [0.01, 0.03], "Transfert": []}, 2)
    assert len(resume) == 1
    assert resume[0]["moyenne_ms"] == pytest.approx(20)
    assert resume[0]["ecart_type_ms"] == pytest.approx(10)
    assert resume[0]["nb_images"] == 2


def test_detection_sur_un_dossier(tmp_path):
    cv2.imwrite(str(tmp_path / "visage.jpg"), portraits_de_test()[0].image)
    cv2.imwrite(str(tmp_path / "vide.jpg"), np.full((300, 300, 3), 120, np.uint8))
    (tmp_path / "notes.txt").write_text("ignoré")
    lignes = mesures.mesurer_detection(tmp_path)
    assert {ligne["portrait"]: ligne["visage_detecte"] for ligne in lignes} == {
        "visage.jpg": "oui", "vide.jpg": "non",
    }


def test_csv_aller_retour(tmp_path):
    chemin = tmp_path / "sous_dossier" / "essai.csv"
    mesures.ecrire_csv(chemin, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    assert mesures.lire_csv(chemin) == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def preparer_sans_fenetre(monkeypatch, touches=None):
    """Remplace la fenêtre et raccourcit les durées pour que les tests restent rapides."""
    touches = iter(touches) if touches else None
    monkeypatch.setattr(mesures.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(mesures.cv2, "waitKey", lambda _: next(touches) if touches else 255)
    monkeypatch.setattr(mesures.cv2, "getWindowProperty", lambda *_: 1.0)
    monkeypatch.setattr(mesures, "DUREE_PREPARATION_S", 0.2)
    monkeypatch.setattr(mesures, "DUREE_LISSAGE_S", 0.4)
    monkeypatch.setattr(mesures, "DUREE_PHASE_S", 0.15)
    monkeypatch.setattr(mesures, "DUREE_PAUSE_S", 0.05)


def camera_de_test():
    return CameraFactice([portraits_de_test()[0].image])


def test_mesure_lissage_complete(monkeypatch, tmp_path):
    preparer_sans_fenetre(monkeypatch)
    with DetecteurReperes(mode_video=True) as detecteur:
        mesures.mesure_lissage(camera_de_test(), detecteur, tmp_path)
    assert (tmp_path / "lissage.png").stat().st_size > 1000
    lignes = mesures.lire_csv(tmp_path / "lissage.csv")
    assert len(lignes) == len(mesures.COEFFICIENTS_LISSAGE)
    assert int(lignes[0]["nb_images"]) > 2


def test_mesure_recalage_complete(monkeypatch, tmp_path):
    preparer_sans_fenetre(monkeypatch)
    with DetecteurReperes(mode_video=True) as detecteur:
        mesures.mesure_recalage(camera_de_test(), detecteur, tmp_path)
    assert (tmp_path / "recalage.png").stat().st_size > 1000
    resume = mesures.lire_csv(tmp_path / "recalage_resume.csv")
    assert resume[0]["mouvement"] == "De face (moins de 5°)"  # l'image ne bouge pas
    detail = mesures.lire_csv(tmp_path / "recalage_detail.csv")
    assert {ligne["phase"] for ligne in detail} == {phase for phase, _ in mesures.PHASES_RECALAGE}


def test_mesure_temps_complete(monkeypatch, tmp_path):
    preparer_sans_fenetre(monkeypatch)
    with DetecteurReperes(mode_video=True) as detecteur:
        mesures.mesure_temps(camera_de_test(), detecteur, portraits_de_test(), tmp_path, nb_images=5)
    assert (tmp_path / "temps.png").stat().st_size > 1000
    lignes = {ligne["etape"]: ligne for ligne in mesures.lire_csv(tmp_path / "temps.csv")}
    assert int(lignes["Boucle complète"]["nb_images"]) == 5
    assert float(lignes["Détection"]["moyenne_ms"]) > 0
    assert "Déformation" in lignes


def test_sans_visage_les_etapes_de_calcul_ne_sont_pas_comptees(monkeypatch, tmp_path):
    preparer_sans_fenetre(monkeypatch)
    images = [portraits_de_test()[0].image] * 8 + [np.zeros((600, 500, 3), np.uint8)]
    camera = CameraFactice(images)
    monkeypatch.setattr(mesures, "capturer_neutre", lambda *_: _neutre())
    with DetecteurReperes(mode_video=True) as detecteur:
        mesures.mesure_temps(camera, detecteur, portraits_de_test(), tmp_path, nb_images=12)
    lignes = {ligne["etape"]: ligne for ligne in mesures.lire_csv(tmp_path / "temps.csv")}
    assert int(lignes["Détection"]["nb_images"]) == 12
    assert int(lignes["Déformation"]["nb_images"]) < 12


def _neutre():
    with DetecteurReperes() as detecteur:
        return detecteur.detecter_3d(portraits_de_test()[0].image)


def test_touche_q_interrompt_la_mesure(monkeypatch):
    preparer_sans_fenetre(monkeypatch, touches=[ord("q")])
    with DetecteurReperes(mode_video=True) as detecteur:
        with pytest.raises(mesures.Interruption):
            mesures.capturer_neutre(camera_de_test(), detecteur)


def test_retracer_les_graphiques_depuis_les_csv(tmp_path, capsys):
    mesures.ecrire_csv(tmp_path / "lissage.csv", [
        {"coefficient": 1.0, "tremblement_px": 0.8}, {"coefficient": 0.5, "tremblement_px": 0.4},
    ])
    mesures.retracer_graphiques(tmp_path)
    assert (tmp_path / "lissage.png").exists()
    assert not (tmp_path / "temps.png").exists()
