"""Interface locale : une page dans le navigateur pour animer un portrait.

Utilisation, depuis le dossier portrait_vivant : python interface.py

La webcam et tous les calculs restent dans ce programme Python. Le serveur n'écoute que
sur cet ordinateur (127.0.0.1) : aucune image ne part sur Internet.
"""
import copy
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

from deformation import charger_portraits, dessiner_triangles, portrait_depuis_image
from main import (
    COEFFICIENT_LISSAGE, INDICE_CAMERA, INTENSITE_MAX, Etat, animer_le_portrait,
    mettre_a_jour_fps,
)
from reperes import DetecteurReperes, LissageExponentiel, dessiner_reperes

CHEMIN_PAGE = Path(__file__).resolve().parent / "interface.html"
ADRESSE = "127.0.0.1"
PORT_PREFERE = 8765
QUALITE_JPEG = 85
HAUTEUR_MINIATURE = 120
TAILLE_IMAGE_MAX = 30 * 1024 * 1024  # 30 Mo
PAUSE_SANS_CAMERA_S = 0.05
ATTENTE_FLUX_S = 0.5  # sans nouvelle image, on renvoie la dernière à ce rythme
LIMITE_BOUNDARY = b"image-suivante"


def encoder_jpeg(image):
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, QUALITE_JPEG])[1].tobytes()


def ouvrir_camera():
    """Ouvre la caméra réglée, sinon essaie la caméra 0. Renvoie None en cas d'échec."""
    for indice in dict.fromkeys((INDICE_CAMERA, 0)):
        camera = cv2.VideoCapture(indice)
        if camera.isOpened() and camera.read()[0]:
            return camera
        camera.release()
    return None


class Moteur:
    """Fait tourner la webcam et l'animation dans un fil d'exécution séparé du serveur."""

    def __init__(self, portraits_nommes):
        self.noms = list(portraits_nommes)
        self.etat = Etat(list(portraits_nommes.values()), afficher_debug=False)
        self.verrou = threading.RLock()
        self.nouvelle_image = threading.Condition()
        self.images = {"webcam": None, "portrait": None}
        self.versions = {"webcam": 0, "portrait": 0}
        self.camera_demandee = False
        self.camera_active = False
        self.visage_vu = False
        self.derniers_reperes = None
        self.fps = 0.0
        self.erreur = None
        self.arret = threading.Event()
        self.fil = threading.Thread(target=self.tourner, daemon=True)

    def demarrer(self):
        self.fil.start()

    def arreter(self):
        self.camera_demandee = False
        self.arret.set()
        with self.nouvelle_image:
            self.nouvelle_image.notify_all()
        self.fil.join(timeout=5)

    # ---------- Images envoyées au navigateur ----------

    def publier(self, nom, octets):
        with self.nouvelle_image:
            self.images[nom] = octets
            self.versions[nom] += 1
            self.nouvelle_image.notify_all()

    def publier_portrait(self):
        """Publie le portrait dans son état actuel, avec ses triangles en mode debug."""
        with self.verrou:
            image, cible = self.etat.rendu, self.etat.cible
            triangles = self.etat.portrait.triangles if self.etat.afficher_debug else None
        if triangles is not None:
            image = image.copy()
            dessiner_triangles(image, cible, triangles)
        self.publier("portrait", encoder_jpeg(image))

    # ---------- Boucle principale ----------

    def tourner(self):
        self.publier_portrait()
        while not self.arret.is_set():
            if self.camera_demandee:
                self.session_camera()
            else:
                self.arret.wait(PAUSE_SANS_CAMERA_S)

    def session_camera(self):
        """Ouvre la caméra, anime le portrait, puis la libère : son voyant s'éteint."""
        camera = ouvrir_camera()
        if camera is None:
            self.erreur = "Webcam introuvable : vérifiez qu'aucune autre application ne l'utilise."
            self.camera_demandee = False
            return
        try:
            with DetecteurReperes(mode_video=True) as detecteur:
                self.boucle_camera(camera, detecteur)
        finally:
            camera.release()
            with self.verrou:
                self.camera_active = False
                self.visage_vu = False
                self.derniers_reperes = None
                self.fps = 0.0
                self.etat.neutre = None
                self.etat.lissage_pose.appliquer(None)
                self.etat.remettre_le_portrait_au_repos()
            self.publier("webcam", None)
            self.publier_portrait()

    def boucle_camera(self, camera, detecteur):
        lissage = LissageExponentiel(COEFFICIENT_LISSAGE)
        self.camera_active = True
        fps, temps_precedent = None, time.perf_counter()
        while self.camera_demandee and not self.arret.is_set():
            lecture_ok, image = camera.read()
            if not lecture_ok:
                self.erreur = "La webcam a cessé de répondre."
                self.camera_demandee = False
                return
            # Effet miroir : le côté droit de l'écran correspond à votre côté droit.
            image = cv2.flip(image, 1)
            reperes = lissage.appliquer(detecteur.detecter_3d(image))
            self.animer_hors_verrou(reperes)
            if reperes is not None and self.etat.afficher_debug:
                dessiner_reperes(image, reperes[:, :2])
            self.publier("webcam", encoder_jpeg(image))
            self.publier_portrait()
            fps, temps_precedent = mettre_a_jour_fps(fps, temps_precedent)
            self.fps = fps

    def animer_hors_verrou(self, reperes):
        """Calcule l'image sur une copie de l'état, sans bloquer la page.

        Le calcul prend presque tout le temps de chaque image : s'il gardait le verrou,
        les demandes de la page attendraient trop longtemps et elle semblerait figée.
        """
        with self.verrou:
            self.visage_vu = reperes is not None
            self.derniers_reperes = reperes
            copie = copy.copy(self.etat)
        animer_le_portrait(copie, reperes)
        with self.verrou:
            # Si le portrait a changé pendant le calcul, ce résultat ne vaut plus rien.
            if self.etat.indice_portrait == copie.indice_portrait:
                self.etat.rendu, self.etat.cible = copie.rendu, copie.cible

    # ---------- Actions demandées par la page ----------

    def enregistrer_neutre(self):
        """Enregistre le visage neutre. Renvoie False si aucun visage n'est visible."""
        with self.verrou:
            if self.derniers_reperes is None:
                return False
            self.etat.neutre = self.derniers_reperes.copy()
            return True

    def choisir_portrait(self, indice):
        with self.verrou:
            if not 0 <= indice < len(self.etat.portraits):
                return False
            self.etat.indice_portrait = indice
            self.etat.remettre_le_portrait_au_repos()
        self.publier_portrait()
        return True

    def ajouter_image(self, octets, nom):
        """Ajoute une image envoyée par la page. Renvoie False si aucun visage n'y est trouvé."""
        tampon = np.frombuffer(octets, np.uint8)
        image = cv2.imdecode(tampon, cv2.IMREAD_COLOR) if tampon.size else None
        portrait = portrait_depuis_image(image)
        if portrait is None:
            return False
        with self.verrou:
            self.noms.append(nom)
            self.etat.ajouter_portrait(portrait)
        self.publier_portrait()
        return True

    def regler_intensite(self, valeur):
        with self.verrou:
            self.etat.intensite = float(np.clip(valeur, 0.0, INTENSITE_MAX))

    def regler_debug(self, actif):
        with self.verrou:
            self.etat.afficher_debug = bool(actif)
        self.publier_portrait()

    def miniature(self, indice):
        with self.verrou:
            if not 0 <= indice < len(self.etat.portraits):
                return None
            image = self.etat.portraits[indice].image
        largeur = round(image.shape[1] * HAUTEUR_MINIATURE / image.shape[0])
        return encoder_jpeg(cv2.resize(image, (largeur, HAUTEUR_MINIATURE), interpolation=cv2.INTER_AREA))

    def etat_public(self):
        """Résume l'état pour la page."""
        with self.verrou:
            erreur, self.erreur = self.erreur, None
            return {
                "camera_demandee": self.camera_demandee,
                "camera_active": self.camera_active,
                "visage": self.visage_vu,
                "neutre": self.etat.neutre is not None,
                "fps": round(self.fps, 1),
                "intensite": self.etat.intensite,
                "intensite_max": INTENSITE_MAX,
                "debug": self.etat.afficher_debug,
                "indice": self.etat.indice_portrait,
                "portraits": self.noms,
                "erreur": erreur,
            }


class Gestionnaire(BaseHTTPRequestHandler):
    """Répond à la page. Le moteur est fourni par creer_serveur."""

    moteur = None

    def log_message(self, *_):
        pass  # le terminal reste lisible

    # ---------- Protection : seule notre page peut piloter la webcam ----------

    def origine_autorisee(self):
        """Refuse les autres sites web, qui peuvent viser les serveurs de cet ordinateur."""
        port = self.server.server_address[1]
        hotes = {f"{ADRESSE}:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hotes:
            return False
        origine = self.headers.get("Origin")
        return origine is None or origine in {f"http://{hote}" for hote in hotes}

    # ---------- Réponses ----------

    def repondre(self, code, contenu, type_contenu):
        self.send_response(code)
        self.send_header("Content-Type", type_contenu)
        self.send_header("Content-Length", str(len(contenu)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(contenu)

    def repondre_json(self, code, donnees):
        self.repondre(code, json.dumps(donnees).encode("utf-8"), "application/json; charset=utf-8")

    def refuser(self, code, message):
        self.repondre_json(code, {"ok": False, "message": message})

    def do_GET(self):
        chemin = self.path.split("?", 1)[0]  # la page ajoute ?t=... pour relancer un flux
        if not self.origine_autorisee():
            self.refuser(403, "Origine refusée.")
        elif chemin == "/":
            self.repondre(200, CHEMIN_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif chemin == "/etat":
            self.repondre_json(200, self.moteur.etat_public())
        elif chemin in ("/flux/webcam", "/flux/portrait"):
            self.envoyer_flux(chemin.rsplit("/", 1)[1])
        elif chemin.startswith("/miniatures/"):
            self.envoyer_miniature()
        else:
            self.refuser(404, "Page introuvable.")

    def envoyer_miniature(self):
        numero = self.path.rsplit("/", 1)[1]
        octets = self.moteur.miniature(int(numero)) if numero.isdigit() else None
        if octets is None:
            self.refuser(404, "Portrait introuvable.")
        else:
            self.repondre(200, octets, "image/jpeg")

    def envoyer_flux(self, nom):
        """Envoie une suite d'images JPEG, que le navigateur affiche comme une vidéo."""
        moteur = self.moteur
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={LIMITE_BOUNDARY.decode()}"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        separateur = b"--" + LIMITE_BOUNDARY + b"\r\n"
        version = -1
        try:
            self.wfile.write(separateur)
            while not moteur.arret.is_set():
                with moteur.nouvelle_image:
                    moteur.nouvelle_image.wait_for(
                        lambda: moteur.versions[nom] != version or moteur.arret.is_set(),
                        timeout=ATTENTE_FLUX_S,
                    )
                    image, version = moteur.images[nom], moteur.versions[nom]
                if image is None:
                    if nom == "webcam" and not moteur.camera_demandee:
                        return
                    continue
                entete = f"Content-Type: image/jpeg\r\nContent-Length: {len(image)}\r\n\r\n"
                self.wfile.write(entete.encode() + image + b"\r\n" + separateur)
        except (BrokenPipeError, ConnectionError):
            pass  # l'onglet a été fermé

    def lire_corps(self):
        """Lit le corps de la requête. Renvoie None s'il est trop gros."""
        taille = int(self.headers.get("Content-Length", 0) or 0)
        if taille > TAILLE_IMAGE_MAX:
            return None
        return self.rfile.read(taille)

    def do_POST(self):
        # Le corps est lu avant toute réponse : sous Windows, refuser sans le lire peut
        # couper la connexion avant que la réponse n'arrive.
        corps = self.lire_corps()
        if corps is None:
            self.close_connection = True
            self.refuser(413, "Image trop lourde (30 Mo au maximum).")
        elif not self.origine_autorisee():
            self.refuser(403, "Origine refusée.")
        elif self.path == "/portraits":
            self.recevoir_image(corps)
        elif self.path.startswith("/portraits/"):
            numero = self.path.rsplit("/", 1)[1]
            reussi = numero.isdigit() and self.moteur.choisir_portrait(int(numero))
            self.repondre_json(200 if reussi else 404, {"ok": bool(reussi)})
        elif self.path == "/neutre":
            self.recevoir_neutre()
        elif self.path in ("/camera", "/intensite", "/debug"):
            self.recevoir_reglage(corps)
        else:
            self.refuser(404, "Action inconnue.")

    def recevoir_image(self, corps):
        nom = Path(unquote(self.headers.get("X-Nom-Fichier", "image"))).stem[:60] or "image"
        if self.moteur.ajouter_image(corps, nom):
            self.repondre_json(200, {"ok": True})
        else:
            self.refuser(422, "Aucun visage trouvé dans cette image : essayez un visage de face et net.")

    def recevoir_neutre(self):
        if self.moteur.enregistrer_neutre():
            self.repondre_json(200, {"ok": True})
        else:
            self.refuser(409, "Aucun visage visible : placez-vous face à la caméra.")

    def recevoir_reglage(self, corps):
        try:
            donnees = json.loads(corps)
            if self.path == "/camera":
                self.moteur.camera_demandee = bool(donnees["active"])
            elif self.path == "/intensite":
                self.moteur.regler_intensite(float(donnees["valeur"]))
            else:
                self.moteur.regler_debug(donnees["actif"])
        except (ValueError, KeyError, TypeError):
            self.refuser(400, "Réglage invalide.")
            return
        self.repondre_json(200, {"ok": True})


def creer_serveur(moteur, port=PORT_PREFERE):
    """Crée le serveur local. Si le port est déjà pris, le système en choisit un autre."""
    gestionnaire = type("GestionnaireDuMoteur", (Gestionnaire,), {"moteur": moteur})
    try:
        serveur = ThreadingHTTPServer((ADRESSE, port), gestionnaire)
    except OSError:
        serveur = ThreadingHTTPServer((ADRESSE, 0), gestionnaire)
    serveur.daemon_threads = True
    return serveur


def main():
    portraits = charger_portraits()
    if not portraits:
        sys.exit("Aucun portrait utilisable dans le dossier portraits/.")
    moteur = Moteur(portraits)
    moteur.demarrer()
    serveur = creer_serveur(moteur)
    adresse = f"http://{ADRESSE}:{serveur.server_address[1]}/"
    print(f"Interface ouverte : {adresse}\nCtrl+C dans ce terminal pour quitter.")
    webbrowser.open(adresse)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\nFermeture.")
    finally:
        serveur.server_close()
        moteur.arreter()


if __name__ == "__main__":
    main()
