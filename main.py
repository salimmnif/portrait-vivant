"""Boucle principale : webcam, détection, portrait animé, affichage et touches.

Touches : N (visage neutre), O (choisir une image), P (portrait suivant),
+ et - (intensité), D (debug), Q (quitter).
"""
import sys
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from deformation import (
    animer_portrait, charger_portraits, charger_un_portrait, dessiner_triangles,
)
from recalage import calculer_grimace_3d, norme_moyenne
from reperes import DetecteurReperes, LissageExponentiel, dessiner_reperes
from transfert import mesurer_regard, ouverture_bouche_mesuree, ouverture_relative, transferer

INDICE_CAMERA = 1
NOM_FENETRE = "Portrait vivant"
COEFFICIENT_FPS = 0.9  # part de l'ancienne valeur dans la moyenne des FPS
COEFFICIENT_LISSAGE = 0.5  # part de la nouvelle mesure ; plus petit = plus stable
COEFFICIENT_LISSAGE_POSE = 0.25  # la rotation déplace tout le visage : lissage plus fort
ATTENTE_TOUCHE_MS = 1
PAS_INTENSITE, INTENSITE_MAX = 0.1, 2.0
ROUGE, VERT, JAUNE = (0, 0, 255), (0, 255, 0), (0, 255, 255)  # couleurs en BGR
AUCUNE_TOUCHE = 255  # valeur de cv2.waitKey quand rien n'est appuyé
TYPES_IMAGES = (("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tous les fichiers", "*.*"))


@dataclass
class Etat:
    """Ce que la boucle garde d'une image à l'autre."""

    portraits: list
    indice_portrait: int = 0
    neutre: np.ndarray = None
    intensite: float = 1.0
    afficher_debug: bool = True
    message: tuple = None  # (texte, couleur) affiché jusqu'à la prochaine touche
    lissage_pose: LissageExponentiel = field(
        default_factory=lambda: LissageExponentiel(COEFFICIENT_LISSAGE_POSE)
    )
    rendu: np.ndarray = field(init=False)  # dernier état du portrait animé
    cible: np.ndarray = field(init=False)  # dernière position de ses points

    def __post_init__(self):
        self.portraits = list(self.portraits)
        self.remettre_le_portrait_au_repos()

    @property
    def portrait(self):
        return self.portraits[self.indice_portrait]

    def remettre_le_portrait_au_repos(self):
        self.rendu = self.portrait.image
        self.cible = self.portrait.points

    def portrait_suivant(self):
        self.indice_portrait = (self.indice_portrait + 1) % len(self.portraits)
        self.remettre_le_portrait_au_repos()

    def ajouter_portrait(self, portrait):
        """Ajoute un portrait à la liste et l'affiche aussitôt."""
        self.portraits.append(portrait)
        self.indice_portrait = len(self.portraits) - 1
        self.remettre_le_portrait_au_repos()

    def changer_intensite(self, variation):
        nouvelle = round(self.intensite + variation, 1)
        self.intensite = min(max(nouvelle, 0.0), INTENSITE_MAX)


def ecrire(image, texte, ligne, couleur=VERT):
    """Écrit du texte ASCII : la police d'OpenCV ne connaît pas les accents."""
    position = (10, 30 + 30 * ligne)
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2, cv2.LINE_AA)


def mettre_a_jour_fps(fps, temps_precedent):
    """Calcule les images par seconde, moyennées pour que l'affichage reste lisible."""
    maintenant = time.perf_counter()
    instantane = 1.0 / max(maintenant - temps_precedent, 1e-6)
    fps = instantane if fps is None else COEFFICIENT_FPS * fps + (1 - COEFFICIENT_FPS) * instantane
    return fps, maintenant


def messages_de_debug(grimace, neutre, intensite):
    """Prépare les mesures affichées en mode debug."""
    maximum = float(np.linalg.norm(grimace.delta, axis=1).max())
    droit = ouverture_relative(grimace.reperes_recales, neutre[:, :2], "droite")
    gauche = ouverture_relative(grimace.reperes_recales, neutre[:, :2], "gauche")
    return [
        (f"Delta moyen {norme_moyenne(grimace.delta):.1f} px, max {maximum:.1f} px", VERT),
        (f"Tete : penchee {grimace.angle_tete:+.0f}, lacet {grimace.lacet:+.0f}, "
         f"tangage {grimace.tangage:+.0f} deg", VERT),
        (f"Ouverture yeux : D {droit:.2f}  G {gauche:.2f}", VERT),
        (f"Regard {mesurer_regard(grimace.reperes_recales, neutre[:, :2]):+.2f}", VERT),
        (f"Intensite {intensite:.1f}", VERT),
    ]


def lisser_la_pose(etat, grimace):
    """Lisse les trois angles de la tête dans le temps."""
    angles = np.float32([grimace.lacet, grimace.tangage, grimace.angle_tete])
    lacet, tangage, roulis = (float(a) for a in etat.lissage_pose.appliquer(angles))
    return replace(grimace, lacet=lacet, tangage=tangage, angle_tete=roulis)


def animer_le_portrait(etat, reperes):
    """Met à jour le portrait et renvoie les messages à afficher.

    Sans visage, le portrait reste simplement dans son dernier état.
    """
    if reperes is None:
        etat.lissage_pose.appliquer(None)
        return [("Aucun visage detecte", ROUGE)]
    if etat.neutre is None:
        return [("Regardez la camera, visage detendu,", JAUNE), ("appuyez sur N", JAUNE)]

    grimace = lisser_la_pose(etat, calculer_grimace_3d(reperes, etat.neutre))
    portrait = etat.portrait
    etat.cible = transferer(portrait, grimace, etat.neutre, etat.intensite)
    ouverture = ouverture_bouche_mesuree(grimace, etat.neutre, etat.intensite)
    etat.rendu = animer_portrait(portrait, etat.cible, ouverture)
    if not etat.afficher_debug:
        return []
    return messages_de_debug(grimace, etat.neutre, etat.intensite)


def composer_affichage(image_webcam, etat):
    """Place la webcam à gauche et le portrait à droite, à la même hauteur."""
    droite = etat.rendu
    if etat.afficher_debug:
        droite = droite.copy()
        dessiner_triangles(droite, etat.cible, etat.portrait.triangles)
    hauteur = droite.shape[0]
    largeur = round(image_webcam.shape[1] * hauteur / image_webcam.shape[0])
    return np.hstack([cv2.resize(image_webcam, (largeur, hauteur)), droite])


def choisir_image():
    """Ouvre la fenêtre de Windows pour choisir un fichier. Renvoie son chemin, ou ''."""
    import tkinter
    from tkinter import filedialog

    racine = tkinter.Tk()
    racine.withdraw()  # seule la fenêtre de choix doit apparaître
    racine.attributes("-topmost", True)  # sinon elle s'ouvre derrière la webcam
    try:
        return filedialog.askopenfilename(
            parent=racine, title="Choisir un portrait", filetypes=TYPES_IMAGES
        )
    finally:
        racine.destroy()


def ouvrir_une_image(etat):
    """Demande une image et l'anime. Sans visage, le portrait actuel reste affiché."""
    chemin = choisir_image()
    if not chemin:
        return
    portrait = charger_un_portrait(chemin)
    if portrait is None:
        etat.message = ("Aucun visage dans cette image : essayez-en une autre", ROUGE)
    else:
        etat.ajouter_portrait(portrait)


def appliquer_touche(etat, touche, reperes):
    """Applique une touche. Renvoie False s'il faut quitter."""
    if touche != AUCUNE_TOUCHE:
        etat.message = None
    lettre = chr(touche).lower()
    if lettre == "q":
        return False
    if lettre == "d":
        etat.afficher_debug = not etat.afficher_debug
    elif lettre == "o":
        ouvrir_une_image(etat)
    elif lettre == "p":
        etat.portrait_suivant()
    elif lettre == "n" and reperes is not None:
        etat.neutre = reperes.copy()
    elif lettre in ("+", "="):  # sur un clavier AZERTY, + demande Maj : = suffit
        etat.changer_intensite(PAS_INTENSITE)
    elif lettre in ("-", "_"):
        etat.changer_intensite(-PAS_INTENSITE)
    return True


def executer(camera, detecteur, portraits):
    """Lit la webcam et affiche le résultat jusqu'à ce que l'on quitte."""
    etat = Etat(portraits)
    lissage = LissageExponentiel(COEFFICIENT_LISSAGE)
    fps = None
    temps_precedent = time.perf_counter()

    while True:
        lecture_ok, image = camera.read()
        if not lecture_ok:
            print("Lecture de la webcam impossible.", file=sys.stderr)
            return

        # Effet miroir : le côté droit de l'écran correspond à votre côté droit.
        image = cv2.flip(image, 1)
        reperes = lissage.appliquer(detecteur.detecter_3d(image))
        messages = animer_le_portrait(etat, reperes)
        if etat.message is not None:
            messages = [etat.message] + messages
        if reperes is not None and etat.afficher_debug:
            dessiner_reperes(image, reperes[:, :2])

        affichage = composer_affichage(image, etat)
        fps, temps_precedent = mettre_a_jour_fps(fps, temps_precedent)
        ecrire(affichage, f"{fps:.0f} images/s", 0)
        for ligne, (texte, couleur) in enumerate(messages, start=1):
            ecrire(affichage, texte, ligne, couleur)
        cv2.imshow(NOM_FENETRE, affichage)

        touche = cv2.waitKey(ATTENTE_TOUCHE_MS) & 0xFF
        if not appliquer_touche(etat, touche, reperes):
            return
        if cv2.getWindowProperty(NOM_FENETRE, cv2.WND_PROP_VISIBLE) < 1:
            return


def main():
    portraits = list(charger_portraits().values())
    if not portraits:
        sys.exit("Aucun portrait utilisable dans le dossier portraits/.")
    camera = cv2.VideoCapture(INDICE_CAMERA)
    if not camera.isOpened():
        sys.exit("Webcam introuvable : vérifiez qu'aucune autre application ne l'utilise.")
    try:
        with DetecteurReperes(mode_video=True) as detecteur:
            executer(camera, detecteur, portraits)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
