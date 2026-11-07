"""Détection des repères du visage avec MediaPipe et choix d'un sous-ensemble.

Les indices proviennent de FaceLandmarksConnections, fourni par MediaPipe, ou du
modèle à 478 repères pour les centres des iris. Ils ne sont pas écrits au hasard.
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

CHEMIN_MODELE = Path(__file__).resolve().parent / "modeles" / "face_landmarker.task"
CONNEXIONS = vision.FaceLandmarksConnections
# Tempes : utiles au recalage, mais exclues des triangles. Sur un visage de trois
# quarts, la tempe du côté éloigné se projette sur l'œil et y créerait des plis.
TEMPES = [127, 356]


def sommets_dans_l_ordre(connexions):
    """Transforme une liste d'arêtes (a, b) en liste de sommets sans doublon."""
    sommets = []
    for arete in connexions:
        for indice in (arete.start, arete.end):
            if indice not in sommets:
                sommets.append(indice)
    return sommets


def chaine(connexions, debut, fin):
    """Renvoie dans l'ordre les sommets d'une tranche de la liste d'arêtes."""
    return sommets_dans_l_ordre(list(connexions)[debut:fin])


def contour_allege():
    """Garde un point sur deux du contour, plus les tempes utiles au recalage."""
    contour = sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_FACE_OVAL)
    return contour[::2] + [i for i in TEMPES if i not in contour[::2]]


# Sous-ensemble utilisé partout, pour le portrait comme pour la webcam.
# Les mots « droit » et « gauche » désignent les côtés de la personne filmée.
GROUPES = {
    "contour": contour_allege(),
    "sourcil_droit": [70, 63, 105, 66, 107],  # bord supérieur du sourcil
    "sourcil_gauche": [300, 293, 334, 296, 336],  # bord supérieur du sourcil
    "oeil_droit": sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_RIGHT_EYE),
    "oeil_gauche": sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_LEFT_EYE),
    # Centre de l'iris seulement : le haut et le bas du pourtour sont cachés sous
    # les paupières, et leurs triangles se retourneraient à chaque clignement.
    "iris_droit": [468],
    "iris_gauche": [473],
    "nez": [168, 6, 195, 4, 2, 98, 327],  # arête, pointe, base et ailes du nez
    "levres": sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_LIPS),
    # Joues : sans ces points, le sourire se perd dans de très grands triangles.
    "joues": [205, 425, 50, 280, 214, 434],
}
INDICES_REPERES = [indice for groupe in GROUPES.values() for indice in groupe]
NOMBRE_REPERES = len(INDICES_REPERES)

# Dans les listes de MediaPipe, le bas de chaque paupière ou lèvre précède le haut.
_OEIL_DROIT = CONNEXIONS.FACE_LANDMARKS_RIGHT_EYE
_OEIL_GAUCHE = CONNEXIONS.FACE_LANDMARKS_LEFT_EYE
_LEVRES = CONNEXIONS.FACE_LANDMARKS_LIPS
PAUPIERES = {  # (paupière basse, paupière haute), de même longueur et même sens
    "droite": (chaine(_OEIL_DROIT, 0, 8), chaine(_OEIL_DROIT, 8, 16)),
    "gauche": (chaine(_OEIL_GAUCHE, 0, 8), chaine(_OEIL_GAUCHE, 8, 16)),
}
LEVRE_BASSE_EXTERIEURE = chaine(_LEVRES, 0, 10)
LEVRE_HAUTE_EXTERIEURE = chaine(_LEVRES, 10, 20)
LEVRE_BASSE_INTERIEURE = chaine(_LEVRES, 20, 30)
LEVRE_HAUTE_INTERIEURE = chaine(_LEVRES, 30, 40)

COULEURS_GROUPES = {  # couleurs en BGR, l'ordre utilisé par OpenCV
    "contour": (255, 255, 255),
    "sourcil_droit": (0, 165, 255),
    "sourcil_gauche": (0, 165, 255),
    "oeil_droit": (0, 255, 0),
    "oeil_gauche": (0, 255, 0),
    "iris_droit": (0, 255, 255),
    "iris_gauche": (0, 255, 255),
    "nez": (255, 128, 0),
    "levres": (180, 80, 255),
    "joues": (200, 200, 0),
}


def positions(indices_mediapipe):
    """Convertit des indices MediaPipe en positions dans le sous-ensemble."""
    return [INDICES_REPERES.index(indice) for indice in indices_mediapipe]


class DetecteurReperes:
    """Détecteur MediaPipe (API Tasks) qui renvoie le sous-ensemble en pixels."""

    def __init__(self, mode_video=False):
        # Le mode vidéo suit le visage d'une image à l'autre, ce qui évite de
        # relancer à chaque fois la recherche complète du visage.
        self.mode_video = mode_video
        self.dernier_temps_ms = -1
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(CHEMIN_MODELE)),
            running_mode=(
                vision.RunningMode.VIDEO if mode_video else vision.RunningMode.IMAGE
            ),
            num_faces=1,
        )
        self.detecteur = vision.FaceLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, *erreur):
        self.detecteur.close()

    def temps_suivant_ms(self):
        """Renvoie un temps strictement croissant, exigé par le mode vidéo."""
        maintenant = int(time.perf_counter() * 1000)
        self.dernier_temps_ms = max(maintenant, self.dernier_temps_ms + 1)
        return self.dernier_temps_ms

    def detecter_3d(self, image_bgr):
        """Renvoie un tableau (N, 3) en pixels, ou None sans visage.

        MediaPipe donne x et y entre 0 et 1. Son z, relatif au visage, utilise à peu
        près la même échelle que x : on le multiplie donc aussi par la largeur.
        """
        # Image numérique : OpenCV range les canaux en BGR, MediaPipe attend du RVB.
        image_rvb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rvb)
        if self.mode_video:
            resultat = self.detecteur.detect_for_video(image_mp, self.temps_suivant_ms())
        else:
            resultat = self.detecteur.detect(image_mp)
        if not resultat.face_landmarks:
            return None

        hauteur, largeur = image_bgr.shape[:2]
        visage = resultat.face_landmarks[0]
        return np.array(
            [
                (visage[i].x * largeur, visage[i].y * hauteur, visage[i].z * largeur)
                for i in INDICES_REPERES
            ],
            dtype=np.float32,
        )

    def detecter(self, image_bgr):
        """Renvoie un tableau (N, 2) de positions en pixels, ou None sans visage."""
        reperes = self.detecter_3d(image_bgr)
        return None if reperes is None else reperes[:, :2]


class LissageExponentiel:
    """Filtre temporel : moyenne pondérée de la mesure et de l'état précédent.

    Plus le coefficient est petit, plus les repères sont stables, mais plus ils
    suivent le visage avec retard.
    """

    def __init__(self, coefficient=0.5):
        self.coefficient = coefficient
        self.etat = None

    def appliquer(self, reperes):
        """Renvoie les repères lissés. Sans visage, on repart de zéro au retour."""
        if reperes is None:
            self.etat = None
        elif self.etat is None:
            self.etat = reperes.copy()
        else:
            self.etat = self.coefficient * reperes + (1 - self.coefficient) * self.etat
        return self.etat


def dessiner_reperes(image, reperes):
    """Dessine les repères sur l'image, avec une couleur par partie du visage."""
    position = 0
    for nom, groupe in GROUPES.items():
        for x, y in reperes[position : position + len(groupe)]:
            cv2.circle(
                image, (round(float(x)), round(float(y))), 2,
                COULEURS_GROUPES[nom], -1, cv2.LINE_AA,
            )
        position += len(groupe)
