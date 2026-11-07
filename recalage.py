"""Enlève le mouvement de la tête pour ne garder que la grimace.

Deux méthodes : la similitude 2D de base et une similitude 3D, qui enlève aussi les
rotations gauche-droite et haut-bas grâce à la profondeur donnée par MediaPipe.
"""
from dataclasses import dataclass
from math import asin, atan2, degrees

import cv2
import numpy as np

from reperes import TEMPES, positions

# Points qui bougent peu pendant une grimace : coins des yeux, haut du nez, tempes.
INDICES_STABLES = [33, 133, 362, 263, 168] + TEMPES
POSITIONS_STABLES = positions(INDICES_STABLES)
# En 3D, on ajoute l'arête du nez et le haut du front pour mieux fixer la rotation.
POSITIONS_STABLES_3D = positions(INDICES_STABLES + [6, 10])
SEUIL_RANSAC_PX = 3.0  # au-delà de cet écart, un point stable est jugé aberrant
MINIMUM_POINTS_3D = 5  # en dessous, le rejet des points aberrants n'est pas appliqué


@dataclass
class Grimace:
    """Résultat du recalage d'une image sur le visage neutre."""

    reperes_recales: np.ndarray  # positions (x, y) une fois la tête remise en place
    delta: np.ndarray  # déplacement de chaque repère par rapport au visage neutre
    angle_tete: float  # rotation de la tête dans le plan de l'image, en degrés
    lacet: float = 0.0  # rotation gauche-droite, en degrés (méthode 3D)
    tangage: float = 0.0  # rotation haut-bas, en degrés (méthode 3D)


def estimer_similitude(reperes, neutre):
    """Estime la similitude qui superpose les points stables sur le visage neutre.

    Similitude (rotation, échelle, translation) : 4 inconnues, donc 2 points suffisent.
    Avec plus de points, c'est un ajustement par moindres carrés, et RANSAC écarte les
    points aberrants. Renvoie une matrice 2×3, ou None si l'estimation échoue.
    """
    matrice, _ = cv2.estimateAffinePartial2D(
        reperes[POSITIONS_STABLES], neutre[POSITIONS_STABLES],
        method=cv2.RANSAC, ransacReprojThreshold=SEUIL_RANSAC_PX,
    )
    return matrice


def angle_de_la_tete(matrice):
    """Lit la rotation de la tête dans la matrice [[s·cos, -s·sin, tx], [s·sin, s·cos, ty]].

    La matrice ramène la tête vers le visage neutre : elle applique la rotation
    inverse de celle de la tête, d'où le signe moins.
    """
    return -degrees(atan2(matrice[1, 0], matrice[0, 0]))


def calculer_grimace(reperes, neutre):
    """Recalage 2D : calcule Δ = Mt recalé − M0. Renvoie None en cas d'échec."""
    matrice = estimer_similitude(reperes, neutre)
    if matrice is None:
        return None
    recales = cv2.transform(reperes[np.newaxis], matrice)[0]
    return Grimace(recales, recales - neutre, angle_de_la_tete(matrice))


def similitude_3d_moindres_carres(source, cible):
    """Calcule s, R et t qui minimisent la somme des ‖s·R·source + t − cible‖².

    Moindres carrés avec une solution exacte (méthode d'Umeyama) : la décomposition en
    valeurs singulières donne la rotation R, puis on en déduit l'échelle s et la
    translation t. Les notations R et t sont celles du modèle de caméra du cours.
    """
    centre_source, centre_cible = source.mean(axis=0), cible.mean(axis=0)
    s_centre, c_centre = source - centre_source, cible - centre_cible
    u, valeurs, vt = np.linalg.svd(c_centre.T @ s_centre)
    # Ce signe impose une vraie rotation, et non une symétrie miroir.
    signes = np.array([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ np.diag(signes) @ vt
    echelle = (valeurs * signes).sum() / (s_centre**2).sum()
    return echelle, rotation, centre_cible - echelle * rotation @ centre_source


def similitude_3d_robuste(source, cible):
    """Applique les moindres carrés, écarte les points trop éloignés, puis recommence.

    Même idée que RANSAC : un point aberrant ne doit pas fausser l'estimation.
    """
    echelle, rotation, translation = similitude_3d_moindres_carres(source, cible)
    ecarts = np.linalg.norm(echelle * source @ rotation.T + translation - cible, axis=1)
    fiables = ecarts <= max(SEUIL_RANSAC_PX, 2 * np.median(ecarts))
    if MINIMUM_POINTS_3D <= fiables.sum() < len(source):
        return similitude_3d_moindres_carres(source[fiables], cible[fiables])
    return echelle, rotation, translation


def rotation_depuis_angles(lacet, tangage, roulis):
    """Construit R = Rz(roulis) · Rx(tangage) · Ry(lacet) à partir d'angles en degrés."""
    y, x, z = np.radians([lacet, tangage, roulis])
    autour_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    autour_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    autour_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return autour_z @ autour_x @ autour_y


def angles_depuis_rotation(rotation):
    """Opération inverse : renvoie (lacet, tangage, roulis) en degrés."""
    return (
        degrees(atan2(-rotation[2, 0], rotation[2, 2])),
        degrees(asin(float(np.clip(rotation[2, 1], -1.0, 1.0)))),
        degrees(atan2(-rotation[0, 1], rotation[1, 1])),
    )


def calculer_grimace_3d(reperes, neutre):
    """Recalage 3D : met la tête dans sa pose neutre, puis garde x et y.

    Les tableaux reçus sont de taille (N, 3). Cette méthode reste fiable quand la tête
    tourne, car ces rotations sont mesurées au lieu d'être prises pour une grimace.
    """
    source = np.float64(reperes)
    echelle, rotation, translation = similitude_3d_robuste(
        source[POSITIONS_STABLES_3D], np.float64(neutre)[POSITIONS_STABLES_3D]
    )
    recales = np.float32((echelle * source @ rotation.T + translation)[:, :2])
    # Rotation neutre ← tête : la pose de la tête est donc sa transposée.
    lacet, tangage, roulis = angles_depuis_rotation(rotation.T)
    return Grimace(recales, recales - neutre[:, :2], roulis, lacet, tangage)


def norme_moyenne(delta):
    """Déplacement moyen des repères, en pixels."""
    return float(np.linalg.norm(delta, axis=1).mean())
