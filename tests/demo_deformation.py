"""Déforme chaque portrait à la main, sans webcam, et sauvegarde avant/après.

Utilisation, depuis portrait_vivant : python tests/demo_deformation.py
Les images sont écrites dans tests/sorties/.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from deformation import animer, charger_portraits, triangles_retournes  # noqa: E402
from reperes import (  # noqa: E402
    GROUPES, LEVRE_BASSE_EXTERIEURE, LEVRE_BASSE_INTERIEURE, PAUPIERES, positions,
)

DOSSIER_SORTIES = Path(__file__).resolve().parent / "sorties"


def distance_yeux(points):
    """Distance entre les centres des iris : l'unité de longueur du visage."""
    droit, gauche = positions(GROUPES["iris_droit"] + GROUPES["iris_gauche"])
    return float(np.linalg.norm(points[droit] - points[gauche]))


def clin_d_oeil(points):
    """Ferme l'œil droit : la paupière haute descend sur la paupière basse."""
    cible = points.copy()
    basse, haute = (positions(paupiere) for paupiere in PAUPIERES["droite"])
    cible[haute] = points[haute] + 0.95 * (points[basse] - points[haute])
    centre = positions(GROUPES["iris_droit"])
    cible[centre] = (cible[haute].mean(axis=0) + cible[basse].mean(axis=0)) / 2
    return cible


def sourire(points):
    """Tire les coins de la bouche vers le haut et vers l'extérieur."""
    cible = points.copy()
    unite = distance_yeux(points)
    droite, gauche = positions([61, 78]), positions([291, 308])
    cible[droite] += np.float32([-0.10, -0.08]) * unite
    cible[gauche] += np.float32([0.10, -0.08]) * unite
    return cible


def sourcils_leves(points):
    """Monte les deux sourcils."""
    cible = points.copy()
    sourcils = positions(GROUPES["sourcil_droit"] + GROUPES["sourcil_gauche"])
    cible[sourcils, 1] -= 0.15 * distance_yeux(points)
    return cible


def bouche_ouverte(points):
    """Descend la lèvre basse et le menton, sans bouger les commissures."""
    cible = points.copy()
    unite = distance_yeux(points)
    coins = {61, 291, 78, 308}
    levre = [i for i in LEVRE_BASSE_EXTERIEURE + LEVRE_BASSE_INTERIEURE if i not in coins]
    positions_levre = positions(levre)
    # La lèvre descend surtout en son milieu, et presque pas près des commissures.
    x = points[positions_levre, 0]
    milieu, demi_largeur = (x.min() + x.max()) / 2, (x.max() - x.min()) / 2
    cible[positions_levre, 1] += 0.35 * unite * (1.0 - ((x - milieu) / demi_largeur) ** 2)
    bas_de_la_bouche = points[positions_levre, 1].max()
    menton = [p for p in positions(GROUPES["contour"]) if points[p, 1] > bas_de_la_bouche]
    cible[menton, 1] += 0.25 * unite
    return cible


SCENARIOS = {
    "clin_d_oeil": clin_d_oeil,
    "sourire": sourire,
    "sourcils": sourcils_leves,
    "bouche_ouverte": bouche_ouverte,
}


def main():
    DOSSIER_SORTIES.mkdir(exist_ok=True)
    for nom, portrait in charger_portraits().items():
        for nom_scenario, scenario in SCENARIOS.items():
            cible = scenario(portrait.points)
            apres = animer(portrait.image, portrait.points, cible, portrait.triangles)
            nb_retournes = triangles_retournes(portrait.points, cible, portrait.triangles)
            print(f"{nom} / {nom_scenario} : {nb_retournes} triangle(s) retourné(s)")
            avant_apres = np.hstack([portrait.image, apres])
            cv2.imwrite(str(DOSSIER_SORTIES / f"{nom}_{nom_scenario}.png"), avant_apres)


if __name__ == "__main__":
    main()
