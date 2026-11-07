"""Affiche les repères et la triangulation de chaque portrait, et les sauvegarde.

Utilisation, depuis portrait_vivant : python tests/voir_triangulation.py
Une touche passe au portrait suivant ; Q quitte.
"""
import sys
from pathlib import Path

import cv2

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from deformation import dessiner_triangles, preparer_portrait  # noqa: E402
from reperes import DetecteurReperes, dessiner_reperes  # noqa: E402

DOSSIER_PORTRAITS = RACINE / "portraits"
DOSSIER_SORTIES = Path(__file__).resolve().parent / "sorties"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_de_controle(portrait):
    """Dessine les triangles et les repères sur une copie du portrait."""
    controle = portrait.image.copy()
    dessiner_triangles(controle, portrait.points, portrait.triangles)
    dessiner_reperes(controle, portrait.points)
    return controle


def main():
    DOSSIER_SORTIES.mkdir(exist_ok=True)
    chemins = sorted(c for c in DOSSIER_PORTRAITS.iterdir() if c.suffix.lower() in EXTENSIONS)
    with DetecteurReperes() as detecteur:
        for chemin in chemins:
            image = cv2.imread(str(chemin))
            reperes = detecteur.detecter_3d(image)
            if reperes is None:
                print(f"{chemin.name} : aucun visage détecté")
                continue
            portrait = preparer_portrait(image, reperes)
            print(f"{chemin.name} : {len(portrait.points)} points, "
                  f"{len(portrait.triangles)} triangles")
            controle = image_de_controle(portrait)
            cv2.imwrite(str(DOSSIER_SORTIES / f"triangulation_{chemin.stem}.png"), controle)
            cv2.imshow("Triangulation", controle)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
