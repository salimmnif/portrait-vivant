"""Vérifie que l'installation permet de lancer le projet.

Utilisation, depuis le dossier portrait_vivant :
    python tests/verifier_installation.py            # sans webcam
    python tests/verifier_installation.py --webcam   # ouvre aussi la webcam
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import scipy
from mediapipe.tasks.python import BaseOptions, vision
from scipy.spatial import Delaunay

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = RACINE / "modeles" / "face_landmarker.task"
DOSSIER_PORTRAITS = RACINE / "portraits"
NB_REPERES_ATTENDU = 478  # 468 repères du maillage + 10 repères des iris
NB_MESURES_DUREE = 30


def afficher_resultat(nom, reussi, detail=""):
    """Affiche une ligne de bilan et renvoie le résultat du contrôle."""
    etat = "OK    " if reussi else "ECHEC "
    print(f"[{etat}] {nom}{' : ' + detail if detail else ''}")
    return reussi


def verifier_transformation_affine():
    """Transformation affine : 3 points → 6 inconnues, donc une solution exacte."""
    source = np.float32([[0, 0], [10, 0], [0, 10]])
    cible = np.float32([[2, 3], [12, 3], [2, 13]])
    matrice = cv2.getAffineTransform(source, cible)
    attendue = np.array([[1, 0, 2], [0, 1, 3]], dtype=float)
    return afficher_resultat("Affine à 3 points", np.allclose(matrice, attendue))


def verifier_ransac():
    """Moindres carrés avec RANSAC : la similitude doit ignorer un point aberrant."""
    source = np.float32([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [2, 8]])
    cible = source + np.float32([4, -2])
    cible[-1] += 50  # point aberrant
    matrice, inliers = cv2.estimateAffinePartial2D(source, cible, method=cv2.RANSAC)
    attendue = np.array([[1, 0, 4], [0, 1, -2]], dtype=float)
    reussi = (
        matrice is not None
        and np.allclose(matrice, attendue, atol=1e-3)
        and inliers[-1, 0] == 0
    )
    return afficher_resultat("Similitude avec RANSAC", reussi)


def verifier_delaunay():
    """La triangulation de Delaunay d'un carré découpe celui-ci en 2 triangles."""
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    nb_triangles = len(Delaunay(points).simplices)
    return afficher_resultat("Delaunay", nb_triangles == 2, f"{nb_triangles} triangles")


def creer_detecteur():
    """Crée le détecteur de repères de l'API Tasks de MediaPipe, en mode image."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(CHEMIN_MODELE)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detecter(detecteur, image_bgr):
    """Lance la détection ; OpenCV fournit du BGR, MediaPipe attend du RVB."""
    image_rvb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rvb)
    return detecteur.detect(image_mp).face_landmarks


def verifier_portraits(detecteur):
    """Détecte le visage de chaque portrait et mesure la durée sur le premier."""
    extensions = {".jpg", ".jpeg", ".png"}
    chemins = sorted(
        c for c in DOSSIER_PORTRAITS.glob("*") if c.suffix.lower() in extensions
    )
    if not chemins:
        return afficher_resultat("Portraits", False, f"aucune image dans {DOSSIER_PORTRAITS}")

    reussi = True
    for chemin in chemins:
        image = cv2.imread(str(chemin))
        if image is None:
            reussi = afficher_resultat(chemin.name, False, "image illisible")
            continue
        visages = detecter(detecteur, image)
        nb_reperes = len(visages[0]) if visages else 0
        reussi &= afficher_resultat(
            chemin.name, nb_reperes == NB_REPERES_ATTENDU, f"{nb_reperes} repères"
        )

    image = cv2.imread(str(chemins[0]))
    if image is not None:
        debut = time.perf_counter()
        for _ in range(NB_MESURES_DUREE):
            detecter(detecteur, image)
        duree_ms = (time.perf_counter() - debut) / NB_MESURES_DUREE * 1000
        print(f"         Durée moyenne de détection : {duree_ms:.1f} ms sur le CPU")
    return reussi


def verifier_mediapipe():
    """Vérifie le modèle, puis la détection sur une image vide et sur les portraits."""
    if not CHEMIN_MODELE.exists():
        return afficher_resultat(
            "Modèle", False, f"{CHEMIN_MODELE} absent, voir le README"
        )
    afficher_resultat("Modèle", True, f"{CHEMIN_MODELE.stat().st_size / 1e6:.1f} Mo")

    with creer_detecteur() as detecteur:
        image_vide = np.zeros((256, 256, 3), np.uint8)
        vide_ok = afficher_resultat(
            "Image vide : aucun visage", not detecter(detecteur, image_vide)
        )
        return verifier_portraits(detecteur) and vide_ok


def verifier_webcam():
    """Ouvre la webcam et lit une image, sans rien afficher."""
    camera = cv2.VideoCapture(0)
    try:
        lecture_ok, image = camera.read()
    finally:
        camera.release()
    detail = f"{image.shape[1]}x{image.shape[0]}" if lecture_ok else "aucune image lue"
    return afficher_resultat("Webcam", lecture_ok, detail)


def main():
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--webcam", action="store_true", help="tester aussi la webcam")
    arguments = analyseur.parse_args()

    print(f"Python {sys.version.split()[0]} | OpenCV {cv2.__version__} | "
          f"MediaPipe {mp.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")

    resultats = [
        verifier_transformation_affine(),
        verifier_ransac(),
        verifier_delaunay(),
        verifier_mediapipe(),
    ]
    if arguments.webcam:
        resultats.append(verifier_webcam())

    print("\nInstallation correcte." if all(resultats) else "\nDes contrôles ont échoué.")
    sys.exit(0 if all(resultats) else 1)


if __name__ == "__main__":
    main()
