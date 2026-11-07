"""Tests sans webcam. Lancement depuis portrait_vivant : python -m pytest tests"""
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from reperes import (
    CONNEXIONS, GROUPES, INDICES_REPERES, DetecteurReperes, LissageExponentiel,
    sommets_dans_l_ordre,
)
from simulation import lancer_boucle

CHEMIN_JOCONDE = Path(__file__).resolve().parent.parent / "portraits" / "joconde.jpg"


@pytest.fixture(scope="module")
def joconde():
    image = cv2.imread(str(CHEMIN_JOCONDE))
    assert image is not None
    return image


def test_sous_ensemble_de_taille_prevue_et_sans_doublon():
    assert 80 <= len(INDICES_REPERES) <= 120
    assert len(set(INDICES_REPERES)) == len(INDICES_REPERES)
    assert all(0 <= indice < 478 for indice in INDICES_REPERES)


def test_indices_ecrits_a_la_main_presents_dans_mediapipe():
    assert set(GROUPES["nez"]) <= set(sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_NOSE))
    assert set(GROUPES["sourcil_droit"]) <= set(
        sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_RIGHT_EYEBROW)
    )
    assert set(GROUPES["sourcil_gauche"]) <= set(
        sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_LEFT_EYEBROW)
    )
    assert {127, 356} <= set(sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_FACE_OVAL))
    assert set(GROUPES["joues"]) <= set(sommets_dans_l_ordre(CONNEXIONS.FACE_LANDMARKS_TESSELATION))


def test_points_stables_du_recalage_disponibles():
    coins_des_yeux = {33, 133, 362, 263}
    haut_du_nez = {168}
    tempes = {127, 356}
    assert coins_des_yeux | haut_du_nez | tempes <= set(INDICES_REPERES)


def test_detection_sur_la_joconde(joconde):
    with DetecteurReperes() as detecteur:
        reperes = detecteur.detecter(joconde)
    assert reperes.shape == (len(INDICES_REPERES), 2)
    hauteur, largeur = joconde.shape[:2]
    assert (reperes >= 0).all()
    assert (reperes[:, 0] < largeur).all() and (reperes[:, 1] < hauteur).all()


def test_detection_3d_donne_x_y_et_z(joconde):
    with DetecteurReperes() as detecteur:
        reperes = detecteur.detecter_3d(joconde)
        en_2d = detecteur.detecter(joconde)
    assert reperes.shape == (len(INDICES_REPERES), 3)
    assert np.allclose(reperes[:, :2], en_2d)
    assert reperes[:, 2].std() > 1  # le z varie vraiment d'un repère à l'autre


def test_lissage_moyenne_ponderee():
    lissage = LissageExponentiel(coefficient=0.25)
    assert np.allclose(lissage.appliquer(np.float32([[0, 0]])), 0)
    assert np.allclose(lissage.appliquer(np.float32([[8, 4]])), [2, 1])
    assert np.allclose(lissage.appliquer(np.float32([[8, 4]])), [3.5, 1.75])


def test_lissage_repart_de_zero_apres_une_perte_du_visage():
    lissage = LissageExponentiel(coefficient=0.25)
    lissage.appliquer(np.float32([[0, 0]]))
    assert lissage.appliquer(None) is None
    assert np.allclose(lissage.appliquer(np.float32([[8, 4]])), [8, 4])


def test_lissage_reduit_le_tremblement():
    bruit = np.random.default_rng(0).normal(0, 1, (300, 1, 2)).astype(np.float32)
    lissage = LissageExponentiel(coefficient=0.3)
    lisse = np.array([lissage.appliquer(mesure) for mesure in bruit])
    assert lisse[20:].std() < bruit[20:].std() / 2


def test_image_vide_sans_visage():
    with DetecteurReperes() as detecteur:
        assert detecteur.detecter(np.zeros((256, 256, 3), np.uint8)) is None


def test_mode_video_temps_croissant(joconde):
    with DetecteurReperes(mode_video=True) as detecteur:
        assert detecteur.detecter(joconde) is not None
        assert detecteur.detecter(joconde) is not None  # deux appels très rapprochés


def compter_boucle(monkeypatch, image, touches):
    """Lance la boucle et compte les images affichées et les dessins de repères."""
    dessins = []
    monkeypatch.setattr(main, "dessiner_reperes", lambda *_: dessins.append(1))
    affichees = lancer_boucle(monkeypatch, [image], touches)
    return {"images": len(affichees), "dessins": len(dessins)}


def test_boucle_touches_d_et_q(monkeypatch, joconde):
    # Image 1 : repères ; D les masque ; image 2 : rien ; D les réaffiche ; image 3 : Q.
    assert compter_boucle(monkeypatch, joconde, ["d", "d", "q"]) == {"images": 3, "dessins": 2}


def test_boucle_sans_visage_ne_plante_pas(monkeypatch):
    image_vide = np.zeros((240, 320, 3), np.uint8)
    assert compter_boucle(monkeypatch, image_vide, ["q"]) == {"images": 1, "dessins": 0}
