"""Tests sans webcam. Lancement depuis portrait_vivant : python -m pytest tests"""
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from recalage import (
    POSITIONS_STABLES, angles_depuis_rotation, calculer_grimace, calculer_grimace_3d,
    norme_moyenne, rotation_depuis_angles,
)
from reperes import DetecteurReperes, positions
from simulation import lancer_boucle

CHEMIN_JOCONDE = Path(__file__).resolve().parent.parent / "portraits" / "joconde.jpg"


@pytest.fixture(scope="module")
def image_joconde():
    return cv2.imread(str(CHEMIN_JOCONDE))


@pytest.fixture(scope="module")
def neutre(image_joconde):
    with DetecteurReperes() as detecteur:
        return detecteur.detecter(image_joconde)


def bouger_la_tete(reperes, angle=10.0, echelle=1.1, translation=(40, -25)):
    """Applique une similitude à tous les repères. Angle > 0 : sens horaire à l'écran."""
    centre = tuple(float(v) for v in reperes.mean(axis=0))
    matrice = cv2.getRotationMatrix2D(centre, -angle, echelle)
    matrice[:, 2] += translation
    return cv2.transform(reperes[np.newaxis], matrice)[0]


def test_sans_mouvement_delta_nul(neutre):
    grimace = calculer_grimace(neutre, neutre)
    assert norme_moyenne(grimace.delta) < 0.01
    assert grimace.angle_tete == pytest.approx(0, abs=0.01)


def test_mouvement_de_tete_sans_grimace_delta_presque_nul(neutre):
    bouge = bouger_la_tete(neutre)
    assert norme_moyenne(bouge - neutre) > 10  # sans recalage, l'écart est grand
    assert norme_moyenne(calculer_grimace(bouge, neutre).delta) < 0.1  # avec, il disparaît


@pytest.mark.parametrize("angle", [-20.0, 7.0, 15.0])
def test_angle_de_la_tete(neutre, angle):
    grimace = calculer_grimace(bouger_la_tete(neutre, angle=angle), neutre)
    assert grimace.angle_tete == pytest.approx(angle, abs=0.1)


def test_la_grimace_survit_au_mouvement_de_tete(neutre):
    levre = positions([14, 17])
    avec_grimace = neutre.copy()
    avec_grimace[levre, 1] += 20.0  # la lèvre basse descend de 20 pixels
    grimace = calculer_grimace(bouger_la_tete(avec_grimace, echelle=1.0), neutre)
    assert np.allclose(grimace.delta[levre], [0, 20], atol=0.5)
    autres = np.delete(grimace.delta, levre, axis=0)
    assert norme_moyenne(autres) < 0.1


def test_ransac_ignore_un_point_stable_aberrant(neutre):
    bouge = bouger_la_tete(neutre)
    aberrant = POSITIONS_STABLES[-1]
    bouge[aberrant] += 40.0
    delta = calculer_grimace(bouge, neutre).delta
    assert norme_moyenne(np.delete(delta, aberrant, axis=0)) < 0.1


def tourner_la_tete_3d(reperes, lacet=0.0, tangage=0.0, roulis=0.0, echelle=1.0):
    """Applique une rotation 3D rigide au visage, autour de son centre."""
    centre = reperes.mean(axis=0)
    rotation = rotation_depuis_angles(lacet, tangage, roulis)
    return np.float32(echelle * (reperes - centre) @ rotation.T + centre + (30, -20, 0))


@pytest.fixture(scope="module")
def neutre_3d(image_joconde):
    with DetecteurReperes() as detecteur:
        return detecteur.detecter_3d(image_joconde)


@pytest.mark.parametrize("lacet, tangage", [(30, 0), (-30, 0), (0, 20), (20, -15)])
def test_recalage_3d_enleve_la_tete_tournee(neutre_3d, lacet, tangage):
    tournee = tourner_la_tete_3d(neutre_3d, lacet, tangage, roulis=5, echelle=1.1)
    sans_3d = calculer_grimace(tournee[:, :2], neutre_3d[:, :2])
    avec_3d = calculer_grimace_3d(tournee, neutre_3d)
    assert norme_moyenne(avec_3d.delta) < 0.1
    assert norme_moyenne(avec_3d.delta) < norme_moyenne(sans_3d.delta) / 10
    assert avec_3d.reperes_recales.shape == (len(neutre_3d), 2)


def test_recalage_3d_garde_la_grimace(neutre_3d):
    levre = positions([14, 17])
    avec_grimace = neutre_3d.copy()
    avec_grimace[levre, 1] += 20.0
    grimace = calculer_grimace_3d(tourner_la_tete_3d(avec_grimace, lacet=25), neutre_3d)
    assert norme_moyenne(np.delete(grimace.delta, levre, axis=0)) < 0.3
    assert np.linalg.norm(grimace.delta[levre], axis=1).min() > 15


def test_recalage_3d_ignore_un_point_stable_aberrant(neutre_3d):
    tournee = tourner_la_tete_3d(neutre_3d, lacet=20)
    aberrant = POSITIONS_STABLES[-1]
    tournee[aberrant] += 40.0
    delta = calculer_grimace_3d(tournee, neutre_3d).delta
    assert norme_moyenne(np.delete(delta, aberrant, axis=0)) < 0.2


@pytest.mark.parametrize("roulis", [-20.0, 10.0])
def test_recalage_3d_lit_le_roulis(neutre_3d, roulis):
    grimace = calculer_grimace_3d(tourner_la_tete_3d(neutre_3d, roulis=roulis), neutre_3d)
    assert grimace.angle_tete == pytest.approx(roulis, abs=0.5)


def test_recalage_3d_lit_les_trois_angles(neutre_3d):
    tournee = tourner_la_tete_3d(neutre_3d, lacet=-22, tangage=12, roulis=8)
    grimace = calculer_grimace_3d(tournee, neutre_3d)
    assert grimace.lacet == pytest.approx(-22, abs=0.5)
    assert grimace.tangage == pytest.approx(12, abs=0.5)
    assert grimace.angle_tete == pytest.approx(8, abs=0.5)


def test_angles_et_rotation_sont_inverses():
    angles = angles_depuis_rotation(rotation_depuis_angles(20, -10, 5))
    assert angles == pytest.approx((20, -10, 5))


def compter_grimaces(monkeypatch, image, touches):
    """Lance la boucle et renvoie le visage neutre utilisé à chaque calcul de grimace."""
    appels = []
    calcul_reel = main.calculer_grimace_3d

    def compter(reperes, neutre):
        appels.append(neutre)
        return calcul_reel(reperes, neutre)

    monkeypatch.setattr(main, "calculer_grimace_3d", compter)
    lancer_boucle(monkeypatch, [image], touches)
    return appels


def test_pas_de_grimace_avant_la_touche_n(monkeypatch, image_joconde):
    assert compter_grimaces(monkeypatch, image_joconde, ["x", "x", "q"]) == []


def test_touche_n_enregistre_le_visage_neutre(monkeypatch, image_joconde):
    assert len(compter_grimaces(monkeypatch, image_joconde, ["n", "x", "q"])) == 2


def test_touche_n_sans_visage_sans_effet(monkeypatch):
    image_vide = np.zeros((240, 320, 3), np.uint8)
    assert compter_grimaces(monkeypatch, image_vide, ["n", "q"]) == []


def test_lissage_de_la_pose_reduit_le_tremblement(monkeypatch):
    from main import Etat, lisser_la_pose
    from recalage import Grimace
    from simulation import portraits_de_test

    etat = Etat(portraits_de_test())
    bruit = np.random.default_rng(0).normal(0, 1.0, 200)
    vide = np.zeros((1, 2), np.float32)
    lisses = [lisser_la_pose(etat, Grimace(vide, vide, 0.0, float(b), 0.0)).lacet for b in bruit]
    assert np.std(lisses[20:]) < np.std(bruit[20:]) / 2
