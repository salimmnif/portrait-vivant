"""Tests sans webcam. Lancement depuis portrait_vivant : python -m pytest tests"""
import cv2
import numpy as np
import pytest

from demo_deformation import SCENARIOS, charger_portraits
from deformation import (
    OUVERTURE_DEBUT, OUVERTURE_PLEINE, aires_triangles, animer, animer_portrait,
    contour_levres_interieur, deformer_triangle, ouverture_bouche, points_de_bord,
    progression, trianguler, triangles_retournes,
)
from reperes import GROUPES, LEVRE_BASSE_INTERIEURE, PAUPIERES, TEMPES, positions
from transfert import AIRE_REPLI_TOLEREE, empecher_retournements

CARRE = 200


@pytest.fixture(scope="module")
def portraits():
    resultat = charger_portraits()
    assert resultat
    return resultat


@pytest.fixture
def grille():
    """Grille de points, triangulée, dont l'intérieur est déplacé au hasard."""
    valeurs = np.linspace(0, CARRE - 1, 6)
    source = np.float32([(x, y) for y in valeurs for x in valeurs])
    interieur = ((source > 0) & (source < CARRE - 1)).all(axis=1)
    cible = source.copy()
    aleatoire = np.random.default_rng(0)
    cible[interieur] += aleatoire.uniform(-8, 8, (interieur.sum(), 2)).astype(np.float32)
    return source, cible, trianguler(source)


def test_sans_deplacement_la_sortie_est_identique(portraits):
    # Le fond est noir : un triangle manquant laisserait donc une zone noire visible.
    for portrait in portraits.values():
        fond = np.zeros_like(portrait.image)
        sortie = animer(
            portrait.image, portrait.points, portrait.points, portrait.triangles, fond
        )
        assert cv2.absdiff(sortie, portrait.image).max() <= 3


def test_deformation_sans_joint_ni_trou(grille):
    source, cible, triangles = grille
    blanc = np.full((CARRE, CARRE, 3), 255, np.uint8)
    sortie = animer(blanc, source, cible, triangles, np.zeros_like(blanc))
    assert sortie.min() == 255


def test_le_portrait_d_origine_n_est_pas_modifie(grille):
    source, cible, triangles = grille
    image = np.random.default_rng(1).integers(0, 255, (CARRE, CARRE, 3), np.uint8)
    copie = image.copy()
    sortie = animer(image, source, cible, triangles)
    assert np.array_equal(image, copie)
    assert not np.array_equal(sortie, copie)


def test_triangle_a_cheval_sur_le_bord():
    src = np.full((50, 50, 3), 200, np.uint8)
    dst = np.zeros_like(src)
    deformer_triangle(src, dst, [(5, 5), (30, 5), (5, 30)], [(-10, -10), (20, -10), (-10, 20)])
    assert dst[2, 2].tolist() == [200, 200, 200]
    assert dst[40, 40].tolist() == [0, 0, 0]


@pytest.mark.parametrize("t_cible", [
    [(100, 100), (130, 100), (100, 130)],  # entièrement hors de l'image
    [(5, 5), (10, 10), (20, 20)],  # 3 points alignés
])
def test_triangle_ignore_sans_erreur(t_cible):
    src = np.full((50, 50, 3), 200, np.uint8)
    dst = np.zeros_like(src)
    deformer_triangle(src, dst, [(5, 5), (30, 5), (5, 30)], t_cible)
    assert dst.max() == 0


def test_triangle_source_degenere_ignore():
    src = np.full((50, 50, 3), 200, np.uint8)
    dst = np.zeros_like(src)
    deformer_triangle(src, dst, [(5, 5), (10, 10), (20, 20)], [(5, 5), (30, 5), (5, 30)])
    assert dst.max() == 0


def test_points_de_bord_dans_l_image():
    points = points_de_bord(100, 50)
    assert len(points) == 8
    assert points.min() == 0 and points[:, 0].max() == 99 and points[:, 1].max() == 49


def test_triangles_couvrent_toute_l_image(portraits):
    for portrait in portraits.values():
        hauteur, largeur = portrait.image.shape[:2]
        aire = aires_triangles(portrait.points, portrait.triangles).sum()
        assert aire == pytest.approx((largeur - 1) * (hauteur - 1), rel=1e-4)


def test_les_paupieres_sont_des_aretes_de_la_triangulation(portraits):
    for portrait in portraits.values():
        aretes = {
            frozenset((int(t[i]), int(t[j])))
            for t in portrait.triangles for i, j in ((0, 1), (1, 2), (0, 2))
        }
        for basse, haute in PAUPIERES.values():
            for paupiere in (positions(basse), positions(haute)):
                for a, b in zip(paupiere, paupiere[1:]):
                    assert frozenset((a, b)) in aretes


def test_centre_de_l_iris_dans_l_oeil(portraits):
    for portrait in portraits.values():
        for cote, iris in (("droite", "iris_droit"), ("gauche", "iris_gauche")):
            basse, haute = PAUPIERES[cote]
            polygone = portrait.points[positions(basse + haute[::-1][1:-1])]
            x, y = portrait.points[positions(GROUPES[iris])[0]]
            assert cv2.pointPolygonTest(polygone, (float(x), float(y)), False) > 0


def test_aucun_sommet_etranger_dans_les_yeux(portraits):
    for portrait in portraits.values():
        sommets = np.unique(portrait.triangles)
        assert not set(positions(TEMPES)) & set(sommets.tolist())
        for cote, nom_oeil, nom_iris in (
            ("droite", "oeil_droit", "iris_droit"), ("gauche", "oeil_gauche", "iris_gauche"),
        ):
            basse, haute = PAUPIERES[cote]
            polygone = portrait.points[positions(basse + haute[::-1][1:-1])]
            autorises = set(positions(GROUPES[nom_oeil] + GROUPES[nom_iris]))
            for sommet in set(sommets.tolist()) - autorises:
                x, y = portrait.points[sommet]
                assert cv2.pointPolygonTest(polygone, (float(x), float(y)), False) < 0


@pytest.mark.parametrize("nom_scenario", SCENARIOS)
def test_garde_fou_supprime_les_replis_visibles(portraits, nom_scenario):
    for portrait in portraits.values():
        cible = empecher_retournements(
            portrait.points, SCENARIOS[nom_scenario](portrait.points),
            portrait.triangles_exterieurs,
        )
        assert triangles_retournes(
            portrait.points, cible, portrait.triangles_exterieurs, AIRE_REPLI_TOLEREE
        ) == 0


def test_bouche_au_repos_rien_n_est_dessine(portraits):
    for portrait in portraits.values():
        sortie = animer_portrait(portrait, portrait.points)
        assert np.array_equal(sortie, portrait.image)


def centre_de_la_bouche_ouverte(cible):
    return tuple(np.int32(contour_levres_interieur(cible).mean(axis=0)))


def test_bouche_ouverte_interieur_peint(portraits):
    for portrait in portraits.values():
        cible = SCENARIOS["bouche_ouverte"](portrait.points)
        assert ouverture_bouche(cible) > ouverture_bouche(portrait.points) + 10
        sortie = animer_portrait(portrait, cible)
        sans_bouche = animer(portrait.image, portrait.points, cible, portrait.triangles)
        x, y = centre_de_la_bouche_ouverte(cible)
        assert cv2.absdiff(sortie, sans_bouche)[y, x].sum() > 0


def zone_bouche(cible, image):
    polygone = contour_levres_interieur(cible)
    x0, y0 = np.int32(polygone.min(axis=0))
    x1, y1 = np.int32(polygone.max(axis=0))
    return image[y0:y1, x0:x1]


def test_dents_claires_en_haut_et_gorge_plus_sombre_au_centre(portraits):
    for portrait in portraits.values():
        cible = SCENARIOS["bouche_ouverte"](portrait.points)
        zone = zone_bouche(cible, animer_portrait(portrait, cible)).astype(float)
        hauteur = zone.shape[0]
        milieu = zone.shape[1] // 2
        dents = zone[int(hauteur * 0.08):int(hauteur * 0.22), milieu - 3:milieu + 3].mean()
        gorge = zone[int(hauteur * 0.45):int(hauteur * 0.55), milieu - 3:milieu + 3].mean()
        assert dents > gorge


def test_petite_ouverture_sans_trou(portraits):
    portrait = portraits["joconde"]
    cible = portrait.points.copy()
    cible[positions(LEVRE_BASSE_INTERIEURE[1:-1]), 1] += 2.0
    sans = animer(portrait.image, portrait.points, cible, portrait.triangles)
    assert np.array_equal(animer_portrait(portrait, cible), sans)


def test_dents_seulement_quand_la_bouche_s_ouvre_franchement(portraits):
    portrait = portraits["joconde"]
    cible = portrait.points.copy()
    cible[positions(LEVRE_BASSE_INTERIEURE[1:-1]), 1] += 20.0
    # A 0,08, l'intérieur est entièrement peint, mais les dents n'ont pas encore commencé.
    sans_dents = zone_bouche(cible, animer_portrait(portrait, cible, ouverture=0.08)).astype(float)
    avec_dents = zone_bouche(cible, animer_portrait(portrait, cible, ouverture=0.20)).astype(float)
    haut, milieu = avec_dents.shape[0] // 4, avec_dents.shape[1] // 2
    assert avec_dents[:haut, milieu - 5:milieu + 5].mean() > sans_dents[:haut, milieu - 5:milieu + 5].mean() + 3


def test_dents_accordees_a_la_clarte_du_tableau(portraits):
    for portrait in portraits.values():
        bleu, vert, rouge = portrait.couleur_dents
        assert 130 <= rouge <= 240 and bleu <= vert <= rouge


def test_couleur_de_la_bouche_est_un_rouge_sombre(portraits):
    for portrait in portraits.values():
        bleu, vert, rouge = portrait.couleur_bouche
        assert 70 <= rouge <= 120
        assert rouge > 5 * vert and rouge > 5 * bleu  # du rouge, et non du marron


def test_interieur_apparait_progressivement(portraits):
    portrait = portraits["joconde"]
    cible = SCENARIOS["bouche_ouverte"](portrait.points)
    x, y = centre_de_la_bouche_ouverte(cible)
    sans = animer(portrait.image, portrait.points, cible, portrait.triangles)
    ecarts = [
        int(cv2.absdiff(animer_portrait(portrait, cible, ouverture=o), sans)[y, x].sum())
        for o in (0.02, 0.05, 0.10)
    ]
    assert ecarts[0] == 0 and ecarts[1] > 0 and ecarts[2] > 0
    assert progression(0.0375, OUVERTURE_DEBUT, OUVERTURE_PLEINE) == pytest.approx(0.5)


def test_le_dessin_reste_autour_de_la_bouche(portraits):
    for portrait in portraits.values():
        cible = SCENARIOS["bouche_ouverte"](portrait.points)
        sans = animer(portrait.image, portrait.points, cible, portrait.triangles)
        avec = animer_portrait(portrait, cible)
        lignes, colonnes = np.nonzero(cv2.absdiff(avec, sans).sum(axis=2))
        polygone = contour_levres_interieur(cible)
        assert lignes.min() >= polygone[:, 1].min() - 6 and lignes.max() <= polygone[:, 1].max() + 6
        assert colonnes.min() >= polygone[:, 0].min() - 6 and colonnes.max() <= polygone[:, 0].max() + 6


def test_bouche_hors_de_l_image_sans_erreur(portraits):
    portrait = portraits["joconde"]
    cible = SCENARIOS["bouche_ouverte"](portrait.points)
    from reperes import GROUPES as groupes

    cible[positions(groupes["levres"])] += 5000.0
    assert animer_portrait(portrait, cible).shape == portrait.image.shape
