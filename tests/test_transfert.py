"""Tests du transfert, sans webcam. Lancement : python -m pytest tests

Pour les tests de bout en bout, un portrait déformé joue le rôle de la personne
filmée, et le premier portrait de la liste doit reproduire sa grimace.
"""
import cv2
import numpy as np
import pytest

from demo_deformation import SCENARIOS
from deformation import animer, triangles_retournes
from recalage import Grimace, calculer_grimace_3d
from reperes import (
    GROUPES, LEVRE_BASSE_EXTERIEURE, LEVRE_BASSE_INTERIEURE, NOMBRE_REPERES, PAUPIERES,
    DetecteurReperes, positions,
)
from simulation import lancer_boucle, portraits_de_test
from test_recalage import tourner_la_tete_3d
from transfert import (
    AIRE_REPLI_TOLEREE, GAIN_POSE, GAIN_REGARD, LACET_MAX, OUVERTURE_FERMEE, ROULIS_MAX,
    ZONE_MORTE_POSE, ZONE_MORTE_REGARD, angle_pour_le_portrait, appliquer_pose, calculer_cible, distance_entre_les_yeux, facteur_echelle,
    limiter_deplacements, mesurer_regard, ouverture_bouche_mesuree, ouverture_portrait,
    ouverture_relative,
    position_iris, transferer,
)

SANS_TOUCHE = "x"
NB_POINTS_BORD = 8


def personne_filmee(nom_scenario):
    """Renvoie un visage neutre et le même visage en train de grimacer."""
    modele = portraits_de_test()[-1]
    cible = SCENARIOS[nom_scenario](modele.points)
    return modele.image, animer(modele.image, modele.points, cible, modele.triangles)


def moitie_droite(affichage, portrait):
    return affichage[:, -portrait.image.shape[1]:]


@pytest.fixture(scope="module")
def neutre_3d():
    with DetecteurReperes() as detecteur:
        return detecteur.detecter_3d(portraits_de_test()[-1].image)


def grimace_oeil_ferme(neutre_3d, cote, ouverture):
    """Construit une grimace où seule la paupière haute d'un œil descend."""
    recales = neutre_3d[:, :2].copy()
    basse, haute = (positions(paupiere) for paupiere in PAUPIERES[cote])
    recales[haute] = recales[basse] + ouverture * (recales[haute] - recales[basse])
    return Grimace(recales, recales - neutre_3d[:, :2], angle_tete=0.0)


def ouverture_sur_le_portrait(portrait, cible, cote):
    return ouverture_relative(cible, portrait.points, cote)


def test_echelle_egale_le_rapport_des_distances():
    points = np.zeros((NOMBRE_REPERES, 2), np.float32)
    droit, gauche = positions([33, 263])
    points[droit], points[gauche] = (0, 0), (100, 0)
    assert distance_entre_les_yeux(points) == 100
    assert facteur_echelle(points * 2, points) == pytest.approx(2)


def test_points_de_bord_immobiles():
    points = np.ones((NOMBRE_REPERES + NB_POINTS_BORD, 2), np.float32)
    delta = np.full((NOMBRE_REPERES, 2), 5, np.float32)
    cible = calculer_cible(points, delta, 2.0)
    assert np.allclose(cible[:NOMBRE_REPERES], 11)
    assert np.allclose(cible[NOMBRE_REPERES:], 1)


def test_limite_garde_la_direction():
    points = np.float32([[0, 0], [0, 0]])
    cible = np.float32([[30, 40], [3, 4]])
    limite = limiter_deplacements(points, cible, 10.0)
    assert np.allclose(limite, [[6, 8], [3, 4]])


@pytest.mark.parametrize("mesure, attendue", [
    (1.0, 1.0), (OUVERTURE_FERMEE, 0.0), (0.1, 0.0), (1.2, 1.2), (5.0, 1.5),
])
def test_conversion_de_l_ouverture(mesure, attendue):
    assert ouverture_portrait(mesure, 1.0) == pytest.approx(attendue)


def test_intensite_nulle_garde_l_oeil_au_repos():
    assert ouverture_portrait(0.0, 0.0) == 1.0


def test_un_oeil_ferme_se_ferme_entierement_sur_chaque_portrait(neutre_3d):
    grimace = grimace_oeil_ferme(neutre_3d, "droite", ouverture=0.25)
    for portrait in portraits_de_test():
        cible = transferer(portrait, grimace, neutre_3d)
        assert ouverture_sur_le_portrait(portrait, cible, "droite") < 0.05
        assert ouverture_sur_le_portrait(portrait, cible, "gauche") > 0.95


@pytest.mark.parametrize("cote", ["droite", "gauche"])
def test_pupille_reste_entre_les_paupieres(neutre_3d, cote):
    iris = positions(GROUPES["iris_droit" if cote == "droite" else "iris_gauche"])[0]
    basse, haute = PAUPIERES[cote]
    contour = positions(basse + haute[::-1][1:-1])
    for ouverture in (1.0, 0.7, 0.5):
        grimace = grimace_oeil_ferme(neutre_3d, cote, ouverture)
        for portrait in portraits_de_test():
            cible = transferer(portrait, grimace, neutre_3d)
            x, y = cible[iris]
            assert cv2.pointPolygonTest(cible[contour], (float(x), float(y)), False) >= 0


@pytest.mark.parametrize("ouverture", [1.0, 0.7, 0.5, 0.25])
def test_aucun_repli_visible_quand_l_oeil_se_ferme(neutre_3d, ouverture):
    for cote in PAUPIERES:
        grimace = grimace_oeil_ferme(neutre_3d, cote, ouverture)
        for portrait in portraits_de_test():
            cible = transferer(portrait, grimace, neutre_3d)
            assert triangles_retournes(
                portrait.points, cible, portrait.triangles_exterieurs, AIRE_REPLI_TOLEREE
            ) == 0


def grimace_de_pose(neutre_3d, lacet=0.0, tangage=0.0, roulis=0.0):
    """Construit une grimace nulle pour une tête tournée de ces angles."""
    xy = neutre_3d[:, :2].copy()
    return Grimace(xy, np.zeros_like(xy), roulis, lacet, tangage)


@pytest.mark.parametrize("angle, attendu", [(10.0, 10.0), (-40.0, -ROULIS_MAX)])
def test_roulis_limite_et_points_de_bord_fixes(neutre_3d, angle, attendu):
    portrait = portraits_de_test()[0]
    grimace = grimace_de_pose(neutre_3d, roulis=angle)
    cible = appliquer_pose(portrait, portrait.points, grimace)
    assert np.array_equal(cible[NOMBRE_REPERES:], portrait.points[NOMBRE_REPERES:])
    centre = portrait.points[:NOMBRE_REPERES].mean(axis=0)
    avant, apres = portrait.points[0] - centre, cible[0] - centre
    mesure = np.degrees(np.arctan2(apres[1], apres[0]) - np.arctan2(avant[1], avant[0]))
    assert (mesure + 180) % 360 - 180 == pytest.approx(attendu, abs=0.1)


def test_angle_pour_le_portrait_est_progressif_et_borne():
    assert angle_pour_le_portrait(ZONE_MORTE_POSE / 2, LACET_MAX) == 0.0
    valeurs = [angle_pour_le_portrait(a, LACET_MAX) for a in np.arange(0.0, 90.0, 1.0)]
    assert all(b >= a for a, b in zip(valeurs, valeurs[1:]))
    assert max(np.diff(valeurs)) <= GAIN_POSE + 1e-6  # pas de saut
    assert valeurs[-1] <= LACET_MAX
    assert angle_pour_le_portrait(-30.0, LACET_MAX) == -angle_pour_le_portrait(30.0, LACET_MAX)


def test_le_front_et_le_menton_bougent_moins_que_le_milieu(neutre_3d):
    milieu, front, menton = positions([168])[0], positions([10])[0], positions([152])[0]
    for portrait in portraits_de_test():
        cible = appliquer_pose(portrait, portrait.points, grimace_de_pose(neutre_3d, lacet=30.0))
        deplacement = np.abs(cible[:, 0] - portrait.points[:, 0])
        assert deplacement[front] < deplacement[milieu] / 2
        assert deplacement[menton] < deplacement[milieu] / 2


def test_tete_presque_immobile_portrait_immobile(neutre_3d):
    grimace = grimace_de_pose(neutre_3d, lacet=1.0, tangage=-1.0)
    for portrait in portraits_de_test():
        cible = appliquer_pose(portrait, portrait.points, grimace)
        assert np.abs(cible - portrait.points).max() < 0.01


@pytest.mark.parametrize("lacet", [-20.0, 20.0])
def test_le_nez_du_portrait_part_du_meme_cote_que_le_mien(neutre_3d, lacet):
    nez = positions([4])[0]
    tournee = tourner_la_tete_3d(neutre_3d, lacet=lacet)
    mon_deplacement = tournee[nez, 0] - tournee[:, 0].mean()
    au_repos = neutre_3d[nez, 0] - neutre_3d[:, 0].mean()
    grimace = calculer_grimace_3d(tournee, neutre_3d)
    for portrait in portraits_de_test():
        cible = transferer(portrait, grimace, neutre_3d)
        deplacement = cible[nez, 0] - portrait.points[nez, 0]
        assert deplacement == 0 or np.sign(deplacement) == np.sign(mon_deplacement - au_repos)
        assert np.array_equal(cible[NOMBRE_REPERES:], portrait.points[NOMBRE_REPERES:])


def test_chaque_portrait_tourne_nettement_d_au_moins_un_cote(neutre_3d):
    nez = positions([4])[0]
    for portrait in portraits_de_test():
        ampleurs = [
            abs(transferer(portrait, grimace_de_pose(neutre_3d, lacet=lacet), neutre_3d)[nez, 0]
                - portrait.points[nez, 0])
            for lacet in (-25.0, 25.0)
        ]
        assert min(ampleurs) > 15  # l'ellipsoïde laisse tourner des deux côtés


@pytest.mark.parametrize("lacet", [-25.0, 25.0])
def test_tourner_la_tete_deplace_tout_le_visage_et_garde_la_silhouette(neutre_3d, lacet):
    for portrait in portraits_de_test():
        cible = transferer(portrait, grimace_de_pose(neutre_3d, lacet=lacet), neutre_3d)
        repos = portrait.points[:NOMBRE_REPERES]
        # Les yeux, la bouche et le nez partent tous du même côté.
        for groupe in ("oeil_droit", "oeil_gauche", "levres", "nez"):
            traits = positions(GROUPES[groupe])
            moyenne = (cible[traits, 0] - portrait.points[traits, 0]).mean()
            assert np.sign(moyenne) == -np.sign(lacet)
            assert abs(moyenne) > 4
        # La silhouette reste dans sa boîte, avec une marge pour le relief.
        largeur = repos[:, 0].max() - repos[:, 0].min()
        assert cible[:NOMBRE_REPERES, 0].min() >= repos[:, 0].min() - 0.15 * largeur
        assert cible[:NOMBRE_REPERES, 0].max() <= repos[:, 0].max() + 0.15 * largeur
        assert np.array_equal(cible[NOMBRE_REPERES:], portrait.points[NOMBRE_REPERES:])
        assert triangles_retournes(
            portrait.points, cible, portrait.triangles_exterieurs, AIRE_REPLI_TOLEREE
        ) == 0


def test_sens_de_rotation_sur_de_vraies_donnees():
    # Un portrait et son miroir regardent de part et d'autre : c'est une vraie mesure
    # de MediaPipe, sans rotation simulée. Le nez se déplace vers la droite de l'image.
    modele = portraits_de_test()[1]
    with DetecteurReperes() as detecteur:
        repos = detecteur.detecter_3d(modele.image)
        miroir = detecteur.detecter_3d(cv2.flip(modele.image, 1))
    nez, tempes = positions([4])[0], positions([127, 356])
    def decalage(points):
        return points[nez, 0] - points[tempes, 0].mean()
    grimace = calculer_grimace_3d(miroir, repos)
    assert abs(grimace.lacet) > 5
    sens_du_nez = np.sign(decalage(miroir) - decalage(repos))
    portrait = portraits_de_test()[0]
    cible = appliquer_pose(portrait, portrait.points, grimace)
    deplacement = cible[nez, 0] - portrait.points[nez, 0]
    assert np.sign(deplacement) == sens_du_nez


def test_rotation_sans_correction_du_garde_fou(neutre_3d):
    # Si le garde-fou devait intervenir, il créerait des zigzags sur le contour.
    for lacet, tangage in ((25, 0), (-25, 0), (0, 15), (0, -15), (20, 10), (-20, -10)):
        grimace = grimace_de_pose(neutre_3d, lacet=lacet, tangage=tangage)
        for portrait in portraits_de_test():
            pose_seule = appliquer_pose(portrait, portrait.points, grimace)
            assert triangles_retournes(
                portrait.points, pose_seule, portrait.triangles_exterieurs, AIRE_REPLI_TOLEREE
            ) == 0


def regard_vers(neutre_3d, ampleur):
    """Construit une grimace où les deux iris glissent à droite de l'image."""
    recales = neutre_3d[:, :2].copy()
    for cote, nom in (("droite", "iris_droit"), ("gauche", "iris_gauche")):
        basse = positions(PAUPIERES[cote][0])
        largeur = abs(recales[basse[0], 0] - recales[basse[-1], 0])
        recales[positions(GROUPES[nom])[0], 0] += ampleur * largeur
    return Grimace(recales, recales - neutre_3d[:, :2], angle_tete=0.0)


@pytest.mark.parametrize("ampleur", [-0.08, 0.08])
def test_les_pupilles_du_portrait_suivent_mon_regard(neutre_3d, ampleur):
    grimace = regard_vers(neutre_3d, ampleur)
    assert mesurer_regard(grimace.reperes_recales, neutre_3d[:, :2]) == pytest.approx(
        np.sign(ampleur) * (abs(ampleur) - ZONE_MORTE_REGARD), abs=0.01  # œil un peu incliné
    )
    for portrait in portraits_de_test():
        cible = transferer(portrait, grimace, neutre_3d)
        for cote, nom in (("droite", "iris_droit"), ("gauche", "iris_gauche")):
            iris = positions(GROUPES[nom])[0]
            basse = positions(PAUPIERES[cote][0])
            largeur = abs(portrait.points[basse[0], 0] - portrait.points[basse[-1], 0])
            deplacement = cible[iris, 0] - portrait.points[iris, 0]
            attendu = GAIN_REGARD * (abs(ampleur) - ZONE_MORTE_REGARD) * largeur
            # Moins si la pupille approche déjà de la limite permise près d'un coin.
            assert np.sign(deplacement) == np.sign(ampleur) or abs(deplacement) < 0.5
            assert abs(deplacement) <= attendu + 1


def test_petit_mouvement_du_regard_ignore(neutre_3d):
    grimace = regard_vers(neutre_3d, ZONE_MORTE_REGARD / 2)
    assert mesurer_regard(grimace.reperes_recales, neutre_3d[:, :2]) == 0.0


def test_pupille_loin_des_coins_meme_avec_un_regard_extreme(neutre_3d):
    grimace = regard_vers(neutre_3d, 0.5)
    for portrait in portraits_de_test():
        cible = transferer(portrait, grimace, neutre_3d)
        for cote in PAUPIERES:
            assert position_iris(cible, cote) < 0.9


def grimace_de_sourire(neutre_3d):
    """Sourire réaliste : la lèvre s'étire d'autant plus qu'on s'éloigne de son centre."""
    recales = neutre_3d[:, :2].copy()
    ecart = distance_entre_les_yeux(neutre_3d[:, :2])
    levres = positions(GROUPES["levres"])
    centre = recales[levres, 0].mean()
    demi_largeur = np.abs(recales[levres, 0] - centre).max()
    position = (recales[levres, 0] - centre) / demi_largeur
    recales[levres, 0] += 0.09 * ecart * position
    recales[levres, 1] -= 0.06 * ecart * position**2
    return Grimace(recales, recales - neutre_3d[:, :2], angle_tete=0.0)


def test_le_garde_fou_conserve_le_sourire(neutre_3d):
    coins = positions([61, 291])
    grimace = grimace_de_sourire(neutre_3d)
    for portrait in portraits_de_test():
        echelle = facteur_echelle(portrait.points, neutre_3d[:, :2])
        voulu = np.linalg.norm(echelle * 1.5 * grimace.delta[coins], axis=1)
        cible = transferer(portrait, grimace, neutre_3d)
        obtenu = np.linalg.norm(cible[coins] - portrait.points[coins], axis=1)
        # De trois quarts, le menton est proche du coin de la bouche : le garde-fou le
        # retient, mais il ne doit jamais annuler le sourire.
        assert (obtenu > 0.5 * voulu).all()
        assert triangles_retournes(
            portrait.points, cible, portrait.triangles_exterieurs, AIRE_REPLI_TOLEREE
        ) == 0


def test_le_sourire_est_amplifie_sur_le_portrait(neutre_3d):
    portrait = portraits_de_test()[0]
    coin = positions([61])[0]
    recales = neutre_3d[:, :2].copy()
    recales[coin] += (0.0, -5.0)
    grimace = Grimace(recales, recales - neutre_3d[:, :2], angle_tete=0.0)
    cible = transferer(portrait, grimace, neutre_3d)
    echelle = facteur_echelle(portrait.points, neutre_3d[:, :2])
    assert cible[coin, 1] - portrait.points[coin, 1] == pytest.approx(-5.0 * echelle * 1.5, abs=0.5)


def test_tete_droite_sans_grimace_portrait_au_repos(neutre_3d):
    grimace = calculer_grimace_3d(neutre_3d, neutre_3d)
    for portrait in portraits_de_test():
        cible = transferer(portrait, grimace, neutre_3d)
        assert np.abs(cible - portrait.points).max() < 1.0


def test_intensite_change_l_ampleur(neutre_3d):
    portrait = portraits_de_test()[0]
    recales = neutre_3d[:, :2].copy()
    sourcils = positions(GROUPES["sourcil_droit"])
    recales[sourcils, 1] -= 4.0
    grimace = Grimace(recales, recales - neutre_3d[:, :2], angle_tete=0.0)
    ampleurs = [
        float(np.abs(transferer(portrait, grimace, neutre_3d, i) - portrait.points)[sourcils].max())
        for i in (0.0, 1.0, 2.0)
    ]
    assert ampleurs[0] < 0.01 and ampleurs[1] > 1 and ampleurs[2] > ampleurs[1] * 1.5


def test_portrait_immobile_avant_la_touche_n(monkeypatch):
    neutre, grimace = personne_filmee("bouche_ouverte")
    portrait = portraits_de_test()[0]
    affichees = lancer_boucle(monkeypatch, [neutre, grimace], ["d", SANS_TOUCHE, "q"])
    assert np.array_equal(moitie_droite(affichees[-1], portrait)[120:], portrait.image[120:])


def test_la_bouche_de_la_personne_filmee_ouvre_celle_du_portrait(monkeypatch):
    neutre, grimace = personne_filmee("bouche_ouverte")
    portrait = portraits_de_test()[0]
    affichees = lancer_boucle(
        monkeypatch, [neutre, neutre, grimace], ["d", "n", SANS_TOUCHE, SANS_TOUCHE, "q"]
    )
    rendu = moitie_droite(affichees[-1], portrait)
    levres = portrait.points[positions(GROUPES["levres"])]
    x0, y0 = levres.min(axis=0).astype(int)
    x1, y1 = levres.max(axis=0).astype(int)
    assert cv2.absdiff(rendu, portrait.image)[y0:y1 + 30, x0:x1].mean() > 3


def test_sans_visage_le_portrait_garde_son_dernier_etat(monkeypatch):
    neutre, grimace = personne_filmee("bouche_ouverte")
    vide = np.zeros_like(neutre)
    portrait = portraits_de_test()[0]
    affichees = lancer_boucle(
        monkeypatch, [neutre, neutre, grimace, vide], ["d", "n", SANS_TOUCHE, SANS_TOUCHE, "q"]
    )
    avant, apres = (moitie_droite(a, portrait)[120:] for a in affichees[-3:-1])
    assert np.array_equal(avant, apres)


def test_touche_p_change_de_portrait(monkeypatch):
    neutre, _ = personne_filmee("sourire")
    portraits = portraits_de_test()
    affichees = lancer_boucle(monkeypatch, [neutre], ["d", "p", SANS_TOUCHE, "q"])
    assert np.array_equal(moitie_droite(affichees[-1], portraits[1])[120:], portraits[1].image[120:])


def test_touches_majuscules(monkeypatch):
    neutre, _ = personne_filmee("sourire")
    assert len(lancer_boucle(monkeypatch, [neutre], ["D", "Q"])) == 2


@pytest.mark.parametrize("touches, attendue", [
    (["+", "q"], 1.1), (["=", "=", "q"], 1.2), (["-", "q"], 0.9), (["_", "q"], 0.9),
    (["+"] * 15 + ["q"], 2.0), (["-"] * 15 + ["q"], 0.0),
])
def test_touches_d_intensite(touches, attendue):
    from main import Etat, appliquer_touche

    etat = Etat(portraits_de_test())
    for touche in touches[:-1]:
        assert appliquer_touche(etat, ord(touche), None)
    assert etat.intensite == pytest.approx(attendue)


def etat_pour_choisir(monkeypatch, chemin):
    from main import Etat
    import main

    monkeypatch.setattr(main, "choisir_image", lambda: str(chemin))
    return Etat(portraits_de_test())


def test_touche_o_affiche_l_image_choisie(monkeypatch, tmp_path):
    from main import appliquer_touche

    # Un chemin avec un accent, comme un fichier choisi à la main sous Windows.
    chemin = tmp_path / "été.jpg"
    assert cv2.imencode(".jpg", portraits_de_test()[0].image)[1].tofile(str(chemin)) is None
    etat = etat_pour_choisir(monkeypatch, chemin)
    assert appliquer_touche(etat, ord("o"), None)
    assert len(etat.portraits) == len(portraits_de_test()) + 1
    assert etat.indice_portrait == len(etat.portraits) - 1
    assert etat.rendu is etat.portraits[-1].image
    assert etat.message is None


def test_touche_o_sans_visage_garde_le_portrait_et_previent(monkeypatch, tmp_path):
    from main import appliquer_touche

    chemin = tmp_path / "vide.jpg"
    cv2.imwrite(str(chemin), np.full((300, 300, 3), 120, np.uint8))
    etat = etat_pour_choisir(monkeypatch, chemin)
    appliquer_touche(etat, ord("o"), None)
    assert etat.indice_portrait == 0 and len(etat.portraits) == len(portraits_de_test())
    assert "Aucun visage" in etat.message[0]
    # Le message reste sans touche, et disparaît à la touche suivante.
    appliquer_touche(etat, 255, None)
    assert etat.message is not None
    appliquer_touche(etat, ord("d"), None)
    assert etat.message is None


def test_touche_o_annulee_ne_change_rien(monkeypatch):
    from main import appliquer_touche

    etat = etat_pour_choisir(monkeypatch, "")
    appliquer_touche(etat, ord("o"), None)
    assert etat.indice_portrait == 0 and etat.message is None


def test_fichier_illisible_ne_plante_pas(tmp_path):
    from deformation import charger_un_portrait

    texte = tmp_path / "pas_une_image.jpg"
    texte.write_text("ceci n'est pas une image")
    vide = tmp_path / "vide.jpg"
    vide.write_bytes(b"")
    assert charger_un_portrait(texte) is None
    assert charger_un_portrait(vide) is None
    assert charger_un_portrait(tmp_path / "absent.jpg") is None


def test_le_message_s_affiche_dans_la_boucle(monkeypatch):
    import main

    messages = []
    ecrire_reel = main.ecrire
    monkeypatch.setattr(main, "ecrire", lambda image, texte, ligne, *c: (messages.append(texte), ecrire_reel(image, texte, ligne, *c)))
    monkeypatch.setattr(main, "choisir_image", lambda: "absent.jpg")
    neutre, _ = personne_filmee("sourire")
    lancer_boucle(monkeypatch, [neutre], ["o", SANS_TOUCHE, "q"])
    assert any("Aucun visage" in texte for texte in messages)


def visage_de_webcam(neutre_3d, ecart=80.0):
    """Un visage de la taille d'une webcam : l'échelle vers le portrait amplifie tout."""
    return neutre_3d * (ecart / distance_entre_les_yeux(neutre_3d[:, :2]))


def bouche_decalee(neutre, ecart_px):
    """Descend toute la lèvre basse d'un bloc, comme une vraie lèvre qui bouge."""
    decale = neutre.copy()
    coins = {61, 291, 78, 308}
    levre = [i for i in LEVRE_BASSE_EXTERIEURE + LEVRE_BASSE_INTERIEURE if i not in coins]
    decale[positions(levre), 1] += ecart_px
    return decale


def rendu_de(portrait, reperes, neutre):
    from deformation import animer_portrait

    grimace = calculer_grimace_3d(reperes, neutre)
    cible = transferer(portrait, grimace, neutre)
    return cible, animer_portrait(portrait, cible, ouverture_bouche_mesuree(grimace, neutre))


@pytest.mark.parametrize("ecart_px", [0.0, 0.5, 1.0, 1.5])
def test_levre_qui_se_detend_pas_de_trou(neutre_3d, ecart_px):
    # 1 px de lèvre sur la webcam devient près de 3 px sur le portrait : c'était le trou.
    neutre = visage_de_webcam(neutre_3d)
    for portrait in portraits_de_test():
        cible, rendu = rendu_de(portrait, bouche_decalee(neutre, ecart_px), neutre)
        sans = animer(portrait.image, portrait.points, cible, portrait.triangles)
        assert np.array_equal(rendu, sans)


@pytest.mark.parametrize("angles", [
    {"lacet": 25.0}, {"lacet": -25.0}, {"tangage": 15.0}, {"tangage": -15.0}, {"roulis": 15.0},
])
def test_tete_qui_bouge_bouche_fermee_pas_de_trou(neutre_3d, angles):
    neutre = visage_de_webcam(neutre_3d)
    for portrait in portraits_de_test():
        cible, rendu = rendu_de(portrait, tourner_la_tete_3d(neutre, **angles), neutre)
        sans = animer(portrait.image, portrait.points, cible, portrait.triangles)
        assert np.array_equal(rendu, sans)


def test_vraie_bouche_ouverte_montre_l_interieur(neutre_3d):
    neutre = visage_de_webcam(neutre_3d)
    for portrait in portraits_de_test():
        cible, rendu = rendu_de(portrait, bouche_decalee(neutre, 20.0), neutre)
        sans = animer(portrait.image, portrait.points, cible, portrait.triangles)
        assert cv2.absdiff(rendu, sans).sum() > 0


def test_intensite_nulle_pas_de_trou(neutre_3d):
    neutre = visage_de_webcam(neutre_3d)
    grimace = calculer_grimace_3d(bouche_decalee(neutre, 20.0), neutre)
    assert ouverture_bouche_mesuree(grimace, neutre, intensite=0.0) == 0.0
