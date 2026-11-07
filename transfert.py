"""Transfère la grimace mesurée sur la webcam vers les points du portrait."""
import numpy as np

from deformation import masque_retournes, ouverture_relative_bouche, triangles_retournes
from recalage import rotation_depuis_angles
from reperes import GROUPES, NOMBRE_REPERES, PAUPIERES, positions

COINS_EXTERIEURS_DES_YEUX = positions([33, 263])
OUVERTURE_FERMEE = 0.35  # ouverture relative à partir de laquelle l'œil est fermé
OUVERTURE_MAX = 1.5  # limite pour un regard très écarquillé
LIMITE_DEPLACEMENT = 0.8  # déplacement maximal, en proportion de l'écart des yeux
ROULIS_MAX, LACET_MAX, TANGAGE_MAX = 15.0, 14.0, 7.0  # rotations limitées, en degrés
GAIN_POSE = 0.6  # part de ma rotation transmise au portrait ; 0 la désactive
ZONE_MORTE_POSE = 1.5  # en degrés : en dessous, c'est du bruit de mesure
RAYON_MINIMAL = 0.15  # plus petit rayon de l'ellipsoïde, en proportion du rayon central
GAINS = {"levres": 1.5, "joues": 1.3, "sourcil_droit": 1.3, "sourcil_gauche": 1.3}
GAIN_REGARD = 2.5  # l'iris bouge peu dans l'œil : on amplifie son déplacement
ZONE_MORTE_REGARD = 0.02  # en largeur d'œil : en dessous, c'est du bruit de mesure
POSITION_IRIS_MIN, POSITION_IRIS_MAX = 0.2, 0.8  # la pupille reste loin des coins
# En pixels carrés. Un repli plus petit reste invisible, et les coins d'un œil qui se
# ferme en produisent de cet ordre : les corriger rouvrirait toute la paupière.
AIRE_REPLI_TOLEREE = 8.0
NB_ESSAIS_GARDE_FOU = 8
POIDS_RELIEF = 0.5  # part du relief (nez, lèvres) ajoutée au mouvement du cylindre
NB_ESSAIS_PARALLAXE = 6  # précision de la recherche par dichotomie : 1/64 du mouvement
IRIS_PAR_COTE = {"droite": "iris_droit", "gauche": "iris_gauche"}


def distance_entre_les_yeux(points):
    """Mesure l'écart entre les coins extérieurs des yeux, qui bougent peu."""
    droit, gauche = COINS_EXTERIEURS_DES_YEUX
    return float(np.linalg.norm(points[droit] - points[gauche]))


def facteur_echelle(points_portrait, neutre):
    """Calcule s : le rapport de taille entre le visage du portrait et le mien."""
    return distance_entre_les_yeux(points_portrait) / distance_entre_les_yeux(neutre)


def calculer_cible(points_portrait, delta, echelle):
    """Calcule P_cible = P_portrait + s · Δ. Les points de bord, à la fin, ne bougent pas."""
    cible = points_portrait.copy()
    cible[: len(delta)] += echelle * delta
    return cible


def limiter_deplacements(points, cible, maximum):
    """Réduit chaque déplacement trop long, sans changer sa direction."""
    deplacement = cible - points
    normes = np.linalg.norm(deplacement, axis=1, keepdims=True)
    reduction = np.minimum(1.0, maximum / np.maximum(normes, 1e-6))
    return points + deplacement * reduction


def ouverture_relative(reperes, neutre, cote):
    """Rapport entre l'ouverture actuelle de l'œil et son ouverture au repos."""
    basse, haute = (positions(paupiere) for paupiere in PAUPIERES[cote])
    actuelle = np.linalg.norm(reperes[haute] - reperes[basse], axis=1).mean()
    repos = np.linalg.norm(neutre[haute] - neutre[basse], axis=1).mean()
    return float(actuelle / max(repos, 1e-6))


def ouverture_portrait(ouverture, intensite):
    """Convertit l'ouverture mesurée en ouverture à donner à l'œil du portrait.

    Même sur un œil bien fermé, les repères de MediaPipe ne se touchent pas : tout ce
    qui est sous OUVERTURE_FERMEE compte donc comme fermé. L'échelle est ensuite étirée
    pour que 1 reste l'œil au repos.
    """
    if ouverture < 1.0:
        ouverture = max(ouverture - OUVERTURE_FERMEE, 0.0) / (1.0 - OUVERTURE_FERMEE)
    ouverture = min(ouverture, OUVERTURE_MAX)
    return max(1.0 + (ouverture - 1.0) * intensite, 0.0)


def position_iris(points, cote):
    """Position de l'iris entre les coins de l'œil : 0 à gauche de l'image, 1 à droite."""
    basse = positions(PAUPIERES[cote][0])
    coin_a, coin_b = points[basse[0]], points[basse[-1]]
    gauche, droite = (coin_a, coin_b) if coin_a[0] <= coin_b[0] else (coin_b, coin_a)
    axe = droite - gauche
    iris = points[positions(GROUPES[IRIS_PAR_COTE[cote]])[0]]
    return float(np.dot(iris - gauche, axe) / max(np.dot(axe, axe), 1e-6))


def mesurer_regard(reperes, neutre):
    """Déplacement horizontal du regard, en largeur d'œil, moyenné sur les deux yeux.

    Les deux yeux regardent dans la même direction : la moyenne réduit le bruit de
    mesure et évite que le portrait louche. Filtrage : une zone morte retire le
    tremblement restant quand le regard ne bouge pas.
    """
    regard = float(np.mean([
        position_iris(reperes, cote) - position_iris(neutre, cote) for cote in PAUPIERES
    ]))
    return float(np.sign(regard)) * max(abs(regard) - ZONE_MORTE_REGARD, 0.0)


def placer_oeil(cible, points, regard, ouverture_mesuree, cote, intensite):
    """Place la paupière haute et la pupille selon l'ouverture relative de l'œil.

    On transfère une proportion et non des pixels : l'œil du portrait se ferme donc
    entièrement, quelles que soient la taille de cet œil et celle de l'œil filmé.
    """
    basse, haute = (positions(paupiere) for paupiere in PAUPIERES[cote])
    ouverture = ouverture_portrait(ouverture_mesuree, intensite)
    cible[haute] = cible[basse] + ouverture * (points[haute] - points[basse])
    placer_pupille(cible, points, regard * intensite, cote, ouverture, basse, haute)


def placer_pupille(cible, points, regard, cote, ouverture, basse, haute):
    """Garde la pupille entre les paupières et lui applique le mouvement du regard.

    Elle garde sa hauteur relative entre les deux paupières : quand l'œil se ferme, elle
    est écrasée avec lui au lieu de traverser la paupière.
    """
    iris = positions(GROUPES[IRIS_PAR_COTE[cote]])[0]
    milieu = len(basse) // 2
    bas, haut = points[basse[milieu]], points[haute[milieu]]
    axe_vertical = haut - bas
    fraction = float(np.dot(points[iris] - bas, axe_vertical) / max(np.dot(axe_vertical, axe_vertical), 1e-6))
    decalage = points[iris] - (bas + fraction * axe_vertical)

    # L'iris se déplace le long de l'œil, sans s'approcher trop près des coins.
    depart = position_iris(points, cote)
    arrivee = np.clip(
        depart + GAIN_REGARD * regard,
        min(POSITION_IRIS_MIN, depart), max(POSITION_IRIS_MAX, depart),
    )
    coin_a, coin_b = cible[basse[0]], cible[basse[-1]]
    axe_oeil = np.abs(coin_b[0] - coin_a[0]) * np.float32([1, 0])
    mouvement = float(arrivee - depart) * axe_oeil

    base = cible[basse[milieu]] + fraction * (cible[haute[milieu]] - cible[basse[milieu]])
    cible[iris] = base + min(ouverture, 1.0) * (decalage + mouvement)


def relief_du_visage(portrait):
    """Renvoie la saillie de chaque repère devant le plan des coins extérieurs des yeux.

    Les points situés derrière ce plan (côtés de la tête) comptent pour zéro : une
    vraie rotation les ferait passer devant le visage, ce qu'une image 2D ne peut pas
    montrer sans se déchirer, car rien ne peut y être caché.
    """
    plan = portrait.profondeurs[COINS_EXTERIEURS_DES_YEUX].mean()
    return np.minimum(portrait.profondeurs - plan, 0.0)


def part_de_parallaxe_sure(portrait, cible, parallaxe):
    """Cherche par dichotomie la plus grande part du mouvement qui ne crée aucun repli.

    Sur un portrait de trois quarts, le nez atteint vite le bord de la joue : ce
    portrait tourne donc moins de ce côté, mais tout son visage reste cohérent.
    """
    triangles = portrait.triangles_exterieurs

    def nb_replis(part):
        decale = cible.copy()
        decale[:NOMBRE_REPERES] += part * parallaxe
        return triangles_retournes(portrait.points, decale, triangles, AIRE_REPLI_TOLEREE)

    deja_presents = nb_replis(0.0)
    if nb_replis(1.0) <= deja_presents:
        return 1.0
    sure, risquee = 0.0, 1.0
    for _ in range(NB_ESSAIS_PARALLAXE):
        milieu = (sure + risquee) / 2
        if nb_replis(milieu) <= deja_presents:
            sure = milieu
        else:
            risquee = milieu
    return sure


def angle_pour_le_portrait(angle, maximum):
    """Convertit une rotation mesurée en rotation à donner au portrait.

    Une zone morte retire le bruit, un gain atténue le mouvement, puis une tangente
    hyperbolique approche la limite en douceur au lieu de bloquer d'un coup.
    """
    utile = np.sign(angle) * max(abs(angle) - ZONE_MORTE_POSE, 0.0)
    return float(maximum * np.tanh(GAIN_POSE * utile / maximum))


def deplacement_sur_ellipsoide(valeurs, transverses, boite, axe, angle):
    """Calcule le déplacement selon un axe quand un ellipsoïde tourne d'un angle.

    Un point à la distance u·r de l'axe de rotation est à l'angle asin(u) sur la tête.
    Après rotation, il se projette en r·sin(asin(u) + angle). Le rayon r diminue vers
    le haut et le bas : le front et le menton bougent donc moins que le milieu du
    visage, et ne se détachent ni des cheveux ni du cou.
    """
    minimum, maximum = boite
    centre = (minimum[axe] + maximum[axe]) / 2
    rayon_central = (maximum[axe] - minimum[axe]) / 2
    autre = 1 - axe
    centre_autre = (minimum[autre] + maximum[autre]) / 2
    rayon_autre = (maximum[autre] - minimum[autre]) / 2
    v = np.clip((transverses - centre_autre) / rayon_autre, -1.0, 1.0)
    rayon = rayon_central * np.maximum(np.sqrt(1.0 - v**2), RAYON_MINIMAL)
    phi = np.arcsin(np.clip((valeurs - centre) / rayon, -1.0, 1.0))
    apres = np.clip(phi + np.radians(angle), -np.pi / 2, np.pi / 2)
    return rayon * (np.sin(apres) - np.sin(phi))


def appliquer_pose(portrait, cible, grimace):
    """Fait tourner la tête du portrait ; les points de bord ne bougent pas.

    La tête est vue comme un ellipsoïde (lacet et tangage), auquel s'ajoute le relief
    du visage : modèle de caméra du cours, avec les termes de R qui multiplient z. Comme
    dans x' = R·p, un lacet positif déplace vers la gauche un point proche de la caméra.
    """
    lacet = angle_pour_le_portrait(grimace.lacet, LACET_MAX)
    tangage = angle_pour_le_portrait(grimace.tangage, TANGAGE_MAX)
    roulis = float(np.clip(grimace.angle_tete, -ROULIS_MAX, ROULIS_MAX))

    repos = portrait.points[:NOMBRE_REPERES]
    boite = (repos.min(axis=0), repos.max(axis=0))
    cible = cible.copy()
    actuels = cible[:NOMBRE_REPERES]
    actuels[:, 0] += deplacement_sur_ellipsoide(actuels[:, 0], repos[:, 1], boite, 0, -lacet)
    actuels[:, 1] += deplacement_sur_ellipsoide(actuels[:, 1], repos[:, 0], boite, 1, tangage)

    rotation_plan = rotation_depuis_angles(0.0, 0.0, roulis)[:2, :2]
    centre = repos.mean(axis=0)
    actuels[:] = (actuels - centre) @ rotation_plan.T + centre

    termes_z = rotation_depuis_angles(lacet, tangage, roulis)[:2, 2]
    parallaxe = POIDS_RELIEF * np.outer(relief_du_visage(portrait), termes_z)
    actuels += part_de_parallaxe_sure(portrait, cible, parallaxe) * parallaxe
    return cible


def moyenner_les_replis(points, cible, triangles, replies):
    """Donne aux sommets de chaque triangle replié leur déplacement moyen.

    Trois sommets qui bougent ensemble forment une translation, qui ne peut rien
    replier : le mouvement est conservé et simplement partagé avec les points voisins.
    Les points de bord, à la fin du tableau, restent fixes.
    """
    deplacements = cible - points
    moyennes = deplacements[triangles[replies]].mean(axis=1)
    sommes = np.zeros_like(cible)
    comptes = np.zeros(len(cible), np.float32)
    for sommets, moyenne in zip(triangles[replies], moyennes):
        sommes[sommets] += moyenne
        comptes[sommets] += 1
    touches = np.nonzero(comptes[:NOMBRE_REPERES])[0]
    resultat = cible.copy()
    resultat[touches] = points[touches] + sommes[touches] / comptes[touches, np.newaxis]
    return resultat


def empecher_retournements(points, cible, triangles):
    """Dernier garde-fou : supprime les replis visibles que la grimace pourrait créer.

    On répartit d'abord le mouvement entre voisins. Si un repli résiste, on réduit le
    déplacement de ses sommets : comme aucun triangle n'est replié au repos, cela finit
    toujours par le supprimer.
    """
    for _ in range(NB_ESSAIS_GARDE_FOU):
        replies = masque_retournes(points, cible, triangles, AIRE_REPLI_TOLEREE)
        if not replies.any():
            return cible
        cible = moyenner_les_replis(points, cible, triangles, replies)
    cible = cible.copy()
    for _ in range(NB_ESSAIS_GARDE_FOU):
        replies = masque_retournes(points, cible, triangles, AIRE_REPLI_TOLEREE)
        if not replies.any():
            break
        sommets = np.unique(triangles[replies])
        sommets = sommets[sommets < NOMBRE_REPERES]
        cible[sommets] = points[sommets] + 0.5 * (cible[sommets] - points[sommets])
    return cible


def ouverture_bouche_mesuree(grimace, neutre, intensite=1.0):
    """Ouverture de la vraie bouche par rapport au visage neutre, tête remise en place."""
    return ouverture_relative_bouche(grimace.reperes_recales, neutre[:, :2]) * intensite


def gains_par_repere():
    """Amplifie les zones où le mouvement mesuré reste trop discret sur un portrait."""
    gains = np.ones((NOMBRE_REPERES, 1), np.float32)
    for groupe, gain in GAINS.items():
        gains[positions(GROUPES[groupe])] = gain
    return gains


GAINS_REPERES = gains_par_repere()


def transferer(portrait, grimace, neutre, intensite=1.0):
    """Enchaîne toutes les étapes qui mènent de la grimace mesurée à P_cible."""
    points = portrait.points
    neutre_xy = neutre[:, :2]
    echelle = facteur_echelle(points, neutre_xy) * intensite
    cible = calculer_cible(points, grimace.delta, echelle * GAINS_REPERES)
    cible = limiter_deplacements(
        points, cible, LIMITE_DEPLACEMENT * distance_entre_les_yeux(points)
    )
    regard = mesurer_regard(grimace.reperes_recales, neutre_xy)
    for cote in PAUPIERES:
        ouverture = ouverture_relative(grimace.reperes_recales, neutre_xy, cote)
        placer_oeil(cible, points, regard, ouverture, cote, intensite)
    cible = appliquer_pose(portrait, cible, grimace)
    return empecher_retournements(points, cible, portrait.triangles_exterieurs)
