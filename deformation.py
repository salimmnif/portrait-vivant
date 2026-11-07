"""Préparation du portrait, découpage en triangles, déformation, intérieur de la bouche."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay

from reperes import (
    GROUPES, LEVRE_BASSE_EXTERIEURE, LEVRE_BASSE_INTERIEURE, LEVRE_HAUTE_EXTERIEURE,
    LEVRE_HAUTE_INTERIEURE, PAUPIERES, TEMPES, DetecteurReperes, positions,
)

DOSSIER_PORTRAITS = Path(__file__).resolve().parent / "portraits"
EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png"}
HAUTEUR_PORTRAIT = 600  # hauteur en pixels du portrait une fois préparé
MARGE_RECADRAGE = 0.6  # marge autour du visage, en proportion de sa taille
AIRE_MINIMALE = 1e-3  # en pixels carrés : en dessous, le triangle est dégénéré
MARGE_SOURCE = 2  # pixels gardés autour du triangle source pour l'interpolation
ASSOMBRISSEMENT_BOUCHE = 0.55  # part de la luminosité des lèvres gardée dans la bouche
LUMINOSITE_BOUCHE_MIN, LUMINOSITE_BOUCHE_MAX = 75.0, 115.0  # reste un rouge sombre visible
TEINTE_BOUCHE = (0.13, 0.10, 1.0)  # proportions de B, V et R : un rouge presque pur
# Ouverture de la bouche, en proportion de l'écart entre les coins des yeux. Sous le
# début, c'est une lèvre qui se détend : aucun trou ne doit apparaître.
OUVERTURE_DEBUT, OUVERTURE_PLEINE = 0.02, 0.055
TAILLE_FLOU_BOUCHE = 5  # taille du filtre gaussien qui adoucit le bord de la bouche
CLARTE_BORD_BOUCHE, CLARTE_GORGE = 0.62, 0.22  # part de la couleur gardée, du bord au fond
ECLAIRCISSEMENT_LANGUE = 1.15  # la langue reste à peine plus claire que la bouche
AJOUT_ROSE_LANGUE = np.array([14.0, 16.0, 8.0])  # l'empêche de paraître rouge vif
OPACITE_LANGUE, OPACITE_DENTS = 0.7, 0.88  # laissent transparaître l'ombre de la bouche
ATTENUATION_DENTS = 0.8  # un ivoire dans l'ombre de la bouche, jamais un blanc pur
LARGEUR_DENTS = 0.72  # les dents ne vont pas jusqu'aux commissures
# Mêmes unités : les dents apparaissent quand la bouche s'ouvre franchement, puis la langue.
DENTS_DEBUT, DENTS_PLEINES = 0.08, 0.14
LANGUE_DEBUT, LANGUE_PLEINE = 0.14, 0.26
LUMINOSITE_DENTS_MIN, LUMINOSITE_DENTS_MAX = 140.0, 235.0  # selon la clarté de la peau
TEINTE_DENTS = (0.90, 0.96, 1.0)  # proportions de B, V et R : un blanc légèrement chaud
NB_DENTS = 8


@dataclass
class Portrait:
    """Portrait prêt à être animé. Les points de bord sont à la fin de `points`."""

    image: np.ndarray
    points: np.ndarray
    triangles: np.ndarray  # triangles intérieurs d'abord, voir separer_interieurs
    nb_interieurs: int = 0
    profondeurs: np.ndarray = None  # z des repères, à l'échelle de l'image préparée
    couleur_bouche: np.ndarray = None  # couleur BGR de l'intérieur de la bouche
    couleur_dents: np.ndarray = None  # couleur BGR des dents, selon la clarté de la peau

    @property
    def triangles_exterieurs(self):
        return self.triangles[self.nb_interieurs :]


def recadrer_sur_visage(image, reperes):
    """Coupe l'image autour du visage et décale les repères de la même translation."""
    hauteur, largeur = image.shape[:2]
    minimum, maximum = reperes.min(axis=0), reperes.max(axis=0)
    marge = (maximum - minimum) * MARGE_RECADRAGE
    x0, y0 = np.maximum(np.floor(minimum - marge).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(maximum + marge).astype(int), (largeur, hauteur))
    return image[y0:y1, x0:x1], reperes - np.float32([x0, y0])


def redimensionner(image, reperes, hauteur_voulue=HAUTEUR_PORTRAIT):
    """Met l'image à la hauteur voulue et applique la même échelle aux repères."""
    hauteur, largeur = image.shape[:2]
    largeur_voulue = round(largeur * hauteur_voulue / hauteur)
    # Image numérique : INTER_AREA évite le crénelage quand on réduit l'image.
    interpolation = cv2.INTER_AREA if hauteur_voulue < hauteur else cv2.INTER_CUBIC
    image = cv2.resize(image, (largeur_voulue, hauteur_voulue), interpolation=interpolation)
    return image, reperes * np.float32([largeur_voulue / largeur, hauteur_voulue / hauteur])


def points_de_bord(largeur, hauteur):
    """Renvoie 8 points fixes (4 coins, 4 milieux) qui immobilisent le fond et le cadre."""
    droite, bas = largeur - 1, hauteur - 1
    milieu_x, milieu_y = droite / 2, bas / 2
    return np.float32([
        (0, 0), (milieu_x, 0), (droite, 0),
        (0, milieu_y), (droite, milieu_y),
        (0, bas), (milieu_x, bas), (droite, bas),
    ])


def aires_signees(points, triangles):
    """Aire de chaque triangle par produit vectoriel ; le signe donne son orientation."""
    a, b, c = (points[triangles[:, i]] for i in range(3))
    ab, ac = b - a, c - a
    return (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2


def aires_triangles(points, triangles):
    """Calcule l'aire de chaque triangle, sans tenir compte de son orientation."""
    return np.abs(aires_signees(points, triangles))


def masque_retournes(points_source, points_cible, triangles, tolerance=0.0):
    """Repère les triangles dont l'orientation s'inverse : l'image s'y replie.

    Les replis d'une aire inférieure à `tolerance` (en pixels carrés) sont ignorés.
    """
    source = aires_signees(points_source, triangles)
    cible = aires_signees(points_cible, triangles)
    return (source * cible < 0) & (np.abs(cible) > tolerance)


def triangles_retournes(points_source, points_cible, triangles, tolerance=0.0):
    """Compte les triangles retournés."""
    return int(masque_retournes(points_source, points_cible, triangles, tolerance).sum())


def trianguler(points, exclus=()):
    """Triangulation de Delaunay, calculée une seule fois sur le portrait.

    Les mêmes triangles servent ensuite à chaque image : seule la position de leurs
    sommets change. Les triangles sans surface sont retirés, car une transformation
    affine ne peut pas être calculée à partir de 3 points alignés. Les points `exclus`
    ne deviennent pas des sommets, mais gardent leur position dans le tableau.
    """
    gardes = np.setdiff1d(np.arange(len(points)), exclus)
    triangles = gardes[Delaunay(points[gardes]).simplices]
    return triangles[aires_triangles(points, triangles) >= AIRE_MINIMALE]


def separer_interieurs(triangles):
    """Place d'abord les triangles entièrement dans un œil ou entre les lèvres.

    Ils se replient forcément quand l'œil se ferme. Dessinés en premier, ils sont
    recouverts par la peau autour, donc ces replis restent invisibles.
    """
    zones = [
        set(positions(GROUPES["oeil_droit"] + GROUPES["iris_droit"])),
        set(positions(GROUPES["oeil_gauche"] + GROUPES["iris_gauche"])),
        set(positions(LEVRE_HAUTE_INTERIEURE + LEVRE_BASSE_INTERIEURE)),
    ]
    interieur = np.array([any(set(t.tolist()) <= zone for zone in zones) for t in triangles])
    return np.vstack([triangles[interieur], triangles[~interieur]]), int(interieur.sum())


def contour_dans_les_yeux(points):
    """Trouve les points du contour qui tombent dans un œil.

    Sur un visage de trois quarts, le bord de la joue éloignée peut se projeter sur
    l'œil. En faire un sommet créerait des plis à chaque clignement.
    """
    trouves = []
    for basse, haute in PAUPIERES.values():
        polygone = points[positions(basse + haute[::-1][1:-1])]
        for position in positions(GROUPES["contour"]):
            x, y = points[position]
            if cv2.pointPolygonTest(polygone, (float(x), float(y)), False) >= 0:
                trouves.append(position)
    return trouves


def contour_levres_exterieur(points):
    """Polygone des lèvres extérieures, parcouru dans un seul sens."""
    return points[positions(LEVRE_BASSE_EXTERIEURE + LEVRE_HAUTE_EXTERIEURE[::-1][1:-1])]


def contour_levres_interieur(points):
    """Polygone de l'ouverture de la bouche, parcouru dans un seul sens."""
    return points[positions(LEVRE_BASSE_INTERIEURE + LEVRE_HAUTE_INTERIEURE[::-1][1:-1])]


def couleur_interieur_bouche(image, points):
    """Tire la luminosité de la bouche de celle des lèvres, avec une teinte rouge.

    Image numérique : on sépare luminosité et teinte. Des lèvres ocre donneraient du
    marron si on gardait leur teinte, alors on garde seulement leur luminosité (le plus
    grand canal), assombrie. La médiane résiste mieux qu'une moyenne aux reflets et aux
    pixels de peau pris par erreur dans le masque.
    """
    masque = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(masque, [np.int32(np.round(contour_levres_exterieur(points)))], 255)
    cv2.fillPoly(masque, [np.int32(np.round(contour_levres_interieur(points)))], 0)
    luminosite = np.median(image[masque > 0].max(axis=1)) * ASSOMBRISSEMENT_BOUCHE
    luminosite = np.clip(luminosite, LUMINOSITE_BOUCHE_MIN, LUMINOSITE_BOUCHE_MAX)
    return luminosite * np.array(TEINTE_BOUCHE)


def couleur_des_dents(image, points):
    """Accorde les dents à la clarté de la peau : un blanc pur trancherait sur un tableau sombre.

    Image numérique : la luminosité vient de la médiane des joues et du nez.
    """
    enveloppe = cv2.convexHull(np.int32(np.round(points[positions(GROUPES["joues"] + GROUPES["nez"])])))
    masque = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(masque, enveloppe, 255)
    peau = np.median(image[masque > 0].max(axis=1))
    return np.clip(peau, LUMINOSITE_DENTS_MIN, LUMINOSITE_DENTS_MAX) * np.array(TEINTE_DENTS)


def preparer_portrait(image, reperes):
    """Enchaîne recadrage, redimensionnement, points de bord et triangulation.

    Les repères reçus sont de taille (N, 3). Le z suit la même échelle que x et y.
    """
    image, positions_xy = recadrer_sur_visage(image, reperes[:, :2])
    echelle = HAUTEUR_PORTRAIT / image.shape[0]
    image, positions_xy = redimensionner(image, positions_xy)
    hauteur, largeur = image.shape[:2]
    points = np.vstack([positions_xy, points_de_bord(largeur, hauteur)])
    exclus = positions(TEMPES) + contour_dans_les_yeux(points)
    triangles, nb_interieurs = separer_interieurs(trianguler(points, exclus))
    return Portrait(
        image, points, triangles, nb_interieurs, reperes[:, 2] * echelle,
        couleur_interieur_bouche(image, points), couleur_des_dents(image, points),
    )


def lire_image(chemin):
    """Lit une image, ou renvoie None. Contrairement à cv2.imread sous Windows, cette
    méthode accepte les accents dans le chemin, fréquents dans un fichier choisi à la main.
    """
    try:
        octets = np.fromfile(str(chemin), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(octets, cv2.IMREAD_COLOR) if octets.size else None


def portrait_depuis_image(image):
    """Prépare une image déjà lue. Renvoie None si elle est vide ou sans visage."""
    if image is None:
        return None
    with DetecteurReperes() as detecteur:
        reperes = detecteur.detecter_3d(image)
    return None if reperes is None else preparer_portrait(image, reperes)


def charger_un_portrait(chemin):
    """Prépare une seule image. Renvoie None si elle est illisible ou sans visage."""
    return portrait_depuis_image(lire_image(chemin))


def charger_portraits(dossier=DOSSIER_PORTRAITS):
    """Prépare les portraits du dossier où un visage est détecté, triés par nom."""
    portraits = {}
    chemins = sorted(c for c in dossier.iterdir() if c.suffix.lower() in EXTENSIONS_IMAGES)
    with DetecteurReperes() as detecteur:
        for chemin in chemins:
            image = lire_image(chemin)
            reperes = None if image is None else detecteur.detecter_3d(image)
            if reperes is None:
                print(f"{chemin.name} : aucun visage détecté, portrait ignoré.")
                continue
            portraits[chemin.stem] = preparer_portrait(image, reperes)
    return portraits


def dessiner_triangles(image, points, triangles, couleur=(255, 200, 0)):
    """Dessine les arêtes des triangles (affichage de contrôle)."""
    contours = [np.int32(np.round(points[t])) for t in triangles]
    cv2.polylines(image, contours, True, couleur, 1, cv2.LINE_AA)


def boite_decoupee(triangle, largeur, hauteur, marge=0):
    """Renvoie (x0, y0, x1, y1), la boîte englobante limitée aux bords de l'image."""
    x, y, l, h = cv2.boundingRect(np.float32([triangle]))
    return (
        max(x - marge, 0), max(y - marge, 0),
        min(x + l + marge, largeur), min(y + h + marge, hauteur),
    )


def deformer_triangle(src, dst, t_src, t_dst):
    """Recopie un triangle de src vers dst par transformation affine.

    Transformation affine : 3 points source → 3 points cible, donc 6 inconnues et une
    solution exacte. Seule la boîte englobante est calculée, pour gagner du temps.
    """
    t_src, t_dst = np.float32(t_src), np.float32(t_dst)
    aires = aires_triangles(np.vstack([t_src, t_dst]), np.array([[0, 1, 2], [3, 4, 5]]))
    if aires.min() < AIRE_MINIMALE:  # 3 points alignés : l'affine n'existe pas
        return

    hauteur, largeur = src.shape[:2]
    xs0, ys0, xs1, ys1 = boite_decoupee(t_src, largeur, hauteur, MARGE_SOURCE)
    x0, y0, x1, y1 = boite_decoupee(t_dst, dst.shape[1], dst.shape[0])
    if x1 <= x0 or y1 <= y0:  # triangle entièrement hors de l'image
        return

    t1 = t_src - np.float32([xs0, ys0])
    t2 = t_dst - np.float32([x0, y0])
    affine = cv2.getAffineTransform(t1, t2)
    morceau = cv2.warpAffine(
        src[ys0:ys1, xs0:xs1], affine, (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    # Masque net en 8 bits : deux triangles voisins partagent leurs pixels de bord,
    # donc aucun joint n'apparaît. Sur un masque float32, OpenCV ignore LINE_AA.
    masque = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(masque, np.int32(np.round(t2)), 255)
    cv2.copyTo(morceau, masque, dst[y0:y1, x0:x1])


def animer(image, points_source, points_cible, triangles, fond=None):
    """Déforme l'image triangle par triangle, toujours à partir du portrait d'origine.

    Repartir de l'original évite d'accumuler le flou des interpolations successives.
    `fond` permet de vérifier que les triangles couvrent bien toute l'image.
    """
    resultat = image.copy() if fond is None else fond.copy()
    for triangle in triangles:
        deformer_triangle(image, resultat, points_source[triangle], points_cible[triangle])
    return resultat


def ouverture_bouche(points):
    """Écart moyen entre les lèvres intérieures, sans compter les commissures."""
    basse = positions(LEVRE_BASSE_INTERIEURE[1:-1])
    haute = positions(LEVRE_HAUTE_INTERIEURE[1:-1])
    return float(np.linalg.norm(points[haute] - points[basse], axis=1).mean())


def progression(valeur, debut, fin):
    """Passe de 0 à 1 entre deux seuils : ce qui apparaît ne surgit jamais d'un coup."""
    return float(np.clip((valeur - debut) / (fin - debut), 0.0, 1.0))


def superposer(peinture, couche, couleur, opacite):
    """Mélange une couleur sur la peinture, là où le masque 8 bits `couche` est rempli."""
    poids = couche[..., np.newaxis].astype(np.float32) / 255.0 * opacite
    peinture[:] = peinture * (1 - poids) + couleur * poids


def peindre_langue(peinture, haute, basse, couleur, opacite):
    """Ajoute la langue : une ellipse claire posée sur la lèvre du bas, adoucie par un flou."""
    milieu = len(basse) // 2
    largeur = float(np.linalg.norm(basse[-1] - basse[0]))
    ecart = float(np.linalg.norm(haute[milieu] - basse[milieu]))
    centre = (round(float(basse[milieu, 0])), round(float(basse[milieu, 1] - 0.06 * ecart)))
    axes = (max(round(0.30 * largeur), 1), max(round(0.30 * ecart), 1))
    couche = np.zeros(peinture.shape[:2], np.uint8)
    cv2.ellipse(couche, centre, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    # Filtre gaussien proportionnel à la langue : son contour reste très doux.
    taille_flou = max(round(0.35 * ecart), 2) * 2 + 1
    couche = cv2.GaussianBlur(couche, (taille_flou, taille_flou), 0)
    teinte = np.clip(couleur * ECLAIRCISSEMENT_LANGUE + AJOUT_ROSE_LANGUE, 0, 255)
    superposer(peinture, couche, teinte, opacite * OPACITE_LANGUE)


def peindre_dents(peinture, haute, basse, couleur, opacite):
    """Ajoute les dents du haut : une bande courte sous la lèvre, qui s'efface sur les côtés."""
    milieu = len(basse) // 2
    x_gauche, x_droite = float(haute[:, 0].min()), float(haute[:, 0].max())
    demi_largeur = max((x_droite - x_gauche) / 2 * LARGEUR_DENTS, 1.0)
    centre_x = (x_gauche + x_droite) / 2
    ecart = float(np.linalg.norm(haute[milieu] - basse[milieu]))
    hauteur = min(0.24 * ecart, 0.13 * demi_largeur * 2)

    relatif = np.clip((haute[:, 0] - centre_x) / demi_largeur, -1.0, 1.0)
    bombe = np.sqrt(np.maximum(1.0 - relatif**2, 0.0))
    bas_des_dents = haute + np.column_stack([np.zeros(len(haute)), hauteur * bombe])
    bande = np.vstack([haute, bas_des_dents[::-1]])
    couche = np.zeros(peinture.shape[:2], np.uint8)
    cv2.fillPoly(couche, [np.int32(np.round(bande))], 255, cv2.LINE_AA)
    # Les colonnes hors de la largeur des dents disparaissent progressivement.
    couche = cv2.GaussianBlur(couche, (5, 5), 0)

    # Ombre vers les commissures, et un très léger creux entre chaque dent.
    colonnes = np.arange(peinture.shape[1], dtype=np.float32)
    u = np.clip((colonnes - centre_x) / demi_largeur, -1.0, 1.0)
    ombre = 0.55 + 0.45 * np.sqrt(np.maximum(1.0 - u**2, 0.0))
    creux = 1.0 - 0.07 * np.abs(np.cos(np.pi * NB_DENTS * (u + 1.0) / 2.0)) ** 8
    teinte = (ombre * creux)[np.newaxis, :, np.newaxis] * couleur * ATTENUATION_DENTS
    superposer(peinture, couche, teinte, opacite * OPACITE_DENTS)


def peindre_bouche(masque, haute, basse, portrait, ouverture):
    """Peint l'intérieur : gorge sombre au centre, puis langue et dents si la bouche s'ouvre.

    Image numérique : la transformée de distance donne, en chaque pixel, son éloignement
    du bord. Plus on s'éloigne des lèvres, plus la couleur s'assombrit, comme en profondeur.
    """
    distance = cv2.distanceTransform(masque, cv2.DIST_L2, 3)
    profondeur = (distance / max(float(distance.max()), 1.0))[..., np.newaxis]
    clarte = CLARTE_BORD_BOUCHE + (CLARTE_GORGE - CLARTE_BORD_BOUCHE) * profondeur
    peinture = (portrait.couleur_bouche * clarte).astype(np.float32)
    langue = progression(ouverture, LANGUE_DEBUT, LANGUE_PLEINE)
    if langue > 0.0:
        peindre_langue(peinture, haute, basse, portrait.couleur_bouche, langue)
    dents = progression(ouverture, DENTS_DEBUT, DENTS_PLEINES)
    if dents > 0.0:
        peindre_dents(peinture, haute, basse, portrait.couleur_dents, dents)
    # Un léger filtre gaussien retire l'aspect de dessin net sur un tableau.
    return cv2.GaussianBlur(peinture, (3, 3), 0)


def ouverture_relative_bouche(points, reference):
    """Ouverture ajoutée par rapport à `reference`, en proportion de l'écart des yeux.

    Une proportion ne dépend ni de la taille du visage ni de sa distance à la caméra.
    """
    droit, gauche = positions([33, 263])
    ecart_yeux = max(float(np.linalg.norm(reference[droit] - reference[gauche])), 1e-6)
    return max(ouverture_bouche(points) - ouverture_bouche(reference), 0.0) / ecart_yeux


def dessiner_interieur_bouche(image, portrait, cible, ouverture=None):
    """Remplit ce qui n'existe pas dans le portrait : l'intérieur de la bouche ouverte.

    Le dessin vient après la déformation, car une bouche fermée n'a aucune surface à
    étirer. Le filtre gaussien sur le masque évite un bord trop net. `ouverture` est celle
    de la vraie bouche filmée : sur le portrait, le moindre mouvement de lèvre est amplifié
    par l'échelle et les gains, et ouvrirait un trou alors que la bouche reste fermée.
    """
    if ouverture is None:
        ouverture = ouverture_relative_bouche(cible, portrait.points)
    opacite = progression(ouverture, OUVERTURE_DEBUT, OUVERTURE_PLEINE)
    if opacite == 0.0:
        return
    polygone = contour_levres_interieur(cible)
    x0, y0, x1, y1 = boite_decoupee(polygone, image.shape[1], image.shape[0], TAILLE_FLOU_BOUCHE)
    if x1 <= x0 or y1 <= y0:
        return
    origine = np.float32([x0, y0])
    masque = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(masque, [np.int32(np.round(polygone - origine))], 255, cv2.LINE_AA)
    peinture = peindre_bouche(
        masque,
        cible[positions(LEVRE_HAUTE_INTERIEURE)] - origine,
        cible[positions(LEVRE_BASSE_INTERIEURE)] - origine,
        portrait, ouverture,
    )
    flou = (TAILLE_FLOU_BOUCHE, TAILLE_FLOU_BOUCHE)
    poids = cv2.GaussianBlur(masque, flou, 0)[..., np.newaxis] / 255.0 * opacite
    zone = image[y0:y1, x0:x1]
    zone[:] = zone * (1 - poids) + peinture * poids


def animer_portrait(portrait, cible, ouverture=None):
    """Produit l'image finale : déformation par triangles, puis intérieur de la bouche."""
    resultat = animer(portrait.image, portrait.points, cible, portrait.triangles)
    dessiner_interieur_bouche(resultat, portrait, cible, ouverture)
    return resultat
