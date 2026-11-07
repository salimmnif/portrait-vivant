"""Mesures pour le rapport : résultats en CSV et graphiques dans le dossier resultats/.

Utilisation, depuis le dossier portrait_vivant :
    python mesures.py portraits     # détection sur chaque portrait, sans webcam
    python mesures.py lissage       # 10 s de tête immobile
    python mesures.py recalage      # 40 s de mouvements de tête, sans grimacer
    python mesures.py temps         # 300 images avec le programme complet
    python mesures.py tout          # tout enchaîner
    python mesures.py graphiques    # retracer les graphiques depuis les CSV

Suivez les consignes affichées dans la fenêtre. Q interrompt la mesure en cours.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # écrit les graphiques dans des fichiers, sans ouvrir de fenêtre
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from deformation import DOSSIER_PORTRAITS, EXTENSIONS_IMAGES, animer_portrait, charger_portraits  # noqa: E402
from main import (  # noqa: E402
    COEFFICIENT_LISSAGE, INDICE_CAMERA, Etat, composer_affichage, ecrire, lisser_la_pose,
)
from recalage import calculer_grimace, calculer_grimace_3d, norme_moyenne  # noqa: E402
from reperes import NOMBRE_REPERES, DetecteurReperes, LissageExponentiel  # noqa: E402
from transfert import ouverture_bouche_mesuree, transferer  # noqa: E402

DOSSIER_RESULTATS = Path(__file__).resolve().parent / "resultats"
NOM_FENETRE = "Mesures"
NB_IMAGES_TEMPS = 300
DUREE_PREPARATION_S = 5.0
DUREE_LISSAGE_S = 10.0
DUREE_PHASE_S = 6.0
DUREE_PAUSE_S = 2.0  # temps laissé pour lire la consigne du mouvement suivant
SEUIL_TETE_DE_FACE = 5.0  # en degrés : en dessous, la tête n'a pas vraiment tourné
NB_IMAGES_NEUTRE = 15  # le visage neutre est la médiane des dernières images
COEFFICIENTS_LISSAGE = (1.0, 0.75, COEFFICIENT_LISSAGE, 0.25)  # 1,0 = sans lissage
PHASES_RECALAGE = [
    ("Translation", "Deplacez la tete dans le cadre"),
    ("Distance", "Approchez et reculez la tete"),
    ("Tête penchée", "Penchez la tete vers chaque epaule"),
    ("Gauche-droite", "Tournez la tete a gauche et a droite"),
    ("Haut-bas", "Levez et baissez la tete"),
]
MOUVEMENTS_MESURES = ("De face (moins de 5°)", "Penchée", "Tournée", "Levée ou baissée")
METHODES_RECALAGE = [("sans", "Sans recalage"), ("recalage_2d", "Recalage 2D"), ("recalage_3d", "Recalage 3D")]

# Palette de référence du guide de visualisation, vérifiée avec son validateur.
COULEURS_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
FOND, TEXTE, TEXTE_SECONDAIRE, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


class Interruption(Exception):
    """La personne a demandé d'arrêter la mesure en cours."""


# ---------- Analyse, sans webcam ----------

def tremblement(serie):
    """Écart-type du déplacement image à image, moyenné sur les repères, en pixels."""
    return float(np.std(np.diff(serie[:, :, :2], axis=0), axis=0).mean())


def lisser_serie(serie, coefficient):
    """Applique à toute une série le même filtre exponentiel que le programme."""
    lissage = LissageExponentiel(coefficient)
    return np.array([lissage.appliquer(reperes) for reperes in serie])


def comparer_lissages(serie, coefficients=COEFFICIENTS_LISSAGE):
    """Mesure le tremblement d'une même série pour chaque coefficient de lissage."""
    return [
        {"coefficient": coefficient, "tremblement_px": tremblement(lisser_serie(serie, coefficient))}
        for coefficient in coefficients
    ]


def comparer_recalages(serie, neutre):
    """Calcule la norme moyenne de Δ, image par image, pour chaque méthode de recalage."""
    lignes = []
    neutre_2d = neutre[:, :2]
    for reperes in serie:
        grimace_2d = calculer_grimace(reperes[:, :2], neutre_2d)
        grimace_3d = calculer_grimace_3d(reperes, neutre)
        lignes.append({
            "sans": norme_moyenne(reperes[:, :2] - neutre_2d),
            "recalage_2d": np.nan if grimace_2d is None else norme_moyenne(grimace_2d.delta),
            "recalage_3d": norme_moyenne(grimace_3d.delta),
            "lacet_deg": grimace_3d.lacet,
            "tangage_deg": grimace_3d.tangage,
            "roulis_deg": grimace_3d.angle_tete,
        })
    return lignes


def classer_mouvement(ligne):
    """Classe une image selon sa rotation dominante, mesurée et non demandée.

    Les consignes ne suffisent pas : une personne peut être en retard sur elles ou
    combiner plusieurs mouvements. Les angles mesurés disent ce qui s'est vraiment passé.
    """
    roulis, lacet, tangage = (
        abs(float(ligne[cle])) for cle in ("roulis_deg", "lacet_deg", "tangage_deg")
    )
    plus_grand = max(roulis, lacet, tangage)
    if plus_grand < SEUIL_TETE_DE_FACE:
        return MOUVEMENTS_MESURES[0]
    if plus_grand == roulis:
        return MOUVEMENTS_MESURES[1]
    return MOUVEMENTS_MESURES[2] if plus_grand == lacet else MOUVEMENTS_MESURES[3]


def resumer_recalages(lignes):
    """Regroupe les images par mouvement réel : la médiane de chaque méthode, par mouvement.

    La médiane résiste aux quelques images où une méthode décroche complètement.
    """
    resume = []
    for mouvement in MOUVEMENTS_MESURES:
        du_mouvement = [ligne for ligne in lignes if classer_mouvement(ligne) == mouvement]
        if du_mouvement:
            resultat = {"mouvement": mouvement, "nb_images": len(du_mouvement)}
            for methode, _ in METHODES_RECALAGE:
                resultat[methode] = float(np.nanmedian([float(ligne[methode]) for ligne in du_mouvement]))
            resume.append(resultat)
    return resume


def resumer_temps(durees, nb_images):
    """Résume les durées de chaque étape, en millisecondes."""
    return [
        {
            "etape": etape,
            "moyenne_ms": float(np.mean(valeurs)) * 1000,
            "ecart_type_ms": float(np.std(valeurs)) * 1000,
            "nb_images": len(valeurs),
        }
        for etape, valeurs in durees.items() if len(valeurs) > 0
    ] if nb_images else []


def mesurer_detection(dossier=DOSSIER_PORTRAITS):
    """Teste la détection automatique sur chaque image du dossier."""
    lignes = []
    chemins = sorted(c for c in Path(dossier).iterdir() if c.suffix.lower() in EXTENSIONS_IMAGES)
    with DetecteurReperes() as detecteur:
        for chemin in chemins:
            image = cv2.imread(str(chemin))
            debut = time.perf_counter()
            reperes = None if image is None else detecteur.detecter(image)
            lignes.append({
                "portrait": chemin.name,
                "visage_detecte": "oui" if reperes is not None else "non",
                "duree_ms": (time.perf_counter() - debut) * 1000,
            })
    return lignes


# ---------- Fichiers CSV ----------

def ecrire_csv(chemin, lignes):
    """Écrit une liste de dictionnaires dans un fichier CSV."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        ecrivain.writeheader()
        ecrivain.writerows(lignes)


def lire_csv(chemin):
    with open(chemin, newline="", encoding="utf-8") as fichier:
        return list(csv.DictReader(fichier))


# ---------- Graphiques ----------

def nombre(valeur, decimales=1):
    """Écrit un nombre avec une virgule décimale."""
    return f"{valeur:.{decimales}f}".replace(".", ",")


def preparer_graphique(titre, sous_titre, axe_valeurs, taille=(8, 4.2)):
    """Crée une figure sobre : fond clair, grille discrète, un seul axe de valeurs."""
    figure, axe = plt.subplots(figsize=taille, dpi=150, facecolor=FOND)
    axe.set_facecolor(FOND)
    for cote in ("top", "right", "left"):
        axe.spines[cote].set_visible(False)
    axe.spines["bottom"].set_color(GRILLE)
    axe.tick_params(colors=TEXTE_SECONDAIRE, length=0, labelsize=9)
    axe.grid(axis=axe_valeurs, color=GRILLE, linewidth=0.8)
    axe.set_axisbelow(True)
    figure.text(0.02, 0.95, titre, fontsize=13, fontweight="bold", color=TEXTE, va="top")
    figure.text(0.02, 0.885, sous_titre, fontsize=9.5, color=TEXTE_SECONDAIRE, va="top")
    return figure, axe


def enregistrer_graphique(figure, chemin, haut=0.8):
    figure.tight_layout(rect=(0.01, 0.01, 0.99, haut))
    figure.savefig(chemin, facecolor=FOND)
    plt.close(figure)


def barres_horizontales(axe, noms, valeurs, unite):
    """Trace une série de barres fines, avec la valeur au bout de chaque barre."""
    positions_y = np.arange(len(noms))
    axe.barh(positions_y, valeurs, height=0.42, color=COULEURS_SERIES[0])
    axe.set_yticks(positions_y, noms)
    axe.invert_yaxis()
    axe.tick_params(axis="y", labelcolor=TEXTE, labelsize=10)
    axe.set_xlim(0, max(valeurs) * 1.18)
    axe.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    for position, valeur in zip(positions_y, valeurs):
        axe.text(
            valeur + max(valeurs) * 0.012, position, f"{nombre(valeur, 2 if valeur < 1 else 1)} {unite}",
            va="center", fontsize=9.5, color=TEXTE,
        )
    axe.set_xlabel(unite, color=TEXTE_SECONDAIRE, fontsize=9)


def tracer_temps(dossier):
    """Graphique : durée moyenne de chaque étape, avec le rythme obtenu."""
    lignes = lire_csv(dossier / "temps.csv")
    boucle = next(ligne for ligne in lignes if ligne["etape"] == "Boucle complète")
    etapes = [ligne for ligne in lignes if ligne["etape"] != "Boucle complète"]
    images_par_seconde = 1000 / float(boucle["moyenne_ms"])
    figure, axe = preparer_graphique(
        "Temps de calcul par étape",
        f"{nombre(images_par_seconde, 0)} images/s en moyenne sur {boucle['nb_images']} images, "
        f"soit {nombre(float(boucle['moyenne_ms']))} ms par image",
        "x",
    )
    barres_horizontales(
        axe, [ligne["etape"] for ligne in etapes],
        [float(ligne["moyenne_ms"]) for ligne in etapes], "ms",
    )
    enregistrer_graphique(figure, dossier / "temps.png")


def tracer_lissage(dossier):
    """Graphique : tremblement des repères selon le coefficient de lissage."""
    lignes = lire_csv(dossier / "lissage.csv")
    noms = []
    for ligne in lignes:
        coefficient = float(ligne["coefficient"])
        if coefficient == 1.0:
            noms.append("Sans lissage")
        elif coefficient == COEFFICIENT_LISSAGE:
            noms.append(f"Coefficient {nombre(coefficient, 2)} (programme)")
        else:
            noms.append(f"Coefficient {nombre(coefficient, 2)}")
    figure, axe = preparer_graphique(
        "Tremblement des repères, tête immobile",
        "Écart-type du déplacement image à image, moyenné sur les repères",
        "x",
    )
    barres_horizontales(axe, noms, [float(ligne["tremblement_px"]) for ligne in lignes], "px")
    enregistrer_graphique(figure, dossier / "lissage.png")


def tracer_recalage(dossier):
    """Graphique : norme de Δ selon le mouvement réel de la tête et la méthode."""
    lignes = resumer_recalages(lire_csv(dossier / "recalage_detail.csv"))
    ecrire_csv(dossier / "recalage_resume.csv", lignes)
    figure, axe = preparer_graphique(
        "Norme de Δ quand la tête bouge sans grimacer",
        "Médiane par image, selon la rotation dominante mesurée. "
        "Plus c'est bas, mieux le mouvement de tête est retiré.",
        "y", taille=(9, 4.8),
    )
    centres = np.arange(len(lignes))
    largeur = 0.24
    for rang, (methode, nom) in enumerate(METHODES_RECALAGE):
        valeurs = [float(ligne[methode]) for ligne in lignes]
        abscisses = centres + (rang - 1) * largeur
        axe.bar(
            abscisses, valeurs, width=largeur - 0.02, color=COULEURS_SERIES[rang],
            label=nom, edgecolor=FOND, linewidth=0.5,
        )
        if methode != "sans":  # les valeurs sans recalage se lisent sur l'axe
            for abscisse, valeur in zip(abscisses, valeurs):
                axe.text(
                    abscisse, valeur, nombre(valeur), ha="center", va="bottom",
                    fontsize=8.5, color=TEXTE,
                )
    axe.set_xticks(
        centres, [f"{ligne['mouvement']}\n{ligne['nb_images']} images" for ligne in lignes]
    )
    axe.tick_params(axis="x", labelcolor=TEXTE, labelsize=10)
    axe.set_ylabel("pixels", color=TEXTE_SECONDAIRE, fontsize=9)
    axe.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0.0, 1.1), fontsize=9.5)
    enregistrer_graphique(figure, dossier / "recalage.png", haut=0.82)


TRACES = {"temps.csv": tracer_temps, "lissage.csv": tracer_lissage, "recalage_detail.csv": tracer_recalage}


def retracer_graphiques(dossier=DOSSIER_RESULTATS):
    """Retrace les graphiques de tous les CSV disponibles."""
    for fichier, tracer in TRACES.items():
        if (Path(dossier) / fichier).exists():
            tracer(Path(dossier))
            print(f"Graphique tracé depuis {fichier}.")


# ---------- Acquisition avec la webcam ----------

def lire_image(camera):
    """Lit une image de la webcam, avec le même effet miroir que le programme."""
    lecture_ok, image = camera.read()
    if not lecture_ok:
        raise Interruption("Lecture de la webcam impossible.")
    return cv2.flip(image, 1)


def afficher_consignes(image, consignes):
    """Affiche l'image et ses consignes. La touche Q interrompt la mesure."""
    for ligne, texte in enumerate(consignes):
        ecrire(image, texte, ligne, (0, 255, 255))
    cv2.imshow(NOM_FENETRE, image)
    if chr(cv2.waitKey(1) & 0xFF).lower() == "q":
        raise Interruption("Mesure interrompue.")
    if cv2.getWindowProperty(NOM_FENETRE, cv2.WND_PROP_VISIBLE) < 1:
        raise Interruption("Fenêtre fermée.")


def enregistrer_serie(camera, detecteur, duree_s, consigne):
    """Enregistre les repères pendant une durée. Renvoie (série, nb d'images sans visage)."""
    reperes_vus = []
    sans_visage = 0
    fin = time.perf_counter() + duree_s
    while (restant := fin - time.perf_counter()) > 0:
        image = lire_image(camera)
        reperes = detecteur.detecter_3d(image)
        if reperes is None:
            sans_visage += 1
        else:
            reperes_vus.append(reperes)
        afficher_consignes(image, [consigne, f"{restant:.0f} s  |  Q : interrompre"])
    vide = np.empty((0, NOMBRE_REPERES, 3), np.float32)
    return (np.array(reperes_vus) if reperes_vus else vide), sans_visage


def capturer_neutre(camera, detecteur):
    """Enregistre le visage neutre : la médiane réduit le bruit d'une seule image."""
    while True:
        serie, _ = enregistrer_serie(
            camera, detecteur, DUREE_PREPARATION_S, "Regardez la camera, visage detendu"
        )
        if len(serie) > 0:
            return np.median(serie[-NB_IMAGES_NEUTRE:], axis=0).astype(np.float32)


def mesure_lissage(camera, detecteur, dossier):
    """Mesure le tremblement des repères quand la tête ne bouge pas."""
    capturer_neutre(camera, detecteur)
    serie, sans_visage = enregistrer_serie(camera, detecteur, DUREE_LISSAGE_S, "Ne bougez plus")
    if len(serie) < 2:
        raise Interruption("Aucun visage détecté pendant la mesure du lissage.")
    lignes = comparer_lissages(serie)
    for ligne in lignes:
        ligne.update(nb_images=len(serie), images_sans_visage=sans_visage)
    ecrire_csv(dossier / "lissage.csv", lignes)
    tracer_lissage(dossier)


def mesure_recalage(camera, detecteur, dossier):
    """Mesure Δ sans grimace, pour chaque type de mouvement et chaque méthode."""
    neutre = capturer_neutre(camera, detecteur)
    lignes = []
    for phase, consigne in PHASES_RECALAGE:
        enregistrer_serie(camera, detecteur, DUREE_PAUSE_S, "Ensuite : " + consigne)
        serie, _ = enregistrer_serie(camera, detecteur, DUREE_PHASE_S, consigne + ", sans grimacer")
        for numero, ligne in enumerate(comparer_recalages(serie, neutre)):
            lignes.append({"phase": phase, "image": numero, **ligne})
    if not lignes:
        raise Interruption("Aucun visage détecté pendant la mesure du recalage.")
    ecrire_csv(dossier / "recalage_detail.csv", lignes)
    tracer_recalage(dossier)


def mesure_temps(camera, detecteur, portraits, dossier, nb_images=NB_IMAGES_TEMPS):
    """Chronomètre chaque étape du programme complet, pendant des grimaces normales."""
    etat = Etat(portraits, afficher_debug=False)
    etat.neutre = capturer_neutre(camera, detecteur)
    lissage = LissageExponentiel(COEFFICIENT_LISSAGE)
    noms = ("Lecture webcam", "Détection", "Lissage et recalage", "Transfert", "Déformation", "Affichage")
    durees = {nom: [] for nom in noms}
    durees["Boucle complète"] = []
    for numero in range(nb_images):
        debut = time.perf_counter()
        image = lire_image(camera)
        apres_lecture = time.perf_counter()
        mesure = detecteur.detecter_3d(image)
        apres_detection = time.perf_counter()
        durees["Lecture webcam"].append(apres_lecture - debut)
        durees["Détection"].append(apres_detection - apres_lecture)
        reperes = lissage.appliquer(mesure)
        if reperes is not None:
            grimace = lisser_la_pose(etat, calculer_grimace_3d(reperes, etat.neutre))
            apres_recalage = time.perf_counter()
            etat.cible = transferer(etat.portrait, grimace, etat.neutre, etat.intensite)
            apres_transfert = time.perf_counter()
            ouverture = ouverture_bouche_mesuree(grimace, etat.neutre, etat.intensite)
            etat.rendu = animer_portrait(etat.portrait, etat.cible, ouverture)
            apres_deformation = time.perf_counter()
            durees["Lissage et recalage"].append(apres_recalage - apres_detection)
            durees["Transfert"].append(apres_transfert - apres_recalage)
            durees["Déformation"].append(apres_deformation - apres_transfert)
        avant_affichage = time.perf_counter()
        affichage = composer_affichage(image, etat)
        afficher_consignes(affichage, [f"Faites des grimaces : image {numero + 1}/{nb_images}"])
        fin = time.perf_counter()
        durees["Affichage"].append(fin - avant_affichage)
        durees["Boucle complète"].append(fin - debut)
    ecrire_csv(dossier / "temps.csv", resumer_temps(durees, nb_images))
    tracer_temps(dossier)


def lancer_mesures(actions, dossier=DOSSIER_RESULTATS):
    """Ouvre la webcam et enchaîne les mesures demandées."""
    camera = cv2.VideoCapture(INDICE_CAMERA)
    if not camera.isOpened():
        sys.exit("Webcam introuvable : vérifiez qu'aucune autre application ne l'utilise.")
    try:
        with DetecteurReperes(mode_video=True) as detecteur:
            if "lissage" in actions:
                mesure_lissage(camera, detecteur, dossier)
            if "recalage" in actions:
                mesure_recalage(camera, detecteur, dossier)
            if "temps" in actions:
                mesure_temps(camera, detecteur, list(charger_portraits().values()), dossier)
    except Interruption as raison:
        print(f"Arrêt : {raison}", file=sys.stderr)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument(
        "action", nargs="?", default="tout",
        choices=["portraits", "lissage", "recalage", "temps", "tout", "graphiques"],
    )
    action = analyseur.parse_args().action
    if action in ("portraits", "tout"):
        lignes = mesurer_detection()
        ecrire_csv(DOSSIER_RESULTATS / "detection.csv", lignes)
        for ligne in lignes:
            print(f"{ligne['portrait']} : visage détecté : {ligne['visage_detecte']}")
    if action == "graphiques":
        retracer_graphiques()
    elif action != "portraits":
        lancer_mesures(("lissage", "recalage", "temps") if action == "tout" else (action,))
    print(f"Résultats dans {DOSSIER_RESULTATS}")


if __name__ == "__main__":
    main()
