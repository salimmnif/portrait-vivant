# Portrait vivant

Un portrait refait vos grimaces en direct, à partir de la webcam et d'une seule image.

## Installation

Testé avec Python 3.14.7 sous Windows 11, sans GPU. Depuis le dossier `portrait_vivant` :

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

N'installez pas `opencv-python` : MediaPipe utilise déjà `opencv-contrib-python`, qui fournit `cv2`.

### Télécharger le modèle

L'API Tasks de MediaPipe a besoin du fichier `face_landmarker.task`, qui n'est pas inclus dans le paquet pip :

```powershell
New-Item -ItemType Directory -Force modeles
curl.exe -L -o modeles\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Vérifier l'installation

```powershell
python tests/verifier_installation.py --webcam
```

La dernière ligne doit être `Installation correcte.`

## Utilisation

### Avec l'interface (recommandé)

```powershell
python interface.py
```

Une page s'ouvre dans votre navigateur. Déposez un portrait ou choisissez-en un dans la galerie, activez la caméra, puis appuyez sur **Visage neutre**. Le programme reste sur votre ordinateur : aucune image n'est envoyée sur Internet. `Ctrl+C` dans le terminal pour quitter.

Une image déposée reste disponible jusqu'à la fermeture du programme. Pour la garder, copiez-la dans `portraits/`.

### Dans une fenêtre OpenCV

```powershell
python main.py
```

Regardez la caméra, visage détendu, puis appuyez sur **N**.

| Touche | Action |
| --- | --- |
| N | enregistrer le visage neutre |
| O | choisir une image sur l'ordinateur |
| P | portrait suivant |
| = ou +, - | augmenter ou diminuer l'intensité |
| D | afficher ou masquer le debug |
| Q | quitter |

La caméra se choisit avec `INDICE_CAMERA` dans `main.py`.

Une image choisie avec O reste disponible jusqu'à la fermeture du programme. Pour la garder, copiez-la dans `portraits/`. Utilisez une œuvre du domaine public, ou la photo d'une personne qui a donné son accord.

## Mesures

```powershell
python mesures.py tout
```

Suivez les consignes affichées (environ 1 min 30). Les CSV et les graphiques sont écrits dans `resultats/`. Chaque mesure peut aussi se lancer seule : `portraits`, `lissage`, `recalage`, `temps`. `python mesures.py graphiques` retrace les graphiques depuis les CSV.

## Portraits

| Fichier | Œuvre | Statut |
| --- | --- | --- |
| `portraits/joconde.jpg` | Léonard de Vinci, *La Joconde* | Domaine public |
| `portraits/vermeer.jpg` | Johannes Vermeer, *La Jeune Fille à la perle* | Domaine public |
| `portraits/van_gogh.jpg` | Vincent van Gogh, *Autoportrait* | Domaine public |
| `portraits/durer.jpg` | Albrecht Dürer, *Autoportrait* | Domaine public |
| `portraits/holbein.jpg` | Hans Holbein le Jeune, *Henri VIII* | Domaine public |
| `portraits/rembrandt.jpg` | Rembrandt, *Autoportrait* | Domaine public |
| `portraits/washington.jpg` | Gilbert Stuart, *George Washington* | Domaine public |
| `portraits/arcimboldo.jpg` | Giuseppe Arcimboldo, *Vertumne* (visage non détecté) | Domaine public |
| `portraits/klimt.jpg` | Gustav Klimt, *Adele Bloch-Bauer I* (visage non détecté) | Domaine public |

## Tests

```powershell
python -m pytest tests
python tests/demo_deformation.py      # images avant/après dans tests/sorties/
python tests/voir_triangulation.py    # repères et triangles de chaque portrait
```
