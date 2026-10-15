# Maquette → arbre hiérarchique

Transforme une image de maquette d'interface en arbre JSON (boîtes, types, texte, styles, Row/Column,
composants répétés). Tout tourne en local, sans service payant ni modèle de langage. Le champ `label`
de chaque nœud est laissé à `null` : il sera rempli plus tard par un MLLM, qui ne doit pas toucher
aux positions.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Utilisation

| Besoin | Commande |
|---|---|
| Interface (déposer une image, curseurs, arbre, vérité terrain) | `.venv/bin/python app.py` puis http://127.0.0.1:7860 |
| Traiter un dossier d'images → JSON + images annotées | `.venv/bin/python batch.py images/ --out out/json` |
| Régénérer les maquettes de test et leur vérité terrain | `.venv/bin/python make_test_image.py` |
| Mesurer contre une vérité terrain | `.venv/bin/python -m uitree.evaluation image.png image_gt.json` |
| Tests | `.venv/bin/python -m pytest` |

Les paramètres sont dans `config.toml` (un nom de paramètre inconnu y est refusé).

## Chaîne de traitement (`uitree/`)

| Étape | Module |
|---|---|
| OCR (RapidOCR, local) | `ocr.py` |
| Détection des boîtes (surfaces, cartes, encre, lignes de texte) | `detection.py` |
| Inclusion (qui contient qui) | `hierarchy.py` |
| Types grossiers (règles, interface remplaçable par un modèle) | `classify.py` |
| Styles (couleurs, hauteur du texte) | `style.py` |
| Arbre : inclusion puis XY-Cut | `tree.py`, `xycut.py` |
| Empreintes : composants répétés, doublons visuels | `similarity.py` |
| Export JSON + validation (schéma `tree.schema.json` + cohérence) | `export.py` |
| Point d'entrée commun (interface, batch, évaluation) | `analysis.py` |

Licences des dépendances : voir `LICENCES.md`.
# detecteur
