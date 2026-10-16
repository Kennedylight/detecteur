# Licences des dépendances

Relevées dans les métadonnées des paquets installés (`importlib.metadata`), pas devinées.
Toutes permissives. Mise à jour à chaque ajout de dépendance.

| Paquet | Version | Licence | Rôle |
|---|---|---|---|
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | détection des formes |
| numpy | 2.5.4 | BSD-3-Clause (+ 0BSD, MIT, Zlib pour des parties vendorisées) | calcul sur les pixels |
| pillow | 12.3.0 | MIT-CMU | lecture/dessin d'images |
| gradio | 6.30.0 | Apache-2.0 | interface locale (app.py) |
| rapidocr | 3.10.0 | Apache-2.0 | OCR local (modèles PP-OCRv6 fournis dans le paquet, aucun appel réseau) |
| onnxruntime | 1.31.0 | MIT | exécution des modèles OCR |
| jsonschema | 4.26.0 | MIT | validation du JSON de l'arbre (+ referencing, rpds-py, jsonschema-specifications, attrs : MIT) |
| pytest | 9.1.1 | MIT | tests (dév.) |

Le dHash (empreinte visuelle) est codé dans uitree/similarity.py (quelques lignes OpenCV) plutôt
que pris dans la bibliothèque imagehash : une variante adaptée aux aplats d'interface était nécessaire.

Les ~45 dépendances indirectes de gradio (voir requirements.txt) sont toutes sous licence MIT, BSD,
Apache, PSF, ISC ou MPL-2.0 (vérifié automatiquement). MPL-2.0 (ex. certifi) est un copyleft faible
au niveau du fichier : sans contrainte tant qu'on ne modifie pas ces paquets.
| iniconfig, pluggy, packaging, Pygments | — | MIT / Apache-2.0 OR BSD-2-Clause / BSD-2-Clause | dépendances de pytest |
