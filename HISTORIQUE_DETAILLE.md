# Historique détaillé du projet « Maquette → arbre hiérarchique »

Ce document décrit tout ce qui a été fait, dans l'ordre, avec les décisions, les mesures, les bugs
rencontrés et ce qui ne marche pas encore. Les chiffres viennent de commandes réellement lancées ;
quand une observation est faite « à l'œil » (sans vérité terrain), c'est indiqué.

---------------------------------------------------------------------------------------------------

## 1. Contexte et objectif

Projet de recherche (mémoire) : transformer une **maquette d'interface (image)** en un **arbre
hiérarchique structuré (JSON)**, qui servira plus tard à générer du code front-end propre (principes
SOLID, DRY). Ce dépôt ne couvre que l'étape **image → JSON d'arbre** et un **outil visuel** pour
l'inspecter.

Contraintes posées dès le départ :

- Un MLLM interviendra **plus tard** pour étiqueter les nœuds ; il ne doit **jamais toucher aux
  positions**. Aucun MLLM et aucune API payante dans ce projet.
- Python 3.10+, bibliothèques maintenues à **licence permissive**, API vérifiée avant usage.
- Code propre : modules séparés, fonctions courtes, types annotés, **paramètres dans un fichier de
  configuration**, tests unitaires pytest, environnement figé, résultats déterministes.
- Méthode : travailler **par étapes** — (a) environnement + tests de départ, (b) détecteur amélioré,
  (c) OCR, (d) types et styles, (e) arbre + JSON, (f) interface — en montrant le résultat de chacune
  et en **mesurant** au lieu de prétendre qu'une détection est exacte.

Critères de réussite demandés :

1. Sur `make_test_image.py` : toutes les boîtes correctes, y compris le badge séparé du texte.
2. Sur de vraies captures : une liste **honnête** des erreurs observées.
3. Une commande pour traiter un dossier d'images et écrire les JSON.

---------------------------------------------------------------------------------------------------

## 2. Point de départ (fichiers fournis)

Quatre fichiers, d'abord absents du dossier (le dossier `Detecteur/` était vide ; je l'ai signalé et
l'utilisateur les a fournis) :

| Fichier d'origine | Rôle | Limites connues |
|---|---|---|
| `xy_cut.py` | découpage récursif en `Row` / `Column` / `Feuille` / `Groupe` (XY-Cut) | ids non reproductibles (compteur jamais remis à zéro) ; une boîte contenue dans une autre devient sa sœur dans un « Groupe » |
| `detect_boxes.py` | détecteur OpenCV simple : fond dominant, masque, dilatation, contours | un fond de couleur = une seule grosse boîte ; éléments proches fusionnés ; texte non lu ; type deviné grossièrement |
| `pipeline.py` | chaîne détecteur → XY-Cut → JSON | — |
| `make_test_image.py` | fabrique la maquette « Contacts » (400×270) | pas de vérité terrain |

**Évolution de ces fichiers.** La consigne était « améliore-les, ne les réécris pas à l'aveugle ».
Je les ai d'abord lus et exécutés tels quels pour mesurer leur comportement, puis modifiés par
étapes mesurées. Au final, `detect_boxes.py` est devenu `uitree/detection.py` et `xy_cut.py` est
devenu `uitree/xycut.py` : leur contenu a été **fortement remanié** (voir §6 et §10). `pipeline.py`
a été **remplacé** par `batch.py`. Les fichiers d'origine n'étaient pas suivis par git, donc leur
historique n'est pas conservé.

---------------------------------------------------------------------------------------------------

## 3. Environnement et choix techniques

### 3.1 Python

- Python système : 3.14.4. J'ai utilisé un **venv Python 3.12** (`.venv`).
- Raison : RapidOCR et onnxruntime n'ont peut-être pas encore de paquets pour 3.14. **Je n'ai pas pu
  le vérifier** (pas de `curl`, et la commande `pip download` de sondage a été refusée par
  l'utilisateur). 3.12 est le choix sûr, pas une conclusion démontrée.

### 3.2 Dépendances (versions figées dans `requirements.txt`)

| Paquet | Version | Licence (relevée dans les métadonnées installées) | Usage |
|---|---|---|---|
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | formes, contours, morphologie |
| numpy | 2.5.4 | BSD-3-Clause (+ 0BSD, MIT, Zlib) | calcul sur les pixels |
| pillow | 12.3.0 | MIT-CMU | images de test, dessin des schémas |
| rapidocr | 3.10.0 | Apache-2.0 | OCR local |
| onnxruntime | 1.31.0 | MIT | exécution des modèles OCR |
| gradio | 6.30.0 | Apache-2.0 | interface locale |
| jsonschema | 4.26.0 | MIT | validation du JSON (+ referencing, rpds-py, jsonschema-specifications, attrs : MIT) |
| pytest | 9.1.1 | MIT | tests |

Un contrôle automatique des ~70 paquets installés (dépendances indirectes de gradio incluses) n'a
trouvé que des licences MIT, BSD, Apache, PSF, ISC ou MPL-2.0. MPL-2.0 (ex. `certifi`) est un
copyleft faible au niveau du fichier : sans contrainte tant qu'on ne modifie pas ces paquets.
Détail : `LICENCES.md`.

### 3.3 Pourquoi RapidOCR

- Gratuit, **local** (aucun réseau, aucune clé), licence Apache-2.0.
- Les modèles PP-OCR (détection, reconnaissance, classification d'angle) sont **fournis dans le
  paquet** et exécutés par onnxruntime : rien à télécharger à l'exécution.
- API vérifiée avant usage : `RapidOCR()(image)` renvoie un objet `RapidOCROutput` avec `boxes`
  (quadrilatères), `txts`, `scores`. Un tableau numpy à 3 canaux est lu comme du **BVR**
  (vérifié dans `load_image.py` du paquet), donc on passe directement l'image OpenCV.
- Environ 1 s par image.

### 3.4 Pourquoi pas la bibliothèque `imagehash`

Le hash perceptuel (dHash) est codé en quelques lignes (`uitree/similarity.py`) : une variante
adaptée aux aplats d'interface était nécessaire (voir §9.2).

---------------------------------------------------------------------------------------------------

## 4. Étape (a) — Environnement et tests de départ

### 4.1 Mesure de référence du code d'origine

`make_test_image.py` d'origine dessinait la maquette mais **ne produisait pas de vérité terrain**.
Mesurer sans vérité terrain aurait été de l'impression. J'ai donc :

1. **Vérifié que l'image régénérée est identique au pixel près** à l'originale (empreinte md5 des
   pixels `f985df0692d3c1838e906013f4372531`, fixée dans un test).
2. Ajouté un `_Recorder` : chaque élément dessiné est enregistré avec sa **boîte réelle, mesurée sur
   les pixels** (on compare l'image avant/après le dessin ; la zone modifiée est la boîte), son type
   attendu et son texte attendu. Résultat : `maquette_test_gt.json`, **18 éléments**.
3. Écrit `uitree/evaluation.py` : appariement **un-à-un glouton par IoU** (les paires de plus fort
   IoU d'abord, chaque boîte utilisée une seule fois, seuil 0,5 par défaut), puis rappel, précision,
   exactitude du type, exactitude du texte et **CER** (taux d'erreur par caractère, distance de
   Levenshtein).

### 4.2 Résultat du code d'origine (référence)

| Mesure | Valeur |
|---|---|
| Rappel | 13/18 = 0,72 |
| Précision | 13/14 = 0,93 |
| Type exact (parmi les appariés) | 11/13 |
| IoU minimum des appariés | 0,995 |

Ratés : `message1` et `badge1` fusionnés en une boîte `(75,86,311,19)` (6 px d'écart, dilatation
horizontale de 12 px) ; les trois textes « MOBILE » à l'intérieur des badges jamais détectés
(`RETR_EXTERNAL` ne garde que le contour extérieur) ; menu et recherche typés « Image ».

### 4.3 Tests de départ

Les limites connues ont été écrites comme **échecs attendus stricts** (`xfail(strict=True)`) : dès
qu'une limite est corrigée, pytest signale un « XPASS » et le test doit devenir un test normal. Ils
ont été promus au fur et à mesure (§5, §10). État à la fin de l'étape : 17 tests réussis, 5 échecs
attendus.

---------------------------------------------------------------------------------------------------

## 5. Étape (b) — Détecteur amélioré

### 5.1 Principe (version 2)

1. Couleur de **fond** de la zone (la plus fréquente).
2. Pixels qui s'en écartent = « il y a quelque chose ».
3. **Composantes connexes AVANT toute fusion**, triées en *formes pleines* (badge, bouton, bandeau :
   assez grandes et remplies) et *encre* (lettres, traits).
4. Chaque forme pleine est une boîte, et on **recommence à l'intérieur avec sa propre couleur comme
   fond** (récursion) : cela corrige « un fond de couleur cache son contenu » et « le texte d'un
   badge est invisible ».
5. On épaissit **l'encre seule** pour fusionner les lettres : les formes pleines ne participent plus
   à la fusion, ce qui sépare le texte du badge.

Paramètres déplacés dans `config.toml` (chargé par `uitree/config.py` en objets immuables ; un nom de
paramètre inconnu est **refusé** par une erreur).

### 5.2 Résultat sur la maquette

Rappel 18/18, précision 18/18, IoU minimum 0,975 (après réglage du seuil, voir 5.3). Le badge est
séparé du message, les trois « MOBILE » sont trouvés.

### 5.3 Défauts trouvés en essayant sur une vraie capture (`GUI-Perceptual-Grouping/data/input/1.jpeg`)

| Défaut | Cause | Correction |
|---|---|---|
| Un texte noir sur badge jaune n'était pas cherché | critère « beaucoup de couleurs = photo » : l'anti-crénelage du texte noir donnait 14 couleurs | critère remplacé (voir §8.2, `palette_ratio`) |
| Écran blanc sur fond bleu clair non séparé | seuil de couleur 60 trop haut (écart réel ≈ 61) | seuil passé à **30** |
| Lignes de texte entières supprimées | règle ajoutée « absorber les boîtes contenues dans une autre » | règle **retirée**, test de régression ajouté |

Le seuil 30 n'a été réglé que sur une image : à ne pas considérer comme universel.

---------------------------------------------------------------------------------------------------

## 6. Étape (c) — OCR

### 6.1 Module `uitree/ocr.py`

- Interface `OcrEngine` (protocole) : remplaçable par un autre moteur. Implémentation
  `RapidOcrEngine`, chargée à la première utilisation.
- `TextLine(bbox, text, score)`.
- Filtrage (`read_text`) : confiance ≥ `min_score` (0,5), texte non vide, et **icônes lues comme du
  texte écartées** : une ligne d'**au plus 1 caractère** dans une boîte **presque carrée**
  (rapport ≤ 1,4) est une icône probable (exemples réels : « 8 » sur l'icône partage, « 日 »).
- Les boîtes OCR sont **bornées à l'image** (l'OCR peut déborder d'1 px au bord).

### 6.2 Fusion OCR + pixels (dans `detection.py`)

Chaque morceau d'encre situé **majoritairement dans une ligne OCR** rejoint cette ligne : on obtient
une boîte « texte » **resserrée sur les vrais pixels**, avec son contenu. Deux lignes ne peuvent donc
plus fusionner. Les morceaux « à cheval » (photo collée au nom) restent dans l'encre. Les lignes OCR
sans aucun pixel détecté (texte très pâle) sont ajoutées telles quelles.

Bug corrigé : une ligne rattachée à **1 seul pixel parasite** produisait une boîte 1×1 (« Dr. Ali
Khan » sur la capture). Un seuil `min_line_coverage` (50 % de la largeur de la ligne) a été ajouté,
avec test de régression.

### 6.3 Résultat sur la maquette

| Mesure | Valeur |
|---|---|
| Boîtes retrouvées | 18/18 (inchangé) |
| Textes exacts | **7/10**, CER **1,6 %** |

Erreurs de lecture, **non corrigées volontairement après coup** (pour garder des mesures honnêtes) :
« isjust » et « notfailed » (espaces perdus), « MOBILE, » (virgule en trop).

### 6.4 Interface

Case « Lire le texte (OCR) » ; l'OCR n'est relancé que si l'image change (cache par empreinte md5) :
bouger un curseur ne recalcule que le détecteur.

---------------------------------------------------------------------------------------------------

## 7. Détection des surfaces pâles (« b bis »)

Problème : les boutons, pastilles et cartes **claires sur fond clair** n'étaient pas détectés (écart
de couleur inférieur au seuil).

Solution en **deux seuils** :

- **Seuil bas (`surface_tol` = 10)** pour trouver les *surfaces* : image débruitée (filtre médian 3×3)
  puis **ouverture morphologique** (3×3) qui supprime le bruit JPEG et les traits fins.
- **Seuil normal (`color_tol` = 30)** pour le *contenu* (texte, icônes), hors des surfaces.

Ajouts :

1. **Cartes de la même couleur que le fond** (`_enclosed_background`) : quand les cartes blanches
   sont plus étendues que le gris qui les sépare, le blanc est la couleur *dominante* et les cartes
   ne ressortent pas. On cherche donc les zones **de la couleur du fond, fermées** (elles ne touchent
   ni le bord de la zone ni l'extérieur) et rectangulaires (remplissage ≥ 0,9, ce qui exclut
   l'intérieur d'un avatar rond : 0,785). Les bordures pâles ont des trous : on les épaissit de 2 px
   pour fermer la carte, puis on rend à la carte cette largeur et sa bordure de 1 px (`_grow_card`).
2. **`_drop_frames`** : les morceaux d'encre qui *entourent* une surface entière (fond gris autour
   des cartes, gouttières) sont retirés : ce sont des cadres, pas des éléments.
3. **`_outermost`** : une surface située *dans* une autre est retrouvée par la récursion ; sans ce
   filtre elle était détectée **deux fois**, et chaque bouton « contenait » son double.
4. **`_split_by_shadow`** : une composante qui n'est pas une forme pleine est retentée avec le seuil
   normal (voir §11.1, barre latérale reliée à une bannière par une ombre portée).

### 7.1 Trois défauts cachés trouvés et corrigés en route (chacun a un test)

1. **Couleur de fond fausse jusqu'à 21 points** : le centre d'un groupe de pas 16 est à 7 près par
   canal, ce qui suffisait à faire passer un fond uni pour une « surface » avec un seuil bas, et les
   tests sont passés de 0 à 19 échecs. `dominant_color` calcule maintenant la **moyenne réelle** des
   pixels du groupe gagnant.
2. **Le filtre médian effaçait les bordures de 1 px** : les cartes à bordure fine n'étaient pas
   fermées. La recherche de cartes utilise l'image brute.
3. **`cv2.erode` ne rognait jamais le bord** d'une forme touchant le cadre de sa découpe (par
   défaut OpenCV traite l'extérieur comme plein) : boucle de détection à chaque profondeur.
   `borderValue=0` corrige.

---------------------------------------------------------------------------------------------------

## 8. Étape (d) — Types et styles

### 8.1 Refonte de l'architecture

Le détecteur ne donne **plus de type** : il renvoie des `Detection` avec `source` (`surface`, `ink`,
`text`), `fill`, `depth`, `text`. Le **type est attribué ensuite, en voyant ce que chaque boîte
contient** (un Bouton se définit par son contenu).

### 8.2 Inclusion (`uitree/hierarchy.py`)

`find_parents` : le parent d'une boîte est la **plus petite** boîte capable de contenir qui la
recouvre à 90 % ; égalité de surface départagée par l'indice (déterministe, sans cycle). Une ligne de
texte ne contient rien.

### 8.3 Règles de typage (`uitree/classify.py`, `RuleClassifier`)

Les boîtes sont traitées **des plus petites aux plus grandes** : quand on type un conteneur, ses
enfants sont déjà typés. Chaque décision garde le **nom de la règle** appliquée (traçabilité).
Interface `ElementClassifier` (protocole) : un modèle appris pourra remplacer les règles. Les scores
des règles sont des valeurs fixes, **non calibrées**.

| Type | Règle |
|---|---|
| Texte | ligne lue par l'OCR |
| Image | sans enfant, ≥ 20 px de côté, ≥ 16 couleurs et `palette_ratio` < 0,7 |
| Icône | petite forme (côté ≤ 48 px, presque carrée) sans contenu |
| Bouton | surface de 24 à 64 px de haut avec **un seul** texte court (≤ 30 car.), éventuellement une icône |
| Champ | comme un bouton, mais ≥ 3× plus large que haut et texte calé à gauche |
| Composant | surface trop basse pour un bouton (badge), ou forme sans contenu (barre, séparateur) |
| Conteneur | surface qui contient plusieurs éléments, ou un seul texte mais forme trop grande / texte long |

`palette_ratio` : part des pixels couverts par les 6 couleurs principales (quantifiées par pas de
32). **Mesuré** sur les captures : cartes et bannières d'interface 0,80 à 0,97 (même en dégradé),
photos et illustrations 0,39 à 0,53. Il a remplacé le comptage de couleurs et le test « couleur
dominante », qui prenaient une carte pleine de tuiles colorées pour une photo.

### 8.4 Styles (`uitree/style.py`)

- **Couleur de fond** : couleur dominante de la boîte.
- **Couleur du texte** : médiane du « cœur » des lettres (pixels du 70ᵉ percentile de distance au
  fond, car les bords sont anti-crénelés).
- **Hauteur du texte** : hauteur de la bande de lignes contenant de l'encre. C'est la hauteur des
  caractères, **pas la taille CSS** : mesuré ≈ **0,7 × la taille de police** (±2 px selon
  hampes/jambages). Couleurs précises à quelques unités près (testé sur le formulaire).

### 8.5 Deuxième maquette de test

`make_form_image()` : un formulaire de connexion (20 éléments) conçu pour mesurer les **types** :
Conteneur, Champ (×2), Bouton plein et Bouton contour, Icônes (loupe, triangle, rond, carré), Image,
badge, barre de progression, lien. **Attention : je l'ai conçue moi-même**, donc les règles risquent
d'y être trop bien adaptées ; les vraies captures restent le vrai juge.

### 8.6 Résultats

| | Maquette Contacts | Maquette formulaire |
|---|---|---|
| Boîtes retrouvées | 18/18, 0 en trop | 20/20, **1 en trop** |
| **Types corrects** | **18/18** (avant : 16/18) | **20/20** |
| Textes exacts | 7/10 | 7/8 (CER 3,4 %) |

La boîte en trop vient de l'OCR : « Mot de passe oublié ? » lu « Mot de passe oubli », et le reste
devient une fausse « Icône ». Sans OCR, un test échoue **volontairement** (xfail) : la loupe, à 8 px
du texte du champ, fusionne avec lui.

---------------------------------------------------------------------------------------------------

## 9. Étape (e) — Arbre, JSON, similarité, commande de lot

### 9.1 Arbre (`uitree/tree.py`, `uitree/xycut.py`)

1. **Inclusion d'abord** : une boîte contenue dans une surface devient son enfant.
2. **Puis XY-Cut** parmi les enfants de chaque parent (et parmi les éléments posés sur la page) :
   Column (bandes empilées) d'abord, sinon Row (blocs côte à côte), sinon « Groupe » (chevauchement).
   Les regroupements intermédiaires trouvés par XY-Cut sont des nœuds `source = "xycut"` sans type.
3. **Ids attribués à la fin en parcours préfixe** : même entrée → mêmes ids.
4. **Jumeaux** : un enfant non texte qui recouvre son parent à IoU ≥ 0,9 (même tuile détectée par son
   contour puis par son intérieur) est fusionné avec lui ; ses enfants remontent.

`xycut.py` est de la géométrie pure : il ne crée ni ids ni nœuds. Les **deux limites de la v1** sont
corrigées : le texte d'un badge est bien son **enfant**, et les **ids sont reproductibles**.

Réglages issus des mesures : `min_gap` passé de 8 à **4** (le badge de la ligne 1 est à 6 px du
texte), puis **couloir de 1 px accepté en dernier recours** après avoir constaté que 32 des 45
« Groupe » d'une capture venaient d'éléments simplement trop proches (icône à 3 px de son texte), pas
de vrais chevauchements. Un couloir ≥ `min_gap` reste prioritaire.

### 9.2 Empreintes pour repérer les composants identiques (`uitree/similarity.py`)

Demande de l'utilisateur : détecter les composants identiques, utile pour le **DRY** (un composant
réutilisable au lieu de code copié).

| Empreinte | Calcul | Trouve |
|---|---|---|
| **De structure** | SHA-1 de `type | geo_type | empreintes des enfants dans l'ordre` — **sans texte ni position** | le même composant, **contenu différent** (3 lignes de contact, 4 cartes de statistiques) |
| **Visuelle** (dHash) | image réduite 9×8 en gris, 1 bit par comparaison de voisins | des éléments qui **se ressemblent** (même badge, même champ) |

Adaptations, chacune motivée par une mesure :

- **dHash avec marge** (`DHASH_MARGIN` = 2 niveaux de gris) : le dHash classique (`>`) donne des bits
  au hasard dans les aplats ; **6 bits sur 64 basculaient** avec un bruit JPEG de ±6.
- **Test de couleur de fond** en plus du dHash : il ne voit que la luminosité, donc un bouton bleu et
  un rouge de même forme auraient été « identiques » (test dédié).
- **Filtres contre les répétitions banales** : au moins 2 descendants, dont un qui n'est pas du texte
  (deux lignes de texte empilées ne font pas un composant) ; tailles proches (hauteur ±30 %, largeur
  ×2) ; on ne garde que les répétitions **les plus hautes** (pas le nom dans chaque ligne répétée).

### 9.3 Export JSON (`uitree/export.py`, `uitree/tree.schema.json`)

Chaque nœud : `id`, `bbox {x,y,width,height}`, `type`, `geo_type`, `source`, `parent`, `children`
(liste ordonnée), `text`, `style`, **`label: null`** (réservé au MLLM), `score`, `hashes`. Le document
contient aussi `image`, `repeated_components`, `visual_duplicates`.

**Validation en deux temps** : le schéma JSON (draft 2020-12) vérifie la forme ; le code vérifie ce
que le schéma ne sait pas exprimer (ids uniques, racine sans parent, `parent` cohérent, enfant
contenu dans son parent, ids des groupes existants). Un test vérifie que **6 types de corruption**
sont détectés.

### 9.4 Commande de lot (`batch.py`)

```
.venv/bin/python batch.py images/ --out out/json
```

Écrit un JSON validé et une image annotée par image ; le moteur OCR est chargé une fois ; une image
en échec n'arrête pas le lot (code de sortie 1 s'il y a des échecs). Remplace `pipeline.py`.

### 9.5 Résultats

- Maquette : les **3 lignes de contact** sont reconnues comme le même composant, les **3 badges**
  comme identiques ; sur le formulaire, les **2 champs** sont identiques.
- Deux exécutions sur la même image donnent **exactement le même JSON** (test).

---------------------------------------------------------------------------------------------------

## 10. Étape (f) — Interface et vérité terrain

### 10.1 Interface Gradio (`app.py`)

Lancement : `.venv/bin/python app.py` puis http://127.0.0.1:7860. Sélection d'une image depuis le PC
(ou presse-papiers), curseurs (seuil de couleur, seuil des surfaces, fusion horizontale et verticale,
écart minimum XY-Cut), case OCR. Trois onglets :

| Onglet | Contenu |
|---|---|
| **Analyse** | image annotée avec **une couleur par type** + légende avec les comptes ; tableau des éléments (type, texte, confiance, fond, couleur du texte, hauteur, boîte) ; vues des composants répétés et des doublons visuels (même couleur et même numéro par groupe) ; boutons de téléchargement du **JSON** et de l'**image annotée** |
| **Arbre** | liste **dépliable** ; **schéma** nœuds-liens à partir de n'importe quel nœud et profondeur réglable ; **squelette** (Row en bleu, Column en vert, Groupe en rouge) |
| **Vérité terrain** | partir des détections ; corriger le tableau (type, texte, coordonnées ; type vidé = ligne supprimée) ; **ajouter une boîte oubliée en cliquant deux coins** ; enregistrer dans `verite_terrain/` ; **Évaluer** (rappel, précision, types, CER) |

Couleurs par type : Bouton rouge, Texte bleu, Image vert, Icône orange, Champ violet, Conteneur gris,
Composant turquoise.

API Gradio vérifiée avant usage (signatures de `Image`, `Slider`, `DownloadButton`, `Dataframe`,
`SelectData`).

### 10.2 Défauts trouvés grâce au squelette et aux tests

1. **Barre latérale non détectée** : l'ombre portée de la bannière la reliait au reste de la page
   (une composante en « L » remplie à 31 %). `_split_by_shadow` : sur DASHBOARD, la racine devient
   bien **Row [barre latérale | contenu]**.
2. **Trop de « Groupe »** : 45 → **20** sur DASHBOARD (7 sur SRMS) grâce au couloir de 1 px.
3. **L'interface modifiait l'image avant l'analyse** : Gradio réenregistrait l'image en **WebP avec
   perte** (240 boîtes au lieu de 272 sur la même capture). `format="png"` ; un test vérifie que
   l'interface donne le même résultat que `batch.py`.

### 10.3 Ce qui n'a PAS été fait

Je n'ai **pas cliqué dans un navigateur** : chaque fonction de l'interface a été testée directement,
puis le vrai serveur a été testé à travers l'API de Gradio (`gradio_client`).

---------------------------------------------------------------------------------------------------

## 11. Tests

`.venv/bin/python -m pytest` → **93 réussis, 1 échec attendu** (la loupe sans OCR, §8.6).

| Fichier | Couvre |
|---|---|
| `test_xycut.py` | XY-Cut : feuille, Column/Row ordonnées, écart < et = `min_gap`, couloir de dernier recours, chevauchement, déterminisme |
| `test_detection.py` | éléments séparés, mots fusionnés, **texte proche d'un badge (écarts 2, 4, 6, 10 px)**, texte dans un badge, fond de couleur, cartes blanches, bordure fine, photo, profondeur max, configuration, **critère « 18/18 boîtes correctes »** |
| `test_ocr.py` | deux lignes proches séparées (faux moteur OCR), boîte resserrée, image collée au texte, ligne sans pixels, pixels parasites, règle des icônes, filtrage, **vrai RapidOCR sur la maquette**, déterminisme de l'OCR |
| `test_classify.py` | une règle à la fois, inclusion (sans cycle), types corrects sur les deux maquettes (avec et sans OCR), couleurs mesurées, hauteur du texte |
| `test_tree.py` | texte enfant du badge, ids reproductibles, jumeaux, structure de la maquette, composants répétés, dHash (bruit, motif, couleur), JSON valide, **6 corruptions détectées**, même image → même JSON, commande de lot |
| `test_ground_truth.py` | maquette identique à l'originale (md5), vérité terrain complète, IoU, appariement |
| `test_interface.py` | vérité terrain (aller-retour, lignes invalides, deux clics), vues de l'arbre, interface = batch, image sans perte, évaluation depuis le tableau |

Les limites connues de la v1 ont toutes été **promues en tests normaux** au fil des corrections,
sauf une (xfail restant).

---------------------------------------------------------------------------------------------------

## 12. Architecture finale

```
Detecteur/
├── app.py                  interface Gradio
├── batch.py                traitement d'un dossier -> JSON + images annotées
├── make_test_image.py      maquettes de test + vérité terrain (Contacts, formulaire)
├── config.toml             tous les paramètres (commentés)
├── requirements.txt        versions figées
├── LICENCES.md             licences relevées des dépendances
├── pyproject.toml          configuration pytest (xfail strict)
├── uitree/
│   ├── analysis.py         point d'entrée unique (interface, batch, évaluation)
│   ├── ocr.py              OCR (RapidOCR derrière une interface)
│   ├── detection.py        surfaces, cartes, encre, lignes de texte
│   ├── hierarchy.py        inclusion : qui contient qui
│   ├── classify.py         types par règles (interface remplaçable)
│   ├── style.py            couleurs, hauteur du texte
│   ├── colors.py           mesures de couleur
│   ├── geometry.py         IoU, recouvrement
│   ├── xycut.py            XY-Cut (géométrie pure)
│   ├── tree.py             inclusion puis XY-Cut -> arbre, ids
│   ├── similarity.py       empreintes : composants répétés, doublons visuels
│   ├── export.py           JSON + validation
│   ├── tree.schema.json    schéma du JSON
│   ├── tree_view.py        liste dépliable, schéma, squelette
│   ├── render.py           dessin des boîtes (couleur par type)
│   ├── groundtruth.py      format et conversion de la vérité terrain
│   ├── evaluation.py       appariement IoU, types, texte, CER
│   └── config.py           chargement de config.toml
└── tests/
```

Chaîne : **image → OCR → détection → inclusion → types → styles → arbre → empreintes → JSON validé**.

---------------------------------------------------------------------------------------------------

## 13. Bilan par rapport aux critères de réussite

| Critère | État |
|---|---|
| Maquette de test : toutes les boîtes correctes, badge séparé du texte | ✅ 18/18, aucune boîte en trop, types 18/18 ; texte 7/10 exacts (CER 1,6 %) |
| Liste honnête des erreurs sur une vraie capture | ⚠ **partielle** : observée à l'œil sur 4 captures, **non chiffrée** (pas de vérité terrain des vraies images) |
| Une commande pour traiter un dossier | ✅ `batch.py` |
| Tests, reproductibilité, configuration, licences | ✅ |

---------------------------------------------------------------------------------------------------

## 14. Erreurs et limites connues sur les vraies captures

Observations **à l'œil** sur les 4 captures (`images/`, non publiées dans ce dépôt) :

- **SRMS** : cartes non détectées (bordure à ≈ 3 niveaux sur 255, au niveau du bruit JPEG : aucune
  méthode fondée sur les pixels ne la voit).
- **Fonds en dégradé** : le contenu des bannières est mal découpé.
- **Barres de progression avec « 60 % »** : prises pour des Boutons.
- **Petits fragments** d'icônes ou de texte non lu → fausses « Icônes ».
- **Répétitions** trouvées en partie seulement : dès qu'un élément est détecté différemment d'une
  occurrence à l'autre, la structure change (les 4 lignes de « Live Activity » forment 2 paires). Les
  variantes de couleur (tuiles « Quick Actions ») ne sont pas réunies.
- **Groupes** : il reste 7 à 20 nœuds « Groupe » par image.
- **OCR** : espaces perdus, ponctuation ajoutée, texte blanc sur fond coloré parfois non lu.
- **Réglages** : `color_tol = 30` n'a été validé que sur peu d'images.
- Temps : 2 à 8 s par capture selon la taille.

Aucun de ces points n'est chiffré : il faut une vérité terrain (onglet « Vérité terrain »).

---------------------------------------------------------------------------------------------------

## 15. Suite proposée

1. **Annoter une capture** dans l'onglet « Vérité terrain » (20 à 40 min) pour obtenir de vrais
   chiffres pour le mémoire et identifier le défaut le plus coûteux.
2. Corriger ce défaut en priorité (détection par contours pour les bordures très pâles, ou typage des
   barres de progression, selon les chiffres).
3. Plus tard : classifieur appris à la place des règles (l'interface `ElementClassifier` est prête),
   puis étiquetage sémantique par un MLLM (champ `label`), sans toucher aux positions.
