"""Arbre (inclusion puis XY-Cut), empreintes de similarité, export JSON et sa validation."""
import copy
import json

import numpy as np
import pytest

from batch import main as batch_main
from make_test_image import make_form_image, make_test_image
from uitree.analysis import analyse_to_tree
from uitree.config import Config
from uitree.detection import Detection
from uitree.export import TreeValidationError, validate_document
from uitree.ocr import RapidOcrEngine
from uitree.similarity import dhash, hamming
from uitree.tree import build_tree


def _bgr(img) -> np.ndarray:
    return np.asarray(img)[:, :, ::-1].copy()


@pytest.fixture(scope="module")
def contacts():
    img, _ = make_test_image()
    return analyse_to_tree(_bgr(img), "maquette_test.png", Config(), ocr=RapidOcrEngine())


# --- arbre -----------------------------------------------------------------------------------

def _det(bbox, source="surface", kind="Conteneur", text=None):
    return Detection(bbox, source, 0, kind=kind, text=text)


def test_texte_dans_un_badge_devient_son_enfant():
    """Était une limite connue de la v1 (le texte devenait le frère du badge dans un 'Groupe')."""
    root = build_tree([_det((0, 0, 50, 20), kind="Composant"),
                       _det((8, 6, 30, 8), "text", "Texte", "OK")], (100, 40), Config())
    badge = root.children[0]
    assert badge.type == "Composant" and [c.text for c in badge.children] == ["OK"]


def test_ids_reproductibles_et_prefixes():
    """Était une limite connue de la v1 (le compteur d'ids n'était pas remis à zéro)."""
    dets = [_det((0, 0, 10, 10), "ink", "Icône"), _det((0, 50, 10, 10), "ink", "Icône")]
    a, b = build_tree(dets, (100, 100), Config()), build_tree(dets, (100, 100), Config())
    assert [n.id for n in a.walk()] == [n.id for n in b.walk()] == [0, 1, 2]


def test_jumeau_imbrique_fusionne():
    dets = [_det((10, 10, 100, 60)), _det((11, 11, 98, 58)), _det((20, 20, 40, 10), "text", "Texte", "a")]
    root = build_tree(dets, (200, 100), Config())
    assert [n.type for n in root.walk()] == [None, "Conteneur", "Texte"]


def test_maquette_trois_lignes_de_contact(contacts):
    root = contacts.tree
    assert root.geo_type == "Column" and len(root.children) == 4          # en-tête + 3 lignes
    for row in root.children[1:]:
        assert row.geo_type == "Row"
        assert [c.type or c.geo_type for c in row.children] == ["Image", "Column", "Composant"]
        assert row.children[2].children[0].text.startswith("MOBILE")


# --- similarité --------------------------------------------------------------------------------

def test_maquette_lignes_repetees_et_badges_identiques(contacts):
    doc = contacts.document
    rows = [n.id for n in contacts.tree.children[1:]]
    badges = [r.children[2].id for r in contacts.tree.children[1:]]
    assert [g["node_ids"] for g in doc["repeated_components"]] == [rows]
    assert [g["node_ids"] for g in doc["visual_duplicates"]] == [badges]


def test_formulaire_champs_visuellement_identiques():
    img, _ = make_form_image()
    result = analyse_to_tree(_bgr(img), "f.png", Config(), ocr=RapidOcrEngine())
    nodes = {n.id: n for n in result.tree.walk()}
    groups = [[nodes[i].type for i in g["node_ids"]] for g in result.document["visual_duplicates"]]
    assert groups == [["Champ", "Champ"]]


def test_dhash_robuste_au_bruit_et_sensible_au_motif():
    rng = np.random.default_rng(0)
    img = np.full((60, 120, 3), 255, np.uint8)
    img[10:50, 10:50] = (200, 80, 30)                          # un carré
    img[:, 60:][:, ::12] = 0                                   # des rayures verticales
    noisy = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    assert hamming(dhash(img, (0, 0, 60, 60)), dhash(noisy, (0, 0, 60, 60))) <= 2
    assert hamming(dhash(img, (0, 0, 60, 60)), dhash(img, (60, 0, 60, 60))) > 8


def test_meme_forme_autre_couleur_pas_doublon_visuel():
    """Le dHash ne voit pas la couleur : c'est le test de couleur de fond qui les sépare."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (300, 80), "white")
    d = ImageDraw.Draw(img)
    for x0, color in ((10, (37, 99, 235)), (160, (220, 38, 38))):
        d.rounded_rectangle([x0, 20, x0 + 120, 56], radius=8, fill=color)
    result = analyse_to_tree(_bgr(img), "b.png", Config(), lines=[])
    assert result.document["visual_duplicates"] == []


# --- JSON ----------------------------------------------------------------------------------------

def test_document_valide_et_serialisable(contacts):
    doc = json.loads(json.dumps(contacts.document, ensure_ascii=False))
    validate_document(doc)
    assert doc["root"]["label"] is None and all(n["label"] is None for n in _walk(doc["root"]))


def _walk(node):
    yield node
    for c in node["children"]:
        yield from _walk(c)


@pytest.mark.parametrize("corrupt,message", [
    (lambda d: d["root"]["children"][0].__setitem__("parent", 99), "parent"),
    (lambda d: d["root"]["children"][1].__setitem__("id", 1), "double"),
    (lambda d: d["root"]["children"][0]["bbox"].__setitem__("x", 900), "déborde"),
    (lambda d: d["root"].__setitem__("type", "Bidule"), "Schéma"),
    (lambda d: d["root"].pop("label"), "Schéma"),
    (lambda d: d["repeated_components"].append(
        {"structure_hash": "0" * 12, "node_ids": [1, 999], "max_visual_distance": 0, "exact": True}), "inconnus"),
])
def test_validation_detecte_les_corruptions(contacts, corrupt, message):
    doc = copy.deepcopy(contacts.document)
    corrupt(doc)
    with pytest.raises(TreeValidationError, match=message):
        validate_document(doc)


def test_meme_image_meme_json(contacts):
    img, _ = make_test_image()
    again = analyse_to_tree(_bgr(img), "maquette_test.png", Config(), ocr=RapidOcrEngine())
    assert again.document == contacts.document


def test_commande_dossier(tmp_path):
    img, _ = make_test_image()
    (tmp_path / "in").mkdir()
    img.save(tmp_path / "in" / "a.png")
    assert batch_main([str(tmp_path / "in"), "--out", str(tmp_path / "out")]) == 0
    validate_document(json.loads((tmp_path / "out" / "a.json").read_text(encoding="utf-8")))
    assert (tmp_path / "out" / "a_annotee.png").exists()
