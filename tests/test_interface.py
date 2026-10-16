"""Vérité terrain (format, tableau), vues de l'arbre, et fonctions de l'interface."""
import gradio as gr
import pytest

import app
from uitree.groundtruth import (GroundTruth, RowError, bbox_from_corners, ground_truth_from_rows,
                                load_ground_truth, save_ground_truth)
from uitree.tree_view import find_node, tree_diagram, tree_html, wireframe


def test_aller_retour_fichier(tmp_path):
    items = [GroundTruth("a", (1, 2, 3, 4), "Bouton", "OK"), GroundTruth("b", (5, 6, 7, 8), "Texte")]
    save_ground_truth(items, tmp_path / "gt.json")
    assert load_ground_truth(tmp_path / "gt.json") == items


def test_lignes_du_tableau():
    rows = [["ok", "Bouton", "10", 20.0, 30, 40, "Valider"],
            ["supprimée", "", 0, 0, 1, 1, ""],
            ["sans_texte", "Icône", 1, 2, 3, 4, None]]
    assert ground_truth_from_rows(rows) == [GroundTruth("ok", (10, 20, 30, 40), "Bouton", "Valider"),
                                            GroundTruth("sans_texte", (1, 2, 3, 4), "Icône", None)]


@pytest.mark.parametrize("row,message", [
    (["x", "Bidule", 0, 0, 5, 5, ""], "inconnu"),
    (["x", "Texte", "a", 0, 5, 5, ""], "numériques"),
    (["x", "Texte", 0, 0, 0, 5, ""], "> 0"),
])
def test_lignes_invalides(row, message):
    with pytest.raises(RowError, match=message):
        ground_truth_from_rows([row])


def test_deux_clics_dans_n_importe_quel_ordre():
    assert bbox_from_corners((50, 40), (10, 20)) == bbox_from_corners((10, 20), (50, 40)) == (10, 20, 41, 21)


@pytest.fixture(scope="module")
def analysed():
    return app.analyse("maquette_test.png", 30, 10, 12, 4, 4, True)


def test_vues_de_l_arbre(analysed):
    root = analysed[0].document["root"]
    assert "<details" in tree_html(root) and "MOBILE" in tree_html(root)
    assert wireframe(root).size == (400, 270)
    full, shallow = tree_diagram(root, 8), tree_diagram(root, 1)
    assert shallow.size[0] < full.size[0]                    # profondeur 1 : moins de feuilles affichées
    assert find_node(root, 5)["id"] == 5 and find_node(root, 999) is None


def test_interface_mesure_comme_batch(analysed):
    """Même image -> mêmes détections que la chaîne en ligne de commande."""
    from uitree.analysis import analyse_to_tree, load_bgr
    from uitree.config import load_config
    direct = analyse_to_tree(load_bgr("maquette_test.png"), "maquette_test.png", load_config())
    assert analysed[0].document == direct.document


def test_image_envoyee_sans_perte():
    """Régression : par défaut Gradio réenregistre l'image en WebP avec perte (240 boîtes au lieu de 272)."""
    ui = app.build_ui()
    inputs = [b for b in ui.blocks.values() if isinstance(b, gr.Image) and b.type == "filepath"]
    assert inputs and all(b.format == "png" for b in inputs)


def test_evaluation_depuis_le_tableau(analysed):
    result = analysed[0]
    rows = app.gt_from_detections(result)
    assert "Rappel    : 18/18" in app.gt_evaluate(result, rows)
    rows[0][1] = ""                                           # une ligne supprimée de la vérité terrain
    assert "Précision : 17/18" in app.gt_evaluate(result, rows)
