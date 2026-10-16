"""Interface locale : déposer une maquette, régler les paramètres, voir les boîtes, l'arbre et les
composants répétés, puis corriger les boîtes pour créer une vérité terrain et mesurer.

Lancement :  .venv/bin/python app.py   puis ouvrir http://127.0.0.1:7860
"""
import dataclasses
import hashlib
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from uitree.analysis import Result, analyse_to_tree, load_bgr
from uitree.classify import KINDS
from uitree.config import load_config
from uitree.evaluation import Item, evaluate, format_report
from uitree.export import write_document
from uitree.groundtruth import (TABLE_HEADERS, RowError, bbox_from_corners, ground_truth_from_rows,
                                rows_from_detections, save_ground_truth)
from uitree.ocr import RapidOcrEngine, TextLine, read_text
from uitree.render import TYPE_COLORS, annotate, annotate_boxes, annotate_groups
from uitree.tree_view import find_node, tree_diagram, tree_html, wireframe

CFG = load_config()
OCR = RapidOcrEngine()
OUT_DIR = Path(tempfile.mkdtemp(prefix="uitree_"))
GT_DIR = Path("verite_terrain")
_ocr_cache: dict[str, list[TextLine]] = {}


def rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cached_lines(bgr: np.ndarray) -> list[TextLine]:
    """L'OCR (~1 s) ne dépend pas des curseurs : on ne le relance que si l'image change."""
    key = hashlib.md5(bgr.tobytes()).hexdigest()
    if key not in _ocr_cache:
        _ocr_cache.clear()
        _ocr_cache[key] = read_text(bgr, OCR, CFG.ocr)
    return _ocr_cache[key]


def legend_html(counts: Counter) -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:0 14px 4px 0">'
        f'<span style="width:14px;height:14px;border:3px solid rgb{color};margin-right:6px"></span>'
        f'{kind} ({counts.get(kind, 0)})</span>'
        for kind, color in TYPE_COLORS.items())
    return f"<div>{items}</div>"


# --- onglet Analyse ----------------------------------------------------------------------------------

def analyse(path: str | None, color_tol: int, surface_tol: int, merge_x: int, merge_y: int,
            min_gap: int, use_ocr: bool):
    if not path:
        raise gr.Error("Dépose d'abord une image.")
    bgr = load_bgr(path)
    cfg = dataclasses.replace(
        CFG,
        detection=dataclasses.replace(CFG.detection, color_tol=color_tol, surface_tol=surface_tol,
                                      merge_x=merge_x, merge_y=merge_y),
        xycut=dataclasses.replace(CFG.xycut, min_gap=min_gap))
    result = analyse_to_tree(bgr, Path(path).name, cfg, lines=cached_lines(bgr) if use_ocr else [])
    dets, doc = result.detections, result.document
    boxes = {n.id: n.bbox for n in result.tree.walk()}
    repeated = annotate_groups(bgr, [[boxes[i] for i in g["node_ids"]] for g in doc["repeated_components"]])
    duplicates = annotate_groups(bgr, [[boxes[i] for i in g["node_ids"]] for g in doc["visual_duplicates"]])
    annotated = annotate(bgr, dets)

    stem = Path(path).stem
    img_path, json_path = OUT_DIR / f"{stem}_annotee.png", OUT_DIR / f"{stem}.json"
    cv2.imwrite(str(img_path), annotated)
    write_document(doc, json_path)
    rows = [[d.kind, d.text or "", round(d.score, 2), d.style.background, d.style.text_color or "",
             d.style.text_height_px or "", *d.bbox] for d in dets]
    n_text = sum(d.text is not None for d in dets)
    summary = (f"**{len(dets)} boîtes détectées, dont {n_text} textes lus — "
               f"{len(doc['repeated_components'])} composants répétés, "
               f"{len(doc['visual_duplicates'])} groupes d'éléments visuellement identiques**")
    return (result, rgb(annotated), legend_html(Counter(d.kind for d in dets)), summary, rows,
            rgb(repeated), rgb(duplicates), str(json_path), str(img_path))


# --- onglet Arbre --------------------------------------------------------------------------------------

def show_tree(result: Result | None, node_id: float, depth: int):
    if result is None:
        return "<i>Analyse d'abord une image.</i>", None, None
    root = result.document["root"]
    start = find_node(root, int(node_id or 0))
    if start is None:
        raise gr.Error(f"Pas de nœud #{int(node_id)} dans l'arbre.")
    return tree_html(root), np.asarray(tree_diagram(start, int(depth))), np.asarray(wireframe(root))


# --- onglet Vérité terrain ------------------------------------------------------------------------------

def gt_preview(path: str | None, rows: list[list]):
    if not path:
        return None, "Analyse d'abord une image."
    try:
        items = ground_truth_from_rows(_rows(rows))
    except RowError as e:
        return None, f"⚠ {e}"
    img = annotate_boxes(load_bgr(path), [(g.bbox, g.kind) for g in items], thickness=2)
    return rgb(img), f"{len(items)} éléments dans la vérité terrain."


def gt_from_detections(result: Result | None):
    if result is None:
        raise gr.Error("Analyse d'abord une image.")
    return rows_from_detections(result.detections)


def gt_click(path: str | None, rows: list[list], corner: list[int] | None, kind: str, text: str,
             evt: gr.SelectData):
    """1er clic : un coin ; 2e clic : le coin opposé -> nouvelle ligne."""
    point = (int(evt.index[0]), int(evt.index[1]))
    rows = _rows(rows)
    if corner is None:
        return rows, list(point), f"Coin noté en {point} : clique le coin opposé."
    x, y, w, h = bbox_from_corners(tuple(corner), point)
    rows.append([f"{kind}_{len(rows) + 1}", kind, x, y, w, h, text or ""])
    return rows, None, f"Boîte ajoutée : {kind} ({x}, {y}, {w}×{h})."


def gt_save(path: str | None, rows: list[list]):
    if not path:
        raise gr.Error("Analyse d'abord une image.")
    try:
        items = ground_truth_from_rows(_rows(rows))
    except RowError as e:
        raise gr.Error(str(e)) from e
    GT_DIR.mkdir(exist_ok=True)
    out = GT_DIR / f"{Path(path).stem}_gt.json"
    save_ground_truth(items, out)
    return str(out), f"Enregistré : {out} ({len(items)} éléments)."


def gt_evaluate(result: Result | None, rows: list[list]):
    if result is None:
        raise gr.Error("Analyse d'abord une image.")
    try:
        truth = [Item(g.bbox, g.kind, g.text, g.name) for g in ground_truth_from_rows(_rows(rows))]
    except RowError as e:
        raise gr.Error(str(e)) from e
    preds = [Item(d.bbox, d.kind, d.text) for d in result.detections]
    return format_report(truth, preds, evaluate(truth, preds))


def _rows(rows) -> list[list]:
    """Le tableau Gradio arrive en DataFrame pandas ou en liste selon les versions."""
    if rows is None:
        return []
    return rows.values.tolist() if hasattr(rows, "values") else [list(r) for r in rows]


# --- mise en page ------------------------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    d = CFG.detection
    with gr.Blocks(title="Maquette -> arbre") as ui:
        result_state = gr.State(None)
        corner_state = gr.State(None)
        gr.Markdown("# Maquette → arbre hiérarchique")
        with gr.Row():
            with gr.Column(scale=1):
                # format="png" : sinon Gradio réenregistre l'image en WebP AVEC PERTE avant l'analyse
                # (mesuré : 240 boîtes au lieu de 272 sur la même capture).
                image_in = gr.Image(type="filepath", format="png", label="Image de la maquette",
                                    sources=["upload", "clipboard"])
                use_ocr = gr.Checkbox(value=CFG.ocr.enabled, label="Lire le texte (OCR)")
                color_tol = gr.Slider(5, 150, value=d.color_tol, step=1, label="Seuil de couleur",
                                      info="Contenu (texte, icônes) : plus bas = détecte le contenu pâle")
                surface_tol = gr.Slider(2, 60, value=d.surface_tol, step=1, label="Seuil des surfaces",
                                        info="Cartes, boutons, pastilles : plus bas = trouve les surfaces très pâles")
                merge_x = gr.Slider(1, 40, value=d.merge_x, step=1, label="Fusion horizontale (px)",
                                    info="Plus haut = fusionne des morceaux plus éloignés")
                merge_y = gr.Slider(1, 20, value=d.merge_y, step=1, label="Fusion verticale (px)")
                min_gap = gr.Slider(1, 40, value=CFG.xycut.min_gap, step=1, label="Écart minimum XY-Cut (px)")
                run = gr.Button("Analyser", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("Analyse"):
                        legend = gr.HTML(legend_html(Counter()))
                        image_out = gr.Image(type="numpy", label="Boîtes détectées", interactive=False)
                        with gr.Row():
                            dl_json = gr.DownloadButton("Télécharger l'arbre (JSON)")
                            dl_img = gr.DownloadButton("Télécharger l'image annotée")
                        table = gr.Dataframe(headers=["Type", "Texte lu", "Confiance", "Fond", "Couleur texte",
                                                      "Haut. texte (px)", "x", "y", "largeur", "hauteur"],
                                             label="Éléments détectés", interactive=False)
                        with gr.Row():
                            repeated_out = gr.Image(type="numpy", interactive=False,
                                                    label="Composants répétés (même structure)")
                            duplicates_out = gr.Image(type="numpy", interactive=False,
                                                      label="Éléments visuellement identiques")
                    with gr.Tab("Arbre"):
                        with gr.Row():
                            node_id = gr.Number(value=0, precision=0, label="Schéma à partir du nœud n°")
                            depth = gr.Slider(1, 8, value=3, step=1, label="Profondeur affichée")
                        diagram = gr.Image(type="numpy", label="Schéma de l'arbre", interactive=False)
                        skeleton = gr.Image(type="numpy", interactive=False,
                                            label="Squelette : Row en bleu, Column en vert, Groupe en rouge")
                        tree_list = gr.HTML(label="Arbre dépliable")
                    with gr.Tab("Vérité terrain"):
                        gr.Markdown(
                            "1. **Partir des détections**, puis corrige le tableau (type, texte, coordonnées). "
                            "Vide la colonne *type* d'une ligne pour la supprimer.\n"
                            "2. Boîte oubliée : choisis son type, puis **clique deux coins opposés** sur l'aperçu.\n"
                            "3. **Enregistrer** (dans `verite_terrain/`), puis **Évaluer** l'analyse actuelle.")
                        with gr.Row():
                            start_gt = gr.Button("Partir des détections")
                            new_kind = gr.Dropdown(list(KINDS), value="Bouton", label="Type de la nouvelle boîte")
                            new_text = gr.Textbox(label="Texte de la nouvelle boîte (facultatif)")
                        gt_status = gr.Markdown()
                        gt_image = gr.Image(type="numpy", interactive=False,
                                            label="Aperçu de la vérité terrain (clique pour ajouter une boîte)")
                        gt_table = gr.Dataframe(headers=TABLE_HEADERS, interactive=True, label="Vérité terrain")
                        with gr.Row():
                            save_gt = gr.Button("Enregistrer")
                            eval_gt = gr.Button("Évaluer", variant="primary")
                            dl_gt = gr.DownloadButton("Télécharger la vérité terrain")
                        report = gr.Textbox(label="Évaluation", lines=20, max_lines=40)

        inputs = [image_in, color_tol, surface_tol, merge_x, merge_y, min_gap, use_ocr]
        outputs = [result_state, image_out, legend, summary, table, repeated_out, duplicates_out, dl_json, dl_img]
        tree_inputs, tree_outputs = [result_state, node_id, depth], [tree_list, diagram, skeleton]
        for event in (run.click, image_in.upload, use_ocr.change,
                      *(s.release for s in (color_tol, surface_tol, merge_x, merge_y, min_gap))):
            event(analyse, inputs, outputs).then(show_tree, tree_inputs, tree_outputs)
        node_id.submit(show_tree, tree_inputs, tree_outputs)
        depth.release(show_tree, tree_inputs, tree_outputs)

        start_gt.click(gt_from_detections, [result_state], [gt_table])
        gt_table.change(gt_preview, [image_in, gt_table], [gt_image, gt_status])
        gt_image.select(gt_click, [image_in, gt_table, corner_state, new_kind, new_text],
                        [gt_table, corner_state, gt_status])
        save_gt.click(gt_save, [image_in, gt_table], [dl_gt, gt_status])
        eval_gt.click(gt_evaluate, [result_state, gt_table], [report])
    return ui


if __name__ == "__main__":
    build_ui().launch()
