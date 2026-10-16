"""Point d'entrée unique : image -> détections typées et stylées (OCR, détecteur, classification,
styles). Utilisé par l'interface, batch.py et l'évaluation, pour qu'ils mesurent la même chose."""
import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np

from uitree.classify import ElementClassifier, RuleClassifier
from uitree.config import Config
from uitree.detection import Detection, detect_in_image
from uitree.export import tree_document, validate_document
from uitree.hierarchy import find_parents
from uitree.ocr import OcrEngine, RapidOcrEngine, TextLine, read_text
from uitree.similarity import add_hashes, repeated_components, visual_duplicates
from uitree.style import measure_style
from uitree.tree import Node, build_tree


def analyse_image(img_bgr: np.ndarray, cfg: Config, ocr: OcrEngine | None = None,
                  classifier: ElementClassifier | None = None,
                  lines: list[TextLine] | None = None) -> list[Detection]:
    """lines : lignes OCR déjà calculées (évite de relancer l'OCR quand seuls les réglages changent)."""
    if lines is None:
        lines = read_text(img_bgr, ocr or RapidOcrEngine(), cfg.ocr) if cfg.ocr.enabled else []
    dets = detect_in_image(img_bgr, cfg.detection, lines)
    parents = find_parents([d.bbox for d in dets], [d.source != "text" for d in dets],
                           cfg.classify.contain_overlap)
    kinds = (classifier or RuleClassifier(cfg.classify))(img_bgr, dets, parents)
    return [dataclasses.replace(d, kind=k.kind, score=k.score,
                                style=measure_style(img_bgr, d.bbox, d.source == "text",
                                                    cfg.detection.color_tol))
            for d, k in zip(dets, kinds)]


@dataclass(frozen=True)
class Result:
    detections: list[Detection]
    tree: Node
    document: dict                  # JSON validé (voir export.py et tree.schema.json)


def analyse_to_tree(img_bgr: np.ndarray, image_file: str, cfg: Config, ocr: OcrEngine | None = None,
                    lines: list[TextLine] | None = None) -> Result:
    """Chaîne complète : détections -> arbre -> empreintes -> document JSON validé."""
    dets = analyse_image(img_bgr, cfg, ocr=ocr, lines=lines)
    root = build_tree(dets, (img_bgr.shape[1], img_bgr.shape[0]), cfg)
    add_hashes(root, img_bgr)
    doc = tree_document(root, image_file, repeated_components(root, cfg.similarity),
                        visual_duplicates(root, cfg.similarity))
    validate_document(doc, cfg.classify.contain_overlap)
    return Result(dets, root, doc)


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable ou illisible : {path}")
    return img
