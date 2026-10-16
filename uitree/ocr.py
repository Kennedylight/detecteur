"""OCR : image -> lignes de texte (boîte, contenu, confiance).

Moteur par défaut : RapidOCR (Apache-2.0), local et hors ligne (modèles PP-OCR fournis dans le
paquet, exécutés par onnxruntime). Interface `OcrEngine` : remplaçable par un autre moteur.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from uitree.config import OcrConfig
from uitree.geometry import BBox


@dataclass(frozen=True)
class TextLine:
    bbox: BBox          # boîte alignée sur les axes, telle que rendue par l'OCR (avec sa marge)
    text: str
    score: float


class OcrEngine(Protocol):
    def __call__(self, img_bgr: np.ndarray) -> list[TextLine]: ...


class RapidOcrEngine:
    """RapidOCR, chargé à la première utilisation (le chargement des modèles prend ~1 s)."""

    def __init__(self) -> None:
        self._engine = None

    def __call__(self, img_bgr: np.ndarray) -> list[TextLine]:
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        result = self._engine(img_bgr)          # un tableau numpy à 3 canaux est lu comme du BVR
        if result.boxes is None:
            return []
        h, w = img_bgr.shape[:2]
        return [TextLine(_quad_to_bbox(q, w, h), t, float(s))
                for q, t, s in zip(result.boxes, result.txts, result.scores)]


def _quad_to_bbox(quad: np.ndarray, width: int, height: int) -> BBox:
    """Quadrilatère OCR -> boîte alignée, limitée à l'image (l'OCR peut déborder d'1 px au bord)."""
    xs, ys = quad[:, 0], quad[:, 1]
    x0, y0 = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
    x1, y1 = min(width, int(np.ceil(xs.max()))), min(height, int(np.ceil(ys.max())))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def is_icon_like(line: TextLine, cfg: OcrConfig) -> bool:
    """Un ou deux symboles dans une boîte presque carrée : souvent une icône lue comme du texte."""
    w, h = line.bbox[2], line.bbox[3]
    return len(line.text.strip()) <= cfg.icon_max_chars and max(w, h) / max(1, min(w, h)) <= cfg.icon_max_aspect


def read_text(img_bgr: np.ndarray, engine: OcrEngine, cfg: OcrConfig) -> list[TextLine]:
    """Lignes retenues : confiance suffisante, et pas une icône probable."""
    return [line for line in engine(img_bgr)
            if line.score >= cfg.min_score and line.text.strip() and not is_icon_like(line, cfg)]
