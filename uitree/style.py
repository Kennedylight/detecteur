"""Style d'un élément, mesuré sur les pixels : couleur dominante, couleur du texte, taille du texte."""
from dataclasses import dataclass

import numpy as np

from uitree.colors import color_distance, dominant_color, to_hex
from uitree.geometry import BBox


@dataclass(frozen=True)
class Style:
    background: str             # couleur dominante de la boîte, "#rrggbb"
    text_color: str | None      # couleur du texte (lignes de texte seulement)
    text_height_px: int | None  # hauteur des caractères mesurée (≈ taille de police, voir plus bas)


def measure_style(img_bgr: np.ndarray, bbox: BBox, is_text: bool, tol: int) -> Style:
    x, y, w, h = bbox
    crop = img_bgr[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return Style("#000000", None, None)
    bg = dominant_color(crop)
    if not is_text:
        return Style(to_hex(bg), None, None)
    dist = color_distance(crop, bg)
    ink = dist > tol
    if not ink.any():
        return Style(to_hex(bg), None, None)
    return Style(to_hex(bg), to_hex(_text_color(crop, dist, ink)), _text_height(ink))


def _text_color(crop: np.ndarray, dist: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Cœur des lettres : les pixels les plus éloignés du fond (les bords sont anti-crénelés)."""
    core = dist >= np.percentile(dist[ink], 70)
    return np.median(crop[core & ink], axis=0)


def _text_height(ink: np.ndarray) -> int:
    """Hauteur de la bande de lignes contenant de l'encre. C'est la hauteur des caractères
    (des hampes aux jambages), PAS la taille CSS : l'écart se mesure sur la maquette de test."""
    rows = np.nonzero(ink.any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1)
