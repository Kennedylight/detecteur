"""Classification grossière : Texte, Image, Icône, Bouton, Champ, Composant, Conteneur.

Interface `ElementClassifier` : reçoit TOUTES les détections et leurs liens d'inclusion, rend un
type par détection. Les règles ci-dessous (RuleClassifier) pourront être remplacées par un modèle
appris sans rien changer d'autre. Les scores des règles sont des valeurs fixes, NON calibrées.

Les boîtes sont traitées des plus petites aux plus grandes : quand on type un conteneur, ses
enfants sont déjà typés (un Bouton = une surface qui contient UN texte court centré).
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from uitree.colors import center_crop, count_colors, palette_ratio
from uitree.config import ClassifyConfig
from uitree.detection import Detection
from uitree.geometry import BBox
from uitree.hierarchy import children_of

TEXT, IMAGE, ICON, BUTTON, FIELD, COMPONENT, CONTAINER = (
    "Texte", "Image", "Icône", "Bouton", "Champ", "Composant", "Conteneur")
KINDS = (TEXT, IMAGE, ICON, BUTTON, FIELD, COMPONENT, CONTAINER)


@dataclass(frozen=True)
class Classification:
    kind: str
    score: float
    rule: str           # règle appliquée (traçabilité ; vide pour un modèle appris)


class ElementClassifier(Protocol):
    def __call__(self, img_bgr: np.ndarray, dets: Sequence[Detection],
                 parents: Sequence[int | None]) -> list[Classification]: ...


@dataclass(frozen=True)
class RuleClassifier:
    cfg: ClassifyConfig = ClassifyConfig()

    def __call__(self, img_bgr: np.ndarray, dets: Sequence[Detection],
                 parents: Sequence[int | None]) -> list[Classification]:
        kids = children_of(parents)
        out: list[Classification | None] = [None] * len(dets)
        for i in sorted(range(len(dets)), key=lambda i: dets[i].bbox[2] * dets[i].bbox[3]):
            children = [(dets[j], out[j]) for j in kids[i]]
            out[i] = self._classify(img_bgr, dets[i], children)
        return out  # type: ignore[return-value]

    def _classify(self, img: np.ndarray, d: Detection,
                  children: list[tuple[Detection, Classification]]) -> Classification:
        if d.source == "text":
            return Classification(TEXT, d.score, "ligne lue par l'OCR")
        x, y, w, h = d.bbox
        crop = img[y:y + h, x:x + w]
        if not children and (self._is_photo(crop) or self._is_photo(self._center(crop))):
            return Classification(IMAGE, 0.6, "beaucoup de couleurs, pas d'aplats")
        if children and self._is_photo_with_overlays(crop, d.bbox, children):
            return Classification(IMAGE, 0.5, "photo au centre, petits éléments posés dessus")
        if children:
            return self._classify_shape_with_content(d, children)
        if min(w, h) <= self.cfg.line_max_thickness:
            return Classification(COMPONENT, 0.5, "trait fin (séparateur)")
        if self._is_small_square(w, h):
            return Classification(ICON, 0.6, "petite forme presque carrée, sans contenu")
        if d.source == "ink" and self._looks_like_unread_text(w, h, d.fill):
            return Classification(TEXT, 0.4, "encre allongée et peu remplie (texte non lu)")
        if d.source == "ink" and min(w, h) > self.cfg.icon_max_side:
            return Classification(IMAGE, 0.4, "grand dessin sans fond (illustration, graphique)")
        return Classification(COMPONENT, 0.4, "forme sans contenu (barre, séparateur, puce)")

    def _classify_shape_with_content(self, d: Detection,
                                     children: list[tuple[Detection, Classification]]) -> Classification:
        texts = [c for c, k in children if k.kind == TEXT]
        others = [k.kind for c, k in children if k.kind != TEXT]
        x, y, w, h = d.bbox
        if not texts and others == [ICON] and self._is_icon_button(d):
            return Classification(BUTTON, 0.5, "forme pleine presque carrée avec une seule icône (bouton icône)")
        if len(texts) != 1 or len(others) > 1 or any(k != ICON for k in others):
            return Classification(CONTAINER, 0.5, "contient plusieurs éléments")
        if h < self.cfg.button_min_height:
            return Classification(COMPONENT, 0.5, "trop bas pour un bouton (badge, étiquette)")
        if h > self.cfg.button_max_height or len(texts[0].text or "") > self.cfg.button_max_chars:
            return Classification(CONTAINER, 0.4, "un seul texte mais forme trop grande / texte long")
        if self._is_left_aligned(d.bbox, texts[0].bbox) and w / h >= self.cfg.field_min_aspect:
            return Classification(FIELD, 0.5, "forme allongée, texte calé à gauche")
        return Classification(BUTTON, 0.6, "forme basse, un seul texte court")

    def _is_photo(self, crop: np.ndarray) -> bool:
        if min(crop.shape[:2]) < self.cfg.photo_min_side:
            return False
        return (count_colors(crop) >= self.cfg.photo_min_colors
                and palette_ratio(crop) < self.cfg.photo_max_palette)

    def _is_photo_with_overlays(self, crop: np.ndarray, box: BBox,
                                children: list[tuple[Detection, Classification]]) -> bool:
        """Avatar avec une pastille "15" posée dessus : la photo est au CENTRE (les coins d'un
        avatar rond sont du fond uni, qui faussait la mesure sur toute la boîte), et les enfants
        sont petits par rapport à elle."""
        area = box[2] * box[3]
        if any(c.bbox[2] * c.bbox[3] > self.cfg.overlay_max_area * area for c, _ in children):
            return False
        return self._is_photo(self._center(crop))

    def _center(self, crop: np.ndarray) -> np.ndarray:
        return center_crop(crop, self.cfg.photo_center)

    def _is_icon_button(self, d: Detection) -> bool:
        w, h = d.bbox[2], d.bbox[3]
        return (d.source == "surface" and max(w, h) / max(1, min(w, h)) <= self.cfg.icon_button_max_aspect
                and self.cfg.button_min_height <= min(w, h) and max(w, h) <= self.cfg.icon_button_max_side)

    def _is_small_square(self, w: int, h: int) -> bool:
        return max(w, h) <= self.cfg.icon_max_side and max(w, h) / max(1, min(w, h)) <= self.cfg.icon_max_aspect

    def _looks_like_unread_text(self, w: int, h: int, fill: float) -> bool:
        return w / h >= self.cfg.text_min_aspect and h <= self.cfg.text_max_height and fill < 0.75

    def _is_left_aligned(self, box: BBox, text: BBox) -> bool:
        start = (text[0] - box[0]) / box[2]
        center_shift = ((box[0] + box[2] / 2) - (text[0] + text[2] / 2)) / box[2]
        return start <= self.cfg.field_text_start and center_shift >= self.cfg.centered_tol
