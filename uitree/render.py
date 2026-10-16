"""Dessin des boîtes détectées, une couleur par type, pour vérifier à l'œil."""
import cv2
import numpy as np

from uitree.detection import Detection
from uitree.geometry import BBox

# Couleurs par type, en RVB (converties en BVR pour OpenCV au moment du dessin).
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "Bouton": (230, 30, 30),
    "Texte": (30, 90, 230),
    "Image": (20, 160, 60),
    "Icône": (245, 140, 0),
    "Champ": (150, 50, 200),
    "Conteneur": (128, 128, 128),
    "Composant": (0, 170, 170),
}
UNKNOWN_COLOR = (0, 0, 0)


GROUP_COLORS: list[tuple[int, int, int]] = [   # RVB, une couleur par groupe de répétition
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40),
]


def annotate_groups(img_bgr: np.ndarray, groups: list[list[BBox]]) -> np.ndarray:
    """Chaque groupe (composant répété, doublons visuels) : même couleur et même numéro."""
    out = (img_bgr * 0.55 + 255 * 0.45).astype(np.uint8)          # image pâlie : les cadres ressortent
    for g, boxes in enumerate(groups):
        r, gr, b = GROUP_COLORS[g % len(GROUP_COLORS)]
        for x, y, w, h in boxes:
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (b, gr, r), 2)
            cv2.putText(out, str(g + 1), (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (b, gr, r), 1,
                        cv2.LINE_AA)
    return out


def annotate(img_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Copie de l'image avec chaque boîte encadrée de la couleur de son type."""
    return annotate_boxes(img_bgr, [(d.bbox, d.kind) for d in detections])


def annotate_boxes(img_bgr: np.ndarray, boxes: list[tuple[BBox, str | None]], thickness: int = 1) -> np.ndarray:
    out = img_bgr.copy()
    # Les plus grandes d'abord : les cadres des petites boîtes restent visibles par-dessus.
    for (x, y, w, h), kind in sorted(boxes, key=lambda bk: -bk[0][2] * bk[0][3]):
        r, g, b = TYPE_COLORS.get(kind, UNKNOWN_COLOR)
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (b, g, r), thickness)
    return out
