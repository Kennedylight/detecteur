"""Opérations sur les boîtes (x, y, largeur, hauteur)."""

BBox = tuple[int, int, int, int]


def intersection_area(a: BBox, b: BBox) -> int:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a: BBox, b: BBox) -> float:
    """Intersection sur union de deux boîtes (0 = disjointes, 1 = identiques)."""
    inter = intersection_area(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def overlap_ratio(inner: BBox, outer: BBox) -> float:
    """Part de la surface de inner qui est dans outer."""
    area = inner[2] * inner[3]
    return intersection_area(inner, outer) / area if area else 0.0
