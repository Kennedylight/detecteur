"""Inclusion : qui contient qui. Utilisé par la classification (que contient ce bouton ?) et,
plus tard, par la construction de l'arbre (inclusion d'abord, puis XY-Cut)."""
from collections.abc import Sequence

from uitree.geometry import BBox, overlap_ratio


def find_parents(boxes: Sequence[BBox], can_contain: Sequence[bool],
                 min_overlap: float = 0.9) -> list[int | None]:
    """Parent de chaque boîte = la plus PETITE boîte capable de contenir qui la recouvre à
    min_overlap près (une boîte n'est jamais son propre parent ; à surface égale, celle de plus
    petit indice contient l'autre, pour que le résultat soit déterministe et sans cycle)."""
    def area(i: int) -> int:
        return boxes[i][2] * boxes[i][3]

    parents: list[int | None] = []
    for i, box in enumerate(boxes):
        candidates = [j for j in range(len(boxes))
                      if j != i and can_contain[j]
                      and (area(j), j) > (area(i), i)
                      and overlap_ratio(box, boxes[j]) >= min_overlap]
        parents.append(min(candidates, key=lambda j: (area(j), j)) if candidates else None)
    return parents


def children_of(parents: Sequence[int | None]) -> list[list[int]]:
    kids: list[list[int]] = [[] for _ in parents]
    for i, p in enumerate(parents):
        if p is not None:
            kids[p].append(i)
    return kids
