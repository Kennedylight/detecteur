"""XY-Cut : découpe récursive d'un ensemble de boîtes en Row / Column (géométrie pure).

Idée (reprise de la version 1, xy_cut.py) :
  1. On regarde toutes les boîtes d'une zone.
  2. On cherche un "couloir vide" qui traverse la zone de bout en bout :
       - couloir horizontal (entre deux bandes)  -> les parties sont empilées -> "Column"
       - couloir vertical (entre deux blocs)     -> les parties sont côte à côte -> "Row"
  3. On coupe, puis on recommence dans chaque morceau (récursion).
  4. S'il ne reste qu'une boîte -> "Feuille". Si on ne peut plus couper alors qu'il y a plusieurs
     boîtes (elles se chevauchent) -> "Groupe".

Différences avec la version 1 : ce module ne crée plus d'ids ni de nœuds (c'est tree.py qui le
fait, de façon reproductible), et l'inclusion (texte DANS un badge) est traitée AVANT, dans tree.py.
"""
from collections.abc import Sequence
from dataclasses import dataclass

from uitree.geometry import BBox

LEAF, ROW, COLUMN, GROUP = "Feuille", "Row", "Column", "Groupe"


@dataclass(frozen=True)
class Cut:
    geo_type: str                       # Feuille | Row | Column | Groupe
    index: int | None = None            # indice de la boîte (Feuille seulement)
    children: tuple["Cut", ...] = ()    # ordonnés : haut -> bas (Column), gauche -> droite (Row)


def split_on_gaps(boxes: Sequence[BBox], indices: list[int], axis: int, min_gap: int) -> list[list[int]]:
    """Groupes séparés par un couloir vide d'au moins min_gap pixels.

    axis = 0 -> axe x (couloirs verticaux) ; axis = 1 -> axe y (couloirs horizontaux).
    """
    ordered = sorted(indices, key=lambda i: (boxes[i][axis], boxes[i][1 - axis], i))
    groups, current = [], [ordered[0]]
    current_end = boxes[ordered[0]][axis] + boxes[ordered[0]][axis + 2]
    for i in ordered[1:]:
        start = boxes[i][axis]
        if start - current_end >= min_gap:          # couloir vide trouvé -> on coupe ici
            groups.append(current)
            current = [i]
        else:
            current.append(i)
        current_end = max(current_end, start + boxes[i][axis + 2])
    groups.append(current)
    return groups


def xy_cut(boxes: Sequence[BBox], min_gap: int = 8, indices: list[int] | None = None) -> Cut:
    indices = list(range(len(boxes))) if indices is None else indices
    if len(indices) == 1:
        return Cut(LEAF, indices[0])
    # Convention web : d'abord des bandes EMPILÉES (Column), sinon des blocs CÔTE À CÔTE (Row).
    # (Tester d'abord "le plus grand couloir" donnait un arbre faux : voir xy_cut.py, version 1.)
    # En dernier recours, un couloir d'1 px suffit : sur les vraies captures, 32 des 45 "Groupe"
    # venaient d'éléments simplement trop proches (icône à 3 px de son texte), pas de chevauchements.
    for gap in (min_gap, 1) if min_gap > 1 else (1,):
        stacked = split_on_gaps(boxes, indices, axis=1, min_gap=gap)
        if len(stacked) > 1:
            return Cut(COLUMN, children=tuple(xy_cut(boxes, min_gap, g) for g in stacked))
        side = split_on_gaps(boxes, indices, axis=0, min_gap=gap)
        if len(side) > 1:
            return Cut(ROW, children=tuple(xy_cut(boxes, min_gap, g) for g in side))
    # Impossible de couper : les boîtes se chevauchent. On les garde ensemble, en ordre de lecture.
    ordered = sorted(indices, key=lambda i: (boxes[i][1], boxes[i][0], i))
    return Cut(GROUP, children=tuple(Cut(LEAF, i) for i in ordered))
