"""Détections -> arbre hiérarchique.

1. INCLUSION d'abord : une boîte contenue dans une surface (texte dans un bouton, cartes dans un
   panneau) devient son enfant (hierarchy.find_parents).
2. Puis XY-CUT parmi les enfants de chaque parent (et parmi les éléments posés sur la page) :
   ils sont rangés en Row / Column. Les regroupements intermédiaires trouvés par XY-Cut deviennent
   des nœuds "Groupe" sans type grossier (source = "xycut").
3. Ids attribués à la fin, en parcours préfixe : même entrée -> mêmes ids.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

from uitree.config import Config
from uitree.detection import Detection
from uitree.geometry import BBox, iou
from uitree.hierarchy import children_of, find_parents
from uitree.style import Style
from uitree.xycut import COLUMN, LEAF, Cut, xy_cut


@dataclass
class Node:
    bbox: BBox
    type: str | None                 # type grossier (None pour un regroupement XY-Cut ou la page)
    geo_type: str                    # Feuille | Row | Column | Groupe
    source: str                      # "page" | "detection" | "xycut"
    text: str | None = None
    style: Style | None = None
    score: float | None = None
    children: list["Node"] = field(default_factory=list)
    id: int = -1
    parent: int | None = None
    label: str | None = None         # étiquette sémantique : remplie plus tard par le MLLM
    structure_hash: str = ""         # voir similarity.py
    visual_hash: str = ""

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def build_tree(dets: Sequence[Detection], size: tuple[int, int], cfg: Config) -> Node:
    """size = (largeur, hauteur) de l'image."""
    parents = find_parents([d.bbox for d in dets], [d.source != "text" for d in dets],
                           cfg.classify.contain_overlap)
    builder = _Builder(dets, children_of(parents), cfg.xycut.min_gap, cfg.xycut.twin_iou)
    roots = [i for i, p in enumerate(parents) if p is None]
    root = Node((0, 0, size[0], size[1]), None, COLUMN, "page")
    builder.arrange(root, roots)
    _number(root)
    return root


@dataclass
class _Builder:
    dets: Sequence[Detection]
    kids: list[list[int]]
    min_gap: int
    twin_iou: float

    def arrange(self, node: Node, indices: list[int]) -> None:
        """Range des détections DANS node : le premier niveau de XY-Cut donne son geo_type."""
        indices = self._without_twins(node.bbox, indices)
        if not indices:
            node.geo_type = LEAF if node.source == "detection" else node.geo_type
            return
        cut = xy_cut([self.dets[i].bbox for i in indices], self.min_gap)
        cut = _reindex(cut, indices)
        if cut.geo_type == LEAF:                       # un seul enfant : on l'empile
            node.geo_type, node.children = COLUMN, [self.convert(cut)]
        else:
            node.geo_type, node.children = cut.geo_type, [self.convert(c) for c in cut.children]

    def _without_twins(self, bbox: BBox, indices: list[int]) -> list[int]:
        """Un enfant non-texte qui recouvre presque tout son parent (même tuile détectée par son
        contour puis par son intérieur) est fusionné avec lui : ses propres enfants remontent."""
        out: list[int] = []
        for i in indices:
            if self.dets[i].source != "text" and iou(self.dets[i].bbox, bbox) >= self.twin_iou:
                out += self._without_twins(bbox, self.kids[i])
            else:
                out.append(i)
        return out

    def convert(self, cut: Cut) -> Node:
        if cut.geo_type == LEAF:
            d = self.dets[cut.index]
            node = Node(d.bbox, d.kind, LEAF, "detection", d.text, d.style, round(d.score, 4))
            self.arrange(node, self.kids[cut.index])
            return node
        children = [self.convert(c) for c in cut.children]
        return Node(_union([c.bbox for c in children]), None, cut.geo_type, "xycut", children=children)


def _reindex(cut: Cut, indices: list[int]) -> Cut:
    """XY-Cut a travaillé sur une sous-liste : on remet les indices de la liste complète."""
    if cut.geo_type == LEAF:
        return Cut(LEAF, indices[cut.index])
    return Cut(cut.geo_type, children=tuple(_reindex(c, indices) for c in cut.children))


def _union(boxes: list[BBox]) -> BBox:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _number(root: Node) -> None:
    """Ids en parcours préfixe (la page = 0) et lien vers le parent."""
    for i, node in enumerate(root.walk()):
        node.id = i
    for node in root.walk():
        for c in node.children:
            c.parent = node.id
