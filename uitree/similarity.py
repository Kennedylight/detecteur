"""Empreintes pour repérer les composants identiques (base du DRY à la génération de code).

- Empreinte VISUELLE (dHash, 64 bits) : calculée sur les pixels. Deux éléments qui se ressemblent
  ont des empreintes à faible distance de Hamming, même avec du bruit JPEG.
- Empreinte de STRUCTURE : type + géométrie + empreintes des enfants dans l'ordre, SANS le texte
  ni la position. Même empreinte = même composant, contenu différent (3 lignes de contact, 4 cartes).
"""
import hashlib
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from uitree.config import SimilarityConfig
from uitree.editdistance import levenshtein
from uitree.geometry import BBox
from uitree.tree import Node


@dataclass(frozen=True)
class RepeatedComponent:
    structure_hash: str
    node_ids: list[int]
    max_visual_distance: int        # 0 = identiques à l'œil ; grand = même structure, autre apparence
    exact: bool                     # False : même composant à une partie optionnelle près (émoji, pastille)


@dataclass(frozen=True)
class VisualDuplicate:
    node_ids: list[int]             # éléments sans enfants qui se ressemblent (même icône, même bouton)


DHASH_MARGIN = 2   # niveaux de gris


def dhash(img_bgr: np.ndarray, bbox: BBox) -> str:
    """Hash de différence : l'image réduite en 9x8 niveaux de gris, 1 bit par comparaison de voisins.

    Variante pour les interfaces : un bit vaut 1 seulement si le voisin est plus clair d'au moins
    DHASH_MARGIN niveaux. Le dHash classique (>) donne des bits au hasard dans les aplats, où deux
    voisins sont égaux au bruit près : mesuré, 6 bits sur 64 basculaient avec un bruit JPEG de ±6."""
    x, y, w, h = bbox
    crop = img_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return "0" * 16
    small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    small = small.astype(np.int16)
    bits = (small[:, 1:] - small[:, :-1] >= DHASH_MARGIN).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def add_hashes(root: Node, img_bgr: np.ndarray) -> None:
    """Remplit structure_hash et visual_hash de chaque nœud (des feuilles vers la racine)."""
    for c in root.children:
        add_hashes(c, img_bgr)
    signature = f"{root.type}|{root.geo_type}|" + ",".join(c.structure_hash for c in root.children)
    root.structure_hash = hashlib.sha1(signature.encode()).hexdigest()[:12]
    root.visual_hash = dhash(img_bgr, root.bbox)


def _descendants(node: Node) -> int:
    return sum(1 for _ in node.walk()) - 1


def repeated_components(root: Node, cfg: SimilarityConfig) -> list[RepeatedComponent]:
    """Sous-arbres de même structure et de taille proche, présents au moins 2 fois.

    - EXACTS : même empreinte de structure.
    - APPROCHÉS (exact = False) : mêmes type et géométrie, et suites de types des enfants à au plus
      approx_max_edits différences. Sur une vraie capture, les 7 lignes de contact ne sont pas
      strictement identiques : un émoji sur certaines, une pastille que l'OCR ne lit pas toujours.
      Pour le DRY, c'est le même composant avec une partie optionnelle.
    Filtres contre les répétitions banales : au moins min_descendants descendants, dont au moins
    un qui n'est pas du texte (deux lignes de texte empilées ne font pas un composant).
    On ne garde que les répétitions les plus hautes (pas le nom DANS chaque ligne répétée), et une
    répétition exacte comprise dans une approchée n'est pas répétée."""
    candidates = [n for n in root.walk()
                  if n.source != "page" and _descendants(n) >= cfg.min_descendants and _has_non_text_part(n)]
    by_hash: dict[str, list[Node]] = defaultdict(list)
    for node in candidates:
        by_hash[node.structure_hash].append(node)
    exact = [g for nodes in by_hash.values() for g in _size_clusters(nodes, cfg) if len(g) >= 2]
    approx = [g for g in _approx_clusters(candidates, cfg)
              if len(g) >= 2 and len({n.structure_hash for n in g}) > 1]
    exact = [g for g in exact if not any({n.id for n in g} <= {n.id for n in a} for a in approx)]
    groups = [(g, True) for g in exact] + [(g, False) for g in approx]
    inside_a_group = {d.id for g, _ in groups for n in g for d in n.walk() if d is not n}
    result = []
    for nodes, is_exact in groups:
        if all(n.id in inside_a_group for n in nodes):
            continue
        dist = max(hamming(a.visual_hash, b.visual_hash) for a in nodes for b in nodes)
        common = max(sorted({n.structure_hash for n in nodes}), key=lambda h: sum(n.structure_hash == h for n in nodes))
        result.append(RepeatedComponent(common, sorted(n.id for n in nodes), dist, is_exact))
    return sorted(result, key=lambda r: r.node_ids[0])


def _has_non_text_part(node: Node) -> bool:
    return any(d.type not in (None, "Texte") for d in node.walk() if d is not node)


def _child_kinds(node: Node) -> list[str]:
    return [c.type or c.geo_type for c in node.children]


def _approx_clusters(nodes: list[Node], cfg: SimilarityConfig) -> list[list[Node]]:
    """Regroupement par liaison simple : un nœud rejoint un groupe s'il ressemble à L'UN de ses
    membres (même type et géométrie, taille proche, enfants à <= approx_max_edits différences).
    Seulement pour des sous-arbres assez riches (approx_min_descendants) : deux rangées de 3
    icônes / textes (barre d'application, barre d'onglets) se ressemblent "à un enfant près"
    sans être le même composant."""
    clusters: list[list[Node]] = []
    for n in (n for n in nodes if len(n.children) >= cfg.approx_min_children
              and _descendants(n) >= cfg.approx_min_descendants):
        matching = [c for c in clusters if any(_approx_same(m, n, cfg) for m in c)]
        merged = [n] + [m for c in matching for m in c]
        clusters = [c for c in clusters if c not in matching] + [merged]
    return clusters


def _approx_same(a: Node, b: Node, cfg: SimilarityConfig) -> bool:
    return (a.type == b.type and a.geo_type == b.geo_type
            and _ratio(a.bbox[3], b.bbox[3]) <= cfg.max_height_ratio
            and _ratio(a.bbox[2], b.bbox[2]) <= cfg.max_width_ratio
            and levenshtein(_child_kinds(a), _child_kinds(b)) <= cfg.approx_max_edits)


def _size_clusters(nodes: list[Node], cfg: SimilarityConfig) -> list[list[Node]]:
    """Regroupement glouton par taille proche (premier élément de chaque groupe = référence)."""
    clusters: list[list[Node]] = []
    for n in nodes:
        for c in clusters:
            if _ratio(c[0].bbox[3], n.bbox[3]) <= cfg.max_height_ratio \
                    and _ratio(c[0].bbox[2], n.bbox[2]) <= cfg.max_width_ratio:
                c.append(n)
                break
        else:
            clusters.append([n])
    return clusters


def _ratio(p: int, q: int) -> float:
    return max(p, q) / max(1, min(p, q))


def visual_duplicates(root: Node, cfg: SimilarityConfig) -> list[VisualDuplicate]:
    """Éléments détectés (hors texte et regroupements) visuellement identiques : même type, taille
    proche, même couleur de fond, et dHash à distance <= max_visual_distance (le dHash ne voit que
    la luminosité : sans le test de couleur, un bouton bleu et un rouge de même forme seraient
    "identiques"). Regroupement glouton, le premier élément de chaque groupe sert de référence."""
    elements = [n for n in root.walk() if n.source == "detection" and n.type != "Texte"]
    groups: list[list[Node]] = []
    for element in elements:
        for g in groups:
            if _look_alike(g[0], element, cfg):
                g.append(element)
                break
        else:
            groups.append([element])
    return [VisualDuplicate([n.id for n in g]) for g in groups if len(g) >= 2]


def _look_alike(a: Node, b: Node, cfg: SimilarityConfig) -> bool:
    return (a.type == b.type
            and hamming(a.visual_hash, b.visual_hash) <= cfg.max_visual_distance
            and _similar_size(a.bbox, b.bbox, cfg.max_size_ratio)
            and _color_gap(a, b) <= cfg.max_color_distance)


def _color_gap(a: Node, b: Node) -> int:
    if a.style is None or b.style is None:
        return 0
    ca, cb = (np.array([int(s.background[i:i + 2], 16) for i in (1, 3, 5)]) for s in (a.style, b.style))
    return int(np.abs(ca - cb).sum())


def _similar_size(a: BBox, b: BBox, max_ratio: float) -> bool:
    return _ratio(a[2], b[2]) <= max_ratio and _ratio(a[3], b[3]) <= max_ratio
