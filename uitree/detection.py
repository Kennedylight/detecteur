"""Détecteur maison, version 3 : image (+ lignes OCR) -> liste de Detection.

Principe (aucune IA dans cette partie, juste des pixels) :
  1. On trouve la couleur de FOND de la zone (la couleur la plus fréquente).
  2. Tout pixel qui s'en écarte nettement = "il y a quelque chose ici".
  3. On découpe ces pixels en morceaux connexes, AVANT toute fusion, et on les trie :
       - forme PLEINE et assez grande (badge, bouton, bandeau de couleur) : c'est une boîte à
         elle seule, et on RECOMMENCE à l'intérieur avec SA couleur comme fond (récursion).
         -> corrige "un fond de couleur cache son contenu" et "le texte d'un badge est invisible".
       - le reste (lettres, traits, icônes fines) : "encre".
  4. Les morceaux d'encre situés surtout dans une ligne lue par l'OCR forment cette ligne
     (une boîte Texte avec son contenu). Deux lignes ne peuvent donc plus fusionner.
  5. On épaissit l'encre restante pour regrouper ses morceaux voisins (icônes en plusieurs traits,
     texte non lu par l'OCR). Les formes pleines ne participent pas à la fusion.
     -> corrige "un texte collé à un badge fusionne avec lui".
  6. Chaque tache = une boîte, resserrée sur ses vrais pixels. (Pas d'absorption des taches
     contenues dans une autre : sur une vraie capture, elle supprimait des lignes de texte entières.)
  7. Pas de type ici : classify.py l'attribue ensuite, en voyant ce que chaque boîte contient.
  8. Les lignes OCR sans aucun pixel détecté (texte très pâle) sont ajoutées telles quelles.

Limites connues (version 3) :
  - Deux morceaux d'encre non lus par l'OCR et plus proches que merge_x fusionnent.
  - Un dégradé ou une image de fond n'a pas de couleur dominante : la récursion n'y aide pas.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from uitree.colors import center_crop, color_distance, dominant_color, palette_ratio
from uitree.config import DetectionConfig
from uitree.geometry import BBox, overlap_ratio
from uitree.ocr import TextLine
from uitree.style import Style


@dataclass(frozen=True)
class Detection:
    bbox: BBox                      # (x, y, largeur, hauteur) dans l'image entière
    source: str                     # comment elle a été trouvée : "surface" | "ink" | "text"
    depth: int                      # 0 = posé sur le fond de page, 1 = dans une surface, etc.
    fill: float = 0.0               # part des pixels de la boîte qui appartiennent à l'élément
    text: str | None = None         # contenu lu par l'OCR
    score: float = 1.0              # confiance : OCR pour une ligne, puis classifieur pour le type
    kind: str | None = None         # type grossier, rempli par la classification (classify.py)
    style: Style | None = None      # rempli par style.py


@dataclass(frozen=True)
class _Context:
    cfg: DetectionConfig
    lines: Sequence[TextLine]       # coordonnées de l'image entière


def foreground_mask(img: np.ndarray, bg: np.ndarray, region: np.ndarray, tol: int) -> np.ndarray:
    """Pixels de la région qui s'écartent du fond de plus de tol (uint8 0/1)."""
    return ((color_distance(img, bg) > tol) & region).astype(np.uint8)


def detect_in_image(img: np.ndarray, cfg: DetectionConfig = DetectionConfig(),
                    lines: Sequence[TextLine] = ()) -> list[Detection]:
    """Boîtes SANS type (kind=None) : le type est attribué ensuite, en voyant le contenu."""
    ctx = _Context(cfg, lines)
    region = np.ones(img.shape[:2], dtype=bool)
    found = _detect_region(img, region, dominant_color(img), (0, 0), 0, ctx)
    found += _unplaced_lines(lines, found)
    return sorted(found, key=lambda d: (d.bbox[1], d.bbox[0], -d.bbox[2] * d.bbox[3]))


def _is_solid(w: int, h: int, area: int, cfg: DetectionConfig) -> bool:
    return min(w, h) >= cfg.min_solid_side and area / (w * h) >= cfg.solid_fill


def surface_mask(img: np.ndarray, bg: np.ndarray, region: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Masque à seuil BAS pour trouver les surfaces pâles (carte blanche sur fond gris clair).

    Image débruitée (médian) puis ouverture morphologique : le bruit JPEG et les traits fins
    disparaissent, seules les zones pleines restent.
    """
    mask = foreground_mask(cv2.medianBlur(img, 3), bg, region, cfg.surface_tol)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.surface_open, cfg.surface_open))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _detect_region(img: np.ndarray, region: np.ndarray, bg: np.ndarray, offset: tuple[int, int],
                   depth: int, ctx: _Context) -> list[Detection]:
    """Détecte les éléments d'une zone (img et region sont des découpes ; offset = leur origine).

    Deux seuils : les formes pleines (surfaces) sont cherchées avec le seuil bas, le contenu
    (encre : texte, icônes) avec le seuil normal, hors des surfaces (traitées par récursion).
    """
    solids = [s for s in _solid_shapes(img, bg, region, ctx.cfg) if not _is_text_shape(s[0], offset, ctx.lines)]
    solid_cover = np.zeros(img.shape[:2], dtype=bool)
    for (x, y, w, h), filled, _ in solids:
        solid_cover[y:y + h, x:x + w] |= filled
    cards = [(box, filled, int(filled.sum()))
             for box, filled in _enclosed_background(img, bg, region, solid_cover, ctx.cfg)]
    covered = np.zeros(img.shape[:2], dtype=bool)
    surfaces: list[BBox] = []                      # coordonnées locales
    found: list[Detection] = []
    for (x, y, w, h), filled, area in _outermost(solids + cards, img.shape[:2]):
        covered[y:y + h, x:x + w] |= filled
        surfaces.append((x, y, w, h))
        found += _detect_solid(img, filled, (x, y, w, h), area, offset, depth, ctx, bg)
    # Le bord anti-crénelé d'une surface (rogné par l'ouverture) ne doit pas devenir de l'encre.
    covered = _dilate(covered, ctx.cfg.edge_margin)
    ink = (foreground_mask(img, bg, region, ctx.cfg.color_tol).astype(bool) & ~covered).astype(np.uint8)
    ink = _drop_frames(ink, surfaces)
    ink = _drop_shadows(ink, img, bg, covered, surfaces, ctx.cfg)
    if depth > 0:
        ink = _drop_edge_ink(ink, region)
    text_found, ink = _extract_text_lines(ink, offset, depth, ctx.lines, ctx.cfg.min_line_coverage)
    return found + text_found + _detect_ink(ink, offset, depth, ctx)


Shape = tuple[BBox, np.ndarray, int]               # (boîte, forme pleine trous bouchés, pixels réels)


def _is_text_shape(box: BBox, offset: tuple[int, int], lines: Sequence[TextLine]) -> bool:
    """Une "surface" presque entièrement dans une ligne OCR, et pas plus haute qu'elle, est du
    texte (lettres grasses avec un halo, qui forment une tache pleine) : sinon le texte est détecté
    deux fois ("Contacts • 318" : 24 px pour une ligne OCR de 28). Une boîte OCR a une marge autour
    des lettres ; un badge, lui, dépasse sa ligne (19 px pour 17 sur la maquette, 26 pour 14)."""
    glob = (box[0] + offset[0], box[1] + offset[1], box[2], box[3])
    return any(overlap_ratio(glob, ln.bbox) >= 0.8 and box[3] <= ln.bbox[3] for ln in lines)


def _solid_shapes(img: np.ndarray, bg: np.ndarray, region: np.ndarray, cfg: DetectionConfig) -> list[Shape]:
    """Formes pleines (seuil bas) : boutons, badges, bandeaux, cartes colorées."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(surface_mask(img, bg, region, cfg), connectivity=8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if _is_solid(w, h, area, cfg):
            shapes.append(((x, y, w, h), _filled(labels[y:y + h, x:x + w] == i), area))
        elif min(w, h) >= cfg.min_solid_side:
            shapes += _split_by_shadow(img, bg, labels == i, (x, y, w, h), cfg)
    return shapes


def _split_by_shadow(img: np.ndarray, bg: np.ndarray, component: np.ndarray, box: BBox,
                     cfg: DetectionConfig) -> list[Shape]:
    """Seconde chance pour une composante qui n'est pas une forme pleine : des surfaces reliées par
    une ombre portée pâle (barre latérale collée à une bannière par son ombre : un "L" rempli à
    31 %). Avec le seuil normal, l'ombre disparaît et les surfaces se séparent."""
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]
    strong = ((color_distance(crop, bg) > cfg.color_tol) & component[y:y + h, x:x + w]).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.surface_open, cfg.surface_open))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(strong, cv2.MORPH_OPEN, kernel),
                                                           connectivity=8)
    shapes = []
    for i in range(1, n):
        sx, sy, sw, sh, area = (int(v) for v in stats[i])
        part = labels[sy:sy + sh, sx:sx + sw] == i
        if _is_solid(sw, sh, area, cfg):
            shapes.append(((x + sx, y + sy, sw, sh), _filled(part), area))
        elif 2 * cfg.min_solid_side <= min(sw, sh) and max(sw, sh) <= cfg.thick_max_side:
            shapes += [((x + sx + bx, y + sy + by, bw, bh), f, a)
                       for (bx, by, bw, bh), f, a in _thick_parts(part, cfg)]
    return shapes


def _thick_parts(part: np.ndarray, cfg: DetectionConfig) -> list[Shape]:
    """Troisième chance, pour de PETITES formes (côté <= thick_max_side) : deux formes qui se
    touchent (bouton flottant rond posé sur un badge) ne font pas une forme pleine. Sur un grand
    bloc (8 écrans de téléphone reliés par leurs ombres), elle était lente (disque de ~385 px) et
    fabriquait une surface géante qui masquait tout : 181 éléments au lieu de 286. Une ouverture par un disque de la moitié du plus petit côté (au moins
    2 x min_solid_side) efface ce qui est plus fin (badge et son ombre : ~36 px) et garde les
    parties épaisses (le rond de 101 px), testées à leur tour."""
    k = max(2 * cfg.min_solid_side, min(part.shape) // 2) | 1
    # PETITS trous bouchés avant l'ouverture : l'icône blanche du bouton flottant y fait un trou, et
    # l'ouverture n'aurait laissé qu'un anneau. Pas les grands : boucher l'intérieur d'un cadre
    # (fond gris autour des cartes) en ferait un rectangle plein de la taille de l'image.
    opened = cv2.morphologyEx(_fill_small_holes(part, 0.3).astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    shapes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        # seulement des parties ÉPAISSES : sinon les coins d'un cadre deviennent des "surfaces"
        if min(w, h) >= k - 1 and _is_solid(w, h, area, cfg):
            shapes.append(((x, y, w, h), labels[y:y + h, x:x + w] == i, area))
    return shapes


def _outermost(shapes: list[Shape], size: tuple[int, int]) -> list[Shape]:
    """Ne garde que les formes qui ne sont pas DANS une autre : celles qui sont dedans seront
    retrouvées en cherchant à l'intérieur de la grande (sinon chacune serait détectée deux fois)."""
    taken = np.zeros(size, dtype=bool)
    kept = []
    for shape in sorted(shapes, key=lambda s: -int(s[1].sum())):
        (x, y, w, h), filled, _ = shape
        if (taken[y:y + h, x:x + w] & filled).sum() > 0.5 * filled.sum():
            continue
        taken[y:y + h, x:x + w] |= filled
        kept.append(shape)
    return kept


def _drop_shadows(ink: np.ndarray, img: np.ndarray, bg: np.ndarray, covered: np.ndarray,
                  surfaces: list[BBox], cfg: DetectionConfig) -> np.ndarray:
    """Restes d'ombre portée autour d'une surface (ombre du bouton flottant : une tache de 112x50
    qui reliait deux lignes de contact et bloquait XY-Cut). Un morceau d'encre est une ombre s'il
    touche une surface, reste dans sa zone d'ombre (halo_max_margin) et est PÂLE : plus proche du
    fond que n'importe quel texte, même gris clair."""
    if not surfaces:
        return ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    touching = set(np.unique(labels[_dilate(covered, 2) & (labels > 0)]).tolist())
    m = cfg.halo_max_margin
    zones = [(x - m, y - m, w + 2 * m, h + 2 * m) for x, y, w, h in surfaces]
    shadows = []
    for i in touching:
        box = tuple(int(v) for v in stats[i][:4])
        if not any(overlap_ratio(box, z) >= 0.95 for z in zones):
            continue
        if np.median(color_distance(img[labels == i], bg)) <= 2 * cfg.color_tol:
            shadows.append(i)
    return (ink.astype(bool) & ~np.isin(labels, shadows)).astype(np.uint8) if shadows else ink


def _drop_edge_ink(ink: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Dans une surface, l'encre qui TOUCHE son bord est l'ombrage du bord (bouton rond en relief),
    pas du contenu : une icône ou un texte ne touche jamais le bord de sa forme."""
    n, labels, _, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    outside = ~region
    outside[[0, -1], :] = True
    outside[:, [0, -1]] = True
    edge = np.unique(labels[_dilate(outside, 1) & (labels > 0)])
    return (ink.astype(bool) & ~np.isin(labels, edge)).astype(np.uint8) if edge.size else ink


def _drop_frames(ink: np.ndarray, surfaces: list[BBox]) -> np.ndarray:
    """Retire de l'encre les morceaux qui ENTOURENT une surface entière : fond gris autour des
    cartes, réseau de gouttières, bordure d'une carte. Ce sont des cadres, pas des éléments."""
    if not surfaces:
        return ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    frames = [i for i in range(1, n)
              if any(overlap_ratio(s, tuple(int(v) for v in stats[i][:4])) == 1.0 for s in surfaces)]
    return (ink.astype(bool) & ~np.isin(labels, frames)).astype(np.uint8) if frames else ink


def _enclosed_background(img: np.ndarray, bg: np.ndarray, region: np.ndarray, covered: np.ndarray,
                         cfg: DetectionConfig) -> list[tuple[BBox, np.ndarray]]:
    """Cartes de la MÊME couleur que le fond, délimitées par une bordure fine ou un espace plus foncé.

    Cas fréquent : cartes blanches plus étendues que le fond gris clair qui les sépare ; le blanc
    est alors la couleur dominante et les cartes ne "ressortent" pas. On cherche donc les zones de
    couleur du fond, fermées (ne touchent ni le bord de la zone ni l'extérieur), et rectangulaires.
    """
    # Image brute (pas de filtre médian : il efface les bordures de 1 px). Le bruit JPEG ne fait
    # que des points isolés, qui ne ferment aucune zone.
    raw_separators = foreground_mask(img, bg, region, cfg.surface_tol).astype(bool)
    # Les bordures pâles ont des trous : on les épaissit pour fermer les cartes, puis on rend à
    # chaque carte la largeur retirée et sa bordure (plus bas).
    separators = _dilate(raw_separators, cfg.card_gap_close)
    background = (region & ~separators & ~covered).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    outside = ~region
    outside[[0, -1], :] = True
    outside[:, [0, -1]] = True
    touching = set(np.unique(labels[_dilate(outside, 1)]).tolist())
    cards = []
    for i in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if i in touching or min(w, h) < cfg.card_min_side:
            continue
        filled = _filled(labels[y:y + h, x:x + w] == i)
        if filled.sum() / (w * h) < cfg.card_rect_fill:
            continue
        cards.append(_grow_card(filled, (x, y, w, h), raw_separators, region, cfg.card_gap_close))
    return cards


def _grow_card(filled: np.ndarray, box: BBox, raw_separators: np.ndarray, region: np.ndarray,
               m: int) -> tuple[BBox, np.ndarray]:
    """Rend à la carte les m px retirés par l'épaississement, plus sa bordure (1 px de séparateur
    collé à elle). Retourne la boîte resserrée et la forme."""
    x, y, w, h = box
    pad = m + 1
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(region.shape[1], x + w + pad), min(region.shape[0], y + h + pad)
    shape = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    shape[y - y0:y - y0 + h, x - x0:x - x0 + w] = filled
    shape = _dilate(shape, m) & region[y0:y1, x0:x1]
    shape |= _dilate(shape, 1) & raw_separators[y0:y1, x0:x1]
    ys, xs = np.nonzero(shape)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return ((int(x0 + bx0), int(y0 + by0), int(bx1 - bx0), int(by1 - by0)), shape[by0:by1, bx0:bx1])


def _detect_solid(img: np.ndarray, filled: np.ndarray, box: BBox, area: int,
                  offset: tuple[int, int], depth: int, ctx: _Context, outer_bg: np.ndarray) -> list[Detection]:
    """Une forme pleine : sa propre boîte, puis ce qu'elle contient (sur sa couleur de fond).

    filled : la forme, trous compris (découpe de taille box) ; area : ses pixels réels.
    Si la forme n'est que le HALO d'ombre d'un élément intérieur, seul l'intérieur est gardé.
    """
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]
    origin = (x + offset[0], y + offset[1])
    inner_bg = _surface_color(crop, filled, outer_bg, ctx.cfg)
    bx, by, bw, bh = _body_box(crop, filled, inner_bg, outer_bg, ctx.cfg)
    own = Detection((origin[0] + bx, origin[1] + by, bw, bh), "surface", depth, fill=area / (w * h))
    inside = _erode(filled, ctx.cfg.edge_margin)
    if (bw, bh) != (w, h):                 # ombre retirée : on ne cherche plus de contenu dedans
        body = np.zeros_like(inside)
        body[by:by + bh, bx:bx + bw] = True
        inside &= _erode(body, ctx.cfg.edge_margin)
    if depth + 1 > ctx.cfg.max_depth or not inside.any():
        return [own]
    if palette_ratio(center_crop(crop, ctx.cfg.photo_center)) < ctx.cfg.min_palette_ratio:
        # Photo / illustration : on n'y cherche que ce qui porte du texte (pastille "15" sur un avatar).
        return [own] + _photo_overlays(crop, inside, inner_bg, origin, depth, ctx, own.bbox)
    inner = _detect_region(crop, inside, inner_bg, origin, depth + 1, ctx)
    return inner if _is_halo(own.bbox, inner, ctx.cfg) else [own] + inner


def _surface_color(crop: np.ndarray, filled: np.ndarray, outer_bg: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Couleur de la surface = la plus fréquente parmi ses pixels NETTEMENT différents du fond
    extérieur. Sur un badge réel, le texte clair et l'ombre douce sont plus nombreux que le vert du
    badge : la couleur majoritaire brute était le blanc-vert, et le texte devenait invisible."""
    distinct = filled & (color_distance(crop, outer_bg) > 2 * cfg.color_tol)
    if distinct.sum() >= 0.3 * filled.sum():
        return dominant_color(crop[distinct])
    return dominant_color(crop[filled])    # surface pâle (carte blanche sur gris) : couleur majoritaire


def _body_box(crop: np.ndarray, filled: np.ndarray, color: np.ndarray, outer_bg: np.ndarray,
              cfg: DetectionConfig) -> BBox:
    """Boîte du CORPS de la forme (pixels de sa couleur), sans l'ombre douce autour : le seuil bas
    des surfaces attrape l'ombre (badge 55x31 au lieu de 48x23, bouton flottant 147x113 au lieu de
    101x101). On ne retire le bord que s'il est fin ET pâle (une ombre), jamais une bordure nette."""
    h, w = filled.shape
    full = (0, 0, w, h)
    body = filled & (color_distance(crop, color) <= cfg.body_tol)
    if not body.any():
        return full
    ys, xs = np.nonzero(body)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if max(x0, y0, w - x1, h - y1) > cfg.halo_max_margin:
        return full                        # corps bien plus petit que la forme : pas une ombre
    ring = filled.copy()
    ring[y0:y1, x0:x1] = False
    if not ring.any():
        return full
    to_outside = np.median(color_distance(crop[ring], outer_bg))
    to_body = np.median(color_distance(crop[ring], color))
    if to_outside >= 0.7 * to_body:
        return full                        # bord aussi loin du fond que du corps : une bordure, gardée
    return (x0, y0, x1 - x0, y1 - y0)      # bord proche du fond : une ombre, retirée


def _photo_overlays(crop: np.ndarray, inside: np.ndarray, bg: np.ndarray, origin: tuple[int, int],
                    depth: int, ctx: _Context, photo: BBox) -> list[Detection]:
    """Dans une photo, on ne garde que les éléments posés dessus qui portent du texte lu par l'OCR
    (et la petite forme qui les entoure) : le reste serait des morceaux de la photo."""
    if not any(overlap_ratio(ln.bbox, photo) >= 0.9 for ln in ctx.lines):
        return []
    inner = _detect_region(crop, inside, bg, origin, depth + 1, ctx)
    texts = [d for d in inner if d.source == "text"]
    small = photo[2] * photo[3] * ctx.cfg.overlay_max_area
    holders = [d for d in inner if d.source == "surface" and d.bbox[2] * d.bbox[3] <= small
               and any(overlap_ratio(t.bbox, d.bbox) >= 0.9 for t in texts)]
    return holders + texts


def _is_halo(box: BBox, inner: list[Detection], cfg: DetectionConfig) -> bool:
    """Ombre douce autour d'un élément : le seuil bas des surfaces la détecte comme une surface
    à part, qui contient l'élément à quelques pixels près (badge "MOBILE" emboîté 3 fois :
    60x36 ⊃ 55x31 ⊃ 50x26). Critères : tout le contenu est dans UNE surface intérieure, et
    celle-ci touche presque les quatre bords (marges <= halo_max_margin)."""
    surfaces = [d for d in inner if d.source == "surface"]
    if not surfaces:
        return False
    core = max(surfaces, key=lambda d: d.bbox[2] * d.bbox[3])
    if any(overlap_ratio(d.bbox, core.bbox) < 0.9 for d in inner if d is not core):
        return False
    (x, y, w, h), (cx, cy, cw, ch) = box, core.bbox
    margins = (cx - x, cy - y, x + w - cx - cw, y + h - cy - ch)
    return max(margins) <= cfg.halo_max_margin


def _filled(shape: np.ndarray) -> np.ndarray:
    """La forme avec ses trous bouchés (c'est dans les trous qu'est le contenu)."""
    contours, _ = cv2.findContours(shape.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(shape.shape, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _fill_small_holes(shape: np.ndarray, max_ratio: float) -> np.ndarray:
    """Bouche les trous dont la surface est inférieure à max_ratio de celle de la boîte."""
    mask = shape.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = mask.copy()
    limit = max_ratio * shape.shape[0] * shape.shape[1]
    for c, h in zip(contours, hierarchy[0] if hierarchy is not None else []):
        if h[3] >= 0 and cv2.contourArea(c) < limit:      # contour intérieur (trou), petit
            cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def _square(margin: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))


def _erode(mask: np.ndarray, margin: int) -> np.ndarray:
    """Rogne le bord de margin px (anti-crénelage des coins arrondis).

    borderValue=0 : hors de la découpe = vide. (Par défaut OpenCV le considère plein, et le bord
    d'une forme qui touche le cadre de sa découpe n'était jamais rogné.)"""
    if margin <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), _square(margin), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0).astype(bool)


def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
    """Élargit de margin px."""
    return cv2.dilate(mask.astype(np.uint8), _square(margin)).astype(bool) if margin > 0 else mask


def _extract_text_lines(ink: np.ndarray, offset: tuple[int, int], depth: int,
                        lines: Sequence[TextLine], min_coverage: float
                        ) -> tuple[list[Detection], np.ndarray]:
    """Chaque morceau d'encre majoritairement dans une ligne OCR rejoint cette ligne.

    Retourne les boîtes Texte et l'encre restante (sans les morceaux attribués).
    Un morceau à cheval (photo collée au nom, icône à côté d'un champ) reste dans l'encre.
    Une ligne dont les morceaux couvrent moins de min_coverage de sa largeur est ignorée ici
    (elle sera placée à un autre niveau, ou ajoutée telle quelle par _unplaced_lines).
    """
    if not lines:
        return [], ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # Boîte OCR élargie de 3 px en hauteur : les bas de lettres juste sous la ligne (jambages
    # coupés) la rejoignent au lieu de devenir des traits "Texte" de 2 px de haut.
    local = [(ln.bbox[0] - offset[0] - 2, ln.bbox[1] - offset[1] - 3, ln.bbox[2] + 4, ln.bbox[3] + 6)
             for ln in lines]
    members: dict[int, list[int]] = {}
    for comp in range(1, n):
        box = tuple(int(v) for v in stats[comp][:4])
        ratios = [overlap_ratio(box, lb) for lb in local]
        best = int(np.argmax(ratios))
        # pas beaucoup plus haut que la ligne : un anneau d'ombre autour du texte n'est pas du texte
        if ratios[best] >= 0.5 and box[3] <= 1.5 * lines[best].bbox[3]:
            members.setdefault(best, []).append(comp)
    found = []
    taken = np.zeros(n, dtype=bool)
    for li, comps in sorted(members.items()):
        x0 = min(stats[c][0] for c in comps)
        y0 = min(stats[c][1] for c in comps)
        x1 = max(stats[c][0] + stats[c][2] for c in comps)
        y1 = max(stats[c][1] + stats[c][3] for c in comps)
        if x1 - x0 < min_coverage * lines[li].bbox[2]:
            continue                       # quelques pixels parasites, pas la ligne : on n'y touche pas
        taken[comps] = True
        found.append(Detection((int(x0) + offset[0], int(y0) + offset[1], int(x1 - x0), int(y1 - y0)),
                               "text", depth, text=lines[li].text, score=lines[li].score))
    rest = (ink.astype(bool) & ~taken[labels]).astype(np.uint8)
    return found, rest


def _detect_ink(ink: np.ndarray, offset: tuple[int, int], depth: int,
                ctx: _Context) -> list[Detection]:
    """Fusionne l'encre en taches (icônes, texte non lu) et en fait des boîtes resserrées."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ctx.cfg.merge_x, ctx.cfg.merge_y))
    n, labels, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, kernel), connectivity=8)
    boxes = [b for b in (_tight_bbox(ink, labels, i) for i in range(1, n)) if b is not None]
    found = []
    for x, y, w, h in _merge_icon_fragments(boxes, ctx.cfg):
        if w * h < ctx.cfg.min_area:
            continue
        found.append(Detection((x + offset[0], y + offset[1], w, h), "ink", depth,
                               fill=float(ink[y:y + h, x:x + w].mean())))
    return found


def _merge_icon_fragments(boxes: list[BBox], cfg: DetectionConfig) -> list[BBox]:
    """Une icône dessinée en traits séparés (menu "hamburger" : 3 barres espacées de 8 px) donne
    plusieurs taches. Deux taches voisines (écart <= icon_gap) dont l'union reste une petite forme
    (côté <= icon_max_side) sont réunies, jusqu'à ce que plus rien ne change."""
    boxes = sorted(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                u = _union_box(boxes[i], boxes[j])
                if _gap(boxes[i], boxes[j]) <= cfg.icon_gap and max(u[2], u[3]) <= cfg.icon_max_side:
                    boxes[i] = u
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _gap(a: BBox, b: BBox) -> int:
    dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return max(dx, dy)


def _union_box(a: BBox, b: BBox) -> BBox:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _tight_bbox(ink: np.ndarray, labels: np.ndarray, label: int) -> BBox | None:
    """Boîte des vrais pixels d'encre d'une tache (on enlève la marge ajoutée par la dilatation)."""
    ys, xs = np.nonzero(ink & (labels == label))
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _unplaced_lines(lines: Sequence[TextLine], found: list[Detection]) -> list[Detection]:
    """Lignes OCR qu'aucun pixel détecté n'a rejointes (texte très pâle) : ajoutées telles quelles.

    Profondeur = nombre de boîtes non-texte qui contiennent le centre de la ligne.
    """
    placed = {(d.text, d.bbox) for d in found if d.text is not None}
    out = []
    for ln in lines:
        if any(t == ln.text and overlap_ratio(b, ln.bbox) > 0.5 for t, b in placed):
            continue
        cx, cy = ln.bbox[0] + ln.bbox[2] / 2, ln.bbox[1] + ln.bbox[3] / 2
        depth = sum(d.text is None and d.bbox[0] <= cx < d.bbox[0] + d.bbox[2]
                    and d.bbox[1] <= cy < d.bbox[1] + d.bbox[3] for d in found)
        out.append(Detection(ln.bbox, "text", depth, text=ln.text, score=ln.score))
    return out
