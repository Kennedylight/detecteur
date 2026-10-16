"""Vues de l'arbre, calculées à partir du document JSON (donc aussi d'un JSON déjà enregistré) :

- liste dépliable (HTML <details>) ;
- schéma nœuds-liens (image), limité en profondeur et à partir d'un nœud choisi ;
- squelette de mise en page : les Row / Column dessinés sur une page blanche.
"""
import html

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from uitree.render import TYPE_COLORS

GEO_COLORS = {"Row": (37, 99, 235), "Column": (22, 163, 74), "Groupe": (220, 38, 38), "Feuille": (120, 120, 120)}
NEUTRAL = (110, 110, 110)


def _color(node: dict) -> tuple[int, int, int]:
    return TYPE_COLORS.get(node["type"], NEUTRAL) if node["type"] else GEO_COLORS.get(node["geo_type"], NEUTRAL)


def _title(node: dict) -> str:
    kind = node["type"] or ("Page" if node["source"] == "page" else "Groupe")
    return f"#{node['id']} {kind} · {node['geo_type']}"


def _snippet(node: dict, n: int = 28) -> str:
    text = node["text"] or ""
    return text if len(text) <= n else text[:n - 1] + "…"


def _font(size: int = 11) -> ImageFont.ImageFont:
    """DejaVu Sans si elle est installée (accents : « Icône »), sinon la police intégrée à Pillow
    (qui n'a pas le « ô »)."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def find_node(root: dict, node_id: int) -> dict | None:
    if root["id"] == node_id:
        return root
    for c in root["children"]:
        found = find_node(c, node_id)
        if found:
            return found
    return None


# --- liste dépliable ----------------------------------------------------------------------------

def tree_html(root: dict, open_depth: int = 2) -> str:
    return f'<div style="font-family:monospace;font-size:13px;line-height:1.5">{_node_html(root, 0, open_depth)}</div>'


def _node_html(node: dict, depth: int, open_depth: int) -> str:
    r, g, b = _color(node)
    b_ = node["bbox"]
    label = (f'<span style="color:rgb({r},{g},{b});font-weight:600">{html.escape(_title(node))}</span>'
             f' <span style="color:#888">({b_["x"]},{b_["y"]}) {b_["width"]}×{b_["height"]}</span>')
    if node["text"]:
        label += f' « {html.escape(_snippet(node, 60))} »'
    if not node["children"]:
        return f'<div style="margin-left:18px">{label}</div>'
    opened = " open" if depth < open_depth else ""
    kids = "".join(_node_html(c, depth + 1, open_depth) for c in node["children"])
    return (f'<details{opened} style="margin-left:{0 if depth == 0 else 6}px">'
            f'<summary>{label} <span style="color:#888">[{len(node["children"])}]</span></summary>'
            f'<div style="border-left:1px dashed #ccc;margin-left:6px">{kids}</div></details>')


# --- schéma nœuds-liens ---------------------------------------------------------------------------

BOX_W, BOX_H, GAP_X, GAP_Y, MARGIN = 150, 40, 14, 46, 20


def tree_diagram(root: dict, max_depth: int = 3) -> Image.Image:
    """Disposition simple : les feuilles affichées sont alignées de gauche à droite, chaque parent
    est centré au-dessus de ses enfants. Sous max_depth, un nœud indique le nombre caché."""
    positions: dict[int, tuple[float, int]] = {}
    next_x = [0]

    def place(node: dict, depth: int) -> float:
        shown = node["children"] if depth < max_depth else []
        if shown:
            xs = [place(c, depth + 1) for c in shown]
            x = (xs[0] + xs[-1]) / 2
        else:
            x = next_x[0]
            next_x[0] += 1
        positions[node["id"]] = (x, depth)
        return x

    place(root, 0)
    depth_max = max(d for _, d in positions.values())
    width = int(MARGIN * 2 + next_x[0] * (BOX_W + GAP_X))
    height = MARGIN * 2 + (depth_max + 1) * (BOX_H + GAP_Y)
    img = Image.new("RGB", (max(width, 200), height), "white")
    draw = ImageDraw.Draw(img)
    font = _font()
    _draw_diagram(draw, font, root, positions, max_depth, 0)
    return img


def _box_xy(pos: tuple[float, int]) -> tuple[int, int]:
    return int(MARGIN + pos[0] * (BOX_W + GAP_X)), MARGIN + pos[1] * (BOX_H + GAP_Y)


def _draw_diagram(draw: ImageDraw.ImageDraw, font, node: dict, positions: dict, max_depth: int, depth: int) -> None:
    x, y = _box_xy(positions[node["id"]])
    shown = node["children"] if depth < max_depth else []
    for c in shown:
        cx, cy = _box_xy(positions[c["id"]])
        draw.line([(x + BOX_W // 2, y + BOX_H), (cx + BOX_W // 2, cy)], fill=(180, 180, 180), width=1)
        _draw_diagram(draw, font, c, positions, max_depth, depth + 1)
    color = _color(node)
    draw.rounded_rectangle([x, y, x + BOX_W, y + BOX_H], radius=6, outline=color, width=2, fill="white")
    draw.text((x + 6, y + 5), _title(node), fill=color, font=font)
    hidden = sum(1 for _ in _walk(node)) - 1 if node["children"] and not shown else 0
    second = f"+{hidden} nœuds cachés" if hidden else _snippet(node, 24)
    draw.text((x + 6, y + 22), second, fill=(60, 60, 60), font=font)


def _walk(node: dict):
    yield node
    for c in node["children"]:
        yield from _walk(c)


# --- squelette de mise en page ----------------------------------------------------------------------

def wireframe(root: dict) -> Image.Image:
    """Chaque nœud dessiné à sa place sur une page blanche : feuilles en aplat de la couleur de
    leur type, Row en bleu, Column en vert, Groupe (chevauchement) en rouge."""
    w, h = root["bbox"]["width"], root["bbox"]["height"]
    img = Image.new("RGB", (w, h), "white")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw, fill = ImageDraw.Draw(img), ImageDraw.Draw(overlay)
    for node in _walk(root):
        b = node["bbox"]
        box = [b["x"], b["y"], b["x"] + b["width"] - 1, b["y"] + b["height"] - 1]
        if node["children"]:
            draw.rectangle(box, outline=GEO_COLORS.get(node["geo_type"], NEUTRAL), width=1)
        else:
            fill.rectangle(box, fill=(*_color(node), 70))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def as_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img)
