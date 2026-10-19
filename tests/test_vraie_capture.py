"""Corrections issues de la vraie capture "Contacts" (resultat/2.jpg) : ombres douces, bouton
flottant, avatars, pastilles, texte gras, répétitions approchées. Chaque cas est reproduit sur une
image synthétique ; le dernier test mesure la vraie capture si elle est présente."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from uitree.analysis import analyse_image, analyse_to_tree, load_bgr
from uitree.config import Config, load_config
from uitree.detection import TextLine, detect_in_image
from uitree.evaluation import Item, evaluate
from uitree.groundtruth import load_ground_truth


def _bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img)[:, :, ::-1].copy()


def _glow_badge(canvas: Image.Image, box: tuple[int, int, int, int], color=(110, 200, 100)) -> None:
    """Badge avec une ombre douce colorée (flou gaussien), comme sur la vraie capture."""
    glow = Image.new("RGB", canvas.size, "white")
    x0, y0, x1, y1 = box
    ImageDraw.Draw(glow).rounded_rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], radius=8, fill=(190, 235, 190))
    glow = glow.filter(ImageFilter.GaussianBlur(4))
    canvas.paste(Image.composite(glow, canvas, Image.new("L", canvas.size, 255)))
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=5, fill=color)


def test_ombre_douce_retiree_autour_d_un_badge():
    """v précédente : badge emboîté 3 fois (60x36 ⊃ 55x31 ⊃ 50x26) à cause de son ombre."""
    img = Image.new("RGB", (160, 80), "white")
    _glow_badge(img, (50, 28, 98, 50))
    surfaces = [d for d in detect_in_image(_bgr(img)) if d.source == "surface"]
    assert len(surfaces) == 1
    x, y, w, h = surfaces[0].bbox
    assert abs(x - 50) <= 2 and abs(y - 28) <= 2 and abs(w - 49) <= 3 and abs(h - 23) <= 3


def test_texte_gras_avec_halo_pas_detecte_deux_fois():
    img = Image.new("RGB", (300, 60), (110, 200, 100))
    font = ImageFont.load_default(size=26)
    halo = Image.new("L", img.size, 0)
    ImageDraw.Draw(halo).text((20, 15), "Contacts", fill=255, font=font, stroke_width=3)
    img.paste((150, 220, 140), mask=halo.filter(ImageFilter.GaussianBlur(2)))
    ImageDraw.Draw(img).text((20, 15), "Contacts", fill="white", font=font)
    x0, y0, x1, y1 = ImageDraw.Draw(img).textbbox((20, 15), "Contacts", font=font)
    line = TextLine((x0 - 3, y0 - 4, x1 - x0 + 6, y1 - y0 + 8), "Contacts", 0.99)
    dets = detect_in_image(_bgr(img), lines=[line])
    assert [d.text for d in dets if d.text] == ["Contacts"]


def _fab_scene() -> tuple[Image.Image, list[TextLine]]:
    """Deux lignes de texte ; un bouton rond flottant posé PAR-DESSUS la seconde, qui reste
    visible des deux côtés (comme sur la capture, où il couvrait la fin du dernier message)."""
    img = Image.new("RGB", (420, 200), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    text = "Un message assez long qui passe sous le bouton rond"
    lines = []
    for y in (30, 120):
        d.text((20, y), text, fill="black", font=font)
        x0, y0, x1, y1 = d.textbbox((20, y), text, font=font)
        lines.append(TextLine((x0 - 2, y0 - 3, x1 - x0 + 4, y1 - y0 + 6), text, 0.99))
    d.ellipse([200, 95, 270, 165], fill=(37, 99, 235))
    d.rectangle([228, 120, 242, 140], fill="white")                # icône dans le bouton
    return img, lines


def test_bouton_flottant_separe_type_et_hors_du_flux():
    img, lines = _fab_scene()
    result = analyse_to_tree(_bgr(img), "fab.png", Config(), lines=lines)
    floating = [n for n in result.tree.walk() if n.overlay]
    assert [(n.type, [c.type for c in n.children]) for n in floating] == [("Bouton", ["Icône"])]
    assert not any(n.geo_type == "Groupe" for n in result.tree.walk())


def test_avatar_avec_pastille_est_une_image():
    rng = np.random.default_rng(0)
    canvas = np.full((120, 120, 3), 255, np.uint8)
    yy, xx = np.mgrid[:120, :120]
    disc = (yy - 60) ** 2 + (xx - 60) ** 2 < 45 ** 2
    canvas[disc] = rng.integers(0, 230, (int(disc.sum()), 3))     # photo ronde
    img = Image.fromarray(canvas[:, :, ::-1])
    d = ImageDraw.Draw(img)
    d.ellipse([78, 14, 104, 40], fill="white", outline=(110, 200, 100), width=2)
    d.text((84, 20), "15", fill=(110, 200, 100), font=ImageFont.load_default(size=12))
    line = TextLine((82, 18, 18, 16), "15", 0.99)
    dets = analyse_image(_bgr(img), Config(), lines=[line])
    photo = max(dets, key=lambda d: d.bbox[2] * d.bbox[3])
    assert photo.kind == "Image"
    assert any(d.text == "15" for d in dets)


def test_lignes_avec_element_optionnel_sont_le_meme_composant():
    """Lignes de contact : avatar, nom + message, badge ; un émoji sur certaines seulement."""
    rng = np.random.default_rng(1)
    img = Image.new("RGB", (420, 330), "white")
    d = ImageDraw.Draw(img)
    big, small = ImageFont.load_default(size=16), ImageFont.load_default(size=11)
    for i, y in enumerate((10, 120, 230)):
        photo = Image.fromarray(rng.integers(0, 230, (70, 70, 3), dtype=np.uint8))
        img.paste(photo, (10, y))
        d.text((100, y + 5), f"Personne {i + 1}", fill="black", font=big)
        d.text((100, y + 40), "Un message assez long", fill=(130, 130, 130), font=small)
        if i != 1:
            d.ellipse([270, y + 38, 290, y + 58], fill=(250, 200, 40))     # émoji optionnel
        d.rounded_rectangle([340, y + 38, 390, y + 58], radius=5, fill=(110, 200, 100))
    doc = analyse_to_tree(_bgr(img), "rows.png", Config(), lines=[]).document
    groups = [g for g in doc["repeated_components"] if len(g["node_ids"]) == 3]
    assert len(groups) == 1 and groups[0]["exact"] is False


CAPTURE = Path("resultat/2.jpg")
CAPTURE_GT = Path("resultat/2_gt.json")


@pytest.mark.skipif(not CAPTURE_GT.exists(), reason="capture réelle non présente dans ce dépôt")
def test_vraie_capture_contacts():
    """Mesuré sur la capture : avant corrections rappel 0,73, précision 0,57, types 31/48."""
    img = load_bgr(str(CAPTURE))
    result = analyse_to_tree(img, CAPTURE.name, load_config())
    truth = [Item(g.bbox, g.kind, g.text, g.name) for g in load_ground_truth(CAPTURE_GT)]
    preds = [Item(d.bbox, d.kind, d.text) for d in result.detections]
    report = evaluate(truth, preds)
    assert report.recall(len(truth)) >= 0.75 and report.precision(len(preds)) >= 0.9
    types_ok = sum(truth[m.truth_index].kind == preds[m.pred_index].kind for m in report.matches)
    assert types_ok / len(report.matches) >= 0.9
    rows = [g for g in result.document["repeated_components"] if len(g["node_ids"]) == 7]
    assert len(rows) == 1                                    # les 7 lignes de contact
