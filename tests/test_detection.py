"""Détecteur : cas de base, cas de fusion d'éléments proches, conteneurs de couleur."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from make_test_image import make_test_image
from uitree.config import Config, DetectionConfig, load_config
from uitree.analysis import analyse_image, load_bgr
from uitree.detection import detect_in_image
from uitree.evaluation import match_boxes


def _save(img: Image.Image, tmp_path: Path) -> str:
    path = tmp_path / "img.png"
    img.save(path)
    return str(path)


def _detect(img: Image.Image) -> list:
    """PIL (RVB) -> OpenCV (BVR) puis détection et typage, sans OCR ni disque."""
    return analyse_image(np.asarray(img)[:, :, ::-1].copy(), Config(), lines=[])


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def test_deux_elements_bien_separes_donnent_deux_boites():
    img = Image.new("RGB", (200, 60), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 50, 40], fill="black")
    d.rectangle([120, 10, 160, 40], fill="black")
    assert sorted(b.bbox for b in _detect(img)) == [(10, 10, 41, 31), (120, 10, 41, 31)]


def test_lettres_d_un_mot_fusionnent_en_une_boite():
    img = Image.new("RGB", (200, 40), "white")
    ImageDraw.Draw(img).text((10, 10), "Bonjour", fill="black", font=_font(16))
    assert len(_detect(img)) == 1


def test_image_illisible_leve_une_erreur(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bgr(str(tmp_path / "absente.png"))


def test_lecture_depuis_le_disque_identique_a_la_memoire(tmp_path):
    img = Image.new("RGB", (100, 40), "white")
    ImageDraw.Draw(img).rectangle([10, 10, 40, 30], fill="black")
    assert analyse_image(load_bgr(_save(img, tmp_path)), Config(), lines=[]) == _detect(img)


@pytest.mark.parametrize("gap", [2, 4, 6, 10])
def test_texte_proche_d_un_badge_reste_separe(gap):
    img = Image.new("RGB", (220, 40), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 14), "Bonjour le monde", fill=(130, 130, 130), font=_font(11))
    right = int(d.textbbox((10, 14), "Bonjour le monde", font=_font(11))[2])
    d.rounded_rectangle([right + gap, 10, right + gap + 50, 28], radius=4, fill=(110, 200, 100))
    assert sorted(b.kind for b in _detect(img)) == ["Composant", "Texte"]


def test_texte_dans_un_badge_est_detecte():
    img = Image.new("RGB", (120, 40), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, 70, 30], radius=4, fill=(110, 200, 100))
    d.text((18, 14), "MOBILE", fill="white", font=_font(11))
    found = _detect(img)
    assert [(b.kind, b.depth) for b in found] == [("Composant", 0), ("Texte", 1)]


def test_texte_sombre_dans_un_badge_est_detecte():
    """Texte d'une 3e couleur : ses pixels touchent le badge dans le masque de la page."""
    img = Image.new("RGB", (120, 40), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, 70, 30], radius=4, fill=(250, 220, 90))
    d.text((18, 14), "NEW", fill="black", font=_font(11))
    assert [b.depth for b in _detect(img)] == [0, 1]


def test_fond_de_couleur_ne_cache_pas_son_contenu():
    img = Image.new("RGB", (300, 120), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 299, 50], fill=(40, 140, 80))         # barre d'en-tête verte
    d.text((20, 18), "Titre", fill="white", font=_font(16))
    d.rectangle([250, 15, 270, 35], fill="white")             # icône dans la barre
    found = _detect(img)
    assert len(found) == 3
    assert found[0].bbox == (0, 0, 300, 51) and {b.depth for b in found[1:]} == {1}


def test_texte_dans_un_cadre_fin_n_est_pas_supprime():
    """Régression (vraie capture) : un cadre fin (encre) ne doit pas effacer le texte qu'il entoure."""
    img = Image.new("RGB", (200, 80), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, 194, 74], outline="black", width=1)
    d.text((40, 30), "Contenu", fill="black", font=_font(16))
    assert len(_detect(img)) == 2


def test_cartes_blanches_plus_etendues_que_le_fond_gris():
    """Vraies captures : le blanc des cartes est la couleur DOMINANTE, le gris du fond ressort."""
    img = Image.new("RGB", (300, 200), (238, 240, 244))
    d = ImageDraw.Draw(img)
    for x0 in (8, 154):
        d.rounded_rectangle([x0, 8, x0 + 137, 191], radius=8, fill="white")
        d.text((x0 + 15, 30), "Carte", fill="black", font=_font(16))
    found = _detect(img)
    cards = [f for f in found if f.bbox[2] > 100]
    assert len(cards) == 2 and all(abs(c.bbox[2] - 138) <= 2 and abs(c.bbox[3] - 184) <= 2 for c in cards)
    assert sorted(f.depth for f in found if f.bbox[2] < 100) == [1, 1]


def test_carte_delimitee_par_une_bordure_fine():
    img = Image.new("RGB", (300, 200), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 20, 279, 179], radius=10, outline=(220, 222, 226), width=1)
    d.text((40, 40), "Titre", fill="black", font=_font(16))
    found = _detect(img)
    assert found[0].bbox == (20, 20, 260, 160) and [f.depth for f in found] == [0, 1]


def test_photo_on_ne_cherche_pas_dedans():
    """Une forme pleine très colorée (photo) reste une seule boîte."""
    rng = np.random.default_rng(0)
    canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
    canvas[20:80, 20:80] = rng.integers(0, 200, (60, 60, 3), dtype=np.uint8)
    found = detect_in_image(canvas)
    assert [b.bbox for b in found] == [(20, 20, 60, 60)]


def test_resultat_deterministe():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8))
    assert _detect(img) == _detect(img)


def test_profondeur_maximale_respectee():
    img = Image.new("RGB", (120, 40), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, 70, 30], radius=4, fill=(110, 200, 100))
    d.text((18, 14), "MOBILE", fill="white", font=_font(11))
    found = detect_in_image(np.asarray(img)[:, :, ::-1].copy(), DetectionConfig(max_depth=0))
    assert len(found) == 1


def test_config_rejette_un_parametre_inconnu(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[detection]\ncolor_toll = 3\n")
    with pytest.raises(ValueError, match="color_toll"):
        load_config(path)


def test_maquette_toutes_les_boites_correctes():
    """Critère de réussite (boîtes) : 18/18 appariées à IoU >= 0.9, aucun faux positif."""
    img, truth = make_test_image()
    found = detect_in_image(np.asarray(img)[:, :, ::-1].copy(), load_config().detection)
    report = match_boxes([g.bbox for g in truth], [f.bbox for f in found], threshold=0.9)
    assert report.missed == [] and report.false_positives == []
