"""OCR et fusion texte/détection. Un faux moteur OCR rend les tests rapides et déterministes ;
un seul test utilise le vrai RapidOCR (sur la maquette)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from make_test_image import make_test_image
from uitree.analysis import analyse_image
from uitree.config import Config, OcrConfig
from uitree.detection import detect_in_image
from uitree.evaluation import Item, evaluate, text_scores
from uitree.ocr import RapidOcrEngine, TextLine, is_icon_like, read_text


class FakeOcr:
    def __init__(self, lines: list[TextLine]) -> None:
        self.lines = lines

    def __call__(self, img_bgr: np.ndarray) -> list[TextLine]:
        return self.lines


def _bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img)[:, :, ::-1].copy()


def _two_close_lines() -> tuple[Image.Image, list[TextLine]]:
    """Deux lignes à 3 px d'écart : la dilatation verticale (4 px) les fusionne sans OCR."""
    img = Image.new("RGB", (220, 60), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    d.text((10, 8), "Bonjour", fill="black", font=font)
    top = d.textbbox((10, 8), "Bonjour", font=font)
    y2 = top[3] + 3 - (d.textbbox((10, 0), "monde", font=font)[1])
    d.text((10, y2), "monde", fill="black", font=font)
    bot = d.textbbox((10, y2), "monde", font=font)
    pad = 2
    return img, [TextLine((top[0] - pad, top[1] - pad, top[2] - top[0] + 2 * pad, top[3] - top[1] + 2 * pad), "Bonjour", 0.99),
                 TextLine((bot[0] - pad, bot[1] - pad, bot[2] - bot[0] + 2 * pad, bot[3] - bot[1] + 2 * pad), "monde", 0.98)]


def test_sans_ocr_deux_lignes_proches_fusionnent():
    img, _ = _two_close_lines()
    assert len(detect_in_image(_bgr(img))) == 1


def test_avec_ocr_deux_lignes_proches_restent_separees():
    img, lines = _two_close_lines()
    found = detect_in_image(_bgr(img), lines=lines)
    assert [(d.source, d.text) for d in found] == [("text", "Bonjour"), ("text", "monde")]


def test_boite_texte_resserree_sur_les_pixels_pas_sur_la_marge_ocr():
    img, lines = _two_close_lines()
    found = detect_in_image(_bgr(img), lines=lines)
    for d, ln in zip(found, lines):
        assert d.bbox[2] < ln.bbox[2] and d.bbox[3] < ln.bbox[3]


def test_image_collee_au_texte_reste_separee():
    img = Image.new("RGB", (200, 60), "white")
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(0)
    for _ in range(400):                                    # "photo" : points colorés serrés
        x, y = rng.integers(10, 50), rng.integers(10, 50)
        d.rectangle([x, y, x + 2, y + 2], fill=tuple(int(v) for v in rng.integers(0, 200, 3)))
    d.text((56, 20), "Alice", fill="black", font=ImageFont.load_default(size=16))
    tb = d.textbbox((56, 20), "Alice", font=ImageFont.load_default(size=16))
    line = TextLine((tb[0] - 2, tb[1] - 2, tb[2] - tb[0] + 4, tb[3] - tb[1] + 4), "Alice", 0.99)
    found = analyse_image(_bgr(img), Config(), lines=[line])
    assert sorted(d.text or d.kind for d in found) == ["Alice", "Image"]


def test_ligne_sans_pixels_ajoutee_telle_quelle():
    img = Image.new("RGB", (100, 40), "white")
    line = TextLine((10, 10, 50, 12), "pâle", 0.8)
    assert detect_in_image(_bgr(img), lines=[line])[0].bbox == (10, 10, 50, 12)


def test_quelques_pixels_parasites_ne_volent_pas_la_ligne():
    """Régression (vraie capture) : une ligne rattachée à 1 seul pixel donnait une boîte 1x1."""
    img = Image.new("RGB", (100, 40), "white")
    img.putpixel((40, 15), (0, 0, 0))
    line = TextLine((10, 10, 60, 12), "Bonjour", 0.9)
    found = detect_in_image(_bgr(img), lines=[line])
    assert [d.bbox for d in found if d.text] == [(10, 10, 60, 12)]


@pytest.mark.parametrize("text,bbox,expected", [
    ("8", (0, 0, 23, 22), True),            # icône partage lue "8"
    ("日", (0, 0, 16, 19), True),
    ("22", (0, 0, 21, 17), False),          # un jour du calendrier : du vrai texte
    ("I", (0, 0, 6, 18), False),            # un seul caractère mais pas carré
    ("Home", (0, 0, 37, 19), False),
])
def test_ligne_ressemblant_a_une_icone(text, bbox, expected):
    assert is_icon_like(TextLine(bbox, text, 0.9), OcrConfig()) is expected


def test_read_text_filtre_confiance_et_icones():
    lines = [TextLine((0, 0, 40, 10), "ok", 0.9), TextLine((0, 0, 40, 10), "bof", 0.3),
             TextLine((0, 0, 20, 20), "8", 0.95), TextLine((0, 0, 40, 10), "  ", 0.9)]
    assert [ln.text for ln in read_text(np.zeros((1, 1, 3), np.uint8), FakeOcr(lines), OcrConfig())] == ["ok"]


def test_ocr_desactive_dans_la_config():
    img, lines = _two_close_lines()
    cfg = Config(ocr=OcrConfig(enabled=False))
    assert all(d.text is None for d in analyse_image(_bgr(img), cfg, ocr=FakeOcr(lines)))


@pytest.fixture(scope="module")
def maquette_avec_vrai_ocr():
    img, truth = make_test_image()
    return truth, analyse_image(_bgr(img), Config(), ocr=RapidOcrEngine())


def test_maquette_vrai_ocr_boites_et_texte(maquette_avec_vrai_ocr):
    """Mesure réelle (RapidOCR) : boîtes inchangées ; au moins 7/10 textes exacts, CER < 3 %."""
    truth, dets = maquette_avec_vrai_ocr
    t = [Item(g.bbox, g.kind, g.text) for g in truth]
    p = [Item(d.bbox, d.kind, d.text) for d in dets]
    report = evaluate(t, p, threshold=0.9)
    exact, n_text, cer = text_scores(t, p, report)
    assert report.missed == [] and report.false_positives == []
    assert n_text == 10 and exact >= 7 and cer < 0.03


def test_vrai_ocr_deterministe():
    img, _ = make_test_image()
    engine = RapidOcrEngine()
    assert engine(_bgr(img)) == engine(_bgr(img))
