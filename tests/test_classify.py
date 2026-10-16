"""Types (règles) et styles. Mesures sur les deux maquettes, plus des cas isolés par règle."""
import numpy as np
import pytest

from make_test_image import make_form_image, make_test_image
from uitree.analysis import analyse_image
from uitree.classify import BUTTON, CONTAINER, FIELD, ICON, TEXT, RuleClassifier
from uitree.config import Config
from uitree.detection import Detection
from uitree.evaluation import match_boxes
from uitree.hierarchy import children_of, find_parents
from uitree.ocr import RapidOcrEngine


def _bgr(img) -> np.ndarray:
    return np.asarray(img)[:, :, ::-1].copy()


def _types_ok(make, use_ocr: bool) -> tuple[int, int, list[str]]:
    img, truth = make()
    dets = analyse_image(_bgr(img), Config(), ocr=RapidOcrEngine()) if use_ocr \
        else analyse_image(_bgr(img), Config(), lines=[])
    report = match_boxes([g.bbox for g in truth], [d.bbox for d in dets], threshold=0.9)
    wrong = [f"{truth[m.truth_index].name}: {dets[m.pred_index].kind}" for m in report.matches
             if truth[m.truth_index].kind != dets[m.pred_index].kind]
    return len(report.matches) - len(wrong), len(truth), wrong


@pytest.mark.parametrize("make,use_ocr", [
    (make_test_image, False), (make_test_image, True), (make_form_image, True),
    pytest.param(make_form_image, False, marks=pytest.mark.xfail(
        strict=True, reason="Limite connue : sans OCR, la loupe (à 8 px) fusionne avec le texte du champ")),
])
def test_maquettes_tous_les_types_corrects(make, use_ocr):
    ok, total, wrong = _types_ok(make, use_ocr)
    assert (ok, wrong) == (total, [])


# --- inclusion -----------------------------------------------------------------------------

def test_parent_est_le_plus_petit_conteneur():
    boxes = [(0, 0, 100, 100), (10, 10, 50, 50), (20, 20, 10, 10)]
    assert find_parents(boxes, [True, True, True]) == [None, 0, 1]


def test_un_texte_ne_contient_rien():
    boxes = [(0, 0, 100, 20), (10, 5, 10, 10)]
    assert find_parents(boxes, [False, True]) == [None, None]


def test_boites_identiques_sans_cycle():
    parents = find_parents([(0, 0, 10, 10), (0, 0, 10, 10)], [True, True])
    assert parents == [1, None]
    assert children_of(parents) == [[], [0]]


# --- règles une par une ----------------------------------------------------------------------

def _classify(dets: list[Detection], img: np.ndarray | None = None) -> list[str]:
    img = np.full((200, 400, 3), 255, np.uint8) if img is None else img
    parents = find_parents([d.bbox for d in dets], [d.source != "text" for d in dets])
    return [c.kind for c in RuleClassifier()(img, dets, parents)]


def test_surface_avec_texte_centre_est_un_bouton():
    dets = [Detection((10, 10, 160, 40), "surface", 0, 1.0),
            Detection((50, 22, 80, 14), "text", 1, text="Valider")]
    assert _classify(dets) == [BUTTON, TEXT]


def test_surface_allongee_avec_texte_a_gauche_est_un_champ():
    dets = [Detection((10, 10, 300, 36), "surface", 0, 1.0),
            Detection((20, 20, 90, 14), "text", 1, text="Votre nom")]
    assert _classify(dets) == [FIELD, TEXT]


def test_bouton_avec_icone_et_texte():
    dets = [Detection((10, 10, 160, 40), "surface", 0, 1.0),
            Detection((40, 20, 18, 18), "ink", 1, 0.4),
            Detection((64, 22, 70, 14), "text", 1, text="Envoyer")]
    assert _classify(dets)[0] == BUTTON


def test_surface_avec_plusieurs_textes_est_un_conteneur():
    dets = [Detection((10, 10, 300, 120), "surface", 0, 1.0),
            Detection((20, 20, 90, 14), "text", 1, text="Titre"),
            Detection((20, 50, 200, 14), "text", 1, text="Description")]
    assert _classify(dets)[0] == CONTAINER


def test_texte_long_n_est_pas_un_bouton():
    dets = [Detection((10, 10, 380, 40), "surface", 0, 1.0),
            Detection((30, 22, 300, 14), "text", 1, text="x" * 60)]
    assert _classify(dets)[0] == CONTAINER


def test_petite_forme_sans_contenu_est_une_icone():
    assert _classify([Detection((10, 10, 24, 24), "ink", 0, 0.3)]) == [ICON]


# --- styles ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def form_styles():
    img, _ = make_form_image()
    return {d.text: d.style for d in analyse_image(_bgr(img), Config(), ocr=RapidOcrEngine()) if d.text}


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)])


@pytest.mark.parametrize("text,color,background", [
    ("Se connecter", (255, 255, 255), (37, 99, 235)),
    ("Annuler", (37, 99, 235), (255, 255, 255)),
    ("Nouveau", (185, 28, 28), (254, 226, 226)),
    ("nom@exemple.com", (150, 150, 150), (255, 255, 255)),
])
def test_couleurs_mesurees(form_styles, text, color, background):
    style = form_styles[text]
    assert np.abs(_rgb(style.text_color) - color).sum() <= 30
    assert np.abs(_rgb(style.background) - background).sum() <= 15


def test_hauteur_de_texte_proportionnelle_a_la_police():
    """Mesuré : hauteur d'encre ≈ 0,7 x taille de police (±2 px selon jambages/hampes)."""
    img, _ = make_test_image()
    dets = analyse_image(_bgr(img), Config(), ocr=RapidOcrEngine())
    heights = {d.text: d.style.text_height_px for d in dets if d.text}
    assert all(10 <= heights[n] <= 14 for n in ("Contacts", "Peter Limonez", "Sofia Morell"))   # police 16
    assert all(7 <= heights[n] <= 11 for n in ("MOBILE",))                                      # police 11
