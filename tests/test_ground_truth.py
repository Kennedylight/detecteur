"""La maquette de test et sa vérité terrain sont stables (sinon toute mesure est faussée)."""
import hashlib

from make_test_image import make_test_image
from uitree.evaluation import match_boxes
from uitree.geometry import iou

# Empreinte des pixels de la maquette d'origine (avant ajout de la vérité terrain).
ORIGINAL_PIXELS_MD5 = "f985df0692d3c1838e906013f4372531"


def test_maquette_identique_a_l_originale():
    img, _ = make_test_image()
    assert hashlib.md5(img.tobytes()).hexdigest() == ORIGINAL_PIXELS_MD5


def test_verite_terrain_complete():
    _, truth = make_test_image()
    assert len(truth) == 18
    assert {g.kind for g in truth} == {"Icône", "Texte", "Image", "Composant"}


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == 50 / 150


def test_appariement_un_a_un():
    truth = [(0, 0, 10, 10), (50, 0, 10, 10)]
    preds = [(0, 0, 10, 10), (1, 0, 10, 10), (200, 0, 5, 5)]
    report = match_boxes(truth, preds)
    assert [(m.truth_index, m.pred_index) for m in report.matches] == [(0, 0)]
    assert report.missed == [1] and report.false_positives == [1, 2]
