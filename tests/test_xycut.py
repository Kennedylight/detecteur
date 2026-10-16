"""XY-Cut (géométrie pure) : mêmes comportements que la version 1, sur la nouvelle API."""
from uitree.xycut import COLUMN, GROUP, LEAF, ROW, Cut, xy_cut


def _leaves(cut: Cut) -> list[int]:
    return [cut.index] if cut.geo_type == LEAF else [i for c in cut.children for i in _leaves(c)]


def test_une_seule_boite_donne_une_feuille():
    assert xy_cut([(5, 5, 10, 10)]) == Cut(LEAF, 0)


def test_boites_empilees_donnent_une_column_ordonnee_haut_bas():
    cut = xy_cut([(0, 50, 10, 10), (0, 0, 10, 10)], min_gap=8)
    assert cut.geo_type == COLUMN and _leaves(cut) == [1, 0]


def test_boites_cote_a_cote_donnent_une_row_ordonnee_gauche_droite():
    cut = xy_cut([(50, 0, 10, 10), (0, 0, 10, 10)], min_gap=8)
    assert cut.geo_type == ROW and _leaves(cut) == [1, 0]


def test_ecart_inferieur_a_min_gap_coupe_en_dernier_recours():
    """v1 : un écart de 7 < 8 donnait un "Groupe". Maintenant : coupe, mais seulement si aucun
    couloir >= min_gap n'existe dans l'autre sens."""
    assert xy_cut([(0, 0, 10, 10), (0, 17, 10, 10)], min_gap=8).geo_type == COLUMN   # écart 7


def test_couloir_large_prioritaire_sur_couloir_etroit():
    # écart vertical de 2 px (Column possible en dernier recours), écart horizontal de 20 px
    boxes = [(0, 0, 10, 10), (30, 12, 10, 10)]
    assert xy_cut(boxes, min_gap=8).geo_type == ROW


def test_contact_sans_couloir_reste_un_groupe():
    assert xy_cut([(0, 0, 10, 10), (10, 10, 10, 10)], min_gap=8).geo_type == GROUP   # coins qui se touchent


def test_ecart_egal_a_min_gap_coupe():
    assert xy_cut([(0, 0, 10, 10), (0, 18, 10, 10)], min_gap=8).geo_type == COLUMN  # écart 8


def test_chevauchement_donne_un_groupe_en_ordre_de_lecture():
    cut = xy_cut([(5, 5, 20, 20), (0, 0, 20, 20)])
    assert cut.geo_type == GROUP and _leaves(cut) == [1, 0]


def test_grille_priorite_aux_bandes_empilees():
    cut = xy_cut([(0, 0, 10, 10), (50, 0, 10, 10), (0, 50, 10, 10), (50, 50, 10, 10)], min_gap=8)
    assert cut.geo_type == COLUMN
    assert [c.geo_type for c in cut.children] == [ROW, ROW]
    assert _leaves(cut) == [0, 1, 2, 3]


def test_chaque_boite_apparait_une_seule_fois():
    boxes = [(10, 10, 30, 20), (150, 12, 100, 18), (10, 60, 50, 50), (75, 62, 120, 16),
             (75, 88, 220, 12), (340, 85, 44, 16)]
    assert sorted(_leaves(xy_cut(boxes))) == list(range(len(boxes)))


def test_deterministe():
    boxes = [(0, 0, 10, 10), (0, 0, 10, 10), (30, 0, 5, 5)]
    assert xy_cut(boxes) == xy_cut(boxes)
