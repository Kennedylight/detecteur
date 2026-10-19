"""Mesure d'un détecteur contre une vérité terrain : appariement un-à-un par IoU, puis types et texte.

Usage :  python -m uitree.evaluation maquette_test.png maquette_test_gt.json
"""
import sys
from dataclasses import dataclass

from uitree.editdistance import levenshtein
from uitree.geometry import BBox, iou


@dataclass(frozen=True)
class Match:
    truth_index: int
    pred_index: int
    iou: float


@dataclass(frozen=True)
class Report:
    matches: list[Match]
    missed: list[int]             # indices de vérité terrain sans prédiction
    false_positives: list[int]    # indices de prédictions sans vérité terrain

    def recall(self, n_truth: int) -> float:
        return len(self.matches) / n_truth if n_truth else 1.0

    def precision(self, n_pred: int) -> float:
        return len(self.matches) / n_pred if n_pred else 1.0


@dataclass(frozen=True)
class Item:
    """Une boîte à comparer (vérité terrain ou prédiction)."""
    bbox: BBox
    kind: str
    text: str | None = None
    name: str = ""


def match_boxes(truth: list[BBox], preds: list[BBox], threshold: float = 0.5) -> Report:
    """Appariement glouton : les paires de plus fort IoU d'abord, chaque boîte une seule fois."""
    pairs = sorted(((iou(t, p), ti, pi) for ti, t in enumerate(truth) for pi, p in enumerate(preds)),
                   reverse=True)
    used_t: set[int] = set()
    used_p: set[int] = set()
    matches = []
    for score, ti, pi in pairs:
        if score < threshold:
            break
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matches.append(Match(ti, pi, score))
    missed = [i for i in range(len(truth)) if i not in used_t]
    false_pos = [i for i in range(len(preds)) if i not in used_p]
    return Report(sorted(matches, key=lambda m: m.truth_index), missed, false_pos)


def text_scores(truth: list[Item], preds: list[Item], report: Report) -> tuple[int, int, float]:
    """(textes exacts, textes attendus parmi les appariés, CER = erreurs de caractères / caractères)."""
    pairs = [(truth[m.truth_index].text, preds[m.pred_index].text or "")
             for m in report.matches if truth[m.truth_index].text is not None]
    exact = sum(t == p for t, p in pairs)
    chars = sum(len(t) for t, _ in pairs)
    errors = sum(levenshtein(t, p) for t, p in pairs)
    return exact, len(pairs), errors / chars if chars else 0.0


def format_report(truth: list[Item], preds: list[Item], report: Report) -> str:
    n_m = len(report.matches)
    good_type = sum(truth[m.truth_index].kind == preds[m.pred_index].kind for m in report.matches)
    exact, n_text, cer = text_scores(truth, preds, report)
    lines = [
        f"Vérité terrain : {len(truth)}   Prédictions : {len(preds)}",
        f"Rappel    : {n_m}/{len(truth)} = {report.recall(len(truth)):.2f}",
        f"Précision : {n_m}/{len(preds)} = {report.precision(len(preds)):.2f}",
        f"Type exact (parmi les appariés) : {good_type}/{n_m}",
        f"Texte exact (parmi les appariés) : {exact}/{n_text}   CER : {cer:.3f}",
    ]
    if n_m:
        lines.append(f"IoU moyen / minimum des appariés : "
                     f"{sum(m.iou for m in report.matches) / n_m:.3f} / {min(m.iou for m in report.matches):.3f}")
    lines += ["", "Détail des appariés :"]
    for m in report.matches:
        t, p = truth[m.truth_index], preds[m.pred_index]
        flag = "ok " if t.kind == p.kind else "TYPE"
        txt = "" if t.text is None else ("  texte ok" if t.text == p.text else f"  texte lu={p.text!r}")
        lines.append(f"  [{flag}] {t.name:<14} IoU={m.iou:.3f}  attendu={t.kind:<9} prédit={p.kind}{txt}")
    lines.append("Ratés (vérité terrain non retrouvée) :")
    for ti in report.missed:
        best = max(range(len(preds)), key=lambda pi: iou(truth[ti].bbox, preds[pi].bbox), default=None)
        hint = (f"  -> meilleure prédiction {preds[best].bbox} IoU={iou(truth[ti].bbox, preds[best].bbox):.2f}"
                if best is not None else "")
        lines.append(f"  {truth[ti].name:<14} {truth[ti].bbox} {truth[ti].kind}{hint}")
    lines.append("Faux positifs (prédiction sans vérité terrain) :")
    lines += [f"  {preds[pi].bbox} {preds[pi].kind} {preds[pi].text or ''}" for pi in report.false_positives]
    return "\n".join(lines)


def evaluate(truth: list[Item], preds: list[Item], threshold: float = 0.5) -> Report:
    return match_boxes([t.bbox for t in truth], [p.bbox for p in preds], threshold)


if __name__ == "__main__":
    from uitree.groundtruth import load_ground_truth
    from uitree.analysis import analyse_image, load_bgr
    from uitree.config import load_config

    image_path, gt_path = sys.argv[1], sys.argv[2]
    truth = [Item(g.bbox, g.kind, g.text, g.name) for g in load_ground_truth(gt_path)]
    preds = [Item(d.bbox, d.kind, d.text) for d in analyse_image(load_bgr(image_path), load_config())]
    print(format_report(truth, preds, evaluate(truth, preds)))
