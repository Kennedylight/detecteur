"""Vérité terrain : format, lecture / écriture, et conversion vers / depuis le tableau de l'interface."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from uitree.classify import KINDS
from uitree.detection import Detection
from uitree.geometry import BBox

TABLE_HEADERS = ["nom", "type", "x", "y", "largeur", "hauteur", "texte"]


@dataclass(frozen=True)
class GroundTruth:
    name: str
    bbox: BBox                           # (x, y, largeur, hauteur)
    kind: str                            # type grossier attendu
    text: str | None = None              # contenu attendu (pour l'OCR)


def load_ground_truth(path: str | Path) -> list[GroundTruth]:
    with open(path, encoding="utf-8") as f:
        return [GroundTruth(g["name"], tuple(g["bbox"]), g["kind"], g.get("text")) for g in json.load(f)]


def save_ground_truth(items: list[GroundTruth], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(g) for g in items], ensure_ascii=False, indent=2), encoding="utf-8")


def rows_from_detections(dets: list[Detection]) -> list[list]:
    """Point de départ de l'annotation : les détections actuelles, à corriger à la main."""
    return [[f"{d.kind or 'élément'}_{i + 1}", d.kind or "", *d.bbox, d.text or ""] for i, d in enumerate(dets)]


class RowError(ValueError):
    pass


def ground_truth_from_rows(rows: list[list]) -> list[GroundTruth]:
    """Lignes du tableau -> vérité terrain. Une ligne au type vide est ignorée (= supprimée)."""
    items = []
    for n, row in enumerate(rows, 1):
        name, kind, x, y, w, h, text = (list(row) + [""] * 7)[:7]
        kind = str(kind or "").strip()
        if not kind:
            continue
        if kind not in KINDS:
            raise RowError(f"Ligne {n} : type « {kind} » inconnu (attendu : {', '.join(KINDS)})")
        try:
            bbox = tuple(int(float(v)) for v in (x, y, w, h))
        except (TypeError, ValueError) as e:
            raise RowError(f"Ligne {n} : coordonnées non numériques") from e
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise RowError(f"Ligne {n} : largeur et hauteur doivent être > 0")
        items.append(GroundTruth(str(name or f"{kind}_{n}"), bbox, kind, str(text) if text else None))
    return items


def bbox_from_corners(a: tuple[int, int], b: tuple[int, int]) -> BBox:
    """Deux clics (coins opposés, dans n'importe quel ordre) -> boîte."""
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    return (x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1))
