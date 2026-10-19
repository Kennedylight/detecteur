"""Arbre -> document JSON, validé (schéma JSON + cohérence que le schéma ne sait pas exprimer)."""
import json
from dataclasses import asdict
from pathlib import Path

import jsonschema

from uitree.geometry import overlap_ratio
from uitree.similarity import RepeatedComponent, VisualDuplicate
from uitree.tree import Node

SCHEMA_PATH = Path(__file__).with_name("tree.schema.json")
SCHEMA_VERSION = "1.1"


def node_to_dict(node: Node) -> dict:
    x, y, w, h = node.bbox
    return {
        "id": node.id,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "type": node.type,
        "geo_type": node.geo_type,
        "source": node.source,
        "parent": node.parent,
        "children": [node_to_dict(c) for c in node.children],
        "text": node.text,
        "style": asdict(node.style) if node.style else None,
        "label": node.label,
        "overlay": node.overlay,
        "score": node.score,
        "hashes": {"structure": node.structure_hash, "visual": node.visual_hash},
    }


def tree_document(root: Node, image_file: str, repeated: list[RepeatedComponent],
                  duplicates: list[VisualDuplicate]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "image": {"file": image_file, "width": root.bbox[2], "height": root.bbox[3]},
        "root": node_to_dict(root),
        "repeated_components": [asdict(r) for r in repeated],
        "visual_duplicates": [asdict(v) for v in duplicates],
    }


class TreeValidationError(ValueError):
    pass


def validate_document(doc: dict, contain_tolerance: float = 0.9) -> None:
    """Lève TreeValidationError au premier problème trouvé."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.Draft202012Validator(schema).validate(doc)
    except jsonschema.ValidationError as e:
        raise TreeValidationError(f"Schéma : {e.message} (à {list(e.absolute_path)})") from e
    nodes = list(_walk(doc["root"]))
    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise TreeValidationError("Ids en double")
    if doc["root"]["parent"] is not None:
        raise TreeValidationError("La racine a un parent")
    for n in nodes:
        for c in n["children"]:
            if c["parent"] != n["id"]:
                raise TreeValidationError(f"Nœud {c['id']} : parent {c['parent']} au lieu de {n['id']}")
            if overlap_ratio(_box(c), _box(n)) < contain_tolerance:
                raise TreeValidationError(f"Nœud {c['id']} déborde de son parent {n['id']}")
    known = set(ids)
    for group in doc["repeated_components"] + doc["visual_duplicates"]:
        if not set(group["node_ids"]) <= known:
            raise TreeValidationError(f"Ids inconnus dans un groupe : {group['node_ids']}")


def write_document(doc: dict, path: Path) -> None:
    validate_document(doc)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _walk(node: dict):
    yield node
    for c in node["children"]:
        yield from _walk(c)


def _box(node: dict) -> tuple[int, int, int, int]:
    b = node["bbox"]
    return (b["x"], b["y"], b["width"], b["height"])
