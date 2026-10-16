"""Traite des images (fichiers ou dossiers) et écrit, pour chacune, l'arbre JSON validé et l'image
annotée.

Usage :
    .venv/bin/python batch.py images/                  # tout un dossier -> out/json/
    .venv/bin/python batch.py a.png b.jpg --out res/   # quelques fichiers
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from uitree.analysis import analyse_to_tree, load_bgr
from uitree.config import DEFAULT_PATH, load_config
from uitree.export import write_document
from uitree.ocr import RapidOcrEngine
from uitree.render import annotate

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        else:
            found.append(p)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maquettes -> arbres JSON")
    parser.add_argument("paths", nargs="+", type=Path, help="images ou dossiers d'images")
    parser.add_argument("--out", type=Path, default=Path("out/json"), help="dossier de sortie")
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ocr = RapidOcrEngine()                       # chargé une fois pour toutes les images
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for path in list_images(args.paths):
        start = time.perf_counter()
        try:
            img = load_bgr(str(path))
            result = analyse_to_tree(img, path.name, cfg, ocr=ocr)
            write_document(result.document, args.out / f"{path.stem}.json")
            cv2.imwrite(str(args.out / f"{path.stem}_annotee.png"), annotate(img, result.detections))
        except Exception as e:                   # une image ratée ne doit pas arrêter le lot
            failures += 1
            print(f"ÉCHEC  {path.name} : {e}", file=sys.stderr)
            continue
        doc = result.document
        print(f"ok     {path.name} : {len(result.detections)} éléments, "
              f"{len(doc['repeated_components'])} composants répétés, "
              f"{len(doc['visual_duplicates'])} groupes de doublons visuels "
              f"({time.perf_counter() - start:.1f} s)")
    print(f"Sorties dans {args.out}/" + (f" — {failures} échec(s)" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
