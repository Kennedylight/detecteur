"""Chargement de config.toml en objets typés et immuables."""
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class DetectionConfig:
    color_tol: int = 30
    surface_tol: int = 10
    surface_open: int = 3
    card_min_side: int = 30
    card_rect_fill: float = 0.9
    card_gap_close: int = 2
    merge_x: int = 12
    merge_y: int = 4
    min_area: int = 30
    solid_fill: float = 0.75
    min_solid_side: int = 14
    edge_margin: int = 2
    min_palette_ratio: float = 0.7
    max_depth: int = 4
    min_line_coverage: float = 0.5


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool = True
    min_score: float = 0.5
    icon_max_chars: int = 1
    icon_max_aspect: float = 1.4


@dataclass(frozen=True)
class ClassifyConfig:
    contain_overlap: float = 0.9
    icon_max_side: int = 48
    icon_max_aspect: float = 2.0
    photo_min_side: int = 20
    photo_min_colors: int = 16
    photo_max_palette: float = 0.7
    button_min_height: int = 24
    button_max_height: int = 64
    button_max_chars: int = 30
    field_min_aspect: float = 3.0
    field_text_start: float = 0.25
    centered_tol: float = 0.15
    text_min_aspect: float = 2.5
    text_max_height: int = 40


@dataclass(frozen=True)
class XYCutConfig:
    min_gap: int = 4
    twin_iou: float = 0.9


@dataclass(frozen=True)
class SimilarityConfig:
    min_descendants: int = 2
    max_height_ratio: float = 1.3
    max_width_ratio: float = 2.0
    max_visual_distance: int = 8
    max_size_ratio: float = 1.2
    max_color_distance: int = 40


@dataclass(frozen=True)
class Config:
    detection: DetectionConfig = DetectionConfig()
    ocr: OcrConfig = OcrConfig()
    classify: ClassifyConfig = ClassifyConfig()
    xycut: XYCutConfig = XYCutConfig()
    similarity: SimilarityConfig = SimilarityConfig()


def _section(cls: type, values: dict) -> object:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Paramètres inconnus pour {cls.__name__} : {sorted(unknown)}")
    return cls(**values)


def load_config(path: Path | str = DEFAULT_PATH) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(detection=_section(DetectionConfig, raw.get("detection", {})),
                  ocr=_section(OcrConfig, raw.get("ocr", {})),
                  classify=_section(ClassifyConfig, raw.get("classify", {})),
                  xycut=_section(XYCutConfig, raw.get("xycut", {})),
                  similarity=_section(SimilarityConfig, raw.get("similarity", {})))
