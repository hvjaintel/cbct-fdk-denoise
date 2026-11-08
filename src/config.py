"""Runtime configuration for the CBCT-FDK-Denoise sample."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "denoise_stub"
REDCNN_DIR = ROOT / "models" / "red_cnn"
OUTPUT_DIR = ROOT / "output"
WALNUT_DATA_DIR = ROOT / "data" / "walnut"


@dataclass
class GeometryConfig:
    """Simplified circular cone-beam geometry (SOD / SDD in arbitrary units)."""

    volume_size: int = 64
    n_angles: int = 90
    detector_size: int = 96
    sod: float = 400.0  # source-to-object distance
    sdd: float = 800.0  # source-to-detector distance
    voxel_size: float = 1.0
    pixel_size: float = 1.0


@dataclass
class PipelineConfig:
    geometry: GeometryConfig
    noise_sigma: float = 0.02
    denoise_mode: str = "auto"  # auto | redcnn | openvino | classical
    device: str = "CPU"  # CPU | GPU | AUTO
    openvino_model: Path = MODELS_DIR / "model.xml"
    redcnn_model: Path = REDCNN_DIR / "model.xml"
    seed: int = 42
    # synthetic | walnut
    data_source: str = "synthetic"
    # Zenodo walnut dose level when data_source=walnut
    walnut_dose: str = "10"
    walnut_root: Path = WALNUT_DATA_DIR
    # Recommended COR correction for Zenodo 11397266
    walnut_cor_shift: int = 4

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["openvino_model"] = str(self.openvino_model)
        data["redcnn_model"] = str(self.redcnn_model)
        data["walnut_root"] = str(self.walnut_root)
        return data


def default_config(**overrides: Any) -> PipelineConfig:
    geo_overrides = overrides.pop("geometry", {}) or {}
    geometry = GeometryConfig(**{**asdict(GeometryConfig()), **geo_overrides})
    cfg = PipelineConfig(geometry=geometry)
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if isinstance(cfg.openvino_model, str):
        cfg.openvino_model = Path(cfg.openvino_model)
    if isinstance(cfg.redcnn_model, str):
        cfg.redcnn_model = Path(cfg.redcnn_model)
    if isinstance(cfg.walnut_root, str):
        cfg.walnut_root = Path(cfg.walnut_root)
    return cfg
