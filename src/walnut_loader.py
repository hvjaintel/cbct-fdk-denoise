"""Loader for Helsinki HelTomo-style walnut CBCT projections (Zenodo 11397266)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ROOT, GeometryConfig

WALNUT_ROOT = ROOT / "data" / "walnut"
WALNUT_DOI = "10.5281/zenodo.11397266"
WALNUT_ZENODO_URL = "https://zenodo.org/records/11397266"
WALNUT_ATTRIBUTION = (
    "Meaney, A. (2024). Cone-Beam Computed Tomography Dataset of a Walnut "
    "Imaged at 4 Different Dose Levels (1.0.0) [Data set]. Zenodo. "
    f"https://doi.org/{WALNUT_DOI}"
)

DOSE_KEYS = ("10", "25", "50", "100")


def _folder_matches_dose(name: str, dose_key: str) -> bool:
    """Match dose folders without confusing dose_10 with dose_100."""
    n = name.lower()
    token = f"dose_{dose_key}"
    return n == token or n.endswith(f"_{token}") or n.endswith(token)


@dataclass
class WalnutScan:
    root: Path
    metadata_path: Path
    metadata: dict[str, str]
    projection_paths: list[Path]
    dose: str


def parse_heltomo_metadata(path: Path) -> dict[str, str]:
    """Parse HelTomo-compatible key=value metadata (section headers ignored)."""
    meta: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("[") or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def _looks_like_scan_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_meta = any(path.rglob("*.txt"))
    has_tif = any(path.rglob("*.tif")) or any(path.rglob("*.tiff"))
    return has_meta and has_tif


def discover_walnut_scan(dose: str = "10", root: Path | None = None) -> WalnutScan:
    """
    Find an extracted walnut dose folder under data/walnut/.

    Accepts layouts such as:
      data/walnut/dose_10/
      data/walnut/2023-08-09_walnut_dose_10/
    """
    root = Path(root) if root else WALNUT_ROOT
    dose_key = str(dose).replace("%", "").strip()
    if dose_key not in DOSE_KEYS:
        raise ValueError(f"Unsupported walnut dose '{dose}'. Use one of: {list(DOSE_KEYS)}")

    if not root.exists():
        raise FileNotFoundError(
            f"Walnut data root missing: {root}. See data/walnut/README.md "
            f"or run: python scripts/download_walnut.py --dose {dose_key}"
        )

    candidates: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and _folder_matches_dose(child.name, dose_key):
            candidates.append(child)

    scan_dir = None
    for cand in candidates:
        if _looks_like_scan_dir(cand):
            scan_dir = cand
            break
    if scan_dir is None:
        raise FileNotFoundError(
            f"No walnut projections found for dose={dose_key} under {root}. "
            f"Expected a folder like dose_{dose_key}/ or *walnut_dose_{dose_key}/. "
            f"Download from {WALNUT_ZENODO_URL} and extract here. "
            f"Attribution (CC BY 4.0): {WALNUT_ATTRIBUTION}"
        )

    meta_files = sorted(scan_dir.rglob("*.txt"))
    metadata_path = None
    metadata: dict[str, str] = {}
    for mf in meta_files:
        parsed = parse_heltomo_metadata(mf)
        if "DistanceSourceOrigin" in parsed or "DistanceSourceDetector" in parsed:
            metadata_path = mf
            metadata = parsed
            break
    if metadata_path is None:
        raise FileNotFoundError(f"No HelTomo geometry metadata (.txt) under {scan_dir}")

    proj_paths = sorted(
        [p for p in scan_dir.rglob("*.tif")] + [p for p in scan_dir.rglob("*.tiff")]
    )
    # Exclude tiny non-projection TIFFs if any; keep numerical / projection-like names first
    if not proj_paths:
        raise FileNotFoundError(f"No .tif projections under {scan_dir}")

    return WalnutScan(
        root=scan_dir,
        metadata_path=metadata_path,
        metadata=metadata,
        projection_paths=proj_paths,
        dose=dose_key,
    )


def list_available_walnut_doses(root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(root) if root else WALNUT_ROOT
    available = []
    for dose in ("10", "25", "50", "100"):
        try:
            scan = discover_walnut_scan(dose, root=root)
            available.append(
                {
                    "dose": dose,
                    "path": str(scan.root),
                    "n_projections": len(scan.projection_paths),
                    "metadata": str(scan.metadata_path),
                }
            )
        except FileNotFoundError:
            continue
    return available


def _meta_float(meta: dict[str, str], key: str, default: float) -> float:
    raw = meta.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_projection(path: Path) -> np.ndarray:
    # IMREAD_UNCHANGED keeps 16-bit depth from Hamamatsu exports
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read projection: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def intensity_to_attenuation(proj: np.ndarray) -> np.ndarray:
    """Flat-fielded intensity → line integral approx via -log."""
    eps = 1e-6
    # Normalize by bright percentile (air / flat) if values are not already in (0,1]
    scale = float(np.percentile(proj, 99.5))
    if scale <= eps:
        scale = float(proj.max()) if float(proj.max()) > eps else 1.0
    norm = np.clip(proj / scale, eps, 1.0)
    return (-np.log(norm)).astype(np.float32)


def load_walnut_projections(
    dose: str = "10",
    root: Path | None = None,
    detector_size: int = 96,
    max_angles: int | None = 90,
    volume_size: int = 64,
    cor_shift: int = 4,
) -> tuple[np.ndarray, np.ndarray, GeometryConfig, dict[str, Any]]:
    """
    Load / downsample walnut projections and build GeometryConfig for FDK.

    Returns
    -------
    projections : [n_angles, det, det]
    angles : radians
    geometry : GeometryConfig
    info : attribution + paths + original sizes
    """
    scan = discover_walnut_scan(dose, root=root)
    meta = scan.metadata
    paths = scan.projection_paths

    # Drop the extra wrap-around frame if present (N+1 acquisitions)
    n_meta = int(float(meta.get("NumberImages", len(paths))))
    if len(paths) > n_meta:
        paths = paths[:n_meta]
    # If still one extra vs 360-interval convention, drop last
    angle_interval = _meta_float(meta, "AngleInterval", 1.0)
    angle_first = _meta_float(meta, "AngleFirst", 0.0)
    if len(paths) > 1 and abs((len(paths) - 1) * angle_interval - 360.0) < 1e-3:
        paths = paths[:-1]

    if max_angles is not None and max_angles > 0 and len(paths) > max_angles:
        idx = np.linspace(0, len(paths) - 1, max_angles, dtype=int)
        paths = [paths[i] for i in idx]
        angles_deg = angle_first + idx.astype(np.float64) * angle_interval
    else:
        angles_deg = angle_first + np.arange(len(paths), dtype=np.float64) * angle_interval

    sod = _meta_float(meta, "DistanceSourceOrigin", 210.66)
    sdd = _meta_float(meta, "DistanceSourceDetector", 553.74)
    pixel_size = _meta_float(meta, "PixelSize", 0.05)

    sample = _read_projection(paths[0])
    h0, w0 = sample.shape[:2]
    # Square crop centered (detector is 2240x2368)
    side = min(h0, w0)
    y0 = (h0 - side) // 2
    x0 = (w0 - side) // 2

    projs = np.empty((len(paths), detector_size, detector_size), dtype=np.float32)
    for i, path in enumerate(paths):
        img = _read_projection(path)
        if cor_shift:
            img = np.roll(img, shift=-int(cor_shift), axis=1)  # left shift
        crop = img[y0 : y0 + side, x0 : x0 + side]
        resized = cv2.resize(crop, (detector_size, detector_size), interpolation=cv2.INTER_AREA)
        projs[i] = intensity_to_attenuation(resized)

    # Effective pixel size after crop+resize
    pixel_eff = pixel_size * (side / float(detector_size))
    # Voxel size so the reconstructed FOV roughly matches the magnified detector FOV
    fov = detector_size * pixel_eff * (sod / sdd)
    voxel_size = fov / float(volume_size)

    geometry = GeometryConfig(
        volume_size=volume_size,
        n_angles=len(paths),
        detector_size=detector_size,
        sod=sod,
        sdd=sdd,
        voxel_size=voxel_size,
        pixel_size=pixel_eff,
    )
    info = {
        "source": "walnut",
        "dose": scan.dose,
        "scan_root": str(scan.root),
        "metadata_path": str(scan.metadata_path),
        "n_projections_loaded": len(paths),
        "original_detector": [int(h0), int(w0)],
        "cor_shift": cor_shift,
        "doi": WALNUT_DOI,
        "license": "CC-BY-4.0",
        "attribution": WALNUT_ATTRIBUTION,
        "zenodo_url": WALNUT_ZENODO_URL,
    }
    return projs, np.deg2rad(angles_deg).astype(np.float32), geometry, info
