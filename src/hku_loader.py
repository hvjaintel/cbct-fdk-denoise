"""Loader for HKU CBCT projection dataset (Hugging Face: hku-mip/cbct_projection).

Head-and-neck cases live under ``{scanner}/hn/{case_id}/*.npz`` (MIT license).
Format and geometry defaults follow MedIP-HKU/CBCT_projection reference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ROOT, GeometryConfig

HKU_ROOT = ROOT / "data" / "hku_hn"
HKU_REPO_ID = "hku-mip/cbct_projection"
HKU_HF_URL = "https://huggingface.co/datasets/hku-mip/cbct_projection"
HKU_CODE_URL = "https://github.com/MedIP-HKU/CBCT_projection"
HKU_LICENSE = "MIT"
HKU_ATTRIBUTION = (
    "HKU MedIP CBCT projection dataset (hku-mip/cbct_projection), MIT License. "
    f"{HKU_HF_URL} — code: {HKU_CODE_URL}"
)

# Defaults from MedIP-HKU reconstruct_cbct.py (TrueBeam-style demo geometry)
DEFAULT_SID = 1000.0
DEFAULT_SDD = 1500.0
DEFAULT_DET_U = 512
DEFAULT_DET_V = 384
DEFAULT_DET_WIDTH_MM = 400.0
DEFAULT_DET_HEIGHT_MM = 300.0
DEFAULT_FIRST_ANGLE_DEG = 270.0


@dataclass
class HkuCase:
    root: Path
    scanner: str
    case_id: str
    projection_paths: list[Path]


def _read_npz_projection(path: Path, include_scatter: bool = True) -> np.ndarray:
    """Return line integrals (-log) from an HKU .npz projection file."""
    data = np.load(path)
    primary_rescaler = data["primary_rescaler"]
    scatter_rescaler = data["scatter_rescaler"]
    primary = data["primary"].astype(np.float32) * float(primary_rescaler[0]) + float(primary_rescaler[1])
    if include_scatter:
        scatter = data["scatter"].astype(np.float32) * float(scatter_rescaler[0]) + float(
            scatter_rescaler[1]
        )
        intensity = primary + scatter
    else:
        intensity = primary
    return (-np.log(np.clip(intensity, 1e-6, None))).astype(np.float32)


def discover_hku_hn_cases(root: Path | None = None) -> list[HkuCase]:
    """Find staged ``scanner/hn/case`` folders with at least one .npz."""
    root = Path(root) if root else HKU_ROOT
    if not root.exists():
        return []
    cases: list[HkuCase] = []
    # Accept data/hku_hn/{scanner}/hn/{case} or data/hku_hn/hn/{case}
    for scanner_dir in sorted(root.iterdir()):
        if not scanner_dir.is_dir():
            continue
        hn_dir = scanner_dir / "hn"
        if hn_dir.is_dir():
            scanner = scanner_dir.name
            case_dirs = [p for p in hn_dir.iterdir() if p.is_dir()]
        elif scanner_dir.name.lower() == "hn":
            scanner = "unknown"
            case_dirs = [p for p in scanner_dir.iterdir() if p.is_dir()]
        else:
            # Flat case folder with npz directly under scanner_dir if named like HN-*
            if scanner_dir.name.upper().startswith("HN") and any(scanner_dir.glob("*.npz")):
                paths = sorted(scanner_dir.glob("*.npz"))
                cases.append(
                    HkuCase(
                        root=scanner_dir,
                        scanner="local",
                        case_id=scanner_dir.name,
                        projection_paths=paths,
                    )
                )
            continue
        for case_dir in sorted(case_dirs):
            paths = sorted(case_dir.glob("*.npz"))
            if paths:
                cases.append(
                    HkuCase(
                        root=case_dir,
                        scanner=scanner,
                        case_id=case_dir.name,
                        projection_paths=paths,
                    )
                )
    return cases


def list_available_hku_hn(root: Path | None = None) -> list[dict[str, Any]]:
    return [
        {
            "case_id": c.case_id,
            "scanner": c.scanner,
            "path": str(c.root),
            "n_projections": len(c.projection_paths),
            "repo_path": f"{c.scanner}/hn/{c.case_id}",
        }
        for c in discover_hku_hn_cases(root)
    ]


def _resolve_case(case_id: str, root: Path | None = None, scanner: str | None = None) -> HkuCase:
    cases = discover_hku_hn_cases(root)
    if not cases:
        raise FileNotFoundError(
            f"No HKU head-and-neck cases under {root or HKU_ROOT}. "
            f"Stage with: python scripts/download_hku_hn.py --case iba/hn/HN-CHUM-006 "
            f"(MIT: {HKU_HF_URL})"
        )
    matches = [
        c
        for c in cases
        if c.case_id == case_id and (scanner is None or c.scanner.lower() == scanner.lower())
    ]
    if not matches and "/" in case_id:
        # Allow "iba/hn/HN-CHUM-006" or "iba/HN-CHUM-006"
        parts = case_id.replace("\\", "/").split("/")
        case_id = parts[-1]
        scanner = parts[0] if len(parts) >= 2 else scanner
        matches = [
            c
            for c in cases
            if c.case_id == case_id and (scanner is None or c.scanner.lower() == scanner.lower())
        ]
    if not matches:
        available = ", ".join(f"{c.scanner}/hn/{c.case_id}" for c in cases)
        raise FileNotFoundError(f"HKU case '{case_id}' not found. Staged: {available}")
    return matches[0]


def load_hku_hn_projections(
    case_id: str = "HN-CHUM-006",
    root: Path | None = None,
    scanner: str | None = None,
    detector_size: int = 96,
    max_angles: int | None = 90,
    volume_size: int = 64,
    include_scatter: bool = True,
) -> tuple[np.ndarray, np.ndarray, GeometryConfig, dict[str, Any]]:
    """
    Load / subsample / downsample HKU HN projections for the NumPy FDK demo.

    Returns projections [n, nv, nu], angles (rad), geometry, info.
    """
    case = _resolve_case(case_id, root=root, scanner=scanner)
    paths = case.projection_paths

    def _frame_index(path: Path) -> int:
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 0

    frame_ids = np.array([_frame_index(p) for p in paths], dtype=np.int64)
    # Infer full orbit length from filenames (smoke downloads may stage a subset)
    n_full = int(max(len(paths), int(frame_ids.max()) + 1 if len(frame_ids) else 1))

    if max_angles is not None and max_angles > 0 and len(paths) > max_angles:
        pick = np.linspace(0, len(paths) - 1, max_angles, dtype=int)
        paths = [paths[i] for i in pick]
        frame_ids = frame_ids[pick]

    # Angles: match MedIP-HKU firstAngle=270, full 360° arc over the original acquisition
    angles = np.deg2rad(
        DEFAULT_FIRST_ANGLE_DEG + frame_ids.astype(np.float64) * (360.0 / max(n_full, 1))
    ).astype(np.float32)

    sample = _read_npz_projection(paths[0], include_scatter=include_scatter)
    nv0, nu0 = sample.shape
    # Target detector size: keep aspect, scale so max(side) ~= detector_size
    scale = float(detector_size) / float(max(nv0, nu0))
    nv = max(int(round(nv0 * scale)), 16)
    nu = max(int(round(nu0 * scale)), 16)

    pixel_u = DEFAULT_DET_WIDTH_MM / float(nu0)
    pixel_v = DEFAULT_DET_HEIGHT_MM / float(nv0)
    # Our FDK uses one pixel_size; HKU demo geometry has du == dv
    pixel_eff = ((pixel_u + pixel_v) * 0.5) / scale

    projs = np.empty((len(paths), nv, nu), dtype=np.float32)
    for i, path in enumerate(paths):
        img = _read_npz_projection(path, include_scatter=include_scatter)
        if img.shape != (nv, nu):
            img = cv2.resize(img, (nu, nv), interpolation=cv2.INTER_AREA)
        projs[i] = img

    sod, sdd = DEFAULT_SID, DEFAULT_SDD
    fov = max(nu, nv) * pixel_eff * (sod / sdd)
    voxel_size = fov / float(volume_size)

    geometry = GeometryConfig(
        volume_size=volume_size,
        n_angles=len(paths),
        detector_size=max(nu, nv),
        sod=sod,
        sdd=sdd,
        voxel_size=voxel_size,
        pixel_size=pixel_eff,
    )
    info = {
        "source": "hku_hn",
        "case_id": case.case_id,
        "scanner": case.scanner,
        "scan_root": str(case.root),
        "n_projections_loaded": len(paths),
        "n_projections_total": len(case.projection_paths),
        "original_detector": [int(nv0), int(nu0)],
        "loaded_detector": [int(nv), int(nu)],
        "include_scatter": include_scatter,
        "license": HKU_LICENSE,
        "attribution": HKU_ATTRIBUTION,
        "url": HKU_HF_URL,
        "repo_id": HKU_REPO_ID,
    }
    return projs, angles, geometry, info
