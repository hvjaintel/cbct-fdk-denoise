"""End-to-end CBCT demo pipeline: synthetic or walnut → FDK → denoise."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import OUTPUT_DIR, PipelineConfig, default_config
from .denoisers import resolve_denoise
from .fdk import reconstruct_fdk
from .metrics import psnr, snr_estimate, timed
from .phantom import shepp_logan_3d
from .projector import forward_project
from .walnut_loader import load_walnut_projections


def _normalize_display(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def mid_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    z, y, x = [s // 2 for s in volume.shape]
    return {
        "axial": volume[z],
        "coronal": volume[:, y, :],
        "sagittal": volume[:, :, x],
    }


def run_pipeline(cfg: PipelineConfig | None = None, save: bool = True) -> dict[str, Any]:
    cfg = cfg or default_config()
    timings: dict[str, float] = {}
    source = (cfg.data_source or "synthetic").lower()
    walnut_info: dict[str, Any] | None = None
    phantom: np.ndarray | None = None

    if source == "walnut":
        with timed(timings, "load_ms"):
            projections, angles, geo, walnut_info = load_walnut_projections(
                dose=cfg.walnut_dose,
                root=cfg.walnut_root,
                detector_size=cfg.geometry.detector_size,
                max_angles=cfg.geometry.n_angles,
                volume_size=cfg.geometry.volume_size,
                cor_shift=cfg.walnut_cor_shift,
            )
        cfg.geometry = geo
        timings["phantom_ms"] = 0.0
        timings["project_ms"] = timings.pop("load_ms")
    else:
        geo = cfg.geometry
        with timed(timings, "phantom_ms"):
            phantom = shepp_logan_3d(geo.volume_size)
        with timed(timings, "project_ms"):
            projections, angles = forward_project(
                phantom, geo, seed=cfg.seed, noise_sigma=cfg.noise_sigma
            )

    with timed(timings, "fdk_ms"):
        recon = reconstruct_fdk(projections, angles, cfg.geometry)

    with timed(timings, "denoise_ms"):
        denoised, denoise_meta = resolve_denoise(
            recon,
            mode=cfg.denoise_mode,
            model_xml=cfg.openvino_model,
            redcnn_model=cfg.redcnn_model,
            device=cfg.device,
        )

    quality: dict[str, Any] = {
        "snr_recon": round(snr_estimate(recon), 3),
        "snr_denoise": round(snr_estimate(denoised), 3),
    }
    slices: dict[str, dict[str, np.ndarray]] = {
        "recon": mid_slices(recon),
        "denoised": mid_slices(denoised),
    }
    volumes: dict[str, np.ndarray] = {
        "recon": recon,
        "denoised": denoised,
        "projections": projections,
    }

    if phantom is not None:
        phantom_n = _normalize_display(phantom)
        quality["psnr_recon_vs_phantom"] = round(psnr(recon, phantom_n), 3)
        quality["psnr_denoise_vs_phantom"] = round(psnr(denoised, phantom_n), 3)
        slices["phantom"] = mid_slices(phantom_n)
        volumes["phantom"] = phantom_n

    result: dict[str, Any] = {
        "source": source,
        "config": cfg.to_dict(),
        "timings_ms": timings,
        "timings_total_ms": round(sum(timings.values()), 2),
        "denoise": denoise_meta,
        "quality": quality,
        "walnut": walnut_info,
        "shapes": {
            "phantom": list(phantom.shape) if phantom is not None else None,
            "projections": list(projections.shape),
            "recon": list(recon.shape),
        },
        "volumes": volumes,
        "slices": slices,
    }

    if save:
        out = OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        if phantom is not None:
            np.save(out / "phantom.npy", volumes["phantom"])
        np.save(out / "recon.npy", recon)
        np.save(out / "denoised.npy", denoised)
        np.save(out / "projections.npy", projections)
        meta = {k: v for k, v in result.items() if k not in {"volumes", "slices"}}
        (out / "last_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _save_preview_pngs(result["slices"], out)

    return result


def _save_preview_pngs(slices: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return

    preview = out / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    for stage, views in slices.items():
        for view, arr in views.items():
            img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(preview / f"{stage}_{view}.png")


def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CBCT FDK + denoise reference pipeline")
    parser.add_argument("--source", choices=["synthetic", "walnut"], default="synthetic")
    parser.add_argument("--walnut-dose", default="10", choices=["10", "25", "50", "100"])
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--angles", type=int, default=90)
    parser.add_argument("--detector", type=int, default=96)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument(
        "--denoise",
        choices=["auto", "redcnn", "openvino", "classical"],
        default="auto",
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = default_config(
        geometry={
            "volume_size": args.volume_size,
            "n_angles": args.angles,
            "detector_size": args.detector,
        },
        noise_sigma=args.noise,
        denoise_mode=args.denoise,
        device=args.device,
        data_source=args.source,
        walnut_dose=args.walnut_dose,
    )
    result = run_pipeline(cfg, save=not args.no_save)
    keys = ("source", "timings_ms", "denoise", "quality", "shapes", "walnut")
    print(json.dumps({k: result.get(k) for k in keys}, indent=2))


if __name__ == "__main__":
    run_cli()
