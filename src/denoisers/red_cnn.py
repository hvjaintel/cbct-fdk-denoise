"""OpenVINO RED-CNN inference with overlapping tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .classical import classical_denoise


def _compile(model_xml: Path, device: str) -> tuple[Any, str, dict[str, Any]]:
    import openvino as ov

    meta: dict[str, Any] = {"backend": "redcnn", "model": str(model_xml)}
    if not model_xml.exists():
        raise FileNotFoundError(f"RED-CNN IR not found: {model_xml}")

    core = ov.Core()
    available = core.available_devices
    target = device.upper()
    if target == "AUTO":
        target = "GPU" if "GPU" in available else "CPU"
    if target not in available and target != "CPU":
        meta["fallback_reason"] = f"{target} unavailable; using CPU"
        target = "CPU"

    compiled = core.compile_model(core.read_model(model_xml), target)
    meta["device"] = target
    return compiled, target, meta


def _tile_params(compiled: Any, default: int = 64) -> int:
    try:
        shape = list(compiled.input(0).shape)
        h = shape[2]
        if isinstance(h, (int, np.integer)) and int(h) > 0:
            return int(h)
    except Exception:  # noqa: BLE001
        pass
    return default


def _denoise_slice_tiled(
    compiled: Any,
    sl: np.ndarray,
    tile: int,
    overlap: int = 16,
) -> np.ndarray:
    """Run fixed-size tiles with Hann-ish overlap blend."""
    h, w = sl.shape
    if h <= tile and w <= tile:
        canvas = np.zeros((tile, tile), dtype=np.float32)
        canvas[:h, :w] = sl
        out = np.squeeze(compiled([canvas[None, None, ...]])[compiled.output(0)]).astype(np.float32)
        return out[:h, :w]

    step = max(tile - overlap, 1)
    acc = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    yy = np.linspace(0, np.pi, tile, dtype=np.float64)
    xx = np.linspace(0, np.pi, tile, dtype=np.float64)
    wy = 0.5 - 0.5 * np.cos(yy)
    wx = 0.5 - 0.5 * np.cos(xx)
    window = np.outer(wy, wx)

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y + tile, h)
            x1 = min(x + tile, w)
            y0 = y1 - tile
            x0 = x1 - tile
            if y0 < 0:
                y0, y1 = 0, min(tile, h)
            if x0 < 0:
                x0, x1 = 0, min(tile, w)

            patch = np.zeros((tile, tile), dtype=np.float32)
            ph, pw = y1 - y0, x1 - x0
            patch[:ph, :pw] = sl[y0:y1, x0:x1]
            pred = np.squeeze(compiled([patch[None, None, ...]])[compiled.output(0)]).astype(np.float32)
            acc[y0:y1, x0:x1] += pred[:ph, :pw] * window[:ph, :pw]
            weight[y0:y1, x0:x1] += window[:ph, :pw]

    weight = np.maximum(weight, 1e-6)
    return (acc / weight).astype(np.float32)


def redcnn_denoise(
    volume: np.ndarray,
    model_xml: Path,
    device: str = "CPU",
    tile: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    compiled, target, meta = _compile(Path(model_xml), device)
    tile_size = tile or _tile_params(compiled, 64)
    meta["tile"] = tile_size
    meta["device"] = target

    out = np.empty_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        sl = np.clip(volume[z].astype(np.float32), 0.0, 1.0)
        out[z] = _denoise_slice_tiled(compiled, sl, tile=tile_size)
    return np.clip(out, 0.0, 1.0), meta


def try_redcnn_or_raise(
    volume: np.ndarray,
    model_xml: Path,
    device: str = "CPU",
) -> tuple[np.ndarray, dict[str, Any]]:
    return redcnn_denoise(volume, model_xml, device=device)
