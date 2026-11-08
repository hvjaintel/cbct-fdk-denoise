"""OpenVINO stub denoiser — loads IR if present, else falls back to classical."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .classical import classical_denoise


def _slicewise_infer(compiled: Any, volume: np.ndarray) -> np.ndarray:
    """Run a 2D stub network on each axial slice (N,C,H,W)."""
    import cv2

    out = np.empty_like(volume, dtype=np.float32)
    try:
        shape = list(compiled.input(0).shape)
        expect_h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) and shape[2] > 0 else None
        expect_w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) and shape[3] > 0 else None
    except Exception:  # noqa: BLE001
        expect_h = expect_w = None

    for z in range(volume.shape[0]):
        sl = volume[z].astype(np.float32)
        h0, w0 = sl.shape
        if expect_h and expect_w and (h0 != expect_h or w0 != expect_w):
            sl_in = cv2.resize(sl, (expect_w, expect_h), interpolation=cv2.INTER_LINEAR)
        else:
            sl_in = sl
        tensor = sl_in[None, None, ...]
        result = np.squeeze(compiled([tensor])[compiled.output(0)]).astype(np.float32)
        if result.shape != (h0, w0):
            result = cv2.resize(result, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out[z] = result
    return np.clip(out, 0.0, 1.0)


def openvino_denoise(
    volume: np.ndarray,
    model_xml: Path,
    device: str = "CPU",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Denoise with a stub OpenVINO IR (small smoothing CNN)."""
    import openvino as ov

    meta: dict[str, Any] = {"backend": "openvino", "device": device, "model": str(model_xml)}
    if not model_xml.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {model_xml}")

    core = ov.Core()
    available = core.available_devices
    target = device.upper()
    if target == "AUTO":
        target = "GPU" if "GPU" in available else "CPU"
    if target not in available and target != "CPU":
        meta["fallback_reason"] = f"{target} unavailable; using CPU"
        target = "CPU"

    model = core.read_model(model_xml)
    compiled = core.compile_model(model, target)
    meta["device"] = target
    denoised = _slicewise_infer(compiled, volume)
    return denoised, meta


def resolve_denoise(
    volume: np.ndarray,
    mode: str = "auto",
    model_xml: Path | None = None,
    device: str = "CPU",
    redcnn_model: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Select denoiser.

    Modes:
      classical — Gaussian/median blend
      openvino  — tiny stub IR (model_xml)
      redcnn    — RED-CNN OpenVINO IR (tiled)
      auto      — redcnn → stub → classical
    """
    from .red_cnn import redcnn_denoise

    mode = (mode or "auto").lower()
    model_xml = Path(model_xml) if model_xml else None
    redcnn_model = Path(redcnn_model) if redcnn_model else None

    if mode == "classical":
        return classical_denoise(volume), {"backend": "classical", "device": "CPU"}

    if mode == "redcnn":
        if not redcnn_model or not redcnn_model.exists():
            raise FileNotFoundError(
                "RED-CNN mode requested but models/red_cnn/model.xml is missing. "
                "Run: python scripts/train_export_red_cnn.py"
            )
        return redcnn_denoise(volume, redcnn_model, device=device)

    if mode == "openvino":
        if not model_xml or not model_xml.exists():
            raise FileNotFoundError("OpenVINO mode requested but stub model.xml is missing")
        return openvino_denoise(volume, model_xml, device=device)

    if mode == "auto":
        if redcnn_model and redcnn_model.exists():
            try:
                return redcnn_denoise(volume, redcnn_model, device=device)
            except Exception as exc:  # noqa: BLE001
                redcnn_err = str(exc)
        else:
            redcnn_err = "RED-CNN IR not staged"

        if model_xml and model_xml.exists():
            try:
                out, meta = openvino_denoise(volume, model_xml, device=device)
                meta["fallback_reason"] = f"auto: stub after redcnn ({redcnn_err})"
                return out, meta
            except Exception as exc:  # noqa: BLE001
                stub_err = str(exc)
        else:
            stub_err = "stub IR not staged"

        out = classical_denoise(volume)
        return out, {
            "backend": "classical",
            "device": "CPU",
            "fallback_reason": f"auto: classical (redcnn={redcnn_err}; stub={stub_err})",
        }

    raise ValueError(f"Unknown denoise mode: {mode}")
