"""OpenVINO compile helpers — NPU needs static spatial shapes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_NPU_HW = 64


def resolve_device(core: Any, device: str, meta: dict[str, Any] | None = None) -> str:
    available = list(core.available_devices or [])
    target = (device or "CPU").upper()
    if target == "AUTO":
        for pref in ("NPU", "GPU", "CPU"):
            if pref in available:
                return pref
        return "CPU"
    if target not in available and target != "CPU":
        if meta is not None:
            meta["fallback_reason"] = f"{target} unavailable; using CPU"
        return "CPU"
    return target


def _needs_static_spatial(partial_shape: Any) -> bool:
    """True when H/W are dynamic (common after ONNX dynamic_axes export)."""
    try:
        dims = list(partial_shape)
    except TypeError:
        return True
    if len(dims) < 4:
        return False
    for dim in dims[2:4]:
        try:
            if int(dim) <= 0:
                return True
        except Exception:  # noqa: BLE001
            return True
    return False


def prepare_model_for_device(
    model: Any,
    device: str,
    *,
    spatial: int = DEFAULT_NPU_HW,
    meta: dict[str, Any] | None = None,
) -> Any:
    """
    NPU (and some GPU paths) reject unbounded dynamic H/W.

    Force NCHW input to [1, 1, spatial, spatial] when needed.
    """
    target = (device or "CPU").upper()
    if target not in {"NPU", "GPU", "AUTO"}:
        return model

    try:
        pshape = model.input(0).partial_shape
    except Exception:  # noqa: BLE001
        return model

    if not _needs_static_spatial(pshape) and target != "NPU":
        return model

    from openvino import PartialShape

    static = PartialShape([1, 1, int(spatial), int(spatial)])
    model.reshape({0: static})
    if meta is not None:
        meta["reshaped_input"] = [1, 1, int(spatial), int(spatial)]
        meta["reshape_reason"] = f"{target} requires static H/W (was {pshape})"
    return model


def compile_ir(
    model_xml: Path,
    device: str,
    *,
    spatial: int = DEFAULT_NPU_HW,
) -> tuple[Any, str, dict[str, Any]]:
    """Read IR, reshape for NPU if needed, compile; fall back to CPU on failure."""
    import openvino as ov

    model_xml = Path(model_xml)
    meta: dict[str, Any] = {"model": str(model_xml)}
    if not model_xml.exists():
        raise FileNotFoundError(f"OpenVINO IR not found: {model_xml}")

    core = ov.Core()
    target = resolve_device(core, device, meta)
    model = prepare_model_for_device(core.read_model(model_xml), target, spatial=spatial, meta=meta)

    try:
        compiled = core.compile_model(model, target)
    except Exception as exc:  # noqa: BLE001
        if target == "CPU":
            raise
        reason = f"{target} compile failed ({exc}); using CPU"
        meta["fallback_reason"] = (
            f"{meta['fallback_reason']}; {reason}" if meta.get("fallback_reason") else reason
        )
        target = "CPU"
        # Fresh model — previous may already be reshaped; CPU accepts either.
        model = prepare_model_for_device(
            core.read_model(model_xml), target, spatial=spatial, meta=None
        )
        compiled = core.compile_model(model, target)

    meta["device"] = target
    return compiled, target, meta
