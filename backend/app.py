"""Flask control plane + SSE for the CBCT-FDK-Denoise dashboard."""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, REDCNN_DIR, default_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.hku_loader import (  # noqa: E402
    HKU_ATTRIBUTION,
    HKU_HF_URL,
    HKU_LICENSE,
    list_available_hku_hn,
)
from src.walnut_loader import (  # noqa: E402
    WALNUT_ATTRIBUTION,
    WALNUT_DOI,
    WALNUT_ZENODO_URL,
    list_available_walnut_doses,
)

app = Flask(__name__, static_folder=None)
CORS(app)

_STATE: dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "message": "Ready",
    "last_result": None,
    "error": None,
}
_LOCK = threading.Lock()
_JOB: threading.Thread | None = None


def _encode_png(arr: np.ndarray) -> str:
    img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _public_result(raw: dict[str, Any]) -> dict[str, Any]:
    slices = {}
    for stage, views in raw["slices"].items():
        slices[stage] = {view: _encode_png(arr) for view, arr in views.items()}
    return {
        "source": raw.get("source", "synthetic"),
        "timings_ms": raw["timings_ms"],
        "timings_total_ms": raw["timings_total_ms"],
        "denoise": raw["denoise"],
        "quality": raw["quality"],
        "shapes": raw["shapes"],
        "config": raw["config"],
        "walnut": raw.get("walnut"),
        "hku": raw.get("hku"),
        "slices": slices,
        "projection_preview": _encode_png(
            _normalize_preview(
                raw["volumes"]["projections"][raw["volumes"]["projections"].shape[0] // 2]
            )
        ),
    }


def _normalize_preview(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _run_job(params: dict[str, Any]) -> None:
    source = (params.get("data_source") or "synthetic").lower()
    if source == "walnut":
        msg0 = "Loading walnut projections…"
    elif source in {"hku_hn", "hku"}:
        msg0 = "Loading HKU HN projections…"
        source = "hku_hn"
    else:
        msg0 = "Building phantom…"
    with _LOCK:
        _STATE.update({"status": "running", "progress": 5, "message": msg0, "error": None})
    try:
        cfg = default_config(
            geometry={
                "volume_size": int(params.get("volume_size", 48)),
                "n_angles": int(params.get("n_angles", 60)),
                "detector_size": int(params.get("detector_size", 64)),
            },
            noise_sigma=float(params.get("noise_sigma", 0.025)),
            denoise_mode=params.get("denoise_mode", "auto"),
            device=params.get("device", "CPU"),
            data_source=source,
            walnut_dose=str(params.get("walnut_dose", "10")),
            hku_case=str(params.get("hku_case", "HN-CHUM-006")),
            hku_scanner=str(params.get("hku_scanner", "iba")),
        )
        with _LOCK:
            _STATE.update(
                {
                    "progress": 25,
                    "message": (
                        "FDK reconstructing…"
                        if source in {"walnut", "hku_hn"}
                        else "Forward projecting…"
                    ),
                }
            )
        result = run_pipeline(cfg, save=True)
        with _LOCK:
            _STATE.update({"progress": 90, "message": "Packaging previews…"})
        public = _public_result(result)
        with _LOCK:
            _STATE.update(
                {
                    "status": "done",
                    "progress": 100,
                    "message": "Complete",
                    "last_result": public,
                }
            )
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _STATE.update({"status": "error", "progress": 0, "message": "Failed", "error": str(exc)})


@app.get("/api/health")
def health():
    devices = ["CPU"]
    try:
        import openvino as ov

        devices = ov.Core().available_devices or ["CPU"]
    except Exception:  # noqa: BLE001
        pass
    return jsonify(
        {
            "ok": True,
            "devices": devices,
            "status": _STATE["status"],
            "models": {
                "stub": (MODELS_DIR / "model.xml").exists(),
                "redcnn": (REDCNN_DIR / "model.xml").exists(),
            },
        }
    )


@app.get("/api/datasets")
def datasets():
    staged = {d["dose"]: d for d in list_available_walnut_doses()}
    walnut_sources = []
    for dose in ("10", "25", "50", "100"):
        info = staged.get(dose)
        walnut_sources.append(
            {
                "id": f"walnut_{dose}",
                "label": f"Walnut {dose}% (Zenodo)",
                "available": info is not None,
                "data_source": "walnut",
                "walnut_dose": dose,
                "path": info["path"] if info else None,
                "n_projections": info["n_projections"] if info else 0,
                "doi": WALNUT_DOI,
                "license": "CC-BY-4.0",
                "attribution": WALNUT_ATTRIBUTION,
                "url": WALNUT_ZENODO_URL,
                "stage_hint": f"python scripts/download_walnut.py --dose {dose}",
            }
        )

    hku_staged = list_available_hku_hn()
    hku_sources = []
    if hku_staged:
        for info in hku_staged:
            hku_sources.append(
                {
                    "id": f"hku_{info['scanner']}_{info['case_id']}",
                    "label": f"HKU HN {info['case_id']} ({info['scanner']})",
                    "available": True,
                    "data_source": "hku_hn",
                    "hku_case": info["case_id"],
                    "hku_scanner": info["scanner"],
                    "path": info["path"],
                    "n_projections": info["n_projections"],
                    "license": HKU_LICENSE,
                    "attribution": HKU_ATTRIBUTION,
                    "url": HKU_HF_URL,
                    "stage_hint": (
                        f"python scripts/download_hku_hn.py --case {info['scanner']}/hn/{info['case_id']}"
                    ),
                }
            )
    else:
        hku_sources.append(
            {
                "id": "hku_hn_placeholder",
                "label": "HKU HN (not staged)",
                "available": False,
                "data_source": "hku_hn",
                "hku_case": "HN-CHUM-006",
                "hku_scanner": "iba",
                "license": HKU_LICENSE,
                "attribution": HKU_ATTRIBUTION,
                "url": HKU_HF_URL,
                "stage_hint": "python scripts/download_hku_hn.py --case iba/hn/HN-CHUM-006",
            }
        )

    return jsonify(
        {
            "sources": [
                {
                    "id": "synthetic",
                    "label": "Synthetic phantom",
                    "available": True,
                    "data_source": "synthetic",
                },
                *walnut_sources,
                *hku_sources,
            ],
            "stage_all_hint": "python scripts/download_walnut.py --dose all",
            "stage_hku_hint": "python scripts/download_hku_hn.py --case iba/hn/HN-CHUM-006",
        }
    )


@app.get("/api/status")
def status():
    with _LOCK:
        payload = {
            "status": _STATE["status"],
            "progress": _STATE["progress"],
            "message": _STATE["message"],
            "error": _STATE["error"],
            "has_result": _STATE["last_result"] is not None,
        }
    return jsonify(payload)


@app.get("/api/result")
def result():
    with _LOCK:
        if _STATE["last_result"] is None:
            return jsonify({"error": "No result yet"}), 404
        return jsonify(_STATE["last_result"])


@app.post("/api/run")
def run():
    global _JOB
    with _LOCK:
        if _STATE["status"] == "running":
            return jsonify({"error": "Job already running"}), 409
        params = request.get_json(silent=True) or {}
        _STATE.update({"status": "running", "progress": 1, "message": "Queued", "error": None})
    _JOB = threading.Thread(target=_run_job, args=(params,), daemon=True)
    _JOB.start()
    return jsonify({"ok": True})


@app.get("/api/events")
def events():
    def stream():
        last = None
        while True:
            with _LOCK:
                snap = {
                    "status": _STATE["status"],
                    "progress": _STATE["progress"],
                    "message": _STATE["message"],
                    "error": _STATE["error"],
                    "has_result": _STATE["last_result"] is not None,
                    "denoise": (_STATE["last_result"] or {}).get("denoise"),
                    "timings_ms": (_STATE["last_result"] or {}).get("timings_ms"),
                    "quality": (_STATE["last_result"] or {}).get("quality"),
                    "source": (_STATE["last_result"] or {}).get("source"),
                }
            blob = json.dumps(snap)
            if blob != last:
                yield f"data: {blob}\n\n"
                last = blob
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream")


@app.get("/api/telemetry")
def telemetry():
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
    except Exception:  # noqa: BLE001
        cpu, mem = 0.0, 0.0
    return jsonify({"cpu_percent": cpu, "memory_percent": mem})


def _dist_dir() -> Path:
    return ROOT / "webapp" / "dist"


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path: str):
    dist = _dist_dir()
    if dist.exists():
        target = dist / path if path else dist / "index.html"
        if target.is_file():
            return send_from_directory(target.parent, target.name)
        return send_from_directory(dist, "index.html")
    return jsonify(
        {
            "message": "API is running. Build the web UI with: cd webapp && npm install && npm run build",
            "health": "/api/health",
        }
    )


def main() -> None:
    host = "0.0.0.0"
    port = int((__import__("os").environ.get("PORT") or "8080"))
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
