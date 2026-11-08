# How It Works

## Pipeline

```text
Shepp-Logan–style phantom
        │
        ▼
Cone-beam forward projector  (+ optional Gaussian noise)
        │
        ▼
FDK reconstruction           (cosine weight → ramp filter → backproject)
        │
        ├──────────► classical denoise (Gaussian + median blend)
        │
        ├──────────► OpenVINO stub IR (slice-wise 3×3 smooth CNN)
        │
        └──────────► RED-CNN OpenVINO IR (tiled axial slices)
                     auto order: redcnn → stub → classical
        │
        ▼
Flask API packs mid-slices as PNG (base64) + KPIs
        │
        ▼
React dashboard (SSE status + REST result)
```

## Services

| Piece | Role |
|-------|------|
| `src/phantom.py` | Synthetic volume (no PHI / no patient data) |
| `src/projector.py` | Simplified circular cone-beam projector |
| `src/fdk.py` | Reference Feldkamp–Davis–Kress implementation |
| `src/denoisers/` | OpenVINO stub + classical fallback |
| `backend/app.py` | REST + SSE control plane; serves `webapp/dist` |
| `webapp/` | Configuration panel, KPI strip, slice grid |

## Why NumPy FDK?

Suite samples that ship for commercial-friendly evaluation often avoid GPL reconstruction toolkits. This reference keeps FDK in Apache/BSD-friendly NumPy/SciPy so integrators can review and replace the kernel with a production library after license and accuracy validation.

## OpenVINO stub

`scripts/generate_stub_ir.py` exports a tiny ONNX Conv→Clip graph and converts it to OpenVINO IR. It exists to prove device selection, IR loading, and fallback behavior — **not** as a clinical denoiser.

## RED-CNN

`scripts/train_export_red_cnn.py` trains a padded RED-CNN-style network on synthetic noisy Shepp-Logan slices, then exports ONNX → OpenVINO IR under `models/red_cnn/`. Runtime uses overlapping tiles (`src/denoisers/red_cnn.py`). Replace with your own LDCT-trained weights for production evaluation; the synthetic demo weights are **not** clinical.
