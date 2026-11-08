# CBCT FDK Denoise — Reference Sample

> [!NOTE]
> This application is for **reference and evaluation purposes only**. It is
> **not intended for direct use in clinical or diagnostic environments** and is not
> validated for such a purpose.

Sample application for **cone-beam CT (CBCT)** evaluation workflows:

1. Synthetic 3D phantom **or** optional **Walnut (Zenodo)** real projections ([CC BY 4.0](https://doi.org/10.5281/zenodo.11397266))
2. Cone-beam forward projection with additive noise (synthetic path)
3. **FDK** reconstruction (NumPy/SciPy reference implementation)
4. Denoising via **RED-CNN** (OpenVINO, tiled) → stub IR → **classical** fallback
5. Flask + React dashboard with axial/coronal/sagittal views and latency KPIs

Packaging mirrors patterns from the Intel® Health and Life Sciences AI Suite
samples (NICU Warmer, Surgical Instrument, Ultrasound Ovarian Segmentation):
disclaimer-first docs, one/two-command launch, staged models, third-party notices,
and both **native Windows** and **Docker** entry points.

## Quick start (Windows)

```powershell
.\setup.ps1
.\run.ps1                 # CLI pipeline → output/
.\run_webapp.ps1          # dashboard → http://127.0.0.1:8080
```

## Quick start (Make / Linux / Git Bash)

```bash
make setup
make stub-ir
make pipeline
make webui
make run                  # serves UI on :8080
```

## Docker

```bash
make docker-up
# open http://localhost:8080
make docker-down
```

## Optional: Walnut (Zenodo) data

```powershell
.\.venv\Scripts\python.exe .\scripts\download_walnut.py --dose all
.\run.ps1 -Source walnut -WalnutDose 10
# dashboard lists Walnut 10% / 25% / 50% / 100% as separate datasets
```

See [data/walnut/README.md](data/walnut/README.md). Attribution (CC BY 4.0) is required.

## What the UI shows

- Data source: Synthetic phantom or Walnut (Zenodo)
- Configuration: volume size, angles, detector, noise, denoise mode, OpenVINO device
- KPI strip: total / FDK / denoise latency, backend, device, PSNR/SNR, CPU
- Slice grid: phantom (synthetic) · FDK recon · denoised × axial / coronal / sagittal
- Mid-angle projection preview + dataset attribution when using walnut

## Project layout

```text
cbct-fdk-denoise/
  src/                 # phantom, projector, FDK, denoisers, pipeline
  backend/             # Flask API + SSE
  webapp/              # React + Vite dashboard
  scripts/             # generate_stub_ir.py
  models/denoise_stub/ # staged OpenVINO IR (generated locally)
  docker/              # Dockerfile + compose
  docs/user-guide/     # get-started, how-it-works, requirements
  setup.ps1 / run.ps1 / run_webapp.ps1
  Makefile
```

## Commercial-use notes

- License: **Apache-2.0** (see [LICENSE](LICENSE))
- Third-party notices: [third_party_programs_cbct-fdk-denoise.txt](third_party_programs_cbct-fdk-denoise.txt)
- Reconstruction uses a **permissive NumPy FDK** path (avoids GPL toolkits such as ASTRA)
- The OpenVINO denoise IR is a **non-clinical stub** for integration testing — replace with your own IR after license/validation review
- This is a **reference sample**, not a medical device

## Docs

- [Get Started](docs/user-guide/get-started.md)
- [How It Works](docs/user-guide/how-it-works.md)
- [System Requirements](docs/user-guide/get-started/system-requirements.md)
