# Get Started

## 1. Clone / open the project

```text
cbct-fdk-denoise/
```

## 2. Windows native setup

```powershell
.\setup.ps1
```

This creates `.venv`, installs dependencies, and generates the OpenVINO stub IR under `models/denoise_stub/`.

### CLI reconstruction

```powershell
.\run.ps1
.\run.ps1 -VolumeSize 48 -Angles 60 -Detector 64 -Denoise auto -Device CPU
.\run.ps1 -Source walnut -WalnutDose 10
```

Artifacts land in `output/` (`*.npy`, `last_run.json`, `preview/*.png`).

### Optional walnut dataset (Zenodo, CC BY 4.0)

```powershell
.\.venv\Scripts\python.exe .\scripts\download_walnut.py --dose 10
```

Then select **Walnut (Zenodo)** in the dashboard, or use `-Source walnut`. See [data/walnut/README.md](../../data/walnut/README.md).

### Web dashboard

Requires [Node.js](https://nodejs.org) 18+ for the first UI build.

```powershell
.\run_webapp.ps1
```

Open http://127.0.0.1:8080 — configure parameters, click **Start reconstruction**.

## 3. Make targets

```bash
make setup
make stub-ir
make pipeline
make webui
make run
```

## 4. Docker

```bash
make docker-up
```

Compose publishes the dashboard on port **8080**. Volumes mount `models/` and `output/` for persistence.

## 5. Model staging

The stub denoiser is created by:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_stub_ir.py
```

If `models/denoise_stub/model.xml` is missing, denoise mode `auto` falls back to classical filtering.
