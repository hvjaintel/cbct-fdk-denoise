# One-time setup: venv + Python deps (+ optional Node note)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Creating .venv"
if (-not (Test-Path .\.venv)) {
  python -m venv .venv
}

Write-Host "==> Installing Python requirements"
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host "==> Generating OpenVINO stub IR"
.\.venv\Scripts\python.exe .\scripts\generate_stub_ir.py

Write-Host ""
Write-Host "Setup complete."
Write-Host "  CLI pipeline:   .\run.ps1"
Write-Host "  Web dashboard:  .\run_webapp.ps1"
Write-Host "  RED-CNN IR:     .\.venv\Scripts\pip.exe install -r requirements-redcnn.txt"
Write-Host "                  .\.venv\Scripts\python.exe .\scripts\train_export_red_cnn.py"
Write-Host "  Docker:         make docker-up"
