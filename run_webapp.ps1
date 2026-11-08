# Build (if needed) and serve the React dashboard via Flask
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Write-Host "venv missing — run .\setup.ps1 first" -ForegroundColor Yellow
  exit 1
}

if (-not (Test-Path .\models\denoise_stub\model.xml)) {
  .\.venv\Scripts\python.exe .\scripts\generate_stub_ir.py
}

if (-not (Test-Path .\webapp\dist\index.html)) {
  Write-Host "==> Building web UI (needs Node.js 18+)"
  Push-Location .\webapp
  npm install
  npm run build
  Pop-Location
}

$env:PORT = if ($env:PORT) { $env:PORT } else { "8080" }
Write-Host "Dashboard → http://127.0.0.1:$env:PORT"
& .\.venv\Scripts\python.exe .\backend\app.py
