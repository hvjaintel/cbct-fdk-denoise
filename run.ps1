# Run the CBCT FDK + denoise CLI pipeline
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Write-Host "venv missing — run .\setup.ps1 first" -ForegroundColor Yellow
  exit 1
}

$volume = 48
$angles = 60
$detector = 64
$denoise = "auto"
$device = "CPU"
$source = "synthetic"
$walnutDose = "10"
$hkuCase = "HN-CHUM-006"
$hkuScanner = "iba"

for ($i = 0; $i -lt $args.Count; $i++) {
  switch ($args[$i]) {
    "-VolumeSize" { $volume = [int]$args[++$i] }
    "-Angles" { $angles = [int]$args[++$i] }
    "-Detector" { $detector = [int]$args[++$i] }
    "-Denoise" { $denoise = $args[++$i] }
    "-Device" { $device = $args[++$i] }
    "-Source" { $source = $args[++$i] }
    "-WalnutDose" { $walnutDose = $args[++$i] }
    "-HkuCase" { $hkuCase = $args[++$i] }
    "-HkuScanner" { $hkuScanner = $args[++$i] }
  }
}

if (-not (Test-Path .\models\denoise_stub\model.xml)) {
  .\.venv\Scripts\python.exe .\scripts\generate_stub_ir.py
}

& .\.venv\Scripts\python.exe -m src.pipeline `
  --source $source `
  --walnut-dose $walnutDose `
  --hku-case $hkuCase `
  --hku-scanner $hkuScanner `
  --volume-size $volume `
  --angles $angles `
  --detector $detector `
  --denoise $denoise `
  --device $device
