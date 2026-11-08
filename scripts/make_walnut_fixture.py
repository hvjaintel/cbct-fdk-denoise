#!/usr/bin/env python3
"""Create a tiny HelTomo-compatible walnut fixture for offline smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "walnut" / "dose_10"

META = """[GENERAL]
ProjectName=fixture_walnut_dose_10
Scanner=Synthetic fixture for CBCT-FDK-Denoise smoke tests
Measurers=fixture
Date=20240101
DateFormat=YYYYMMDD

[GEOMETRY]
GeometryType=Cone
DistanceSourceDetector=553.74
DistanceSourceOrigin=210.66
DistanceUnit=mm

[ACQUISITION]
NumberImages=36
AngleFirst=0
AngleInterval=10
AngleLast=360

[DETECTOR]
Detector=Fixture
DetectorType=EID
Binning=1x1
PixelSize=0.050
PixelSizeUnit=mm
ExposureTime=1000
ExposureTimeUnit=ms

[XRAY]
Tube=Fixture
Target=Mo
Voltage=40
VoltageUnit=kV
Current=0.2
CurrentUnit=mA
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fixture_walnut.txt").write_text(META, encoding="utf-8")
    yy, xx = np.mgrid[-63.5:64.5, -63.5:64.5]
    disk = ((xx / 28) ** 2 + (yy / 36) ** 2) <= 1.0
    base = np.full((128, 128), 20000.0, dtype=np.float32)
    base[disk] = 8000.0
    for i in range(36):
        # Rotate attenuation pattern in intensity space for crude motion
        angle = i * 10.0
        M = np.array(
            [
                [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
            ],
            dtype=np.float32,
        )
        # PIL rotate for simplicity
        arr = np.clip(base, 0, 65535).astype(np.uint16)
        img = Image.fromarray(arr)
        rot = img.rotate(angle, resample=Image.BILINEAR, fillcolor=20000)
        rot.save(OUT / f"proj_{i:03d}.tif")
    print(f"Fixture written to {OUT}")
    print("Note: this is NOT the Zenodo dataset — for loader/UI smoke tests only.")


if __name__ == "__main__":
    main()
