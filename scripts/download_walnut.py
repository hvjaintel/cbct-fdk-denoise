#!/usr/bin/env python3
"""Download + extract walnut dose zips from Zenodo record 11397266 (CC BY 4.0)."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "walnut"

FILES = {
    "10": "2023-08-09_walnut_dose_10.zip",
    "25": "2023-08-09_walnut_dose_25.zip",
    "50": "2023-08-09_walnut_dose_50.zip",
    "100": "2023-08-09_walnut_dose_100.zip",
}
BASE = "https://zenodo.org/records/11397266/files"


def _has_projections(extract_dir: Path) -> bool:
    if not extract_dir.is_dir():
        return False
    tifs = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff"))
    txts = list(extract_dir.rglob("*.txt"))
    # Real Zenodo extracts are large; ignore tiny smoke fixtures (< 8 frames)
    return len(tifs) >= 8 and any(
        "DistanceSourceOrigin" in p.read_text(encoding="utf-8", errors="ignore")
        for p in txts
    )


def download_dose(dose: str, out: Path, keep_zip: bool = False) -> Path:
    name = FILES[dose]
    url = f"{BASE}/{name}?download=1"
    zip_path = out / name
    extract_dir = out / f"dose_{dose}"

    if _has_projections(extract_dir):
        print(f"[skip] dose {dose}% already staged at {extract_dir}")
        return extract_dir

    print(f"\n=== Walnut dose {dose}% ===")
    print(f"Downloading {url}")
    print(f" → {zip_path}  (~1.5–2.2 GB)")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = 100.0 * done / total
        mb = done / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:.0f} MB)")
        sys.stdout.flush()

    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        urlretrieve(url, zip_path, reporthook=_progress)
        print()
    else:
        print(f"  using existing zip ({zip_path.stat().st_size / (1024**3):.2f} GB)")

    print("Extracting…")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    if not keep_zip:
        try:
            zip_path.unlink()
            print(f"Removed zip {zip_path.name}")
        except OSError:
            pass

    if not _has_projections(extract_dir):
        raise RuntimeError(f"Extract finished but no usable projections in {extract_dir}")
    print(f"Ready → {extract_dir}")
    return extract_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dose",
        choices=[*sorted(FILES, key=lambda x: int(x)), "all"],
        default="all",
        help="Dose level to download (default: all)",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip after extract")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print("Dataset: Zenodo 10.5281/zenodo.11397266 (CC BY 4.0)")
    print("Attribution: Meaney, A. (2024). https://doi.org/10.5281/zenodo.11397266")

    doses = list(FILES) if args.dose == "all" else [args.dose]
    # Stable order: 10 → 25 → 50 → 100
    doses = sorted(doses, key=lambda d: int(d))
    for dose in doses:
        download_dose(dose, args.out, keep_zip=args.keep_zip)

    print("\nAll requested doses staged under", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
