# Walnut (Zenodo) CBCT projections

Optional real CBCT input for this sample (not shipped in git).

## Dataset

**Cone-Beam Computed Tomography Dataset of a Walnut Imaged at 4 Different Dose Levels**  
Alexander Meaney, University of Helsinki  
DOI: [10.5281/zenodo.11397266](https://doi.org/10.5281/zenodo.11397266)  
License: **CC BY 4.0** (commercial use allowed with attribution)

> This is a walnut phantom dataset, not clinical patient data. The sample app remains for evaluation only.

## Stage data

Download one dose zip from Zenodo (≈1.5–2.2 GB each), then extract under this folder:

```powershell
# All four dose datasets (10 / 25 / 50 / 100%) — ~7+ GB total
.\.venv\Scripts\python.exe .\scripts\download_walnut.py --dose all

# Or a single dose
.\.venv\Scripts\python.exe .\scripts\download_walnut.py --dose 10
```

Each dose becomes its own dashboard dataset:

- Walnut 10% (Zenodo)
- Walnut 25% (Zenodo)
- Walnut 50% (Zenodo)
- Walnut 100% (Zenodo)

For a tiny offline smoke-test layout (not the real Zenodo data):

```powershell
.\.venv\Scripts\python.exe .\scripts\make_walnut_fixture.py
```

Or manually:

1. Download `2023-08-09_walnut_dose_10.zip` (or 25 / 50 / 100) from the Zenodo record  
2. Extract so a metadata `.txt` and projection `.tif` files are under one of:

```text
data/walnut/dose_10/
data/walnut/2023-08-09_walnut_dose_10/
```

The loader searches recursively for HelTomo-style `*.txt` metadata and `*.tif` / `*.tiff` projections.

## COR note (from dataset authors)

Shift each projection **left by 4 pixels** (circular) before reconstruction. The loader applies this by default (`cor_shift=4`).

## Attribution (required)

When redistributing results that include this data, credit:

> Meaney, A. (2024). Cone-Beam Computed Tomography Dataset of a Walnut Imaged at 4 Different Dose Levels (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11397266
