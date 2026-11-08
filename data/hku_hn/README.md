# HKU Head-and-Neck CBCT projections

Optional real **patient CBCT projection** input for this sample (not shipped in git).

## Dataset

**hku-mip/cbct_projection** (University of Hong Kong MedIP)  
Hugging Face: https://huggingface.co/datasets/hku-mip/cbct_projection  
Code/scripts: https://github.com/MedIP-HKU/CBCT_projection  
License: **MIT** (dataset card + repository LICENSE)

> De-identified research projections for evaluation only. This sample app is **not** for clinical or diagnostic use.

Head-and-neck cases are under `{scanner}/hn/{case_id}/` (e.g. `iba/hn/HN-CHUM-006`), with one `.npz` file per angle (`primary` / `scatter` + rescalers).

## Stage a case

```powershell
.\.venv\Scripts\pip.exe install -r requirements-hku.txt
# Full case (~720 projections)
.\.venv\Scripts\python.exe .\scripts\download_hku_hn.py --case iba/hn/HN-CHUM-006
# Smoke test (subset of angles)
.\.venv\Scripts\python.exe .\scripts\download_hku_hn.py --case iba/hn/HN-CHUM-006 --max-files 60
```

Expected layout:

```text
data/hku_hn/iba/hn/HN-CHUM-006/*.npz
```

## Run

```powershell
.\run.ps1 -Source hku_hn -HkuCase HN-CHUM-006
# or pick “HKU HN …” in the dashboard
```

## Attribution

When redistributing results that include this data, credit the HKU MedIP CBCT projection dataset and retain the MIT notice.
