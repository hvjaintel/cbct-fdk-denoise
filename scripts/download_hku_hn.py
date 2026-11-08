#!/usr/bin/env python3
"""Download HKU head-and-neck CBCT projection cases (MIT) from Hugging Face.

Dataset: https://huggingface.co/datasets/hku-mip/cbct_projection
Default case: iba/hn/HN-CHUM-006
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hku_hn"
REPO_ID = "hku-mip/cbct_projection"
DEFAULT_CASE = "iba/hn/HN-CHUM-006"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help="Repo folder prefix, e.g. iba/hn/HN-CHUM-006 or truebeam/hn/...",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of .npz files (0 = all). Useful for smoke tests.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required.\n  pip install -r requirements-hku.txt"
        ) from exc

    case = args.case.strip().replace("\\", "/").strip("/")
    if not case.endswith("/"):
        prefix = case + "/"
    else:
        prefix = case
        case = case.rstrip("/")

    print("Dataset: hku-mip/cbct_projection (MIT License)")
    print("https://huggingface.co/datasets/hku-mip/cbct_projection")
    print(f"Downloading case prefix: {prefix}")
    print(f" → {args.out}")

    files = list_repo_files(REPO_ID, repo_type="dataset")
    targets = sorted(f for f in files if f.startswith(prefix) and f.endswith(".npz"))
    if not targets:
        # Helpful hint if user passed only case id
        hn = sorted({"/".join(f.split("/")[:3]) for f in files if "/hn/" in f})
        sample = ", ".join(hn[:8]) if hn else "(rate-limited or empty listing)"
        raise SystemExit(f"No .npz files under '{prefix}'. Example HN folders: {sample}")

    if args.max_files and args.max_files > 0:
        # Even subsample across the orbit
        idx = [
            int(round(i * (len(targets) - 1) / (args.max_files - 1)))
            for i in range(args.max_files)
        ]
        targets = [targets[i] for i in sorted(set(idx))]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {f}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type="dataset",
            local_dir=str(args.out),
        )

    local_case = args.out / case
    print(f"\nReady → {local_case}")
    print("Use dashboard dataset 'HKU HN …' or:")
    print(f"  python -m src.pipeline --source hku_hn --hku-case {case.split('/')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
