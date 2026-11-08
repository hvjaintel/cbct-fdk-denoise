#!/usr/bin/env python3
"""Train a tiny RED-CNN on synthetic noisy phantom slices and export OpenVINO IR.

This produces a demo-ready model (not clinical). Replace with your own weights
trained on real LDCT pairs when moving beyond the sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "models" / "red_cnn"


def _make_pairs(n: int, size: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from src.phantom import shepp_logan_3d

    rng = np.random.default_rng(seed)
    vol = shepp_logan_3d(size)
    # Use axial + a few coronal/sagittal-ish crops as 2D patches
    clean_list = []
    for z in range(vol.shape[0]):
        clean_list.append(vol[z])
    for y in range(0, vol.shape[1], max(vol.shape[1] // 8, 1)):
        clean_list.append(vol[:, y, :])
    cleans = []
    noisies = []
    while len(cleans) < n:
        img = clean_list[rng.integers(0, len(clean_list))].astype(np.float32)
        if img.shape[0] != size or img.shape[1] != size:
            # center crop / pad
            canvas = np.zeros((size, size), dtype=np.float32)
            h, w = img.shape
            h2, w2 = min(h, size), min(w, size)
            canvas[:h2, :w2] = img[:h2, :w2]
            img = canvas
        noisy = np.clip(img + rng.normal(0.0, noise, size=img.shape).astype(np.float32), 0.0, 1.0)
        cleans.append(img)
        noisies.append(noisy)
    return np.stack(noisies), np.stack(cleans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Export randomly initialized weights only")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for RED-CNN train/export.\n"
            "  pip install -r requirements-redcnn.txt"
        ) from exc

    from src.denoisers.red_cnn_arch import REDCNN

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = REDCNN(channels=64).to(device)  # slightly smaller for demo IR size

    if not args.skip_train:
        print(f"Building synthetic pairs (n={args.samples}, patch={args.patch})…")
        noisy, clean = _make_pairs(args.samples, args.patch, args.noise, args.seed)
        ds = TensorDataset(
            torch.from_numpy(noisy[:, None, ...]),
            torch.from_numpy(clean[:, None, ...]),
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()
        model.train()
        for epoch in range(1, args.epochs + 1):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.size(0)
            print(f"  epoch {epoch}/{args.epochs}  mse={total / len(ds):.6f}")
    else:
        print("Skipping train — exporting random initialization")

    model.eval()
    ckpt = args.out / "red_cnn.pt"
    torch.save({"state_dict": model.state_dict(), "channels": 64, "patch": args.patch}, ckpt)
    print(f"Saved checkpoint → {ckpt}")

    onnx_path = args.out / "model.onnx"
    dummy = torch.randn(1, 1, args.patch, args.patch, device=device)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}},
        opset_version=13,
    )
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch without dynamo= flag
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    print(f"ONNX → {onnx_path}")

    import openvino as ov

    ov_model = ov.convert_model(str(onnx_path))
    xml_path = args.out / "model.xml"
    ov.save_model(ov_model, str(xml_path))
    meta = {
        "name": "red_cnn_demo",
        "note": "Trained on synthetic Shepp-Logan noise pairs for demo only — not clinical.",
        "patch": args.patch,
        "channels": 64,
        "epochs": 0 if args.skip_train else args.epochs,
        "citation": "Chen et al., Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN), IEEE TMI 2017.",
    }
    (args.out / "model_card.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OpenVINO IR → {xml_path}")
    print("Done. Use --denoise redcnn  or dashboard mode 'redcnn'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
