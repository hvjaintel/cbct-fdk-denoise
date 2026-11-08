#!/usr/bin/env python3
"""Generate a tiny OpenVINO IR stub denoiser (identity-ish 3x3 smooth)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "denoise_stub"


def build_onnx(path: Path, size: int = 64) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Mild 3x3 average with center bias (stub "denoiser")
    kernel = np.array(
        [[[[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]]],
        dtype=np.float32,
    )

    # Dynamic H/W so demo volumes (e.g. 48³) do not require a rebuild
    _ = size  # kept for CLI compatibility / documentation
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, "H", "W"])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "H", "W"])
    w = numpy_helper.from_array(kernel, name="W")
    min_c = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="clip_min")
    max_c = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="clip_max")

    conv = helper.make_node(
        "Conv",
        inputs=["input", "W"],
        outputs=["conv_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    clip = helper.make_node(
        "Clip",
        inputs=["conv_out", "clip_min", "clip_max"],
        outputs=["output"],
    )
    graph = helper.make_graph(
        [conv, clip],
        "cbct_denoise_stub",
        [x],
        [y],
        [w, min_c, max_c],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def convert_to_ir(onnx_path: Path, out_dir: Path) -> Path:
    import openvino as ov

    out_dir.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path))
    xml_path = out_dir / "model.xml"
    ov.save_model(model, str(xml_path))
    return xml_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=64, help="Expected slice H=W")
    args = parser.parse_args()

    onnx_path = args.out / "model.onnx"
    print(f"Writing ONNX stub → {onnx_path}")
    build_onnx(onnx_path, size=args.size)
    print("Converting to OpenVINO IR…")
    xml = convert_to_ir(onnx_path, args.out)
    print(f"Stub IR ready: {xml}")
    print(f"Companion bin: {xml.with_suffix('.bin')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
