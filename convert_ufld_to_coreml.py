"""
convert_ufld_to_coreml.py
=========================

Ultra‑Fast Lane Detection (UFLD) is a lightweight lane detector that operates
efficiently on mobile devices.  Pre‑trained checkpoints are available for
datasets such as CULane and TuSimple.  This script demonstrates how to convert
a UFLD model from ONNX to Core ML using `coremltools`.  You can either train
your own UFLD model (see the official repository for instructions) or download
a pre‑trained ONNX checkpoint.

Notes:
  * The UFLD model outputs a row‑wise classification of lane positions.  After
    conversion, you will need to replicate the decoding logic in Swift to
    reconstruct polylines for the lane markers.
  * Core ML does not currently support dynamic input shapes for all layers,
    therefore set a fixed input image size (e.g. 288×800) matching the model
    training configuration.

References:
  * The UFLD authors provide trained models and inference scripts at
    https://github.com/cfzd/Ultra-Fast-Lane-Detection
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import coremltools as ct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a UFLD ONNX model to Core ML.")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the UFLD ONNX model (e.g. resnet18_culane_288x800.onnx)",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=288,
        help="Input image height used during training",
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=800,
        help="Input image width used during training",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output path for the converted .mlmodel",
    )
    return parser.parse_args()


def convert_ufld(onnx_path: Path, h: int, w: int, out_path: Path | None) -> None:
    # Load the ONNX model
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # Define input shape.  UFLD models expect a normalized float32 tensor
    # with shape [1,3,H,W].  We fix H and W to the values used during training.
    input_name = model.graph.input[0].name
    mlmodel = ct.convert(
        model,
        inputs=[ct.ImageType(name=input_name, shape=(1, 3, h, w), scale=1/255.0, bias=[0, 0, 0])],
    )

    # Save the Core ML model
    out_path = out_path or onnx_path.with_suffix(".mlmodel")
    mlmodel.save(str(out_path))
    print(f"Saved Core ML lane model to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    convert_ufld(Path(args.onnx_path), args.input_height, args.input_width, Path(args.output_path) if args.output_path else None)