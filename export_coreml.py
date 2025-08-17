"""
export_coreml.py
=================

This script converts a trained YOLOv8 model into Apple's Core ML format (.mlpackage)
for deployment on iOS devices.  Ultralytics' export API supports direct
conversion to Core ML with optional quantization and NMS (non‑maximum suppression)
embedded in the model.  See the official Ultralytics export documentation for
details【458114777348310†L240-L258】.

Usage:
  1. Ensure you have trained a model using `train_yolo_traffic.py`.  The best
     checkpoint will be stored in `runs/detect/<run_name>/weights/best.pt`.
  2. Run this script, specifying the path to the trained checkpoint and desired
     output directory:

        python export_coreml.py --weights runs/detect/yolov8s_driving/weights/best.pt --imgsz 640 --int8

  3. The resulting `.mlpackage` will be saved in the same directory as the
     weights file.

Notes:
  * `imgsz` determines the input image size that Core ML will expect.  Set
    this to match the size used during training (typically 640 for YOLOv8s).
  * Passing `--int8` enables 8‑bit quantization for smaller models and faster
    inference on the Neural Engine【458114777348310†L324-L326】.  Provide a
    representative dataset via the `data` argument for better calibration if
    possible.  Without this flag, the exported model uses full precision (FP16).
  * Use `--nms` to include non‑maximum suppression inside the model.  Including
    NMS simplifies the Swift integration but reduces flexibility; leaving NMS
    external lets you tune thresholds in Swift.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained YOLO model to Core ML format.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to a trained YOLO weights file (.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (height/width) for the exported model",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 quantization for smaller model size and faster inference",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include built‑in non‑maximum suppression in the exported model",
    )
    return parser.parse_args()


def export_model(cfg: argparse.Namespace) -> None:
    weight_path = Path(cfg.weights).resolve()
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weight_path}")

    # Load the trained model
    model = YOLO(str(weight_path))

    # Determine export options
    export_kwargs = {
        "format": "coreml",
        "imgsz": cfg.imgsz,
        "nms": cfg.nms,
    }
    if cfg.int8:
        # Provide a representative dataset path for quantization if available
        # Here we leave it None; Ultralytics will use a small random sample.
        export_kwargs["int8"] = True
    
    # Perform export
    print(f"Exporting model {weight_path} to Core ML…")
    coreml_path = model.export(**export_kwargs)
    print(f"Core ML model saved at {coreml_path}")


if __name__ == "__main__":
    args = parse_args()
    export_model(args)