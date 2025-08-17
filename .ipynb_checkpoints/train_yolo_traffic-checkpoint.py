"""
train_yolo_traffic.py
======================

This script trains an object‑detection model based on Ultralytics' YOLO implementation.  It
targets the set of features required for a driving assistant application: U.S. traffic
signs and signals, traffic lights with state, road users (pedestrians, cyclists and
vehicles), and optionally speed‑limit reading via OCR.  The script uses transfer
learning from a pre‑trained lightweight YOLO architecture (e.g. `yolov8s.pt`) and
supports multi‑dataset training by merging annotations from BDD100K, LISA Traffic Sign,
Mapillary Traffic Sign Dataset (MTSD) and the Bosch Small Traffic Lights Dataset (BSTLD).

Datasets
---------
* **BDD100K** provides bounding boxes for common road objects (car, bus, truck, person,
  bicycle, motorbike) and also includes lane and drivable area labels【222217897337216†L62-L92】.  The dataset
  comprises 100k video keyframes captured in diverse conditions across the U.S.  You
  can download the detection annotations from the official site and convert them to
  YOLO format.
* **LISA Traffic Sign** contains 47 U.S. sign types (stop, yield, speed‑limit, no
  entry, etc.) with 7 855 annotated signs in 6 610 frames【62780691709535†L90-L101】.  It ensures good
  coverage of U.S. signs and is critical for your application.
* **Mapillary Traffic Sign Dataset (MTSD)** includes around 100 000 high‑resolution
  images with over 300 sign classes and provides 257 543 manually annotated traffic
  signs【334726012655485†L39-L46】【334726012655485†L110-L116】.  Using MTSD improves generalization to
  varied sign shapes and appearances.
* **Bosch Small Traffic Lights Dataset (BSTLD)** offers RGB images with 10 756
  annotated traffic lights.  Lights are labeled with states (red, yellow, green, off) and the
  dataset provides bounding boxes even for tiny lights as small as 1 × 1 pixel【119639528357021†L88-L110】.

To train on these datasets you should first convert each annotation set to the YOLO
bounding‑box format (one text file per image with `class x_center y_center width height`)
and then define a combined dataset configuration YAML (see below).  Several open
source tools are available to assist with conversion (e.g. `roboflow`, `fiftyone` or
custom scripts).

Usage
-----
1. Install the Ultralytics package (v8 or higher) and any dependencies:

```bash
pip install ultralytics
```

2. Prepare your dataset directory structure as follows:

```
datasets/
  traffic/
    images/
      train/        # all training images (combined from BDD100K, LISA, MTSD, BSTLD)
      val/          # validation images
    labels/
      train/        # YOLO text label files for training images
      val/          # YOLO text label files for validation images
  traffic.yaml      # dataset configuration (see below)
```

3. Update the `DATASET_CONFIG` path in this script to point to your `traffic.yaml`.

4. Run the training:

```bash
python train_yolo_traffic.py --weights yolov8s.pt --epochs 50 --imgsz 640 --batch 16
```

After training, the best model will be saved in the `runs/detect/` directory.  You can then
export it to Core ML using the provided `export_coreml.py` script.

Notes
-----
* A small model variant (e.g. `yolov8s.pt`) is recommended to achieve real‑time
  performance on iPhone 13 Pro and above.  Ultralytics provides several model
  checkpoints (n, s, m, l, x) trading off speed and accuracy.  Starting with
  `yolov8s` yields good accuracy while maintaining low latency.
* If your GPU has enough memory you can increase the input resolution (`--imgsz`)
  to improve small‑object detection; however inference on mobile devices will
  typically run at 640×640 or 512×512.
* Training on a merged dataset may require class re‑mapping (e.g. unify various
  speed‑limit sign IDs into a single class).  The `CLASSES` list in the
  dataset YAML should reflect the union of all classes used in your project.

"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for a driving assistant.")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/traffic/traffic.yaml",
        help="Path to dataset config YAML file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8s.pt",
        help="Pre‑trained weights checkpoint (e.g. yolov8s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square). Higher values may improve small object recall but slow down inference",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size per GPU/CPU during training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (e.g. 0,1,2,3 or 'cpu' or 'auto')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8s_driving",
        help="Run name for saving results",
    )
    return parser.parse_args()


def train_model(cfg: argparse.Namespace) -> None:
    """Load a YOLO model, set up training parameters and run training."""
    # Resolve dataset YAML path
    data_path = Path(cfg.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {data_path}")

    # Load the model from pre‑trained weights.  Ultralytics automatically
    # downloads the file if it does not exist locally.
    model = YOLO(cfg.weights)

    # Train the model.  See https://docs.ultralytics.com/modes/train/ for full options.
    model.train(
        data=str(data_path),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        name=cfg.name,
        project="runs/detect",
        close_mosaic=15,  # disable mosaic augmentation in last epochs for better fine‑tuning
        patience=10,       # early stopping if no improvement
    )

    # Evaluate on the validation set and print metrics
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)