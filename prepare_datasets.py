"""
prepare_datasets_clean.py
==========================

This script extracts bounding boxes from four road‑scene datasets (LISA Traffic Sign,
Bosch Small Traffic Lights Dataset (BSTLD), Mapillary Traffic Sign Dataset (MTSD), and
BDD100K) and writes them into the YOLO format expected by Ultralytics.

The implementation here is intentionally simple and tailored to the directory structures
shown in the provided screenshots.  It avoids the heuristics and wildcards of the
previous script: each dataset is processed independently using explicit paths and
parsers for the known annotation formats.  Adjust the `RAW_BASE` and `OUT_BASE`
constants below if your folder names differ.

For each dataset, images and corresponding label files are copied into
``datasets/traffic/images/{split}`` and ``datasets/traffic/labels/{split}``, where
``split`` is one of ``train``, ``val``, or ``test``.  LISA does not provide an
official split, so all of its images are placed into the ``train`` split by default.

Usage::

    python prepare_datasets_clean.py

Ensure that the following folder structure exists before running the script::

    datasets/
      raw/
        bdd100k/
          images/100k/{train,val,test}            # BDD100K images
          labels/det_20/{train,val}.json          # optional aggregated detection labels
          labels/{train,val,test}/*.json          # per‑image detection labels
        bstld/
          train/img/                             # images
          train/ann/                             # annotation files (.png.json)
          test/img/
          test/ann/
        lisa/
          daySequence*/dayClip*/frames/*.jpg     # images
          dayTrain/dayClip*/frames/*.jpg
          nightSequence*/nightClip*/frames/*.jpg
          nightTrain/nightClip*/frames/*.jpg
          **/frameAnnotationsBOX.csv             # bounding boxes
          **/frameAnnotationsBULB.csv            # optional second pass
        mtsd/
          mtsd_fully_annotated_annotation/
            mtsd_v2_fully_annotated/
              annotations/*.json                # per‑image annotations
              splits/{train,val,test}.txt        # lists of IDs
          mtsd_fully_annotated_images.train.*/*   # image folders
          mtsd_fully_annotated_images.val/
          mtsd_fully_annotated_images.test/

After running the script, ``datasets/traffic/`` will contain populated
``images`` and ``labels`` folders ready for training.

"""

import os
import json
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Location of the raw datasets.  Update this if your raw datasets live
# elsewhere.  The folder should contain subfolders named ``bdd100k``,
# ``bstld``, ``lisa``, and ``mtsd`` with the structures shown in the
# module docstring.
RAW_BASE = Path('datasets/raw')

# Location where the YOLO‑formatted dataset will be written.  The script
# creates ``images/{train,val,test}`` and ``labels/{train,val,test}``
# subdirectories here.
OUT_BASE = Path('datasets/traffic')

# Ordered list of class names.  These names are used to map category
# strings in the raw annotations to integer indices (0‑based).  The
# ordering must match the one used in your YOLO training configuration
# (traffic.yaml).  Feel free to add or remove classes as needed.
CLASSES: List[str] = [
    'speed_sign',      # generic speed limit sign
    'stop',            # stop sign
    'yield',           # yield sign
    'no_entry',        # do‑not‑enter sign
    'other_sign',      # fallback for miscellaneous signs
    'traffic_light_red',
    'traffic_light_yellow',
    'traffic_light_green',
    'car',
    'bus',
    'truck',
    'motorcycle',
    'bicycle',
    'pedestrian',
]

# Map LISA annotation tags to class names.  These mappings are case‑sensitive;
# additional prefixes are handled in code below.
LISA_MAP: Dict[str, str] = {
    'stop': 'stop',
    'yield': 'yield',
    'doNotEnter': 'no_entry',
    'do_not_enter': 'no_entry',
    'warning': 'other_sign',
    # Generic mapping for speed signs; see logic in convert_lisa
}

# Map BSTLD class titles to class names.  Only red/yellow/green are used.
BSTLD_MAP: Dict[str, Optional[str]] = {
    'red': 'traffic_light_red',
    'yellow': 'traffic_light_yellow',
    'green': 'traffic_light_green',
    'off': None,  # ignore unlit lights
}

# Map BDD100K categories to class names.  Unknown categories are ignored.
BDD100K_MAP: Dict[str, Optional[str]] = {
    'traffic sign': 'other_sign',
    'traffic light': None,  # BDD labels lights but not their state
    'car': 'car',
    'bus': 'bus',
    'truck': 'truck',
    'train': None,
    'motorcycle': 'motorcycle',
    'motor': 'motorcycle',
    'bike': 'bicycle',
    'bicycle': 'bicycle',
    'person': 'pedestrian',
    'pedestrian': 'pedestrian',
    'rider': 'motorcycle',
}

# Heuristic mapping for MTSD labels.  Some codes like
# ``regulatory--maximum-speed-limit-50--g2`` are mapped via substring checks.
def map_mtsd_label(label: str) -> Optional[str]:
    """Map a raw MTSD label string to one of the defined class names.

    If the label encodes a speed limit, stop, yield, or do‑not‑enter sign,
    return the corresponding class name.  Otherwise return 'other_sign'.
    """
    low = label.lower()
    # Speed limits contain 'speed' or 'maximum-speed-limit'
    if 'speed' in low or 'maximum-speed-limit' in low:
        return 'speed_sign'
    if 'stop' in low:
        return 'stop'
    if 'yield' in low:
        return 'yield'
    if 'do-not-enter' in low or 'do_not_enter' in low or 'no-entry' in low:
        return 'no_entry'
    # Could extend with more mappings here
    return 'other_sign'


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def ensure_dirs(base: Path) -> None:
    """Create images/labels subdirectories for train, val and test splits."""
    for split in ['train', 'val', 'test']:
        (base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base / 'labels' / split).mkdir(parents=True, exist_ok=True)


def write_yolo_label(file_path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    """Write YOLO bounding boxes to a file.  Boxes are (cls, x_c, y_c, w, h)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w') as f:
        for cls_idx, x_c, y_c, bw, bh in boxes:
            f.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")


def copy_image(src: Path, dst: Path) -> None:
    """Copy an image from src to dst, creating any necessary directories."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ----------------------------------------------------------------------------
# Conversion functions
# ----------------------------------------------------------------------------

def convert_lisa(raw_dir: Path, out_dir: Path, seed: int = 42) -> None:
    """Convert the LISA Traffic Sign dataset to YOLO format.

    This version supports an optional random split: all images are first
    collected with their annotations, then assigned to train/val/test
    splits using an 80/10/10 ratio.  If you prefer to place all
    samples in a single split, set the random seed to ``None`` and all
    files will default to ``train``.

    The function walks through every CSV file matching ``frameAnnotations*.csv``
    under ``raw_dir``.  Each row in the CSV provides a filename, a sign type,
    and bounding box coordinates.  If the sign type starts with
    ``speedLimit`` (case‑insensitive), it is mapped to ``speed_sign``.
    Otherwise the mapping is looked up in ``LISA_MAP``; unknown tags
    become ``other_sign``.
    """
    # Gather all annotations keyed by relative image path
    annotations: Dict[str, List[Tuple[str, int, int, int, int]]] = {}
    csv_files = list(raw_dir.rglob('frameAnnotationsBOX.csv')) + list(raw_dir.rglob('frameAnnotationsBULB.csv'))
    for csv_path in csv_files:
        try:
            with csv_path.open(newline='') as f:
                reader = csv.reader(f, delimiter=';')
                # Skip header if present
                _ = next(reader, None)
                for row in reader:
                    if len(row) < 6:
                        continue
                    fn = row[0].strip()
                    tag = row[1].strip()
                    try:
                        x1 = int(row[2]); y1 = int(row[3]); x2 = int(row[4]); y2 = int(row[5])
                    except Exception:
                        continue
                    annotations.setdefault(fn, []).append((tag, x1, y1, x2, y2))
        except Exception as e:
            print(f"LISA: failed to read {csv_path}: {e}")

    # Determine split for each image
    rel_paths = list(annotations.keys())
    # Shuffle with seed if provided
    if seed is not None:
        import random
        random.Random(seed).shuffle(rel_paths)
    # Compute counts
    total = len(rel_paths)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    # Remainder goes to test
    # Map each relative path to a split
    split_map: Dict[str, str] = {}
    for i, rel in enumerate(rel_paths):
        if seed is None:
            # Everything to train
            split = 'train'
        else:
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
        split_map[rel] = split

    # Copy images and write labels for each split
    for rel_path, boxes in annotations.items():
        img_path = raw_dir / rel_path
        if not img_path.exists():
            # Try to locate by basename within raw_dir
            found = list(raw_dir.rglob(os.path.basename(rel_path)))
            if found:
                img_path = found[0]
            else:
                print(f"LISA: image not found for {rel_path}")
                continue
        # Determine split
        split = split_map.get(rel_path, 'train')
        # Output directories
        images_out = out_dir / 'images' / split
        labels_out = out_dir / 'labels' / split
        # Copy image
        out_img = images_out / img_path.name
        copy_image(img_path, out_img)
        # Image dimensions
        try:
            w, h = Image.open(img_path).size
        except Exception:
            w, h = 1, 1
        yolo_boxes: List[Tuple[int, float, float, float, float]] = []
        for tag, x1, y1, x2, y2 in boxes:
            # Determine class
            tag_low = tag.lower()
            if tag_low.startswith('speedlimit'):
                cls_name = 'speed_sign'
            elif tag_low.startswith('donotenter') or tag_low.startswith('do_not_enter'):
                cls_name = 'no_entry'
            elif tag_low.startswith('yield'):
                cls_name = 'yield'
            elif tag_low.startswith('stop'):
                cls_name = 'stop'
            else:
                cls_name = LISA_MAP.get(tag, 'other_sign')
            if cls_name is None or cls_name not in CLASSES:
                continue
            cls_idx = CLASSES.index(cls_name)
            # Normalise
            x_c = ((x1 + x2) / 2) / w
            y_c = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_boxes.append((cls_idx, x_c, y_c, bw, bh))
        # Write label file
        lbl_path = labels_out / (img_path.stem + '.txt')
        write_yolo_label(lbl_path, yolo_boxes)


def convert_bstld(raw_dir: Path, out_dir: Path) -> None:
    """Convert the BSTLD dataset to YOLO format.

    Processes the ``train`` and ``test`` splits in the directory structure
    ``bstld/train/img``, ``bstld/train/ann`` and similarly for ``test``.  The
    images are copied into ``train`` and ``test`` splits respectively; BSTLD
    does not contain a validation split.  Each annotation JSON (``*.png.json``
    or ``*.jpg.json``) contains a list of ``objects`` with a ``classTitle``
    (e.g. ``red``, ``green``, ``yellow``) and bounding box coordinates under
    ``points`` → ``exterior``.
    """
    for split in ['train', 'test']:
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        img_dir = split_dir / 'img'
        ann_dir = split_dir / 'ann'
        out_img_base = out_dir / 'images' / split
        out_lbl_base = out_dir / 'labels' / split
        for img_path in img_dir.glob('*.*'):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            # Find corresponding annotation file; try <name>.<ext>.json then <name>.json
            ann_path = ann_dir / (img_path.name + '.json')
            if not ann_path.exists():
                ann_path = ann_dir / (img_path.stem + '.json')
            # Copy image
            out_img = out_img_base / img_path.name
            copy_image(img_path, out_img)
            # Read annotation
            yolo_boxes: List[Tuple[int, float, float, float, float]] = []
            if ann_path.exists():
                try:
                    data = json.load(ann_path.open())
                except Exception as e:
                    print(f"BSTLD: failed to read {ann_path}: {e}")
                    data = {}
                objects = data.get('objects', [])
                # Image dimensions
                try:
                    w, h = Image.open(img_path).size
                except Exception:
                    w, h = 1, 1
                for obj in objects:
                    state = obj.get('classTitle')  # 'red', 'yellow', 'green', 'off'
                    cls_name = BSTLD_MAP.get(state)
                    if cls_name is None or cls_name not in CLASSES:
                        continue
                    cls_idx = CLASSES.index(cls_name)
                    points = obj.get('points', {}).get('exterior', [])
                    if len(points) != 2:
                        continue
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_c = ((x1 + x2) / 2) / w
                    y_c = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    yolo_boxes.append((cls_idx, x_c, y_c, bw, bh))
            # Write label file
            lbl_path = out_lbl_base / (img_path.stem + '.txt')
            write_yolo_label(lbl_path, yolo_boxes)


def convert_bdd100k(raw_dir: Path, out_dir: Path) -> None:
    """Convert BDD100K detection labels to YOLO format.

    This function reads images from ``images/100k/{train,val,test}`` (or
    ``images/{split}`` as a fallback) and per‑image detection labels from
    ``labels/{split}/*.json``.  Images and labels are written into the
    matching splits (``train``, ``val`` and ``test``).  If no label file
    exists for a particular image, an empty label file is created so
    that the image can still be used for inference or evaluation.
    """
    for split in ['train', 'val', 'test']:
        # Determine image and label directories
        # img_dir = raw_dir / 'images' / '100k' / split
        # if not img_dir.exists():
        img_dir = raw_dir / 'images' / split
        label_dir = raw_dir / 'labels' / split
        # if not img_dir.exists():
        #     print(f"BDD100K: image directory not found for split {split}; skipping.")
        #     continue
        # if not label_dir.exists():
        #     # Even if labels are missing, copy images and create empty labels
        #     print(f"BDD100K: label directory not found for split {split}; images will have empty labels.")
        out_img_base = out_dir / 'images' / split
        out_lbl_base = out_dir / 'labels' / split
        for img_path in img_dir.glob('*.jpg'):
            # Copy image
            out_img = out_img_base / img_path.name
            copy_image(img_path, out_img)
            # Prepare boxes
            yolo_boxes: List[Tuple[int, float, float, float, float]] = []
            # If label directory exists, try to load annotation
            if label_dir.exists():
                json_path = label_dir / f"{img_path.stem}.json"
                if json_path.exists():
                    try:
                        data = json.load(json_path.open())
                    except Exception as e:
                        print(f"BDD100K: failed to read {json_path}: {e}")
                        data = {}
                    # Determine object list: may live under 'labels' or within 'frames'
                    objects = data.get('labels', [])
                    if not objects and 'frames' in data:
                        objects = []
                        for frame in data.get('frames', []):
                            objects.extend(frame.get('objects', []))
                    # Load image size
                    try:
                        w, h = Image.open(img_path).size
                    except Exception:
                        w, h = 1, 1
                    for obj in objects:
                        cat = obj.get('category') or obj.get('label')
                        cls_name = BDD100K_MAP.get(cat)
                        if cls_name is None or cls_name not in CLASSES:
                            continue
                        cls_idx = CLASSES.index(cls_name)
                        box = obj.get('box2d') or obj.get('bbox')
                        if not box:
                            continue
                        x1 = box.get('x1') or box.get('xmin') or box.get('left')
                        y1 = box.get('y1') or box.get('ymin') or box.get('top')
                        x2 = box.get('x2') or box.get('xmax') or box.get('right')
                        y2 = box.get('y2') or box.get('ymax') or box.get('bottom')
                        if None in (x1, y1, x2, y2):
                            continue
                        x_c = ((x1 + x2) / 2) / w
                        y_c = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        yolo_boxes.append((cls_idx, x_c, y_c, bw, bh))
            # Write label file (even if empty)
            lbl_path = out_lbl_base / (img_path.stem + '.txt')
            write_yolo_label(lbl_path, yolo_boxes)


def convert_mtsd(raw_dir: Path, out_dir: Path) -> None:
    """Convert the Mapillary Traffic Sign Dataset (fully annotated) to YOLO format.

    The directory ``raw_dir`` should contain a folder ``mtsd_fully_annotated_annotation``
    with subfolders ``mtsd_v2_fully_annotated/annotations`` (per‑image JSON) and
    ``mtsd_v2_fully_annotated/splits`` (txt files listing IDs for train, val and test).
    It should also contain image folders whose names start with
    ``mtsd_fully_annotated_images``.  This function reads the annotation JSONs,
    maps labels to the defined classes via heuristics, and writes images and
    YOLO labels into the appropriate split.
    """
    # Find annotation folder
    ann_dir = None
    for candidate in raw_dir.rglob('annotations'):
        if candidate.is_dir():
            ann_dir = candidate
            break
    if ann_dir is None:
        print("MTSD: annotations folder not found; skipping.")
        return
    # Load split lists if present
    split_ids: Dict[str, set] = {'train': set(), 'val': set(), 'test': set()}
    splits_dir = ann_dir.parent / 'splits'
    if splits_dir.exists():
        for split in ['train', 'val', 'test']:
            split_file = splits_dir / f'{split}.txt'
            if split_file.exists():
                with split_file.open() as f:
                    split_ids[split] = {ln.strip() for ln in f if ln.strip()}
    # Preindex images by ID (stem)
    image_lookup: Dict[str, Path] = {}
    for img_root in raw_dir.glob('mtsd_fully_annotated_images*'):
        for img_path in img_root.rglob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            image_lookup[img_path.stem] = img_path
    # Process each annotation file
    for ann_path in ann_dir.glob('*.json'):
        img_id = ann_path.stem
        # Determine split
        target_split = 'train'
        for split in ['train', 'val', 'test']:
            if split_ids[split] and img_id in split_ids[split]:
                target_split = split
                break
        # Load annotation
        try:
            data = json.load(ann_path.open())
        except Exception as e:
            print(f"MTSD: failed to read {ann_path}: {e}")
            continue
        # Find image path
        img_path = image_lookup.get(img_id)
        if img_path is None or not img_path.exists():
            # fallback: some JSONs include 'image' field
            ref = data.get('image') or data.get('filename') or data.get('file')
            if ref:
                possible = raw_dir / ref
                if possible.exists():
                    img_path = possible
                else:
                    # search by basename
                    name_only = os.path.basename(ref)
                    for p in raw_dir.rglob(name_only):
                        img_path = p
                        break
        if img_path is None or not img_path.exists():
            print(f"MTSD: image file for {img_id} not found; skipping.")
            continue
        # Copy image
        out_img = out_dir / 'images' / target_split / img_path.name
        copy_image(img_path, out_img)
        # Prepare boxes
        try:
            w, h = Image.open(img_path).size
        except Exception:
            w, h = 1, 1
        yolo_boxes: List[Tuple[int, float, float, float, float]] = []
        objects = data.get('objects') or data.get('labels') or []
        for obj in objects:
            label = obj.get('label') or obj.get('class') or obj.get('sign') or obj.get('category')
            if not label:
                continue
            cls_name = map_mtsd_label(label)
            if cls_name is None or cls_name not in CLASSES:
                continue
            cls_idx = CLASSES.index(cls_name)
            bbox = obj.get('bbox') or obj.get('box') or obj.get('bounds') or obj.get('rectangle')
            if not bbox:
                continue
            # Handle dict or list bboxes
            if isinstance(bbox, dict):
                x1 = bbox.get('x1') or bbox.get('xmin') or bbox.get('left')
                y1 = bbox.get('y1') or bbox.get('ymin') or bbox.get('top')
                x2 = bbox.get('x2') or bbox.get('xmax') or bbox.get('right')
                y2 = bbox.get('y2') or bbox.get('ymax') or bbox.get('bottom')
            elif isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                continue
            if None in (x1, y1, x2, y2):
                continue
            x_c = ((x1 + x2) / 2) / w
            y_c = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_boxes.append((cls_idx, x_c, y_c, bw, bh))
        # Write label file
        lbl_path = out_dir / 'labels' / target_split / (img_path.stem + '.txt')
        write_yolo_label(lbl_path, yolo_boxes)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    ensure_dirs(OUT_BASE)
    # Convert each dataset
    lisa_dir = RAW_BASE / 'lisa'
    if lisa_dir.exists():
        print('Converting LISA...')
        convert_lisa(lisa_dir, OUT_BASE)
    else:
        print('LISA dataset not found; skipping.')
    bstld_dir = RAW_BASE / 'bstld'
    if bstld_dir.exists():
        print('Converting BSTLD...')
        convert_bstld(bstld_dir, OUT_BASE)
    else:
        print('BSTLD dataset not found; skipping.')
    bdd_dir = RAW_BASE / 'bdd100k'
    if bdd_dir.exists():
        print('Converting BDD100K...')
        convert_bdd100k(bdd_dir, OUT_BASE)
    else:
        print('BDD100K dataset not found; skipping.')
    mtsd_dir = RAW_BASE / 'mtsd'
    if mtsd_dir.exists():
        print('Converting MTSD...')
        convert_mtsd(mtsd_dir, OUT_BASE)
    else:
        print('MTSD dataset not found; skipping.')
    print('Conversion finished.')


if __name__ == '__main__':
    main()