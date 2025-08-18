#!/usr/bin/env python3
# coding: utf-8
import os, sys, json, shutil, glob, random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -------- paths (match your screenshots) ----------
ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "datasets" / "raw"
OUT  = ROOT / "datasets" / "traffic"   # detector export
LANE_OUT = ROOT / "datasets" / "lanes" # lane masks export

# your raw structure (from screenshots)
BDD_ROOT  = RAW / "bdd100k"
MTSD_ROOT = RAW / "mtsd"
# LISA/BSTLD intentionally unused for detector now
# LISA_ROOT = RAW / "lisa"
# BSTLD_ROOT = RAW / "bstld"

# -------- unified detector classes (order for YOLO txt indices) ----------
NAMES = [
    "person","bicycle","car","motorcycle","bus","truck",
    "traffic_light_red","traffic_light_yellow","traffic_light_green",
    # Signs (MTSD provides rich classes)
    "stop","yield","no_entry","speed_limit_sign","pedestrian_crossing_sign",
    "no_left_turn","no_right_turn","no_u_turn","one_way","turn_left","turn_right",
    "go_straight","roundabout","keep_right","keep_left","pass_either_side",
    "priority_road","no_parking","no_stopping","height_limit","children_crossing",
    "road_bump","curve_left","curve_right","roadworks","parking_info",
    "bicycles_only","other_sign"
]
NAME2ID = {n:i for i,n in enumerate(NAMES)}

# ---- MTSD label-string -> our class id (compact mapping) -------------
def mtsd_label_to_id(label: str):
    l = label.lower()
    if l.startswith("regulatory--stop"):                    return NAME2ID["stop"]
    if l.startswith("regulatory--yield"):                   return NAME2ID["yield"]
    if l.startswith("regulatory--no-entry"):                return NAME2ID["no_entry"]
    if "maximum-speed-limit" in l:                           return NAME2ID["speed_limit_sign"]
    if l.startswith("warning--pedestrians-crossing") or l.startswith("information--pedestrians-crossing"):
        return NAME2ID["pedestrian_crossing_sign"]
    if l.startswith("regulatory--no-left-turn"):            return NAME2ID["no_left_turn"]
    if l.startswith("regulatory--no-right-turn"):           return NAME2ID["no_right_turn"]
    if l.startswith("regulatory--no-u-turn"):               return NAME2ID["no_u_turn"]
    if "one-way-left" in l or "one-way-right" in l or "one-way-straight" in l:
        return NAME2ID["one_way"]
    if l.startswith("regulatory--turn-left"):               return NAME2ID["turn_left"]
    if l.startswith("regulatory--turn-right"):              return NAME2ID["turn_right"]
    if l.startswith("regulatory--go-straight"):             return NAME2ID["go_straight"]
    if l.startswith("regulatory--roundabout"):              return NAME2ID["roundabout"]
    if l.startswith("regulatory--keep-right") or l.startswith("complementary--go-right"):
        return NAME2ID["keep_right"]
    if l.startswith("regulatory--keep-left") or l.startswith("complementary--go-left"):
        return NAME2ID["keep_left"]
    if l.startswith("regulatory--pass-on-either-side"):     return NAME2ID["pass_either_side"]
    if l.startswith("regulatory--priority-road"):           return NAME2ID["priority_road"]
    if l.startswith("regulatory--no-parking"):              return NAME2ID["no_parking"]
    if l.startswith("regulatory--no-stopping"):             return NAME2ID["no_stopping"]
    if l.startswith("regulatory--height-limit"):            return NAME2ID["height_limit"]
    if l.startswith("warning--children"):                   return NAME2ID["children_crossing"]
    if l.startswith("warning--road-bump"):                  return NAME2ID["road_bump"]
    if l.startswith("warning--curve-left"):                 return NAME2ID["curve_left"]
    if l.startswith("warning--curve-right"):                return NAME2ID["curve_right"]
    if l.startswith("warning--roadworks"):                  return NAME2ID["roadworks"]
    if l.startswith("information--parking"):                return NAME2ID["parking_info"]
    if l.startswith("regulatory--bicycles-only"):           return NAME2ID["bicycles_only"]
    # fallback
    if "sign" in l:                                         return NAME2ID["other_sign"]
    return None

# ---- helpers ----------------------------------------------------------
def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def yolo_line(cls_id, cx, cy, w, h):
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

def norm_bbox(x1,y1,x2,y2, W,H):
    # clip and normalize
    x1 = max(0.0, min(float(x1), W-1))
    y1 = max(0.0, min(float(y1), H-1))
    x2 = max(0.0, min(float(x2), W-1))
    y2 = max(0.0, min(float(y2), H-1))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w  = bw / W
    h  = bh / H
    return cx, cy, w, h

def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# ---- coverage by scanning exported labels (robust) --------------------
def compute_coverage(lbl_root: Path):
    counts = [0]*len(NAMES)
    for split in ("train","val","test"):
        for p in (lbl_root/split).glob("*.txt"):
            with open(p, "r") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        cid = int(line.split()[0])
                        if 0 <= cid < len(counts):
                            counts[cid]+=1
                    except Exception:
                        pass
    return counts

# ===================== MTSD (SIGNS) ======================
def find_mtsd_paths():
    # annotations
    ann_dirs = sorted(glob.glob(str(MTSD_ROOT / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated" / "annotations")))
    if not ann_dirs:
        # accept any *annotated_annotation*/mtsd_v2_fully_annotated/annotations
        ann_dirs = sorted(glob.glob(str(MTSD_ROOT / "*annotated*_annotation" / "mtsd_v2_fully_annotated" / "annotations")))
    ann_dir = Path(ann_dirs[0]) if ann_dirs else None

    # images are split across train.0/1/2, val, test (as in your screenshot)
    img_dirs = []
    for pat in [
        "mtsd_fully_annotated_images.train.0",
        "mtsd_fully_annotated_images.train.1",
        "mtsd_fully_annotated_images.train.2",
        "mtsd_fully_annotated_images.val",
        "mtsd_fully_annotated_images.test",
    ]:
        p = MTSD_ROOT / pat
        if p.is_dir():
            img_dirs.append(p)
    return ann_dir, img_dirs

def build_mtsd_image_index(img_dirs):
    # map 'filename.ext' -> full path
    idx = {}
    for d in img_dirs:
        for ext in ("*.jpg","*.jpeg","*.png"):
            for p in d.rglob(ext):
                idx[p.name] = p
    return idx

def convert_mtsd_signs():
    ann_dir, img_dirs = find_mtsd_paths()
    if not ann_dir or not img_dirs:
        print(f"MTSD: expected {MTSD_ROOT}/.../annotations and image dirs train.0/1/2,val,test, skipping.")
        return

    print("Converting MTSD (signs)...")
    img_index = build_mtsd_image_index(img_dirs)

    # use provided splits if present
    splits_dir = ann_dir.parent / "splits"
    split_files = {
        "train": splits_dir / "train.txt",
        "val":   splits_dir / "val.txt",
        "test":  splits_dir / "test.txt",
    }

    for split, split_file in split_files.items():
        if not split_file.exists():  # if no split files, just skip MTSD (avoid guessing)
            print(f"MTSD: missing split file {split_file}, skipping this split.")
            continue

        out_img_dir = OUT / "images" / split
        out_lbl_dir = OUT / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        with open(split_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        processed = 0
        for rel in lines:
            # 'rel' is like 'images/train/xxx.jpg' or similar; use basename to locate both image and json
            img_name = os.path.basename(rel)
            img_path = img_index.get(img_name)
            if not img_path:
                continue

            # the MTSD annotation json usually shares basename with image
            json_name = Path(img_name).with_suffix(".json").name
            json_path = ann_dir / json_name
            if not json_path.exists():
                # some zips may prefix names; try glob by stem
                stem = Path(img_name).stem
                candidates = list(ann_dir.glob(f"*{stem}*.json"))
                if candidates:
                    json_path = candidates[0]
                else:
                    continue

            try:
                data = json.loads(Path(json_path).read_text())
            except Exception:
                continue

            # image size (some MTSD jsons might not include W/H reliably, so read the image)
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue

            yolo_lines = []
            # MTSD bbox style: each object -> { "label": "<string>", "bbox": [x, y, w, h] }  (varies; fallback to x1,y1,x2,y2 keys)
            objects = data.get("objects") or data.get("labels") or []
            for obj in objects:
                label = obj.get("label") or obj.get("class") or obj.get("category") or ""
                cid = mtsd_label_to_id(label)
                if cid is None:
                    continue
                bbox = obj.get("bbox") or obj.get("box") or None
                if bbox and len(bbox) >= 4:
                    # bbox as [x,y,w,h] or [x1,y1,w,h]
                    x = float(bbox[0]); y = float(bbox[1]); w = float(bbox[2]); h = float(bbox[3])
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                else:
                    # try x1,y1,x2,y2 keys
                    x1 = obj.get("x1", obj.get("xmin")); y1 = obj.get("y1", obj.get("ymin"))
                    x2 = obj.get("x2", obj.get("xmax")); y2 = obj.get("y2", obj.get("ymax"))
                    if None in (x1,y1,x2,y2): 
                        continue
                cx, cy, nw, nh = norm_bbox(x1,y1,x2,y2,W,H)
                yolo_lines.append(yolo_line(cid, cx, cy, nw, nh))

            # copy & write
            if yolo_lines:
                copy_image(img_path, out_img_dir / img_name)
                (out_lbl_dir / (Path(img_name).stem + ".txt")).write_text("".join(yolo_lines))
                processed += 1

        print(f"MTSD {split}: wrote {processed} labeled images.")

# ===================== BDD100K (USERS + TLIGHTS + LANES) ======================
# category mapping
BDD_CAT2DET = {
    "person": "person",
    "rider": "person",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "car": "car",
}

def bdd_light_to_id(color: str):
    c = (color or "").lower()
    if c == "red":    return NAME2ID["traffic_light_red"]
    if c == "yellow": return NAME2ID["traffic_light_yellow"]
    if c == "green":  return NAME2ID["traffic_light_green"]
    return None

def parse_bdd_image_labels(obj):
    """Handles either per-image JSON (with 'labels') or tracking JSON (with 'frames'). Returns (img_name, W,H, list of [cid,x1,y1,x2,y2])"""
    # A) per-image JSON (detection 2020): keys often include "name" and "labels"
    if "labels" in obj:
        img_name = obj.get("name")
        W = obj.get("attributes", {}).get("width") or None
        H = obj.get("attributes", {}).get("height") or None
        dets = []
        for lab in obj["labels"]:
            cat = lab.get("category")
            if cat == "traffic light":
                color = lab.get("attributes", {}).get("trafficLightColor", "none")
                cid = bdd_light_to_id(color)
                if cid is None: 
                    continue
            elif cat == "traffic sign":
                # BDD does not give sign type => bucket to other_sign
                cid = NAME2ID["other_sign"]
            else:
                mapped = BDD_CAT2DET.get(cat)
                if not mapped: 
                    continue
                cid = NAME2ID[mapped]
            box = lab.get("box2d")
            if not box: 
                continue
            x1,y1,x2,y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            dets.append((cid, x1,y1,x2,y2))
        return img_name, W, H, dets

    # B) tracking-style JSON (with 'frames'): use the single image name from top-level and take first frame's objects
    if "frames" in obj and obj.get("frames"):
        img_name = obj.get("name")
        dets = []
        W = H = None
        for lab in obj["frames"][0].get("objects", []):
            cat = lab.get("category")
            if cat == "traffic light":
                color = lab.get("attributes", {}).get("trafficLightColor", "none")
                cid = bdd_light_to_id(color)
                if cid is None:
                    continue
            elif cat == "traffic sign":
                cid = NAME2ID["other_sign"]
            else:
                mapped = BDD_CAT2DET.get(cat)
                if not mapped:
                    continue
                cid = NAME2ID[mapped]
            box = lab.get("box2d")
            if not box:
                continue
            x1,y1,x2,y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            dets.append((cid, x1,y1,x2,y2))
        return img_name, W, H, dets

    return None, None, None, []

def convert_bdd_detector():
    print("Converting BDD100K (road users + traffic lights)...")
    for split in ("train","val","test"):
        lbl_dir = BDD_ROOT / "labels" / split
        img_dir = BDD_ROOT / "images" / split
        if not (lbl_dir.is_dir() and img_dir.is_dir()):
            print(f"BDD100K: expected {lbl_dir} and {img_dir}, skipping {split}.")
            continue

        out_img_dir = OUT / "images" / split
        out_lbl_dir = OUT / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        files = list(lbl_dir.glob("*.json"))
        processed = 0
        for i, jp in enumerate(files, 1):
            try:
                obj = json.loads(jp.read_text())
            except Exception:
                continue

            img_name, W, H, dets = parse_bdd_image_labels(obj)
            if not img_name or not dets:
                continue

            # infer image path (jpg or png)
            for ext in (".jpg",".png",".jpeg"):
                candidate = img_dir / (img_name + ext if not img_name.lower().endswith(ext) else img_name)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                # sometimes 'name' already includes extension
                candidate = img_dir / img_name
                if candidate.exists():
                    img_path = candidate
                else:
                    continue

            if W is None or H is None:
                try:
                    with Image.open(img_path) as im:
                        W, H = im.size
                except Exception:
                    continue

            yolo_lines = []
            for cid,x1,y1,x2,y2 in dets:
                cx,cy,w,h = norm_bbox(x1,y1,x2,y2,W,H)
                yolo_lines.append(yolo_line(cid,cx,cy,w,h))

            if yolo_lines:
                copy_image(img_path, out_img_dir / Path(img_path).name)
                (out_lbl_dir / (Path(img_path).stem + ".txt")).write_text("".join(yolo_lines))
                processed += 1

            if i % 5000 == 0:
                print(f"BDD100K {split}: processed {i} annotations...")

        print(f"BDD100K {split}: wrote {processed} labeled images.")

# ---- BDD lanes (rasterize poly2d to mask PNG) ------------------------
LANE_CLASSES = [
    "lane/single white","lane/single yellow","lane/double white","lane/double yellow",
    "lane/road curb","lane/crosswalk","lane/single other","lane/double other"
]
LANE_COLOR = 255  # binary mask (1 class). If you want multi-class, map per type.

def export_bdd_lanes():
    print("Exporting BDD100K lane masks...")
    for split in ("train","val","test"):
        lbl_dir = BDD_ROOT / "labels" / split
        img_dir = BDD_ROOT / "images" / split
        if not (lbl_dir.is_dir() and img_dir.is_dir()):
            print(f"LANES {split}: expected {lbl_dir} and {img_dir}, skipping.")
            continue

        out_mask_dir = LANE_OUT / split
        ensure_clean_dir(out_mask_dir)

        files = list(lbl_dir.glob("*.json"))
        written = 0
        for i, jp in enumerate(files, 1):
            try:
                obj = json.loads(jp.read_text())
            except Exception:
                continue

            # we need the image name regardless of per-image or frames style
            img_name = obj.get("name")
            if not img_name:
                continue

            # find the corresponding image to get size
            img_path = None
            for ext in (".jpg",".png",".jpeg"):
                candidate = img_dir / (img_name + ext if not img_name.lower().endswith(ext) else img_name)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                candidate = img_dir / img_name
                if candidate.exists():
                    img_path = candidate
                else:
                    continue

            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue

            mask = Image.new("L", (W,H), 0)
            draw = ImageDraw.Draw(mask)

            # gather polylines from either 'labels' (per-image) or first frame in 'frames'
            polys = []
            if "labels" in obj:
                for lab in obj["labels"]:
                    if lab.get("category") in LANE_CLASSES and "poly2d" in lab:
                        polys.extend(lab["poly2d"])
            elif obj.get("frames"):
                for lab in obj["frames"][0].get("objects", []):
                    if lab.get("category") in LANE_CLASSES and "poly2d" in lab:
                        polys.extend(lab["poly2d"])

            # poly2d format: each polygon is a list like [ [x, y, 'L' or 'C'], ... ]
            for poly in polys:
                pts = []
                for item in poly:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        x, y = float(item[0]), float(item[1])
                        pts.append((x,y))
                if len(pts) >= 2:
                    draw.line(pts, fill=LANE_COLOR, width=3)

            mask.save(out_mask_dir / (Path(img_path).stem + ".png"))
            written += 1

            if i % 2000 == 0:
                print(f"LANES {split}: exported {i} masks...")

        print(f"LANES {split}: exported {written} masks.")

# ===================== main ======================
def main():
    # clean detector output dirs
    ensure_clean_dir(OUT / "images" / "train")
    ensure_clean_dir(OUT / "images" / "val")
    ensure_clean_dir(OUT / "images" / "test")
    ensure_clean_dir(OUT / "labels" / "train")
    ensure_clean_dir(OUT / "labels" / "val")
    ensure_clean_dir(OUT / "labels" / "test")

    # MTSD (signs)
    convert_mtsd_signs()

    # BDD road users + traffic lights
    convert_bdd_detector()

    # BDD lanes
    export_bdd_lanes()

    # coverage
    print("\nCoverage (instances per class) by scanning exported labels:")
    counts = compute_coverage(OUT / "labels")
    for i, n in enumerate(NAMES):
        print(f"{i:2d} {n:25s}: {counts[i]}")

if __name__ == "__main__":
    main()