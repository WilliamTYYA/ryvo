#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_datasets.py  (COPY images, no symlinks)

Build a unified YOLO detection dataset from:
  - MTSD (traffic signs)
  - BDD100K (road users + traffic-light states)

Outputs:
  datasets/traffic/
    images/{train,val,test}/*.jpg|.png
    labels/{train,val,test}/*.txt   # YOLO cls cx cy w h

Optional:
  --export_lanes datasets/lanes  # exports lane masks from BDD100K for later seg model
  --clean                        # removes previous images/ and labels/ under --out first

Usage:
  conda activate automind
  python prepare_datasets.py \
    --mtsd datasets/raw/mtsd \
    --bdd  datasets/raw/bdd100k \
    --out  datasets/traffic \
    --export_lanes datasets/lanes   # optional
"""

import argparse, json, os, re, shutil, sys, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

random.seed(0)

# -------------------- Classes (ORDER MUST MATCH traffic.yaml) --------------------
CLASSES = [
    # road users (BDD100K)
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    # traffic light states (BDD100K)
    "traffic_light_red", "traffic_light_yellow", "traffic_light_green",
    # traffic signs (MTSD)
    "stop", "yield", "no_entry", "speed_limit_sign",
    "pedestrian_crossing_sign",
    "no_left_turn", "no_right_turn", "no_u_turn",
    "one_way", "turn_left", "turn_right", "go_straight",
    "roundabout", "keep_right", "keep_left", "pass_either_side",
    "priority_road", "no_parking", "no_stopping", "height_limit",
    "children_crossing", "road_bump", "curve_left", "curve_right",
    "roadworks", "parking_info", "bicycles_only",
    "other_sign"
]
NAME2ID = {n:i for i,n in enumerate(CLASSES)}

# -------------------- FS helpers --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def write_yolo_txt(txt_path: Path, recs: List[Tuple[int,float,float,float,float]]):
    ensure_dir(txt_path.parent)
    if not recs:
        txt_path.write_text("")  # YOLO is fine with empty label files
        return
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cid,cx,cy,w,h) in recs]
    txt_path.write_text("\n".join(lines))

def norm_bbox(x1, y1, x2, y2, W, H):
    # clip & sanitize
    x1 = max(0.0, min(float(x1), W - 1))
    y1 = max(0.0, min(float(y1), H - 1))
    x2 = max(0.0, min(float(x2), W - 1))
    y2 = max(0.0, min(float(y2), H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    if w <= 0 or h <= 0:
        return None
    return (cx, cy, w, h)

# -------------------- MTSD label mapping --------------------
def map_mtsd_label(label: str) -> Optional[int]:
    s = (label or "").lower()

    if "regulatory--stop" in s: return NAME2ID["stop"]
    if "regulatory--yield" in s: return NAME2ID["yield"]
    if "regulatory--no-entry" in s: return NAME2ID["no_entry"]

    # Speed limits (collapse variants)
    if "regulatory--maximum-speed-limit" in s or "regulatory--speed-limit" in s:
        return NAME2ID["speed_limit_sign"]

    # Pedestrian crossing
    if "pedestrians-crossing" in s: return NAME2ID["pedestrian_crossing_sign"]

    # No turns
    if "regulatory--no-left-turn" in s:  return NAME2ID["no_left_turn"]
    if "regulatory--no-right-turn" in s: return NAME2ID["no_right_turn"]
    if "regulatory--no-u-turn" in s:     return NAME2ID["no_u_turn"]

    # Directions / one-way
    if "regulatory--one-way" in s:                     return NAME2ID["one_way"]
    if "regulatory--turn-left" in s or "go-left" in s: return NAME2ID["turn_left"]
    if "regulatory--turn-right" in s or "go-right" in s: return NAME2ID["turn_right"]
    if "regulatory--go-straight" in s or "one-way-straight" in s: return NAME2ID["go_straight"]

    # Keep / pass / priority / roundabout
    if "keep-right" in s:  return NAME2ID["keep_right"]
    if "keep-left" in s:   return NAME2ID["keep_left"]
    if "pass-on-either-side" in s: return NAME2ID["pass_either_side"]
    if "priority-road" in s: return NAME2ID["priority_road"]
    if "roundabout" in s:    return NAME2ID["roundabout"]

    # Parking family
    if "information--parking" in s: return NAME2ID["parking_info"]
    if "regulatory--no-parking" in s: return NAME2ID["no_parking"]
    if "regulatory--no-stopping" in s: return NAME2ID["no_stopping"]

    # Warnings / restrictions
    if "height-limit" in s:       return NAME2ID["height_limit"]
    if "children" in s:           return NAME2ID["children_crossing"]
    if "road-bump" in s or "speed-bump" in s or "hump" in s: return NAME2ID["road_bump"]
    if "curve-left" in s:         return NAME2ID["curve_left"]
    if "curve-right" in s:        return NAME2ID["curve_right"]
    if "roadworks" in s or "construction" in s: return NAME2ID["roadworks"]
    if "bicycles-only" in s or "bike-only" in s: return NAME2ID["bicycles_only"]

    if "other-sign" in s: return NAME2ID["other_sign"]
    return None

# -------------------- BDD100K category mapping --------------------
def map_bdd_category(cat: str, attrs: dict) -> Optional[int]:
    c = (cat or "").lower()
    if c == "person": return NAME2ID["person"]
    if c in ("bike","bicycle"): return NAME2ID["bicycle"]
    if c in ("motor","motorcycle","motorbike"): return NAME2ID["motorcycle"]
    if c == "car": return NAME2ID["car"]
    if c == "bus": return NAME2ID["bus"]
    if c == "truck": return NAME2ID["truck"]

    # Traffic lights with color attribute
    if c in ("traffic light","traffic_light","tl","tlight"):
        color = (attrs or {}).get("trafficLightColor","").lower()
        if color == "red":    return NAME2ID["traffic_light_red"]
        if color == "yellow": return NAME2ID["traffic_light_yellow"]
        if color == "green":  return NAME2ID["traffic_light_green"]
        return None

    # Do NOT take BDD traffic sign (we use MTSD for signs)
    return None

# -------------------- Image indexing --------------------
def index_images(root: Path) -> Dict[str, Path]:
    """basename -> full path (covers nested shards in MTSD images)."""
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp"):
            idx[p.name.lower()] = p
    return idx

# -------------------- MTSD ingest --------------------
def ingest_mtsd(mtsd_root: Path, out_root: Path, coverage: Dict[str, Dict[int,int]]):
    ann_dir = mtsd_root/"mtsd_fully_annotated_annotation"/"mtsd_v2_fully_annotated"/"annotations"
    img_root = mtsd_root/"mtsd_fully_annotated_images"
    if not ann_dir.exists() or not img_root.exists():
        print(f"MTSD: expected {ann_dir} and {img_root}, skipping.")
        return

    idx = index_images(img_root)

    def guess_split(ann_path: Path) -> str:
        n = ann_path.name.lower()
        if ".train." in n: return "train"
        if ".val"   in n:  return "val"
        return "train"

    from PIL import Image

    count = 0
    for j in sorted(ann_dir.glob("*.json")):
        try:
            data = json.loads(j.read_text())
        except Exception as e:
            print("MTSD: bad JSON:", j, e); continue

        # Try to locate image file
        cand_names = []
        for k in ("img","image","img_name","name","id","filename","file","path"):
            v = data.get(k)
            if isinstance(v, str) and v.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
                cand_names.append(Path(v).name)
        cand_names += [j.stem + ".jpg", j.stem + ".png", j.stem + ".jpeg"]

        img_path = None
        for nm in cand_names:
            p = idx.get(nm.lower())
            if p: img_path = p; break
        if not img_path:
            # last resort: stem match
            stem = j.stem.lower()
            for k,v in idx.items():
                if Path(k).stem.lower() == stem:
                    img_path = v; break
        if not img_path:
            continue

        try:
            with Image.open(img_path) as im:
                W,H = im.width, im.height
        except:
            continue

        recs = []
        for o in data.get("objects", []):
            label = o.get("label","")
            cid = map_mtsd_label(label)
            if cid is None:
                continue

            x1=y1=x2=y2=None
            if "bbox" in o:
                bb = o["bbox"]
                if isinstance(bb, (list,tuple)) and len(bb)>=4:
                    x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                    x1,y1,x2,y2 = x, y, x+w, y+h
                elif isinstance(bb, dict):
                    x, y, w, h = float(bb.get("x",0)), float(bb.get("y",0)), float(bb.get("w",0)), float(bb.get("h",0))
                    x1,y1,x2,y2 = x, y, x+w, y+h
            if x1 is None:
                poly = o.get("polygon") or o.get("poly") or []
                if isinstance(poly, list) and len(poly)>=3:
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x1 is None:
                continue

            bb = norm_bbox(x1,y1,x2,y2, W,H)
            if bb:
                recs.append((cid,*bb))
                coverage["mtsd"][cid] += 1

        split = guess_split(j)
        out_img = out_root/"images"/split/img_path.name
        out_txt = out_root/"labels"/split/(img_path.stem + ".txt")
        copy_file(img_path, out_img)
        write_yolo_txt(out_txt, recs)

        count += 1
        if count % 2000 == 0:
            print(f"MTSD: processed {count} annotations...")

# -------------------- BDD100K ingest --------------------
def _bdd_find_image(bdd_root: Path, split: str, name: str) -> Optional[Path]:
    # images/<split>/<name>.jpg|.png
    for ext in (".jpg",".jpeg",".png"):
        p = bdd_root/"images"/split/(name + ext)
        if p.exists(): return p
    # tracking-style: images/track/<split>/<name>/000000.ext
    track_dir = bdd_root/"images"/"track"/split/name
    if track_dir.exists():
        for k in range(0, 6):
            for ext in (".jpg",".jpeg",".png"):
                q = track_dir/(f"{k:06d}"+ext)
                if q.exists(): return q
    return None

def ingest_bdd100k(bdd_root: Path, out_root: Path, coverage: Dict[str, Dict[int,int]]):
    from PIL import Image
    for split in ("train","val","test"):
        labels_dir = bdd_root/"labels"/split
        if not labels_dir.exists():
            continue

        count = 0
        for j in labels_dir.glob("*.json"):
            try:
                data = json.loads(j.read_text())
            except Exception as e:
                print("BDD100K: bad JSON:", j, e)
                continue

            name = data.get("name") or j.stem
            img_path = _bdd_find_image(bdd_root, split, name)
            if not img_path:
                continue

            try:
                with Image.open(img_path) as im:
                    W,H = im.width, im.height
            except:
                continue

            recs = []

            if "labels" in data:
                # detection-style
                for o in data["labels"]:
                    cat = o.get("category","")
                    attrs = o.get("attributes", {})
                    cid = map_bdd_category(cat, attrs)
                    if cid is None:
                        continue
                    box = o.get("box2d")
                    if not box:
                        continue
                    x1,y1,x2,y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                    bb = norm_bbox(x1,y1,x2,y2, W,H)
                    if bb:
                        recs.append((cid,*bb))
                        coverage["bdd"][cid] += 1

            elif "frames" in data:
                # tracking-style: pick first frame with usable objects
                for fr in data["frames"]:
                    tmp=[]
                    for o in fr.get("objects", []):
                        cat = o.get("category","")
                        attrs = o.get("attributes", {})
                        cid = map_bdd_category(cat, attrs)
                        if cid is None:
                            continue
                        box = o.get("box2d")
                        if not box:
                            continue
                        x1,y1,x2,y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                        bb = norm_bbox(x1,y1,x2,y2, W,H)
                        if bb:
                            tmp.append((cid,*bb))
                    if tmp:
                        recs = tmp
                        break

            out_img = out_root/"images"/split/img_path.name
            out_txt = out_root/"labels"/split/(img_path.stem + ".txt")
            copy_file(img_path, out_img)
            write_yolo_txt(out_txt, recs)

            count += 1
            if count % 5000 == 0:
                print(f"BDD100K {split}: processed {count} annotations...")

# -------------------- Optional Lane Export (copy) --------------------
def export_bdd_lanes(bdd_root: Path, lanes_out: Path):
    """
    Creates a binary lane mask for each BDD image that has lane polygons.
    Saves images and masks (copies images).
    """
    from PIL import Image, ImageDraw
    for split in ("train","val","test"):
        ensure_dir(lanes_out/"images"/split)
        ensure_dir(lanes_out/"masks"/split)

    def draw_mask(polys, W,H):
        mask = Image.new("L", (W,H), 0)
        dr = ImageDraw.Draw(mask)
        for poly in polys:
            if len(poly) >= 3:
                dr.polygon(poly, outline=255, fill=255)
        return mask

    for split in ("train","val","test"):
        labels_dir = bdd_root/"labels"/split
        if not labels_dir.exists(): continue
        count = 0
        for j in labels_dir.glob("*.json"):
            data = json.loads(j.read_text())
            name = data.get("name") or j.stem
            img_path = _bdd_find_image(bdd_root, split, name)
            if not img_path: continue

            # gather lane polygons
            if "labels" in data:
                objs = data["labels"]
            elif "frames" in data and data["frames"]:
                objs = data["frames"][0].get("objects", [])
            else:
                objs = []

            polys=[]
            for o in objs:
                cat = (o.get("category","") or "").lower()
                if not cat.startswith("lane/"):
                    continue
                for poly in o.get("poly2d", []):
                    pts = [(float(px), float(py)) for (px,py,*_) in poly]
                    polys.append(pts)
            if not polys:
                continue

            from PIL import Image
            with Image.open(img_path) as im:
                W,H = im.width, im.height
            mask = draw_mask(polys, W,H)

            out_img = lanes_out/f"images/{split}"/img_path.name
            out_msk = lanes_out/f"masks/{split}"/(img_path.stem + ".png")
            copy_file(img_path, out_img)
            mask.save(out_msk)

            count += 1
            if count % 2000 == 0:
                print(f"LANES {split}: exported {count} masks...")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtsd", type=str, required=True, help="Path to MTSD root")
    ap.add_argument("--bdd",  type=str, required=True, help="Path to BDD100K root")
    ap.add_argument("--out",  type=str, default="datasets/traffic", help="Output root for YOLO detection")
    ap.add_argument("--export_lanes", type=str, default=None, help="Optional output root for BDD lane masks")
    ap.add_argument("--clean", action="store_true", help="Remove existing images/ and labels/ under --out before writing")
    args = ap.parse_args()

    mtsd_root = Path(args.mtsd)
    bdd_root  = Path(args.bdd)
    out_root  = Path(args.out)

    # Prepare folder tree
    if args.clean:
        for sub in ("images","labels"):
            p = out_root/sub
            if p.exists():
                print(f"Cleaning {p} ...")
                shutil.rmtree(p)
    for split in ("train","val","test"):
        ensure_dir(out_root/"images"/split)
        ensure_dir(out_root/"labels"/split)

    coverage = defaultdict(lambda: defaultdict(int))

    print("Converting MTSD (signs)...")
    ingest_mtsd(mtsd_root, out_root, coverage)

    print("Converting BDD100K (road users + traffic lights)...")
    ingest_bdd100k(bdd_root, out_root, coverage)

    if args.export_lanes:
        lanes_out = Path(args.export_lanes)
        print("Exporting BDD100K lane masks...")
        export_bdd_lanes(bdd_root, lanes_out)

    # Summary
    print("\nCoverage (instances per class):")
    inv = {v:k for k,v in NAME2ID.items()}
    totals = defaultdict(int)
    for src, d in coverage.items():
        for cid, n in d.items():
            totals[cid] += n
    for cid in range(len(CLASSES)):
        print(f"{cid:2d} {inv[cid]:24s} : {totals.get(cid,0)}")

if __name__ == "__main__":
    main()
