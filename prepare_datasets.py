#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil
from pathlib import Path
from glob import glob
from collections import Counter
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(".").resolve()
RAW = PROJECT_ROOT / "datasets" / "raw"
OUT = PROJECT_ROOT / "datasets" / "traffic"

def ensure_det_dirs():
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)

def open_image_size(path: Path):
    with Image.open(path) as im:
        return im.size  # (W, H)

def yolo_line(cls_id, x1, y1, x2, y2, W, H) -> str:
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0); bh = min(max(bh, 0.0), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

# ---------------------------------------------------------------------
# Unified class list (must match traffic.yaml: names:)
# Reduced to core users + lights + key signs (no other_sign / bumps / height-limit / bicycles-only)
# ---------------------------------------------------------------------
CLASSES = [
    "person",
    # "car",
    "traffic_light_red","traffic_light_yellow","traffic_light_green",
    "stop","yield","no_entry","speed_limit_sign","pedestrian_crossing_sign",
    "no_left_turn","no_right_turn","no_u_turn","one_way","turn_left","turn_right",
    "go_straight","roundabout","keep_right","keep_left","pass_either_side",
    "children_crossing","curve_left","curve_right",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------
# MTSD (signs) conversion
# ---------------------------------------------------------------------
def _resolve_mtsd_ann_dir():
    explicit = RAW / "mtsd" / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated" / "annotations"
    if explicit.exists():
        return explicit
    candidates = list(RAW.glob("mtsd/**/mtsd_v2_fully_annotated/annotations"))
    return candidates[0] if candidates else None

MTSD_ANN_DIR = _resolve_mtsd_ann_dir()
MTSD_IMG_ROOTS = sorted(map(Path, glob(str(RAW / "mtsd" / "mtsd_fully_annotated_images*"))))
MTSD_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Your consolidated mapping (unmapped labels are skipped)
MTSD_TO_SIGN = {
    # Core regulatory
    "regulatory--stop--g1": "stop",
    "regulatory--stop--g2": "stop",
    "regulatory--stop--g10": "stop",

    "regulatory--yield--g1": "yield",
    "regulatory--no-entry--g1": "no_entry",

    # Speed limits → one class
    "regulatory--maximum-speed-limit-5--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-10--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-20--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-25--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-25--g2": "speed_limit_sign",
    "regulatory--maximum-speed-limit-30--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-30--g3": "speed_limit_sign",
    "regulatory--maximum-speed-limit-35--g2": "speed_limit_sign",
    "regulatory--maximum-speed-limit-40--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-40--g3": "speed_limit_sign",
    "regulatory--maximum-speed-limit-40--g6": "speed_limit_sign",
    "regulatory--maximum-speed-limit-45--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-45--g3": "speed_limit_sign",
    "regulatory--maximum-speed-limit-50--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-50--g6": "speed_limit_sign",
    "regulatory--maximum-speed-limit-55--g2": "speed_limit_sign",
    "regulatory--maximum-speed-limit-60--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-65--g2": "speed_limit_sign",
    "regulatory--maximum-speed-limit-70--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-80--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-90--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-100--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-100--g3": "speed_limit_sign",
    "regulatory--maximum-speed-limit-110--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-120--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-led-60--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-led-80--g1": "speed_limit_sign",
    "regulatory--maximum-speed-limit-led-100--g1": "speed_limit_sign",

    # Keep / roundabout / one-way / go-straight
    "regulatory--keep-right--g1": "keep_right",
    "regulatory--keep-right--g2": "keep_right",
    "regulatory--keep-right--g4": "keep_right",
    "regulatory--keep-right--g6": "keep_right",
    "regulatory--keep-left--g1": "keep_left",
    "regulatory--keep-left--g2": "keep_left",

    "regulatory--roundabout--g1": "roundabout",
    "regulatory--roundabout--g2": "roundabout",
    "warning--roundabout--g1": "roundabout",
    "warning--roundabout--g25": "roundabout",

    "regulatory--one-way-left--g1": "one_way",
    "regulatory--one-way-left--g2": "one_way",
    "regulatory--one-way-left--g3": "one_way",
    "regulatory--one-way-right--g1": "one_way",
    "regulatory--one-way-right--g2": "one_way",
    "regulatory--one-way-right--g3": "one_way",

    "regulatory--go-straight--g1": "go_straight",
    "regulatory--go-straight--g3": "go_straight",
    "regulatory--one-way-straight--g1": "go_straight",

    # Turn prohibitions / directions
    "regulatory--no-left-turn--g1": "no_left_turn",
    "regulatory--no-left-turn--g2": "no_left_turn",
    "regulatory--no-left-turn--g3": "no_left_turn",
    "regulatory--no-right-turn--g1": "no_right_turn",
    "regulatory--no-right-turn--g2": "no_right_turn",
    "regulatory--no-right-turn--g3": "no_right_turn",
    "regulatory--no-u-turn--g1": "no_u_turn",
    "regulatory--no-u-turn--g2": "no_u_turn",
    "regulatory--no-u-turn--g3": "no_u_turn",

    "regulatory--turn-left--g1": "turn_left",
    "regulatory--turn-left--g2": "turn_left",
    "regulatory--turn-left--g3": "turn_left",
    "regulatory--turn-left-ahead--g1": "turn_left",

    "regulatory--turn-right--g1": "turn_right",
    "regulatory--turn-right--g2": "turn_right",
    "regulatory--turn-right--g3": "turn_right",
    "regulatory--turn-right-ahead--g1": "turn_right",
    "regulatory--turn-right-ahead--g2": "turn_right",

    # Pedestrian crossing (info + warning variants)
    "information--pedestrians-crossing--g1": "pedestrian_crossing_sign",
    "information--pedestrians-crossing--g2": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g1": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g4": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g5": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g9": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g10": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g11": "pedestrian_crossing_sign",
    "warning--pedestrians-crossing--g12": "pedestrian_crossing_sign",

    # Children crossing
    "warning--children--g1": "children_crossing",
    "warning--children--g2": "children_crossing",
    "information--children--g1": "children_crossing",

    # Curves (left/right) – related variants
    "warning--curve-left--g1": "curve_left",
    "warning--curve-left--g2": "curve_left",
    "warning--horizontal-alignment-left--g1": "curve_left",
    "warning--hairpin-curve-left--g1": "curve_left",
    "warning--hairpin-curve-left--g3": "curve_left",
    "warning--double-curve-first-left--g1": "curve_left",
    "warning--double-curve-first-left--g2": "curve_left",
    "warning--winding-road-first-left--g2": "curve_left",

    "warning--curve-right--g1": "curve_right",
    "warning--curve-right--g2": "curve_right",
    "warning--horizontal-alignment-right--g1": "curve_right",
    "warning--horizontal-alignment-right--g3": "curve_right",
    "warning--hairpin-curve-right--g4": "curve_right",
    "warning--double-curve-first-right--g1": "curve_right",
    "warning--double-curve-first-right--g2": "curve_right",
    "warning--winding-road-first-right--g1": "curve_right",

    # Pass either side
    "regulatory--pass-on-either-side--g1": "pass_either_side",
    "regulatory--pass-on-either-side--g2": "pass_either_side",
    "warning--pass-left-or-right--g1": "pass_either_side",
    "warning--pass-left-or-right--g2": "pass_either_side",
}

def mtsd_find_image(stem: str):
    for root in MTSD_IMG_ROOTS:
        for ext in MTSD_EXTS:
            p = root / f"{stem}{ext}"
            if p.exists(): return p
    return None

def mtsd_split_from_path(p: Path) -> str:
    s = str(p)
    if "train" in s: return "train"
    if "val" in s: return "val"
    if "test" in s: return "test"
    return "train"

def parse_bbox_generic(obj):
    # MTSD variants: bbox dicts, lists, or polygons
    if "bbox" in obj:
        b = obj["bbox"]
        if isinstance(b, dict):
            if {"xmin","ymin","xmax","ymax"} <= set(b.keys()):
                return float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])
            if {"x","y","w","h"} <= set(b.keys()):
                x1 = float(b["x"]); y1 = float(b["y"])
                return x1, y1, x1+float(b["w"]), y1+float(b["h"])
        elif isinstance(b, (list,tuple)) and len(b) == 4:
            x1,y1,a,b2 = map(float, b)
            return (x1, y1, a, b2) if (a>x1 and b2>y1) else (x1, y1, x1+a, y1+b2)
    for key in ("polygon","poly2d","points"):
        if key in obj:
            pts = obj[key]
            if isinstance(pts, dict) and {"x","y"} <= set(pts.keys()):
                xs, ys = pts["x"], pts["y"]
                if xs and ys: return min(xs), min(ys), max(xs), max(ys)
            elif isinstance(pts, list) and pts and isinstance(pts[0], (list,tuple)) and len(pts[0])>=2:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                return min(xs), min(ys), max(xs), max(ys)
    return None

def convert_mtsd(limit=None, debug=False, args=None):
    ensure_det_dirs()
    if MTSD_ANN_DIR is None or not MTSD_ANN_DIR.exists():
        print(f"MTSD: annotations not found; looked for 'mtsd_v2_fully_annotated/annotations'. Skipping.")
        return
    if not MTSD_IMG_ROOTS:
        print(f"MTSD: no 'mtsd_fully_annotated_images*' volumes found. Skipping.")
        return

    ann_files = sorted(MTSD_ANN_DIR.glob("*.json"))
    if limit: ann_files = ann_files[:limit]

    wrote = Counter(); boxes = Counter(); cov = Counter()

    for i, jp in enumerate(ann_files, 1):
        try:
            data = json.loads(jp.read_text())
        except Exception as e:
            if debug: print(f"[MTSD] JSON error {jp.name}: {e}")
            continue

        stem = jp.stem
        img_path = mtsd_find_image(stem)
        if img_path is None:
            if debug: print(f"[MTSD] image missing for {stem}")
            continue

        split = mtsd_split_from_path(img_path)
        W,H = open_image_size(img_path)

        objs = data.get("objects", data.get("labels", []))
        lines = []
        for obj in objs:
            lab = obj.get("label") or obj.get("classTitle") or obj.get("category") or ""
            if not isinstance(lab, str):
                continue
            lab = lab.strip()
            canon = MTSD_TO_SIGN.get(lab)  # None if unmapped
            if not canon:
                continue
            if canon not in CLASS_TO_ID:
                continue
            bb = parse_bbox_generic(obj)
            if not bb:
                continue
            x1,y1,x2,y2 = bb
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            if x2<=x1 or y2<=y1: continue
            if (x2-x1)<2 or (y2-y1)<2: continue
            lines.append(yolo_line(CLASS_TO_ID[canon], x1,y1,x2,y2, W,H))
            cov[canon]+=1

        if not lines:
            continue
        out_img = OUT/"images"/split/img_path.name
        out_lbl = OUT/"labels"/split/(img_path.stem + ".txt")
        copy_image(img_path, out_img)
        out_lbl.write_text("".join(lines))
        wrote[split]+=1; boxes[split]+=len(lines)

        if debug and i % 500 == 0:
            print(f"[MTSD] {i}/{len(ann_files)} wrote so far: {dict(wrote)}")

    print("MTSD summary:")
    for s in ("train","val","test"):
        print(f"  {s:5s}: wrote {wrote[s]} labeled images, {boxes[s]} boxes")
    print("\nMTSD coverage (instances per class):")
    if cov:
        for cname in sorted(cov, key=lambda k: CLASS_TO_ID[k]):
            print(f"{CLASS_TO_ID[cname]:2d} {cname:24s}: {cov[cname]}")
    else:
        print("  (no boxes kept; check paths or mapping)")

# ---------------------------------------------------------------------
# BDD100K (road users + traffic lights; STRICT filter to person/car + TL colors)
# ---------------------------------------------------------------------
def bdd_get_objects(record: dict):
    if "frames" in record and record["frames"]:
        return record["frames"][0].get("objects", [])
    if "objects" in record:
        return record["objects"]
    if "labels" in record:
        return record["labels"]
    return []

def convert_bdd(limit=None, debug=False):
    ensure_det_dirs()
    bdd_root = RAW / "bdd100k"
    img_root = bdd_root / "images"
    lbl_root = bdd_root / "labels"
    if not (img_root.exists() and lbl_root.exists()):
        print(f"BDD100K: expected {img_root} and {lbl_root}. Skipping.")
        return

    cov = Counter(); wrote = Counter(); boxes = Counter()

    # Only these categories from BDD:
    cat_map = {
        "person": "person",
        # "rider":  "person",  # treat riders as person (remove this line if undesired)
        # "car":    "car",
    }

    for split in ("train","val","test"):
        lbl_dir = lbl_root / split
        img_dir = img_root / split
        if not lbl_dir.exists() or not img_dir.exists():
            print(f"BDD100K: missing split {split} — skipping.")
            continue

        jsons = sorted(lbl_dir.glob("*.json"))
        if limit: jsons = jsons[:limit]

        for i, jp in enumerate(jsons, 1):
            try:
                rec = json.loads(jp.read_text())
            except Exception as e:
                if debug: print(f"[BDD] JSON error {jp.name}: {e}")
                continue

            stem = jp.stem
            img_path = None
            for ext in (".jpg",".png",".JPG",".PNG"):
                p = img_dir / f"{stem}{ext}"
                if p.exists(): img_path = p; break
            if img_path is None:
                if debug: print(f"[BDD] image missing for {stem}")
                continue

            W,H = open_image_size(img_path)
            objs = bdd_get_objects(rec)
            lines = []

            for obj in objs:
                cat = obj.get("category","")
                # traffic light with color
                if cat == "traffic light":
                    color = obj.get("attributes",{}).get("trafficLightColor","none")
                    if color == "red":      cls = "traffic_light_red"
                    elif color == "yellow": cls = "traffic_light_yellow"
                    elif color == "green":  cls = "traffic_light_green"
                    else: continue  # skip 'none' / unknown
                elif cat == "traffic sign":
                    # ALWAYS skip BDD signs (MTSD supplies sign types)
                    continue
                else:
                    cls = cat_map.get(cat)
                    if cls is None:
                        continue

                bb = obj.get("box2d")
                if not bb or not {"x1","y1","x2","y2"} <= set(bb.keys()):
                    continue
                x1,y1,x2,y2 = float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                if x2<=x1 or y2<=y1: continue
                if (x2-x1)<2 or (y2-y1)<2: continue

                if cls not in CLASS_TO_ID:
                    continue
                lines.append(yolo_line(CLASS_TO_ID[cls], x1,y1,x2,y2, W,H))
                cov[cls]+=1

            if not lines: continue
            out_img = OUT/"images"/split/img_path.name
            out_lbl = OUT/"labels"/split/(img_path.stem + ".txt")
            copy_image(img_path, out_img)
            out_lbl.write_text("".join(lines))
            wrote[split]+=1; boxes[split]+=len(lines)

            if debug and i % 5000 == 0:
                print(f"[BDD] {split}: processed {i}/{len(jsons)} ...")

        print(f"BDD100K {split}: wrote {wrote[split]} labeled images, {boxes[split]} boxes")

    print("\nBDD100K coverage (instances per class):")
    if cov:
        for cname in sorted(cov, key=lambda k: CLASS_TO_ID[k]):
            print(f"{CLASS_TO_ID[cname]:2d} {cname:24s}: {cov[cname]}")
    else:
        print("  (no boxes kept; check paths or mapping)")

# ---------------------------------------------------------------------
# BDD100K lane masks (optional)
# ---------------------------------------------------------------------
def poly2d_to_points(poly2d):
    pts = []
    if not isinstance(poly2d, list) or not poly2d:
        return pts
    first = poly2d[0]
    if isinstance(first, dict) and "vertices" in first:
        for comp in poly2d:
            for x,y in comp.get("vertices", []):
                pts.append((float(x), float(y)))
    elif isinstance(first, (list,tuple)) and len(first) >= 2:
        for item in poly2d:
            if isinstance(item, (list,tuple)) and len(item) >= 2:
                pts.append((float(item[0]), float(item[1])))
    return pts

def export_bdd_lanes(lanes_out: Path, debug=False):
    bdd_root = RAW / "bdd100k"
    img_root = bdd_root / "images"
    lbl_root = bdd_root / "labels"
    if not (img_root.exists() and lbl_root.exists()):
        print(f"BDD100K lanes: expected {img_root} and {lbl_root}, skipping.")
        return

    for split in ("train","val","test"):
        out_dir = lanes_out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir = lbl_root / split
        img_dir = img_root / split
        if not lbl_dir.exists() or not img_dir.exists():
            print(f"BDD100K lanes: split {split} missing, skipping.")
            continue

        jsons = sorted(lbl_dir.glob("*.json"))
        count = 0
        for i, jp in enumerate(jsons, 1):
            try:
                rec = json.loads(jp.read_text())
            except Exception:
                continue

            stem = jp.stem
            img_path = None
            for ext in (".jpg",".png",".JPG",".PNG"):
                p = img_dir / f"{stem}{ext}"
                if p.exists(): img_path = p; break
            if img_path is None:
                continue

            try:
                W,H = open_image_size(img_path)
            except Exception:
                continue

            mask = Image.new("L", (W,H), 0)
            draw = ImageDraw.Draw(mask)
            objs = bdd_get_objects(rec)
            drew = False
            for obj in objs:
                cat = obj.get("category","")
                if isinstance(cat,str) and cat.startswith("lane/"):
                    poly2d = obj.get("poly2d")
                    if not poly2d: continue
                    pts = poly2d_to_points(poly2d)
                    if len(pts) >= 3:
                        draw.polygon(pts, outline=255, fill=255)
                        drew = True
            if drew:
                mask.save(out_dir / f"{stem}.png")
                count += 1

            if i % 2000 == 0:
                print(f"LANES {split}: exported {count} masks...")

        print(f"LANES {split}: exported {count} masks.")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert MTSD + BDD100K to YOLO; optionally export BDD lane masks.")
    ap.add_argument("--clean", action="store_true", help="remove datasets/traffic/images and labels before writing")
    ap.add_argument("--limit_mtsd", type=int, default=None, help="limit MTSD JSONs")
    ap.add_argument("--limit_bdd", type=int, default=None, help="limit BDD JSONs per split")
    ap.add_argument("--export_lanes", action="store_true", help="also export BDD lane masks")
    ap.add_argument("--lanes_out", type=str, default=str(PROJECT_ROOT / "datasets" / "lanes"),
                    help="lane masks output folder")
    ap.add_argument("--debug", action="store_true", help="verbose logs")
    args = ap.parse_args()

    if args.clean:
        for d in (OUT/"images", OUT/"labels"):
            if d.exists(): shutil.rmtree(d)
        print("Cleaned datasets/traffic/{images,labels}")

    print("Converting MTSD (signs)…")
    convert_mtsd(limit=args.limit_mtsd, debug=args.debug, args=args)

    print("Converting BDD100K (person/car + traffic light colors only)…")
    convert_bdd(limit=args.limit_bdd, debug=args.debug)

    if args.export_lanes:
        print("Exporting BDD100K lane masks…")
        export_bdd_lanes(Path(args.lanes_out), debug=args.debug)

if __name__ == "__main__":
    main()