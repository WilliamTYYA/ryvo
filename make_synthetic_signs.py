#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_signs.py

A. build-stickers: Extracts sign "stickers" (PNG+alpha) from MTSD fully-annotated JSONs.
   - Uses polygon masks when available; else falls back to bbox (with soft-edged mask).
   - Saves under: {out_dir}/{class_name}/{stem}_{idx}.png

B. paste: Pastes 0–3 signs per image onto background frames (e.g., your prepared BDD set).
   - Small rotation, mild perspective, tiny blur + JPEG noise for realism.
   - Appends YOLO labels for pasted signs; preserves existing labels.

NOTE: Keep test set real (no synthetic).
"""

import argparse, json, math, os, random, io
from pathlib import Path
from glob import glob
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# --------------------
# Canonical classes (reduced; matches your project)
# --------------------
CLASSES = [
    "person",
    "traffic_light_red","traffic_light_yellow","traffic_light_green",
    "stop","yield","no_entry","speed_limit_sign","pedestrian_crossing_sign",
    "no_left_turn","no_right_turn","no_u_turn","one_way","turn_left","turn_right",
    "go_straight","roundabout","keep_right","keep_left","pass_either_side",
    "children_crossing",
    "curve_left","curve_right"
]
CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}

# --------------------
# MTSD sign mapping (your approved set)
# --------------------
MTSD_TO_SIGN = {
    # Core regulatory shapes
    "regulatory--stop--g1": "stop",
    "regulatory--stop--g2": "stop",
    "regulatory--stop--g10": "stop",

    "regulatory--yield--g1": "yield",
    "regulatory--no-entry--g1": "no_entry",

    # Speed limits (collapsed)
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

    # Keep / roundabout / one-way / straight
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

    # Ped crossing (info + warning)
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

    # Curves
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

# --------------------
# Utility helpers
# --------------------
def _resolve_mtsd_ann_dir(mtsd_root: Path) -> Path|None:
    explicit = mtsd_root / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated" / "annotations"
    if explicit.exists(): return explicit
    cands = list(mtsd_root.glob("**/mtsd_v2_fully_annotated/annotations"))
    return cands[0] if cands else None

def _resolve_mtsd_images(mtsd_root: Path):
    return sorted(map(Path, glob(str(mtsd_root / "mtsd_fully_annotated_images*"))))

def _mtsd_find_image(stem: str, roots) -> Path|None:
    exts = (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")
    for r in roots:
        for e in exts:
            p = r / f"{stem}{e}"
            if p.exists(): return p
    return None

def _get_polygon(obj):
    # Try common polygon containers
    for k in ("polygon","poly2d","points"):
        if k in obj:
            pts = obj[k]
            if isinstance(pts, dict) and "x" in pts and "y" in pts:
                return list(zip(pts["x"], pts["y"]))
            if isinstance(pts, list) and len(pts) and isinstance(pts[0], (list,tuple)) and len(pts[0])>=2:
                return [(float(a), float(b)) for a,b in pts]
    return None

def _get_bbox(obj):
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
    # Fallback: derive bbox from polygon
    poly = _get_polygon(obj)
    if poly:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
    return None

def _soft_rect_mask(w, h, feather=3):
    m = Image.new("L", (w,h), 0)
    dr = ImageDraw.Draw(m)
    dr.rectangle([0,0,w-1,h-1], fill=255)
    if feather>0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return m

def _rand_perspective_coeffs(w, h, mag=0.06):
    # Build mild perspective by jittering corners
    jitter = lambda v: v + random.uniform(-mag, mag)
    src = [(0,0),(w,0),(w,h),(0,h)]
    dst = [(jitter(0)*w, jitter(0)*h),
           ((1+jitter(0))*w, jitter(0)*h),
           ((1+jitter(0))*w, (1+jitter(0))*h),
           (jitter(0)*w, (1+jitter(0))*h)]
    return _find_perspective_coeffs(src, dst)

def _find_perspective_coeffs(pa, pb):
    # From PIL docs recipe
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0,
                      -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1,
                      -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=np.float32)
    B = np.array(pb, dtype=np.float32).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res)

def _yolo_line(cls_id, x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0); bh = min(max(bh, 0.0), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

# --------------------
# A) Build sticker library
# --------------------
def cmd_build_stickers(args):
    mtsd_root = Path(args.mtsd_root)
    out_dir   = Path(args.out_dir)
    ann_dir = _resolve_mtsd_ann_dir(mtsd_root)
    img_roots = _resolve_mtsd_images(mtsd_root)

    if not ann_dir or not img_roots:
        print("[ERR] Could not resolve MTSD annotations/images.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = Counter()
    total = 0

    jsons = sorted(ann_dir.glob("*.json"))
    if args.limit and args.limit > 0:
        jsons = jsons[:args.limit]

    for j in jsons:
        try:
            data = json.loads(j.read_text())
        except Exception as e:
            if args.debug: print(f"[WARN] read {j.name}: {e}")
            continue

        stem = j.stem
        imgp = _mtsd_find_image(stem, img_roots)
        if not imgp:
            if args.debug: print(f"[WARN] image missing for {stem}")
            continue

        try:
            im = Image.open(imgp).convert("RGB")
        except Exception:
            continue
        W, H = im.size

        objs = data.get("objects", data.get("labels", []))
        for k, obj in enumerate(objs):
            lab_raw = (obj.get("label") or obj.get("classTitle") or obj.get("category") or "").strip()
            canon = MTSD_TO_SIGN.get(lab_raw)
            if not canon:  # skip unmapped & "other-sign"
                continue

            bb = _get_bbox(obj)
            if not bb: continue
            x1,y1,x2,y2 = bb
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            if x2<=x1 or y2<=y1: continue

            # Expand a little to catch border/plate
            pad = 0.06
            px = int((x2-x1)*pad); py = int((y2-y1)*pad)
            cx1 = max(0, int(x1 - px)); cy1 = max(0, int(y1 - py))
            cx2 = min(W, int(x2 + px)); cy2 = min(H, int(y2 + py))

            crop = im.crop((cx1, cy1, cx2, cy2))

            # Build a mask
            poly = _get_polygon(obj)
            if poly:
                # Shift poly to crop coords
                poly_shift = [(p[0]-cx1, p[1]-cy1) for p in poly]
                m = Image.new("L", (crop.width, crop.height), 0)
                ImageDraw.Draw(m).polygon(poly_shift, fill=255, outline=255)
                # feather edges a bit
                m = m.filter(ImageFilter.GaussianBlur(1.5))
            else:
                m = _soft_rect_mask(crop.width, crop.height, feather=2)

            # Save RGBA sticker
            sticker = crop.convert("RGBA")
            sticker.putalpha(m)

            out_c = out_dir / canon
            out_c.mkdir(parents=True, exist_ok=True)
            name = f"{stem}_{k}.png"
            sticker.save(out_c / name)
            kept[canon]+=1; total+=1

    print(f"[stickers] saved {total} PNGs under {out_dir}")
    for c in sorted(kept, key=lambda z: CLASS_TO_ID[z]):
        print(f"  {CLASS_TO_ID[c]:2d} {c:24s}: {kept[c]}")

# --------------------
# B) Paste on backgrounds
# --------------------
def _read_yolo_labels(lbl_path: Path):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            p = line.strip().split()
            if len(p) != 5: continue
            c = int(float(p[0])); cx,cy,bw,bh = map(float, p[1:5])
            boxes.append((c,cx,cy,bw,bh))
    return boxes

def _write_yolo_labels(lbl_path: Path, lines: list[str]):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    txt = "".join(lines)
    lbl_path.write_text(txt)

def _jitter_sticker(img_rgba: Image.Image, max_rot=7.0, persp=0.05, blur_p=0.35):
    # random small rotation
    angle = random.uniform(-max_rot, max_rot)
    rot = img_rgba.rotate(angle, resample=Image.BICUBIC, expand=True)

    # mild perspective
    coeffs = _rand_perspective_coeffs(rot.width, rot.height, mag=persp)
    persp_img = rot.transform(rot.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

    # occasional blur
    if random.random() < blur_p:
        persp_img = persp_img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.0)))

    # slight brightness/contrast jitter
    # (avoid external libs, simple gamma)
    if random.random() < 0.6:
        arr = np.asarray(persp_img).astype(np.float32)
        gamma = random.uniform(0.9, 1.1)
        arr[..., :3] = np.clip((arr[..., :3]/255.0) ** gamma * 255.0, 0, 255)
        persp_img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    return persp_img

def _paste_once(bg: Image.Image, sticker: Image.Image, target_w_px: int, pos_xy: tuple[int,int]):
    # resize sticker to target width, keep aspect
    ratio = target_w_px / max(1, sticker.width)
    new_w = max(2, int(sticker.width * ratio))
    new_h = max(2, int(sticker.height * ratio))
    st = sticker.resize((new_w, new_h), resample=Image.BICUBIC)
    st = _jitter_sticker(st)

    x, y = pos_xy
    # clamp within bg
    x = max(-new_w//2, min(x, bg.width - new_w//2))
    y = max(-new_h//2, min(y, bg.height - new_h//2))

    # paste
    bg.alpha_composite(st, (x, y))

    # compute bbox from alpha
    alpha = np.array(st.split()[-1])
    ys, xs = np.where(alpha > 10)
    if xs.size == 0 or ys.size == 0:
        return None  # too transparent
    x1 = x + int(xs.min())
    y1 = y + int(ys.min())
    x2 = x + int(xs.max())
    y2 = y + int(ys.max())

    # clip
    x1 = max(0, min(x1, bg.width-1)); x2 = max(0, min(x2, bg.width-1))
    y1 = max(0, min(y1, bg.height-1)); y2 = max(0, min(y2, bg.height-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1,y1,x2,y2)

def _jpeg_noise(im: Image.Image, quality_range=(72, 92)):
    buf = io.BytesIO()
    q = random.randint(*quality_range)
    im.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def cmd_paste(args):
    stickers_root = Path(args.stickers)
    in_images = Path(args.in_images)
    in_labels = Path(args.in_labels)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Load sticker inventory per class
    inv = {c: sorted((stickers_root / c).glob("*.png")) for c in CLASSES if (stickers_root / c).exists()}
    inv = {k: v for k, v in inv.items() if len(v) > 0}
    if not inv:
        print("[ERR] No stickers found. Run build-stickers first.")
        return

    # Background list (flat folder)
    bg_list = sorted(
        sum(
            [
                glob(str(in_images / "*.jpg")),
                glob(str(in_images / "*.png")),
                glob(str(in_images / "*.JPG")),
                glob(str(in_images / "*.PNG")),
            ],
            [],
        )
    )
    if args.limit and args.limit > 0:
        bg_list = bg_list[: args.limit]

    # Class sampling weights (uniform)
    class_weights = {c: 1.0 for c in inv.keys()}
    keys = list(class_weights.keys())
    weights = np.array([class_weights[k] for k in keys], dtype=np.float32)
    weights = (weights / max(weights.sum(), 1e-9)).tolist()

    made = 0
    pasted_boxes = 0

    for i, imgp in enumerate(bg_list, 1):
        imgp = Path(imgp)
        lblp = in_labels / (imgp.stem + ".txt")

        # Load background once
        try:
            base_bg_rgb = Image.open(imgp).convert("RGB")
        except Exception:
            continue
        W, H = base_bg_rgb.size

        # Preserve existing labels
        existing = _read_yolo_labels(lblp)
        existing_lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for (c, cx, cy, bw, bh) in existing]

        copies = max(1, int(getattr(args, "copies_per_bg", 1)))
        for cidx in range(copies):
            # Fresh canvas for each copy
            bg = base_bg_rgb.copy().convert("RGBA")

            # Per-copy RNG so results change even with same --seed
            rng = random.Random(args.seed * 1000003 + i * 10007 + cidx)

            # Decide how many stickers to add to this copy
            n_to_add = max(0, int(args.per_image))
            if args.randomize:
                n_to_add = rng.randint(0, max(n_to_add, 1))
            n_to_add = min(n_to_add, int(args.max_paste))

            out_lines = existing_lines.copy()

            for _ in range(n_to_add):
                # Pick class & sticker
                cls = rng.choices(keys, weights=weights, k=1)[0]
                if cls not in CLASS_TO_ID:
                    continue
                st_list = inv.get(cls, [])
                if not st_list:
                    continue
                st_path = rng.choice(st_list)
                try:
                    st = Image.open(st_path).convert("RGBA")
                except Exception:
                    continue

                # Scale sticker relative to frame
                short_side = min(W, H)
                target_w = int(short_side * rng.uniform(0.035, 0.12))

                # Plausible roadside placement (left/right shoulder bands & vertical band)
                if rng.random() < 0.5:
                    x = int(rng.uniform(W * 0.55, W * 0.92))  # right
                else:
                    x = int(rng.uniform(W * 0.08, W * 0.45))  # left
                y = int(rng.uniform(H * 0.30, H * 0.78))

                bbox = _paste_once(bg, st, target_w, (x, y))
                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox
                out_lines.append(_yolo_line(CLASS_TO_ID[cls], x1, y1, x2, y2, W, H))
                pasted_boxes += 1

            # JPEG noise and save with unique suffix per copy
            final_rgb = _jpeg_noise(bg, quality_range=(76, 92))
            out_imgp = out_images / f"{imgp.stem}_syn{cidx:02d}.jpg"
            out_lblp = out_labels / f"{imgp.stem}_syn{cidx:02d}.txt"
            final_rgb.save(out_imgp, quality=92)
            _write_yolo_labels(out_lblp, out_lines)
            made += 1

        if i % 500 == 0:
            print(f"[{i}/{len(bg_list)}] variants={made}, pasted_boxes={pasted_boxes}")

    print(f"[done] synthesized {made} images to {out_images}")
    print(f"       pasted sign boxes: {pasted_boxes}")

# --------------------
# CLI
# --------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("build-stickers", help="Extract sign PNGs from MTSD")
    a.add_argument("--mtsd_root", type=str, required=True)
    a.add_argument("--out_dir", type=str, required=True)
    a.add_argument("--limit", type=int, default=None, help="limit JSONs")
    a.add_argument("--debug", action="store_true")

    b = sub.add_parser("paste", help="Paste signs onto backgrounds and write YOLO labels")
    b.add_argument("--stickers", type=str, required=True)
    b.add_argument("--in_images", type=str, required=True)
    b.add_argument("--in_labels", type=str, required=True)
    b.add_argument("--out_images", type=str, required=True)
    b.add_argument("--out_labels", type=str, required=True)
    b.add_argument("--per_image", type=int, default=2, help="target signs per image")
    b.add_argument("--max_paste", type=int, default=3)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--randomize", action="store_true", help="randomize count per image (0..per_image)")

    # in main() -> paste subparser
    b.add_argument("--copies_per_bg", type=int, default=1,
               help="# of synthetic variants to produce per background image")

    args = ap.parse_args()
    if args.cmd == "build-stickers":
        cmd_build_stickers(args)
    else:
        cmd_paste(args)

if __name__ == "__main__":
    main()