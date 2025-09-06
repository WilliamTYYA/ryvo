#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly downsample per-class YOLO labels to reduce imbalance.

- Edit ONLY the label .txt files inside your *mix* tree (e.g. datasets/traffic_mix/labels/train).
- Supports per-class CAPS (absolute max kept across all files) and/or KEEP_FRACs.
- Optionally delete label files that become empty, and also remove their paired image symlink.

USAGE (dry-run first):
  python downsample_labels.py \
    --yaml datasets/traffic_mix/traffic.yaml \
    --labels_dirs datasets/traffic_mix/labels/train \
    --cap person=40000 traffic_light_red=40000 traffic_light_green=40000 \
    --keep_frac traffic_light_yellow=1.0 \
    --delete-empty --rm-image \
    --seed 42 --dry-run

Apply for real (same args, no --dry-run):
  python downsample_labels.py ...[same args without --dry-run]

Notes
- Works fine on symlinks; originals in datasets/traffic and datasets/traffic_syn are untouched.
- If you omit --rm-image, empty-label frames can remain as *negative* samples (image kept).
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except Exception:
    raise SystemExit("Please `pip install pyyaml` to run this script.")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# --------- helpers ---------
def parse_kv_list(kvs: List[str], value_type):
    out: Dict[str, float | int] = {}
    for item in kvs or []:
        if "=" not in item:
            raise ValueError(f"Bad spec '{item}', expected name=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if value_type is int:
            out[k] = int(v.replace("_", ""))
        else:
            out[k] = float(v)
    return out


def load_class_list(yaml_path: str) -> List[str]:
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # {0:'person', 1:'tl_red', ...}
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list):
        raise ValueError("YAML 'names' must be a list or {idx:name} dict")
    return names


def find_image_for_label(txt_path: Path, images_dir: Path) -> Path | None:
    """Return the first matching image path under images_dir with same stem."""
    stem = txt_path.stem
    for e in IMG_EXTS:
        p = images_dir / f"{stem}{e}"
        if p.exists() or p.is_symlink():
            return p
    return None


def infer_images_dir_from_labels_dir(labels_dir: Path) -> Path:
    # datasets/traffic_mix/labels/train  ->  datasets/traffic_mix/images/train
    parts = list(labels_dir.parts)
    try:
        i = parts.index("labels")
        parts[i] = "images"
        return Path(*parts)
    except ValueError:
        # fallback: sibling 'images' under the same parent
        return labels_dir.parent.parent / "images" / labels_dir.name


# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="Downsample YOLO labels by class.")
    ap.add_argument("--yaml", required=True, help="dataset yaml (to read class list/order)")
    ap.add_argument(
        "--labels_dirs",
        nargs="+",
        required=True,
        help="one or more label dirs (e.g. datasets/traffic_mix/labels/train)",
    )
    ap.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="images dir for deleting images when labels go empty. "
             "If omitted, inferred from each labels_dir.",
    )
    ap.add_argument(
        "--rm-image",
        action="store_true",
        help="when a label file becomes empty AND --delete-empty is set, also remove its paired image",
    )
    ap.add_argument("--cap", nargs="*", default=[], help="caps like person=40000 tl_red=35000")
    ap.add_argument("--keep_frac", nargs="*", default=[], help="fractions like tl_green=0.30")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delete-empty", action="store_true", help="delete label files that become empty")
    ap.add_argument("--dry-run", action="store_true", help="show what would change (no writes)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    class_names = load_class_list(args.yaml)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    caps = parse_kv_list(args.cap, int)                   # name -> max keep
    fracs = parse_kv_list(args.keep_frac, float)          # name -> keep prob

    # Validate names
    for n in list(caps.keys()) + list(fracs.keys()):
        if n not in name_to_id:
            raise ValueError(f"Unknown class in spec: '{n}'. YAML classes: {class_names}")

    # Convert to id-based specs
    cap_id = {name_to_id[n]: caps[n] for n in caps}
    frac_id = {name_to_id[n]: max(0.0, min(1.0, fracs[n])) for n in fracs}

    # Build label file list (all dirs), then shuffle to avoid early-file bias for caps
    label_files: List[Path] = []
    ldir_to_idir: Dict[Path, Path] = {}

    for ld in args.labels_dirs:
        ldir = Path(ld)
        if not ldir.exists():
            raise FileNotFoundError(f"labels dir not found: {ld}")
        label_files.extend(sorted([p for p in ldir.glob("**/*.txt") if p.is_file()]))
        if args.images_dir:
            ldir_to_idir[ldir] = Path(args.images_dir)
        else:
            ldir_to_idir[ldir] = infer_images_dir_from_labels_dir(ldir)

    rng.shuffle(label_files)

    kept_global = {i: 0 for i in range(len(class_names))}
    orig_global = {i: 0 for i in range(len(class_names))}
    files_changed = 0
    files_emptied = 0
    images_removed = 0

    for txt in label_files:
        try:
            raw = txt.read_text()
        except Exception:
            continue
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        parsed = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(float(parts[0]))
                # validate floats but keep original tokens
                _ = [float(x) for x in parts[1:5]]
            except Exception:
                continue
            parsed.append((cid, parts))

        # Count originals
        for cid, _ in parsed:
            orig_global[cid] += 1

        rng.shuffle(parsed)
        out_lines: List[str] = []

        for cid, parts in parsed:
            keep = True
            # fractional keeping
            if cid in frac_id and rng.random() > frac_id[cid]:
                keep = False
            # cap (global)
            if keep and cid in cap_id and kept_global[cid] >= cap_id[cid]:
                keep = False
            if keep:
                out_lines.append(" ".join(parts) + "\n")
                kept_global[cid] += 1

        changed = len(out_lines) != len(parsed)
        will_be_empty = len(out_lines) == 0

        if changed:
            files_changed += 1

        if args.dry_run:
            # no writes
            pass
        else:
            if will_be_empty and args.delete_empty:
                # remove label file
                try:
                    os.remove(txt)
                    files_emptied += 1
                except Exception:
                    pass

                if args.rm_image:
                    # figure images_dir associated with this labels_dir
                    # find the matching labels root for this txt
                    matched_ldir = None
                    for ldir in ldir_to_idir.keys():
                        try:
                            txt.relative_to(ldir)
                            matched_ldir = ldir
                            break
                        except Exception:
                            continue
                    if matched_ldir is None:
                        matched_ldir = list(ldir_to_idir.keys())[0]

                    img_dir = ldir_to_idir[matched_ldir]
                    img_path = find_image_for_label(txt, img_dir)
                    if img_path is not None:
                        try:
                            os.remove(img_path)
                            images_removed += 1
                        except Exception:
                            pass
            else:
                # write back (even if unchanged is fine)
                try:
                    txt.write_text("".join(out_lines))
                except Exception:
                    pass

    # ---- summary
    def pct(a, b): 
        return (100.0 * a / b) if b > 0 else 0.0

    print("\n=== Downsample Summary ===")
    print(f"label files scanned : {len(label_files)}")
    print(f"files changed       : {files_changed}")
    if args.delete_empty:
        print(f"empty label files {'(would be) ' if args.dry_run else ''}removed: {files_emptied}")
        if args.rm_image:
            print(f"paired images {'(would be) ' if args.dry_run else ''}removed: {images_removed}")
    print("")
    for i, name in enumerate(class_names):
        o = orig_global[i]
        k = kept_global[i]
        print(f"{i:2d} {name:28s} kept {k:7d} / {o:7d} ({pct(k,o):5.1f}%)")

    if args.dry_run:
        print("\n(dry-run only; rerun without --dry-run to apply changes)")


if __name__ == "__main__":
    main()
