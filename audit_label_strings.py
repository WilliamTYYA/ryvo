#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit raw label strings for LISA / MTSD / BSTLD / BDD100K.
- Robust CSV delimiter sniffing for LISA (commas/semicolons, BOM-safe)
- Supervisely JSON for BSTLD (DatasetNinja format)
- MTSD fully-annotated JSON taxonomy strings
- BDD100K detection-style ('labels') and tracking-style ('frames') JSONs

Usage (from repo root):
  conda activate automind
  python audit_label_strings.py
  # or with explicit paths:
  python audit_label_strings.py \
    --lisa "/path/to/lisa" \
    --mtsd "/path/to/mtsd" \
    --bstld "/path/to/bstld" \
    --bdd  "/path/to/bdd100k" \
    --top  60
"""

import argparse
import csv
import io
import json
import re
from pathlib import Path
from collections import Counter

# ---------- Helpers ----------

def norm(s: str) -> str:
    """Normalize label strings for counting/preview."""
    return re.sub(r"[\s\-_]+", "", (s or "").strip().lower()).replace("\ufeff", "")

def first_existing(paths):
    for p in paths:
        if not p:
            continue
        p = Path(p).expanduser()
        if p.exists():
            return p
    return None

def open_csv_any(path: Path) -> csv.DictReader:
    """Open CSV with robust delimiter sniffing (handles BOM; ; or ,)."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sample = "\n".join(raw.splitlines()[:50])  # sniff across multiple lines
    try:
        dialect = csv.Sniffer().sniff(sample)
        delim = dialect.delimiter
    except Exception:
        header = raw.splitlines()[0] if raw else ""
        delim = ";" if ";" in header else ","
    return csv.DictReader(io.StringIO(raw), delimiter=delim)

# ---------- Scanners ----------

def scan_lisa(root: Path, top: int):
    """LISA: frameAnnotationsBOX.csv (traffic signs) and frameAnnotationsBULB.csv (traffic light bulbs)"""
    box = Counter()
    bulb = Counter()

    for csv_path in root.rglob("frameAnnotationsBOX.csv"):
        try:
            reader = open_csv_any(csv_path)
        except Exception as e:
            print(f"  !! Failed to open {csv_path}: {e}")
            continue
        for r in reader:
            # Primary label column in LISA is "Annotation tag"
            tag = r.get("Annotation tag") or r.get("Annotation Tag") or r.get("Tag") or r.get("Label") or ""
            box[norm(tag)] += 1

    for csv_path in root.rglob("frameAnnotationsBULB.csv"):
        try:
            reader = open_csv_any(csv_path)
        except Exception as e:
            print(f"  !! Failed to open {csv_path}: {e}")
            continue
        for r in reader:
            # For bulbs we often have LightColor; sometimes tag columns exist too
            tag = r.get("LightColor") or r.get("Annotation tag") or r.get("Annotation Tag") or r.get("Tag") or r.get("Label") or ""
            bulb[norm(tag)] += 1

    print(f"\n[LISA] root: {root}")
    print("  BOX tags (top {}):".format(top), box.most_common(top) or "— none —")
    print("  BULB tags (light state):", bulb.most_common(10) or "— none —")


def scan_mtsd(root: Path, top: int):
    """MTSD fully-annotated taxonomy strings (e.g., regulatory--speed-limit-50--g1)."""
    c = Counter()
    ann_dir = root / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated" / "annotations"
    if not ann_dir.exists():
        print(f"\n[MTSD] SKIP (annotations not found at {ann_dir})")
        return
    for j in ann_dir.glob("*.json"):
        try:
            data = json.loads(j.read_text())
        except Exception as e:
            print(f"  !! Could not read {j}: {e}")
            continue
        for o in data.get("objects", []):
            c[(o.get("label", "") or "").lower()] += 1
    print(f"\n[MTSD] root: {root}")
    print("  label strings (top {}):".format(top), c.most_common(top) or "— none —")


def scan_bstld_supervisely(root: Path, top: int):
    """BSTLD (DatasetNinja/Supervisely JSON) uses objects[].classTitle and rectangle in points.exterior."""
    c = Counter()
    found = False
    for split in ["train", "val", "test"]:
        ann_dir = root / split / "ann"
        if not ann_dir.exists():
            continue
        found = True
        for j in ann_dir.glob("*.json"):
            try:
                data = json.loads(j.read_text())
            except Exception as e:
                print(f"  !! Could not read {j}: {e}")
                continue
            for o in data.get("objects", []):
                c[(o.get("classTitle", "") or "").lower()] += 1
    if not found:
        print(f"\n[BSTLD] SKIP (no {root}/(train|val|test)/ann)")
        return
    print(f"\n[BSTLD] root: {root}")
    print("  classTitle values (top {}):".format(top), c.most_common(top) or "— none —")


def scan_bdd(root: Path, top: int):
    """BDD100K detection-style JSON ('labels') and tracking-style JSON ('frames')."""
    c_cat = Counter()
    c_tl  = Counter()
    found = False
    for split in ["train","val","test"]:
        labels_dir = root / "labels" / split
        if not labels_dir.exists():
            continue
        found = True
        for j in labels_dir.glob("*.json"):
            try:
                data = json.loads(j.read_text())
            except Exception as e:
                print(f"  !! Could not read {j}: {e}")
                continue
            # tracking style
            if "frames" in data:
                for fr in data["frames"]:
                    for o in fr.get("objects", []):
                        cat = (o.get("category","") or "").lower()
                        c_cat[cat] += 1
                        tlc = (o.get("attributes",{}).get("trafficLightColor","") or "").lower()
                        if tlc:
                            c_tl[tlc] += 1
            # detection style
            elif "labels" in data:
                for o in data["labels"]:
                    cat = (o.get("category","") or "").lower()
                    c_cat[cat] += 1
                    tlc = (o.get("attributes",{}).get("trafficLightColor","") or "").lower()
                    if tlc:
                        c_tl[tlc] += 1
    if not found:
        print(f"\n[BDD100K] SKIP (no labels under {root}/labels/*)")
        return
    print(f"\n[BDD100K] root: {root}")
    print("  categories (top {}):".format(top), c_cat.most_common(top) or "— none —")
    print("  trafficLightColor:", c_tl.most_common() or "— none —")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Audit raw label strings in LISA / MTSD / BSTLD / BDD100K")
    ap.add_argument("--lisa",  type=str, help="Path to LISA root")
    ap.add_argument("--mtsd",  type=str, help="Path to MTSD root")
    ap.add_argument("--bstld", type=str, help="Path to BSTLD root")
    ap.add_argument("--bdd",   type=str, help="Path to BDD100K root")
    ap.add_argument("--top",   type=int, default=60, help="Top-N to display")
    args = ap.parse_args()

    cwd = Path.cwd()

    LISA  = first_existing([args.lisa] if args.lisa else [
        cwd/"datasets/raw/lisa", cwd/"../datasets/raw/lisa", cwd/"lisa"
    ])
    MTSD  = first_existing([args.mtsd] if args.mtsd else [
        cwd/"datasets/raw/mtsd", cwd/"../datasets/raw/mtsd", cwd/"mtsd"
    ])
    BSTLD = first_existing([args.bstld] if args.bstld else [
        cwd/"datasets/raw/bstld", cwd/"../datasets/raw/bstld", cwd/"bstld"
    ])
    BDD   = first_existing([args.bdd] if args.bdd else [
        cwd/"datasets/raw/bdd100k", cwd/"../datasets/raw/bdd100k", cwd/"bdd100k"
    ])

    if not any([LISA, MTSD, BSTLD, BDD]):
        print("Nothing to audit: no dataset roots found.\n"
              "Pass explicit paths, e.g.:\n"
              "  python audit_label_strings.py --lisa '/abs/path/LISA' --mtsd '/abs/path/mtsd' "
              "--bstld '/abs/path/bstld' --bdd '/abs/path/bdd100k'")
        return

    if LISA:  scan_lisa(LISA, args.top)
    else:     print("\n[LISA] SKIP (not found)")

    if MTSD:  scan_mtsd(MTSD, args.top)
    else:     print("\n[MTSD] SKIP (not found)")

    if BSTLD: scan_bstld_supervisely(BSTLD, args.top)
    else:     print("\n[BSTLD] SKIP (not found)")

    if BDD:   scan_bdd(BDD, args.top)
    else:     print("\n[BDD100K] SKIP (not found)")

if __name__ == "__main__":
    main()
