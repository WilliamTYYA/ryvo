#!/usr/bin/env python3
# filter_syn_labels_keep_yellow.py
# Keep only: traffic signs + traffic_light_yellow in synthetic labels.
# CLASSES index map (must match your project):
#  0 person
#  1 traffic_light_red
#  2 traffic_light_yellow   <-- keep
#  3 traffic_light_green
#  4..22 traffic sign classes <-- keep all

import argparse
from pathlib import Path

CLASSES = [
    "person",
    "traffic_light_red","traffic_light_yellow","traffic_light_green",
    "stop","yield","no_entry","speed_limit_sign","pedestrian_crossing_sign",
    "no_left_turn","no_right_turn","no_u_turn","one_way","turn_left","turn_right",
    "go_straight","roundabout","keep_right","keep_left","pass_either_side",
    "children_crossing","curve_left","curve_right",
]

# Keep yellow light (2) and all signs (4..22)
ALLOWED = {2} | set(range(4, len(CLASSES)))

def filter_split(lbl_dir: Path, keep_empty: bool, dry_run: bool):
    files = sorted(lbl_dir.rglob("*.txt"))
    kept_lines = removed_lines = changed_files = emptied_files = 0

    for txt in files:
        lines = txt.read_text().splitlines() if txt.exists() else []
        new = []
        removed_here = 0

        for ln in lines:
            p = ln.strip().split()
            if len(p) != 5:
                # Skip malformed line
                removed_here += 1
                continue
            try:
                cls_id = int(float(p[0]))
            except Exception:
                removed_here += 1
                continue

            if cls_id in ALLOWED:
                # Re-emit normalized line
                new.append(f"{cls_id} {p[1]} {p[2]} {p[3]} {p[4]}\n")
            else:
                removed_here += 1

        if removed_here > 0:
            changed_files += 1

        if not dry_run:
            if new:
                txt.write_text("".join(new))
            else:
                if keep_empty:
                    txt.write_text("")  # keep empty file
                    emptied_files += 1
                else:
                    txt.unlink(missing_ok=True)
                    emptied_files += 1

        kept_lines += len(new)
        removed_lines += removed_here

    return {
        "files": len(files),
        "changed": changed_files,
        "emptied": emptied_files,
        "kept_lines": kept_lines,
        "removed_lines": removed_lines,
    }

def main():
    ap = argparse.ArgumentParser(description="Filter synthetic YOLO labels to keep only signs + yellow light.")
    ap.add_argument("--labels-root", type=str, default="datasets/traffic_syn/labels",
                    help="Root of synthetic labels (contains train/val/test).")
    ap.add_argument("--splits", nargs="+", default=["train"], help="Which splits to process.")
    ap.add_argument("--delete-empty", action="store_true",
                    help="Delete label files that end up empty (default: keep empty files).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; do not modify files.")
    args = ap.parse_args()

    root = Path(args.labels_root)
    if not root.exists():
        print(f"[ERR] labels root not found: {root}")
        return

    total = {"files":0,"changed":0,"emptied":0,"kept_lines":0,"removed_lines":0}
    for split in args.splits:
        d = root / split
        if not d.exists():
            print(f"[WARN] split not found: {d}")
            continue
        stats = filter_split(d, keep_empty=not args.delete_empty, dry_run=args.dry_run)
        print(f"[{split}] files={stats['files']} changed={stats['changed']} emptied={stats['emptied']} "
              f"kept_lines={stats['kept_lines']} removed_lines={stats['removed_lines']}")
        for k in total: total[k] += stats[k]

    print(f"[ALL ] files={total['files']} changed={total['changed']} emptied={total['emptied']} "
          f"kept_lines={total['kept_lines']} removed_lines={total['removed_lines']}")

if __name__ == "__main__":
    main()