#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, pathlib, concurrent.futures, itertools
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

def list_imgs(p: Path):
    if not p.exists(): return []
    return [q for q in p.iterdir() if q.is_file() and q.suffix in IMG_EXTS]

def list_txts(p: Path):
    if not p.exists(): return []
    return [q for q in p.iterdir() if q.is_file() and q.suffix == ".txt"]

def ensure_tree(root: Path):
    for split in ("train","val","test"):
        (root/"images"/split).mkdir(parents=True, exist_ok=True)
        (root/"labels"/split).mkdir(parents=True, exist_ok=True)

def symlink_one(src: Path, dst: Path, make_abs: bool):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # remove any existing file/symlink at destination (handles broken links too)
        try:
            dst.unlink()
        except FileNotFoundError:
            pass

        # choose target style
        tgt = src.resolve() if make_abs else Path(os.path.relpath(src, dst.parent))
        dst.symlink_to(tgt)
        return True, None
    except Exception as e:
        return False, f"{dst} -> {src}: {e}"

def build_tasks(real: Path, syn: Path, mix: Path, make_abs: bool):
    tasks = []

    # helper to add (src,dst)
    def add(src: Path, dst: Path):
        tasks.append((src, dst, make_abs))

    # ---- REAL (train/val/test) ----
    existing_imgs = { }  # split -> set of basenames
    existing_lbls = { }
    for split in ("train","val","test"):
        existing_imgs[split] = set()
        existing_lbls[split] = set()
        # images
        for f in list_imgs(real/"images"/split):
            bn = f.name
            existing_imgs[split].add(bn)
            add(f, mix/"images"/split/bn)
        # labels
        for f in list_txts(real/"labels"/split):
            bn = f.name
            existing_lbls[split].add(bn)
            add(f, mix/"labels"/split/bn)

    # ---- SYN (train only) ----
    # keep image/label names aligned; if collision, prefix syn_, then syn2_, …
    syn_name_map = {}  # stem -> new_stem
    split = "train"

    # images
    for f in list_imgs(syn/"images"/split):
        stem, ext = f.stem, f.suffix
        bn = f.name
        new_bn = bn
        n = 1
        while new_bn in existing_imgs[split]:
            prefix = "syn_" if n == 1 else f"syn{n}_"
            new_bn = prefix + bn
            n += 1
        existing_imgs[split].add(new_bn)
        syn_name_map[stem] = Path(new_bn).stem  # remember for label
        add(f, mix/"images"/split/new_bn)

    # labels (use mapping from images so they match)
    for f in list_txts(syn/"labels"/split):
        stem = f.stem
        if stem in syn_name_map:
            new_bn = syn_name_map[stem] + ".txt"
        else:
            # no corresponding image seen (rare); still avoid clashes
            bn = f.name
            new_bn = bn
            n = 1
            while new_bn in existing_lbls[split]:
                prefix = "syn_" if n == 1 else f"syn{n}_"
                new_bn = prefix + bn
                n += 1
        existing_lbls[split].add(new_bn)
        add(f, mix/"labels"/split/new_bn)

    return tasks

def main():
    ap = argparse.ArgumentParser(description="Fast parallel symlink mixer for REAL + SYN datasets")
    ap.add_argument("--real", default="datasets/traffic", type=str)
    ap.add_argument("--syn",  default="datasets/traffic_syn", type=str)
    ap.add_argument("--mix",  default="datasets/traffic_mix", type=str)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--abs", action="store_true", help="use absolute symlinks")
    g.add_argument("--rel", action="store_true", help="use relative symlinks (default)")
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 4)*4))
    args = ap.parse_args()

    real = Path(args.real); syn = Path(args.syn); mix = Path(args.mix)
    if not (real.exists() and (real/"images").exists() and syn.exists()):
        print("[ERR] Check --real/--syn paths.", file=sys.stderr); sys.exit(1)

    if mix.exists():
        for p in mix.rglob("*"):
            if p.is_symlink(): p.unlink()
        # keep dirs; faster than rm -rf for large trees
    ensure_tree(mix)

    make_abs = bool(args.abs)
    tasks = build_tasks(real, syn, mix, make_abs)
    total = len(tasks)
    ok = 0
    errs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for success, msg in ex.map(lambda t: symlink_one(*t), tasks, chunksize=256):
            if success: ok += 1
            else: errs.append(msg)

    print(f"[done] linked {ok}/{total} items -> {mix}")
    if errs:
        print(f"[warn] {len(errs)} errors (showing first 10):")
        for m in errs[:10]:
            print("  -", m)

    # quick broken-link scan
    broken = [p for p in mix.rglob("*") if p.is_symlink() and not p.exists()]
    if broken:
        print(f"[warn] broken links: {len(broken)} (first 10)")
        for p in broken[:10]:
            print("  -", p, "->", os.readlink(p))

if __name__ == "__main__":
    main()
