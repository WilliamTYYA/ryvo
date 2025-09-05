python - <<'PY'
import os, glob, yaml
from collections import defaultdict

YAML_PATH = "datasets/traffic/traffic.yaml"
REAL_ROOT = "datasets/traffic"
SYN_ROOT  = "datasets/traffic_syn"   # change if you used a different folder

with open(YAML_PATH, "r") as f:
    y = yaml.safe_load(f)
names = y["names"]
nc = len(names)

def count_dataset(root, label_subdir="labels"):
    out = {}
    for split in ("train","val","test"):
        base = os.path.join(root, label_subdir, split)
        files = sorted(glob.glob(os.path.join(base, "**", "*.txt"), recursive=True))
        cls_counts = [0]*nc
        bad = 0
        for p in files:
            try:
                for line in open(p, "r"):
                    s = line.strip().split()
                    if not s: continue
                    c = int(float(s[0]))
                    if 0 <= c < nc:
                        cls_counts[c] += 1
            except Exception:
                bad += 1
        out[split] = {"files": len(files), "instances": sum(cls_counts), "cls": cls_counts, "bad": bad}
    # overall
    total_cls = [0]*nc
    total_files = 0
    total_inst = 0
    for split in ("train","val","test"):
        total_files += out[split]["files"]
        total_inst  += out[split]["instances"]
        for i,v in enumerate(out[split]["cls"]):
            total_cls[i] += v
    out["ALL"] = {"files": total_files, "instances": total_inst, "cls": total_cls, "bad": sum(out[s]["bad"] for s in ("train","val","test"))}
    return out

def pretty_print(title, stats):
    print(f"\n[{title}]")
    for split in ("train","val","test","ALL"):
        s = stats[split]
        print(f"  {split:5s}: files={s['files']:6d}  instances={s['instances']:7d}  bad_files={s['bad']}")
    print("  per-class instances (ALL):")
    for i,(name,count) in enumerate(zip(names, stats["ALL"]["cls"])):
        pct = (count / max(1, stats["ALL"]["instances"])) * 100.0
        print(f"    {i:2d} {name:28s} {count:7d}  ({pct:5.2f}%)")

real_stats = count_dataset(REAL_ROOT)
pretty_print("REAL  datasets/traffic", real_stats)

if os.path.exists(SYN_ROOT):
    syn_stats  = count_dataset(SYN_ROOT)
    pretty_print("SYN   datasets/traffic_syn", syn_stats)
else:
    print(f"\n[INFO] synthetic root '{SYN_ROOT}' not found; skipping.")
PY
