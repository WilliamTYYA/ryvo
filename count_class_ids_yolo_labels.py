import os, glob, yaml
from collections import Counter, defaultdict

yaml_path = "datasets/traffic/traffic.yaml"

with open(yaml_path, "r") as f:
    y = yaml.safe_load(f)
names = y["names"] if isinstance(y["names"], list) else [y["names"][k] for k in sorted(y["names"], key=lambda x:int(x))]
nc = y.get("nc", len(names))
root = y.get("path",".")
if not os.path.isabs(root):
    root = os.path.normpath(os.path.join(os.path.dirname(yaml_path), root))

splits = {
    "train": y.get("train","images/train").replace("images","labels",1),
    "val"  : y.get("val","images/val").replace("images","labels",1),
    "test" : y.get("test","images/test").replace("images","labels",1),
}

def scan(split, rel):
    lbl_dir = os.path.join(root, rel)
    files = sorted(glob.glob(os.path.join(lbl_dir, "**/*.txt"), recursive=True))
    counts = Counter()
    bad = 0
    for p in files:
        with open(p,"r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: 
                    continue
                try:
                    c = int(float(parts[0]))
                except:
                    bad += 1
                    continue
                counts[c] += 1
    total = sum(counts.values())
    print(f"\n[{split}] label dir: {lbl_dir}")
    print(f"  files: {len(files)}  instances: {total}  (bad lines: {bad})")
    for i in range(nc):
        n = counts.get(i,0)
        pct = (n/total*100) if total>0 else 0
        nm = names[i] if i < len(names) else f"class_{i}"
        print(f"  {i:2d} {nm:28s} {n:7d} ({pct:5.2f}%)")
    missing = [i for i in range(nc) if counts.get(i,0)==0]
    if missing:
        print("  -> classes with ZERO instances:", missing)
    return counts

all_counts = {sp: scan(sp, rel) for sp,rel in splits.items()}