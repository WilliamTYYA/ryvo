import argparse
from custom_yolo import train_cli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", default="datasets/traffic/traffic.yaml")
    p.add_argument("--img", type=int, default=512)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="runs/custom_yolo")
    # scheduler
    p.add_argument("--sched", choices=["cosine","onecycle"], default="cosine")
    # warmup+cosine
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    # one-cycle
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--div", type=float, default=25.0, help="div_factor")
    p.add_argument("--final_div", type=float, default=1e4, help="final_div_factor")
    p.add_argument("--pct_start", type=float, default=0.3)
    # EarlyStopping
    p.add_argument("--patience", type=int, default=8)
    a = p.parse_args()

    train_cli(a.yaml, img_size=a.img, grid_size=a.grid,
              batch=a.batch, epochs=a.epochs,
              cache=a.cache, limit=a.limit, out_path=a.out,
              sched=a.sched,
              lr0=a.lr0, lrf=a.lrf, warmup_epochs=a.warmup_epochs,
              max_lr=a.max_lr, div_factor=a.div, final_div_factor=a.final_div, pct_start=a.pct_start,
              patience=a.patience)

#python train_custom_yolo.py --yaml datasets/traffic/traffic.yaml \
#  --sched cosine \
#  --img 512 --grid 16 --batch 80 --epochs 10 --limit 2000 \
#  --lr0 1e-3 --lrf 0.01 --warmup_epochs 3 --patience 6

#python train_custom_yolo.py --yaml datasets/traffic/traffic.yaml \
#  --sched onecycle \
#  --img 512 --grid 16 --batch 80 --epochs 10 --limit 2000 \
#  --max_lr 1e-3 --div 25 --final_div 1e4 --pct_start 0.3 --patience 6
