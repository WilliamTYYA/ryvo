import argparse
from custom_yolo import train_cli

def main():
    p = argparse.ArgumentParser(
        description="Train custom YOLO-like model (2-scale FPN, focal+GIoU) on YOLO-formatted data."
    )

    # Data & IO
    p.add_argument("--yaml", default="datasets/traffic/traffic.yaml",
                   help="Path to YOLO-style dataset YAML (train/val/test, class names).")
    p.add_argument("--out", default="runs/custom_yolo",
                   help="Directory to save checkpoints and final model.")
    p.add_argument("--cache", action="store_true",
                   help="Cache tf.data pipeline in RAM (use for subsets, not full 70GB).")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of training images (e.g., 2000 for quick runs).")
    p.add_argument("--deterministic", action="store_true",
                   help="Make tf.data deterministic (slightly slower but reproducible).")

    # Model / image
    p.add_argument("--img", type=int, default=512, help="Square image size.")
    p.add_argument("--grid", type=int, default=16,
                   help="Main prediction grid size (model also predicts at 2x this).")

    # Training
    p.add_argument("--batch", type=int, default=32, help="Batch size.")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    p.add_argument("--patience", type=int, default=8, help="EarlyStopping patience (epochs).")

    # Scheduler
    p.add_argument("--sched", choices=["cosine", "onecycle"], default="cosine",
                   help="Learning rate schedule.")
    # Cosine params (Ultralytics-style warmup + cosine decay)
    p.add_argument("--lr0", type=float, default=1e-3, help="Base LR for cosine schedule.")
    p.add_argument("--lrf", type=float, default=0.01, help="Final LR factor for cosine (lr_end = lr0*lrf).")
    p.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs for cosine.")
    # One-cycle params
    p.add_argument("--max_lr", type=float, default=1e-3, help="Max LR for one-cycle.")
    p.add_argument("--div", type=float, default=25.0, help="div_factor for one-cycle (base_lr = max_lr/div).")
    p.add_argument("--final_div", type=float, default=1e4, help="final_div_factor for one-cycle.")
    p.add_argument("--pct_start", type=float, default=0.3, help="Warmup % of total steps for one-cycle.")

    # Optimizer
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"],
                   help="Optimizer choice.")
    p.add_argument("--wd", type=float, default=1e-2, help="Weight decay (for AdamW/SGD).")

    a = p.parse_args()

    train_cli(
        yaml_path=a.yaml,
        img_size=a.img,
        grid_size=a.grid,
        batch=a.batch,
        epochs=a.epochs,
        cache=a.cache,
        limit=a.limit,
        out_path=a.out,
        # schedule
        sched=a.sched,
        lr0=a.lr0, lrf=a.lrf, warmup_epochs=a.warmup_epochs,
        max_lr=a.max_lr, div_factor=a.div, final_div_factor=a.final_div, pct_start=a.pct_start,
        # early stopping
        patience=a.patience,
        # optimizer
        optimizer_name=a.optimizer,
        weight_decay=a.wd,
        deterministic=a.deterministic,
    )


if __name__ == "__main__":
    main()

# Cosine (Ultralytics-style) + AdamW + grid 32
#python train_custom_yolo.py --yaml datasets/traffic/traffic.yaml \
#  --sched cosine --img 512 --grid 32 --batch 80 --epochs 10 --limit 20000 \
#  --lr0 1e-3 --lrf 0.01 --warmup_epochs 3 --patience 6 \
#  --optimizer adamw --wd 0.01

# One-cycle + AdamW + grid 24
#python train_custom_yolo.py --yaml datasets/traffic/traffic.yaml \
#  --sched onecycle --img 512 --grid 24 --batch 80 --epochs 10 --limit 20000 \
#  --max_lr 1e-3 --div 25 --final_div 1e4 --pct_start 0.3 --patience 6 \
#  --optimizer adamw --wd 0.01
