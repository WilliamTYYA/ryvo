# eval_custom_yolo.py
# Evaluate custom YOLO-like model: mAP@0.5 and mAP@[.5:.95], per-class AP, and latency.

import os, argparse, math, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from custom_yolo_like_streaming import (
    parse_ultra_yaml,
    images_and_labels_for_split,
    decode_resize_image,  # for consistent resizing pipeline
)

# ---------- IOU / NMS utils ----------

def box_xywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2

def iou_xyxy(a, b):
    # a: [N,4], b: [M,4], coords in [0,1]
    ax1, ay1, ax2, ay2 = np.split(a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(b, 4, axis=1)
    inter_x1 = np.maximum(ax1, bx1.T)
    inter_y1 = np.maximum(ay1, by1.T)
    inter_x2 = np.minimum(ax2, bx2.T)
    inter_y2 = np.minimum(ay2, by2.T)
    inter_w = np.clip(inter_x2 - inter_x1, 0, 1)
    inter_h = np.clip(inter_y2 - inter_y1, 0, 1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b.T - inter
    union = np.clip(union, 1e-9, None)
    return (inter / union)

def nms_per_class(boxes, scores, iou_thr=0.5, max_dets=300):
    # boxes: [N,4] xyxy, scores: [N]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= max_dets: break
        if order.size == 1: break
        ious = iou_xyxy(boxes[i:i+1], boxes[order[1:]]).ravel()
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    return keep

# ---------- Load GT labels (YOLO txt) ----------

def load_gt_yolo_txt(label_path):
    gts = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    x1, y1, x2, y2 = box_xywh_to_xyxy(cx, cy, w, h)
                    gts.append((c, x1, y1, x2, y2))
    return gts

# ---------- Decode model outputs ----------

def decode_preds_single(pred, conf_thr, grid_size, num_classes, nms_iou=0.5):
    """
    pred: [G, G, 1+4+C] where:
          obj = pred[...,0], box = pred[...,1:5] with (dx,dy,w,h) in [0,1],
          cls = pred[...,5:] softmax.
    Returns: list of detections (c, score, x1,y1,x2,y2)
    """
    G = grid_size
    obj = pred[..., 0]          # [G,G]
    box = pred[..., 1:5]        # [G,G,4] dx,dy in [0,1], w,h in [0,1]
    cls = pred[..., 5:]         # [G,G,C]
    H, W, C = cls.shape

    # Per-cell best class score = obj * max(cls)
    cls_idx = np.argmax(cls, axis=-1)                 # [G,G]
    cls_score = cls[np.arange(G)[:,None], np.arange(G)[None,:], cls_idx]  # [G,G]
    scores = (obj * cls_score).reshape(-1)            # [G*G]
    keep = np.where(scores >= conf_thr)[0]

    dets = []
    if keep.size == 0:
        return dets

    # Compute absolute centers
    yy, xx = np.indices((G, G))
    dx = box[..., 0]
    dy = box[..., 1]
    w  = box[..., 2]
    h  = box[..., 3]
    cx = (xx + dx) / G
    cy = (yy + dy) / G

    cx = cx.reshape(-1); cy = cy.reshape(-1)
    w  = w.reshape(-1);  h  = h.reshape(-1)
    cls_idx_flat = cls_idx.reshape(-1)
    boxes_xyxy = np.stack(box_xywh_to_xyxy(cx, cy, w, h), axis=1)  # [G*G,4]

    # filter by conf
    boxes_xyxy = boxes_xyxy[keep]
    scores_k   = scores[keep]
    classes_k  = cls_idx_flat[keep]

    # NMS per-class
    results = []
    for c in np.unique(classes_k):
        inds = np.where(classes_k == c)[0]
        if inds.size == 0: continue
        b = boxes_xyxy[inds]
        s = scores_k[inds]
        keep_idx = nms_per_class(b, s, iou_thr=nms_iou)
        for j in keep_idx:
            results.append((int(c), float(s[j]), float(b[j,0]), float(b[j,1]), float(b[j,2]), float(b[j,3])))
    return results

# ---------- AP / mAP computation ----------

def average_precision(recalls, precisions):
    # standard trapezoidal area under PR curve
    # (monotonic precision envelope can be applied; here we use raw trapezoidal)
    # ensure sorted by recall
    order = np.argsort(recalls)
    r = np.array(recalls)[order]
    p = np.array(precisions)[order]
    ap = 0.0
    for i in range(1, len(r)):
        ap += (r[i] - r[i-1]) * ((p[i] + p[i-1]) / 2.0)
    return ap

def eval_ap_for_class(gt_by_image, det_by_image, class_id, iou_thr=0.5):
    """
    gt_by_image: dict img_id -> list of gt boxes [(c,x1,y1,x2,y2),...]
    det_by_image: dict img_id -> list of detections [(c,score,x1,y1,x2,y2),...]
    Returns AP, precision/recall arrays.
    """
    # Flatten detections across images, keep image id
    records = []
    npos = 0
    for img_id, gts in gt_by_image.items():
        npos += sum(1 for g in gts if g[0] == class_id)
        dets = det_by_image.get(img_id, [])
        for d in dets:
            if d[0] == class_id:
                records.append((img_id, d[1], d[2], d[3], d[4], d[5]))  # (img, score, x1,y1,x2,y2)
    if npos == 0:
        return 0.0, [], []

    # sort by score desc
    records.sort(key=lambda x: -x[1])

    tp = np.zeros(len(records), dtype=np.float32)
    fp = np.zeros(len(records), dtype=np.float32)

    matched = {img_id: [] for img_id in gt_by_image.keys()}
    for i, (img_id, score, x1, y1, x2, y2) in enumerate(records):
        gt = [(k, np.array([g[1], g[2], g[3], g[4]], dtype=np.float32))
              for k, g in enumerate(gt_by_image.get(img_id, [])) if g[0] == class_id]
        if not gt:
            fp[i] = 1.0
            continue
        gtb = np.stack([g[1] for g in gt], axis=0)  # [M,4]
        ious = iou_xyxy(np.array([[x1,y1,x2,y2]], dtype=np.float32), gtb).ravel()
        j = np.argmax(ious)
        if ious[j] >= iou_thr and j not in matched[img_id]:
            tp[i] = 1.0
            matched[img_id].append(j)
        else:
            fp[i] = 1.0

    # precision-recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recalls = tp / max(npos, 1)
    precisions = tp / np.maximum(tp + fp, 1e-9)
    ap = average_precision(recalls, precisions)
    return ap, precisions, recalls

def eval_map(gt_by_image, det_by_image, num_classes, iou_thrs=[0.5]):
    ap_table = {}
    for thr in iou_thrs:
        ap_per_class = []
        for c in range(num_classes):
            ap, _, _ = eval_ap_for_class(gt_by_image, det_by_image, c, iou_thr=thr)
            ap_per_class.append(ap)
        ap_table[thr] = (ap_per_class, float(np.mean(ap_per_class)) if ap_per_class else 0.0)
    return ap_table

# ---------- Evaluation runner ----------

def build_image_dataset(img_label_pairs, img_size, batch):
    img_paths = tf.constant([p[0] for p in img_label_pairs])
    ds = tf.data.Dataset.from_tensor_slices(img_paths)
    ds = ds.map(lambda p: decode_resize_image(p, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def evaluate(weights_path, yaml_path, split="val",
             img_size=512, grid_size=16,
             conf=0.25, nms_iou=0.5,
             batch=64):
    y = parse_ultra_yaml(yaml_path)
    root, names, nc = y["root"], y["names"], y["nc"]

    if split == "val":
        pairs = images_and_labels_for_split(root, y["val_rel"])
    elif split == "test":
        pairs = images_and_labels_for_split(root, y["test_rel"])
    else:
        pairs = images_and_labels_for_split(root, y["train_rel"])

    # Ground truth dict
    gt_by_image = {}
    for idx, (_, lbl) in enumerate(pairs):
        gt_by_image[idx] = load_gt_yolo_txt(lbl)

    # Load model (no need to compile for inference)
    model = keras.models.load_model(weights_path, compile=False)

    # Latency warmup
    ds = build_image_dataset(pairs, img_size, batch)
    it = iter(ds)
    warm = next(it)
    _ = model(warm, training=False)  # warmup

    # Latency benchmark
    t0 = time.time()
    nimg = 0
    detections_by_image = {}
    for b, batch_imgs in enumerate(ds):
        preds = model(batch_imgs, training=False).numpy()  # [B,G,G,1+4+C]
        B, G, _, K = preds.shape
        for i in range(B):
            img_id = nimg + i
            dets = decode_preds_single(preds[i], conf_thr=conf, grid_size=grid_size,
                                       num_classes=nc, nms_iou=nms_iou)
            detections_by_image[img_id] = dets
        nimg += B
    t1 = time.time()
    ms_per_image = (t1 - t0) * 1000.0 / max(nimg, 1)

    # mAP@0.5 and mAP@[.5:.95]
    ap_tbl = eval_map(gt_by_image, detections_by_image, nc, iou_thrs=[0.5])
    ap_tbl_coco = eval_map(
        gt_by_image, detections_by_image, nc,
        iou_thrs=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    )
    map50 = ap_tbl[0.5][1]
    map5095 = float(np.mean([ap_tbl_coco[t][1] for t in ap_tbl_coco]))

    # Per-class AP at 0.5
    per_class_ap_05 = ap_tbl[0.5][0]
    lines = [f"AP@0.5 per class:"]
    for i, ap in enumerate(per_class_ap_05):
        cname = names[i] if i < len(names) else f"class_{i}"
        lines.append(f"  {i:2d} {cname:<22s} {ap:.3f}")
    lines.append(f"\nSummary:")
    lines.append(f"  mAP@0.5     : {map50:.3f}")
    lines.append(f"  mAP@0.5:0.95: {map5095:.3f}")
    lines.append(f"  Latency     : {ms_per_image:.1f} ms/image (batch={batch})")

    report = "\n".join(lines)
    print(report)
    return {
        "map50": map50,
        "map5095": map5095,
        "per_class_ap_05": per_class_ap_05,
        "latency_ms": ms_per_image
    }

# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="datasets/traffic/traffic.yaml")
    ap.add_argument("--weights", required=True,
                    help="Path to saved Keras model, e.g. runs/custom_yolo_like/model_512_16.keras")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    evaluate(args.weights, args.yaml, split=args.split,
             img_size=args.img, grid_size=args.grid,
             conf=args.conf, nms_iou=args.nms, batch=args.batch)
