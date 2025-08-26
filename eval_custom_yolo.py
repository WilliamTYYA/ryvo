# eval_custom_yolo.py
# Evaluate a YOLO-like Keras model on a YOLO-formatted dataset (yaml).
# - Uses the SAME resize policy as training (stretch to square, no letterbox).
# - Decodes 2 heads (P4: G, P3: 2G) with probabilities already post-activated.
# - Per-image, per-class NMS (fixed indexing bug).
# - Computes AP@0.50 and mAP@[0.50:0.95] across classes.

import os, glob, argparse, yaml
import numpy as np
from PIL import Image
import tensorflow as tf

# Ensure custom layer is available when deserializing the saved model
try:
    from custom_yolo import SiLU  # uses the training-time definition
except Exception:
    # Fallback: define a compatible SiLU so load_model can deserialize
    class SiLU(tf.keras.layers.Layer):
        def call(self, x):
            return x * tf.nn.sigmoid(x)

# ---------------- YAML helpers ----------------
def parse_ultra_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    root = y.get("path", ".")
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(yaml_path), root))
    train_rel = y.get("train", "images/train")
    val_rel   = y.get("val", "images/val")
    test_rel  = y.get("test", "images/test")
    names = y.get("names", [])
    if isinstance(names, dict):  # handle dict form
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    nc = y.get("nc", len(names))
    return {"root": root, "train": train_rel, "val": val_rel, "test": test_rel, "names": names, "nc": nc}

def list_image_label_pairs(root: str, images_rel: str):
    images_dir = os.path.join(root, images_rel)
    labels_dir = os.path.join(root, images_rel.replace("images", "labels", 1))
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(images_dir, "**", e), recursive=True))
    paths.sort()
    pairs = []
    for imgp in paths:
        stem = os.path.splitext(os.path.basename(imgp))[0]
        pairs.append((imgp, os.path.join(labels_dir, stem + ".txt")))
    return pairs

# ---------------- Data I/O & preprocessing ----------------
def load_image_stretch(path: str, img_size: int):
    # Same as training: convert to RGB, float32 [0,1], resize to (img_size,img_size) (stretch).
    im = Image.open(path).convert("RGB")
    im = im.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    return arr

def load_yolo_labels(txt_path: str, num_classes: int):
    """
    Returns list of dicts: {"cls": int, "cx":float, "cy":float, "w":float, "h":float}
    All normalized in [0,1] (YOLO format).
    """
    gts = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
                if 0 <= c < num_classes:
                    gts.append({"cls": c, "cx": cx, "cy": cy, "w": w, "h": h})
    return gts

# ---------------- Geometry utils ----------------
def cxcywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w*0.5
    y1 = cy - h*0.5
    x2 = cx + w*0.5
    y2 = cy + h*0.5
    return x1, y1, x2, y2

def box_iou_xyxy(a, b):
    """
    IoU between box arrays a[N,4], b[M,4] in xyxy (normalized [0,1]).
    Returns IoU matrix [N,M].
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0] if b.ndim==2 else 0), dtype=np.float32)
    a = a.astype(np.float32); b = b.astype(np.float32)
    Ax1, Ay1, Ax2, Ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    Bx1, By1, Bx2, By2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(Ax1[:,None], Bx1[None,:])
    inter_y1 = np.maximum(Ay1[:,None], By1[None,:])
    inter_x2 = np.minimum(Ax2[:,None], Bx2[None,:])
    inter_y2 = np.minimum(Ay2[:,None], By2[None,:])
    inter_w  = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h  = np.maximum(inter_y2 - inter_y1, 0.0)
    inter    = inter_w * inter_h
    area_a = np.maximum(Ax2 - Ax1, 0.0) * np.maximum(Ay2 - Ay1, 0.0)
    area_b = np.maximum(Bx2 - Bx1, 0.0) * np.maximum(By2 - By1, 0.0)
    union  = area_a[:,None] + area_b[None,:] - inter + 1e-9
    return inter / union

# ---------------- Corrected NMS (per-class) ----------------
def nms_per_class(boxes_xyxy, scores, iou_thr=0.5, topk=300):
    """
    boxes_xyxy: [N,4], scores: [N]
    Returns indices (into original arrays) kept by NMS.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]  # descending
    keep = []

    while order.size > 0 and len(keep) < topk:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_xyxy(boxes_xyxy[i:i+1], boxes_xyxy[order[1:]]).reshape(-1)
        # survivors are indices (RELATIVE to order[1:]) where IoU <= thr
        inds = np.where(ious <= iou_thr)[0]
        # BUGFIX: shift by +1 because inds are for order[1:]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)

# ---------------- Decode heads ----------------
def decode_heads(pred_m, pred_s, conf_th=0.25, max_det=300):
    """
    pred_m: [H1,W1,1+4+C], pred_s: [H2,W2,1+4+C]
    Returns list of dets dicts: {"cls":int, "score":float, "xyxy":[x1,y1,x2,y2]} in normalized [0,1].
    Heads are already sigmoid/softmax (as trained).
    """
    def decode_single(grid_pred):
        H, W, D = grid_pred.shape
        C = D - 5
        dets = []
        for gy in range(H):
            row = grid_pred[gy]  # [W, D]
            for gx in range(W):
                v = row[gx]       # [D]
                obj = float(v[0])
                if obj < conf_th:
                    continue
                tx, ty, bw, bh = float(v[1]), float(v[2]), float(v[3]), float(v[4])
                cls_probs = v[5:]  # already softmax
                cls_idx = int(np.argmax(cls_probs))
                cls_p = float(cls_probs[cls_idx])
                score = obj * cls_p
                if score < conf_th:
                    continue
                cx = (gx + tx) / float(W)
                cy = (gy + ty) / float(H)
                x1, y1, x2, y2 = cxcywh_to_xyxy(cx, cy, bw, bh)
                # clamp to [0,1]
                x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
                dets.append({"cls": cls_idx, "score": score, "xyxy": [x1, y1, x2, y2]})
        return dets

    dets = decode_single(pred_m) + decode_single(pred_s)

    # Per-class NMS merge
    out = []
    if len(dets) == 0:
        return out

    dets_by_cls = {}
    for d in dets:
        dets_by_cls.setdefault(d["cls"], []).append(d)

    for c, lst in dets_by_cls.items():
        boxes = np.array([d["xyxy"] for d in lst], dtype=np.float32)
        scores = np.array([d["score"] for d in lst], dtype=np.float32)
        keep = nms_per_class(boxes, scores, iou_thr=0.50, topk=max_det)
        for k in keep:
            out.append(lst[int(k)])

    # Sort final by score desc, cap topK
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:max_det]

# ---------------- mAP computation ----------------
def average_precision(rec, prec):
    """
    COCO-style area under precision-recall curve (interpolated).
    rec, prec are 1D arrays sorted by recall ascending.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    # Make precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    # Integrate where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def evaluate_map(all_preds, all_gts, nc, iou_thresholds=np.arange(0.50, 0.96, 0.05)):
    """
    all_preds: list per image of dets [{"cls":int,"score":float,"xyxy":[x1,y1,x2,y2]}, ...]
    all_gts  : list per image of gts  [{"cls":int,"xyxy":[x1,y1,x2,y2]}, ...]
    Returns dict with per-class AP@0.50, per-class mAP@[.50:.95], and overall means.
    """
    # Prepare GT by image and class
    gt_by_img_cls = []
    for gts in all_gts:
        d = {}
        for g in gts:
            d.setdefault(g["cls"], []).append({"xyxy": g["xyxy"], "used": False})
        gt_by_img_cls.append(d)

    results_ap50 = np.zeros(nc, dtype=np.float32)
    results_map = np.zeros(nc, dtype=np.float32)

    for cls in range(nc):
        # gather predictions for this class across images
        records = []  # (img_idx, score, xyxy)
        npos = 0
        for img_idx, (preds, gt_map) in enumerate(zip(all_preds, gt_by_img_cls)):
            # gt count for this class
            if cls in gt_map:
                npos += len(gt_map[cls])
            # preds for this class
            for p in preds:
                if p["cls"] == cls:
                    records.append((img_idx, p["score"], np.array(p["xyxy"], dtype=np.float32)))
        if npos == 0:
            results_ap50[cls] = 0.0
            results_map[cls] = 0.0
            continue
        if len(records) == 0:
            results_ap50[cls] = 0.0
            results_map[cls] = 0.0
            continue

        # Sort predictions by score desc
        records.sort(key=lambda x: x[1], reverse=True)
        preds_img = np.array([r[0] for r in records], dtype=np.int64)
        preds_scr = np.array([r[1] for r in records], dtype=np.float32)
        preds_box = np.stack([r[2] for r in records], axis=0)  # [N,4]

        # Compute AP for each IoU threshold and average
        aps = []
        for thr in iou_thresholds:
            tp = np.zeros(len(records), dtype=np.float32)
            fp = np.zeros(len(records), dtype=np.float32)

            # Reset GT used flags for this thr
            for gm in gt_by_img_cls:
                if cls in gm:
                    for g in gm[cls]:
                        g["used"] = False

            for i in range(len(records)):
                img_i = preds_img[i]
                box_i = preds_box[i]
                gmap = gt_by_img_cls[img_i]
                if cls not in gmap or len(gmap[cls]) == 0:
                    fp[i] = 1.0
                    continue
                gt_boxes = np.array([g["xyxy"] for g in gmap[cls]], dtype=np.float32)
                ious = box_iou_xyxy(box_i[None,:], gt_boxes).reshape(-1)
                j = int(np.argmax(ious))
                if ious[j] >= thr and (not gmap[cls][j]["used"]):
                    tp[i] = 1.0
                    gmap[cls][j]["used"] = True
                else:
                    fp[i] = 1.0

            # cumulative precision/recall
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(npos, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = average_precision(rec, prec)
            aps.append(ap)

        results_ap50[cls] = aps[0]  # first threshold is 0.50
        results_map[cls] = float(np.mean(aps))

    summary = {
        "per_class_AP50": results_ap50,
        "per_class_mAP": results_map,
        "mAP50": float(np.mean(results_ap50)),
        "mAP50_95": float(np.mean(results_map))
    }
    return summary

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser("Evaluate YOLO-like Keras model on YOLO dataset (test split).")
    p.add_argument("--keras", required=True, help="Path to Keras model (.keras)")
    p.add_argument("--yaml", required=True, help="Path to dataset yaml")
    p.add_argument("--img", type=int, default=384, help="Square image size used by the model (e.g., 384 or 512)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold before NMS")
    p.add_argument("--nms_iou", type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--max_det", type=int, default=300, help="Max detections per image after NMS")
    p.add_argument("--limit", type=int, default=None, help="Limit number of test images")
    args = p.parse_args()

    # Load model
    print("[load]", args.keras)
    model = tf.keras.models.load_model(args.keras, custom_objects={"SiLU": SiLU}, compile=False)

    # Parse dataset
    y = parse_ultra_yaml(args.yaml)
    pairs = list_image_label_pairs(y["root"], y["test"])
    if len(pairs) == 0:
        print("[warn] No test images found in YAML 'test'. Falling back to 'val'.")
        pairs = list_image_label_pairs(y["root"], y["val"])
    if args.limit:
        pairs = pairs[:args.limit]
    nc = y["nc"]; names = y["names"] if y["names"] else [f"class_{i}" for i in range(nc)]

    print(f"[data] images={len(pairs)} | nc={nc}")
    print(f"[cfg] img={args.img} conf={args.conf} nms_iou={args.nms_iou} max_det={args.max_det}")

    all_preds = []
    all_gts   = []

    for idx, (imgp, lblp) in enumerate(pairs):
        # Input (stretch to square)
        arr = load_image_stretch(imgp, args.img)  # [H,W,3] float32 [0,1]
        inp = arr[None, ...]  # [1,H,W,3]

        # Inference
        outs = model.predict(inp, verbose=0)
        if isinstance(outs, (list, tuple)) and len(outs) == 2:
            out_m = outs[0][0]  # [G,G,1+4+C]
            out_s = outs[1][0]  # [2G,2G,1+4+C]
        elif isinstance(outs, dict) and "out_m" in outs and "out_s" in outs:
            out_m = outs["out_m"][0]; out_s = outs["out_s"][0]
        else:
            raise RuntimeError("Unexpected model outputs. Expect two heads (out_m, out_s).")

        # Decode detections (per-image NMS inside)
        dets = decode_heads(out_m, out_s, conf_th=args.conf, max_det=args.max_det)

        # Convert to normalized xyxy
        all_preds.append(dets)

        # Load GT, convert to xyxy
        gts_raw = load_yolo_labels(lblp, nc)
        gts_xyxy = []
        for g in gts_raw:
            x1,y1,x2,y2 = cxcywh_to_xyxy(g["cx"], g["cy"], g["w"], g["h"])
            gts_xyxy.append({"cls": g["cls"], "xyxy": [x1,y1,x2,y2]})
        all_gts.append(gts_xyxy)

        if (idx+1) % 50 == 0:
            print(f"[{idx+1}/{len(pairs)}] processed")

    # Evaluate
    metrics = evaluate_map(all_preds, all_gts, nc)

    # Pretty print
    print("\n=== Results ===")
    print(f"mAP@0.50     : {metrics['mAP50']:.4f}")
    print(f"mAP@0.50:0.95: {metrics['mAP50_95']:.4f}\n")

    ap50 = metrics["per_class_AP50"]
    amap = metrics["per_class_mAP"]
    for i in range(nc):
        name = names[i] if i < len(names) else f"class_{i}"
        print(f"{i:2d} {name:30s}  AP50={ap50[i]:.4f}  mAP={amap[i]:.4f}")

if __name__ == "__main__":
    main()