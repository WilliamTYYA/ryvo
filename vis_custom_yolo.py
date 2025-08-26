import os, argparse, yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# --- Make sure SiLU is available when loading the Keras model ---
try:
    from custom_yolo import SiLU  # training-time class
except Exception:
    class SiLU(tf.keras.layers.Layer):
        def call(self, x): return x * tf.nn.sigmoid(x)

# --- YAML helpers ---
def parse_ultra_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names if names else [f"class_{i}" for i in range(y.get("nc", 1))]

# --- Geometry & NMS ---
def cxcywh_to_xyxy(cx, cy, w, h):
    return cx - w*0.5, cy - h*0.5, cx + w*0.5, cy + h*0.5

def box_iou_xyxy(a, b):
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

def nms_per_class(boxes_xyxy, scores, iou_thr=0.5, topk=300):
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < topk:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_xyxy(boxes_xyxy[i:i+1], boxes_xyxy[order[1:]]).reshape(-1)
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]  # shift by +1 (relative to order[1:])
    return np.array(keep, dtype=np.int64)

# --- Decode heads (prob outputs) ---
def decode_heads(out_m, out_s, conf_th=0.25, iou_th=0.50, max_det=300):
    def decode_single(pred):
        H, W, D = pred.shape
        C = D - 5
        dets = []
        for gy in range(H):
            row = pred[gy]
            for gx in range(W):
                v = row[gx]
                obj = float(v[0])
                if obj < conf_th: continue
                tx, ty, bw, bh = float(v[1]), float(v[2]), float(v[3]), float(v[4])
                cls_probs = v[5:]  # already softmax
                cls_idx = int(np.argmax(cls_probs))
                cls_p = float(cls_probs[cls_idx])
                score = obj * cls_p
                if score < conf_th: continue
                cx = (gx + tx) / float(W)
                cy = (gy + ty) / float(H)
                x1,y1,x2,y2 = cxcywh_to_xyxy(cx, cy, bw, bh)
                x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
                dets.append({"cls": cls_idx, "score": score, "xyxy":[x1,y1,x2,y2]})
        return dets

    raw = decode_single(out_m) + decode_single(out_s)

    # per-class NMS
    final = []
    by_cls = {}
    for d in raw:
        by_cls.setdefault(d["cls"], []).append(d)
    for c, lst in by_cls.items():
        boxes = np.array([d["xyxy"] for d in lst], dtype=np.float32)
        scores = np.array([d["score"] for d in lst], dtype=np.float32)
        keep = nms_per_class(boxes, scores, iou_thr=iou_th, topk=max_det)
        for k in keep: final.append(lst[int(k)])
    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:max_det]

# --- Draw ---
def draw_dets(img_rgb, dets, labels):
    im = img_rgb.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("SFNS.ttf", 14)
    except:
        font = ImageFont.load_default()

    W, H = im.size
    for d in dets:
        x1,y1,x2,y2 = d["xyxy"]
        x1,y1,x2,y2 = x1*W, y1*H, x2*W, y2*H
        cls = d["cls"]; score = d["score"]
        name = labels[cls] if 0 <= cls < len(labels) else str(cls)

        # box
        draw.rectangle([x1,y1,x2,y2], outline=(255,255,0), width=2)
        # label
        text = f"{name} {score:.2f}"
        tw, th = draw.textlength(text, font=font), (font.size + 6)
        bx, by = x1, max(0, y1 - th)
        draw.rectangle([bx, by, bx+tw+6, by+th], fill=(0,0,0,180))
        draw.text((bx+3, by+3), text, fill=(255,255,255), font=font)

    return im

# --- Main ---
def main():
    a = argparse.ArgumentParser("Visualize YOLO-like Keras model predictions on one image.")
    a.add_argument("--keras", required=True)
    a.add_argument("--yaml", required=True)
    a.add_argument("--img", type=int, default=384)
    a.add_argument("--input", required=True, help="Path to input image")
    a.add_argument("--conf", type=float, default=0.25)
    a.add_argument("--nms_iou", type=float, default=0.50)
    a.add_argument("--max_det", type=int, default=200)
    a.add_argument("--out", default="vis.jpg")
    args = a.parse_args()

    labels = parse_ultra_yaml(args.yaml)

    print("[load]", args.keras)
    model = tf.keras.models.load_model(args.keras, custom_objects={"SiLU": SiLU}, compile=False)

    # preprocess like training (stretch to square)
    im = Image.open(args.input).convert("RGB").resize((args.img, args.img), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = x[None, ...]

    outs = model.predict(x, verbose=0)
    if isinstance(outs, (list,tuple)) and len(outs)==2:
        m = outs[0][0]; s = outs[1][0]
    elif isinstance(outs, dict) and "out_m" in outs and "out_s" in outs:
        m = outs["out_m"][0]; s = outs["out_s"][0]
    else:
        raise RuntimeError("Unexpected model outputs. Expect two heads (out_m, out_s).")

    dets = decode_heads(m, s, conf_th=args.conf, iou_th=args.nms_iou, max_det=args.max_det)
    vis = draw_dets(im, dets, labels)
    vis.save(args.out)
    print(f"[done] wrote {args.out}  (detections: {len(dets)})")

if __name__ == "__main__":
    main()