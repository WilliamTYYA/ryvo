# custom_yolo_like_streaming.py
# Advanced YOLO-like trainer for TensorFlow 2.x / Keras 3
# Adds: 2-scale FPN head, Focal (obj+cls), GIoU box loss, HSV jitter, horiz flip.
# Still supports: --sched cosine|onecycle, EarlyStopping, robust LR setter, Adam/AdamW/SGD.

import os, glob, yaml, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# ---------- util: robust LR setter (handles Keras variants) ----------
def _set_lr(optimizer, lr: float):
    try:
        lr_attr = optimizer.learning_rate
    except Exception:
        try:
            optimizer.learning_rate = float(lr)
            return
        except Exception:
            return
    try:
        lr_attr.assign(lr); return
    except Exception:
        pass
    try:
        tf.keras.backend.set_value(lr_attr, lr); return
    except Exception:
        pass
    try:
        optimizer.learning_rate = float(lr)
    except Exception:
        pass

# ---------------- YAML & path helpers ----------------
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
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return {"root": root, "train_rel": train_rel, "val_rel": val_rel,
            "test_rel": test_rel, "names": names, "nc": y.get("nc", len(names))}

def images_and_labels_for_split(root: str, images_rel: str):
    images_dir = os.path.join(root, images_rel)
    labels_dir = os.path.join(root, images_rel.replace("images", "labels", 1))
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    img_files = []
    for e in exts:
        img_files.extend(glob.glob(os.path.join(images_dir, "**", e), recursive=True))
    img_files.sort()
    pairs = []
    for img in img_files:
        stem = os.path.splitext(os.path.basename(img))[0]
        pairs.append((img, os.path.join(labels_dir, stem + ".txt")))
    return pairs

# ---------------- Label encoding (supports flip, 2 scales) ----------------
def _encode_single_scale(txt_path: str, grid: int, num_classes: int, flip_flag: int):
    G, C = grid, num_classes
    obj = np.zeros((G, G, 1), np.float32)
    box = np.zeros((G, G, 4), np.float32)
    cls = np.zeros((G, G, C), np.float32)
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
                if flip_flag == 1:
                    cx = 1.0 - cx  # horizontal flip for labels
                gx = int(np.clip(cx * G, 0, G-1)); gy = int(np.clip(cy * G, 0, G-1))
                # single-anchor per cell: keep larger target if conflict
                if obj[gy, gx, 0] == 1.0:
                    prev = box[gy, gx]
                    if (w*h) <= (prev[2]*prev[3]):
                        continue
                obj[gy, gx, 0] = 1.0
                box[gy, gx] = [cx*G - gx, cy*G - gy, w, h]  # offsets in [0,1], w/h in [0,1]
                if 0 <= c < C:
                    cls[gy, gx, :] = 0.0
                    cls[gy, gx, c] = 1.0
    return np.concatenate([obj, box, cls], axis=-1).astype(np.float32)

def encode_labels_multi(label_txt: tf.Tensor, grid_main: int, num_classes: int, do_flip: tf.Tensor):
    def _py(txt, flip):
        txt = txt.decode("utf-8")
        flip = int(flip)
        y_m = _encode_single_scale(txt, grid_main,     num_classes, flip)   # main scale G
        y_s = _encode_single_scale(txt, grid_main * 2, num_classes, flip)   # fine scale 2G
        return y_m, y_s
    y1, y2 = tf.numpy_function(_py, [label_txt, do_flip], [tf.float32, tf.float32])
    y1.set_shape((grid_main, grid_main, 1 + 4 + num_classes))
    y2.set_shape((grid_main*2, grid_main*2, 1 + 4 + num_classes))
    return {"out_m": y1, "out_s": y2}

# ---------------- Image decode & augment (HSV + optional flip) ----------------
def hsv_jitter(img, hgain=0.015, sgain=0.7, vgain=0.4):
    # Ultralytics-like HSV jitter (scaled down)
    x = tf.image.rgb_to_hsv(img)
    h = x[..., 0:1]; s = x[..., 1:2]; v = x[..., 2:3]
    dh = tf.random.uniform((), -hgain, hgain)
    ds = tf.random.uniform((), 1.0 - sgain, 1.0 + sgain)
    dv = tf.random.uniform((), 1.0 - vgain, 1.0 + vgain)
    h = tf.math.floormod(h + dh, 1.0)
    s = tf.clip_by_value(s * ds, 0.0, 1.0)
    v = tf.clip_by_value(v * dv, 0.0, 1.0)
    x = tf.concat([h, s, v], axis=-1)
    return tf.image.hsv_to_rgb(x)

def decode_resize_image(img_path: tf.Tensor, img_size: int):
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return img

# ---------------- tf.data builders ----------------
def make_dataset(pairs, img_size: int, grid_size: int, num_classes: int,
                 batch_size: int, shuffle=True, cache=False, limit=None, deterministic=False):
    if limit is not None:
        pairs = pairs[:limit]
    img_paths = tf.constant([p[0] for p in pairs])
    lbl_paths = tf.constant([p[1] for p in pairs])

    ds = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    if shuffle:
        ds = ds.shuffle(min(8192, len(pairs)), reshuffle_each_iteration=True)

    def _map(img_p, lbl_p):
        # decide flip once and apply to both image and labels
        flip_flag = tf.cast(tf.less(tf.random.uniform((), 0, 1), 0.5), tf.int32)
        img = decode_resize_image(img_p, img_size)
        img = hsv_jitter(img)
        img = tf.cond(tf.equal(flip_flag, 1), lambda: tf.image.flip_left_right(img), lambda: img)
        y = encode_labels_multi(lbl_p, grid_size, num_classes, flip_flag)  # dict {'out_m':..., 'out_s':...}
        return img, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    if not deterministic:
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        ds = ds.with_options(opts)
    ds = ds.prefetch(AUTOTUNE)
    return ds, len(pairs)

# ---------------- Model: backbone + 2-scale FPN heads ----------------
def cbl(x, c, k=3, s=1):
    x = layers.Conv2D(c, k, s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x

def build_yolo_like_model(img_size: int, grid_size: int, num_classes: int):
    """
    Backbone downsamples to grid_size. We keep the feature at 2*grid_size (route_2x)
    and build a small FPN: upsample + concat for the fine scale head.
    Outputs:
      out_m: [G,   G,   1+4+C]
      out_s: [2G,  2G,  1+4+C]
    """
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    filters = 16
    size = img_size
    route_2x = None

    # Downsample pyramid
    while size > grid_size:
        x = cbl(x, filters, 3, 1)
        x = cbl(x, filters, 3, 1)
        x = cbl(x, filters * 2, 3, 2)  # stride-2 downsample
        size //= 2
        if size == grid_size * 2:
            route_2x = x  # save feature at 2*grid_size
        filters *= 2
        if size <= grid_size:
            break

    # Main (GxG) neck
    m = cbl(x, 256); m = cbl(m, 256)
    out_m_obj = layers.Conv2D(1, 1, activation="sigmoid")(m)
    out_m_box = layers.Conv2D(4, 1, activation="sigmoid")(m)
    out_m_cls = layers.Conv2D(num_classes, 1, activation="softmax")(m)
    out_m = layers.Concatenate(axis=-1, name="out_m")([out_m_obj, out_m_box, out_m_cls])

    # Fine (2Gx2G) FPN head
    up = layers.UpSampling2D(size=2, interpolation="nearest")(m)  # upsample main neck
    if route_2x is not None:
        # align channels then concat
        p2 = cbl(route_2x, 128, 1, 1)
        s = layers.Concatenate(axis=-1)([up, p2])
    else:
        s = up
    s = cbl(s, 192); s = cbl(s, 192)
    out_s_obj = layers.Conv2D(1, 1, activation="sigmoid")(s)
    out_s_box = layers.Conv2D(4, 1, activation="sigmoid")(s)
    out_s_cls = layers.Conv2D(num_classes, 1, activation="softmax")(s)
    out_s = layers.Concatenate(axis=-1, name="out_s")([out_s_obj, out_s_box, out_s_cls])

    return keras.Model(inputs=inputs, outputs=[out_m, out_s], name="custom_yolo_like_2scale")

# ---------------- Losses: focal (obj+cls) + GIoU box ----------------
def giou_loss_map(y_true, y_pred, G: int):
    """Return per-cell GIoU loss map [B,G,G]. Uses normalized coords in [0,1]."""
    obj = y_true[..., 0:1]  # [B,G,G,1]
    t = y_true[..., 1:5]
    p = y_pred[..., 1:5]

    # grid centers to absolute cx,cy
    g = tf.range(G, dtype=tf.float32)
    gx, gy = tf.meshgrid(g, g, indexing="xy")  # [G,G]
    gx = tf.reshape(gx, (1, G, G, 1))
    gy = tf.reshape(gy, (1, G, G, 1))

    # true / pred absolute boxes (cx,cy,w,h) -> corners
    cx_t = (gx + t[..., 0:1]) / G; cy_t = (gy + t[..., 1:2]) / G
    w_t  = t[..., 2:3]; h_t = t[..., 3:4]
    cx_p = (gx + p[..., 0:1]) / G; cy_p = (gy + p[..., 1:2]) / G
    w_p  = p[..., 2:3]; h_p = p[..., 3:4]

    def to_corners(cx, cy, w, h):
        x1 = tf.clip_by_value(cx - w/2.0, 0.0, 1.0)
        y1 = tf.clip_by_value(cy - h/2.0, 0.0, 1.0)
        x2 = tf.clip_by_value(cx + w/2.0, 0.0, 1.0)
        y2 = tf.clip_by_value(cy + h/2.0, 0.0, 1.0)
        return x1,y1,x2,y2

    x1_t, y1_t, x2_t, y2_t = to_corners(cx_t, cy_t, w_t, h_t)
    x1_p, y1_p, x2_p, y2_p = to_corners(cx_p, cy_p, w_p, h_p)

    # areas
    area_t = tf.maximum(x2_t - x1_t, 0.0) * tf.maximum(y2_t - y1_t, 0.0)
    area_p = tf.maximum(x2_p - x1_p, 0.0) * tf.maximum(y2_p - y1_p, 0.0)

    # intersection
    xi1 = tf.maximum(x1_t, x1_p); yi1 = tf.maximum(y1_t, y1_p)
    xi2 = tf.minimum(x2_t, x2_p); yi2 = tf.minimum(y2_t, y2_p)
    inter = tf.maximum(xi2 - xi1, 0.0) * tf.maximum(yi2 - yi1, 0.0)
    union = area_t + area_p - inter + 1e-9
    iou = inter / union

    # smallest enclosing box
    xc1 = tf.minimum(x1_t, x1_p); yc1 = tf.minimum(y1_t, y1_p)
    xc2 = tf.maximum(x2_t, x2_p); yc2 = tf.maximum(y2_t, y2_p)
    area_c = tf.maximum(xc2 - xc1, 0.0) * tf.maximum(yc2 - yc1, 0.0) + 1e-9

    giou = iou - (area_c - union) / area_c
    giou_loss = 1.0 - giou  # [B,G,G,1]
    giou_loss = tf.squeeze(giou_loss, axis=-1)  # [B,G,G]
    # only where object exists
    mask = tf.squeeze(obj, -1)
    return giou_loss * mask  # [B,G,G]

def focal_weight(prob, target, alpha=0.25, gamma=2.0):
    # prob is predicted prob for the target class (or obj prob), same shape as target
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    return alpha * tf.pow(1.0 - p_t, gamma)

def yolo_2scale_loss(grid_main: int, num_classes: int,
                     lambda_box=1.0, lambda_obj=1.0, lambda_cls=1.0,
                     focal_alpha=0.25, focal_gamma=2.0):
    """
    Returns a dict of losses for outputs 'out_m' (G) and 'out_s' (2G).
    Each loss is focal(obj+cls) + GIoU(box).
    """
    bce = keras.losses.BinaryCrossentropy(reduction="none")
    cce = keras.losses.CategoricalCrossentropy(reduction="none")

    def _single_scale_loss(y_true, y_pred, G):
        obj_t  = y_true[..., 0:1]     # [B,G,G,1]
        box_t  = y_true[..., 1:5]
        cls_t  = y_true[..., 5:]
        obj_p  = y_pred[..., 0:1]
        box_p  = y_pred[..., 1:5]
        cls_p  = y_pred[..., 5:]

        # --- focal objectness BCE ---
        obj_map = bce(obj_t, obj_p)                         # [B,G,G]
        obj_prob = obj_p                                     # [B,G,G,1]
        fw_obj = focal_weight(obj_prob, obj_t, focal_alpha, focal_gamma)  # [B,G,G,1]
        obj_l = tf.reduce_sum(obj_map * tf.squeeze(fw_obj, -1), axis=[1,2])  # [B]

        # --- GIoU box loss (objects only) ---
        giou_map = giou_loss_map(y_true, y_pred, G)         # [B,G,G]
        box_l = tf.reduce_sum(giou_map, axis=[1,2])         # [B]

        # --- focal class CE (objects only) ---
        ce_map = cce(cls_t, cls_p)                          # [B,G,G]
        # prob of true class (since one-hot)
        p_t = tf.reduce_sum(cls_t * cls_p, axis=-1, keepdims=False)  # [B,G,G]
        fw_cls = focal_weight(p_t, tf.ones_like(p_t), focal_alpha, focal_gamma)  # target=1 for chosen class prob
        cls_mask = tf.squeeze(obj_t, -1)                    # [B,G,G]
        cls_l = tf.reduce_sum(ce_map * fw_cls * cls_mask, axis=[1,2])

        total = lambda_obj * obj_l + lambda_box * box_l + lambda_cls * cls_l
        return tf.reduce_mean(total)

    # return a dict of callables keyed by output names
    return {
        "out_m": lambda y_true, y_pred: _single_scale_loss(y_true, y_pred, grid_main),
        "out_s": lambda y_true, y_pred: _single_scale_loss(y_true, y_pred, grid_main * 2)
    }

# ---------------- LR Schedulers (callbacks) ----------------
class WarmupCosineLR(tf.keras.callbacks.Callback):
    def __init__(self, lr0, lrf, steps_per_epoch, epochs, warmup_epochs=3.0, verbose=0):
        super().__init__()
        self.lr0 = float(lr0); self.lrf = float(lrf)
        self.steps_per_epoch = int(max(steps_per_epoch, 1))
        self.total_steps = int(self.steps_per_epoch * max(epochs, 1))
        self.warmup_steps = int(self.steps_per_epoch * max(warmup_epochs, 0.0))
        self.step = 0; self.verbose = verbose
    def _cosine(self, t):
        return self.lrf + 0.5 * (1.0 - self.lrf) * (1 + math.cos(math.pi * t))
    def on_train_begin(self, logs=None):
        _set_lr(self.model.optimizer, 0.0); self.step = 0
    def on_train_batch_begin(self, batch, logs=None):
        if self.step < self.warmup_steps and self.warmup_steps > 0:
            lr = self.lr0 * (self.step / self.warmup_steps)
        else:
            if self.total_steps > self.warmup_steps:
                t = (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps); t = min(max(t,0.0),1.0)
            else:
                t = 1.0
            lr = self.lr0 * self._cosine(t)
        _set_lr(self.model.optimizer, lr); self.step += 1

class OneCycleLR(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, steps_per_epoch, epochs, pct_start=0.3, div_factor=25.0, final_div_factor=1e4, verbose=0):
        super().__init__()
        self.total_steps = max(int(steps_per_epoch) * int(epochs), 1)
        self.warmup = max(int(self.total_steps * pct_start), 1)
        self.max_lr = float(max_lr)
        self.base_lr = float(max_lr) / float(div_factor)
        self.final_lr = float(max_lr) / float(final_div_factor)
        self.step = 0; self.verbose = verbose
    def _lr_at(self, step):
        if step <= self.warmup:
            return self.base_lr + (self.max_lr - self.base_lr) * (step / self.warmup)
        t = (step - self.warmup) / max(self.total_steps - self.warmup, 1)
        return self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * t))
    def on_train_begin(self, logs=None):
        _set_lr(self.model.optimizer, self.base_lr); self.step = 0
    def on_train_batch_begin(self, batch, logs=None):
        _set_lr(self.model.optimizer, self._lr_at(self.step)); self.step += 1

# ---------------- Build datasets + train CLI ----------------
def build_datasets_from_yaml(yaml_path: str, img_size: int, grid_size: int,
                             batch: int, cache=False, train_limit=None,
                             val_limit=None, test_limit=None, deterministic=False):
    y = parse_ultra_yaml(yaml_path)
    root, names, nc = y["root"], y["names"], y["nc"]
    train_pairs = images_and_labels_for_split(root, y["train_rel"])
    val_pairs   = images_and_labels_for_split(root, y["val_rel"])
    test_pairs  = images_and_labels_for_split(root, y["test_rel"])
    train_ds, n_train = make_dataset(train_pairs, img_size, grid_size, nc, batch, True,  cache, train_limit, deterministic)
    val_ds,   n_val   = make_dataset(val_pairs,   img_size, grid_size, nc, batch, False, cache, val_limit, deterministic)
    test_ds,  n_test  = make_dataset(test_pairs,  img_size, grid_size, nc, batch, False, cache, test_limit, deterministic)
    return (train_ds, val_ds, test_ds), (n_train, n_val, n_test), names, nc

def train_cli(yaml_path: str,
              img_size: int = 512,
              grid_size: int = 16,       # try 24 or 32 for tighter localization
              batch: int = 32,
              epochs: int = 10,
              cache: bool = False,
              limit: int | None = None,
              out_path: str = "runs/custom_yolo_like",
              # schedule selector
              sched: str = "cosine",
              # cosine params
              lr0: float = 1e-3, lrf: float = 0.01, warmup_epochs: float = 3.0,
              # onecycle params
              max_lr: float = 1e-3, div_factor: float = 25.0, final_div_factor: float = 1e4, pct_start: float = 0.3,
              # EarlyStopping
              patience: int = 8,
              # optimizer choice
              optimizer_name: str = "adamw",  # 'adamw'|'adam'|'sgd'
              weight_decay: float = 1e-2,
              deterministic: bool = False):
    (train_ds, val_ds, _), counts, names, nc = build_datasets_from_yaml(
        yaml_path, img_size, grid_size, batch, cache,
        train_limit=limit, val_limit=min(limit, 1000) if limit else None,
        deterministic=deterministic
    )
    n_train = counts[0]
    steps_per_epoch = max(n_train // batch, 1)
    print(f"Dataset sizes (images): train={counts[0]}, val={counts[1]}  (limit={limit})")
    print(f"steps_per_epoch={steps_per_epoch}, batch={batch}, epochs={epochs}")
    print(f"Scheduler: {sched}  |  Grid: {grid_size} & {grid_size*2}")

    # Build model
    model = build_yolo_like_model(img_size, grid_size, nc)

    # Optimizer
    lr_init = 0.0 if sched.lower()=="cosine" else (max_lr / div_factor)
    if optimizer_name.lower() == "adamw":
        try:
            Optim = keras.optimizers.AdamW
        except AttributeError:
            try:
                from tensorflow.keras.optimizers import legacy as _legacy
                Optim = _legacy.AdamW
            except Exception:
                from tensorflow.keras.optimizers.experimental import AdamW as Optim
        optimizer = Optim(learning_rate=lr_init, weight_decay=weight_decay, beta_1=0.9, beta_2=0.999)
    elif optimizer_name.lower() == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr_init)

    # Loss dict for 2 outputs, with weights (fine scale a bit lighter)
    loss_dict = yolo_2scale_loss(grid_size, nc, lambda_box=1.0, lambda_obj=1.0, lambda_cls=1.0,
                                 focal_alpha=0.25, focal_gamma=2.0)
    loss_weights = {"out_m": 1.0, "out_s": 0.7}

    model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weights)

    # Callbacks
    os.makedirs(out_path, exist_ok=True)
    ckpt_path = os.path.join(out_path, f"best_{img_size}_{grid_size}_2scale.keras")
    callbacks = []
    if sched.lower() == "cosine":
        callbacks.append(WarmupCosineLR(lr0=lr0, lrf=lrf, steps_per_epoch=steps_per_epoch,
                                        epochs=epochs, warmup_epochs=warmup_epochs))
        print(f"cosine params: lr0={lr0}, lrf={lrf}, warmup_epochs={warmup_epochs}")
    else:
        callbacks.append(OneCycleLR(max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                    pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor))
        print(f"onecycle params: max_lr={max_lr}, div={div_factor}, final_div={final_div_factor}, pct_start={pct_start}")

    callbacks += [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True,
                                        save_weights_only=False, verbose=1),
        keras.callbacks.TerminateOnNaN()
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    final_path = os.path.join(out_path, f"model_{img_size}_{grid_size}_2scale.keras")
    model.save(final_path)
    print("Saved models to:", {"best": ckpt_path, "final": final_path})

if __name__ == "__main__":
    yaml_path = "datasets/traffic/traffic.yaml"
    train_cli(yaml_path, img_size=512, grid_size=16, batch=32, epochs=5,
              cache=False, limit=1000, sched="cosine",
              lr0=1e-3, lrf=0.01, warmup_epochs=3.0,
              max_lr=1e-3, div_factor=25.0, final_div_factor=1e4, pct_start=0.3,
              patience=8, optimizer_name="adamw", weight_decay=1e-2)
