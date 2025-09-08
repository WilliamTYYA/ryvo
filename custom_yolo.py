# custom_yolo.py
# YOLO-like trainer with C2f backbone, SPPF, simple FPN, 2 heads (P4 & P3)
# Heads are aligned to labels: out_m -> stride16 (G), out_s -> stride8 (2G)
# Loss: focal (obj+cls) + GIoU (prob-based). Schedulers: WarmupCosine / OneCycle.

import os, glob, yaml, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

AUTOTUNE = tf.data.AUTOTUNE

# ---------- util: robust LR setter ----------
def _set_lr(optimizer, lr: float):
    try:
        lr_attr = optimizer.learning_rate
    except Exception:
        try:
            optimizer.learning_rate = float(lr); return
        except Exception:
            return
    for setter in (lambda: lr_attr.assign(lr),
                   lambda: tf.keras.backend.set_value(lr_attr, lr),
                   lambda: setattr(optimizer, "learning_rate", float(lr))):
        try: setter(); return
        except Exception: pass

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

# ---------------- Label encoding (flip + 2 scales) ----------------
def _encode_single_scale(txt_path: str, grid: int, num_classes: int, flip_flag: int):
    G, C = grid, num_classes
    obj = np.zeros((G, G, 1), np.float32)
    box = np.zeros((G, G, 4), np.float32)
    cls = np.zeros((G, G, C), np.float32)
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5: continue
                c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
                if flip_flag == 1: cx = 1.0 - cx
                gx = int(np.clip(cx * G, 0, G-1)); gy = int(np.clip(cy * G, 0, G-1))
                if obj[gy, gx, 0] == 1.0:
                    prev = box[gy, gx]
                    if (w*h) <= (prev[2]*prev[3]):  # keep the bigger target
                        continue
                obj[gy, gx, 0] = 1.0
                box[gy, gx] = [cx*G - gx, cy*G - gy, w, h]  # offsets in [0,1], w/h in [0,1]
                if 0 <= c < C:
                    cls[gy, gx, :] = 0.0
                    cls[gy, gx, c] = 1.0
    return np.concatenate([obj, box, cls], axis=-1).astype(np.float32)

def encode_labels_multi(label_txt: tf.Tensor, grid_main: int, num_classes: int, do_flip: tf.Tensor):
    def _py(txt, flip):
        txt = txt.decode("utf-8"); flip = int(flip)
        y_m = _encode_single_scale(txt, grid_main,     num_classes, flip)   # main G (stride16)
        y_s = _encode_single_scale(txt, grid_main * 2, num_classes, flip)   # fine 2G (stride8)
        return y_m, y_s
    y1, y2 = tf.numpy_function(_py, [label_txt, do_flip], [tf.float32, tf.float32])
    y1.set_shape((grid_main, grid_main, 1 + 4 + num_classes))
    y2.set_shape((grid_main*2, grid_main*2, 1 + 4 + num_classes))
    return {"out_m": y1, "out_s": y2}

# ---------------- Image decode & augment ----------------
def hsv_jitter(img, hgain=0.015, sgain=0.7, vgain=0.4):
    x = tf.image.rgb_to_hsv(img)
    h = x[..., 0:1]; s = x[..., 1:2]; v = x[..., 2:3]
    dh = tf.random.uniform((), -hgain, hgain)
    ds = tf.random.uniform((), 1.0 - sgain, 1.0 + sgain)
    dv = tf.random.uniform((), 1.0 - vgain, 1.0 + vgain)
    h = tf.math.floormod(h + dh, 1.0)
    s = tf.clip_by_value(s * ds, 0.0, 1.0)
    v = tf.clip_by_value(v * dv, 0.0, 1.0)
    return tf.image.hsv_to_rgb(tf.concat([h, s, v], axis=-1))

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
        ds = ds.shuffle(max(1, min(8192, len(pairs))), reshuffle_each_iteration=True)

    # def _map(img_p, lbl_p):
    #     flip_flag = tf.cast(tf.less(tf.random.uniform((), 0, 1), 0.5), tf.int32) if shuffle else tf.constant(0, tf.int32)
    #     img = decode_resize_image(img_p, img_size)
    #     if shuffle: img = hsv_jitter(img)
    #     img = tf.cond(tf.equal(flip_flag, 1), lambda: tf.image.flip_left_right(img), lambda: img)
    #     y = encode_labels_multi(lbl_p, grid_size, num_classes, flip_flag)  # dict {'out_m':..., 'out_s':...}
    #     return img, y

    def _map(img_p, lbl_p):
        # Disable flips to keep left/right semantics correct
        flip_flag = tf.constant(0, tf.int32)
        img = decode_resize_image(img_p, img_size)
        if shuffle:
            img = hsv_jitter(img)
        # DO NOT flip the image anymore
        # img = tf.cond(tf.equal(flip_flag, 1), lambda: tf.image.flip_left_right(img), lambda: img)
        y = encode_labels_multi(lbl_p, grid_size, num_classes, flip_flag)  # flip_flag=0 → labels unchanged
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

# ---------------- Blocks: SiLU, C2f, SPPF, etc. ----------------
class SiLU(layers.Layer):
    def call(self, x): return x * tf.nn.sigmoid(x)

def conv_bn_act(x, c, k=3, s=1, act=True):
    x = layers.Conv2D(c, k, s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return SiLU()(x) if act else x

def Bottleneck(x, c, shortcut=True):
    y = conv_bn_act(x, c, 1, 1, True)
    y = conv_bn_act(y, c, 3, 1, True)
    return layers.Add()([x, y]) if shortcut and x.shape[-1] == c else y

def C2f(x, c, n=2, shortcut=True):
    c_ = c // 2
    x1 = conv_bn_act(x, c_, 1, 1, True)
    x2 = conv_bn_act(x, c_, 1, 1, True)
    y = x1
    for _ in range(n):
        y = Bottleneck(y, c_, shortcut=True)
    y = layers.Concatenate()([y, x2])
    return conv_bn_act(y, c, 1, 1, True)

def SPPF(x, c, k=5):
    x = conv_bn_act(x, c, 1, 1, True)
    y1 = layers.MaxPooling2D(k, 1, padding="same")(x)
    y2 = layers.MaxPooling2D(k, 1, padding="same")(y1)
    y3 = layers.MaxPooling2D(k, 1, padding="same")(y2)
    y = layers.Concatenate()([x, y1, y2, y3])
    return conv_bn_act(y, c, 1, 1, True)

def pred_head(feat, num_classes, name):
    """Head outputs probabilities (sigmoid/softmax) to match the prob-based loss."""
    c_mid = max(96, feat.shape[-1] // 2)  # light mid channels
    x = conv_bn_act(feat, c_mid, 3, 1, True)
    x = conv_bn_act(x, c_mid, 3, 1, True)
    obj = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)
    box = layers.Conv2D(4, 1, padding="same", activation="sigmoid")(x)
    cls = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(x)
    return layers.Concatenate(axis=-1, name=name)([obj, box, cls])

def build_yolo_like_model(img_size: int, grid_size: int, num_classes: int):
    """
    Backbone: strides 2,4,8,16,32 → P3 (s8), P4 (s16), P5 (s32)
    FPN: P5 ↑ + P4 → f4  (stride16)
         f4 ↑ + P3 → f3  (stride8)
    Heads: out_m on f4 (G = img//16), out_s on f3 (2G = img//8)
    """
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Stem
    x = conv_bn_act(inputs, 32, 3, 2, True)   # 1/2
    x = conv_bn_act(x, 64, 3, 2, True)        # 1/4
    x = C2f(x, 64, n=1)

    # P3 (stride 8)
    x = conv_bn_act(x, 128, 3, 2, True)       # 1/8
    p3 = C2f(x, 128, n=2)

    # P4 (stride 16)
    x = conv_bn_act(p3, 256, 3, 2, True)      # 1/16
    p4 = C2f(x, 256, n=2)

    # P5 (stride 32)
    x = conv_bn_act(p4, 512, 3, 2, True)      # 1/32
    p5 = C2f(x, 512, n=1)
    p5 = SPPF(p5, 512)

    # FPN
    up4 = layers.UpSampling2D(size=2, interpolation="nearest")(p5)  # to stride16
    f4 = C2f(layers.Concatenate()([up4, p4]), 256, n=1)             # -> out_m (G)

    up3 = layers.UpSampling2D(size=2, interpolation="nearest")(f4)  # to stride8
    f3 = C2f(layers.Concatenate()([up3, p3]), 128, n=1)             # -> out_s (2G)

    # Heads (prob outputs)
    out_m = pred_head(f4, num_classes, name="out_m")  # shape: [B, img/16, img/16, 1+4+C]
    out_s = pred_head(f3, num_classes, name="out_s")  # shape: [B, img/8,  img/8,  1+4+C]

    return keras.Model(inputs=inputs, outputs=[out_m, out_s], name="custom_yolo_c2f_fpn")

# ---------------- Losses: focal (obj+cls) + GIoU (prob-based) ----------------
def giou_loss_map(y_true, y_pred, G: int):
    obj = y_true[..., 0:1]
    t = y_true[..., 1:5]
    p = y_pred[..., 1:5]  # already sigmoid in head, so in [0,1]

    g = tf.range(G, dtype=tf.float32)
    gx, gy = tf.meshgrid(g, g, indexing="xy")
    gx = tf.reshape(gx, (1, G, G, 1))
    gy = tf.reshape(gy, (1, G, G, 1))

    cx_t = (gx + t[..., 0:1]) / tf.cast(G, tf.float32)
    cy_t = (gy + t[..., 1:2]) / tf.cast(G, tf.float32)
    w_t  = t[..., 2:3]; h_t = t[..., 3:4]

    cx_p = (gx + p[..., 0:1]) / tf.cast(G, tf.float32)
    cy_p = (gy + p[..., 1:2]) / tf.cast(G, tf.float32)
    w_p  = p[..., 2:3]; h_p = p[..., 3:4]

    def to_corners(cx, cy, w, h):
        x1 = tf.clip_by_value(cx - w/2.0, 0.0, 1.0)
        y1 = tf.clip_by_value(cy - h/2.0, 0.0, 1.0)
        x2 = tf.clip_by_value(cx + w/2.0, 0.0, 1.0)
        y2 = tf.clip_by_value(cy + h/2.0, 0.0, 1.0)
        return x1,y1,x2,y2

    x1_t, y1_t, x2_t, y2_t = to_corners(cx_t, cy_t, w_t, h_t)
    x1_p, y1_p, x2_p, y2_p = to_corners(cx_p, cy_p, w_p, h_p)

    area_t = tf.maximum(x2_t - x1_t, 0.0) * tf.maximum(y2_t - y1_t, 0.0)
    area_p = tf.maximum(x2_p - x1_p, 0.0) * tf.maximum(y2_p - y1_p, 0.0)

    xi1 = tf.maximum(x1_t, x1_p); yi1 = tf.maximum(y1_t, y1_p)
    xi2 = tf.minimum(x2_t, x2_p); yi2 = tf.minimum(y2_t, y2_p)
    inter = tf.maximum(xi2 - xi1, 0.0) * tf.maximum(yi2 - yi1, 0.0)
    union = area_t + area_p - inter + 1e-9
    iou = inter / union

    xc1 = tf.minimum(x1_t, x1_p); yc1 = tf.minimum(y1_t, y1_p)
    xc2 = tf.maximum(x2_t, x2_p); yc2 = tf.maximum(y2_t, y2_p)
    area_c = tf.maximum(xc2 - xc1, 0.0) * tf.maximum(yc2 - yc1, 0.0) + 1e-9

    giou = iou - (area_c - union) / area_c
    giou_loss = 1.0 - giou
    giou_loss = tf.squeeze(giou_loss, -1)
    mask = tf.squeeze(obj, -1)
    return giou_loss * mask

def focal_weight(prob, target, alpha=0.25, gamma=2.0):
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    return alpha * tf.pow(1.0 - p_t, gamma)

def yolo_2scale_loss(grid_main: int, num_classes: int,
                     lambda_box=1.0, lambda_obj=1.0, lambda_cls=1.0,
                     focal_alpha=0.25, focal_gamma=2.0):
    bce = keras.losses.BinaryCrossentropy(reduction="none")
    cce = keras.losses.CategoricalCrossentropy(reduction="none")

    def _single_scale(y_true, y_pred, G):
        obj_t  = y_true[..., 0:1]
        box_t  = y_true[..., 1:5]
        cls_t  = y_true[..., 5:]

        obj_p  = y_pred[..., 0:1]    # sigmoid in head
        box_p  = y_pred[..., 1:5]    # sigmoid in head
        cls_p  = y_pred[..., 5:]     # softmax in head

        obj_map = bce(obj_t, obj_p)                          # [B,G,G]
        fw_obj  = tf.squeeze(focal_weight(obj_p, obj_t, focal_alpha, focal_gamma), -1)
        obj_l   = tf.reduce_sum(obj_map * fw_obj, axis=[1,2])

        giou_map = giou_loss_map(y_true, y_pred, G)
        box_l    = tf.reduce_sum(giou_map, axis=[1,2])

        ce_map = cce(cls_t, cls_p)
        p_t    = tf.reduce_sum(cls_t * cls_p, axis=-1)       # prob of true class
        fw_cls = focal_weight(p_t, tf.ones_like(p_t), focal_alpha, focal_gamma)
        cls_m  = tf.squeeze(obj_t, -1)
        cls_l  = tf.reduce_sum(ce_map * fw_cls * cls_m, axis=[1,2])

        total = lambda_obj * obj_l + lambda_box * box_l + lambda_cls * cls_l
        return tf.reduce_mean(total)

    return {
        "out_m": lambda y_true, y_pred: _single_scale(y_true, y_pred, grid_main),
        "out_s": lambda y_true, y_pred: _single_scale(y_true, y_pred, grid_main * 2),
    }

# ---------------- LR schedulers ----------------
class WarmupCosineLR(tf.keras.callbacks.Callback):
    def __init__(self, lr0, lrf, steps_per_epoch, epochs, warmup_epochs=3.0):
        super().__init__()
        self.lr0=float(lr0); self.lrf=float(lrf)
        self.spe=int(max(steps_per_epoch,1)); self.epochs=int(max(epochs,1))
        self.total=self.spe*self.epochs; self.warm=int(self.spe*max(warmup_epochs,0.0)); self.step=0
    def _cos(self, t): return self.lrf + 0.5*(1.0-self.lrf)*(1+math.cos(math.pi*t))
    def on_train_begin(self, logs=None): _set_lr(self.model.optimizer, 0.0); self.step=0
    def on_train_batch_begin(self, batch, logs=None):
        if self.step < self.warm and self.warm>0:
            lr = self.lr0 * (self.step / self.warm)
        else:
            t = (self.step - self.warm) / max(self.total - self.warm, 1)
            t = min(max(t, 0.0), 1.0)
            lr = self.lr0 * self._cos(t)
        _set_lr(self.model.optimizer, lr); self.step += 1

class OneCycleLR(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, steps_per_epoch, epochs, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        super().__init__()
        self.total=max(int(steps_per_epoch)*int(epochs),1)
        self.warm=max(int(self.total*pct_start),1)
        self.max_lr=float(max_lr)
        self.base=float(max_lr)/float(div_factor)
        self.final=float(max_lr)/float(final_div_factor)
        self.step=0
    def _lr_at(self, s):
        if s<=self.warm: return self.base + (self.max_lr - self.base)*(s/self.warm)
        t=(s-self.warm)/max(self.total-self.warm,1)
        return self.final + 0.5*(self.max_lr-self.final)*(1+math.cos(math.pi*t))
    def on_train_begin(self, logs=None): _set_lr(self.model.optimizer, self.base); self.step=0
    def on_train_batch_begin(self, batch, logs=None): _set_lr(self.model.optimizer, self._lr_at(self.step)); self.step+=1

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
              grid_size: int = 16,       # user hint; we will snap to img//16
              batch: int = 32,
              epochs: int = 10,
              cache: bool = False,
              limit: int | None = None,
              out_path: str = "runs/custom_yolo",
              # schedule
              sched: str = "cosine",
              lr0: float = 1e-3, lrf: float = 0.01, warmup_epochs: float = 3.0,
              max_lr: float = 1e-3, div_factor: float = 25.0, final_div_factor: float = 1e4, pct_start: float = 0.3,
              patience: int = 8,
              optimizer_name: str = "adamw",
              weight_decay: float = 1e-2,
              deterministic: bool = False):
    # Snap grid to architecture (stride16 main). For 512px, G = 32.
    G_arch = img_size // 16
    if grid_size != G_arch:
        print(f"[note] Requested grid={grid_size}, but backbone is stride-16 main; using G={G_arch} and 2G={G_arch*2}.")
        grid_size = G_arch

    (train_ds, val_ds, _), counts, names, nc = build_datasets_from_yaml(
        yaml_path, img_size, grid_size, batch, cache,
        train_limit=limit, val_limit=min(limit, 1000) if limit else None,
        deterministic=deterministic
    )
    n_train = counts[0]
    steps_per_epoch = max(n_train // batch, 1)
    print(f"Dataset sizes (images): train={counts[0]}, val={counts[1]}  (limit={limit})")
    print(f"steps_per_epoch={steps_per_epoch}, batch={batch}, epochs={epochs}")
    print(f"Scheduler: {sched}  |  Grids: G={grid_size}, 2G={grid_size*2}  |  nc={nc}")

    # Build model (C2f + SPPF + FPN)
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

    # Loss dict (prob-based heads)
    loss_dict = yolo_2scale_loss(grid_size, nc, lambda_box=1.0, lambda_obj=1.0, lambda_cls=1.0,
                                 focal_alpha=0.25, focal_gamma=2.0)
    loss_weights = {"out_m": 1.0, "out_s": 0.7}

    model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weights)

    # Callbacks
    os.makedirs(out_path, exist_ok=True)
    ckpt_path = os.path.join(out_path, f"best_{img_size}_G{grid_size}.keras")
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

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    final_path = os.path.join(out_path, f"final_{img_size}_G{grid_size}.keras")
    model.save(final_path)
    print("Saved models to:", {"best": ckpt_path, "final": final_path})

if __name__ == "__main__":
    yaml_path = "datasets/traffic_mix/traffic.yaml"
    train_cli(
        yaml_path, img_size=512, grid_size=16, batch=32, epochs=5,
        cache=False, limit=None,  # use all data by default
        sched="cosine",
        lr0=1e-3, lrf=0.01, warmup_epochs=3.0,
        max_lr=1e-3, div_factor=25.0, final_div_factor=1e4, pct_start=0.3,
        patience=8, optimizer_name="adamw", weight_decay=1e-2
    )