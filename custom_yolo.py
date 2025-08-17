# custom_yolo.py
# Streaming YOLO-like trainer for TensorFlow 2.x (Apple Silicon friendly)
# Features:
# - tf.data streaming (no giant pre-load)
# - Conv+BN backbone
# - Shape-safe YOLO-like loss (single anchor/cell)
# - EarlyStopping + Checkpoint
# - LR schedule selectable: warmup+cosine OR one-cycle (via --sched)

import os, glob, yaml, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# -----------------------------
# YAML parsing and path helpers
# -----------------------------
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
    if isinstance(names, dict):  # {0:'car',1:'bus',...}
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

# -----------------------------
# Label encoding (YOLO-like)
# -----------------------------
def encode_labels_to_grid_txt(label_txt: tf.Tensor, grid_size: int, num_classes: int):
    def _py(path):
        path = path.decode("utf-8")
        G, C = grid_size, num_classes
        obj = np.zeros((G, G, 1), np.float32)
        box = np.zeros((G, G, 4), np.float32)
        cls = np.zeros((G, G, C), np.float32)
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 5: continue
                    c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
                    gx = int(np.clip(cx * G, 0, G-1)); gy = int(np.clip(cy * G, 0, G-1))
                    # single-anchor: keep larger
                    if obj[gy, gx, 0] == 1.0:
                        prev = box[gy, gx]
                        if (w*h) <= (prev[2]*prev[3]): continue
                    obj[gy, gx, 0] = 1.0
                    box[gy, gx] = [cx*G - gx, cy*G - gy, w, h]  # offsets in [0,1], w/h in [0,1]
                    if 0 <= c < C:
                        cls[gy, gx, :] = 0.0
                        cls[gy, gx, c] = 1.0
        y = np.concatenate([obj, box, cls], axis=-1)
        return y
    y = tf.numpy_function(_py, [label_txt], tf.float32)
    y.set_shape((grid_size, grid_size, 1 + 4 + num_classes))
    return y

# -----------------------------
# Image decoding / augment
# -----------------------------
def decode_resize_image(img_path: tf.Tensor, img_size: int):
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return img

def augment(img: tf.Tensor):
    # light, safe augment; comment out for determinism
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

# -----------------------------
# tf.data builders
# -----------------------------
def make_dataset(pairs, img_size: int, grid_size: int, num_classes: int,
                 batch_size: int, shuffle=True, cache=False, limit=None):
    if limit is not None:
        pairs = pairs[:limit]
    img_paths = tf.constant([p[0] for p in pairs])
    lbl_paths = tf.constant([p[1] for p in pairs])
    ds = tf.data.Dataset.from_tensor_slices((img_paths, lbl_paths))
    if shuffle:
        ds = ds.shuffle(min(8192, len(pairs)), reshuffle_each_iteration=True)
    def _map(img_p, lbl_p):
        x = decode_resize_image(img_p, img_size)
        x = augment(x)
        y = encode_labels_to_grid_txt(lbl_p, grid_size, num_classes)
        return x, y
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    if cache: ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds, len(pairs)

# -----------------------------
# Model (Conv+BN backbone)
# -----------------------------
def build_yolo_like_model(img_size: int, grid_size: int, num_classes: int):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    filters = 16
    size = img_size
    while size > grid_size:
        x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)

        x = layers.Conv2D(filters*2, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)

        filters *= 2
        size //= 2
        if size <= grid_size:
            break

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    objectness = layers.Conv2D(1, 1, activation="sigmoid", name="objectness")(x)
    bbox       = layers.Conv2D(4, 1, activation="sigmoid", name="bbox")(x)
    class_prob = layers.Conv2D(num_classes, 1, activation="softmax", name="class")(x)
    out = layers.Concatenate(axis=-1)([objectness, bbox, class_prob])
    return keras.Model(inputs=inputs, outputs=out, name="custom_yolo")

# -----------------------------
# Loss (shape-safe)
# -----------------------------
def yolo_like_loss(grid_size: int, num_classes: int,
                   lambda_box=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    bce = keras.losses.BinaryCrossentropy(reduction="none")
    cce = keras.losses.CategoricalCrossentropy(reduction="none")
    def _loss(y_true, y_pred):
        obj_t  = y_true[..., 0:1]    # [B,G,G,1]
        box_t  = y_true[..., 1:5]    # [B,G,G,4]
        cls_t  = y_true[..., 5:]     # [B,G,G,C]
        obj_p  = y_pred[..., 0:1]
        box_p  = y_pred[..., 1:5]
        cls_p  = y_pred[..., 5:]

        obj_map  = bce(obj_t, obj_p)                     # [B,G,G]
        obj_mask = tf.squeeze(obj_t, -1)                 # [B,G,G]
        obj_l = lambda_obj * obj_mask * obj_map + lambda_noobj * (1.0 - obj_mask) * obj_map
        obj_l = tf.reduce_sum(obj_l, axis=[1,2])         # [B]

        box_l = tf.reduce_sum(tf.square(box_t - box_p), axis=-1)  # [B,G,G]
        box_l = lambda_box * tf.reduce_sum(box_l * obj_mask, axis=[1,2])

        cls_map = cce(cls_t, cls_p)                      # [B,G,G]
        cls_l = lambda_cls * tf.reduce_sum(cls_map * obj_mask, axis=[1,2])

        total = obj_l + box_l + cls_l
        return tf.reduce_mean(total)
    return _loss

# -----------------------------
# LR Schedulers (callbacks)
# -----------------------------

def _set_lr(optimizer, lr: float):
    """Robustly set optimizer learning rate across Keras/TF variants."""
    try:
        lr_attr = optimizer.learning_rate
    except Exception:
        try:
            optimizer.learning_rate = float(lr)
            return
        except Exception:
            return
    # Try assign() first (Variable)
    try:
        lr_attr.assign(lr)
        return
    except Exception:
        pass
    # Then try K.set_value on tensors
    try:
        tf.keras.backend.set_value(lr_attr, lr)
        return
    except Exception:
        pass
    # Fallback: plain attribute set
    try:
        optimizer.learning_rate = float(lr)
    except Exception:
        pass

class WarmupCosineLR(tf.keras.callbacks.Callback):
    """
    Ultralytics-like schedule:
      warmup (0→lr0) for warmup_steps, then cosine (lr0→lr0*lrf) to the end.
    """
    def __init__(self, lr0, lrf, steps_per_epoch, epochs, warmup_epochs=3.0, verbose=0):
        super().__init__()
        self.lr0 = float(lr0)
        self.lrf = float(lrf)
        self.steps_per_epoch = int(max(steps_per_epoch, 1))
        self.total_steps = int(self.steps_per_epoch * max(epochs, 1))
        self.warmup_steps = int(self.steps_per_epoch * max(warmup_epochs, 0.0))
        self.step = 0
        self.verbose = verbose

    def _cosine(self, t):
        # t in [0,1]
        return self.lrf + 0.5 * (1.0 - self.lrf) * (1 + math.cos(math.pi * t))

    def on_train_begin(self, logs=None):
        # start near 0 safely
        _set_lr(self.model.optimizer, 0.0)
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self.step < self.warmup_steps and self.warmup_steps > 0:
            lr = self.lr0 * (self.step / self.warmup_steps)
        else:
            if self.total_steps > self.warmup_steps:
                t = (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                t = min(max(t, 0.0), 1.0)
            else:
                t = 1.0
            lr = self.lr0 * self._cosine(t)
        _set_lr(self.model.optimizer, lr)
        self.step += 1
        if self.verbose and self.step % 200 == 0:
            print(f"[WarmupCosine] step {self.step}/{self.total_steps}, lr={lr:.6f}")


class OneCycleLR(tf.keras.callbacks.Callback):
    """
    One-Cycle schedule with cosine anneal:
      warmup base_lr→max_lr over pct_start of total steps,
      then anneal max_lr→final_lr.
    """
    def __init__(self, max_lr, steps_per_epoch, epochs,
                 pct_start=0.3, div_factor=25.0, final_div_factor=1e4, verbose=0):
        super().__init__()
        total_steps = int(steps_per_epoch) * int(epochs)
        self.total_steps = max(total_steps, 1)
        self.warmup = max(int(self.total_steps * pct_start), 1)
        self.max_lr = float(max_lr)
        self.base_lr = float(max_lr) / float(div_factor)
        self.final_lr = float(max_lr) / float(final_div_factor)
        self.step = 0
        self.verbose = verbose

    def _lr_at(self, step):
        if step <= self.warmup:
            return self.base_lr + (self.max_lr - self.base_lr) * (step / self.warmup)
        t = (step - self.warmup) / max(self.total_steps - self.warmup, 1)
        return self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * t))

    def on_train_begin(self, logs=None):
        _set_lr(self.model.optimizer, self.base_lr)
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        lr = self._lr_at(self.step)
        _set_lr(self.model.optimizer, lr)
        self.step += 1
        if self.verbose and self.step % 200 == 0:
            print(f"[OneCycle] step {self.step}/{self.total_steps}, lr={lr:.6f}")

# -----------------------------
# Build datasets + CLI train
# -----------------------------
def build_datasets_from_yaml(yaml_path: str, img_size: int, grid_size: int,
                             batch: int, cache=False, train_limit=None,
                             val_limit=None, test_limit=None):
    y = parse_ultra_yaml(yaml_path)
    root, names, nc = y["root"], y["names"], y["nc"]
    train_pairs = images_and_labels_for_split(root, y["train_rel"])
    val_pairs   = images_and_labels_for_split(root, y["val_rel"])
    test_pairs  = images_and_labels_for_split(root, y["test_rel"])
    train_ds, n_train = make_dataset(train_pairs, img_size, grid_size, nc, batch, True,  cache, train_limit)
    val_ds,   n_val   = make_dataset(val_pairs,   img_size, grid_size, nc, batch, False, cache, val_limit)
    test_ds,  n_test  = make_dataset(test_pairs,  img_size, grid_size, nc, batch, False, cache, test_limit)
    return (train_ds, val_ds, test_ds), (n_train, n_val, n_test), names, nc

def train_cli(yaml_path: str,
              img_size: int = 512,
              grid_size: int = 16,
              batch: int = 32,
              epochs: int = 10,
              cache: bool = False,
              limit: int | None = None,
              out_path: str = "runs/custom_yolo",
              # schedule selector
              sched: str = "cosine",
              # warmup+cosine params
              lr0: float = 1e-3,
              lrf: float = 0.01,
              warmup_epochs: float = 3.0,
              # one-cycle params
              max_lr: float = 1e-3,
              div_factor: float = 25.0,
              final_div_factor: float = 1e4,
              pct_start: float = 0.3,
              # EarlyStopping
              patience: int = 8):
    (train_ds, val_ds, _), counts, names, nc = build_datasets_from_yaml(
        yaml_path, img_size, grid_size, batch, cache,
        train_limit=limit, val_limit=min(limit, 1000) if limit else None
    )
    n_train = counts[0]
    steps_per_epoch = max(n_train // batch, 1)
    print(f"Dataset sizes (images): train={counts[0]}, val={counts[1]}  (limit={limit})")
    print(f"steps_per_epoch={steps_per_epoch}, batch={batch}, epochs={epochs}")
    print(f"Scheduler: {sched}")

    model = build_yolo_like_model(img_size, grid_size, nc)

    callbacks = []
    if sched.lower() == "cosine":
        #optimizer = keras.optimizers.Adam(learning_rate=0.0)  # set by callback
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.0,              # scheduler will set per batch
            weight_decay=wd, beta_1=0.9, beta_2=0.999
        )
        callbacks.append(
            WarmupCosineLR(lr0=lr0, lrf=lrf, steps_per_epoch=steps_per_epoch,
                           epochs=epochs, warmup_epochs=warmup_epochs)
        )
        print(f"cosine params: lr0={lr0}, lrf={lrf}, warmup_epochs={warmup_epochs}")
    elif sched.lower() == "onecycle":
        optimizer = keras.optimizers.AdamW(
            learning_rate=max_lr / div_factor,  # base LR for one-cycle warmup
            weight_decay=wd, beta_1=0.9, beta_2=0.999
        )
        callbacks.append(
            OneCycleLR(max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs,
                       pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor)
        )
        print(f"onecycle params: max_lr={max_lr}, div={div_factor}, final_div={final_div_factor}, pct_start={pct_start}")
    else:
        raise ValueError("--sched must be 'cosine' or 'onecycle'")

    model.compile(optimizer=optimizer, loss=yolo_like_loss(grid_size, nc))

    os.makedirs(out_path, exist_ok=True)
    ckpt_path = os.path.join(out_path, f"best_{img_size}_{grid_size}.keras")
    callbacks += [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                        save_best_only=True, save_weights_only=False, verbose=1),
        keras.callbacks.TerminateOnNaN()
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    final_path = os.path.join(out_path, f"model_{img_size}_{grid_size}.keras")
    model.save(final_path)
    print("Saved models to:", {"best": ckpt_path, "final": final_path})

if __name__ == "__main__":
    yaml_path = "datasets/traffic/traffic.yaml"
    train_cli(yaml_path, img_size=512, grid_size=16, batch=32, epochs=5,
              cache=False, limit=1000, sched="cosine",
              lr0=1e-3, lrf=0.01, warmup_epochs=3.0,
              max_lr=1e-3, div_factor=25.0, final_div_factor=1e4, pct_start=0.3,
              patience=8)
