#!/usr/bin/env python3
"""
convert_to_coreml.py
Convert a Keras .keras model to a Core ML .mlmodel ready for Xcode.

Usage examples:
  python convert_to_coreml.py \
    --keras runs/custom_yolo/best_384_G24.keras \
    --img 384 \
    --out Models/CustomYOLO.mlmodel \
    --fp16
"""

import argparse, os, sys, shutil
import tensorflow as tf
import coremltools as ct
from tensorflow.keras import layers

# ---- Handle custom layers (e.g., SiLU) ----
try:
    # If your project already defines/exports SiLU:
    from custom_yolo import SiLU as _SiLU  # noqa
except Exception:
    @tf.keras.saving.register_keras_serializable()
    class _SiLU(layers.Layer):
        def call(self, x):
            return tf.nn.silu(x)

CUSTOM_OBJECTS = {"SiLU": _SiLU}

def base_name(tensor_name: str) -> str:
    # Keras tensors often look like 'input_1:0' → return 'input_1'
    return str(tensor_name).split(":")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True, help="Path to .keras model")
    ap.add_argument("--out",   required=True, help="Output .mlmodel path")
    ap.add_argument("--img",   type=int, default=None, help="Square input size (e.g., 384). If omitted, try to infer.")
    ap.add_argument("--input_name", default=None, help="Override input layer name (optional)")
    ap.add_argument("--scale", type=float, default=1/255.0, help="Core ML input scale (default 1/255)")
    ap.add_argument("--bias_r", type=float, default=0.0, help="Red bias")
    ap.add_argument("--bias_g", type=float, default=0.0, help="Green bias")
    ap.add_argument("--bias_b", type=float, default=0.0, help="Blue bias")
    ap.add_argument("--fp16", action="store_true", help="Use float16 weights (smaller & faster)")
    args = ap.parse_args()

    if not os.path.exists(args.keras):
        sys.exit(f"[error] Keras model not found: {args.keras}")

    print(f"[load] {args.keras}")
    model = tf.keras.models.load_model(
        args.keras,
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,   # allow custom classes
    )

    # ---- Inspect I/O ----
    in_t = model.inputs[0]
    in_name = args.input_name or base_name(in_t.name)
    ishape = tuple(in_t.shape)  # e.g., (None, 384, 384, 3)
    print(f"[info] Keras input: name={in_name}, shape={ishape}")

    outs = model.outputs
    out_names = [base_name(o.name) for o in outs]
    out_shapes = [tuple(o.shape) for o in outs]
    print(f"[info] Keras outputs: {list(zip(out_names, out_shapes))}")

    # ---- Decide image size ----
    img_size = args.img
    if img_size is None:
        if len(ishape) == 4 and ishape[1] and ishape[2]:
            img_size = int(ishape[1])
            print(f"[info] Inferred image size from model: {img_size}")
        else:
            sys.exit("[error] Couldn't infer input size. Pass --img <size> (e.g., 384).")

    if ishape[-1] != 3:
        print(f"[warn] Input channels != 3 ({ishape[-1]}). This script assumes RGB.")

    # ---- Build Core ML input description (image input) ----
    img_input = ct.ImageType(
        name=in_name,
        shape=(1, img_size, img_size, 3),   # NHWC with batch=1
        color_layout="RGB",
        scale=args.scale,
        bias=[args.bias_r, args.bias_g, args.bias_b],
    )

    # ---- Export a SavedModel with an explicit serving signature ----
    # This bypasses Keras 2.x internals coremltools expects, and is stable with Keras 3 / TF 2.19.
    tmp_sm = "tmp_savedmodel_for_coreml"
    if os.path.isdir(tmp_sm):
        shutil.rmtree(tmp_sm)

    sig = tf.TensorSpec([1, img_size, img_size, 3], tf.float32, name=in_name)

    @tf.function(input_signature=[sig])
    def serving_fn(x):
        y = model(x, training=False)
        # Return stable, named outputs for Core ML
        if isinstance(y, dict):
            return y
        elif isinstance(y, (list, tuple)):
            # Use conventional YOLO names for two heads if available
            if len(y) >= 2:
                return {"out_m": y[0], "out_s": y[1]}
            return {"out": y[0]}
        else:
            return {"out": y}

    tf.saved_model.save(model, tmp_sm, signatures={"serving_default": serving_fn})
    print(f"[savedmodel] exported → {tmp_sm}")

    # ---- Convert SavedModel → Core ML ----
    precision = ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32
    print(f"[convert] to mlprogram, precision={precision.name}, input={img_size}x{img_size}, scale={args.scale}")
    mlmodel = ct.convert(
        tmp_sm,
        source="tensorflow",
        inputs=[img_input],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,  # ANE+GPU+CPU on Apple Silicon
    )

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mlmodel.save(args.out)
    print(f"[done] Saved Core ML model → {args.out}")

    # Print final Core ML interface summary
    spec = mlmodel.get_spec()
    coreml_in = [f"{f.name} {list(f.type.imageType.shape)}" if f.type.WhichOneof("Type")=="imageType"
                 else f"{f.name}" for f in spec.description.input]
    coreml_out = [f.name for f in spec.description.output]
    print(f"[coreml] inputs : {coreml_in}")
    print(f"[coreml] outputs: {coreml_out}")

if __name__ == "__main__":
    main()