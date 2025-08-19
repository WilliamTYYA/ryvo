#!/usr/bin/env python3
"""
convert_to_coreml.py
Convert a Keras .keras model to a Core ML .mlmodel ready for Xcode.

Usage examples:
  python scripts/convert_to_coreml.py --keras best_384_G24.keras --img 384 \
      --out Models/CustomYOLOAdvanced.mlmodel --fp16

  # If your preprocessing already feeds 0..1 floats to Core ML (rare on iOS),
  # set --scale 1.0 so the converter doesn't divide by 255:
  python scripts/convert_to_coreml.py --keras best_384_G24.keras --img 384 --scale 1.0
"""

import argparse, os, sys
import tensorflow as tf
import coremltools as ct
from tensorflow.keras import layers

# Try to import the custom class from your project; fall back to a minimal definition.
try:
    from custom_yolo import SiLU as _SiLU
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
    ap.add_argument("--rename_outputs", action="store_true",
                    help="Rename first two outputs to 'out_m' and 'out_s' (if names differ)")
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

    # ---- Build Core ML input description ----
    img_input = ct.ImageType(
        name=in_name,
        shape=(1, img_size, img_size, 3),   # NHWC
        color_layout="RGB",
        scale=args.scale,
        bias=[args.bias_r, args.bias_g, args.bias_b],
    )

    # ---- Convert ----
    precision = ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32
    print(f"[convert] to mlprogram, precision={precision.name}, input={img_size}x{img_size}, scale={args.scale}")
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        inputs=[img_input],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,  # use ANE + GPU + CPU
    )

    # (Optional) Standardize output names to 'out_m'/'out_s' if user wants
    if args.rename_outputs and len(out_names) >= 2:
        try:
            ct.utils.rename_feature(mlmodel, out_names[0], "out_m")
            ct.utils.rename_feature(mlmodel, out_names[1], "out_s")
            print(f"[info] Renamed outputs: {out_names[:2]} -> ['out_m','out_s']")
        except Exception as e:
            print(f"[warn] Could not rename outputs: {e}")

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