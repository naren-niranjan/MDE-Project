#!/usr/bin/env python3
"""
probe_da3_prediction.py

Runs one conditioned DA3 inference and dumps every field of the returned
Prediction object, to determine whether a metrically aligned world-frame
pointmap already exists.

Context
-------
The pipeline currently reads a depth tensor, applies K by hand, and transforms
by the extrinsic (~540 ms of back-projection). The reconstruction comes out
~1.4x shallow with ~270 mm of inter-camera disagreement, despite exact
intrinsics and extrinsics being accepted by the model. Hypothesis: the depth
field being read is in the model's normalised scale while a separate metric
field is being ignored.

Verdict
-------
    A field with median near --expect-depth
        -> use it directly, delete the back-projection stage.
    Everything clusters near expect-depth / 1.4
        -> conditioning is not reaching the depth head; read DA3's
           inference() source to find what align_to_input_ext_scale rescales.

Calibration is read from JSON. The loader searches recursively for the usual
key names and prints what it found. If it guesses wrong, run with
--inspect-calib to dump the structure, then adjust KEY_ALIASES.

Example
-------
SNAP=/home/jetson/Projects/Image_Capture_4_2/snapshots/20260805_100601_125
CAL=/home/jetson/Projects/Calibration_4_1/results

python probe_da3_prediction.py \
    --image left=$SNAP/left.png     --image center=$SNAP/center.png \
    --image right=$SNAP/right.png   --image top=$SNAP/top.png \
    --intrinsics-json $CAL/intrinsics_{name}.json \
    --extrinsics-json $CAL/extrinsics.json \
    --reference center --expect-depth 3.15 --also-noprior \
    --out probe_prediction.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


KEY_ALIASES = {
    "K": ["K", "camera_matrix", "cameraMatrix", "intrinsic_matrix",
          "intrinsics", "matrix", "M"],
    "R": ["R", "rotation", "rotation_matrix", "rmat"],
    "t": ["t", "T", "tvec", "translation", "translation_vector"],
    "RT": ["RT", "T", "transform", "extrinsic", "extrinsics", "pose",
           "matrix", "T_wc", "T_cw", "world_to_camera", "camera_to_world"],
    "size": ["image_size", "resolution", "size", "img_size"],
}


# --------------------------------------------------------------------------
# JSON traversal
# --------------------------------------------------------------------------

def coerce_matrix(v):
    """Turn a JSON value into an ndarray, handling the OpenCV dict form."""
    if isinstance(v, dict):
        if "data" in v and ("rows" in v or "cols" in v):
            rows = int(v.get("rows", 0))
            cols = int(v.get("cols", 0))
            arr = np.asarray(v["data"], float)
            if rows and cols:
                return arr.reshape(rows, cols)
            return arr
        return None
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v, float)
        except Exception:
            return None
    return None


def find_key(obj, aliases, want_shape=None, _depth=0):
    """Depth-first search for the first key matching any alias."""
    if _depth > 6 or not isinstance(obj, dict):
        return None
    lower = {k.lower(): k for k in obj}
    for a in aliases:
        k = lower.get(a.lower())
        if k is None:
            continue
        m = coerce_matrix(obj[k])
        if m is None:
            continue
        if want_shape is None:
            return m
        if m.size == int(np.prod(want_shape)):
            return m.reshape(want_shape)
    for v in obj.values():
        if isinstance(v, dict):
            r = find_key(v, aliases, want_shape, _depth + 1)
            if r is not None:
                return r
    return None


def find_subdict(obj, name, _depth=0):
    """Find the sub-dictionary keyed by a camera name, at any depth."""
    if _depth > 5 or not isinstance(obj, dict):
        return None
    lower = {k.lower(): k for k in obj}
    if name.lower() in lower:
        v = obj[lower[name.lower()]]
        if isinstance(v, dict):
            return v
    for v in obj.values():
        if isinstance(v, dict):
            r = find_subdict(v, name, _depth + 1)
            if r is not None:
                return r
    return None


def dump_structure(obj, indent=0, max_depth=4):
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}({len(v)})")
                dump_structure(v, indent + 1, max_depth)
            else:
                print(f"{pad}{k}: {type(v).__name__} = {v!r}"[:110])
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], (dict, list)):
            print(f"{pad}[0]:")
            dump_structure(obj[0], indent + 1, max_depth)
        else:
            print(f"{pad}{obj!r}"[:110])


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def parse_named(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def load_images(pairs):
    from PIL import Image
    imgs, names, shapes = [], [], []
    for name, path in pairs:
        if not path.exists():
            sys.exit(f"missing image: {path}")
        arr = np.asarray(Image.open(path).convert("RGB"))
        imgs.append(arr)
        names.append(name)
        shapes.append(arr.shape[:2])
        print(f"[image] {name:<8} {arr.shape} {arr.dtype}  {path.name}")
    return names, imgs, shapes


def load_intrinsics(template, name):
    path = Path(str(template).replace("{name}", name)).expanduser()
    if not path.exists():
        sys.exit(f"missing intrinsics json: {path}")
    data = json.loads(path.read_text())

    K = find_key(data, KEY_ALIASES["K"], (3, 3))
    if K is None:
        def scal(keys):
            lower = {k.lower(): k for k in data}
            for kk in keys:
                if kk in lower:
                    return float(data[lower[kk]])
            return None
        fx, fy = scal(["fx", "f_x"]), scal(["fy", "f_y"])
        cx, cy = scal(["cx", "c_x", "ppx"]), scal(["cy", "c_y", "ppy"])
        if None not in (fx, fy, cx, cy):
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    if K is None:
        print(f"\ncould not find intrinsics in {path}. Structure:", file=sys.stderr)
        dump_structure(data)
        sys.exit(1)
    return K.astype(np.float64), data


def load_extrinsics(path, names, reference, convention):
    path = Path(path).expanduser()
    if not path.exists():
        sys.exit(f"missing extrinsics json: {path}")
    data = json.loads(path.read_text())

    Es = {}
    for n in names:
        sub = find_subdict(data, n)
        if sub is None and n == reference:
            Es[n] = np.eye(4)
            continue
        if sub is None:
            print(f"\nno entry for camera {n!r} in {path}. Structure:",
                  file=sys.stderr)
            dump_structure(data)
            sys.exit(1)

        E = find_key(sub, KEY_ALIASES["RT"], (4, 4))
        if E is None:
            R = find_key(sub, KEY_ALIASES["R"], (3, 3))
            t = find_key(sub, KEY_ALIASES["t"], (3,))
            if R is None or t is None:
                print(f"\ncould not parse extrinsic for {n!r}. Structure:",
                      file=sys.stderr)
                dump_structure(sub)
                sys.exit(1)
            E = np.eye(4)
            E[:3, :3] = R
            E[:3, 3] = t
        Es[n] = E.astype(np.float64)

    if convention == "c2w":
        Es = {k: np.linalg.inv(v) for k, v in Es.items()}

    if reference in Es:
        Eref = Es[reference]
        Es = {k: v @ np.linalg.inv(Eref) for k, v in Es.items()}

    return np.stack([Es[n] for n in names])


def camera_centre(E):
    return -E[:3, :3].T @ E[:3, 3]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(model_id, device):
    errors = []
    for label, fn in [
        ("depth_anything_3.api",
         lambda: __import__("depth_anything_3.api", fromlist=["DepthAnything3"])
                 .DepthAnything3.from_pretrained(model_id)),
        ("depth_anything_3",
         lambda: __import__("depth_anything_3", fromlist=["DepthAnything3"])
                 .DepthAnything3.from_pretrained(model_id)),
        ("transformers.AutoModel",
         lambda: __import__("transformers", fromlist=["AutoModel"])
                 .AutoModel.from_pretrained(model_id, trust_remote_code=True)),
    ]:
        try:
            m = fn()
            if hasattr(m, "to"):
                m = m.to(device)
            print(f"[model] {label}  {model_id}")
            return m
        except Exception as e:
            errors.append(f"{label}: {e}")
    print("could not load the model. Tried:", file=sys.stderr)
    for e in errors:
        print("  -", e, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def to_numpy(v):
    if torch is not None and isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "detach"):
        try:
            return v.detach().cpu().numpy()
        except Exception:
            return None
    return None


def describe(name, v, records, depth=0, prefix=""):
    full = f"{prefix}{name}"
    arr = to_numpy(v)

    if arr is not None:
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            print(f"  {full:<32} {str(arr.shape):<24} {arr.dtype}")
            return
        f = arr[np.isfinite(arr)]
        if f.size == 0:
            print(f"  {full:<32} {str(arr.shape):<24} {arr.dtype}  non-finite")
            return
        rec = {"field": full, "shape": list(arr.shape), "dtype": str(arr.dtype),
               "min": float(f.min()), "median": float(np.median(f)),
               "max": float(f.max()),
               "finite_frac": float(f.size / arr.size)}
        records.append(rec)
        print(f"  {full:<32} {str(arr.shape):<24} {arr.dtype}  "
              f"min={f.min():>9.4f} med={rec['median']:>9.4f} "
              f"max={f.max():>9.4f}  finite={rec['finite_frac']:.1%}")
        return

    if isinstance(v, dict) and depth < 1:
        print(f"  {full:<32} dict len={len(v)}")
        for k, sub in v.items():
            describe(str(k), sub, records, depth + 1, prefix=f"{full}.")
        return

    if isinstance(v, (list, tuple)) and depth < 1:
        print(f"  {full:<32} {type(v).__name__} len={len(v)}")
        for i, sub in enumerate(v[:8]):
            describe(f"[{i}]", sub, records, depth + 1, prefix=full)
        return

    if isinstance(v, (int, float, str, bool, type(None))):
        print(f"  {full:<32} {v!r}")
        return

    print(f"  {full:<32} {type(v).__name__}")


def probe(pred, expect_depth, tag=""):
    print(f"\n===== PREDICTION PROBE {tag} =====")
    print("type:", type(pred))
    names = [a for a in dir(pred) if not a.startswith("_")]
    print("attrs:", names, "\n")

    records = []
    for a in names:
        try:
            v = getattr(pred, a)
        except Exception as e:
            print(f"  {a:<32} <error: {e}>")
            continue
        if callable(v):
            continue
        describe(a, v, records)

    cands = []
    for r in records:
        shp = r["shape"]
        is_pm = len(shp) >= 2 and shp[-1] == 3
        is_dm = len(shp) in (2, 3, 4) and not is_pm
        if not (is_pm or is_dm) or r["median"] <= 0:
            continue
        c = dict(r)
        c["kind"] = "pointmap" if is_pm else "depthlike"
        c["ratio_vs_expected"] = round(expect_depth / c["median"], 4)
        cands.append(c)
    cands.sort(key=lambda r: abs(np.log(max(r["ratio_vs_expected"], 1e-9))))

    print(f"\n--- candidates vs expected depth {expect_depth} m ---")
    hdr = f"  {'field':<32}{'kind':<11}{'median':>10}{'expect/med':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in cands[:12]:
        print(f"  {r['field']:<32}{r['kind']:<11}"
              f"{r['median']:>10.4f}{r['ratio_vs_expected']:>12.4f}")

    if cands:
        b = cands[0]
        if 0.95 <= b["ratio_vs_expected"] <= 1.05:
            print(f"\n  VERDICT: {b['field']} is already metric. Use it "
                  f"directly and delete the back-projection stage.")
        else:
            print(f"\n  VERDICT: nearest candidate {b['field']} is "
                  f"{b['ratio_vs_expected']:.3f}x off. No metric field "
                  f"returned; read DA3 inference() source next.")
    print("=" * 60 + "\n")
    return records, cands


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Probe the DA3 Prediction object for a metric pointmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", action="append", required=True,
                    type=parse_named, metavar="NAME=PATH")
    ap.add_argument("--intrinsics-json", required=True,
                    help="path template containing {name}")
    ap.add_argument("--extrinsics-json", required=True)
    ap.add_argument("--reference", default="center")
    ap.add_argument("--extrinsic-convention", choices=["w2c", "c2w"],
                    default="w2c")
    ap.add_argument("--inspect-calib", action="store_true",
                    help="dump calibration JSON structure and exit")
    ap.add_argument("--model",
                    default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--expect-depth", type=float, required=True)
    ap.add_argument("--also-noprior", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("probe_prediction.json"))
    args = ap.parse_args()

    if args.inspect_calib:
        for name, _ in args.image:
            p = Path(str(args.intrinsics_json).replace("{name}", name))
            if p.exists():
                print(f"\n===== {p} =====")
                dump_structure(json.loads(p.read_text()))
                break
        print(f"\n===== {args.extrinsics_json} =====")
        dump_structure(json.loads(Path(args.extrinsics_json).read_text()))
        return

    names, imgs, shapes = load_images(args.image)

    Ks = []
    for n in names:
        K, _ = load_intrinsics(args.intrinsics_json, n)
        Ks.append(K)
        print(f"[calib] {n:<8} fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
              f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    Ks = np.stack(Ks)

    Es = load_extrinsics(args.extrinsics_json, names, args.reference,
                         args.extrinsic_convention)
    print()
    for n, E in zip(names, Es):
        print(f"[pose ] {n:<8} C={camera_centre(E).round(4)}")
    print("  (reference should be ~[0 0 0]; check baselines against your "
          "calibration report before trusting the run)\n")

    model = load_model(args.model, args.device)

    report = {"model": args.model, "cameras": names,
              "expect_depth_m": args.expect_depth,
              "extrinsic_convention": args.extrinsic_convention,
              "image_shapes": [list(s) for s in shapes], "runs": {}}

    print("[run] conditioned inference "
          "(intrinsics + extrinsics, align_to_input_ext_scale=True)")
    pred = model.inference(imgs, intrinsics=Ks, extrinsics=Es,
                           align_to_input_ext_scale=True,
                           process_res=args.process_res)
    recs, cands = probe(pred, args.expect_depth, tag="PRIOR")
    report["runs"]["prior"] = {"fields": recs, "candidates": cands}

    if args.also_noprior:
        print("[run] unconditioned inference")
        pred2 = model.inference(imgs, process_res=args.process_res)
        recs2, cands2 = probe(pred2, args.expect_depth, tag="NOPRIOR")
        report["runs"]["noprior"] = {"fields": recs2, "candidates": cands2}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()