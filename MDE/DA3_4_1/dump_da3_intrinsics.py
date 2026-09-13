#!/usr/bin/env python3
"""
dump_da3_intrinsics.py
======================

Diagnostic only. Answers one question: does DA3's returned `intrinsics` field
agree with the intrinsics that da3_multicam_fusion.py derived by rescaling the
calibrated K to the depth-map resolution?

Runs two passes on the same snapshot:

  PASS A  raw images, no rotation, no priors
  PASS B  upright-rotated images with calibrated intrinsics + extrinsics
          (exactly what the fusion script fed the model)

For each view it prints the depth grid size, DA3's returned fx/fy/cx/cy, the
derived fx/fy/cx/cy, and the ratio between them. A ratio of 1.00 everywhere
means the resize assumption was correct and the layering lies elsewhere. Any
other value, or a cx/cy offset that is not simply w/2, h/2, means DA3 padded,
cropped or letterboxed and the back-projection must use DA3's numbers instead.

Usage
-----
    source ~/venvs/da3/bin/activate          # your DA3 venv
    python dump_da3_intrinsics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SNAP = Path("/home/jetson/Projects/Image_Capture_4_1/snapshots/20260804_092615_018")
CALIB = Path("/home/jetson/Projects/Calibration_4_1/results")
MODEL = "depth-anything/da3nested-giant-large-1.1"
CAMS = ["left", "center", "right", "top"]


# --------------------------------------------------------------------------- #

def roll_deg_from_R(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def rot90_K(K, W, H, k):
    """Adjust K for k counter-clockwise 90 deg rotations (np.rot90 semantics)."""
    K = K.copy()
    for _ in range(int(k) % 4):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Kn = np.eye(3)
        Kn[0, 0], Kn[1, 1] = fy, fx
        Kn[0, 2], Kn[1, 2] = cy, (W - 1) - cx
        K = Kn
        W, H = H, W
    return K, W, H


def scale_K(K, src_wh, dst_wh):
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    if isinstance(x, (list, tuple)):
        try:
            return np.stack([to_np(e) for e in x])
        except Exception:
            return None
    return np.asarray(x)


def get(obj, names):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            v = getattr(obj, n)
            if not callable(v):
                return v
    return None


def report(tag, pred, derived_K_list, rot_shapes):
    depth = to_np(get(pred, ("depth", "depths", "depth_map")))
    intr = to_np(get(pred, ("intrinsics", "pred_intrinsics")))

    print(f"\n{'=' * 74}\n{tag}\n{'=' * 74}")
    if depth is None:
        print("  no depth field found")
        return
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[:, 0] if depth.shape[1] == 1 else depth[..., 0]
    print(f"  depth array shape : {depth.shape}")

    if intr is None:
        print("  DA3 returned intrinsics = None  ->  cannot cross-check, "
              "must rely on the resize assumption")
        return
    intr = np.asarray(intr)
    print(f"  intrinsics shape  : {intr.shape}")

    for i, name in enumerate(CAMS):
        h, w = depth[i].shape[-2:]
        Ki = intr[i] if intr.ndim == 3 else intr
        fx_d, fy_d = float(Ki[0, 0]), float(Ki[1, 1])
        cx_d, cy_d = float(Ki[0, 2]), float(Ki[1, 2])

        print(f"\n  --- {name} ---   depth grid {w} x {h}")
        print(f"      DA3      fx={fx_d:9.3f} fy={fy_d:9.3f} "
              f"cx={cx_d:8.3f} cy={cy_d:8.3f}")

        if derived_K_list is not None:
            src_wh = rot_shapes[i]
            Kd = scale_K(derived_K_list[i], src_wh, (w, h))
            fx_o, fy_o = float(Kd[0, 0]), float(Kd[1, 1])
            cx_o, cy_o = float(Kd[0, 2]), float(Kd[1, 2])
            print(f"      derived  fx={fx_o:9.3f} fy={fy_o:9.3f} "
                  f"cx={cx_o:8.3f} cy={cy_o:8.3f}")
            print(f"      ratio    fx={fx_d / fx_o:6.4f}   fy={fy_d / fy_o:6.4f}   "
                  f"dcx={cx_d - cx_o:+7.2f} px  dcy={cy_d - cy_o:+7.2f} px")

        print(f"      grid centre would be cx={w / 2:.1f} cy={h / 2:.1f} "
              f"(DA3 offset {cx_d - w / 2:+.2f}, {cy_d - h / 2:+.2f})")


# --------------------------------------------------------------------------- #

def main():
    # ---- load calibration -------------------------------------------------- #
    K_orig, sizes = [], []
    for n in CAMS:
        d = json.loads((CALIB / f"intrinsics_{n}.json").read_text())
        K_orig.append(np.asarray(d["camera_matrix"], dtype=np.float64))
        sizes.append(tuple(d["image_size"]))  # (W, H)

    ext = json.loads((CALIB / "extrinsics.json").read_text())
    rot_k = []
    for n in CAMS:
        R = np.asarray(ext[n]["R"], dtype=np.float64)
        rot_k.append(int(round(roll_deg_from_R(R) / 90.0)) % 4)
    print(f"upright rotation counts: {dict(zip(CAMS, rot_k))}")

    # ---- load images ------------------------------------------------------- #
    raw, rotated, K_rot, rot_wh = [], [], [], []
    for i, n in enumerate(CAMS):
        bgr = cv2.imread(str(SNAP / f"{n}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f"could not read {SNAP / f'{n}.png'}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw.append(rgb)
        rotated.append(np.ascontiguousarray(np.rot90(rgb, k=rot_k[i])))
        Kr, Wr, Hr = rot90_K(K_orig[i], sizes[i][0], sizes[i][1], rot_k[i])
        K_rot.append(Kr)
        rot_wh.append((Wr, Hr))
    print(f"raw sizes     : {[r.shape[:2] for r in raw]}")
    print(f"rotated sizes : {[r.shape[:2] for r in rotated]}")

    # ---- model ------------------------------------------------------------- #
    from depth_anything_3.api import DepthAnything3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nloading {MODEL} on {dev} ...")
    model = DepthAnything3.from_pretrained(MODEL).to(dev)
    model.eval()

    # ---- PASS A: raw images, no priors ------------------------------------- #
    with torch.inference_mode():
        pa = model.inference(raw)
    report("PASS A  raw images, no rotation, no priors",
           pa, K_orig, sizes)

    # ---- PASS B: rotated images, calibrated priors ------------------------- #
    E = []
    for n in CAMS:
        M = np.eye(4)
        M[:3, :3] = np.asarray(ext[n]["R"], dtype=np.float64)
        M[:3, 3] = np.asarray(ext[n]["t"], dtype=np.float64)
        E.append(M)

    with torch.inference_mode():
        pb = model.inference(
            rotated,
            extrinsics=np.stack(E).astype(np.float32),
            intrinsics=np.stack(K_rot).astype(np.float32),
        )
    report("PASS B  upright-rotated images, calibrated intrinsics + extrinsics "
           "(what the fusion script ran)", pb, K_rot, rot_wh)

    print("\n" + "=" * 74)
    print("READ THIS")
    print("=" * 74)
    print("  ratio fx/fy = 1.00 and dcx/dcy = 0  ->  resize assumption correct,")
    print("                                          layering is NOT an intrinsics bug")
    print("  anything else                       ->  back-projection must use DA3's")
    print("                                          returned intrinsics")


if __name__ == "__main__":
    main()