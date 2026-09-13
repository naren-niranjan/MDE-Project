#!/usr/bin/env python3
"""
fuse_da3_intr_only.py  —  DIAGNOSTIC / MODE.

Give DA3 your KNOWN intrinsics but let it PREDICT the extrinsics (poses).
Isolates the extrinsics as the variable while keeping calibration you trust.

  - Merges cleanly here but not in the pose-conditioned run
        -> your extrinsic ROTATIONS were the problem. You can adopt DA3's
           predicted poses (printed below) or re-check calibration.
  - Predicted baselines now land near 0.78 m
        -> scale is consistent; residual was rotational.
    Predicted baselines still far from 0.78 m
        -> genuine scale disagreement between your rig and DA3's metric depth.

Run from DA3_v2:
    python fuse_da3_intr_only.py
"""

import os
import numpy as np
import open3d as o3d

from fuse_da3 import DEFAULTS, CAM_ORDER, CONF_PERCENTILE, load_intrinsics
from fuse_da3_refined import unproject, make_o3d, EDGE_REL_THRESH


def to4x4(e):
    M = np.eye(4)
    M[:3, :4] = np.asarray(e)[:3, :4]
    return M


def main():
    import torch
    from depth_anything_3.api import DepthAnything3

    snap = DEFAULTS["snapshot"]
    intr_paths = {"center": DEFAULTS["intr_center"],
                  "left": DEFAULTS["intr_left"],
                  "right": DEFAULTS["intr_right"]}
    img_paths = [os.path.join(snap, f"{n}.png") for n in CAM_ORDER]
    intr_in = np.stack([load_intrinsics(intr_paths[n]) for n in CAM_ORDER], 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(DEFAULTS["model"]).to(device)

    # <<< intrinsics given, extrinsics NOT given -> DA3 predicts poses >>>
    pred = model.inference(
        image=img_paths,
        intrinsics=intr_in,               # your known K's
        # no extrinsics, no align_to_input_ext_scale
        export_format="mini_npz",
        conf_thresh_percentile=CONF_PERCENTILE,
    )

    depth = np.asarray(pred.depth)
    conf = np.asarray(pred.conf) if getattr(pred, "conf", None) is not None else None
    imgs = np.asarray(pred.processed_images)
    K = np.asarray(pred.intrinsics)        # processed res, consistent with depth
    E = np.asarray(pred.extrinsics)        # DA3-predicted w2c (N,3,4)

    Hp, Wp = depth.shape[1], depth.shape[2]
    print(f"Processed res {Wp}x{Hp}. K[0]:\n{np.round(K[0], 2)}")

    centers = {n: np.linalg.inv(to4x4(E[i]))[:3, 3] for i, n in enumerate(CAM_ORDER)}
    print("\nDA3-predicted baselines (m)  [yours: c-l 0.784, c-r 0.777, l-r 1.562]:")
    for a, b in (("center", "left"), ("center", "right"), ("left", "right")):
        print(f"  {a}-{b}: {np.linalg.norm(centers[a] - centers[b]):.4f}")
    print("\nDA3-predicted camera centres (world, m):")
    for n in CAM_ORDER:
        print(f"  {n:>6}: {centers[n].round(4)}")

    merged = o3d.geometry.PointCloud()
    for i, name in enumerate(CAM_ORDER):
        c_i = conf[i] if conf is not None else None
        pts, col = unproject(depth[i], K[i], to4x4(E[i]), imgs[i],
                             c_i, CONF_PERCENTILE, EDGE_REL_THRESH)
        print(f"  {name}: {len(pts)} pts")
        merged += make_o3d(pts, col)

    out = os.path.join(os.path.dirname(DEFAULTS["out"]), "fused_intr_only.ply")
    o3d.io.write_point_cloud(out, merged)
    print(f"\nWrote {out}  ({len(merged.points)} points)  -- DA3-predicted poses")


if __name__ == "__main__":
    main()