#!/usr/bin/env python3
"""
fuse_da3_nopose.py  —  DIAGNOSTIC.

Runs DA3 with NO extrinsics / intrinsics, so it estimates its own poses and
metric scale. Unprojects with DA3's OWN predicted poses and concatenates
(no ICP). Purpose: compare against your pose-conditioned result.

  - If this cloud merges cleanly but the pose-conditioned one doubled,
    your provided calibration/scale is the limiter.
  - It also prints DA3's predicted camera baselines. Compare to your
    extrinsics' 0.784 m / 0.777 m. If DA3's baselines differ a lot -> the
    metric SCALES disagree (that's your systematic ~5.7 cm offset source).
    If they match ~0.78 m -> scale is fine, residual is rotational.

Run from DA3_v2:
    python fuse_da3_nopose.py
"""

import os
import numpy as np
import open3d as o3d

from fuse_da3 import DEFAULTS, CAM_ORDER, CONF_PERCENTILE
from fuse_da3_refined import unproject, make_o3d, EDGE_REL_THRESH


def to4x4(e):
    M = np.eye(4)
    M[:3, :4] = np.asarray(e)[:3, :4]
    return M


def main():
    import torch
    from depth_anything_3.api import DepthAnything3

    snap = DEFAULTS["snapshot"]
    img_paths = [os.path.join(snap, f"{n}.png") for n in CAM_ORDER]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(DEFAULTS["model"]).to(device)

    # <<< the whole point: NO extrinsics, NO intrinsics >>>
    pred = model.inference(
        image=img_paths,
        export_format="mini_npz",
        conf_thresh_percentile=CONF_PERCENTILE,
    )

    depth = np.asarray(pred.depth)                 # (N,Hp,Wp) metres
    conf = np.asarray(pred.conf) if getattr(pred, "conf", None) is not None else None
    imgs = np.asarray(pred.processed_images)       # (N,Hp,Wp,3) uint8
    K = np.asarray(pred.intrinsics)                # (N,3,3) predicted, processed res
    E = np.asarray(pred.extrinsics)                # (N,3,4) predicted w2c

    Hp, Wp = depth.shape[1], depth.shape[2]
    print(f"Processed res {Wp}x{Hp}. Predicted K[0] (check cx~{Wp/2:.0f}, cy~{Hp/2:.0f}):")
    print(np.round(K[0], 2))

    # DA3's own predicted camera centres + baselines
    centers = {n: np.linalg.inv(to4x4(E[i]))[:3, 3] for i, n in enumerate(CAM_ORDER)}
    print("\nDA3-predicted baselines (m)  [your extrinsics: c-l 0.784, c-r 0.777]:")
    for a, b in (("center", "left"), ("center", "right"), ("left", "right")):
        print(f"  {a}-{b}: {np.linalg.norm(centers[a] - centers[b]):.4f}")

    merged = o3d.geometry.PointCloud()
    for i, name in enumerate(CAM_ORDER):
        c_i = conf[i] if conf is not None else None
        pts, col = unproject(depth[i], K[i], to4x4(E[i]), imgs[i],
                             c_i, CONF_PERCENTILE, EDGE_REL_THRESH)
        print(f"  {name}: {len(pts)} pts")
        merged += make_o3d(pts, col)

    out = os.path.join(os.path.dirname(DEFAULTS["out"]), "fused_nopose.ply")
    o3d.io.write_point_cloud(out, merged)
    print(f"\nWrote {out}  ({len(merged.points)} points)  -- no ICP, DA3's own poses")


if __name__ == "__main__":
    main()