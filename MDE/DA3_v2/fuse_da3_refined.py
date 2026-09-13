#!/usr/bin/env python3
"""
fuse_da3_refined.py  —  Fix "overlaps but won't merge".

Two changes vs fuse_da3.py:
  (1) Edge / flying-pixel filtering when back-projecting, so depth
      discontinuities don't spray "comet tail" points (the ground ribbing).
  (2) Instead of blindly concatenating the 3 per-view clouds, register them
      with multi-scale COLORED ICP inside a pose graph (multiway registration).
      They already start in a common world frame from your extrinsics, so ICP
      only has to remove the small residual (calibration error / per-view depth
      bias) that was causing the doubled surfaces.

Reuses the loaders from fuse_da3.py (must be in the same folder).
Needs open3d:  pip install open3d   (aarch64 wheels exist for Jetson).

Run from DA3_v2:
    python fuse_da3_refined.py
"""

import os
import numpy as np
from PIL import Image
import open3d as o3d

from fuse_da3 import (
    DEFAULTS, CAM_ORDER, MIN_DEPTH_M, MAX_DEPTH_M, CONF_PERCENTILE,
    load_intrinsics, load_extrinsics, scale_K,
)

# ---- registration tuning (metres — tune to your scene scale) ----------------
VOXEL_COARSE = 0.10       # coarse ICP / downsample voxel
VOXEL_FINE   = 0.03       # fine ICP voxel
ICP_ITERS    = (60, 35, 20)
ICP_VOXELS   = (0.10, 0.05, 0.02)
MERGE_VOXEL  = 0.005       # final dedup voxel for the merged cloud (0 = off)

# ---- edge filtering ---------------------------------------------------------
# Drop a pixel if its depth differs from a neighbour by more than this FRACTION
# of the local depth (relative gradient). 0.03 = 3%.
EDGE_REL_THRESH = 0.03


def unproject(depth, K, w2c, rgb, conf, conf_pct, edge_thresh):
    """Back-project one view to world coords with edge + confidence filtering."""
    H, W = depth.shape
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)

    # relative-gradient edge filter (kills flying pixels at silhouettes)
    if edge_thresh > 0:
        gy, gx = np.gradient(z)
        grad = np.sqrt(gx * gx + gy * gy)
        valid &= grad < (edge_thresh * np.maximum(z, MIN_DEPTH_M))

    if conf is not None and conf_pct > 0:
        finite = np.isfinite(conf)
        thr = np.percentile(conf[finite], conf_pct)
        valid &= conf >= thr

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us, vs, z = us[valid], vs[valid], z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts_cam = np.stack([x, y, z], 1)

    c2w = np.linalg.inv(w2c)
    pts_world = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return pts_world.astype(np.float64), rgb[valid].astype(np.float64) / 255.0


def make_o3d(pts, col):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc


def pairwise(src, tgt):
    """Multi-scale colored ICP, identity init (clouds already ~aligned)."""
    T = np.identity(4)
    for v, it in zip(ICP_VOXELS, ICP_ITERS):
        s = src.voxel_down_sample(v)
        t = tgt.voxel_down_sample(v)
        for pc in (s, t):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
        try:
            res = o3d.pipelines.registration.registration_colored_icp(
                s, t, v, T,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
            T = res.transformation
        except RuntimeError:
            # fall back to point-to-plane if colored ICP can't converge at this scale
            res = o3d.pipelines.registration.registration_icp(
                s, t, v, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
            T = res.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, VOXEL_FINE * 1.5, T)
    return T, info


def build_pose_graph(pcds):
    reg = o3d.pipelines.registration
    pg = reg.PoseGraph()
    odom = np.identity(4)
    pg.nodes.append(reg.PoseGraphNode(odom))
    n = len(pcds)
    for i in range(n):
        for j in range(i + 1, n):
            T, info = pairwise(pcds[i], pcds[j])
            print(f"  ICP {CAM_ORDER[i]}->{CAM_ORDER[j]}: "
                  f"|t|={np.linalg.norm(T[:3, 3]):.4f} m")
            if j == i + 1:
                odom = T @ odom
                pg.nodes.append(reg.PoseGraphNode(np.linalg.inv(odom)))
                pg.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=False))
            else:
                pg.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=True))
    return pg


def main():
    import argparse, torch
    from depth_anything_3.api import DepthAnything3
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    args = ap.parse_args()

    intr_paths = {"center": args.intr_center, "left": args.intr_left, "right": args.intr_right}
    img_paths, K_orig, orig_size = [], [], []
    for name in CAM_ORDER:
        p = os.path.join(args.snapshot, f"{name}.png")
        img_paths.append(p)
        K_orig.append(load_intrinsics(intr_paths[name]))
        with Image.open(p) as im:
            orig_size.append(im.size)

    poses = load_extrinsics(args.extrinsics, CAM_ORDER)
    extr_w2c = np.stack([poses[n] for n in CAM_ORDER], 0)
    intr_in = np.stack(K_orig, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(args.model).to(device)
    pred = model.inference(
        image=img_paths, extrinsics=extr_w2c, intrinsics=intr_in,
        align_to_input_ext_scale=True, export_format="mini_npz",
        conf_thresh_percentile=CONF_PERCENTILE)

    # --- CRITICAL: verify DA3 kept your poses AND your view order ---
    pe = np.asarray(pred.extrinsics)
    print("Pose/order check (should be ~0):")
    for i, name in enumerate(CAM_ORDER):
        d = np.abs(pe[i][:3, :4] - extr_w2c[i][:3, :4]).max()
        flag = "" if d < 1e-2 else "   <-- MISMATCH, fix correspondence first!"
        print(f"  {name}: {d:.4f}{flag}")

    depth = np.asarray(pred.depth)
    conf = np.asarray(pred.conf) if getattr(pred, "conf", None) is not None else None
    imgs = np.asarray(pred.processed_images)
    Hp, Wp = depth.shape[1], depth.shape[2]

    pcds = []
    for i, name in enumerate(CAM_ORDER):
        w_o, h_o = orig_size[i]
        K_i = scale_K(K_orig[i], w_o, h_o, Wp, Hp)
        c_i = conf[i] if conf is not None else None
        pts, col = unproject(depth[i], K_i, poses[name], imgs[i], c_i,
                             CONF_PERCENTILE, EDGE_REL_THRESH)
        print(f"  {name}: {len(pts)} points after filtering")
        pcds.append(make_o3d(pts, col))

    print("Registering (multiway colored ICP)...")
    pg = build_pose_graph(pcds)
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=VOXEL_FINE * 1.5,
            edge_prune_threshold=0.25, reference_node=0))

    merged = o3d.geometry.PointCloud()
    for i, pc in enumerate(pcds):
        pc.transform(pg.nodes[i].pose)
        merged += pc
    if MERGE_VOXEL > 0:
        merged = merged.voxel_down_sample(MERGE_VOXEL)

    out = args.out.replace(".ply", "_refined.ply")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_point_cloud(out, merged)
    print(f"\nWrote refined fused cloud -> {out}  ({len(merged.points)} points)")


if __name__ == "__main__":
    main()