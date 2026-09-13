#!/usr/bin/env python3
"""
align_to_gt.py  —  Measure & reduce the gap between the fused DA3 cloud and
ground truth (e.g. LiDAR).

Steps:
  1. Rigid point-to-plane ICP  (fixes the frame offset: DA3 is in the
     center-camera frame, GT is in the LiDAR/world frame).
  2. Similarity ICP (with_scaling)  ->  also recovers a global SCALE.
     Scale != 1 means DA3's metric scale is biased (usually because the
     0.784 m baseline used for align_to_input_ext_scale is slightly off).
  3. Reports mean/median/RMS/p95 point-to-GT distance before and after.
  4. Writes the aligned cloud and an ERROR-COLORED cloud (blue=close,
     red=far) so you can see whether the residual is uniform (noise floor)
     or structured (fixable bias).

Usage:
    python align_to_gt.py --gt /path/to/ground_truth.ply
    python align_to_gt.py --gt gt.ply --fused results/fused_pointcloud_refined.ply
"""

import os
import argparse
import numpy as np
import open3d as o3d

from fuse_da3 import DEFAULTS


def prep(pcd, voxel):
    p = pcd.voxel_down_sample(voxel) if voxel > 0 else pcd
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return p


def stats(src, gt, label):
    d = np.asarray(src.compute_point_cloud_distance(gt))
    print(f"  [{label}] mean={d.mean()*100:.2f} cm  median={np.median(d)*100:.2f} cm  "
          f"RMS={np.sqrt((d**2).mean())*100:.2f} cm  p95={np.percentile(d,95)*100:.2f} cm")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="ground-truth point cloud (.ply/.pcd)")
    ap.add_argument("--fused", default=DEFAULTS["out"].replace(".ply", "_refined.ply"))
    ap.add_argument("--voxel", type=float, default=0.02, help="registration voxel (m)")
    ap.add_argument("--max-corr", type=float, default=0.30, help="ICP max corr dist (m)")
    args = ap.parse_args()

    da3 = o3d.io.read_point_cloud(args.fused)
    gt = o3d.io.read_point_cloud(args.gt)
    print(f"DA3: {len(da3.points)} pts   GT: {len(gt.points)} pts")

    ds_src = prep(da3, args.voxel)
    ds_gt = prep(gt, args.voxel)

    print("Before alignment:")
    stats(ds_src, ds_gt, "raw")

    reg = o3d.pipelines.registration
    # 1) rigid point-to-plane
    r1 = reg.registration_icp(
        ds_src, ds_gt, args.max_corr, np.identity(4),
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=80))
    # 2) similarity (adds scale), initialized from the rigid result
    r2 = reg.registration_icp(
        ds_src, ds_gt, args.max_corr * 0.6, r1.transformation,
        reg.TransformationEstimationPointToPoint(with_scaling=True),
        reg.ICPConvergenceCriteria(max_iteration=80))

    T = r2.transformation
    R = T[:3, :3]
    scale = float(np.cbrt(np.linalg.det(R)))     # uniform scale factor from Sim3
    print(f"\nRecovered similarity transform:")
    print(f"  scale     = {scale:.4f}   (1.0 = DA3 metric scale already correct)")
    print(f"  translation = {T[:3,3].round(4)} m")
    print(f"  fitness={r2.fitness:.3f}  inlier_rmse={r2.inlier_rmse*100:.2f} cm")

    da3.transform(T)
    ds_src.transform(T)
    print("After alignment:")
    d = stats(ds_src, ds_gt, "aligned")

    outdir = os.path.dirname(args.fused)
    aligned_path = os.path.join(outdir, "fused_aligned_to_gt.ply")
    o3d.io.write_point_cloud(aligned_path, da3)

    # error-colored cloud on the full-res DA3 (blue<=1cm ... red>=cap)
    dfull = np.asarray(da3.compute_point_cloud_distance(gt))
    cap = max(np.percentile(dfull, 95), 0.02)
    t = np.clip(dfull / cap, 0, 1)
    colors = np.stack([t, np.zeros_like(t), 1 - t], axis=1)  # blue->red
    err = o3d.geometry.PointCloud()
    err.points = da3.points
    err.colors = o3d.utility.Vector3dVector(colors)
    err_path = os.path.join(outdir, "fused_error_colored.ply")
    o3d.io.write_point_cloud(err_path, err)

    print(f"\nWrote {aligned_path}")
    print(f"Wrote {err_path}   (blue=0, red>={cap*100:.1f} cm)")


if __name__ == "__main__":
    main()