#!/usr/bin/env python3
"""
Interactive visual merge of three camera clouds with Open3D.

Workflow per side camera (left, then right), aligned to the fixed center cloud:
  1. A window opens showing the SIDE cloud. Shift+Left-click 3-6 distinct
     features (box corners work best). Shift+Right-click undoes the last pick.
     Press Q / close the window when done.
  2. A window opens showing the CENTER cloud. Click the SAME features in the
     SAME order.
  3. The tool computes a similarity transform (scale+R+t) from those pairs,
     optionally refines with point-to-plane ICP, applies it to the full-res
     side cloud, and moves on.
Finally it shows the merged cloud and writes fused_open3d.ply.

Requires a display. Best run on a desktop (x86, py3.10/3.11):
    pip install open3d
    python open3d_manual_merge.py --dir /path/to/Moge2

Controls in each pick window:
    Shift + Left click   : pick a point
    Shift + Right click  : undo last pick
    Q or close window    : done picking
"""

import argparse
import os
import numpy as np
import open3d as o3d


def load(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise SystemExit(f"empty/missing: {path}")
    return pcd


def pick_points(pcd, title):
    print(f"\n>>> PICK on [{title}]: shift+click features, shift+right-click "
          f"to undo, then press Q / close window.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"pick: {title}", width=1280, height=800)
    vis.add_geometry(pcd)
    vis.run()            # blocks until the user closes the window
    vis.destroy_window()
    idx = vis.get_picked_points()
    print(f"    picked {len(idx)} points")
    return idx


def similarity_from_corr(src, dst, picks_src, picks_dst, with_scaling):
    if len(picks_src) < 3 or len(picks_src) != len(picks_dst):
        raise SystemExit("need >=3 matched picks, equal count on both clouds "
                         f"(got {len(picks_src)} vs {len(picks_dst)})")
    corr = np.stack([picks_src, picks_dst], axis=1).astype(np.int32)
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=with_scaling)
    T = est.compute_transformation(
        src, dst, o3d.utility.Vector2iVector(corr))
    return T


def icp_refine(src, dst, T_init, voxel):
    src_d = src.voxel_down_sample(voxel)
    dst_d = dst.voxel_down_sample(voxel)
    dst_d.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30))
    result = T_init
    for thr in (voxel * 8, voxel * 3):
        reg = o3d.pipelines.registration.registration_icp(
            src_d, dst_d, thr, result,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        result = reg.transformation
    return result, reg.inlier_rmse


def describe(T):
    R = T[:3, :3]
    scale = np.cbrt(max(np.linalg.det(R), 1e-12))
    ang = np.degrees(np.arccos(np.clip(((np.trace(R / scale)) - 1) / 2, -1, 1)))
    t = T[:3, 3]
    return scale, ang, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--no-scale", action="store_true",
                    help="lock scale to 1.0 (rigid only)")
    ap.add_argument("--no-icp", action="store_true",
                    help="skip ICP refinement after picking")
    ap.add_argument("--pick-voxel", type=float, default=0.004,
                    help="downsample (m) for the interactive pick windows")
    ap.add_argument("--out-voxel", type=float, default=0.004,
                    help="dedup voxel (m) for the saved merge; 0 = keep all")
    ap.add_argument("--tint", action="store_true",
                    help="color center gray / left red / right blue to judge fit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    full = {c: load(os.path.join(args.dir, f"cloud_{c}.ply"))
            for c in ("center", "left", "right")}
    # lighter copies for picking (5M pts is sluggish to click)
    pick = {c: (full[c].voxel_down_sample(args.pick_voxel)
                if args.pick_voxel > 0 else full[c]) for c in full}
    print({c: len(pick[c].points) for c in pick})

    center_pick = pick["center"]
    for cam in ("left", "right"):
        ps = pick_points(pick[cam], f"{cam} (source)")
        pd = pick_points(center_pick, "center (target)")
        T = similarity_from_corr(pick[cam], center_pick, ps, pd,
                                 with_scaling=not args.no_scale)
        if not args.no_icp:
            T, rmse = icp_refine(full[cam], full["center"], T, args.pick_voxel)
            s, ang, t = describe(T)
            print(f"[{cam}] after ICP: scale={s:.4f} rot={ang:.2f}deg "
                  f"|t|={np.linalg.norm(t)*1000:.1f}mm inlier_rmse={rmse*1000:.1f}mm")
        else:
            s, ang, t = describe(T)
            print(f"[{cam}] picked: scale={s:.4f} rot={ang:.2f}deg "
                  f"|t|={np.linalg.norm(t)*1000:.1f}mm")
        full[cam].transform(T)

    if args.tint:
        full["center"].paint_uniform_color([0.6, 0.6, 0.6])
        full["left"].paint_uniform_color([0.85, 0.2, 0.2])
        full["right"].paint_uniform_color([0.2, 0.4, 0.85])

    merged = full["center"] + full["left"] + full["right"]
    if args.out_voxel > 0:
        merged = merged.voxel_down_sample(args.out_voxel)

    out = args.out or os.path.join(args.dir, "fused_open3d.ply")
    o3d.io.write_point_cloud(out, merged)
    print(f"\nwrote {out}  ({len(merged.points):,} pts)")

    print("showing merged result -- close window to exit")
    o3d.visualization.draw_geometries([merged], window_name="merged")


if __name__ == "__main__":
    main()