#!/usr/bin/env python3
"""
plane_separation.py

Measures the perpendicular separation between the two dominant parallel planes
in a point cloud (conveyor deck and floor), per camera, and optionally solves
the affine depth correction that reconciles the measurement with ground truth.

Why this matters
----------------
A uniform depth scale error and an affine (scale + shift) error are
distinguishable using relief alone:

    z_true = a * z_pred + b

Absolute distance carries both a and b. A height DIFFERENCE between two
surfaces carries only a, because b cancels in the subtraction. So:

    relief ratio == absolute ratio   ->  b = 0, a single scalar fixes it
    relief ratio >  absolute ratio   ->  b != 0, no scalar will ever fix it

Running this per camera also separates two confounded effects: if each camera
gives a tight but DIFFERENT separation, the spread is inter-camera layering.
If each camera individually gives a spatially varying separation, the bias is
genuinely non-uniform and needs a spatially varying correction.

Requires Open3D.

Example
-------
python plane_separation.py \
    --cloud left=cloud_left.ply --cloud center=cloud_center.ply \
    --cloud right=cloud_right.ply --cloud top=cloud_top.ply \
    --gt-separation 0.80 \
    --gt-distance 3.50 \
    --out plane_separation.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not found. Activate the Python 3.11 conda environment.")


def parse_cloud_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def fit_plane(points, thresh, iters=4000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, inliers = pcd.segment_plane(thresh, 3, iters)
    a, b, c, d = model
    n = np.array([a, b, c], float)
    s = np.linalg.norm(n)
    return n / s, d / s, np.asarray(inliers)


def find_parallel_pair(points, thresh, max_angle_deg, min_frac, max_planes=6):
    """Extract planes sequentially; return the two largest near-parallel ones."""
    remaining = points.copy()
    found = []
    total = len(points)

    for _ in range(max_planes):
        if len(remaining) < max(500, int(min_frac * total)):
            break
        n, d, inl = fit_plane(remaining, thresh)
        if len(inl) < min_frac * total:
            break
        pts_in = remaining[inl]
        found.append({
            "normal": n, "d": d,
            "n_points": int(len(inl)),
            "centroid": pts_in.mean(axis=0),
        })
        keep = np.ones(len(remaining), bool)
        keep[inl] = False
        remaining = remaining[keep]

    if len(found) < 2:
        return None, found

    # Largest plane is the reference; find the biggest plane parallel to it.
    ref = found[0]
    cos_lim = np.cos(np.radians(max_angle_deg))
    for cand in found[1:]:
        if abs(float(np.dot(ref["normal"], cand["normal"]))) >= cos_lim:
            return (ref, cand), found
    return None, found


def separation(p1, p2):
    """Perpendicular distance between two near-parallel planes, metres."""
    n = p1["normal"]
    # Project the second plane's centroid onto the first plane's normal.
    return abs(float(np.dot(p2["centroid"] - p1["centroid"], n)))


def solve_affine(pred_sep, pred_dist, gt_sep, gt_dist, space):
    """Solve z_true = a*z_pred + b, in linear or inverse depth space."""
    if space == "inverse":
        # Relief in inverse space is not a simple difference; solve on the two
        # absolute surface distances instead.
        z1p, z2p = pred_dist, pred_dist + pred_sep
        z1t, z2t = gt_dist, gt_dist + gt_sep
        A = np.array([[1.0 / z1p, 1.0], [1.0 / z2p, 1.0]])
        y = np.array([1.0 / z1t, 1.0 / z2t])
    else:
        z1p, z2p = pred_dist, pred_dist + pred_sep
        z1t, z2t = gt_dist, gt_dist + gt_sep
        A = np.array([[z1p, 1.0], [z2p, 1.0]])
        y = np.array([z1t, z2t])
    try:
        a, b = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        return None
    return {"a": float(a), "b": float(b), "space": space}


def main():
    ap = argparse.ArgumentParser(
        description="Measure conveyor-to-floor separation and solve the affine fix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cloud", action="append", required=True,
                    type=parse_cloud_arg, metavar="NAME=PATH")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--plane-thresh", type=float, default=0.010)
    ap.add_argument("--max-angle-deg", type=float, default=8.0,
                    help="max angle for two planes to count as parallel")
    ap.add_argument("--min-plane-frac", type=float, default=0.04,
                    help="min inlier fraction for a plane to be accepted")
    ap.add_argument("--gt-separation", type=float, default=None,
                    help="true floor-to-conveyor height, metres (e.g. 0.80)")
    ap.add_argument("--gt-distance", type=float, default=None,
                    help="true camera-to-conveyor distance, metres. TAPE MEASURE "
                         "THIS, do not assume it.")
    ap.add_argument("--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="reference camera centre in world coords")
    ap.add_argument("--out", type=Path, default=Path("plane_separation.json"))
    args = ap.parse_args()

    origin = np.array(args.origin, float)
    report = {"gt_separation_m": args.gt_separation,
              "gt_distance_m": args.gt_distance,
              "cameras": {}}

    rows = []
    for name, path in args.cloud:
        if not path.exists():
            sys.exit(f"missing cloud: {path}")
        pcd = o3d.io.read_point_cloud(str(path))
        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)
        pts = np.asarray(pcd.points, float)

        pair, all_planes = find_parallel_pair(
            pts, args.plane_thresh, args.max_angle_deg, args.min_plane_frac
        )
        entry = {"n_points": int(len(pts)),
                 "n_planes_found": len(all_planes)}

        if pair is None:
            entry["status"] = "no parallel plane pair found"
            report["cameras"][name] = entry
            rows.append((name, None, None, None))
            print(f"[{name}] no parallel pair among {len(all_planes)} planes")
            continue

        p1, p2 = pair
        sep = separation(p1, p2)
        # Which plane is nearer the camera: that is the conveyor.
        d1 = float(np.linalg.norm(p1["centroid"] - origin))
        d2 = float(np.linalg.norm(p2["centroid"] - origin))
        near_dist = min(d1, d2)
        angle = float(np.degrees(np.arccos(
            np.clip(abs(np.dot(p1["normal"], p2["normal"])), -1, 1))))

        entry.update({
            "status": "ok",
            "separation_mm": round(sep * 1e3, 1),
            "plane_angle_deg": round(angle, 3),
            "near_plane_dist_m": round(near_dist, 4),
            "far_plane_dist_m": round(max(d1, d2), 4),
            "plane_point_counts": [p1["n_points"], p2["n_points"]],
        })

        if args.gt_separation:
            entry["relief_ratio"] = round(args.gt_separation / sep, 4)
        if args.gt_distance:
            entry["absolute_ratio"] = round(args.gt_distance / near_dist, 4)
        if args.gt_separation and args.gt_distance:
            for space in ("linear", "inverse"):
                sol = solve_affine(sep, near_dist,
                                   args.gt_separation, args.gt_distance, space)
                if sol:
                    entry[f"affine_{space}"] = {
                        "a": round(sol["a"], 5), "b_mm": round(sol["b"] * 1e3, 1)
                    }

        report["cameras"][name] = entry
        rows.append((name, sep * 1e3,
                     entry.get("relief_ratio"), entry.get("absolute_ratio")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))

    hdr = f"{'camera':<10}{'sep_mm':>10}{'relief_x':>11}{'absolute_x':>13}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, sep, rr, ar in rows:
        s = f"{sep:.1f}" if sep is not None else "n/a"
        r = f"{rr:.3f}" if rr else "-"
        a = f"{ar:.3f}" if ar else "-"
        print(f"{name:<10}{s:>10}{r:>11}{a:>13}")

    if args.gt_separation and args.gt_distance:
        print("\nIf relief_x and absolute_x agree, a single scalar fixes it.")
        print("If relief_x exceeds absolute_x, the shift term is nonzero and")
        print("no scalar correction will work. Get a THIRD known distance to")
        print("decide between the linear and inverse affine solutions.")

    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()