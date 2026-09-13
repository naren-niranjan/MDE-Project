#!/usr/bin/env python3
"""
fused_layer_profile.py

Detects and measures surface layering in a single fused point cloud, without
requiring the per-camera intermediates.

Method
------
1. RANSAC-fit the dominant support plane (conveyor deck or floor).
2. Take the signed distance of every point in a band around that plane.
3. Estimate the distribution of those signed distances with a Gaussian KDE and
   locate its modes. One mode means the cameras agree. N modes means N sheets,
   and the mode spacing is the layer separation.
4. Repeat per tile on a spatial grid so that the spatial behaviour of the
   separation can be read off.

Interpretation
--------------
    separation roughly constant across tiles   -> residual per-camera shift
    separation grows across the scene          -> residual per-camera scale
    single broad mode, high spread             -> bias is smeared rather than
                                                  discretely layered; look at
                                                  spread_mm per tile instead

Requires Open3D and SciPy. On the Jetson use the Python 3.11 conda environment.

Example
-------
python fused_layer_profile.py \
    --cloud /home/jetson/Projects/MDE/DA3_4_2/runs/20260805_100915_018_v2/noprior/fused.ply \
    --plane-band 0.06 \
    --grid 4 \
    --out layer_profile.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not found. Activate the Python 3.11 conda environment.")

try:
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
except ImportError:
    sys.exit("scipy not found. pip install scipy in the same environment.")


def fit_plane(points, dist_thresh, iters=4000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh, ransac_n=3, num_iterations=iters
    )
    a, b, c, d = model
    n = np.array([a, b, c], dtype=np.float64)
    scale = np.linalg.norm(n)
    n /= scale
    d /= scale
    mask = np.zeros(len(points), dtype=bool)
    mask[np.asarray(inliers)] = True
    return n, d, mask


def plane_basis(normal):
    """Two unit vectors spanning the plane, for tiling in-plane."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def find_modes(signed, bandwidth_mm, min_prominence, max_modes, rng, sample=60000):
    """Locate modes of the signed-distance distribution. Units in, mm out."""
    if len(signed) < 200:
        return None

    s = signed
    if len(s) > sample:
        s = s[rng.choice(len(s), size=sample, replace=False)]
    s_mm = s * 1e3

    lo, hi = np.percentile(s_mm, [0.5, 99.5])
    if hi - lo < 1e-6:
        return None
    grid = np.linspace(lo, hi, 512)

    bw = bandwidth_mm / max(np.std(s_mm), 1e-9)
    try:
        kde = gaussian_kde(s_mm, bw_method=bw)
    except np.linalg.LinAlgError:
        return None
    dens = kde(grid)

    peaks, props = find_peaks(dens, prominence=min_prominence * dens.max())
    if len(peaks) == 0:
        order = [int(np.argmax(dens))]
        proms = [float(dens.max())]
    else:
        order = list(peaks)
        proms = list(props["prominences"])

    ranked = sorted(zip(order, proms), key=lambda t: -t[1])[:max_modes]
    ranked = sorted(ranked, key=lambda t: grid[t[0]])

    modes = [round(float(grid[i]), 2) for i, _ in ranked]
    seps = [round(modes[i + 1] - modes[i], 2) for i in range(len(modes) - 1)]

    return {
        "n_points": int(len(signed)),
        "n_modes": len(modes),
        "mode_positions_mm": modes,
        "mode_separations_mm": seps,
        "spread_mm": round(float(np.std(s_mm)), 2),
        "p5_p95_span_mm": round(
            float(np.percentile(s_mm, 95) - np.percentile(s_mm, 5)), 2
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Measure surface layering in a fused point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cloud", type=Path, required=True)
    ap.add_argument(
        "--voxel",
        type=float,
        default=0.003,
        help="voxel down-sample size in metres, 0 to disable. Keep this well "
             "below the expected layer separation or the sheets merge.",
    )
    ap.add_argument("--plane-thresh", type=float, default=0.008,
                    help="RANSAC inlier threshold, metres")
    ap.add_argument("--plane-band", type=float, default=0.06,
                    help="half-thickness of the band analysed around the plane, metres")
    ap.add_argument("--bandwidth-mm", type=float, default=3.0,
                    help="KDE bandwidth in mm; lower resolves closer sheets but "
                         "starts to pick up noise")
    ap.add_argument("--min-prominence", type=float, default=0.06,
                    help="peak prominence as a fraction of peak density")
    ap.add_argument("--max-modes", type=int, default=5)
    ap.add_argument("--grid", type=int, default=4,
                    help="tile the plane into GRID x GRID cells for the spatial "
                         "profile; 1 disables tiling")
    ap.add_argument("--min-tile-points", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("layer_profile.json"))
    args = ap.parse_args()

    if not args.cloud.exists():
        sys.exit(f"missing cloud: {args.cloud}")

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    pcd = o3d.io.read_point_cloud(str(args.cloud))
    if len(pcd.points) == 0:
        sys.exit("cloud contains no points")
    raw_n = len(pcd.points)
    if args.voxel > 0.0:
        pcd = pcd.voxel_down_sample(args.voxel)
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"[load] {raw_n} -> {len(pts)} pts  {args.cloud}")

    normal, d, inl = fit_plane(pts, args.plane_thresh)
    signed_all = pts @ normal + d
    if np.median(signed_all) < 0.0:
        normal, d = -normal, -d
        signed_all = -signed_all
    print(
        f"[plane] normal=[{normal[0]:+.4f} {normal[1]:+.4f} {normal[2]:+.4f}] "
        f"d={d:+.4f} inliers={inl.mean():.1%}"
    )

    band = np.abs(signed_all) <= args.plane_band
    band_pts = pts[band]
    band_signed = signed_all[band]
    print(f"[band] {len(band_pts)} pts within +/-{args.plane_band * 1e3:.0f} mm")

    report = {
        "cloud": str(args.cloud),
        "n_points_raw": raw_n,
        "n_points_used": int(len(pts)),
        "plane": {
            "normal": normal.tolist(),
            "d": float(d),
            "inlier_frac": float(inl.mean()),
        },
        "params": {
            "voxel_m": args.voxel,
            "plane_thresh_m": args.plane_thresh,
            "plane_band_m": args.plane_band,
            "bandwidth_mm": args.bandwidth_mm,
            "min_prominence": args.min_prominence,
        },
    }

    glob = find_modes(
        band_signed, args.bandwidth_mm, args.min_prominence, args.max_modes, rng
    )
    report["global"] = glob

    tiles = []
    if args.grid > 1 and len(band_pts) > 0:
        u, v = plane_basis(normal)
        cu = band_pts @ u
        cv = band_pts @ v
        u_edges = np.linspace(cu.min(), cu.max(), args.grid + 1)
        v_edges = np.linspace(cv.min(), cv.max(), args.grid + 1)

        for i in range(args.grid):
            for j in range(args.grid):
                sel = (
                    (cu >= u_edges[i]) & (cu < u_edges[i + 1] if i < args.grid - 1 else cu <= u_edges[i + 1])
                    & (cv >= v_edges[j]) & (cv < v_edges[j + 1] if j < args.grid - 1 else cv <= v_edges[j + 1])
                )
                if sel.sum() < args.min_tile_points:
                    continue
                res = find_modes(
                    band_signed[sel], args.bandwidth_mm,
                    args.min_prominence, args.max_modes, rng
                )
                if res is None:
                    continue
                res["tile"] = [i, j]
                res["centre_uv_m"] = [
                    round(float(0.5 * (u_edges[i] + u_edges[i + 1])), 4),
                    round(float(0.5 * (v_edges[j] + v_edges[j + 1])), 4),
                ]
                tiles.append(res)
    report["tiles"] = tiles
    report["elapsed_s"] = round(time.perf_counter() - t0, 3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print()
    if glob:
        print(f"global: {glob['n_modes']} mode(s) at {glob['mode_positions_mm']} mm")
        if glob["mode_separations_mm"]:
            print(f"        separations {glob['mode_separations_mm']} mm")
        print(f"        spread {glob['spread_mm']} mm, "
              f"p5-p95 span {glob['p5_p95_span_mm']} mm")

    if tiles:
        print()
        hdr = f"{'tile':<10}{'modes':>7}{'max_sep_mm':>13}{'spread_mm':>12}{'pts':>9}"
        print(hdr)
        print("-" * len(hdr))
        for t in tiles:
            sep = max(t["mode_separations_mm"]) if t["mode_separations_mm"] else 0.0
            print(
                f"{str(t['tile']):<10}{t['n_modes']:>7}{sep:>13.2f}"
                f"{t['spread_mm']:>12.2f}{t['n_points']:>9d}"
            )

    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()