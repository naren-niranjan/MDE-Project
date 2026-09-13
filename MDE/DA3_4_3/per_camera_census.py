#!/usr/bin/env python3
"""
Per-camera height census and cross-camera support test.

The two-plane bias test only inspects points near the floor and near the
deck. A population sitting between those bands, or above them, is invisible
to it. This script censuses every point by height above the fitted floor,
per camera, and separately reports how much of each camera's cloud has no
corroborating point from any other camera.

A region that is large, single-coloured in a fused render, and has near
zero cross-camera support is either unmatched coverage (a surface only one
camera can see) or flying pixels. A region with support that is displaced
is a genuine geometric disagreement.

Usage
-----
    python per_camera_census.py \
        --cloud left=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_left.ply \
        --cloud center=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_center.ply \
        --cloud right=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_right.ply \
        --cloud top=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_top.ply \
        --extrinsics /home/jetson/Projects/Calibration_4_1/results/extrinsics.json \
        --support-radius 0.015 \
        --out per_camera_census.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def fit_plane(points: np.ndarray, thresh: float, iters: int = 4000):
    if points.shape[0] < 100:
        return None, None
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    model, _ = pc.segment_plane(thresh, 3, iters)
    a, b, c, d = model
    n = np.array([a, b, c], dtype=float)
    k = np.linalg.norm(n)
    return n / k, d / k


def load_extrinsics(path: Path) -> dict:
    raw = json.loads(path.read_text())
    cams = raw.get("cameras", raw)
    out = {}
    for name, e in cams.items():
        if not isinstance(e, dict) or "R" not in e or "t" not in e:
            continue
        R = np.asarray(e["R"], dtype=float).reshape(3, 3)
        t = np.asarray(e["t"], dtype=float).reshape(3)
        out[name] = {"C": -R.T @ t}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-camera height census and cross-camera support.")
    ap.add_argument("--cloud", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--extrinsics", required=True, type=Path)
    ap.add_argument("--plane-thresh", type=float, default=0.010)
    ap.add_argument("--bin", type=float, default=0.050,
                    help="census bin width in metres")
    ap.add_argument("--max-height", type=float, default=1.600,
                    help="upper limit of the census in metres above the floor")
    ap.add_argument("--support-radius", type=float, default=0.015,
                    help="radius for a corroborating point from another camera")
    ap.add_argument("--support-subsample", type=int, default=200000,
                    help="cap on points per camera for the support test")
    ap.add_argument("--out", type=Path, default=Path("per_camera_census.json"))
    args = ap.parse_args()

    clouds = {}
    for spec in args.cloud:
        if "=" not in spec:
            print(f"malformed --cloud argument: {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"missing cloud: {p}", file=sys.stderr)
            return 2
        pts = np.asarray(o3d.io.read_point_cloud(str(p)).points, dtype=float)
        if pts.shape[0] == 0:
            print(f"empty cloud: {p}", file=sys.stderr)
            return 2
        clouds[name] = pts

    ext = load_extrinsics(args.extrinsics)
    names = list(clouds.keys())

    union = np.vstack([clouds[n] for n in names])
    n_f, d_f = fit_plane(union, args.plane_thresh)
    if n_f is None:
        print("floor fit failed", file=sys.stderr)
        return 1

    centroid = np.mean([ext[n]["C"] for n in names if n in ext], axis=0)
    if float(n_f @ centroid + d_f) < 0.0:
        n_f, d_f = -n_f, -d_f

    edges = np.arange(-0.100, args.max_height + args.bin, args.bin)

    heights = {n: clouds[n] @ n_f + d_f for n in names}
    hist = {n: np.histogram(heights[n], bins=edges)[0] for n in names}

    # Cross-camera support.
    rng = np.random.default_rng(0)
    trees = {}
    for n in names:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(clouds[n])
        trees[n] = o3d.geometry.KDTreeFlann(pc)

    support = {}
    for n in names:
        pts = clouds[n]
        if pts.shape[0] > args.support_subsample:
            idx = rng.choice(pts.shape[0], args.support_subsample, replace=False)
            sample = pts[idx]
            sample_h = heights[n][idx]
        else:
            sample = pts
            sample_h = heights[n]

        others = [m for m in names if m != n]
        supported = np.zeros(sample.shape[0], dtype=bool)
        for m in others:
            tree = trees[m]
            for i, p in enumerate(sample):
                if supported[i]:
                    continue
                k, _, _ = tree.search_radius_vector_3d(p, args.support_radius)
                if k > 0:
                    supported[i] = True

        per_bin_total = np.histogram(sample_h, bins=edges)[0]
        per_bin_unsup = np.histogram(sample_h[~supported], bins=edges)[0]
        support[n] = {
            "sampled": int(sample.shape[0]),
            "unsupported_fraction": float(np.mean(~supported)),
            "per_bin_total": per_bin_total,
            "per_bin_unsupported": per_bin_unsup,
        }

    # Report.
    width = max(len(n) for n in names)
    print("height census, points per bin, thousands\n")
    header = "height (mm)".ljust(14) + "".join(n.rjust(width + 3) for n in names)
    print(header)
    for i in range(len(edges) - 1):
        row_counts = [hist[n][i] for n in names]
        if max(row_counts) < 500:
            continue
        lo = edges[i] * 1000.0
        label = f"{lo:7.0f}".ljust(14)
        print(label + "".join(f"{c / 1000.0:>{width + 3}.1f}" for c in row_counts))

    print("\nunsupported fraction, no neighbour from another camera within "
          f"{args.support_radius * 1000:.0f} mm\n")
    for n in names:
        print(f"{n.ljust(width)}  {support[n]['unsupported_fraction'] * 100:6.2f} %"
              f"   of {support[n]['sampled']} sampled")

    print("\nunsupported points by height, thousands\n")
    print(header)
    for i in range(len(edges) - 1):
        row = [support[n]["per_bin_unsupported"][i] for n in names]
        if max(row) < 500:
            continue
        lo = edges[i] * 1000.0
        print(f"{lo:7.0f}".ljust(14)
              + "".join(f"{c / 1000.0:>{width + 3}.1f}" for c in row))

    payload = {
        "floor_normal": [round(v, 6) for v in n_f.tolist()],
        "bin_m": args.bin,
        "support_radius_m": args.support_radius,
        "bin_edges_mm": [round(float(e) * 1000.0, 1) for e in edges],
        "cameras": {
            n: {
                "n_points": int(clouds[n].shape[0]),
                "histogram": hist[n].tolist(),
                "unsupported_fraction": support[n]["unsupported_fraction"],
                "unsupported_histogram": support[n]["per_bin_unsupported"].tolist(),
            } for n in names
        },
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())