#!/usr/bin/env python3
"""
Decompose per-camera disagreement in a multi-view fused cloud into
depth scale, axial offset, and rotation.

Method
------
Two roughly parallel reference surfaces are present in the scene at
different ranges: the factory floor and the conveyor deck. For every
camera, both surfaces are fitted from that camera's points alone, and
the perpendicular distance from that camera's optical centre to each
fitted surface is measured. The same distances are measured against
consensus planes fitted from the remaining cameras. Solving

    d_measured = s * d_consensus + c

for the two surfaces yields a per-camera depth scale s and axial
offset c in closed form.

Interpretation
--------------
    s != 1, c ~ 0   depth scale error; the intrinsics the network was
                    conditioned on disagree with the image it received
    s ~ 1, c != 0   translation along the optical axis; extrinsics
    s ~ 1, c ~ 0    geometry agrees; a visible discrepancy in the fused
                    render is unmatched coverage or flying pixels
    normal tilt     extrinsic rotation, reported separately

Usage
-----
    python per_camera_bias.py \
        --cloud left=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_left.ply \
        --cloud center=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_center.ply \
        --cloud right=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_right.ply \
        --cloud top=/home/jetson/Projects/MDE/DA3_4_3/runs/20260807_105802/prior/cloud_top.ply \
        --extrinsics /home/jetson/Projects/Calibration_4_1/results/extrinsics.json \
        --out per_camera_bias.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------
# plane utilities
# --------------------------------------------------------------------------

def fit_plane(points: np.ndarray, thresh: float, iters: int = 4000):
    """RANSAC plane fit. Returns unit normal, offset d, inlier mask.

    The plane satisfies n . x + d = 0 with |n| = 1.
    """
    if points.shape[0] < 100:
        return None, None, None
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    model, inliers = pc.segment_plane(thresh, 3, iters)
    a, b, c, d = model
    n = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(n)
    return n / norm, d / norm, np.asarray(inliers, dtype=int)


def orient_towards(n: np.ndarray, d: float, target: np.ndarray):
    """Flip the plane so the signed distance to target is positive."""
    if float(n @ target + d) < 0.0:
        return -n, -d
    return n, d


def signed_distance(n: np.ndarray, d: float, points: np.ndarray) -> np.ndarray:
    return points @ n + d


def perpendicular_distance(n: np.ndarray, d: float, centre: np.ndarray) -> float:
    return abs(float(centre @ n + d))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two unit vectors, sign-insensitive."""
    c = abs(float(np.clip(a @ b, -1.0, 1.0)))
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------
# extrinsics
# --------------------------------------------------------------------------

def load_extrinsics(path: Path) -> dict:
    """Return {cam: {'R': 3x3, 't': 3, 'C': 3, 'axis': 3}} in world frame.

    Convention is world to camera: x_cam = R x_world + t. The optical
    centre is therefore C = -R^T t and the viewing axis is the third row
    of R expressed in world coordinates.
    """
    raw = json.loads(path.read_text())
    cams = raw.get("cameras", raw)
    out = {}
    for name, entry in cams.items():
        if not isinstance(entry, dict):
            continue
        if "R" not in entry or "t" not in entry:
            continue
        R = np.asarray(entry["R"], dtype=float).reshape(3, 3)
        t = np.asarray(entry["t"], dtype=float).reshape(3)
        out[name] = {
            "R": R,
            "t": t,
            "C": -R.T @ t,
            "axis": R.T @ np.array([0.0, 0.0, 1.0]),
        }
    return out


# --------------------------------------------------------------------------
# surface extraction
# --------------------------------------------------------------------------

def extract_surfaces(points: np.ndarray,
                     floor_ref: tuple,
                     deck_sep: float,
                     band: float,
                     thresh: float):
    """Fit floor and deck for one point set, using a global floor as a guide.

    The largest plane in this scene is the floor, not the deck, so the
    deck is taken as the band at deck_sep above the global floor and
    refitted locally.
    """
    n_ref, d_ref = floor_ref

    sd = signed_distance(n_ref, d_ref, points)

    floor_pts = points[np.abs(sd) < band]
    deck_pts = points[np.abs(sd - deck_sep) < band]

    floor = fit_plane(floor_pts, thresh)
    deck = fit_plane(deck_pts, thresh)

    return floor, deck, floor_pts.shape[0], deck_pts.shape[0]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-camera depth scale, axial offset, and rotation.")
    ap.add_argument("--cloud", action="append", required=True, metavar="NAME=PATH",
                    help="per-camera cloud, repeatable, e.g. left=cloud_left.ply")
    ap.add_argument("--extrinsics", required=True, type=Path,
                    help="extrinsics.json, world to camera relative to center")
    ap.add_argument("--deck-sep", type=float, default=0.8375,
                    help="floor to deck separation in metres, ChArUco ground truth")
    ap.add_argument("--band", type=float, default=0.040,
                    help="half width of the surface selection band in metres")
    ap.add_argument("--plane-thresh", type=float, default=0.010,
                    help="RANSAC inlier threshold in metres")
    ap.add_argument("--out", type=Path, default=Path("per_camera_bias.json"))
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
        pc = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pc.points, dtype=float)
        if pts.shape[0] == 0:
            print(f"empty cloud: {p}", file=sys.stderr)
            return 2
        clouds[name] = pts

    ext = load_extrinsics(args.extrinsics)
    missing = [c for c in clouds if c not in ext]
    if missing:
        print(f"no extrinsics for: {', '.join(missing)}", file=sys.stderr)
        return 2

    names = list(clouds.keys())

    # Global floor from the union, oriented so the camera side is positive.
    union = np.vstack([clouds[n] for n in names])
    n_g, d_g, _ = fit_plane(union, args.plane_thresh)
    if n_g is None:
        print("global floor fit failed", file=sys.stderr)
        return 1
    centroid = np.mean([ext[n]["C"] for n in names], axis=0)
    n_g, d_g = orient_towards(n_g, d_g, centroid)

    # Per-camera surface fits.
    fits = {}
    for name in names:
        floor, deck, n_floor_pts, n_deck_pts = extract_surfaces(
            clouds[name], (n_g, d_g), args.deck_sep, args.band, args.plane_thresh)

        entry = {"n_floor_points": int(n_floor_pts),
                 "n_deck_points": int(n_deck_pts)}

        if floor[0] is not None:
            nf, df = orient_towards(floor[0], floor[1], ext[name]["C"])
            entry["floor"] = (nf, df)
        if deck[0] is not None:
            nd, dd = orient_towards(deck[0], deck[1], ext[name]["C"])
            entry["deck"] = (nd, dd)

        fits[name] = entry

    # Consensus surfaces, leave one out.
    results = {}
    for name in names:
        others = [m for m in names if m != name]
        entry = fits[name]

        if "floor" not in entry or "deck" not in entry:
            results[name] = {
                "status": "insufficient surface coverage",
                "n_floor_points": entry["n_floor_points"],
                "n_deck_points": entry["n_deck_points"],
            }
            continue

        other_union = np.vstack([clouds[m] for m in others])
        cons_floor, cons_deck, _, _ = extract_surfaces(
            other_union, (n_g, d_g), args.deck_sep, args.band, args.plane_thresh)

        if cons_floor[0] is None or cons_deck[0] is None:
            results[name] = {"status": "consensus fit failed"}
            continue

        cnf, cdf = orient_towards(cons_floor[0], cons_floor[1], ext[name]["C"])
        cnd, cdd = orient_towards(cons_deck[0], cons_deck[1], ext[name]["C"])

        C = ext[name]["C"]

        d_meas = np.array([
            perpendicular_distance(entry["floor"][0], entry["floor"][1], C),
            perpendicular_distance(entry["deck"][0], entry["deck"][1], C),
        ])
        d_cons = np.array([
            perpendicular_distance(cnf, cdf, C),
            perpendicular_distance(cnd, cdd, C),
        ])

        # d_meas = s * d_cons + c, two equations, closed form.
        denom = d_cons[0] - d_cons[1]
        if abs(denom) < 1e-6:
            results[name] = {"status": "reference surfaces not separated"}
            continue
        s = float((d_meas[0] - d_meas[1]) / denom)
        c = float(d_meas[0] - s * d_cons[0])

        tilt_floor = angle_between(entry["floor"][0], cnf)
        tilt_deck = angle_between(entry["deck"][0], cnd)

        # Residual direction against the camera axis. A residual parallel
        # to the axis is a depth error; a perpendicular one is not.
        resid = float(np.mean(signed_distance(cnd, cdd, clouds[name][
            np.abs(signed_distance(n_g, d_g, clouds[name]) - args.deck_sep) < args.band])))
        axis_alignment = angle_between(cnd, ext[name]["axis"])

        if abs(s - 1.0) > 0.01 and abs(c) < 0.010:
            verdict = "depth scale error, check intrinsics passed at inference"
        elif abs(s - 1.0) <= 0.01 and abs(c) > 0.010:
            verdict = "axial offset, check extrinsics translation"
        elif abs(s - 1.0) > 0.01 and abs(c) > 0.010:
            verdict = "mixed scale and offset"
        elif max(tilt_floor, tilt_deck) > 0.5:
            verdict = "rotation only, check extrinsics orientation"
        else:
            verdict = "geometry consistent, discrepancy is coverage or flying pixels"

        results[name] = {
            "status": "ok",
            "scale": round(s, 5),
            "offset_mm": round(c * 1000.0, 2),
            "camera_to_floor_measured_m": round(float(d_meas[0]), 5),
            "camera_to_floor_consensus_m": round(float(d_cons[0]), 5),
            "camera_to_deck_measured_m": round(float(d_meas[1]), 5),
            "camera_to_deck_consensus_m": round(float(d_cons[1]), 5),
            "floor_tilt_deg": round(tilt_floor, 4),
            "deck_tilt_deg": round(tilt_deck, 4),
            "mean_deck_residual_mm": round(resid * 1000.0, 2),
            "deck_normal_to_optical_axis_deg": round(axis_alignment, 3),
            "n_floor_points": entry["n_floor_points"],
            "n_deck_points": entry["n_deck_points"],
            "verdict": verdict,
        }

    payload = {
        "deck_separation_gt_m": args.deck_sep,
        "band_m": args.band,
        "plane_threshold_m": args.plane_thresh,
        "global_floor_normal": [round(v, 6) for v in n_g.tolist()],
        "cameras": results,
    }

    args.out.write_text(json.dumps(payload, indent=2))

    width = max(len(n) for n in names)
    print(f"{'camera'.ljust(width)}  {'scale':>8}  {'offset':>9}  "
          f"{'floor tilt':>10}  {'deck tilt':>9}   verdict")
    for name in names:
        r = results[name]
        if r.get("status") != "ok":
            print(f"{name.ljust(width)}  {r.get('status')}")
            continue
        print(f"{name.ljust(width)}  {r['scale']:8.4f}  "
              f"{r['offset_mm']:7.1f}mm  {r['floor_tilt_deg']:9.3f}d  "
              f"{r['deck_tilt_deg']:8.3f}d   {r['verdict']}")

    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())