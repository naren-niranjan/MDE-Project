#!/usr/bin/env python3
"""
layer_align.py

Diagnose and remove inter-camera layering in a fused DA3 cloud, and measure
whether the correction generalises beyond the surfaces it was fitted on.

The problem
-----------
Concatenating four per-camera clouds produces parallel offset sheets of the
same physical surface. The extrinsics are good to 0.24-0.30 px, which puts the
geometric floor at 2-5 mm, so separation an order larger than that is not a
pose error. It is per-view depth scale: DA3's absolute scale is framing
dependent, and four cameras framing the scene differently receive four
different implicit scales.

The decomposition
-----------------
Layering is RELATIVE disagreement between views. Absolute scale error is a
separate problem. This tool solves them in that order.

  stage 1  fit the deck plane and the box-top planes in the reference camera's
           cloud, and measure how far every other camera's samples of those
           same planes sit from them. That signed offset, per camera, is the
           layering, quantified.

  stage 2  for each non-reference camera, compute where each of its rays
           SHOULD intersect the reference planes, and regress its predicted
           depth against that target. Two planes at different distances give
           the two equations needed to solve Z_true = a*Z_pred + b per view.

  stage 3  apply the correction, rebuild, and re-measure. The residual after
           correction is the spatially structured component of the bias, which
           an affine model cannot remove by construction. That residual is a
           result worth reporting, not a failure.

What stage 3 could not previously see
-------------------------------------
The residual was reported only on the planes the coefficients were fitted on,
where it is near zero by construction. Measurements on this rig then showed
the four cameras agreeing on the deck plane to 1.6 mm while disagreeing by
44.3 mm on a parcel top 270 mm above it. A correction validated only on the
deck therefore tells you nothing about the surfaces the robot actually picks
from. Two additions close that gap:

  --holdout       leave-one-plane-out validation. Solve on every plane but
                  one, measure on the one left out. This is the only figure
                  in the report that is not fitted on its own test set, and it
                  is the figure to quote.

  residual maps   the residual after correction, binned over the image, per
                  camera. A per-view affine can only remove a constant and a
                  slope in depth; anything that varies with WHERE in the frame
                  the surface appears survives it. The reported
                  spatial_structure_mm is the p05 to p95 range of the cell
                  medians and is the direct measurement of what a scalar
                  correction cannot reach. If it is large, the answer is not a
                  better affine, it is spatially varying anchoring.

Honesty constraint
------------------
The affine coefficients are fitted on planes present in this scene. Fitting
and evaluating on the same frame is circular. For the validation campaign,
solve the coefficients once on a calibration scene, freeze them, and apply the
frozen values to every subsequent frame. --load-affine exists for that.

The filter defaults here must match the ones the capture was produced under,
or the coefficients are solved on a different population of samples from the
one they are applied to. They now match da3_stream.py, including the 70 degree
incidence limit.

Input is a capture directory containing depth_<cam>.npy, conf_<cam>.npy,
K_<cam>.npy and E_<cam>.npy, as written by da3_stream.py or da3_fuse.py.

Examples
--------
Diagnose only:
    python layer_align.py --capture-dir runs/live_.../capture_00000 --mode diagnose

Solve, apply, validate and write the corrected cloud:
    python layer_align.py --capture-dir runs/live_.../capture_00000 \\
        --mode both --holdout --out-dir runs/aligned

Apply frozen coefficients to a later frame:
    python layer_align.py --capture-dir runs/live_.../capture_00007 \\
        --mode apply --load-affine runs/aligned/depth_affine.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d is required")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from da3_fuse import (points_camera, edge_mask, incidence_mask,
                          to_world, to_o3d, PALETTE,
                          GT_SEPARATION_M, GT_DECK_PERP_M)
except ImportError as exc:
    raise SystemExit(f"cannot import da3_fuse.py from the working directory: {exc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Diagnose and remove inter-camera layering, and measure "
                    "whether the correction generalises.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <capture-dir>/aligned")
    ap.add_argument("--cameras", nargs="+", default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--mode", choices=["diagnose", "solve", "apply", "both"],
                    default="both")
    ap.add_argument("--load-affine", type=Path, default=None,
                    help="apply frozen coefficients instead of solving")

    # plane extraction
    ap.add_argument("--plane-thresh", type=float, default=0.010,
                    help="RANSAC inlier distance for plane fitting, metres")
    ap.add_argument("--plane-iters", type=int, default=2000)
    ap.add_argument("--n-planes", type=int, default=3,
                    help="planes extracted from the reference cloud: the deck "
                         "plus box-top planes. At least 2 are needed to solve "
                         "both affine coefficients, and at least 3 before "
                         "--holdout can leave one out and still solve a full "
                         "affine on the rest")
    ap.add_argument("--min-plane-frac", type=float, default=0.02,
                    help="stop extracting once a plane holds less than this "
                         "fraction of the remaining points")
    ap.add_argument("--assign-tol", type=float, default=0.060,
                    help="a point is treated as belonging to a reference plane "
                         "if it lies within this distance of it, metres. Must "
                         "exceed the layer separation being corrected")
    ap.add_argument("--max-tilt-deg", type=float, default=8.0,
                    help="reject a reference plane whose normal is further "
                         "than this from the deck normal, so that box SIDE "
                         "faces are not used as depth references")
    ap.add_argument("--seed", type=int, default=0,
                    help="RANSAC seed; without it Open3D returns a different "
                         "plane partition on every run")
    ap.add_argument("--hist-bin", type=float, default=0.005,
                    help="bin width for the parallel-plane histogram, metres")
    ap.add_argument("--hist-smooth", type=float, default=0.020,
                    help="smoothing width, metres; must exceed the warp within "
                         "a single surface or one plane yields several peaks")
    ap.add_argument("--peak-min-frac", type=float, default=0.05,
                    help="ignore histogram peaks below this fraction of the "
                         "tallest")
    ap.add_argument("--plane-min-gap", type=float, default=0.15,
                    help="minimum separation between accepted planes, metres")
    ap.add_argument("--legacy-ransac", action="store_true",
                    help="use the old sequential RANSAC extraction instead of "
                         "the parallel-plane histogram")
    ap.add_argument("--merge-offset", type=float, default=0.05,
                    help="merge reference planes whose perpendicular offsets "
                         "differ by less than this, metres. Near-duplicate "
                         "planes make --assign-tol claim the same points twice "
                         "and corrupt the regression")
    ap.add_argument("--merge-tilt", type=float, default=4.0,
                    help="merge only if the normals also agree within this angle")

    # robust regression
    ap.add_argument("--huber-iters", type=int, default=5)
    ap.add_argument("--huber-k", type=float, default=1.345)
    ap.add_argument("--min-plane-points", type=int, default=500,
                    help="stop extracting planes once fewer than this many "
                         "points remain")
    ap.add_argument("--min-samples", type=int, default=2000,
                    help="minimum assigned points per camera before its affine "
                         "fit is trusted")
    ap.add_argument("--subsample", type=int, default=200000,
                    help="cap on points used in the regression per camera")

    # validation
    ap.add_argument("--holdout", action="store_true", default=True,
                    help="leave-one-plane-out validation: solve on every plane "
                         "but one and measure on the one left out. This is the "
                         "only residual in the report that is not fitted on "
                         "its own test set")
    ap.add_argument("--no-holdout", dest="holdout", action="store_false")
    ap.add_argument("--residual-cells", type=int, default=8,
                    help="grid over the image used to map the residual after "
                         "correction. A per-view affine removes a constant and "
                         "a slope in depth; whatever varies across this grid "
                         "is what it cannot reach")
    ap.add_argument("--residual-depth-bins", type=int, default=8,
                    help="depth bins used to test whether the residual is "
                         "flat with distance, which is the test of additive "
                         "against multiplicative")
    ap.add_argument("--no-residual-maps", dest="residual_maps",
                    action="store_false", default=True,
                    help="skip writing residual_map_<cam>.png")

    # filtering, matching da3_stream.py defaults exactly
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--conf-min", type=float, default=0.0,
                    help="absolute confidence floor on top of the percentile, "
                         "matching da3_stream.py")
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0,
                    help="drop samples whose surface normal exceeds this angle "
                         "from the view ray. MUST match the value the capture "
                         "was produced under, or the coefficients are solved "
                         "on a different population of samples from the one "
                         "they are applied to")
    ap.add_argument("--voxel", type=float, default=0.004)

    # cross-view consistency, applied after the affine correction
    ap.add_argument("--consistency", type=float, default=0.0,
                    help="after correction, keep only points with a neighbour "
                         "from a DIFFERENT camera within this radius in metres. "
                         "0 disables. This enforces a single layer directly, "
                         "at the cost of thinning single-view regions")
    ap.add_argument("--write-tsdf", action="store_true",
                    help="also integrate the corrected depth maps into a TSDF "
                         "volume and write the extracted mesh, which produces "
                         "one surface by construction. Note that this HIDES a "
                         "disagreement wider than --tsdf-trunc rather than "
                         "resolving it, so it is a presentation aid and not a "
                         "measurement")
    ap.add_argument("--tsdf-voxel", type=float, default=0.005)
    ap.add_argument("--tsdf-trunc", type=float, default=0.02)
    return ap.parse_args()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_capture(capture_dir: Path, names):
    data = {}
    for n in names:
        req = {k: capture_dir / f"{k}_{n}.npy" for k in ("depth", "K", "E")}
        missing = [str(p) for p in req.values() if not p.exists()]
        if missing:
            raise SystemExit(f"missing arrays for {n}: {', '.join(missing)}")
        entry = {k: np.load(p) for k, p in req.items()}
        conf_p = capture_dir / f"conf_{n}.npy"
        entry["conf"] = np.load(conf_p) if conf_p.exists() else None
        rgb_p = capture_dir / f"{n}.png"
        if rgb_p.exists() and cv2 is not None:
            img = cv2.imread(str(rgb_p))
            entry["rgb"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        else:
            entry["rgb"] = None
        data[n] = entry
    return data


def valid_mask(depth, conf, K, args, name=""):
    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    stages = {"finite": int(valid.sum())}

    if args.conf_percentile > 0 and conf is not None and valid.any():
        thr = float(np.percentile(conf[valid], args.conf_percentile))
        valid &= conf >= thr
        stages["after_conf"] = int(valid.sum())

    if args.conf_min > 0 and conf is not None:
        valid &= conf >= args.conf_min
        stages["after_conf_floor"] = int(valid.sum())

    if args.edge_thresh > 0:
        valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
        stages["after_edge"] = int(valid.sum())

    if args.max_incidence < 90:
        # The incidence test must be computed on metrically correct camera-frame
        # points. Using identity intrinsics here makes every normal appear
        # grazing and rejects the entire frame.
        ok, _ = incidence_mask(points_camera(d, K), args.max_incidence)
        valid &= ok
        stages["after_grazing"] = int(valid.sum())

    frac = valid.sum() / max(d.size, 1)
    if frac < 0.05:
        print(f"[WARN] {name}: only {valid.sum()} of {d.size} pixels survive "
              f"filtering ({frac * 100:.2f}%). Stage counts: {stages}. "
              f"Relax whichever stage collapses the count.")
    return valid, stages


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------

def refine_plane(points, normal, offset, thresh, iters=3):
    """Least-squares refinement of a plane over its inliers."""
    n = np.asarray(normal, np.float64)
    d = float(offset)
    for _ in range(iters):
        sd = points @ n - d
        inl = points[np.abs(sd) <= thresh]
        if len(inl) < 10:
            break
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n_new = vt[-1]
        if np.dot(n_new, n) < 0:
            n_new = -n_new
        n, d = n_new, float(n_new @ c)
    return n, d


def extract_parallel_planes(points, args):
    """Find the dominant plane, then its parallel companions by histogram.

    Sequential RANSAC is the wrong primitive when the surface being fitted is
    itself warped: it splits one physical plane into several patches at
    slightly different tilts, competes with genuine second surfaces for the
    plane budget, and returns a different partition on every run because the
    sampling is unseeded.

    Instead: fit the dominant plane once, project every point onto its normal,
    and read the parallel surfaces off the resulting 1D histogram. The deck and
    the floor share a normal in this workcell, so the peaks are exactly the
    references wanted, they cannot be split, and the result is deterministic.
    """
    if len(points) < args.min_plane_points:
        print(f"[WARN] only {len(points)} points supplied to plane extraction")
        return []

    o3d.utility.random.seed(args.seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, inliers = pcd.segment_plane(args.plane_thresh, 3, args.plane_iters)
    a, b, c, d = model
    n = np.array([a, b, c], np.float64)
    norm = np.linalg.norm(n)
    n, d0 = n / norm, float(-d / norm)
    if d0 < 0:
        n, d0 = -n, -d0
    n, d0 = refine_plane(points, n, d0, args.plane_thresh)

    sd = points @ n - d0
    lo, hi = np.percentile(sd, [0.5, 99.5])
    bins = max(32, int(round((hi - lo) / args.hist_bin)))
    counts, edges = np.histogram(sd, bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Smooth so that a single surface produces one peak rather than several.
    k = max(1, int(round(args.hist_smooth / args.hist_bin)))
    if k > 1:
        kern = np.ones(k) / k
        counts = np.convolve(counts.astype(np.float64), kern, mode="same")

    order = np.argsort(counts)[::-1]
    peaks = []
    for i in order:
        if counts[i] < args.peak_min_frac * counts.max():
            break
        if any(abs(centres[i] - p) < args.plane_min_gap for p in peaks):
            continue
        peaks.append(float(centres[i]))
        if len(peaks) >= args.n_planes:
            break

    planes = []
    for s in sorted(peaks, key=lambda v: -np.sum(np.abs(sd - v) <= args.plane_thresh)):
        offset = d0 + s
        cnt = int(np.sum(np.abs(sd - s) <= args.plane_thresh))
        planes.append({"normal": n.tolist(), "offset": float(offset),
                       "n_inliers": cnt, "tilt_from_deck_deg": 0.0,
                       "histogram_peak_m": round(s, 5)})
    return planes


def extract_planes(points, args):
    """Sequential RANSAC. Returns a list of (normal, offset, n_inliers)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    total = len(points)
    planes = []
    remaining = pcd
    if total < args.min_plane_points:
        print(f"[WARN] only {total} points supplied to plane extraction; "
              f"nothing will be fitted")
        return planes
    while len(planes) < args.n_planes and len(remaining.points) > args.min_plane_points:
        model, inliers = remaining.segment_plane(args.plane_thresh, 3, args.plane_iters)
        if len(inliers) < args.min_plane_frac * total:
            break
        a, b, c, d = model
        n = np.array([a, b, c], np.float64)
        norm = np.linalg.norm(n)
        # store as n . X = offset with unit normal
        planes.append({"normal": (n / norm).tolist(),
                       "offset": float(-d / norm),
                       "n_inliers": int(len(inliers))})
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


def filter_planes_by_tilt(planes, max_tilt_deg):
    """Keep planes roughly parallel to the largest one, which is the deck.

    Box side faces are near perpendicular to the deck and are seen by only
    some of the cameras, so including them would bias the regression toward
    whichever views happen to see them.
    """
    if not planes:
        return []
    ref_n = np.asarray(planes[0]["normal"])
    kept = []
    for p in planes:
        n = np.asarray(p["normal"])
        cos = abs(float(np.dot(n, ref_n)))
        tilt = np.degrees(np.arccos(np.clip(cos, 0, 1)))
        p["tilt_from_deck_deg"] = round(float(tilt), 3)
        if tilt <= max_tilt_deg:
            kept.append(p)
    return kept


def merge_planes(planes, merge_offset, merge_tilt):
    """Collapse near-duplicate planes, keeping the one with more inliers.

    Sequential RANSAC frequently splits one physical surface into several
    patches. Left in place they overlap under --assign-tol, so the same points
    are claimed by two references and the regression sees contradictory
    targets at nominally different depths.
    """
    kept = []
    for p in sorted(planes, key=lambda q: -q["n_inliers"]):
        n = np.asarray(p["normal"])
        duplicate = False
        for q in kept:
            nq = np.asarray(q["normal"])
            sign = np.sign(np.dot(n, nq)) or 1.0
            tilt = np.degrees(np.arccos(np.clip(abs(float(np.dot(n, nq))), 0, 1)))
            gap = abs(float(p["offset"]) * sign - float(q["offset"]))
            if tilt <= merge_tilt and gap <= merge_offset:
                q["n_inliers"] += p["n_inliers"]
                q.setdefault("merged", 0)
                q["merged"] += 1
                duplicate = True
                break
        if not duplicate:
            kept.append(dict(p))
    return sorted(kept, key=lambda q: -q["n_inliers"])


def per_camera_plane_fit(names, clouds_w, planes_w, args):
    """Fit each reference plane independently in every camera's own cloud.

    This separates the two failure modes that a signed-distance median cannot
    distinguish:

      same normal, different offset  ->  depth scale or shift error, which a
                                         per-view affine can correct
      different normal               ->  extrinsic rotation error, which it
                                         cannot; the disagreement then varies
                                         with lateral position rather than
                                         with depth
    """
    out = []
    for pi, plane in enumerate(planes_w):
        ref_n = np.asarray(plane["normal"])
        entry = {"plane": pi, "cameras": {}}
        for n in names:
            pts = clouds_w[n]
            if len(pts) == 0:
                continue
            sd = signed_distance(pts, plane["normal"], plane["offset"])
            near = np.abs(sd) <= args.assign_tol
            if near.sum() < args.min_plane_points:
                continue
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(pts[near])
            model, inl = sub.segment_plane(args.plane_thresh, 3, args.plane_iters)
            a, b, c, d = model
            nv = np.array([a, b, c], np.float64)
            nv /= np.linalg.norm(nv)
            sign = np.sign(np.dot(nv, ref_n)) or 1.0
            nv, off = nv * sign, float(-d / np.linalg.norm([a, b, c])) * sign
            tilt = np.degrees(np.arccos(np.clip(abs(float(np.dot(nv, ref_n))), 0, 1)))
            entry["cameras"][n] = {
                "n_points": int(near.sum()),
                "n_inliers": int(len(inl)),
                "tilt_vs_reference_deg": round(float(tilt), 4),
                "offset_vs_reference_mm": round((off - float(plane["offset"])) * 1e3, 3),
            }
        out.append(entry)
    return out


def report_per_camera_planes(per_cam_planes, extent_m=1.5):
    print("\n[fit  ] independent plane fit per camera")
    print("        tilt vs reference, and the lateral depth swing that tilt "
          f"implies over a {extent_m:.1f} m scene")
    for e in per_cam_planes:
        for cam, v in e["cameras"].items():
            swing = np.tan(np.radians(v["tilt_vs_reference_deg"])) * extent_m * 1e3
            print(f"        plane {e['plane']}  {cam:<8} "
                  f"tilt {v['tilt_vs_reference_deg']:6.3f} deg  "
                  f"offset {v['offset_vs_reference_mm']:+8.2f} mm  "
                  f"implied swing {swing:6.1f} mm")


def plane_to_camera(normal_w, offset_w, E):
    """World plane n.X = d expressed in the camera frame of a w2c extrinsic."""
    R, t = E[:3, :3], E[:3, 3]
    n_c = R @ np.asarray(normal_w, np.float64)
    d_c = float(offset_w) + float(n_c @ t)
    return n_c, d_c


def signed_distance(points_w, normal_w, offset_w):
    return points_w @ np.asarray(normal_w, np.float64) - float(offset_w)


def target_depth(depth_shape, K, E, plane):
    """Depth at which each ray meets the plane, and where that is meaningful."""
    rays = points_camera(np.ones(depth_shape, dtype=np.float64), K)
    n_c, d_c = plane_to_camera(plane["normal"], plane["offset"], E)
    denom = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d_c / denom
    ok = np.isfinite(z) & (z > 0) & (np.abs(denom) > 1e-6)
    return z, ok


# --------------------------------------------------------------------------
# affine solve
# --------------------------------------------------------------------------

def huber_fit(z_pred, z_true, iters, k):
    """Robust least squares for z_true = a * z_pred + b."""
    A = np.stack([z_pred, np.ones_like(z_pred)], axis=1)
    w = np.ones_like(z_pred)
    a, b = 1.0, 0.0
    for _ in range(max(1, iters)):
        Aw = A * w[:, None]
        sol, *_ = np.linalg.lstsq(Aw, z_true * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = z_true - (a * z_pred + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if s <= 1e-9:
            break
        u = np.abs(r) / s
        w = np.where(u <= k, 1.0, k / np.maximum(u, 1e-9))
    return a, b


def solve_affine(name, depth, K, E, valid, planes_w, args):
    """Regress predicted depth against the depth implied by the reference planes."""
    d = depth.astype(np.float64)

    z_pred_all, z_true_all, per_plane = [], [], []
    for pi, plane in enumerate(planes_w):
        z_target, usable_geom = target_depth(d.shape, K, E, plane)
        usable = valid & usable_geom
        # a point belongs to this plane if its CURRENT depth is close to the
        # depth the plane predicts along the same ray
        near = usable & (np.abs(d - z_target) <= args.assign_tol)
        cnt = int(near.sum())
        per_plane.append({"plane": pi, "n_assigned": cnt,
                          "median_residual_m": float(np.median(d[near] - z_target[near]))
                          if cnt else None})
        if cnt:
            z_pred_all.append(d[near])
            z_true_all.append(z_target[near])

    if not z_pred_all:
        return {"camera": name, "solved": False, "reason": "no points assigned",
                "a": 1.0, "b": 0.0, "per_plane": per_plane}

    z_pred = np.concatenate(z_pred_all)
    z_true = np.concatenate(z_true_all)

    if z_pred.size > args.subsample:
        idx = np.random.default_rng(0).choice(z_pred.size, args.subsample, replace=False)
        z_pred, z_true = z_pred[idx], z_true[idx]

    spread = float(z_true.max() - z_true.min())
    n_planes_used = sum(1 for p in per_plane if p["n_assigned"] > 0)

    if z_pred.size < args.min_samples:
        return {"camera": name, "solved": False, "reason": "too few samples",
                "a": 1.0, "b": 0.0, "n_samples": int(z_pred.size),
                "per_plane": per_plane}

    if n_planes_used < 2 or spread < 0.10:
        # Only one effective distance is present, so scale and offset are not
        # separable. Solve the offset alone and say so.
        b = float(np.median(z_true - z_pred))
        return {"camera": name, "solved": True, "model": "offset_only",
                "reason": "single depth present; a fixed at 1",
                "a": 1.0, "b": b, "n_samples": int(z_pred.size),
                "depth_spread_m": round(spread, 4),
                "per_plane": per_plane}

    a, b = huber_fit(z_pred, z_true, args.huber_iters, args.huber_k)
    r = z_true - (a * z_pred + b)
    r0 = z_true - z_pred
    return {"camera": name, "solved": True, "model": "affine",
            "a": a, "b": b,
            "n_samples": int(z_pred.size),
            "n_planes_used": n_planes_used,
            "depth_spread_m": round(spread, 4),
            "residual_before": {"median_mm": round(float(np.median(r0)) * 1e3, 3),
                                "mad_mm": round(float(np.median(np.abs(r0 - np.median(r0)))) * 1e3, 3),
                                "rms_mm": round(float(np.sqrt(np.mean(r0 ** 2))) * 1e3, 3)},
            "residual_after": {"median_mm": round(float(np.median(r)) * 1e3, 3),
                               "mad_mm": round(float(np.median(np.abs(r - np.median(r)))) * 1e3, 3),
                               "rms_mm": round(float(np.sqrt(np.mean(r ** 2))) * 1e3, 3)},
            "fitted_and_evaluated_on_same_planes": True,
            "per_plane": per_plane}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def plane_residual(depth, K, E, valid, plane, a, b, args):
    """Residual against one plane after applying Z -> a Z + b."""
    d = depth.astype(np.float64)
    dc = a * d + b
    z_target, usable_geom = target_depth(d.shape, K, E, plane)
    near = valid & usable_geom & (np.abs(dc - z_target) <= args.assign_tol)
    cnt = int(near.sum())
    if cnt < 100:
        return None
    r0 = (d - z_target)[near]
    r1 = (dc - z_target)[near]
    return {
        "n_points": cnt,
        "before": {"median_mm": round(float(np.median(r0)) * 1e3, 2),
                   "rms_mm": round(float(np.sqrt(np.mean(r0 ** 2))) * 1e3, 2)},
        "after": {"median_mm": round(float(np.median(r1)) * 1e3, 2),
                  "rms_mm": round(float(np.sqrt(np.mean(r1 ** 2))) * 1e3, 2)},
    }


def holdout_validation(names, data, masks, planes_w, args):
    """Leave one plane out, solve on the rest, measure on the one left out.

    Every other residual in this report is measured on the planes the
    coefficients were fitted to, where it is small by construction. This is the
    only figure that answers the question actually being asked: does the
    correction transfer to a surface it has not seen. If the held-out residual
    is comparable to the uncorrected one, the affine has learned the scene
    rather than the camera.
    """
    if len(planes_w) < 2:
        return {"ok": False,
                "reason": "leave-one-out needs at least two planes; with one "
                          "plane the solve has nothing left to fit"}
    if len(planes_w) == 2:
        note = ("only two planes, so each fold solves an offset alone and the "
                "scale term is untested")
    else:
        note = None

    folds = []
    for hp in range(len(planes_w)):
        train = [p for i, p in enumerate(planes_w) if i != hp]
        held = planes_w[hp]
        entry = {"held_out_plane": hp,
                 "held_out_offset_m": round(float(held["offset"]), 4),
                 "cameras": {}}
        for n in names:
            if n == args.reference:
                continue
            e = solve_affine(n, data[n]["depth"], data[n]["K"], data[n]["E"],
                             masks[n], train, args)
            res = plane_residual(data[n]["depth"], data[n]["K"], data[n]["E"],
                                 masks[n], held, e["a"], e["b"], args)
            if res is None:
                continue
            entry["cameras"][n] = {
                "a": round(e["a"], 6), "b_mm": round(e["b"] * 1e3, 3),
                "model": e.get("model"),
                "held_out_median_before_mm": res["before"]["median_mm"],
                "held_out_median_after_mm": res["after"]["median_mm"],
                "held_out_rms_before_mm": res["before"]["rms_mm"],
                "held_out_rms_after_mm": res["after"]["rms_mm"],
                "improvement_mm": round(abs(res["before"]["median_mm"])
                                        - abs(res["after"]["median_mm"]), 2),
                "n_points": res["n_points"],
            }
        folds.append(entry)
    return {"ok": True, "note": note, "folds": folds}


def residual_analysis(depth, K, E, valid, planes_w, a, b, args):
    """Where the residual lives after correction, in the image and in depth.

    A per-view affine can remove a constant and a term proportional to depth.
    It cannot remove anything that depends on where in the frame the surface
    appears. Binning the residual over the image separates the two: a flat map
    means the model fits, and structure in the map is the part of the bias that
    is out of reach of any per-camera scalar.
    """
    d = depth.astype(np.float64)
    dc = a * d + b
    h, w = d.shape

    resid = np.full(d.shape, np.nan)
    best = np.full(d.shape, np.inf)
    for plane in planes_w:
        z_target, usable_geom = target_depth(d.shape, K, E, plane)
        r = dc - z_target
        near = valid & usable_geom & (np.abs(r) <= args.assign_tol)
        take = near & (np.abs(r) < best)
        best[take] = np.abs(r)[take]
        resid[take] = r[take]

    finite = np.isfinite(resid)
    if int(finite.sum()) < 500:
        return None

    ny = nx = max(2, int(args.residual_cells))
    grid = np.full((ny, nx), np.nan)
    counts = np.zeros((ny, nx), dtype=np.int64)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    for iy in range(ny):
        for ix in range(nx):
            block = resid[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
            ok = np.isfinite(block)
            counts[iy, ix] = int(ok.sum())
            if ok.sum() >= 50:
                grid[iy, ix] = float(np.median(block[ok])) * 1e3

    cells = grid[np.isfinite(grid)]
    spatial = (round(float(np.percentile(cells, 95) - np.percentile(cells, 5)), 2)
               if cells.size >= 4 else None)

    # Residual against distance: flat means the bias is additive, a trend means
    # it is multiplicative and the scale term has not absorbed it.
    zz = dc[finite]
    rr = resid[finite] * 1e3
    nb = max(2, int(args.residual_depth_bins))
    edges = np.percentile(zz, np.linspace(0, 100, nb + 1))
    depth_bins = []
    for i in range(nb):
        sel = (zz >= edges[i]) & (zz <= edges[i + 1] if i == nb - 1 else zz < edges[i + 1])
        if int(sel.sum()) < 50:
            continue
        depth_bins.append({"z_centre_m": round(float(0.5 * (edges[i] + edges[i + 1])), 3),
                           "n": int(sel.sum()),
                           "median_mm": round(float(np.median(rr[sel])), 2)})
    trend = (round(depth_bins[-1]["median_mm"] - depth_bins[0]["median_mm"], 2)
             if len(depth_bins) >= 2 else None)

    return {
        "n_points": int(finite.sum()),
        "overall_median_mm": round(float(np.median(rr)), 2),
        "overall_rms_mm": round(float(np.sqrt(np.mean(rr ** 2))), 2),
        "cell_grid_mm": np.where(np.isfinite(grid), np.round(grid, 2), None).tolist(),
        "cell_counts": counts.tolist(),
        "spatial_structure_mm": spatial,
        "depth_bins": depth_bins,
        "depth_trend_mm": trend,
        "_grid": grid,
    }


def draw_residual_map(grid, name, spatial, cell_px=64):
    """Render the residual grid as an annotated image, in millimetres."""
    if cv2 is None:
        return None
    ny, nx = grid.shape
    lim = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    lim = max(lim, 1.0)
    canvas = np.full((ny * cell_px + 40, nx * cell_px, 3), 20, np.uint8)
    for iy in range(ny):
        for ix in range(nx):
            v = grid[iy, ix]
            y0, x0 = 40 + iy * cell_px, ix * cell_px
            if not np.isfinite(v):
                colour = (35, 35, 35)
                text = "-"
            else:
                t = float(np.clip((v / lim + 1.0) / 2.0, 0.0, 1.0))
                colour = tuple(int(c) for c in cv2.applyColorMap(
                    np.array([[int(t * 255)]], np.uint8),
                    cv2.COLORMAP_COOL if hasattr(cv2, "COLORMAP_COOL")
                    else cv2.COLORMAP_JET)[0, 0])
                text = f"{v:+.1f}"
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1),
                          colour, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1),
                          (15, 15, 15), 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            org = (x0 + (cell_px - tw) // 2, y0 + (cell_px + th) // 2)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
    head = (f"{name}: residual after correction, mm per image cell   "
            f"p05-p95 spread {spatial:.1f} mm" if spatial is not None
            else f"{name}: residual after correction, mm per image cell")
    cv2.putText(canvas, head, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure_layering(names, clouds_w, planes_w, args):
    """Signed distance of every camera's samples to each reference plane."""
    out = []
    for pi, plane in enumerate(planes_w):
        entry = {"plane": pi,
                 "normal": [round(v, 5) for v in plane["normal"]],
                 "offset_m": round(plane["offset"], 5),
                 "cameras": {}}
        for n in names:
            pts = clouds_w[n]
            if len(pts) == 0:
                continue
            sd = signed_distance(pts, plane["normal"], plane["offset"])
            near = np.abs(sd) <= args.assign_tol
            if near.sum() < 100:
                continue
            v = sd[near]
            entry["cameras"][n] = {
                "n_points": int(near.sum()),
                "median_mm": round(float(np.median(v)) * 1e3, 3),
                "mad_mm": round(float(np.median(np.abs(v - np.median(v)))) * 1e3, 3),
            }
        meds = [c["median_mm"] for c in entry["cameras"].values()]
        entry["inter_camera_spread_mm"] = round(max(meds) - min(meds), 3) if len(meds) > 1 else None
        out.append(entry)
    return out


def consistency_filter(clouds_w, names, radius):
    """Keep points corroborated by at least one other camera.

    The previous implementation ran one KD-tree radius query per point from
    Python, which on a two-million-point cloud takes tens of minutes. This
    hashes every point into a voxel of the given radius and asks which cameras
    occupy that voxel or any of the twenty-six around it, which is the same
    question answered in one vectorised pass over the OCCUPIED VOXELS rather
    than over the points.
    """
    order = [n for n in names if len(clouds_w[n])]
    if len(order) < 2 or radius <= 0:
        return dict(clouds_w)
    pts = np.vstack([clouds_w[n] for n in order])
    src = np.concatenate([np.full(len(clouds_w[n]), i, np.int32)
                          for i, n in enumerate(order)])

    q = np.floor(pts / radius).astype(np.int64)
    q -= q.min(axis=0)
    q += 1
    dims = q.max(axis=0) + 2
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = inv.ravel()
    q_u = q[first]

    own = np.zeros(len(uniq), np.int32)
    np.bitwise_or.at(own, inv, (1 << src))
    merged = own.copy()
    for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
        if dx == dy == dz == 0:
            continue
        nk = (((q_u[:, 0] + dx) * dims[1] + (q_u[:, 1] + dy)) * dims[2]
              + (q_u[:, 2] + dz))
        pos = np.clip(np.searchsorted(uniq, nk), 0, len(uniq) - 1)
        hit = uniq[pos] == nk
        merged[hit] |= own[pos[hit]]

    # A point is corroborated if some camera OTHER than its own is present.
    supported = (merged[inv] & ~(1 << src)) != 0
    kept = {}
    at = 0
    for i, n in enumerate(order):
        m = len(clouds_w[n])
        kept[n] = clouds_w[n][supported[at:at + m]]
        at += m
    for n in names:
        kept.setdefault(n, clouds_w[n])
    return kept


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    names = list(args.cameras)
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} not among {names}")
    out_dir = args.out_dir or (args.capture_dir / "aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(f"[filt ] conf p{args.conf_percentile:.0f}"
          + (f" floor {args.conf_min}" if args.conf_min > 0 else "")
          + f", edge {args.edge_thresh}, incidence {args.max_incidence:.0f} deg")
    print("        these must match the values the capture was produced under, "
          "or the\n        coefficients are solved on a different population "
          "of samples from the\n        one they will be applied to")

    data = load_capture(args.capture_dir, names)
    masks, filter_stages = {}, {}
    for n in names:
        masks[n], filter_stages[n] = valid_mask(
            data[n]["depth"], data[n]["conf"], data[n]["K"], args, name=n)

    def world_points(n, depth=None):
        d = data[n]["depth"].astype(np.float64) if depth is None else depth
        pts = points_camera(d, data[n]["K"])
        return to_world(pts[masks[n]], data[n]["E"])

    clouds_w = {n: world_points(n) for n in names}
    for n in names:
        total = data[n]["depth"].size
        print(f"[load ] {n:<8} {len(clouds_w[n]):>8d} of {total} pixels "
              f"({100 * len(clouds_w[n]) / max(total, 1):5.1f}%)  "
              f"stages={filter_stages[n]}")

    # ---- reference planes ---------------------------------------------
    if args.legacy_ransac:
        planes_all = extract_planes(clouds_w[args.reference], args)
        planes_tilt = filter_planes_by_tilt(planes_all, args.max_tilt_deg)
        planes_w = merge_planes(planes_tilt, args.merge_offset, args.merge_tilt)
        print(f"\n[plane] sequential RANSAC: {len(planes_all)} extracted, "
              f"{len(planes_tilt)} after tilt filter, {len(planes_w)} after merge")
    else:
        planes_all = extract_parallel_planes(clouds_w[args.reference], args)
        planes_tilt = planes_all
        planes_w = planes_all
        print(f"\n[plane] parallel-plane histogram from {args.reference}: "
              f"{len(planes_w)} surfaces")
    for i, p in enumerate(planes_w):
        peak = (f"  peak={p['histogram_peak_m']:+.4f} m"
                if "histogram_peak_m" in p else "")
        merged = f"  (merged {p['merged']})" if p.get("merged") else ""
        print(f"        plane {i}: n={np.round(p['normal'], 4)} "
              f"offset={p['offset']:+.4f} m  inliers={p['n_inliers']}{peak}{merged}")
    if len(planes_w) < 2:
        print("[WARN] fewer than two usable planes. Scale and offset cannot be "
              "separated, so only an offset will be solved. Capture a scene "
              "with boxes of differing height, or include the floor as well as "
              "the deck.")

    before = measure_layering(names, clouds_w, planes_w, args)
    print("\n[layer] signed distance to reference planes, before correction")
    for e in before:
        cams = "  ".join(f"{k} {v['median_mm']:+7.1f}" for k, v in e["cameras"].items())
        print(f"        plane {e['plane']}: {cams}   "
              f"spread {e['inter_camera_spread_mm']} mm")

    per_cam_planes = per_camera_plane_fit(names, clouds_w, planes_w, args)
    report_per_camera_planes(per_cam_planes)

    report = {"capture_dir": str(args.capture_dir),
              "reference": args.reference,
              "cameras": names,
              "filters": {"conf_percentile": args.conf_percentile,
                          "conf_min": args.conf_min,
                          "edge_thresh": args.edge_thresh,
                          "max_incidence_deg": args.max_incidence},
              "planes": planes_w,
              "planes_before_merge": planes_tilt,
              "layering_before": before,
              "per_camera_plane_fit": per_cam_planes}

    # The reference view's own flatness bounds what any inter-camera alignment
    # can achieve. If the reference cannot reconstruct a flat surface, the
    # residual after alignment cannot be smaller than its own warp.
    if planes_w:
        p0 = planes_w[0]
        sd_ref = signed_distance(clouds_w[args.reference], p0["normal"], p0["offset"])
        near = np.abs(sd_ref) <= args.assign_tol
        if near.sum() > args.min_plane_points:
            v = sd_ref[near]
            flat = {"n_points": int(near.sum()),
                    "rms_mm": round(float(np.sqrt(np.mean(v ** 2))) * 1e3, 3),
                    "mad_mm": round(float(np.median(np.abs(v - np.median(v)))) * 1e3, 3),
                    "p05_mm": round(float(np.percentile(v, 5)) * 1e3, 3),
                    "p95_mm": round(float(np.percentile(v, 95)) * 1e3, 3)}
            report["reference_flatness"] = flat
            print(f"\n[flat ] {args.reference} against its own dominant plane: "
                  f"rms {flat['rms_mm']:.2f} mm, "
                  f"p05-p95 {flat['p05_mm']:+.1f} to {flat['p95_mm']:+.1f} mm")
            print("        This is the reference view's own warp. No "
                  "inter-camera alignment can produce a residual below it.")

    # ---- solve ---------------------------------------------------------
    affine = None
    if args.mode in ("solve", "both"):
        affine = {}
        print("\n[solve] per-camera depth affine, Z_true = a * Z_pred + b")
        for n in names:
            if n == args.reference:
                affine[n] = {"camera": n, "solved": True, "model": "reference",
                             "a": 1.0, "b": 0.0}
            else:
                affine[n] = solve_affine(n, data[n]["depth"], data[n]["K"],
                                         data[n]["E"], masks[n], planes_w, args)
            e = affine[n]
            extra = ""
            if "residual_after" in e:
                extra = (f"  rms {e['residual_before']['rms_mm']:.1f} -> "
                         f"{e['residual_after']['rms_mm']:.1f} mm")
            print(f"        {n:<8} a={e['a']:.6f}  b={e['b']:+.6f} m  "
                  f"[{e.get('model', 'unsolved')}]{extra}")
        print("        These residuals are measured on the planes the "
              "coefficients were fitted\n        to, so they are small by "
              "construction. Read the holdout block below\n        instead.")
        (out_dir / "depth_affine.json").write_text(json.dumps(
            {"reference": args.reference,
             "source_capture": str(args.capture_dir),
             "filters": report["filters"],
             "coefficients": {n: {"a": affine[n]["a"], "b": affine[n]["b"]} for n in names},
             "detail": affine}, indent=2))

    elif args.mode == "apply":
        if args.load_affine is None:
            raise SystemExit("--mode apply requires --load-affine")
        loaded = json.loads(args.load_affine.read_text())
        affine = {n: {"camera": n, "a": loaded["coefficients"][n]["a"],
                      "b": loaded["coefficients"][n]["b"], "model": "loaded"}
                  for n in names}
        print(f"\n[apply] loaded coefficients from {args.load_affine}")
        src_f = loaded.get("filters")
        if src_f and src_f.get("max_incidence_deg") != args.max_incidence:
            print(f"[WARN] those coefficients were solved with an incidence "
                  f"limit of {src_f.get('max_incidence_deg')} deg and are "
                  f"being applied under {args.max_incidence:.0f} deg. The two "
                  f"populations of samples are not the same.")

    if args.load_affine is not None and args.mode == "both":
        loaded = json.loads(args.load_affine.read_text())
        for n in names:
            affine[n]["a"] = loaded["coefficients"][n]["a"]
            affine[n]["b"] = loaded["coefficients"][n]["b"]
            affine[n]["model"] = "loaded"
        print(f"[apply] overriding solved values with {args.load_affine}")

    # ---- generalisation -------------------------------------------------
    if args.holdout and args.mode in ("solve", "both"):
        hv = holdout_validation(names, data, masks, planes_w, args)
        report["holdout"] = hv
        if hv.get("ok"):
            print("\n[hold ] leave-one-plane-out: solved on the other planes, "
                  "measured on the one held out")
            if hv.get("note"):
                print(f"        note: {hv['note']}")
            for fold in hv["folds"]:
                for cam, v in fold["cameras"].items():
                    print(f"        plane {fold['held_out_plane']} at "
                          f"{fold['held_out_offset_m']:+.3f} m  {cam:<8} "
                          f"median {v['held_out_median_before_mm']:+7.2f} -> "
                          f"{v['held_out_median_after_mm']:+7.2f} mm   "
                          f"rms {v['held_out_rms_before_mm']:6.2f} -> "
                          f"{v['held_out_rms_after_mm']:6.2f} mm")
            print("        A held-out residual no better than the uncorrected "
                  "one means the affine\n        has learned this scene rather "
                  "than this camera.")
        else:
            print(f"\n[hold ] not run: {hv.get('reason')}")

    # ---- apply ---------------------------------------------------------
    if affine is not None and args.mode in ("apply", "both"):
        corrected = {}
        for n in names:
            a, b = affine[n]["a"], affine[n]["b"]
            corrected[n] = world_points(n, a * data[n]["depth"].astype(np.float64) + b)
        after = measure_layering(names, corrected, planes_w, args)
        report["layering_after"] = after
        report["affine"] = {n: {"a": affine[n]["a"], "b": affine[n]["b"]} for n in names}

        print("\n[layer] signed distance to reference planes, after correction")
        for e in after:
            cams = "  ".join(f"{k} {v['median_mm']:+7.1f}" for k, v in e["cameras"].items())
            print(f"        plane {e['plane']}: {cams}   "
                  f"spread {e['inter_camera_spread_mm']} mm")

        # ---- where the residual actually lives --------------------------
        spatial = {}
        print("\n[resid] residual after correction, by image cell and by depth")
        for n in names:
            ra = residual_analysis(data[n]["depth"], data[n]["K"], data[n]["E"],
                                   masks[n], planes_w, affine[n]["a"],
                                   affine[n]["b"], args)
            if ra is None:
                continue
            grid = ra.pop("_grid")
            spatial[n] = ra
            print(f"        {n:<8} rms {ra['overall_rms_mm']:6.2f} mm   "
                  f"spatial p05-p95 "
                  f"{ra['spatial_structure_mm'] if ra['spatial_structure_mm'] is not None else float('nan'):6.2f} mm"
                  f"   depth trend "
                  f"{ra['depth_trend_mm'] if ra['depth_trend_mm'] is not None else float('nan'):+6.2f} mm")
            if args.residual_maps and cv2 is not None:
                img = draw_residual_map(grid, n, ra["spatial_structure_mm"])
                if img is not None:
                    cv2.imwrite(str(out_dir / f"residual_map_{n}.png"), img)
        report["residual_analysis"] = spatial
        worst = [v["spatial_structure_mm"] for v in spatial.values()
                 if v.get("spatial_structure_mm") is not None]
        if worst:
            print(f"        The spatial spread reaches {max(worst):.1f} mm. "
                  f"A per-camera scalar cannot\n        remove a bias that "
                  f"varies across the frame; that part needs anchoring at "
                  f"control\n        points, not a better affine.")

        if args.consistency > 0:
            t_c = time.perf_counter()
            n_before = sum(len(v) for v in corrected.values())
            corrected = consistency_filter(corrected, names, args.consistency)
            n_after = sum(len(v) for v in corrected.values())
            print(f"\n[cons ] cross-view consistency at {args.consistency * 1e3:.0f} mm: "
                  f"{n_before} -> {n_after} points "
                  f"({100 * (1 - n_after / max(n_before, 1)):.1f}% removed) in "
                  f"{(time.perf_counter() - t_c) * 1e3:.0f} ms")
            report["consistency"] = {"radius_m": args.consistency,
                                     "points_before": int(n_before),
                                     "points_after": int(n_after)}

        fused = o3d.geometry.PointCloud()
        tinted = o3d.geometry.PointCloud()
        for i, n in enumerate(names):
            rgb = data[n]["rgb"]
            cols = None
            if rgb is not None and rgb.shape[:2] == masks[n].shape:
                cols = rgb[masks[n]]
                if args.consistency > 0:
                    cols = None      # indices no longer align after filtering
            fused += to_o3d(corrected[n], colors=cols)
            tinted += to_o3d(corrected[n], rgb01=PALETTE[i % len(PALETTE)])
        if args.voxel > 0:
            fused = fused.voxel_down_sample(args.voxel)
        o3d.io.write_point_cloud(str(out_dir / "fused_aligned.ply"), fused)
        o3d.io.write_point_cloud(str(out_dir / "fused_aligned_by_camera.ply"), tinted)
        print(f"\n[write] {out_dir / 'fused_aligned.ply'}  {len(fused.points)} points")

        if args.write_tsdf:
            print("[tsdf ] integrating corrected depth maps")
            if args.tsdf_trunc * 1e3 < 60:
                print(f"[note ] truncation {args.tsdf_trunc * 1e3:.0f} mm is "
                      f"below the inter-view disagreement measured on parcel "
                      f"tops, so two views further apart than that will still "
                      f"produce two surfaces rather than one")
            vol = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=args.tsdf_voxel,
                sdf_trunc=args.tsdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            for n in names:
                a, b = affine[n]["a"], affine[n]["b"]
                d = (a * data[n]["depth"].astype(np.float64) + b).astype(np.float32)
                d[~masks[n]] = 0.0
                h, w = d.shape
                K = data[n]["K"]
                intr = o3d.camera.PinholeCameraIntrinsic(
                    w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                depth_img = o3d.geometry.Image(d)
                grey = o3d.geometry.Image(np.zeros((h, w, 3), np.uint8))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    grey, depth_img, depth_scale=1.0,
                    depth_trunc=float(np.nanmax(d)) + 1.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd, intr, data[n]["E"])
            mesh = vol.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(out_dir / "fused_tsdf.ply"), mesh)
            print(f"[tsdf ] {out_dir / 'fused_tsdf.ply'}  "
                  f"{len(mesh.vertices)} vertices  {len(mesh.triangles)} triangles")

    # ---- absolute anchor ----------------------------------------------
    if len(planes_w) >= 2:
        n0 = np.asarray(planes_w[0]["normal"])
        seps = []
        for p in planes_w[1:]:
            n1 = np.asarray(p["normal"])
            s = abs(p["offset"] * float(np.sign(np.dot(n0, n1))) - planes_w[0]["offset"])
            seps.append(round(float(s), 4))
        report["plane_separations_m"] = seps
        print(f"\n[scale] reference plane separations: {seps} m")
        print(f"        ground truth floor-to-deck: {GT_SEPARATION_M} m")

        # Separation constrains scale alone; absolute distance constrains scale
        # and shift together. Solving both tells you which one is wrong.
        best = min(seps, key=lambda s: abs(s - GT_SEPARATION_M))
        a_est = best / GT_SEPARATION_M
        deck = min(p["offset"] for p in planes_w)
        b_est = deck - a_est * GT_DECK_PERP_M
        report["absolute"] = {
            "measured_separation_m": best,
            "scale_estimate": round(float(a_est), 6),
            "scale_error_pct": round(float((a_est - 1) * 100), 4),
            "nearest_plane_offset_m": round(float(deck), 4),
            "gt_deck_perp_m": GT_DECK_PERP_M,
            "shift_estimate_m": round(float(b_est), 4),
        }
        print(f"        implied scale : {a_est:.6f}  ({(a_est - 1) * 100:+.2f}%)")
        print(f"        implied shift : {b_est * 1e3:+.1f} mm, assuming the "
              f"nearest plane is the deck at {GT_DECK_PERP_M} m")
        print("        A large shift with near-unity scale means the depth "
              "field is displaced, not stretched. Confirm that the ground "
              "truth standoff is measured from the REFERENCE camera's optical "
              "centre before acting on this.")
        if abs(deck - GT_DECK_PERP_M) > 0.05:
            print(f"[WARN] the nearest plane sits at {deck:.3f} m while the "
                  f"ground truth standoff in da3_fuse.py is "
                  f"{GT_DECK_PERP_M:.3f} m, a difference of "
                  f"{(deck - GT_DECK_PERP_M) * 1e3:+.0f} mm. Either the "
                  f"ground truth constant is stale or the nearest plane is not "
                  f"the deck. The whole shift estimate, and therefore every "
                  f"dimension produced with --apply-absolute, rests on which "
                  f"of those it is.")

    report["runtime_s"] = round(time.perf_counter() - t_start, 3)
    (out_dir / "layer_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'layer_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())