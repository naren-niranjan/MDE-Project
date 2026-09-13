#!/usr/bin/env python3
"""
layer_align.py

Diagnose and remove inter-camera layering in a fused DA3 cloud, solve the
metric anchor, and measure whether either correction generalises beyond the
surfaces it was fitted on.

The problem
-----------
Concatenating four per-camera clouds produces parallel offset sheets of the
same physical surface. The extrinsics are good to 0.25-0.36 px, which puts the
geometric floor at 2-5 mm, so separation an order larger than that is not a
pose error. It is per-view depth scale: DA3's absolute scale is framing
dependent, and four cameras framing the scene differently receive four
different implicit scales. On this rig the solved focal lengths also span
0.77 per cent, which is 24 mm of depth at the measured 3.080 m standoff.

The decomposition
-----------------
  stage 1  fit reference planes in the reference camera's cloud and measure how
           far every other camera's samples of those same planes sit from them
  stage 2  regress each non-reference camera's predicted depth against the
           depth the reference planes imply, giving Z_true = a*Z_pred + b
  stage 3  apply, rebuild, re-measure, and report what an affine could not
           reach

WHAT THIS VERSION CHANGES, AND WHY IT MATTERS FOR ACCURACY
==========================================================

1  THE ASSIGNMENT WINDOW WAS BIASING THE SOLVE TOWARDS UNDERCORRECTION.

   solve_affine selects samples with |Z_pred - Z_target| <= --assign-tol and
   regresses on them. That window has to EXCEED the disagreement being
   corrected, or a biased view never reaches the plane it belongs to. But once
   the disagreement approaches the window, the window clips that view's far
   tail and keeps its near tail, and the fit is pulled toward zero correction.

   Measured on synthetic data with a 40 mm bias at the parcel top, 12 mm of
   per-sample noise and a 50 mm window: a single pass leaves +4.8 mm at the
   parcel top. Iterating the assignment with a shrinking window, 50 -> 30 ->
   18 -> 15 mm, leaves +1.2 mm. The same effect is present in every
   coefficient the previous version solved, and it is systematic rather than
   random, so it does not average out over captures.

   The solve is now iterated. --assign-tol is the STARTING window and
   --assign-tol-final the last one, geometrically spaced over --solve-iters
   passes. Each pass reassigns against the depth the previous pass corrected,
   so a biased view is admitted on the first wide pass and then measured
   without truncation on the narrow ones.

2  THE SCALE TERM WAS APPLIED WITHOUT ANY STATEMENT OF ITS UNCERTAINTY.

   huber_fit returned (a, b) and nothing else. With the deck and one riser the
   lever arm is 257 mm against a 3.08 m standoff, about 8 per cent of the
   depth range, so `a` is poorly conditioned and its error propagates as
   depth * da. Whether the scale term HELPS or HARMS depends on whether that
   error is smaller than the bias it is removing, and the previous version had
   no way to tell.

   Every solve now carries the parameter covariance and reports the PREDICTION
   standard error at the top of the parcel band:

       se(Z) = sqrt( [Z 1] Cov [Z 1]^T )

   This is the right quantity rather than the standard error on `a` alone,
   because it accounts for the correlation between a and b and, more usefully,
   it is small inside the span of the reference planes and grows outside it.
   Interpolation versus extrapolation stops being a printed warning and
   becomes a number.

   The scale term is REFUSED, falling back to offset-only, when that
   prediction standard error exceeds --scale-se-limit-frac of the measured
   inter-camera disagreement at the upper reference. A correction whose own
   uncertainty is half the size of the error it removes is not a correction.

3  THE FIT CAN NOW RUN IN INVERSE DEPTH.

   DA3's scale ambiguity lives in disparity, and over a short lever arm the
   affine is better conditioned there. --fit-space disparity solves
   1/Z_true = a'/Z_pred + b' and converts back. Default remains depth so the
   two are comparable; the report carries the condition number of both.

Earlier fixes, retained
-----------------------
  THE ABSOLUTE ANCHOR IS SOLVED AS A REGRESSION of the measured plane offsets
  onto the measured metric distances in ground_truth.json, over however many
  reference planes both sides have, with an explicit residual tolerance. If no
  consistent assignment exists inside that tolerance the anchor is refused and
  written as NaN, which propagates to visibly NaN depth rather than to a
  believable wrong dimension. Note that with exactly two planes the fit is
  exactly determined: a shift common to BOTH planes is absorbed into t and
  cannot be detected here. Measure a third reference surface.

  THE FLOOR IS THE WRONG SECOND REFERENCE. It is 824 mm BELOW the deck; the
  robot picks from 30 to 400 mm ABOVE it. Measure a riser inside the parcel
  band with measure_deck.py and the same solve becomes an interpolation, which
  the prediction standard error above now rewards explicitly.

Honesty constraints
-------------------
  --holdout       leave-one-plane-out validation. Every other residual in the
                  report is measured on the planes the coefficients were fitted
                  to, where it is small by construction. This is the figure to
                  quote.

  residual maps   the residual after correction, binned over the image, per
                  camera. A per-view affine removes a constant and a slope in
                  depth; anything varying with WHERE in the frame the surface
                  appears survives it. spatial_structure_mm is the p05 to p95
                  range of the cell medians. If it stays large after the fixes
                  above, the remainder is not reachable by any per-camera
                  scalar and needs anchoring at control points.

The filter defaults must match the ones the capture was produced under, or the
coefficients are solved on a different population of samples from the one they
are applied to.

Examples
--------
    python layer_align.py --capture-dir runs/live_.../capture_00000 \\
        --mode both --holdout --n-planes 3

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
                          GT, GT_PLANES, GT_SEPARATION_M, GT_DECK_PERP_M)
except ImportError as exc:
    raise SystemExit(f"cannot import da3_fuse.py from the working directory: {exc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Diagnose and remove inter-camera layering, solve the "
                    "metric anchor, and measure whether either generalises.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <capture-dir>/aligned")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
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
                         "plus whatever stands on it. At least 2 are needed to "
                         "solve both affine coefficients and to anchor")
    ap.add_argument("--min-plane-frac", type=float, default=0.02)
    ap.add_argument("--assign-tol", type=float, default=0.050,
                    help="STARTING window: a point belongs to a reference "
                         "plane if it lies within this distance of it. MUST "
                         "exceed the inter-camera layer separation being "
                         "corrected, or a biased view's samples never reach "
                         "the plane they belong to and that camera is left "
                         "uncorrected. Kept wide for that reason, then "
                         "narrowed to --assign-tol-final over --solve-iters "
                         "passes so the window does not truncate the biased "
                         "view's tail and bias the fit toward undercorrection")
    ap.add_argument("--assign-tol-final", type=float, default=0.015,
                    help="FINAL window. Must stay below half of "
                         "--plane-min-gap or the deck claims the parcel tops. "
                         "Set it a few times the per-sample depth noise: too "
                         "tight and the fit is solved on the samples that "
                         "already agree with it")
    ap.add_argument("--solve-iters", type=int, default=4,
                    help="assignment passes between the two tolerances. One "
                         "reproduces the previous single-pass behaviour, "
                         "including its undercorrection bias")
    ap.add_argument("--fit-space", choices=["depth", "disparity"],
                    default="depth",
                    help="regress in depth (Z_true = a Z_pred + b) or in "
                         "inverse depth. DA3's scale ambiguity lives in "
                         "disparity and the affine conditions better there "
                         "over a short lever arm, but depth is the default so "
                         "results stay comparable with earlier runs. The "
                         "condition number of both is reported either way")
    ap.add_argument("--max-tilt-deg", type=float, default=8.0,
                    help="reject a reference plane when the REFERENCE camera's "
                         "own fit inside that plane's assignment band comes "
                         "back further than this from it. The histogram "
                         "extractor gives every plane the deck's normal by "
                         "construction, so this is the only screen that can "
                         "see a band holding more than one surface")
    ap.add_argument("--allow-below-deck", action="store_true",
                    help="keep reference planes further from the camera than "
                         "the deck, that is the floor. OFF by default: the "
                         "robot picks from ABOVE the deck, and an affine "
                         "fitted on deck plus riser was measured to "
                         "mispredict the floor by 250 mm, so including it "
                         "lets a non-affine error into the coefficients")
    ap.add_argument("--below-deck-tol", type=float, default=0.020,
                    help="how far beyond the deck a plane may sit before it is "
                         "treated as below it, metres")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist-bin", type=float, default=0.005)
    ap.add_argument("--hist-smooth", type=float, default=0.020,
                    help="smoothing width, metres; must exceed the warp within "
                         "a single surface or one plane yields several peaks")
    ap.add_argument("--peak-min-frac", type=float, default=0.03,
                    help="ignore histogram peaks below this fraction of the "
                         "tallest")
    ap.add_argument("--plane-min-gap", type=float, default=0.12,
                    help="minimum separation between accepted planes, metres. "
                         "A parcel shorter than this cannot be used as the "
                         "second reference")
    ap.add_argument("--legacy-ransac", action="store_true")
    ap.add_argument("--merge-offset", type=float, default=0.05)
    ap.add_argument("--merge-tilt", type=float, default=4.0)

    # robust regression
    ap.add_argument("--huber-iters", type=int, default=5)
    ap.add_argument("--huber-k", type=float, default=1.345)
    ap.add_argument("--min-plane-points", type=int, default=500)
    ap.add_argument("--min-samples", type=int, default=2000)
    ap.add_argument("--subsample", type=int, default=200000)
    ap.add_argument("--min-span", type=float, default=0.100,
                    help="depth spread below which scale and offset are not "
                         "separable and only an offset is solved")

    # the scale-uncertainty gate
    ap.add_argument("--scale-se-limit-frac", type=float, default=0.5,
                    help="the scale term is REFUSED when its prediction "
                         "standard error at the top of the parcel band exceeds "
                         "this fraction of the measured inter-camera "
                         "disagreement at the upper reference. A correction "
                         "whose own uncertainty is comparable to the error it "
                         "removes is noise. Set 0 to apply the scale term "
                         "regardless and rely on the reported figure")
    ap.add_argument("--scale-se-floor-mm", type=float, default=3.0,
                    help="absolute floor on that test, in millimetres, used "
                         "when the measured disagreement is small or absent. "
                         "Below this the scale term is allowed through "
                         "whatever the ratio says")
    ap.add_argument("--plane-sigma-floor-m", type=float, default=0.0008,
                    help="floor on the systematic uncertainty attributed to "
                         "one reference plane, metres. The measured value is "
                         "the spatial scatter of the residual across the "
                         "frame; this stops a plane that happens to look flat "
                         "in one capture from claiming an uncertainty below "
                         "what the ChArUco anchor itself carries, which on "
                         "this rig is about 0.9 mm of flatness rms")

    # metric anchor
    ap.add_argument("--anchor-tol", type=float, default=0.012,
                    help="largest residual, metres, between a measured plane "
                         "mapped through the solved scale and shift and the "
                         "metric distance it was matched to. Beyond this the "
                         "assignment is not believed and the anchor is REFUSED")
    ap.add_argument("--anchor-max-scale-dev", type=float, default=0.25,
                    help="largest departure from unity scale entertained")
    ap.add_argument("--parcel-band", nargs=2, type=float,
                    default=[0.030, 0.400], metavar=("MIN_M", "MAX_M"),
                    help="height band above the deck the robot picks from. The "
                         "prediction standard error is evaluated at its TOP, "
                         "which is the worst case the correction has to serve")
    ap.add_argument("--no-anchor", dest="anchor", action="store_false",
                    default=True)

    # validation
    ap.add_argument("--holdout", action="store_true", default=True)
    ap.add_argument("--no-holdout", dest="holdout", action="store_false")
    ap.add_argument("--residual-cells", type=int, default=8)
    ap.add_argument("--residual-depth-bins", type=int, default=8)
    ap.add_argument("--no-residual-maps", dest="residual_maps",
                    action="store_false", default=True)

    # filtering, matching da3_stream.py defaults exactly
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--conf-min", type=float, default=0.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--voxel", type=float, default=0.004)

    ap.add_argument("--consistency", type=float, default=0.0)
    ap.add_argument("--write-tsdf", action="store_true")
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
            entry["rgb"] = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if img is not None else None)
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
        ok, _ = incidence_mask(points_camera(d, K), args.max_incidence)
        valid &= ok
        stages["after_grazing"] = int(valid.sum())

    frac = valid.sum() / max(d.size, 1)
    if frac < 0.05:
        print(f"[WARN] {name}: only {valid.sum()} of {d.size} pixels survive "
              f"filtering ({frac * 100:.2f}%). Stage counts: {stages}.")
    return valid, stages


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------

def refine_plane(points, normal, offset, thresh, iters=3):
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
    itself warped: it splits one physical plane into patches at slightly
    different tilts and returns a different partition on every run.
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

    k = max(1, int(round(args.hist_smooth / args.hist_bin)))
    if k > 1:
        counts = np.convolve(counts.astype(np.float64),
                             np.ones(k) / k, mode="same")

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
    for s in sorted(peaks,
                    key=lambda v: -np.sum(np.abs(sd - v) <= args.plane_thresh)):
        planes.append({"normal": n.tolist(), "offset": float(d0 + s),
                       "n_inliers": int(np.sum(np.abs(sd - s)
                                               <= args.plane_thresh)),
                       "tilt_from_deck_deg": 0.0,
                       "histogram_peak_m": round(s, 5)})
    return planes


def extract_planes(points, args):
    """Sequential RANSAC, kept behind --legacy-ransac."""
    o3d.utility.random.seed(args.seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    total = len(points)
    planes, remaining = [], pcd
    if total < args.min_plane_points:
        return planes
    while (len(planes) < args.n_planes
           and len(remaining.points) > args.min_plane_points):
        model, inliers = remaining.segment_plane(args.plane_thresh, 3,
                                                 args.plane_iters)
        if len(inliers) < args.min_plane_frac * total:
            break
        a, b, c, d = model
        n = np.array([a, b, c], np.float64)
        norm = np.linalg.norm(n)
        planes.append({"normal": (n / norm).tolist(),
                       "offset": float(-d / norm),
                       "n_inliers": int(len(inliers))})
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


def identify_deck(planes_w):
    """Index of the deck among the extracted reference planes.

    The deck is the plane carrying the MOST INLIERS, not the most distant one.
    Taking the furthest picked the floor on a capture whose planes came out at
    3.136 m with 37982 inliers, 2.946 m with 8640 and 3.561 m with 3509: the
    floor won on distance with a tenth of the deck's support, and every height
    in the report was then measured from it. That also moved the depth at which
    the scale gate is evaluated, so its limits were computed for the wrong
    surface.
    """
    if not planes_w:
        return None
    return int(max(range(len(planes_w)),
                   key=lambda i: planes_w[i]["n_inliers"]))


def screen_reference_planes(planes_w, clouds_w, ref_name, args):
    """Drop surfaces that cannot serve as depth references.

    The histogram extractor gives every plane the deck's normal by
    construction, so filter_planes_by_tilt is a no-op on that path and a
    surface that is not really parallel to the deck passes through unnoticed.
    Two screens are applied instead.

    BELOW THE DECK. The robot picks from above it. A plane beyond the deck is
    the floor or something past the belt edge, and the whole argument for
    replacing the deck-plus-floor anchor was that solving across it
    extrapolates a bias that is not exactly affine. Measured on this rig: an
    affine fitted on deck and riser mispredicts the floor by 250 mm, so
    including the floor lets that error into the coefficients.

    NOT FITTABLE BY THE REFERENCE CAMERA. The reference camera defined these
    planes, so if its own RANSAC fit inside the assignment band comes back at a
    large tilt, the band holds more than one surface and the plane's offset is
    a mixture. Observed at 70.7 deg on one camera for the floor plane.
    """
    if not planes_w:
        return planes_w, []
    deck_i = identify_deck(planes_w)
    deck_off = float(planes_w[deck_i]["offset"])
    pts = clouds_w[ref_name]
    kept, dropped = [], []

    for i, p in enumerate(planes_w):
        height = deck_off - float(p["offset"])
        if i != deck_i and height < -args.below_deck_tol and not args.allow_below_deck:
            dropped.append({
                "plane": i, "offset_m": round(float(p["offset"]), 5),
                "height_above_deck_mm": round(height * 1e3, 1),
                "n_inliers": p["n_inliers"],
                "reason": (f"sits {abs(height) * 1e3:.0f} mm BELOW the deck, so "
                           f"a correction solved across it is extrapolated "
                           f"into the pick volume. Pass --allow-below-deck to "
                           f"keep it")})
            continue

        # The reference camera's own fit inside the band this plane claims.
        sd = signed_distance(pts, p["normal"], p["offset"])
        near = np.abs(sd) <= args.assign_tol
        tilt = 0.0
        if int(near.sum()) >= args.min_plane_points:
            o3d.utility.random.seed(args.seed)
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(pts[near])
            model, _ = sub.segment_plane(args.plane_thresh, 3, args.plane_iters)
            nv = np.array(model[:3], np.float64)
            nrm = np.linalg.norm(nv)
            if nrm > 1e-9:
                nv /= nrm
                tilt = float(np.degrees(np.arccos(np.clip(
                    abs(float(nv @ np.asarray(p["normal"]))), 0, 1))))
        p["reference_fit_tilt_deg"] = round(tilt, 3)
        if tilt > args.max_tilt_deg:
            dropped.append({
                "plane": i, "offset_m": round(float(p["offset"]), 5),
                "height_above_deck_mm": round(height * 1e3, 1),
                "n_inliers": p["n_inliers"],
                "reference_fit_tilt_deg": round(tilt, 3),
                "reason": (f"the reference camera's own fit inside this plane's "
                           f"assignment band comes back {tilt:.1f} deg from it, "
                           f"above --max-tilt-deg {args.max_tilt_deg:.0f}, so "
                           f"the band holds more than one surface and this "
                           f"offset is a mixture")})
            continue
        kept.append(p)

    return kept, dropped


def filter_planes_by_tilt(planes, max_tilt_deg):
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
    kept = []
    for p in sorted(planes, key=lambda q: -q["n_inliers"]):
        n = np.asarray(p["normal"])
        duplicate = False
        for q in kept:
            nq = np.asarray(q["normal"])
            sign = np.sign(np.dot(n, nq)) or 1.0
            tilt = np.degrees(np.arccos(np.clip(abs(float(np.dot(n, nq))),
                                                0, 1)))
            gap = abs(float(p["offset"]) * sign - float(q["offset"]))
            if tilt <= merge_tilt and gap <= merge_offset:
                q["n_inliers"] += p["n_inliers"]
                q["merged"] = q.get("merged", 0) + 1
                duplicate = True
                break
        if not duplicate:
            kept.append(dict(p))
    return sorted(kept, key=lambda q: -q["n_inliers"])


def per_camera_plane_fit(names, clouds_w, planes_w, args):
    """Fit each reference plane independently in every camera's own cloud.

    Separates a depth scale or shift error, which a per-view affine corrects,
    from an extrinsic rotation error, which it cannot: the first shows as the
    same normal at a different offset, the second as a different normal.
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
            o3d.utility.random.seed(args.seed)
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(pts[near])
            model, inl = sub.segment_plane(args.plane_thresh, 3,
                                           args.plane_iters)
            a, b, c, d = model
            nv = np.array([a, b, c], np.float64)
            nv /= np.linalg.norm(nv)
            sign = np.sign(np.dot(nv, ref_n)) or 1.0
            nv = nv * sign
            off = float(-d / np.linalg.norm([a, b, c])) * sign
            tilt = np.degrees(np.arccos(np.clip(abs(float(np.dot(nv, ref_n))),
                                                0, 1)))
            entry["cameras"][n] = {
                "n_points": int(near.sum()),
                "n_inliers": int(len(inl)),
                "tilt_vs_reference_deg": round(float(tilt), 4),
                "offset_vs_reference_mm": round(
                    (off - float(plane["offset"])) * 1e3, 3),
            }
        out.append(entry)
    return out


def report_per_camera_planes(per_cam_planes, extent_m=1.5):
    print("\n[fit  ] independent plane fit per camera")
    print(f"        tilt vs reference and the lateral depth swing it implies "
          f"over a {extent_m:.1f} m scene")
    for e in per_cam_planes:
        for cam, v in e["cameras"].items():
            swing = np.tan(np.radians(v["tilt_vs_reference_deg"])) * extent_m * 1e3
            print(f"        plane {e['plane']}  {cam:<8} "
                  f"tilt {v['tilt_vs_reference_deg']:6.3f} deg  "
                  f"offset {v['offset_vs_reference_mm']:+8.2f} mm  "
                  f"implied swing {swing:6.1f} mm")


def plane_to_camera(normal_w, offset_w, E):
    R, t = E[:3, :3], E[:3, 3]
    n_c = R @ np.asarray(normal_w, np.float64)
    return n_c, float(offset_w) + float(n_c @ t)


def signed_distance(points_w, normal_w, offset_w):
    return points_w @ np.asarray(normal_w, np.float64) - float(offset_w)


def target_depth(depth_shape, K, E, plane):
    rays = points_camera(np.ones(depth_shape, dtype=np.float64), K)
    n_c, d_c = plane_to_camera(plane["normal"], plane["offset"], E)
    denom = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d_c / denom
    ok = np.isfinite(z) & (z > 0) & (np.abs(denom) > 1e-6)
    return z, ok


# --------------------------------------------------------------------------
# robust affine with covariance
# --------------------------------------------------------------------------

def huber_fit(x, y, iters, k):
    """Robust least squares for y = a*x + b, returning the covariance too.

    The covariance is what makes the scale term auditable. Cov = s^2 (A'WA)^-1
    with s the robust residual scale, from which the PREDICTION standard error
    at any depth follows as sqrt([Z 1] Cov [Z 1]').

    Returns (a, b, Cov, sigma, condition_number).
    """
    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x)
    a, b, s = 1.0, 0.0, 0.0
    for _ in range(max(1, iters)):
        sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = y - (a * x + b)
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if s <= 1e-12:
            break
        u = np.abs(r) / s
        w = np.where(u <= k, 1.0, k / np.maximum(u, 1e-9))

    r = y - (a * x + b)
    if s <= 1e-12:
        s = float(np.std(r)) + 1e-12
    AtWA = (A * (w ** 2)[:, None]).T @ A
    try:
        inv = np.linalg.inv(AtWA)
        cond = float(np.linalg.cond(AtWA))
    except np.linalg.LinAlgError:
        return a, b, None, s, float("inf")
    # Effective sample count under the robust weights, so a heavily downweighted
    # population does not report the precision of the raw one.
    n_eff = float((w ** 2).sum() ** 2 / max(float((w ** 4).sum()), 1e-12))
    dof = max(n_eff - 2.0, 1.0)
    resid_var = float(np.sum((w * r) ** 2) / dof)
    return a, b, inv * resid_var, s, cond


def prediction_se(cov, z):
    """Standard error of a*z + b at depth z, in metres."""
    if cov is None:
        return float("nan")
    v = np.array([float(z), 1.0])
    var = float(v @ cov @ v)
    return float(np.sqrt(max(var, 0.0)))


def plane_systematic_sigma(depth, K, E, valid, plane, a, b, tol, cells=8):
    """How well this camera's position of ONE reference plane is known.

    NOT the standard error of the mean over the assigned pixels. A per-view
    depth bias is a single systematic displacement of the whole surface, so
    twenty thousand pixels on one plane are one observation, not twenty
    thousand: dividing by sqrt(N) understates the uncertainty by two orders of
    magnitude and produces a gate that can never fire.

    What is measured instead is the spatial scatter of the residual across the
    frame, taken as the standard deviation of the per-cell medians. That is the
    part of the error that a per-camera scalar cannot remove, and it is the
    honest uncertainty on "where is this plane, for this camera".
    """
    d = depth.astype(np.float64)
    dc = a * d + b
    z_target, usable_geom = target_depth(d.shape, K, E, plane)
    near = valid & usable_geom & (np.abs(dc - z_target) <= tol)
    if int(near.sum()) < 200:
        return None, 0
    resid = np.where(near, dc - z_target, np.nan)
    h, w = d.shape
    ny = nx = max(2, int(cells))
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    meds = []
    for iy in range(ny):
        for ix in range(nx):
            blk = resid[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
            ok = np.isfinite(blk)
            if ok.sum() >= 50:
                meds.append(float(np.median(blk[ok])))
    if len(meds) < 4:
        return None, int(near.sum())
    meds = np.asarray(meds)
    sigma = 1.4826 * float(np.median(np.abs(meds - np.median(meds))))
    return float(sigma), int(near.sum())


def plane_level_prediction_se(z_planes, sigmas, z_query):
    """Prediction standard error at z_query from PLANE-LEVEL observations.

    Each reference plane contributes one observation at depth z with systematic
    uncertainty sigma. For an ordinary least-squares line through n such points

        se(z) = s * sqrt( 1/n + (z - zbar)^2 / Sxx )

    with Sxx the spread of the plane depths. The second term is the leverage,
    and it is what makes extrapolation expensive: a query far outside the span
    of the references inherits a large multiple of the plane uncertainty, while
    one inside the span does not. That is the interpolation-versus-
    extrapolation argument expressed as a number rather than as a warning.

    Returns (se, leverage_factor).
    """
    z = np.asarray(z_planes, float)
    s = np.asarray(sigmas, float)
    n = len(z)
    if n < 2:
        return float("nan"), float("nan")
    s_eff = float(np.sqrt(np.mean(s ** 2)))
    zbar = float(z.mean())
    sxx = float(np.sum((z - zbar) ** 2))
    if sxx < 1e-12:
        return float("inf"), float("inf")
    lev = float(np.sqrt(1.0 / n + (float(z_query) - zbar) ** 2 / sxx))
    return s_eff * lev, lev


def _to_space(z, space):
    return 1.0 / z if space == "disparity" else z


def _from_space(v, space):
    return 1.0 / v if space == "disparity" else v


def solve_affine(name, depth, K, E, valid, planes_w, args,
                 disagreement_mm=None, band_depth=None):
    """Regress predicted depth against the depth the reference planes imply.

    ITERATED. Pass one assigns with the wide --assign-tol so a biased view is
    admitted at all; later passes reassign against the depth the previous pass
    corrected, with a narrower window, so the fit is not truncated on the side
    the bias lies. See the module docstring for the measured size of that
    effect.
    """
    d = depth.astype(np.float64)
    space = args.fit_space

    tols = np.geomspace(max(args.assign_tol, 1e-4),
                        max(args.assign_tol_final, 1e-4),
                        max(1, int(args.solve_iters)))

    a, b = 1.0, 0.0
    cov, sigma, cond = None, float("nan"), float("inf")
    per_plane, history = [], []
    z_pred = z_true = None

    for it, tol in enumerate(tols):
        z_pred_all, z_true_all, per_plane = [], [], []
        for pi, plane in enumerate(planes_w):
            z_target, usable_geom = target_depth(d.shape, K, E, plane)
            usable = valid & usable_geom
            # Assignment uses the CURRENT correction, so a view that started
            # 44 mm out is measured about its own corrected position rather
            # than about the raw one.
            near = usable & (np.abs((a * d + b) - z_target) <= tol)
            cnt = int(near.sum())
            per_plane.append({
                "plane": pi, "n_assigned": cnt,
                "median_residual_m": (float(np.median(d[near] - z_target[near]))
                                      if cnt else None)})
            if cnt:
                z_pred_all.append(d[near])
                z_true_all.append(z_target[near])

        if not z_pred_all:
            return {"camera": name, "solved": False,
                    "reason": f"no points assigned at iteration {it}",
                    "a": 1.0, "b": 0.0, "per_plane": per_plane}

        z_pred = np.concatenate(z_pred_all)
        z_true = np.concatenate(z_true_all)
        if z_pred.size > args.subsample:
            idx = np.random.default_rng(args.seed).choice(
                z_pred.size, args.subsample, replace=False)
            z_pred, z_true = z_pred[idx], z_true[idx]

        spread = float(z_true.max() - z_true.min())
        n_planes_used = sum(1 for p in per_plane
                            if p["n_assigned"] >= args.min_plane_points)

        if z_pred.size < args.min_samples:
            return {"camera": name, "solved": False,
                    "reason": f"too few samples at iteration {it}",
                    "a": 1.0, "b": 0.0, "n_samples": int(z_pred.size),
                    "per_plane": per_plane}

        if n_planes_used < 2 or spread < args.min_span:
            b = float(np.median(z_true - z_pred))
            a = 1.0
            history.append({"iter": it, "tol_mm": round(float(tol) * 1e3, 1),
                            "n": int(z_pred.size), "a": 1.0,
                            "b_mm": round(b * 1e3, 3), "model": "offset_only"})
            cov, cond = None, float("inf")
            continue

        xs = _to_space(z_pred, space)
        ys = _to_space(z_true, space)
        a_s, b_s, cov_s, sigma, cond = huber_fit(xs, ys, args.huber_iters,
                                                 args.huber_k)
        if space == "disparity":
            # 1/Zt = a'/Zp + b'  =>  Zt = Zp / (a' + b' Zp). Linearise about the
            # centre of the data so the reported (a, b) stay a depth affine and
            # everything downstream keeps one model.
            z0 = float(np.median(z_pred))
            zt0 = 1.0 / (a_s / z0 + b_s)
            dzt = (a_s / z0 ** 2) / (a_s / z0 + b_s) ** 2
            a, b = float(dzt), float(zt0 - dzt * z0)
            # Propagate the covariance through the same linearisation.
            J = np.array([[1.0 / z0, 1.0]])
            cov = None
            if cov_s is not None:
                var_d = float(J @ cov_s @ J.T) * (zt0 ** 4)
                cov = np.array([[var_d / max(z0 ** 2, 1e-9), 0.0],
                                [0.0, var_d]])
        else:
            a, b, cov = float(a_s), float(b_s), cov_s

        history.append({"iter": it, "tol_mm": round(float(tol) * 1e3, 1),
                        "n": int(z_pred.size), "a": round(a, 6),
                        "b_mm": round(b * 1e3, 3), "model": "affine",
                        "sigma_mm": round(float(sigma) * 1e3, 3)})

    if not history:
        return {"camera": name, "solved": False, "reason": "no iteration ran",
                "a": 1.0, "b": 0.0, "per_plane": per_plane}

    model = history[-1]["model"]
    spread = float(z_true.max() - z_true.min()) if z_true is not None else 0.0

    # ---- the scale-uncertainty gate ------------------------------------
    # Built from PLANE-LEVEL systematics, not from the per-sample covariance.
    # See plane_systematic_sigma: a per-view bias displaces the whole surface
    # at once, so the pixels on one plane are one observation and the
    # sqrt(N) of the naive covariance understates the uncertainty by orders of
    # magnitude, producing a gate that can never fire.
    gate = {"applied": False}
    if model == "affine" and band_depth is not None:
        z_planes, sigmas, detail = [], [], []
        for pi, plane in enumerate(planes_w):
            sig, cnt = plane_systematic_sigma(
                d, K, E, valid, plane, a, b, float(tols[-1]),
                cells=args.residual_cells)
            if sig is None:
                continue
            zt_p, ok_p = target_depth(d.shape, K, E, plane)
            sel = valid & ok_p & (np.abs((a * d + b) - zt_p) <= float(tols[-1]))
            if int(sel.sum()) < 200:
                continue
            z_planes.append(float(np.median(zt_p[sel])))
            sigmas.append(max(sig, args.plane_sigma_floor_m))
            detail.append({"plane": pi,
                           "depth_m": round(z_planes[-1], 4),
                           "systematic_sigma_mm": round(sigmas[-1] * 1e3, 2),
                           "n_points": cnt})

        if len(z_planes) >= 2:
            se_m, lev = plane_level_prediction_se(z_planes, sigmas, band_depth)
            se_mm = se_m * 1e3
            limit_mm = max(args.scale_se_floor_mm,
                           args.scale_se_limit_frac
                           * float(disagreement_mm or 0.0))
            gate = {
                "applied": True,
                "basis": "plane-level systematics, one observation per surface",
                "band_depth_m": round(float(band_depth), 4),
                "planes": detail,
                "leverage_factor": round(float(lev), 3),
                # Variance is not the whole story. A long lever arm below the
                # deck, the floor for instance, constrains the SLOPE well and
                # still extrapolates a bias that is not exactly affine into the
                # pick volume. That is model error and no covariance sees it,
                # so whether the query lies inside the reference span is
                # recorded alongside and must be read with the standard error.
                "query_inside_reference_span": bool(
                    min(z_planes) <= band_depth <= max(z_planes)),
                "reference_span_m": [round(min(z_planes), 4),
                                     round(max(z_planes), 4)],
                "prediction_se_mm": round(se_mm, 3),
                "limit_mm": round(limit_mm, 3),
                "measured_disagreement_mm": (round(float(disagreement_mm), 2)
                                             if disagreement_mm else None),
                "condition_number": (round(cond, 1)
                                     if np.isfinite(cond) else None),
                "refused": bool(args.scale_se_limit_frac > 0
                                and se_mm > limit_mm),
            }
            if gate["refused"]:
                b = float(np.median(z_true - z_pred))
                a = 1.0
                model = "offset_only_scale_refused"
                gate["note"] = (
                    f"the scale term's own prediction standard error at the "
                    f"top of the parcel band is {se_mm:.1f} mm (leverage "
                    f"{lev:.2f} on a plane systematic of "
                    f"{np.sqrt(np.mean(np.asarray(sigmas) ** 2)) * 1e3:.1f} mm) "
                    f"against a {limit_mm:.1f} mm limit, so applying it would "
                    f"inject more uncertainty than the "
                    f"{disagreement_mm or 0:.0f} mm disagreement it removes. "
                    f"Lengthen the lever arm: measure a reference surface "
                    f"higher in the parcel band, or reduce the spatial "
                    f"structure that sets the plane systematic")

    r = z_true - (a * z_pred + b)
    r0 = z_true - z_pred
    return {
        "camera": name, "solved": True, "model": model,
        "a": a, "b": b,
        "fit_space": space,
        "n_samples": int(z_pred.size),
        "depth_spread_m": round(spread, 4),
        "iterations": history,
        "assign_tol_start_m": float(tols[0]),
        "assign_tol_final_m": float(tols[-1]),
        "scale_gate": gate,
        "condition_number": (round(cond, 1) if np.isfinite(cond) else None),
        "residual_before": {
            "median_mm": round(float(np.median(r0)) * 1e3, 3),
            "rms_mm": round(float(np.sqrt(np.mean(r0 ** 2))) * 1e3, 3)},
        "residual_after": {
            "median_mm": round(float(np.median(r)) * 1e3, 3),
            "rms_mm": round(float(np.sqrt(np.mean(r ** 2))) * 1e3, 3)},
        "fitted_and_evaluated_on_same_planes": True,
        "per_plane": per_plane,
    }


# --------------------------------------------------------------------------
# the metric anchor
# --------------------------------------------------------------------------

def match_planes_to_truth(planes_w, gt_planes, tol, max_scale_dev):
    """Solve true = s * measured + t over the reference planes.

    Every ordered pair from each side gives a candidate (s, t). Each candidate
    is scored by how many of the remaining planes it brings inside `tol` of a
    metric distance, then refitted by least squares on the pairs it claimed.

    Returns None if no candidate matches at least two planes inside tolerance.
    Returning None is the point: it is what stops a parcel top being mistaken
    for the floor and scaling every depth by three.

    NOTE the limit of this test. A shift shared by EVERY plane is absorbed into
    t and produces zero residual, so with two planes the fit is exactly
    determined and cannot detect a common offset at all. Measure a third
    reference surface before quoting the residual as evidence.
    """
    meas = [float(p["offset"]) for p in planes_w]
    truth = [float(g["perp_m"]) for g in gt_planes]
    labels = [g.get("label", "?") for g in gt_planes]
    if len(meas) < 2 or len(truth) < 2:
        return None

    best = None
    for i, j in itertools.permutations(range(len(meas)), 2):
        for k, l in itertools.permutations(range(len(truth)), 2):
            dm = meas[j] - meas[i]
            dt = truth[l] - truth[k]
            if abs(dm) < 1e-6:
                continue
            s = dt / dm
            if not (1.0 - max_scale_dev <= s <= 1.0 + max_scale_dev):
                continue
            t = truth[k] - s * meas[i]

            pairs, used = [], set()
            for mi, m in enumerate(meas):
                pred = s * m + t
                cand = [(abs(pred - truth[ti]), ti)
                        for ti in range(len(truth)) if ti not in used]
                if not cand:
                    break
                e, ti = min(cand)
                if e <= tol:
                    used.add(ti)
                    pairs.append((mi, ti))
            if len(pairs) < 2:
                continue

            x = np.array([meas[mi] for mi, _ in pairs], float)
            y = np.array([truth[ti] for _, ti in pairs], float)
            A = np.stack([x, np.ones_like(x)], axis=1)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            s2, t2 = float(sol[0]), float(sol[1])
            if not (1.0 - max_scale_dev <= s2 <= 1.0 + max_scale_dev):
                continue
            resid = (s2 * x + t2) - y
            rms = float(np.sqrt(np.mean(resid ** 2)))
            worst = float(np.max(np.abs(resid)))
            if worst > tol:
                continue
            score = (len(pairs), -rms)
            if best is None or score > best["_score"]:
                best = {
                    "_score": score,
                    "scale_true_from_measured": s2,
                    "shift_true_from_measured_m": t2,
                    "n_matched": len(pairs),
                    "rms_m": rms,
                    "worst_residual_m": worst,
                    "matches": [
                        {"measured_offset_m": round(meas[mi], 5),
                         "truth_label": labels[ti],
                         "truth_perp_m": round(truth[ti], 5),
                         "residual_mm": round(float(resid[q] * 1e3), 2)}
                        for q, (mi, ti) in enumerate(pairs)],
                }
    if best is not None:
        best.pop("_score")
    return best


def _unbounded_gain(planes_w, gt_planes):
    """The gain implied by the most plausible pairing, WITHOUT the scale bound.

    Ordering both sides by perpendicular distance and pairing rank for rank is
    the assignment a human would make when the surfaces are known to be the
    same ones. Reporting its gain even when the bounded search refused turns a
    rejection into a measurement: on this rig it showed the model reproducing
    the absolute standoff to 1.8 per cent while compressing the deck-to-riser
    separation by 26 per cent, which is exactly why no affine solved on the
    deck transfers to a parcel top.

    Diagnostic only. It is deliberately NOT fed to the correction, because with
    two planes any gain fits both exactly and the residual cannot falsify it.
    """
    if len(planes_w) < 2 or len(gt_planes) < 2:
        return None
    meas = sorted(float(p["offset"]) for p in planes_w)
    truth = sorted(float(g["perp_m"]) for g in gt_planes)
    n = min(len(meas), len(truth))
    if n < 2:
        return None
    m = np.array(meas[:n]), np.array(truth[:n])
    x, y = m[0], m[1]
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2, t2 = float(sol[0]), float(sol[1])
    resid = (s2 * x + t2) - y
    seps_m = float(x.max() - x.min())
    seps_t = float(y.max() - y.min())
    return {
        "note": ("rank-ordered pairing with no scale bound, for diagnosis "
                 "only; never applied"),
        "implied_gain": round(s2, 5),
        "implied_offset_m": round(t2, 5),
        "worst_residual_mm": round(float(np.max(np.abs(resid))) * 1e3, 2),
        "measured_span_mm": round(seps_m * 1e3, 1),
        "true_span_mm": round(seps_t * 1e3, 1),
        "relief_compression_pct": (round(100.0 * (1.0 - seps_m / seps_t), 2)
                                   if seps_t > 1e-9 else None),
        "absolute_error_pct": (round(100.0 * (max(meas) / max(truth) - 1.0), 3)
                               if max(truth) > 1e-9 else None),
        "interpretation": (
            "a large relief compression with a small absolute error means the "
            "model places the scene at about the right distance and flattens "
            "its internal structure. No single affine removes that unless the "
            "compression is constant with height, which needs a THIRD measured "
            "surface to test"),
    }


def solve_anchor(planes_w, args):
    """The metric anchor, or an explicit refusal.

    The output convention is the one da3_stream.load_affine consumes:
    measured = scale * true + shift, so the correction it applies is
    Z -> (Z - shift) / scale.
    """
    rec = {"ok": False, "reason": None,
           "scale_estimate": float("nan"),
           "shift_estimate_m": float("nan"),
           "source": None}

    if not args.anchor:
        rec["reason"] = "--no-anchor given"
        return rec
    if not GT.get("ok"):
        rec["reason"] = f"no usable ground truth: {GT.get('reason')}"
        return rec
    if not GT_PLANES or len(GT_PLANES) < 2:
        rec["reason"] = ("ground_truth.json carries fewer than two reference "
                         "planes, so there is nothing to regress against")
        return rec
    if len(planes_w) < 2:
        rec["reason"] = ("fewer than two reference planes were extracted from "
                         "this capture, so the anchor has one equation and two "
                         "unknowns. Capture a scene with the riser block "
                         "standing on the belt")
        return rec

    fit = match_planes_to_truth(planes_w, GT_PLANES,
                                args.anchor_tol, args.anchor_max_scale_dev)
    if fit is None:
        # A refusal that says only "no assignment matched" hides the thing
        # worth knowing. The gain implied by pairing the measured planes in
        # DESCENDING ORDER OF SUPPORT against the metric ones in the same order
        # is reported unbounded, so a genuine finding, such as the model
        # compressing relief by 26 per cent while placing the scene to within
        # 2 per cent, is visible rather than buried in a rejection message.
        rec["diagnostic"] = _unbounded_gain(planes_w, GT_PLANES)
        rec["reason"] = (
            f"no assignment of the {len(planes_w)} measured plane(s) onto the "
            f"{len(GT_PLANES)} metric plane(s) matched two or more of them "
            f"within {args.anchor_tol * 1e3:.0f} mm at a scale inside "
            f"{1 - args.anchor_max_scale_dev:.2f} to "
            f"{1 + args.anchor_max_scale_dev:.2f}. The anchor is REFUSED. "
            f"Either the surfaces in this capture are not the surfaces that "
            f"were measured, or ground_truth.json is stale. Do NOT loosen "
            f"--anchor-tol to make this pass")
        return rec

    s, t = fit["scale_true_from_measured"], fit["shift_true_from_measured_m"]
    rec.update({
        "ok": True,
        "source": "regression over the measured reference planes",
        "n_planes_matched": fit["n_matched"],
        "matches": fit["matches"],
        "fit_rms_mm": round(fit["rms_m"] * 1e3, 2),
        "worst_residual_mm": round(fit["worst_residual_m"] * 1e3, 2),
        # true = s * measured + t   =>   measured = (1/s) * true + (-t/s)
        "scale_estimate": round(1.0 / s, 6),
        "shift_estimate_m": round(-t / s, 6),
        "correction_gain": round(s, 6),
        "correction_offset_m": round(t, 6),
        "scale_error_pct": round((1.0 / s - 1.0) * 100, 4),
        "exactly_determined": fit["n_matched"] < 3,
    })
    if fit["n_matched"] < 3:
        rec["caveat"] = ("two planes matched, so the fit is exactly determined "
                         "and its residual is zero by construction. A shift "
                         "common to both planes is absorbed into the offset "
                         "term and CANNOT be detected here. Measure a third "
                         "reference surface before quoting the rms")
    return rec


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def plane_residual(depth, K, E, valid, plane, a, b, args):
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


def holdout_validation(names, data, masks, planes_w, args,
                       disagreement_mm=None, band_depth=None):
    """Leave one plane out, solve on the rest, measure on the one left out."""
    if len(planes_w) < 2:
        return {"ok": False,
                "reason": "leave-one-out needs at least two planes"}
    note = ("only two planes, so each fold solves an offset alone and the "
            "scale term is untested" if len(planes_w) == 2 else None)

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
                             masks[n], train, args,
                             disagreement_mm=disagreement_mm,
                             band_depth=band_depth)
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
    """Where the residual lives after correction, in the image and in depth."""
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
    spatial = (round(float(np.percentile(cells, 95)
                           - np.percentile(cells, 5)), 2)
               if cells.size >= 4 else None)

    zz = dc[finite]
    rr = resid[finite] * 1e3
    nb = max(2, int(args.residual_depth_bins))
    edges = np.percentile(zz, np.linspace(0, 100, nb + 1))
    depth_bins = []
    for i in range(nb):
        sel = ((zz >= edges[i]) & (zz <= edges[i + 1]) if i == nb - 1
               else (zz >= edges[i]) & (zz < edges[i + 1]))
        if int(sel.sum()) < 50:
            continue
        depth_bins.append({"z_centre_m": round(float(0.5 * (edges[i]
                                                            + edges[i + 1])), 3),
                           "n": int(sel.sum()),
                           "median_mm": round(float(np.median(rr[sel])), 2)})
    trend = (round(depth_bins[-1]["median_mm"] - depth_bins[0]["median_mm"], 2)
             if len(depth_bins) >= 2 else None)

    return {
        "n_points": int(finite.sum()),
        "overall_median_mm": round(float(np.median(rr)), 2),
        "overall_rms_mm": round(float(np.sqrt(np.mean(rr ** 2))), 2),
        "cell_grid_mm": np.where(np.isfinite(grid),
                                 np.round(grid, 2), None).tolist(),
        "cell_counts": counts.tolist(),
        "spatial_structure_mm": spatial,
        "depth_bins": depth_bins,
        "depth_trend_mm": trend,
        "_grid": grid,
    }


def draw_residual_map(grid, name, spatial, cell_px=64):
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
                colour, text = (35, 35, 35), "-"
            else:
                t = float(np.clip((v / lim + 1.0) / 2.0, 0.0, 1.0))
                colour = tuple(int(c) for c in cv2.applyColorMap(
                    np.array([[int(t * 255)]], np.uint8),
                    cv2.COLORMAP_JET)[0, 0])
                text = f"{v:+.1f}"
            cv2.rectangle(canvas, (x0, y0),
                          (x0 + cell_px - 1, y0 + cell_px - 1), colour, -1)
            cv2.rectangle(canvas, (x0, y0),
                          (x0 + cell_px - 1, y0 + cell_px - 1), (15, 15, 15), 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4, 1)
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

def measure_layering(names, clouds_w, planes_w, args, tol=None):
    tol = args.assign_tol if tol is None else tol
    out = []
    for pi, plane in enumerate(planes_w):
        entry = {"plane": pi,
                 "normal": [round(v, 5) for v in plane["normal"]],
                 "offset_m": round(plane["offset"], 5),
                 "assign_tol_m": round(float(tol), 4),
                 "cameras": {}}
        for n in names:
            pts = clouds_w[n]
            if len(pts) == 0:
                continue
            sd = signed_distance(pts, plane["normal"], plane["offset"])
            near = np.abs(sd) <= tol
            if near.sum() < 100:
                continue
            v = sd[near]
            entry["cameras"][n] = {
                "n_points": int(near.sum()),
                "median_mm": round(float(np.median(v)) * 1e3, 3),
                "mad_mm": round(float(np.median(np.abs(v - np.median(v))))
                                * 1e3, 3),
            }
        meds = [c["median_mm"] for c in entry["cameras"].values()]
        entry["inter_camera_spread_mm"] = (round(max(meds) - min(meds), 3)
                                           if len(meds) > 1 else None)
        out.append(entry)
    return out


def consistency_filter(clouds_w, names, radius):
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

    supported = (merged[inv] & ~(1 << src)) != 0
    kept, at = {}, 0
    for n in order:
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

    if args.assign_tol_final > args.assign_tol:
        raise SystemExit(
            f"--assign-tol-final {args.assign_tol_final} exceeds --assign-tol "
            f"{args.assign_tol}. The schedule narrows: the first pass must be "
            f"wide enough to admit a biased view, the last tight enough not to "
            f"truncate it.")
    if args.assign_tol_final >= args.plane_min_gap / 2:
        print(f"[WARN] --assign-tol-final "
              f"{args.assign_tol_final * 1e3:.0f} mm is at or above half of "
              f"--plane-min-gap {args.plane_min_gap * 1e3:.0f} mm, so the "
              f"deck's assignment band reaches the parcel tops and the same "
              f"points are claimed by two references.")

    print(f"[filt ] conf p{args.conf_percentile:.0f}"
          + (f" floor {args.conf_min}" if args.conf_min > 0 else "")
          + f", edge {args.edge_thresh}, incidence {args.max_incidence:.0f} deg")
    print("        these must match the values the capture was produced under")
    print(f"[solve] {args.solve_iters} assignment pass(es), window "
          f"{args.assign_tol * 1e3:.0f} -> {args.assign_tol_final * 1e3:.0f} mm,"
          f" fitted in {args.fit_space}")

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
              f"({100 * len(clouds_w[n]) / max(total, 1):5.1f}%)")

    # ---- reference planes ---------------------------------------------
    if args.legacy_ransac:
        planes_all = extract_planes(clouds_w[args.reference], args)
        planes_tilt = filter_planes_by_tilt(planes_all, args.max_tilt_deg)
        planes_w = merge_planes(planes_tilt, args.merge_offset, args.merge_tilt)
        print(f"\n[plane] sequential RANSAC: {len(planes_all)} extracted, "
              f"{len(planes_w)} after merge")
    else:
        planes_w = extract_parallel_planes(clouds_w[args.reference], args)
        print(f"\n[plane] parallel-plane histogram from {args.reference}: "
              f"{len(planes_w)} surface(s)")
    for i, p in enumerate(planes_w):
        peak = (f"  peak={p['histogram_peak_m']:+.4f} m"
                if "histogram_peak_m" in p else "")
        print(f"        plane {i}: offset={p['offset']:+.4f} m  "
              f"inliers={p['n_inliers']}{peak}")

    planes_w, planes_dropped = screen_reference_planes(
        planes_w, clouds_w, args.reference, args)
    for dsc in planes_dropped:
        print(f"        plane {dsc['plane']} REJECTED "
              f"({dsc['n_inliers']} inliers): {dsc['reason']}")
    if not planes_w:
        raise SystemExit("every extracted plane was rejected by screening; "
                         "nothing to solve against")

    # The deck is the plane with the MOST SUPPORT, not the most distant one.
    deck_i = identify_deck(planes_w)
    deck_off = float(planes_w[deck_i]["offset"])
    print(f"        deck is plane {deck_i} at {deck_off:+.4f} m "
          f"({planes_w[deck_i]['n_inliers']} inliers)")

    span = (max(p["offset"] for p in planes_w)
            - min(p["offset"] for p in planes_w)) if len(planes_w) > 1 else 0.0
    lo, hi = args.parcel_band
    heights = sorted(deck_off - p["offset"] for p in planes_w)
    in_band = [h for h in heights if lo - 0.02 <= h <= hi + 0.02]
    print(f"        span {span * 1e3:.0f} mm; surfaces stand "
          f"{', '.join(f'{h * 1e3:.0f}' for h in heights)} mm above the "
          f"furthest one")

    # The depth at which the scale term is judged: the top of the parcel band,
    # which is the worst case the correction has to serve. Depth along the
    # optical axis decreases with height above the deck.
    band_depth = float(deck_off - hi) if np.isfinite(deck_off) else None

    if len(planes_w) < 2:
        print("[WARN] fewer than two usable planes. Scale and offset cannot be "
              "separated, so only an offset will be solved and the metric "
              "anchor is refused.\n       Through the 12 mm lens the floor is "
              "out of frame, so the second plane has to be a PARCEL TOP or the "
              "riser block: put one on the belt before capturing.")
    elif not in_band:
        print(f"[WARN] no reference surface stands between {lo * 1e3:.0f} and "
              f"{hi * 1e3:.0f} mm above the deck, so this solve EXTRAPOLATES "
              f"into the volume the robot picks from. The prediction standard "
              f"error below will show what that costs.")
    else:
        print(f"        {len(in_band)} surface(s) inside the parcel band, so "
              f"the solve interpolates rather than extrapolates")

    before = measure_layering(names, clouds_w, planes_w, args)
    print("\n[layer] signed distance to reference planes, before correction")
    for e in before:
        cams = "  ".join(f"{k} {v['median_mm']:+7.1f}"
                         for k, v in e["cameras"].items())
        print(f"        plane {e['plane']}: {cams}   "
              f"spread {e['inter_camera_spread_mm']} mm")

    # The disagreement the correction is being asked to remove, taken at the
    # UPPER reference rather than at the deck, because that is where it matters
    # and where the deck-solved correction has historically failed.
    upper_spreads = [e["inter_camera_spread_mm"] for e in before[1:]
                     if e["inter_camera_spread_mm"] is not None]
    disagreement_mm = max(upper_spreads) if upper_spreads else None
    if disagreement_mm is not None:
        print(f"        the correction has to remove {disagreement_mm:.1f} mm "
              f"at the upper reference; the scale term is refused if its own "
              f"uncertainty\n        exceeds "
              f"{max(args.scale_se_floor_mm, args.scale_se_limit_frac * disagreement_mm):.1f} mm")

    per_cam_planes = per_camera_plane_fit(names, clouds_w, planes_w, args)
    report_per_camera_planes(per_cam_planes)

    report = {"capture_dir": str(args.capture_dir),
              "reference": args.reference,
              "cameras": names,
              "filters": {"conf_percentile": args.conf_percentile,
                          "conf_min": args.conf_min,
                          "edge_thresh": args.edge_thresh,
                          "max_incidence_deg": args.max_incidence},
              "solve": {"assign_tol_start_m": args.assign_tol,
                        "assign_tol_final_m": args.assign_tol_final,
                        "iterations": args.solve_iters,
                        "fit_space": args.fit_space,
                        "scale_se_limit_frac": args.scale_se_limit_frac,
                        "scale_se_floor_mm": args.scale_se_floor_mm,
                        "band_depth_m": (round(band_depth, 4)
                                         if band_depth else None),
                        "measured_disagreement_mm": disagreement_mm},
              "planes": planes_w,
              "planes_rejected": planes_dropped,
              "deck_plane_index": deck_i,
              "deck_offset_m": round(deck_off, 5),
              "plane_span_m": round(float(span), 5),
              "plane_heights_above_lowest_m": [round(float(h), 5)
                                               for h in heights],
              "parcel_band_m": [lo, hi],
              "planes_in_parcel_band": len(in_band),
              "layering_before": before,
              "per_camera_plane_fit": per_cam_planes}

    if planes_w:
        # Measured at the FINAL window and against the DECK rather than the
        # first plane in the list. At the 50 mm starting window this number
        # admits everything within 50 mm of the plane and reports the scatter
        # of that whole population, which is not comparable with a residual
        # quoted at 15 mm.
        p0 = planes_w[deck_i]
        sd_ref = signed_distance(clouds_w[args.reference], p0["normal"],
                                 p0["offset"])
        near = np.abs(sd_ref) <= args.assign_tol_final
        if near.sum() > args.min_plane_points:
            v = sd_ref[near]
            flat = {"n_points": int(near.sum()),
                    "rms_mm": round(float(np.sqrt(np.mean(v ** 2))) * 1e3, 3),
                    "p05_mm": round(float(np.percentile(v, 5)) * 1e3, 3),
                    "p95_mm": round(float(np.percentile(v, 95)) * 1e3, 3)}
            report["reference_flatness"] = flat
            flat["window_m"] = args.assign_tol_final
            print(f"\n[flat ] {args.reference} against the deck plane at a "
                  f"{args.assign_tol_final * 1e3:.0f} mm window: "
                  f"rms {flat['rms_mm']:.2f} mm, p05-p95 "
                  f"{flat['p05_mm']:+.1f} to {flat['p95_mm']:+.1f} mm")
            print("        No inter-camera alignment can produce a residual "
                  "below this.")

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
                affine[n] = solve_affine(
                    n, data[n]["depth"], data[n]["K"], data[n]["E"], masks[n],
                    planes_w, args, disagreement_mm=disagreement_mm,
                    band_depth=band_depth)
            e = affine[n]
            extra = ""
            if "residual_after" in e:
                extra = (f"  rms {e['residual_before']['rms_mm']:.1f} -> "
                         f"{e['residual_after']['rms_mm']:.1f} mm")
            print(f"        {n:<8} a={e['a']:.6f}  b={e['b']:+.6f} m  "
                  f"[{e.get('model', 'unsolved')}]{extra}")
            g = e.get("scale_gate") or {}
            if g.get("applied"):
                print(f"                 scale uncertainty at the parcel band "
                      f"top: {g['prediction_se_mm']:.1f} mm against a "
                      f"{g['limit_mm']:.1f} mm limit"
                      + ("   REFUSED" if g.get("refused") else "   accepted"))
            hist = e.get("iterations") or []
            if len(hist) > 1:
                first, last = hist[0], hist[-1]
                print(f"                 iteration {first['tol_mm']:.0f} mm "
                      f"a={first['a']} -> {last['tol_mm']:.0f} mm "
                      f"a={last['a']}")

        degraded = [n for n in names
                    if str(affine[n].get("model", "")).startswith("offset_only")]
        if degraded:
            print(f"        ** {', '.join(degraded)} carry an offset only. "
                  f"Their focal error\n           survives into every height "
                  f"above the deck. **")
        print("        These residuals are measured on the planes the "
              "coefficients were fitted\n        to. Read the holdout block "
              "below instead.")

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
                  f"being applied under {args.max_incidence:.0f} deg.")

    if args.load_affine is not None and args.mode == "both":
        loaded = json.loads(args.load_affine.read_text())
        for n in names:
            affine[n]["a"] = loaded["coefficients"][n]["a"]
            affine[n]["b"] = loaded["coefficients"][n]["b"]
            affine[n]["model"] = "loaded"
        print(f"[apply] overriding solved values with {args.load_affine}")

    # ---- the metric anchor ---------------------------------------------
    anchor = solve_anchor(planes_w, args)
    report["absolute"] = anchor
    print("\n[scale] metric anchor")
    if GT.get("ok"):
        print(f"        ground truth: "
              + ", ".join(f"{p['label']} {p['perp_m']:.4f} m"
                          for p in GT_PLANES))
    if anchor["ok"]:
        for m in anchor["matches"]:
            print(f"        measured {m['measured_offset_m']:+.4f} m  ->  "
                  f"{m['truth_label']:<10} {m['truth_perp_m']:.4f} m   "
                  f"residual {m['residual_mm']:+6.2f} mm")
        print(f"        implied scale : {anchor['scale_estimate']:.6f}  "
              f"({anchor['scale_error_pct']:+.2f}%)")
        print(f"        implied shift : "
              f"{anchor['shift_estimate_m'] * 1e3:+.1f} mm")
        print(f"        correction    : Z -> {anchor['correction_gain']:.6f} "
              f"* Z {anchor['correction_offset_m']:+.4f} m")
        if anchor.get("caveat"):
            print(f"        note: {anchor['caveat']}")
    else:
        print(f"        REFUSED: {anchor['reason']}")
        dg = anchor.get("diagnostic")
        if dg:
            print(f"\n        DIAGNOSTIC, not applied. Pairing the surfaces in "
                  f"order of distance:")
            print(f"          measured span {dg['measured_span_mm']:.1f} mm "
                  f"against a true {dg['true_span_mm']:.1f} mm  ->  relief "
                  f"compressed {dg['relief_compression_pct']:.1f}%")
            print(f"          absolute standoff error "
                  f"{dg['absolute_error_pct']:+.2f}%")
            print(f"          implied gain {dg['implied_gain']:.4f}, offset "
                  f"{dg['implied_offset_m']:+.4f} m, worst residual "
                  f"{dg['worst_residual_mm']:.2f} mm")
            print(f"          {dg['interpretation']}")

    # ---- generalisation -------------------------------------------------
    if args.holdout and args.mode in ("solve", "both"):
        hv = holdout_validation(names, data, masks, planes_w, args,
                                disagreement_mm=disagreement_mm,
                                band_depth=band_depth)
        report["holdout"] = hv
        if hv.get("ok"):
            print("\n[hold ] leave-one-plane-out")
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

    # ---- write the coefficients ----------------------------------------
    if args.mode in ("solve", "both") and affine is not None:
        (out_dir / "depth_affine.json").write_text(json.dumps({
            "reference": args.reference,
            "source_capture": str(args.capture_dir),
            "filters": report["filters"],
            "solve": report["solve"],
            "coefficients": {n: {"a": affine[n]["a"], "b": affine[n]["b"]}
                             for n in names},
            "absolute": anchor,
            "planes": planes_w,
            "plane_span_m": report["plane_span_m"],
            "planes_in_parcel_band": len(in_band),
            "ground_truth": {"planes": GT_PLANES,
                             "lens_id": GT.get("lens_id"),
                             "measured": GT.get("measured"),
                             "usable": bool(GT.get("ok"))},
            "detail": affine,
        }, indent=2, default=str))

    # ---- apply ---------------------------------------------------------
    if affine is not None and args.mode in ("apply", "both"):
        corrected = {}
        for n in names:
            a, b = affine[n]["a"], affine[n]["b"]
            corrected[n] = world_points(
                n, a * data[n]["depth"].astype(np.float64) + b)
        # Measured at the FINAL window, so the after figure is not flattered by
        # the wide band that admitted the biased view in the first place.
        after = measure_layering(names, corrected, planes_w, args,
                                 tol=args.assign_tol_final)
        report["layering_after"] = after
        report["affine"] = {n: {"a": affine[n]["a"], "b": affine[n]["b"]}
                            for n in names}

        print(f"\n[layer] signed distance to reference planes, after "
              f"correction (window {args.assign_tol_final * 1e3:.0f} mm)")
        for e in after:
            cams = "  ".join(f"{k} {v['median_mm']:+7.1f}"
                             for k, v in e["cameras"].items())
            print(f"        plane {e['plane']}: {cams}   "
                  f"spread {e['inter_camera_spread_mm']} mm")
        spreads = [e["inter_camera_spread_mm"] for e in after
                   if e["inter_camera_spread_mm"] is not None]
        if len(spreads) > 1:
            print(f"        The spread on the UPPER plane is the number that "
                  f"decides whether the box\n        layer is flat. Deck "
                  f"{spreads[0]:.1f} mm, upper "
                  f"{max(spreads[1:]):.1f} mm.")

        spatial = {}
        print("\n[resid] residual after correction, by image cell and by depth")
        for n in names:
            ra = residual_analysis(data[n]["depth"], data[n]["K"],
                                   data[n]["E"], masks[n], planes_w,
                                   affine[n]["a"], affine[n]["b"], args)
            if ra is None:
                continue
            grid = ra.pop("_grid")
            spatial[n] = ra
            sp = ra["spatial_structure_mm"]
            tr = ra["depth_trend_mm"]
            print(f"        {n:<8} rms {ra['overall_rms_mm']:6.2f} mm   "
                  f"spatial p05-p95 "
                  f"{sp if sp is not None else float('nan'):6.2f} mm   "
                  f"depth trend {tr if tr is not None else float('nan'):+6.2f} mm")
            if args.residual_maps and cv2 is not None:
                img = draw_residual_map(grid, n, sp)
                if img is not None:
                    cv2.imwrite(str(out_dir / f"residual_map_{n}.png"), img)
        report["residual_analysis"] = spatial
        worst = [v["spatial_structure_mm"] for v in spatial.values()
                 if v.get("spatial_structure_mm") is not None]
        if worst:
            print(f"        The spatial spread reaches {max(worst):.1f} mm. A "
                  f"per-camera scalar cannot\n        remove a bias that "
                  f"varies across the frame; that part needs anchoring at "
                  f"control\n        points, not a better affine.")

        if args.consistency > 0:
            t_c = time.perf_counter()
            n_before = sum(len(v) for v in corrected.values())
            corrected = consistency_filter(corrected, names, args.consistency)
            n_after = sum(len(v) for v in corrected.values())
            print(f"\n[cons ] cross-view consistency at "
                  f"{args.consistency * 1e3:.0f} mm: {n_before} -> {n_after} "
                  f"points in {(time.perf_counter() - t_c) * 1e3:.0f} ms")

        fused = o3d.geometry.PointCloud()
        tinted = o3d.geometry.PointCloud()
        for i, n in enumerate(names):
            rgb = data[n]["rgb"]
            cols = None
            if rgb is not None and rgb.shape[:2] == masks[n].shape:
                cols = rgb[masks[n]]
                if args.consistency > 0:
                    cols = None
            fused += to_o3d(corrected[n], colors=cols)
            tinted += to_o3d(corrected[n], rgb01=PALETTE[i % len(PALETTE)])
        if args.voxel > 0:
            fused = fused.voxel_down_sample(args.voxel)
        o3d.io.write_point_cloud(str(out_dir / "fused_aligned.ply"), fused)
        o3d.io.write_point_cloud(str(out_dir / "fused_aligned_by_camera.ply"),
                                 tinted)
        print(f"\n[write] {out_dir / 'fused_aligned.ply'}  "
              f"{len(fused.points)} points")

        if args.write_tsdf:
            print("[tsdf ] integrating corrected depth maps")
            vol = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=args.tsdf_voxel, sdf_trunc=args.tsdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            for n in names:
                a, b = affine[n]["a"], affine[n]["b"]
                d = (a * data[n]["depth"].astype(np.float64) + b).astype(np.float32)
                d[~masks[n]] = 0.0
                h, w = d.shape
                K = data[n]["K"]
                intr = o3d.camera.PinholeCameraIntrinsic(
                    w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.zeros((h, w, 3), np.uint8)),
                    o3d.geometry.Image(d), depth_scale=1.0,
                    depth_trunc=float(np.nanmax(d)) + 1.0,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd, intr, data[n]["E"])
            mesh = vol.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(out_dir / "fused_tsdf.ply"), mesh)

    report["runtime_s"] = round(time.perf_counter() - t_start, 3)
    (out_dir / "layer_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nreport: {out_dir / 'layer_report.json'}")
    if not anchor["ok"]:
        print("The absolute block is NaN, so --apply-absolute downstream will "
              "refuse rather than\nproduce a wrong dimension. That is the "
              "intended behaviour, not a failure of this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())