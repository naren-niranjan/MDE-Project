#!/usr/bin/env python3
"""
scan_align.py

Fit the per-view depth correction against a reference SCAN instead of a
ChArUco board, and write depth_correction.json for da3_stream.py to consume.

Why this replaces the board fit
-------------------------------
depth_align.py fits the same model, h_measured = p_v * h_true + q_v, against
board poses. On this rig that fit is badly conditioned for two reasons that a
scan removes outright.

1.  THE LEVER ARM. The board runs are deck, 130 mm and 218 mm, so p is
    estimated over 218 mm of relief with a per-capture drift of 10 to 33 mm
    sitting on top of it. The bootstrap said so: the shared correction carried
    a one-sigma of 14.8 mm at 270 mm, and the leave-one-height-out test missed
    by 16 to 37 mm rms. A scan of the loaded cell covers roughly -810 mm, the
    floor, to +340 mm, the tallest parcel, in a single capture. That is five
    times the lever arm, and it is measured at hundreds of thousands of
    pixels rather than at three heights.

2.  IDENTIFIABILITY WITHIN A CAPTURE. With a board, all four views see one
    height per capture, so p_v is identified only by how each view's deviation
    grows across captures, which is why two of the four cameras had to keep an
    offset-only correction. With a scan every view sees the whole height range
    at once, so p_v is identified inside a single capture and the per-capture
    drift is separable rather than confounded with it.

The consequence measured on this rig: a board-fitted alpha of 1.171 left the
fused cloud reporting heights as 1.105 times true, which is +21 mm on a 220 mm
parcel and +126 mm on the floor.

The model, unchanged
--------------------
Heights are perpendicular to the deck, in each camera's OWN frame:

    h = z * (n_cam . r) - d_cam       r = ((u-cx)/fx, (v-cy)/fy, 1)

The forward model is fitted, h_measured against h_true, because the scan is
the exact quantity and the depth map is the noisy one. Regressing the exact
quantity on the noisy one attenuates the slope, which is the error that turned
a true compression of 0.17 into 0.06 on synthetic data in depth_align.py.

    h_meas = p_v * h_true + q_v + e_f
    p_v = p_bar + a_v   (sum_v a_v = 0)
    q_v = q_bar + b_v   (sum_v b_v = 0)
    sum_f e_f = 0

e_f is the drift shared by all four views inside one capture, which exists
because DA3 solves the four frames jointly. It is absorbed here and never
applied downstream, because nothing at run time reveals which one a live
capture has drawn. It is reported as an absolute height uncertainty instead.

The correction written is the inverse, alpha_v = 1/p_v, beta_v = -q_v/p_v,
together with the height range it was fitted over. da3_stream.py holds the
correction constant outside that range rather than extrapolating it.

Inputs
------
Captures written by da3_stream.py holding depth_raw_<cam>.npy, K_<cam>.npy and
E_<cam>.npy, plus a reference scan and the frozen scan_to_rig.json from
scan_register.py. The RAW depth is required: fitting against corrected depth
would compound two corrections.

Example
-------
    python scan_align.py \
        --capture runs/live_.../frame_00000 \
        --capture runs/live_.../frame_00001 \
        --capture runs/live_.../frame_00002 \
        --scan scans/cell.ply --transform scan_to_rig.json \
        --calib-dir /home/jetson/Projects/Calibration_4_5/results \
        --out depth_correction.json

Keep this file beside scan_frame.py and da3_stream.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

import scan_frame as sf


# --------------------------------------------------------------------------
# rendering the scan into a camera
# --------------------------------------------------------------------------

def _zbuf(scan_cam, K, shape, front_band, min_points):
    """One z-buffer pass: nearest surface per pixel, then the median of the
    points within front_band of it so a single stray point cannot set it."""
    h, w = shape
    z = scan_cam[:, 2]
    ok = z > 1e-6
    P = scan_cam[ok]
    z = z[ok]
    iu = np.round(K[0, 0] * P[:, 0] / z + K[0, 2]).astype(np.int64)
    iv = np.round(K[1, 1] * P[:, 1] / z + K[1, 2]).astype(np.int64)
    inb = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    iu, iv, z = iu[inb], iv[inb], z[inb]
    flat = iv * w + iu

    order = np.lexsort((z, flat))
    flat, z = flat[order], z[order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    ends = np.r_[starts[1:], len(flat)]

    out = np.full(h * w, np.nan)
    for a, b in zip(starts, ends):
        if b - a < min_points:
            continue
        zz = z[a:b]
        out[flat[a]] = np.median(zz[zz <= zz[0] + front_band])
    return out.reshape(h, w)


def render_scan(scan_cam, K, shape, front_band=0.015, min_points=1,
                levels=2):
    """Depth the scan predicts at every pixel of one camera's depth grid.

    Rendered at the full grid, then holes filled from progressively coarser
    renders of the same points.

    The pyramid is not a refinement, it is what makes the tool work at all
    above process_res 504. A two-million-point scan spread over an 840 by 1008
    grid leaves only a few points in each pixel, so demanding three of them
    returns five per cent coverage; every other pixel comes back NaN and the
    reference is a sieve. Halving the resolution quadruples the points per
    pixel, and a pixel filled from the level above is a genuine measurement of
    the same surface at coarser sampling, not an interpolation between
    measurements.

    NaN remains where no level covered the pixel. Those carry no reference and
    must not be guessed at.
    """
    out = _zbuf(scan_cam, K, shape, front_band, min_points)
    h, w = shape
    for lvl in range(1, max(0, levels) + 1):
        holes = ~np.isfinite(out)
        if not holes.any():
            break
        f = 2 ** lvl
        Ks = np.array(K, dtype=float, copy=True)
        Ks[0, 0] /= f
        Ks[1, 1] /= f
        Ks[0, 2] = (K[0, 2] + 0.5) / f - 0.5
        Ks[1, 2] = (K[1, 2] + 0.5) / f - 0.5
        coarse = _zbuf(scan_cam, Ks, (max(2, h // f), max(2, w // f)),
                       front_band, min_points)
        up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_NEAREST)
        out = np.where(holes & np.isfinite(up), up, out)
    return out, np.isfinite(out)


def edge_reject(z, rel_thresh=0.010, dilate=2):
    """Pixels far enough from a depth discontinuity to be a fair comparison.

    A depth model does not reproduce a step, it smooths across it. Comparing
    those pixels against a scan that resolves the step measures the smoothing
    rather than the relief error the fit is after, and a handful of them at a
    300 mm step will dominate any least squares that admits them.

    The step is measured as the range over a 3 by 3 window of the FINITE
    neighbours only, and it is the steep mask that is then grown, not the valid
    mask. Growing the valid mask instead is wrong on a rendered reference,
    which is full of small holes wherever the scan did not cover a pixel: every
    hole becomes a discontinuity, the erosion eats outward from all of them at
    once, and a five per cent perforated map comes back completely empty with
    nothing to say which stage did it.
    """
    d = np.asarray(z, dtype=float)
    finite = np.isfinite(d) & (d > 0)
    if not finite.any():
        return finite
    k3 = np.ones((3, 3), np.uint8)
    lo = np.where(finite, d, 1e6).astype(np.float32)
    hi = np.where(finite, d, -1e6).astype(np.float32)
    rng = cv2.dilate(hi, k3) - cv2.erode(lo, k3)
    steep = finite & (rng > 0) & (rng / np.maximum(d, 1e-6) > rel_thresh)
    if dilate > 0:
        steep = cv2.dilate(steep.astype(np.uint8),
                           np.ones((2 * dilate + 1,) * 2, np.uint8)).astype(bool)
    return finite & ~steep


def incidence_ok(z_true, K, max_deg):
    """Drop pixels where the reference surface is seen edge on.

    A grazing sample carries the largest depth error for the smallest
    reprojection error, and it is also where the scan and the depth map are
    least likely to be looking at the same physical point.
    """
    pts = sf.pixel_rays(z_true.shape, K) * np.nan_to_num(z_true, nan=0.0)[..., None]
    du = np.zeros_like(pts)
    dv = np.zeros_like(pts)
    du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dv[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(du, dv)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    r = pts / np.maximum(np.linalg.norm(pts, axis=-1, keepdims=True), 1e-12)
    cos = np.abs(np.sum(n * r, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    ang[~np.isfinite(ang)] = 90.0
    return ang <= max_deg


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

def irls_line(x, y, huber=2.0, iters=6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x)
    p = q = 0.0
    for _ in range(max(1, iters if huber else 1)):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        p, q = float(coef[0]), float(coef[1])
        if not huber:
            break
        r = np.abs(p * x + q - y)
        s = max(1.4826 * float(np.median(r)), 1e-12)
        w = np.where(r <= huber * s, 1.0, huber * s / np.maximum(r, 1e-12))
    return p, q


def lstsq_constrained(A, y, C, big=1e5):
    A2 = np.vstack([A, big * np.asarray(C, dtype=float)])
    y2 = np.concatenate([y, np.zeros(len(C))])
    sol, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    return sol


def solve_effects(rows, cams, slope_cams, huber=2.0):
    """Shared relief loss, then per-view deviations with a per-capture offset.

    Gauge: the per-view slopes sum to zero over the cameras that keep one, the
    per-view offsets sum to zero, and the capture offsets sum to zero. Without
    those three constraints the second stage is rank deficient by exactly
    three and the split between shared and per-view is arbitrary.
    """
    x = np.array([r["h_true"] for r in rows], dtype=float)
    y = np.array([r["h_meas"] for r in rows], dtype=float)
    w = np.array([r.get("w", 1.0) for r in rows], dtype=float)

    p_bar, q_bar = irls_line(x, y, huber)
    resid = y - (p_bar * x + q_bar)

    keys = sorted({r["key"] for r in rows})
    kidx = {k: i for i, k in enumerate(keys)}
    cidx = {c: i for i, c in enumerate(cams)}
    sidx = {c: i for i, c in enumerate(slope_cams)}
    nS, nC, nF = len(slope_cams), len(cams), len(keys)

    A = np.zeros((len(rows), nS + nC + nF))
    for i, r in enumerate(rows):
        if r["camera"] in sidx:
            A[i, sidx[r["camera"]]] = x[i]
        A[i, nS + cidx[r["camera"]]] = 1.0
        A[i, nS + nC + kidx[r["key"]]] = 1.0
    A *= w[:, None]
    rhs = resid * w

    C = []
    if nS >= 2:
        row = np.zeros(A.shape[1]); row[:nS] = 1.0; C.append(row)
    row = np.zeros(A.shape[1]); row[nS:nS + nC] = 1.0; C.append(row)
    row = np.zeros(A.shape[1]); row[nS + nC:] = 1.0; C.append(row)

    sol = lstsq_constrained(A, rhs, np.stack(C))
    a = {c: 0.0 for c in cams}
    for c, i in sidx.items():
        a[c] = float(sol[i])
    b = {c: float(sol[nS + cidx[c]]) for c in cams}
    e = {k: float(sol[nS + nC + kidx[k]]) for k in keys}
    return {"p_bar": p_bar, "q_bar": q_bar, "a": a, "b": b, "e": e}


def bootstrap(rows, cams, slope_cams, huber, draws, seed=0):
    """Cluster bootstrap over captures when there are several, over height
    bins when there is only one.

    Four views of one capture share the drift that dominates this rig, so they
    are one cluster and resampling them independently would understate every
    sigma by about a factor of two. With a single capture there is no drift to
    resample and the bootstrap measures fit precision only, which is stated
    rather than passed off as the full uncertainty.
    """
    rng = np.random.default_rng(seed)
    keys = sorted({r["key"] for r in rows})
    by_key = {k: [] for k in keys}
    for r in rows:
        by_key[r["key"]].append(r)
    single = len(keys) < 2

    out = {"p_bar": [], "q_bar": [],
           "a": {c: [] for c in cams}, "b": {c: [] for c in cams}}
    for _ in range(draws):
        if single:
            sub = [dict(r) for r in
                   np.asarray(rows, dtype=object)[
                       rng.choice(len(rows), len(rows), replace=True)]]
        else:
            sub = []
            for j, p in enumerate(rng.choice(len(keys), len(keys), replace=True)):
                for r in by_key[keys[p]]:
                    q = dict(r)
                    q["key"] = (r["key"], j)
                    sub.append(q)
        counts = {c: 0 for c in cams}
        for r in sub:
            counts[r["camera"]] += 1
        if min(counts.values()) < 4:
            continue
        if len({round(r["h_true"], 3) for r in sub}) < 3:
            continue
        try:
            fit = solve_effects(sub, cams, slope_cams, huber)
        except np.linalg.LinAlgError:
            continue
        out["p_bar"].append(fit["p_bar"])
        out["q_bar"].append(fit["q_bar"])
        for c in cams:
            out["a"][c].append(fit["a"][c])
            out["b"][c].append(fit["b"][c])
    out["single_capture"] = single
    return out


def sd(v):
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 2 else float("nan")


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def deck_in_camera(n_world, d_world, E):
    """The deck plane expressed in one camera's own frame.

    n_w . x_w = d_w and x_cam = R x_w + t give n_c = R n_w and
    d_c = d_w + n_c . t. The normal is then oriented back towards the camera,
    so d_c is negative and n_c . r is about -1, which is the sign convention
    every height below assumes.
    """
    E = np.asarray(E, dtype=float)
    n_c = E[:3, :3] @ np.asarray(n_world, dtype=float)
    d_c = float(d_world) + float(n_c @ E[:3, 3])
    if d_c > 0:
        n_c, d_c = -n_c, -d_c
    return n_c, d_c


def collect(args, T_scan, scans):
    """One row per camera per capture per height bin.

    Binning matters. The belt covers most of the frame and the parcel tops
    cover little of it, so an unbinned least squares would be a measurement of
    the belt with the relief range thrown in as a rounding error. Equal weight
    per height bin makes the fit a measurement of the slope, which is the
    quantity that was wrong.

    scans is one cloud per capture. Several captures of ONE static scene share
    a scan and are the arrangement that measures the drift; captures of
    different arrangements need one scan each, and pooling them widens the
    range of positions the fit sees.
    """
    worlds = [sf.apply_T(T_scan, s) for s in scans]
    n_w, d_w = sf.deck_plane(sf.voxel(worlds[0], 0.006), args.deck_rig)
    print(f"[deck ] world deck plane n={np.round(n_w, 5)} d={d_w:.4f} m "
          f"(from the scan, so it is metrology rather than DA3)")

    lo, hi = args.range
    edges = np.arange(lo, hi + args.bin_m, args.bin_m)
    rows, per_view_planes, coverage = [], {}, {}

    for ci, cap in enumerate(args.capture):
        arrays = sf.load_capture_depths(cap, args.cameras, raw=True)
        scan_world = worlds[ci]
        key = Path(cap).name if len(args.capture) > 1 else "single"
        for name, a in arrays.items():
            K, E, z_meas = a["K"], a["E"], a["depth"]
            n_c, d_c = deck_in_camera(n_w, d_w, E)
            per_view_planes[name] = (n_c, d_c)

            scan_cam = np.asarray(scan_world, dtype=float) @ E[:3, :3].T + E[:3, 3]
            z_true, cnt = render_scan(scan_cam, K, z_meas.shape,
                                      args.front_band, args.min_scan_points,
                                      args.render_levels)

            # Every stage is counted. A filter chain that ends in zero is
            # useless to debug from the total alone, and this one runs behind
            # a registration, a render and a plane fit, any of which could be
            # the cause.
            stage = []
            ok = np.isfinite(z_meas) & (z_meas > 0)
            stage.append(("depth", int(ok.sum())))
            ok &= cnt
            stage.append(("scan covers", int(ok.sum())))
            ok &= edge_reject(z_true, args.edge_thresh, args.edge_dilate)
            stage.append(("scan edges", int(ok.sum())))
            ok &= edge_reject(z_meas, args.edge_thresh, args.edge_dilate)
            stage.append(("depth edges", int(ok.sum())))
            ok &= incidence_ok(z_true, K, args.max_incidence)
            stage.append(("incidence", int(ok.sum())))
            ok &= np.abs(z_meas - z_true) <= args.max_residual_m
            stage.append(("agree", int(ok.sum())))

            nr = sf.pixel_rays(z_meas.shape, K) @ n_c
            h_true = z_true * nr - d_c
            h_meas = z_meas * nr - d_c
            ok &= np.isfinite(h_true) & np.isfinite(h_meas)
            ok &= (h_true >= lo) & (h_true <= hi)
            stage.append(("in range", int(ok.sum())))

            ht = h_true[ok]
            hm = h_meas[ok]
            coverage[(key, name)] = int(ok.sum())
            if args.verbose or len(ht) < 500:
                print(f"[pix  ] {key} {name:<7} "
                      + "  ".join(f"{k} {v}" for k, v in stage))
            if len(ht) < 500:
                print(f"[skip ] {key} {name}: only {len(ht)} usable pixels. "
                      f"The stage where the count collapses is the one to "
                      f"change.")
                continue

            which = np.digitize(ht, edges) - 1
            for b in range(len(edges) - 1):
                m = which == b
                if int(m.sum()) < args.min_bin_points:
                    continue
                rows.append({"camera": name, "capture": key,
                             "key": key, "bin": b,
                             "h_true": float(np.median(ht[m])),
                             "h_meas": float(np.median(hm[m])),
                             "n": int(m.sum()), "w": 1.0})
    return rows, n_w, d_w, per_view_planes, coverage


# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Fit a per-view height-space depth correction against a "
                    "reference scan and write depth_correction.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture", action="append", type=Path, required=True,
                    metavar="DIR",
                    help="a frame_NNNNN directory holding depth_raw_<cam>.npy, "
                         "K_<cam>.npy and E_<cam>.npy. Repeat it: one capture "
                         "fits the correction, three or more also measure the "
                         "drift that no correction can remove")
    ap.add_argument("--scan", action="append", type=Path, required=True,
                    help="the reference scan. Give it once when every capture "
                         "is of the same static scene, which is the "
                         "arrangement that also measures the drift, or once "
                         "per capture in the same order when the parcels were "
                         "rearranged between them")
    ap.add_argument("--transform", type=Path, default=Path("scan_to_rig.json"),
                    help="the frozen transform from scan_register.py")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="only for the lens tag written into the output, so a "
                         "correction solved on the 12 mm rig cannot be applied "
                         "to the 8 mm one")
    ap.add_argument("--lens", default="",
                    help="lens tag, if --calib-dir is not available")
    ap.add_argument("--deck-rig", type=float, default=None,
                    help="rough reference-camera-to-deck distance, m, used to "
                         "seed the deck plane fit in the world frame")
    ap.add_argument("--out", type=Path, default=Path("depth_correction.json"))

    ap.add_argument("--range", nargs=2, type=float, default=[-0.06, 0.45],
                    metavar=("LO_M", "HI_M"),
                    help="height range about the deck the correction is fitted "
                         "over, and the range it is held constant outside of. "
                         "Cover the deck and the tallest parcel and stop "
                         "there. Reaching down to the floor looks like free "
                         "lever arm and is not: the floor lies outside the "
                         "region the registration aligned, DA3 reconstructs it "
                         "several degrees out of parallel with the deck, and "
                         "that is a spatially structured error which a map on "
                         "height alone cannot express. Measured on this rig, "
                         "including it spent the per-view slopes on the floor "
                         "and made the layering on the parcels three times "
                         "worse")
    ap.add_argument("--bin-m", type=float, default=0.010,
                    help="height bin width. Each bin contributes one row per "
                         "view per capture, so the belt does not outvote the "
                         "parcel tops")
    ap.add_argument("--min-bin-points", type=int, default=150)
    ap.add_argument("--front-band", type=float, default=0.015,
                    help="depth window behind the nearest scan point taken as "
                         "the front surface of a pixel")
    ap.add_argument("--min-scan-points", type=int, default=1,
                    help="scan points a pixel needs before it carries a "
                         "reference depth. Above 1 the render starves at high "
                         "--process-res: a 2 M point scan over an 840 by 1008 "
                         "grid gives only a few points per pixel")
    ap.add_argument("--render-levels", type=int, default=2,
                    help="coarser render passes used to fill the holes the "
                         "full-resolution one leaves")
    ap.add_argument("--edge-thresh", type=float, default=0.010,
                    help="relative depth gradient above which a pixel is "
                         "treated as a discontinuity in either cloud")
    ap.add_argument("--edge-dilate", type=int, default=3,
                    help="how far the discontinuity rejection is grown, in "
                         "depth-grid pixels. Generous on purpose: the fit "
                         "wants flat interior only")
    ap.add_argument("--max-incidence", type=float, default=65.0)
    ap.add_argument("--max-residual-m", type=float, default=0.35,
                    help="absolute reject before the robust fit; a pixel this "
                         "far from the scan is not looking at the same surface")
    ap.add_argument("--huber", type=float, default=2.0)

    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--predict-height", type=float, default=0.270,
                    help="height at which the correction and its uncertainty "
                         "are reported, for comparison against parcel tops")
    ap.add_argument("--max-view-sigma-mm", type=float, default=5.0,
                    help="refuse to write when a view's delayering correction "
                         "at --predict-height is this uncertain")
    ap.add_argument("--max-view-slope-dev", type=float, default=0.04,
                    help="largest per-view departure from the rig-wide relief "
                         "factor that is still treated as a relief difference. "
                         "Beyond this the view is corrected by an offset "
                         "alone. Real per-view relief differences on this rig "
                         "span about 0.026; a view fitting 0.09 from its "
                         "neighbours is not reconstructing relief differently, "
                         "it carries a structured error that a map on height "
                         "cannot express, and fitting a slope to it spends the "
                         "gauge on that view and drags the others with it")
    ap.add_argument("--slope-sigma-ratio", type=float, default=1.0,
                    help="a view keeps its fitted slope only when the slope's "
                         "contribution at --predict-height exceeds this many "
                         "of its own bootstrap sigmas")
    ap.add_argument("--no-view-slope", dest="view_slope", action="store_false",
                    default=True,
                    help="fit per-view offsets but no per-view slopes")
    ap.add_argument("--shared-only", action="store_true",
                    help="write the rig-wide term alone, with no per-view "
                         "deviations at all. The right choice when the [prof ] "
                         "table shows each view's error is not a function of "
                         "height: a per-view map on height then has nothing to "
                         "grip, and fitting one moves the four sheets apart "
                         "rather than together. The shared relief compression "
                         "is still real, still well determined, and still "
                         "worth removing. The per-view terms are omitted from "
                         "the file rather than zeroed, so nothing downstream "
                         "can apply them by accident")
    ap.add_argument("--verbose", action="store_true",
                    help="print the per-stage pixel survivor counts for every "
                         "camera, not only when one of them collapses")
    ap.add_argument("--force", action="store_true",
                    help="write even when a gate fails")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    lo, hi = float(args.range[0]), float(args.range[1])
    args.range = (lo, hi)
    H = float(args.predict_height)
    if not (lo < 0 < hi):
        raise SystemExit("--range must bracket the deck, that is lo < 0 < hi")
    if H > hi:
        raise SystemExit(f"--predict-height {H} lies outside --range; the "
                         f"correction would be reported where it was never "
                         f"fitted")

    lens = args.lens
    if not lens and args.calib_dir:
        try:
            data = json.loads((args.calib_dir /
                               f"intrinsics_{args.cameras[0]}.json").read_text())
            lens = data.get("lens_id") or ""
        except (OSError, ValueError):
            pass
    if not lens:
        print("[WARN ] no lens tag. da3_stream.py will accept this file for "
              "any optical configuration, including one it was not solved on.")

    T_scan, tmeta = sf.load_transform(args.transform)
    print(f"[frame] scan_to_rig from {args.transform}  registration residual "
          f"{tmeta.get('icp_median_mm', float('nan')):.1f} mm median")
    if len(args.scan) == 1:
        scans = [sf.read_ply(args.scan[0])] * len(args.capture)
        print(f"[load ] one scan, {len(scans[0])} points, reused for all "
              f"{len(args.capture)} capture(s). That is the right arrangement "
              f"only if the scene did not move between them")
    elif len(args.scan) == len(args.capture):
        scans = [sf.read_ply(p) for p in args.scan]
        print(f"[load ] {len(scans)} scans, one per capture")
    else:
        raise SystemExit(f"{len(args.scan)} scans for {len(args.capture)} "
                         f"captures; give one scan, or one per capture in the "
                         f"same order")

    rows, n_w, d_w, planes, coverage = collect(args, T_scan, scans)
    if not rows:
        raise SystemExit("no usable pixels; check --range, the transform, and "
                         "that the captures hold depth_raw arrays")
    cams = [c for c in args.cameras if any(r["camera"] == c for r in rows)]
    caps = sorted({r["capture"] for r in rows})
    print(f"\n[data ] {len(rows)} height bins over {len(cams)} cameras and "
          f"{len(caps)} capture(s)")
    print(f"[data ] {'camera':<8} {'bins':>5} {'h_true span mm':>16} "
          f"{'pixels':>9}")
    for c in cams:
        sub = [r for r in rows if r["camera"] == c]
        ht = [r["h_true"] * 1e3 for r in sub]
        print(f"[data ] {c:<8} {len(sub):>5} {min(ht):>7.0f} to {max(ht):<6.0f} "
              f"{sum(r['n'] for r in sub):>9}")
    hs = np.sort(np.array([r["h_true"] for r in rows]))
    if len(hs) > 10:
        gaps = np.diff(np.unique(np.round(hs, 3)))
        if gaps.size and float(gaps.max()) > 0.20:
            k = int(np.argmax(gaps))
            edge = float(np.unique(np.round(hs, 3))[k])
            print(f"[WARN ] the heights fall into two clusters with "
                  f"{gaps.max() * 1e3:.0f} mm of nothing between them, the gap "
                  f"starting at {edge * 1e3:+.0f} mm. A line fitted across a "
                  f"void is a lever: a small error in the far cluster swings "
                  f"the slope hard, and the near cluster pays for it. Narrow "
                  f"--range to the cluster the parcels are in.")

    if len(caps) < 3:
        print("[note ] fewer than three captures, so the drift shared by all "
              "four views is not measurable here. The correction is still "
              "valid; its absolute offset simply carries this capture's drift, "
              "and the height uncertainty below is a lower bound.")

    # ---- before ---------------------------------------------------------
    def layering(rows, fit=None):
        out = {}
        for b in sorted({r["bin"] for r in rows}):
            for k in sorted({r["key"] for r in rows}):
                grp = [r for r in rows if r["bin"] == b and r["key"] == k]
                if len(grp) < 2:
                    continue
                hb = [r["h_meas"] for r in grp]
                pre = (max(hb) - min(hb)) * 1e3
                post = None
                if fit is not None:
                    ha = [fit["alpha"][r["camera"]] * r["h_meas"]
                          + fit["beta"][r["camera"]] for r in grp]
                    post = (max(ha) - min(ha)) * 1e3
                out.setdefault(round(grp[0]["h_true"], 2), []).append((pre, post))
        return out

    def layering_summary(stats, lo_h=-0.05):
        """Median before and after, split at the deck.

        Split because they are different measurements. Above the deck is the
        working volume, the only place a pick happens and the only place the
        correction has to be right. Below it is whatever else the range
        admitted, and improving that at the expense of the volume above is not
        a better correction, it is a worse one that reports a better number.
        """
        out = {}
        for tag, sel in (("working volume", lambda h: h >= lo_h),
                         ("below the deck", lambda h: h < lo_h)):
            pre = [v[0] for h, vals in stats.items() if sel(h) for v in vals]
            post = [v[1] for h, vals in stats.items() if sel(h)
                    for v in vals if v[1] is not None]
            if pre:
                out[tag] = (float(np.median(pre)),
                            float(np.median(post)) if post else None,
                            len(pre))
        return out

    before = layering(rows)

    huber = args.huber or None
    slope_cams = ([] if args.shared_only
                  else (list(cams) if args.view_slope else []))
    fit = solve_effects(rows, cams, slope_cams, huber)
    p_bar, q_bar = fit["p_bar"], fit["q_bar"]
    if p_bar < 0.2:
        raise SystemExit(f"the fitted surviving fraction of relief is "
                         f"{p_bar:.4f}. Either the registration is wrong or "
                         f"the depth maps are not looking at this scene; open "
                         f"the transformed scan beside the cloud.")
    scale = 1.0 / p_bar
    print(f"\n[shared] p={p_bar:.6f}  q={q_bar * 1e3:+.2f} mm: the rig keeps "
          f"{p_bar * 100:.1f} per cent of every rise, mean compression "
          f"k={1 - p_bar:.4f}")
    print(f"[shared] as a correction, alpha={1 / p_bar:.6f} "
          f"beta={-q_bar / p_bar * 1e3:+.2f} mm; it moves a deck point by "
          f"{-q_bar * 1e3:+.1f} mm and a {H * 1e3:.0f} mm point by "
          f"{(H - (p_bar * H + q_bar)) * 1e3:+.1f} mm")

    gate_failed = []
    boot = (bootstrap(rows, cams, slope_cams, huber, args.bootstrap)
            if args.bootstrap > 0 else None)

    if boot and slope_cams and not args.shared_only:
        print()
        keep = []
        for c in slope_cams:
            s = sd(boot["a"][c]) * H * 1e3 * scale
            v = abs(fit["a"][c]) * H * 1e3 * scale
            dev = abs(fit["a"][c])
            ok = np.isfinite(s) and s > 0 and v >= args.slope_sigma_ratio * s
            why = "" if ok else ", not resolved"
            if ok and dev > args.max_view_slope_dev:
                ok = False
                why = (f", {dev:.3f} from the rig mean, above "
                       f"--max-view-slope-dev {args.max_view_slope_dev:.3f}")
            print(f"[slope ] {c:<8} contributes {v:5.1f} mm at {H * 1e3:.0f} mm "
                  f"against its own sigma {s:5.1f} mm, deviation {dev:.3f}  "
                  + ("kept" if ok else f"dropped, offset only{why}"))
            if ok:
                keep.append(c)
        if len(keep) < 2:
            print("[slope ] fewer than two views resolve a slope, so the gauge "
                  "would force the rest to zero anyway; offset only.")
            keep = []
        if keep != slope_cams:
            slope_cams = keep
            fit = solve_effects(rows, cams, slope_cams, huber)
            boot = bootstrap(rows, cams, slope_cams, huber, args.bootstrap)
            p_bar, q_bar = fit["p_bar"], fit["q_bar"]
            scale = 1.0 / p_bar

    if args.shared_only:
        # The gauge already sums these to zero across views; here they are
        # dropped outright, so every view carries the rig-wide map and the
        # spread between views is left exactly as measured rather than being
        # rearranged by a term that does not describe it.
        fit["a"] = {c: 0.0 for c in cams}
        fit["b"] = {c: 0.0 for c in cams}
    p_v = {c: p_bar + fit["a"][c] for c in cams}
    q_v = {c: q_bar + fit["b"][c] for c in cams}
    bad = [c for c in cams if p_v[c] < 0.2]
    if bad:
        raise SystemExit(f"views {bad} fitted a surviving relief fraction "
                         f"below 0.2, which is a singularity rather than a "
                         f"correction. Re-run with --no-view-slope.")
    fit["alpha"] = {c: 1.0 / p_v[c] for c in cams}
    fit["beta"] = {c: -q_v[c] / p_v[c] for c in cams}

    sig_shared = (float(np.std([H - (p * H + q) for p, q
                                in zip(boot["p_bar"], boot["q_bar"])], ddof=1))
                  * 1e3 if boot and len(boot["p_bar"]) > 2 else None)

    print(f"\n[view  ] per-view deviation from the rig mean, in millimetres of "
          f"reconstructed height. This is the delayering term.")
    print(f"[view  ] {'camera':<8} {'p_v':>9} {'q_v mm':>9} {'at deck':>9} "
          f"{'at ' + str(int(H * 1e3)) + ' mm':>11} {'sigma':>8} {'slope':>7}")
    view_out = {}
    for c in cams:
        d_deck = fit["b"][c] * 1e3 * scale
        d_high = (fit["a"][c] * H + fit["b"][c]) * 1e3 * scale
        s_high = None
        if boot and len(boot["b"][c]) > 2:
            s_high = float(np.std([a * H + b for a, b
                                   in zip(boot["a"][c], boot["b"][c])],
                                  ddof=1)) * 1e3 * scale
        n_c, d_c = planes[c]
        view_out[c] = {
            "p_v": p_v[c], "q_v": q_v[c],
            "alpha": fit["alpha"][c], "beta": fit["beta"][c],
            "a_v": fit["a"][c], "b_v": fit["b"][c],
            "slope_fitted": c in slope_cams,
            "n_cam": n_c.tolist(), "d_cam": float(d_c),
            "delayering_at_deck_mm": round(d_deck, 3),
            "delayering_at_predict_height_mm": round(d_high, 3),
            "delayering_sigma_mm": (None if s_high is None
                                    else round(s_high, 3)),
            "n_bins": sum(1 for r in rows if r["camera"] == c),
        }
        print(f"[view  ] {c:<8} {p_v[c]:>9.5f} {q_v[c] * 1e3:>+8.2f} "
              f"{d_deck:>+8.2f} {d_high:>+10.2f} "
              + (f"{s_high:>7.2f}" if s_high is not None else f"{'n/a':>7}")
              + f" {'yes' if c in slope_cams else 'no':>7}")
        if (s_high is not None and s_high > args.max_view_sigma_mm
                and not args.shared_only):
            gate_failed.append(f"{c}: delayering sigma {s_high:.2f} mm at "
                               f"{H * 1e3:.0f} mm, above "
                               f"--max-view-sigma-mm {args.max_view_sigma_mm}")

    # ---- what it does ---------------------------------------------------
    after = layering(rows, fit)
    print(f"\n[layer ] spread across views on the same surface, the quantity a "
          f"per-view correction can move")
    print(f"[layer ] {'height mm':>10} {'n':>4} {'before':>10} {'after':>10}")
    for h_, vals in sorted(after.items()):
        pre = np.median([v[0] for v in vals])
        post = np.median([v[1] for v in vals if v[1] is not None])
        print(f"[layer ] {h_ * 1e3:>10.0f} {len(vals):>4} {pre:>7.1f} mm "
              f"{post:>7.1f} mm")

    summ = layering_summary(after)
    print()
    for tag, (pre, post, n) in summ.items():
        print(f"[layer ] {tag:<15} {n:>4} bins   before {pre:6.1f} mm   "
              f"after {post:6.1f} mm")

    work = summ.get("working volume")
    if args.shared_only:
        print("[layer ] --shared-only: every view gets the same map, so the "
              "spread between them is unchanged by construction. It is left "
              "as measured because nothing here describes it.")
    elif work and work[1] is not None and work[1] > 0.8 * work[0]:
        gate_failed.append(
            f"the layering in the working volume went from {work[0]:.1f} mm "
            f"to {work[1]:.1f} mm. A per-view correction that does not reduce "
            f"the spread between views is not correcting a per-view error")

    # Whether each view's error is a function of height at all. A view whose
    # error grows smoothly with height is carrying a relief difference, which
    # this model can remove. A view whose error jumps about is carrying
    # something structured, which it cannot, and fitting a slope to it is how
    # one bad camera drags the other three off through the gauge.
    bands = [(-0.06, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.35),
             (0.35, 0.45)]
    print(f"\n[prof  ] median height error per view against height, before the "
          f"fit, in mm. Smooth means a relief difference; ragged means "
          f"something this model cannot remove")
    print(f"[prof  ] {'camera':<8}" + "".join(
        f"{int(a * 1e3):>5}..{int(b * 1e3):<5}" for a, b in bands))
    for c in cams:
        cells = []
        for a, b in bands:
            sub = [(r["h_meas"] - r["h_true"]) * 1e3 for r in rows
                   if r["camera"] == c and a <= r["h_true"] < b]
            cells.append(f"{np.median(sub):>9.1f} " if len(sub) >= 2
                         else f"{'-':>9} ")
        print(f"[prof  ] {c:<8}" + "".join(cells))

    print(f"\n[resid ] height error against the scan, before and after, in mm")
    print(f"[resid ] {'camera':<8} {'before rms':>11} {'after rms':>11} "
          f"{'after max':>11}")
    resid_out = {}
    for c in cams:
        sub = [r for r in rows if r["camera"] == c]
        b0 = np.array([(r["h_meas"] - r["h_true"]) * 1e3 for r in sub])
        a1 = np.array([(fit["alpha"][c] * r["h_meas"] + fit["beta"][c]
                        - r["h_true"]) * 1e3 for r in sub])
        resid_out[c] = {"before_rms_mm": round(float(np.sqrt((b0 ** 2).mean())), 2),
                        "after_rms_mm": round(float(np.sqrt((a1 ** 2).mean())), 2),
                        "after_max_mm": round(float(np.abs(a1).max()), 2)}
        print(f"[resid ] {c:<8} {np.sqrt((b0 ** 2).mean()):>8.2f} mm "
              f"{np.sqrt((a1 ** 2).mean()):>8.2f} mm "
              f"{np.abs(a1).max():>8.2f} mm")

    # ---- the part no correction removes ---------------------------------
    uncertainty = None
    if len(caps) >= 3:
        drift = np.array(list(fit["e"].values())) * 1e3
        uncertainty = {
            "model": "constant, measured as the capture-to-capture drift",
            "capture_drift_sd_mm": round(float(drift.std(ddof=1)), 2),
            "n_captures": len(caps),
            "at_predict_height_mm": round(float(drift.std(ddof=1)), 2),
            "shared_correction_sigma_mm": (None if sig_shared is None
                                           else round(sig_shared, 2)),
            "note": ("one sigma on a reconstructed absolute height. It is the "
                     "term shared across all four views inside one capture, so "
                     "it is NOT reduced by view consensus and must not be "
                     "estimated from inter-view agreement."),
        }
        print(f"\n[unc   ] capture-to-capture drift {drift.std(ddof=1):.1f} mm "
              f"one sigma, over {len(caps)} captures. Carry this into "
              f"height_uncertainty_mm downstream.")
    else:
        print("\n[unc   ] absolute height uncertainty not measured: it needs "
              "three or more captures of the same scene. Add them and re-run; "
              "the correction itself will barely move.")

    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lens_id": lens,
        "model": "deck_height_linear",
        "source": "reference_scan",
        "formula": ("h = z * (n_cam . r) - d_cam ; h' = alpha * h + beta ; "
                    "z' = (h' + d_cam) / (n_cam . r)"),
        "reference": f"{len(args.scan)} scan(s) via {args.transform.name}",
        "valid_range_m": [lo, hi],
        "range_note": ("the correction was fitted only inside valid_range_m. "
                       "Outside it da3_stream.py holds the correction at its "
                       "endpoint and passes the excess through unchanged, "
                       "because an affine map extrapolated to the floor "
                       "fabricated 126 mm on this rig"),
        "captures": [str(c) for c in args.capture],
        "scans": [str(s) for s in args.scan],
        "scan_to_rig": {"file": str(args.transform), **tmeta},
        "world_deck": {"normal": n_w.tolist(), "offset_m": float(d_w)},
        "predict_height_m": H,
        "shared": {"p": p_bar, "q": q_bar,
                   "alpha": 1.0 / p_bar, "beta": -q_bar / p_bar,
                   "compression": 1.0 - p_bar,
                   "sigma_at_predict_height_mm": (None if sig_shared is None
                                                  else round(sig_shared, 3))},
        "per_view_terms": not args.shared_only,
        "views": view_out,
        "residuals": resid_out,
        "layering": {str(round(k * 1e3)): {
            "before_median_mm": round(float(np.median([v[0] for v in vals])), 2),
            "after_median_mm": round(float(np.median(
                [v[1] for v in vals if v[1] is not None])), 2)}
            for k, vals in sorted(after.items())},
        "capture_effects_mm": {str(k): round(v * 1e3, 2)
                               for k, v in fit["e"].items()},
        "height_uncertainty": uncertainty,
        "bootstrap_draws": (len(boot["p_bar"]) if boot else 0),
        "bootstrap_over": ("height bins, single capture" if
                           (boot and boot.get("single_capture"))
                           else "captures"),
    }

    if gate_failed:
        print()
        for g in gate_failed:
            print(f"[GATE  ] {g}")
    if gate_failed and not args.force:
        raise SystemExit(
            "refusing to write. An unresolved per-view term moves the four "
            "sheets apart rather than together. Add captures, which shrinks "
            "these sigmas as the square root, or re-run with --no-view-slope "
            "to keep the offsets only, or --force.")

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.out}")
    print(f"apply it with:  python da3_stream.py --correction {args.out} ...")
    print("then grade it with gt_compare.py; the gain it reports should come "
          "back to 1.00 within a few thousandths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())