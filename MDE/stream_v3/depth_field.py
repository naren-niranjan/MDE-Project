#!/usr/bin/env python3
"""
depth_field.py

A four-degree-of-freedom depth correction per camera, fitted against EVERY
measured reference plane, replacing both of the corrections that came before it.

WHY THIS EXISTS
===============
Two scripts already correct per-camera depth and each is missing half the model.

  layer_align.py    Z' = a*Z + b.  Has the ADDITIVE term and the metric anchor.
                    Has no tilt term, so it cannot touch a plane orientation
                    error. Measured on this rig: after correction the deck
                    agreed to 2.2 mm and a surface 218 mm above it to 12.3 mm,
                    with one camera still showing 4.2 deg of apparent tilt and
                    45.8 mm of spatial structure.

  layer_collapse.py d' = d*(s + a*x~ + b*y~).  Has the TILT terms, and they
                    work: measured tilt fell from 1.20-1.60 deg to 0.24-0.34
                    deg on all four cameras, and the deck rms from 17-78 mm to
                    3.1-4.2 mm. It has no additive term and it anchors to ONE
                    plane, which is the deck, so the correction serving the
                    parcel tops is extrapolated from the surface below them.

Neither model alone fits the measured planes. On capture_00001, DA3 read the
deck at 3.1435 m and a riser at 2.9247 m against ChArUco measurements of
3.07527 and 2.85715:

    additive only        c = -68.2 mm      residuals +0.6, -0.6 mm
    multiplicative only  s =  0.9788       residuals +1.5, +5.4 mm, and the
                                           218 mm riser height reads 214 mm

So the radial part of DA3's error over this interval is closer to a constant
offset than to a gain, while the part that varies across the frame is a tilt.
A model with both, and only both, can remove them together:

    d'(u,v) = d(u,v) * ( s + a*x~ + b*y~ ) + c

    x~ = (u - cx)/fx,   y~ = (v - cy)/fy

WHY IT IS STILL A LINEAR LEAST SQUARES
--------------------------------------
Scaling depth by g and adding c maps the back-projected point X = ray*d to

    X' = ray*(d*g + c) = g*X + c*ray

so the perpendicular residual against a plane n.X = offset is

    n.X' - offset = s*q + a*x~*q + b*y~*q + c*r - offset

with q = n.X and r = n.ray. Linear in (s, a, b, c): closed form, no iteration
inside the solve. Only the ASSIGNMENT of points to planes is iterated, with a
narrowing window, for the same reason layer_align iterates it.

FIT THE RAW DEPTH, NOT THE CORRECTED DEPTH
==========================================
da3_stream.py writes depth_{cam}.npy AFTER its correction chain and
depth_raw_{cam}.npy before it. This script now reads the raw array by default,
and refuses to solve on the corrected one, because the two DO NOT COMPOSE.

At stream time --load-field applies the field to RAW depth and runs the online
aligner afterwards. A field fitted against corrected depth is therefore wrong by
exactly whatever the aligner happened to be doing on the frame that was
captured, which is a per-frame quantity recorded nowhere per pixel.

Measured. Run live_20260818_095143 streamed on auto-align alone and finished at
left +5.2, center +0.0, right +25.0, top +34.5 mm. Those offsets sat inside
depth_{cam}.npy for capture_00002. The field solved from that array was applied
to raw depth in run live_20260818_095612, whose online tracker then converged to
left -4.2, center +0.0, right +3.6, top +30.6 mm: it spent the whole run
clawing back the 34.5 mm baked into top. Inter-view spread on parcel tops rose
from 15.2 mm without the field to 22.4 mm with it, and the c values collapsed
from a genuine +177/+67/+28/+74 mm to a meaningless -8/-12/-8/-12 mm.

The field itself was sound. Its holdout, fitted on the deck alone, predicted the
risers to within 1 to 9 mm. It was simply describing a depth map that no longer
existed by the time it was applied. See --depth-source.

WHAT CHANGES ARCHITECTURALLY
----------------------------
Every camera is fitted to PHYSICALLY MEASURED planes, including the reference
camera. There is no reference camera whose scale everyone else inherits, and
there is no separate metric anchor to solve afterwards: the absolute scale
comes from the same fit, because the targets are ChArUco measurements rather
than another camera's opinion. layer_align's two-stage structure, relative
alignment then a metric anchor over two planes, collapses into one stage.

That matters for the numbers, not just the tidiness. The per-camera absolute
deck error on capture_00001 spans 83 mm across the rig, from -15 mm on left to
+68 mm on center. Correcting three cameras onto the fourth and then shifting
all four by one anchor cannot remove a spread that is per-camera in origin.

THE ONE NUMBER TO READ
----------------------
--holdout fits on the deck alone and evaluates on the riser. Every other
residual here is measured on the planes the coefficients were fitted to, where
it is small by construction. The holdout residual is the only figure that says
whether a correction solved below the parcels serves the parcels, which is the
question the whole pipeline turns on.

WHAT THIS DOES NOT FIX
----------------------
A plane constrains three degrees of freedom: the standoff and two tilt axes. It
is blind to in-plane translation and to yaw, because a plane is invariant under
both. A clean residual here does NOT mean the cameras are registered. It means
the depth field is right. If parcels still fail to coincide after this, the
remainder is in-plane and belongs to the extrinsics, not to depth.

Nor does a radial field reach a bias that varies with the CONTENT of the frame
rather than with position in it. The field is smooth and low order by
construction. Whatever survives it is the residual that needs anchoring at
control points, and the spatial_structure figure reported per camera is the
measurement of how much that is. Measured after a good field on this rig:
15.6, 18.2, 20.7, 15.1 mm across the four cameras, which is comparable to the
inter-view parcel-top spread the field is trying to remove. That is the floor.

Example
-------
  python depth_field.py \\
      --capture-dir runs/live_20260817_144844/capture_00001 \\
      --gt ground_truth.json --holdout \\
      --out-dir runs/live_20260817_144844/capture_00001/field
"""

from __future__ import annotations

import argparse
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
    from da3_fuse import (points_camera, pixel_rays, edge_mask, incidence_mask,
                          to_world, to_o3d, PALETTE)
except ImportError as exc:
    raise SystemExit(f"cannot import da3_fuse.py from the working directory: {exc}")


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Four-DOF per-camera depth field fitted to measured planes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("ground_truth.json"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <capture-dir>/field")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center",
                    help="the camera the ChArUco planes were measured from. "
                         "Used ONLY to place those planes in the world; it "
                         "receives a correction like every other camera")

    ap.add_argument("--depth-source", choices=["auto", "raw", "corrected"],
                    default="auto",
                    help="WHICH DEPTH ARRAY TO FIT. da3_stream.py writes "
                         "depth_{cam}.npy AFTER its correction chain and "
                         "depth_raw_{cam}.npy before it, and --load-field "
                         "applies the field to RAW depth at stream time, so a "
                         "field fitted against the corrected array does not "
                         "compose: each camera comes out short by whatever the "
                         "online aligner was doing on the captured frame. "
                         "'auto' uses depth_raw when present and refuses "
                         "depth_{cam}.npy when capture.json records a "
                         "correction. 'raw' requires depth_raw. 'corrected' "
                         "forces the corrected array and is for DIAGNOSIS "
                         "ONLY, since the coefficients will not compose")

    ap.add_argument("--dof", type=int, choices=[3, 4], default=4,
                    help="4 solves s, a, b and c. 3 drops the additive term "
                         "and reproduces layer_collapse.py's model, for a "
                         "controlled comparison")
    ap.add_argument("--fix-scale", action="store_true",
                    help="HOLD s AT 1 and solve only the tilt terms and the "
                         "offset. Measured across all four cameras and all "
                         "three planes, the residual is flat in height to 1-4 "
                         "mm over the 218 mm anchor span (left "
                         "178.2/177.0/176.3, right 27.2/28.1/27.2, center "
                         "69.1/68.0/65.3), which bounds |s-1| below about 1.8 "
                         "per cent. A 218 mm lever arm carrying 6-7 mm of "
                         "noise cannot resolve that, so s is not measured here "
                         "so much as guessed, and the guess costs real "
                         "accuracy outside the span: with the guards opened, "
                         "`top` walked to s=1.058, c=-292 mm over six "
                         "iterations while every anchor residual stayed under "
                         "7 mm, which is 25 mm of error at a parcel top 400 mm "
                         "up. This is the HARD form of --ridge-s. It also "
                         "removes the single-plane degeneracy: a camera that "
                         "can only see the deck still gets a proper offset "
                         "instead of a gain extrapolated from it")
    ap.add_argument("--assign-tol", type=float, default=0.100,
                    help="STARTING half-width for assigning a point to a "
                         "reference plane. MUST EXCEED THE ERROR BEING "
                         "CORRECTED, measured against the PHYSICAL plane and "
                         "not against another camera. On this rig the "
                         "per-camera deck offsets run -15 to +177 mm, so a "
                         "window of 100 mm leaves the worst camera fitting "
                         "PARCEL TOPS instead of the deck: left reads 177 mm "
                         "long, so the surfaces landing near the deck plane in "
                         "its depth map are boxes 78 to 278 mm tall. The upper "
                         "bound is half the gap between the closest pair of "
                         "reference planes, which is why stage one fits one "
                         "plane alone and can afford a wide window. Check the "
                         "assigned counts per plane if a camera comes back "
                         "unsolved")
    ap.add_argument("--assign-tol-final", type=float, default=0.015)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--bootstrap-plane", default="deck",
                    help="THE SOLVE IS TWO-STAGE, and it has to be. Stage one "
                         "fits this plane ALONE with the wide --assign-tol, "
                         "because a single plane cannot be confused with "
                         "another one however wide the window is. Stage two "
                         "then refits every plane starting from that field, "
                         "with the much narrower --assign-tol-stage2, which is "
                         "now affordable because the residual after stage one "
                         "is under about 12 mm. Set empty to fit every plane "
                         "in one stage, which on a three-plane anchor whose "
                         "closest pair is 88 mm apart does not work: see "
                         "--assign-tol-stage2")
    ap.add_argument("--assign-tol-stage2", type=float, default=0.025,
                    help="STARTING window for stage two. MUST STAY BELOW HALF "
                         "THE GAP BETWEEN THE CLOSEST PAIR OF REFERENCE "
                         "PLANES, or a point near one is claimed by both and "
                         "the fit is solved on a mixture. Measured on this "
                         "rig: planes at 0, 130 and 218 mm above the deck, so "
                         "the closest pair is 87 mm and the ceiling is 44 mm. "
                         "This is why one-stage fitting fails here, since the "
                         "uncorrected error reaches 177 mm and no single "
                         "window can be both wider than the error and narrower "
                         "than half the gap")

    ap.add_argument("--max-scale-dev", type=float, default=0.05,
                    help="bound on |s - 1|. The measured values on this rig "
                         "run 0.979 to 1.005, so 5 per cent is generous and "
                         "still excludes a degenerate fit. Inert under "
                         "--fix-scale")
    ap.add_argument("--max-gain-span", type=float, default=0.06,
                    help="bound on the peak-to-peak variation of the gain "
                         "across the frame. Measured: 1.7 to 2.3 per cent. A "
                         "camera whose accepted fit sits exactly ON this "
                         "ceiling with earlier iterations rejected for "
                         "exceeding it is not converging, it is being clipped: "
                         "check its post-correction tilt, which should be "
                         "under a tenth of a degree like its rig-mates")
    ap.add_argument("--max-offset-m", type=float, default=0.30,
                    help="bound on |c|. Measured: up to 177 mm")
    ap.add_argument("--ridge-s", type=float, default=0.02,
                    help="prior standard deviation on (s - 1). The scale and "
                         "the offset are 99.98 per cent correlated over a "
                         "218 mm lever arm, so the data barely constrains the "
                         "direction along which they trade off and the fit "
                         "slides freely along it. Measured consequence: `top` "
                         "solved s=1.0443 with c=-239 mm where the deck-only "
                         "stage gave s=0.9797, c=0. Both fit the reference "
                         "planes; they disagree by 33 mm at 420 mm above the "
                         "deck. A weak prior toward (s=1, c=0) picks the "
                         "smallest member of that degenerate family, which "
                         "BOUNDS THE EXTRAPOLATION rather than finding the "
                         "true split, and it is nearly free inside the span "
                         "because that is the direction the data does "
                         "constrain. Set 0 to disable, or use --fix-scale for "
                         "the hard version")
    ap.add_argument("--ridge-c", type=float, default=0.030,
                    help="prior standard deviation on c, metres. See --ridge-s")
    ap.add_argument("--min-plane-points", type=int, default=500)
    ap.add_argument("--min-samples", type=int, default=2000)

    ap.add_argument("--holdout", action="store_true",
                    help="also fit on the DECK ALONE and evaluate on the "
                         "remaining planes. This is the only residual in the "
                         "report that is not measured on the planes it was "
                         "fitted to")
    ap.add_argument("--residual-cells", type=int, default=8)

    # filters, matching da3_stream.py and layer_align.py exactly. layer_collapse
    # applied neither the edge nor the incidence mask, which is why its point
    # counts and plane fits are not comparable with layer_align's.
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--load-field", type=Path, default=None,
                    help="APPLY a stored depth_field.json instead of solving. "
                         "This is how later captures are corrected: solve once "
                         "on a capture that has the risers standing on the "
                         "belt, then apply those coefficients to every "
                         "subsequent capture. The field is a property of the "
                         "cameras and the lens, not of the scene, so it "
                         "transfers as long as nothing is moved or refocused")
    ap.add_argument("--write-capture", type=Path, default=None,
                    help="write a CORRECTED CAPTURE DIRECTORY: depth_<cam>.npy "
                         "holding the corrected depth, alongside copies of "
                         "K_, E_, conf_ and the images. Every downstream tool "
                         "reads a capture directory, so box_segment.py, "
                         "layer_align.py and layer_collapse.py all run on the "
                         "output unchanged and see anchored metric depth")
    ap.add_argument("--no-write-clouds", dest="write_clouds",
                    action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------

def gt_planes_world(gt_path, E_ref):
    """The ChArUco planes, in world coordinates, as n.X = offset.

    ground_truth.json records each plane's normal and perpendicular distance in
    the REFERENCE CAMERA's frame. The normal is flipped where necessary so that
    it points away from the camera, which makes the offset the positive
    perpendicular distance and removes the sign ambiguity that
    layer_collapse.normalise_gt has to carry.

    x_cam = R x_world + t, so n_cam.x_cam = offset becomes
    (R^T n_cam).x_world = offset - n_cam.t.
    """
    d = json.loads(Path(gt_path).read_text())
    if d.get("usable") is False:
        raise SystemExit(f"{gt_path} is marked unusable: {d.get('problems')}")
    R, t = E_ref[:3, :3], E_ref[:3, 3]
    out = []
    for p in d.get("reference_planes") or []:
        perp = p.get("perp_m")
        n = p.get("normal")
        if not isinstance(perp, (int, float)) or n is None:
            continue
        n_c = np.asarray(n, np.float64)
        n_c = n_c / np.linalg.norm(n_c)
        if n_c[2] < 0:                      # point it away from the camera
            n_c = -n_c
        n_w = R.T @ n_c
        off_w = float(perp) - float(n_c @ t)
        out.append({"label": p.get("label", "?"),
                    "normal": n_w.tolist(),
                    "offset": off_w,
                    "perp_ref_m": float(perp),
                    "height_above_deck_m": p.get("height_above_deck_m"),
                    "flatness_rms_mm": p.get("flatness_rms_mm")})
    if len(out) < 1:
        raise SystemExit(f"{gt_path} carries no usable reference planes")
    # nearest the camera first, deck last
    out.sort(key=lambda e: e["perp_ref_m"])
    return out


def plane_in_camera(plane, E):
    """World plane -> that camera's frame. n_c = R n_w, offset_c = offset + n_c.t"""
    R, t = E[:3, :3], E[:3, 3]
    n_c = R @ np.asarray(plane["normal"], np.float64)
    return n_c, float(plane["offset"]) + float(n_c @ t)


def target_depth(shape, K, n_c, off_c):
    """Depth at which each pixel's ray meets the plane, and where that is real."""
    rays = pixel_rays(shape, K)
    denom = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = off_c / denom
    ok = np.isfinite(z) & (z > 0) & (np.abs(denom) > 1e-6)
    return z, ok, denom


# --------------------------------------------------------------------------
# the field
# --------------------------------------------------------------------------

def gain(shape, K, theta):
    """g(u,v) = s + a*x~ + b*y~ over the whole grid."""
    s, a, b = theta[0], theta[1], theta[2]
    rays = pixel_rays(shape, K)
    return s + a * rays[..., 0] + b * rays[..., 1]


def apply_field(depth, K, theta):
    c = theta[3] if len(theta) > 3 else 0.0
    return depth * gain(depth.shape, K, theta) + c


def solve_field(depth, K, E, valid, planes, args, dof=None,
                tol_start=None, tol_final=None, init_theta=None):
    """Least squares for (s, a, b, c) against every plane at once.

    Rows are [q, x~q, y~q, r] with target offset_c, so the residual minimised is
    the PERPENDICULAR distance to the plane in metres, not the depth difference.
    The two differ by the factor r = n.ray, which is near unity on axis and
    falls away at the field edge; using the perpendicular form keeps the fit
    consistent with how the residual is reported.

    Under --fix-scale the first column moves to the target side, since s is held
    at 1, and the system becomes [x~q, y~q, r] against (offset_c - q). That is
    three unknowns with no degenerate direction among them, so it solves from a
    single plane as readily as from three.
    """
    dof = args.dof if dof is None else dof
    d = depth.astype(np.float64)
    rays = pixel_rays(d.shape, K)
    xt, yt = rays[..., 0], rays[..., 1]

    t0 = args.assign_tol if tol_start is None else tol_start
    t1 = args.assign_tol_final if tol_final is None else tol_final
    tols = np.geomspace(max(t0, 1e-4), max(t1, 1e-4), max(1, args.iters))

    theta = (np.array([1.0, 0.0, 0.0, 0.0]) if init_theta is None
             else np.asarray(init_theta, np.float64).copy())
    cond = {}
    history, per_plane = [], []
    guard_notes = []
    last_good = theta.copy()
    last_per_plane = []

    for it, tol in enumerate(tols):
        rows, targets, per_plane = [], [], []
        dc = apply_field(d, K, theta)
        for pi, plane in enumerate(planes):
            n_c, off_c = plane_in_camera(plane, E)
            z_t, ok, r = target_depth(d.shape, K, n_c, off_c)
            near = valid & ok & (np.abs(dc - z_t) <= tol)
            cnt = int(near.sum())
            per_plane.append({
                "plane": pi, "label": plane["label"], "n_assigned": cnt,
                "target_depth_median_m": (round(float(np.median(z_t[near])), 5)
                                          if cnt else None),
                "median_residual_mm": (round(float(np.median(dc[near]
                                                             - z_t[near])) * 1e3, 2)
                                       if cnt else None)})
            if cnt >= args.min_plane_points:
                q = d[near] * r[near]          # n.X, since X = ray * d
                rows.append(np.stack([q, xt[near] * q, yt[near] * q, r[near]],
                                     axis=1))
                targets.append(np.full(cnt, off_c))

        n_used = len(rows)
        if n_used == 0:
            if history:
                guard_notes.append(
                    f"no plane kept {args.min_plane_points} points at the "
                    f"{tol * 1e3:.0f} mm window; keeping the previous solution")
                theta, per_plane = last_good, last_per_plane
                break
            return {"solved": False, "reason": "no plane had enough points",
                    "theta": [1.0, 0.0, 0.0, 0.0], "per_plane": per_plane}

        A = np.concatenate(rows)
        y = np.concatenate(targets)
        if A.shape[0] < args.min_samples:
            if history:
                guard_notes.append(
                    f"only {A.shape[0]} samples at the {tol * 1e3:.0f} mm "
                    f"window; keeping the previous solution")
                theta, per_plane = last_good, last_per_plane
                break
            return {"solved": False, "reason": f"only {A.shape[0]} samples",
                    "theta": [1.0, 0.0, 0.0, 0.0], "per_plane": per_plane}

        if args.fix_scale:
            # s held at unity. The q column is known, so it moves to the target
            # side and the remaining three unknowns are independent. Note that
            # this also solves the ONE-PLANE case correctly: with s fixed, a
            # single plane determines c outright, where the free-scale model has
            # to drop c and extrapolate a gain instead.
            use_dof = 3
            Ac, yc = A[:, 1:4], y - A[:, 0]
            ridge_rows = []
            if args.ridge_c > 0:
                sol0, *_ = np.linalg.lstsq(Ac, yc, rcond=None)
                sigma_r = float(np.std(yc - Ac @ sol0)) + 1e-9
                w = np.sqrt(max(Ac.shape[0], 1))
                k = w * sigma_r / args.ridge_c
                ridge_rows.append((np.array([0.0, 0.0, k]), 0.0))
            if ridge_rows:
                Ac = np.vstack([Ac] + [r[0][None, :] for r in ridge_rows])
                yc = np.concatenate([yc, np.array([r[1] for r in ridge_rows])])
            sol, *_ = np.linalg.lstsq(Ac, yc, rcond=None)
            cand = np.array([1.0, sol[0], sol[1], sol[2]])
            cond_cols = 4
        else:
            # s and c are separable only across two DIFFERENT plane depths: at
            # one depth an offset and a gain are the same correction. With one
            # plane the additive term is dropped, which is exactly
            # layer_collapse's model. --fix-scale is the better answer to this.
            use_dof = dof if n_used >= 2 else 3
            if use_dof == 3 and dof == 4 and n_used < 2:
                note = ("only one reference plane was assigned, so s and c are "
                        "degenerate and the additive term is dropped. This is "
                        "the deck-only model, and its correction at parcel "
                        "height is extrapolated. --fix-scale removes this "
                        "degeneracy and solves c from the single plane instead")
                if note not in guard_notes:
                    guard_notes.append(note)
            cols = 4 if use_dof == 4 else 3
            cond_cols = cols
            Ac, yc = A[:, :cols], y
            ridge_rows = []
            if cols == 4 and (args.ridge_s > 0 or args.ridge_c > 0):
                # A prior on a parameter with standard deviation sigma_p is
                # worth one observation of weight sigma_r / sigma_p. sigma_r is
                # the residual scale of the unregularised fit, so the prior is
                # scaled against the noise the data actually carries rather
                # than against an arbitrary constant.
                sol0, *_ = np.linalg.lstsq(Ac, yc, rcond=None)
                sigma_r = float(np.std(yc - Ac @ sol0)) + 1e-9
                # The data supplies almost no information along the s/c
                # degenerate direction, so a prior worth even a handful of
                # observations dominates there while remaining negligible in
                # every direction the data does constrain. Scaled by sqrt(n) so
                # it does not vanish as the sample count grows.
                w = np.sqrt(max(Ac.shape[0], 1))
                if args.ridge_s > 0:
                    k = w * sigma_r / args.ridge_s
                    ridge_rows.append((np.array([k, 0.0, 0.0, 0.0]), k * 1.0))
                if args.ridge_c > 0:
                    k = w * sigma_r / args.ridge_c
                    ridge_rows.append((np.array([0.0, 0.0, 0.0, k]), 0.0))
            if ridge_rows:
                Ac = np.vstack([Ac] + [r[0][None, :] for r in ridge_rows])
                yc = np.concatenate([yc, np.array([r[1] for r in ridge_rows])])
            sol, *_ = np.linalg.lstsq(Ac, yc, rcond=None)
            cand = np.array([sol[0], sol[1], sol[2],
                             sol[3] if cols == 4 else 0.0])

        g = gain(d.shape, K, cand)
        span = float(g.max() - g.min())
        bad = None
        if not args.fix_scale and abs(cand[0] - 1.0) > args.max_scale_dev:
            bad = (f"s = {cand[0]:.5f}, {abs(cand[0] - 1) * 100:.1f}% from "
                   f"unity, beyond --max-scale-dev "
                   f"{args.max_scale_dev * 100:.0f}%")
        elif span > args.max_gain_span:
            bad = (f"gain span {span * 100:.2f}% across the frame, beyond "
                   f"--max-gain-span {args.max_gain_span * 100:.0f}%")
        elif abs(cand[3]) > args.max_offset_m:
            bad = (f"c = {cand[3] * 1e3:+.0f} mm, beyond --max-offset-m "
                   f"{args.max_offset_m * 1e3:.0f} mm")
        elif not np.isfinite(g).all() or g.min() <= 0:
            bad = "the gain is non-positive somewhere in the frame"

        if bad is not None:
            guard_notes.append(f"iteration {it} rejected: {bad}")
            history.append({"iter": it, "tol_mm": round(float(tol) * 1e3, 1),
                            "n": int(A.shape[0]), "n_planes": n_used,
                            "rejected": bad})
            # do NOT adopt it: the next pass assigns through the current field
            continue

        theta = cand
        last_good, last_per_plane = theta.copy(), per_plane
        history.append({"iter": it, "tol_mm": round(float(tol) * 1e3, 1),
                        "n": int(A.shape[0]), "n_planes": n_used,
                        "dof": use_dof, "fix_scale": bool(args.fix_scale),
                        "s": round(float(theta[0]), 6),
                        "a": round(float(theta[1]), 6),
                        "b": round(float(theta[2]), 6),
                        "c_mm": round(float(theta[3]) * 1e3, 2),
                        "gain_span_pct": round(span * 100, 3)})
        cond = conditioning(A[:, :cond_cols], y, per_plane, args)

    g = gain(d.shape, K, theta)
    accepted = [h for h in history if "rejected" not in h]
    return {"solved": True, "theta": [float(v) for v in theta],
            "n_accepted_iterations": len(accepted),
            "dof_used": (accepted[-1].get("dof") if accepted else None),
            "fix_scale": bool(args.fix_scale),
            "s": float(theta[0]), "a": float(theta[1]),
            "b": float(theta[2]), "c_m": float(theta[3]),
            "gain_min": float(g.min()), "gain_max": float(g.max()),
            "gain_span_pct": float((g.max() - g.min()) * 100),
            "implied_scale_error_pct": float((theta[0] - 1.0) * 100),
            "conditioning": cond,
            "iterations": history, "guard_notes": guard_notes,
            "per_plane": per_plane}


def closest_plane_gap_m(planes):
    """Smallest separation between adjacent reference planes, in metres.

    The ceiling on any assignment window is half of this. Both layer_align and
    run.sh already enforce the same rule on their own tolerances; depth_field
    has to as well, and for the same reason: each plane claims points within the
    window of itself in BOTH directions, so two windows must fit in the gap.
    """
    d = sorted(float(p["perp_ref_m"]) for p in planes)
    gaps = [b - a for a, b in zip(d, d[1:])]
    return min(gaps) if gaps else 0.0


def solve_bootstrapped(depth, K, E, valid, planes, args):
    """Stage one on one plane with a wide window, stage two on all with a narrow.

    WHY THIS IS NOT OPTIONAL, measured on capture_00001.

    Fitting all three planes in one stage from a cold start put every camera's
    assignment on the wrong surfaces. The uncorrected error runs to 177 mm, so
    the first window has to exceed that; the closest pair of planes is 87 mm
    apart, so it also has to stay under 44 mm. No single window satisfies both.
    What happened instead: center ended up fitting 901 deck points against 2433
    riser1 points out of 81663 deck points available, `right` had 138 points on
    riser1, and `top` had every one of its four iterations rejected by the
    guards and came out COMPLETELY UNCORRECTED at s=1, c=0. That is one of the
    sheets in the render.

    A single plane cannot be confused with another one, however wide the window.
    So stage one fits the deck alone, which is the largest and nearest surface,
    and after it the residual at the risers is 7 to 11 mm. Stage two can then
    use a 25 mm window, comfortably under the 44 mm ceiling, and the additive
    term becomes solvable because two planes are properly populated.

    Stage two is ACCEPTED ONLY IF IT IMPROVES the residual on the planes stage
    one did not see. Otherwise stage one is kept. On this rig stage one already
    generalises, and a stage that fits its own planes better while predicting
    the others worse is overfitting, not progress.
    """
    out = {"stages": []}
    boot = [p for p in planes if p["label"] == args.bootstrap_plane]
    rest = [p for p in planes if p["label"] != args.bootstrap_plane]

    if not args.bootstrap_plane or not boot:
        f = solve_field(depth, K, E, valid, planes, args)
        f["stage"] = "single"
        if args.bootstrap_plane and not boot:
            f.setdefault("guard_notes", []).append(
                f"no plane labelled {args.bootstrap_plane!r}, so the bootstrap "
                f"was skipped and every plane was fitted in one stage. On an "
                f"anchor whose closest pair is "
                f"{closest_plane_gap_m(planes) * 1e3:.0f} mm apart with an "
                f"uncorrected error of tens of millimetres, expect the "
                f"assignment to land on the wrong surfaces")
        return f

    # ---- stage one: the bootstrap plane alone, wide window ---------------
    # dof=3 here is the single-plane model. Under --fix-scale that means s=1
    # with a real offset; otherwise it means a gain with c=0, extrapolated.
    s1 = solve_field(depth, K, E, valid, boot, args, dof=3,
                     tol_start=args.assign_tol,
                     tol_final=args.assign_tol_final)
    out["stages"].append({"stage": 1, "planes": [args.bootstrap_plane],
                          "theta": s1.get("theta"),
                          "solved": s1["solved"],
                          "guard_notes": s1.get("guard_notes")})
    if not s1["solved"]:
        s1["stage"] = "stage1_failed"
        return s1

    th1 = np.asarray(s1["theta"])
    ev1 = evaluate(depth, K, E, valid, rest, th1, args) if rest else []
    held1 = [e["after"]["rms_mm"] for e in ev1
             if e.get("after") and e["after"].get("rms_mm") is not None]

    if not rest:
        s1["stage"] = "stage1_only"
        s1["stage_detail"] = out["stages"]
        s1["held_out_after_stage1"] = ev1
        return s1

    # ---- stage two: every plane, narrow window, started from stage one ----
    s2 = solve_field(depth, K, E, valid, planes, args,
                     tol_start=args.assign_tol_stage2,
                     tol_final=args.assign_tol_final,
                     init_theta=th1)
    out["stages"].append({"stage": 2, "planes": [p["label"] for p in planes],
                          "theta": s2.get("theta"),
                          "solved": s2["solved"],
                          "guard_notes": s2.get("guard_notes")})

    keep_s1 = True
    reason = None
    if not s2["solved"]:
        reason = f"stage two did not solve: {s2.get('reason')}"
    elif not s2.get("n_accepted_iterations"):
        reason = ("every stage-two iteration was rejected by the guards, so it "
                  "produced no field of its own")
    else:
        th2 = np.asarray(s2["theta"])
        ev2 = evaluate(depth, K, E, valid, planes, th2, args)
        worst2 = [e["after"]["rms_mm"] for e in ev2
                  if e.get("after") and e["after"].get("rms_mm") is not None]
        if not held1:
            # Stage one could not be evaluated on ANY plane it did not see, so
            # there is no evidence either way. Keeping stage one is the
            # conservative choice, but the camera has to be flagged: its
            # correction above the bootstrap plane is unverified, and a camera
            # nobody checked is exactly how a sheet reaches the fused cloud.
            reason = (f"stage one could not be evaluated on any plane it did "
                      f"not see, so there is NO EVIDENCE either way. Keeping "
                      f"stage one and flagging this camera as UNVERIFIED above "
                      f"{args.bootstrap_plane!r}")
            s1["unverified_above_bootstrap"] = True
        elif worst2 and max(worst2) <= max(held1) + 0.5:
            keep_s1 = False
        else:
            reason = (f"stage two's worst per-plane rms is "
                      f"{max(worst2) if worst2 else float('nan'):.2f} mm "
                      f"against stage one's {max(held1):.2f} "
                      f"mm on the planes stage one did not see, so it fits its "
                      f"own planes at the cost of the others")

    if keep_s1:
        s1["stage"] = "stage1_kept"
        s1["stage_detail"] = out["stages"]
        s1["held_out_after_stage1"] = ev1
        s1.setdefault("guard_notes", []).append(
            f"STAGE ONE KEPT: {reason}")
        return s1

    s2["stage"] = "stage2"
    s2["stage_detail"] = out["stages"]
    s2["stage1_theta"] = s1["theta"]
    s2["held_out_after_stage1"] = ev1
    return s2


def conditioning(A, y, per_plane, args):
    """How well the scale and the offset are SEPARATELY determined.

    They are not, over a short lever arm, and the report has to say so. s
    multiplies depth and c does not, so the two are distinguishable only across
    different plane depths: the shorter the span, the more nearly collinear they
    become, and the fit slides along that direction while the residual barely
    moves.

    Measured on synthetic data with the real plane geometry, 218 mm apart, and
    8 mm of per-sample depth noise: the true field (s=0.9973, c=-59.6 mm) was
    recovered as (s=0.9769, c=+2.1 mm). Both describe the same correction to
    within 4 mm ANYWHERE INSIDE the span, and diverge to 7 mm at a parcel top
    400 mm above the deck and 9 mm at 470 mm. So the fit is trustworthy where
    the references are and degrades linearly outside them.

    Under --fix-scale the correlation reported here is still that of the
    UNCONSTRAINED problem, because it is a property of the plane geometry rather
    than of the solve. It is worth reading as a measure of how much the
    constraint is doing.

    sigma is taken PER PLANE, not per sample: a depth bias displaces a whole
    surface at once, so twenty thousand pixels on one plane are one observation.
    Dividing by sqrt(N) here would understate the uncertainty by two orders of
    magnitude, which is the mistake layer_align.plane_systematic_sigma exists to
    avoid.
    """
    out = {}
    try:
        AtA = A.T @ A
        out["condition_number"] = round(float(np.linalg.cond(AtA)), 1)
        inv = np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        return {"condition_number": None,
                "note": "the normal equations are singular"}

    depths = [p["target_depth_median_m"] for p in per_plane
              if p["n_assigned"] >= args.min_plane_points
              and p.get("target_depth_median_m") is not None]
    if len(depths) >= 2:
        out["plane_depths_m"] = [round(v, 4) for v in sorted(depths)]
        out["lever_arm_mm"] = round((max(depths) - min(depths)) * 1e3, 1)
    else:
        out["lever_arm_mm"] = 0.0

    # one observation per plane, sigma = scatter of the plane medians
    meds = [p["median_residual_mm"] for p in per_plane
            if p.get("median_residual_mm") is not None]
    sigma_mm = (float(np.std(meds)) if len(meds) >= 2 else None)
    out["plane_median_scatter_mm"] = (round(sigma_mm, 2)
                                      if sigma_mm is not None else None)

    if A.shape[1] >= 4 and inv is not None:
        # correlation between the s and c columns of the parameter covariance
        vs, vc, vsc = inv[0, 0], inv[3, 3], inv[0, 3]
        denom = np.sqrt(max(vs * vc, 1e-30))
        rho = float(vsc / denom) if denom > 0 else None
        out["scale_offset_correlation"] = (round(rho, 4) if rho is not None
                                          else None)
        if rho is not None and abs(rho) > 0.99:
            out["note"] = (
                f"the scale and the offset are {abs(rho) * 100:.2f}% "
                f"correlated over a {out.get('lever_arm_mm', 0):.0f} mm lever "
                f"arm, so their individual values are NOT separately "
                f"determined"
                + (", which is why --fix-scale is on: it picks the s=1 member "
                   "of that family rather than letting the pair walk"
                   if getattr(args, "fix_scale", False) else
                   ". The CORRECTION is still well determined inside the span "
                   "of the reference planes and degrades roughly linearly "
                   "outside it. Do not quote s or c on their own, and consider "
                   "--fix-scale or a reference plane near the top of the "
                   "parcel band"))
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(depth, K, E, valid, planes, theta, args, tol=None):
    """Per-plane residual and apparent tilt, before and after the field.

    THE TWO WINDOWS DIFFER, DELIBERATELY. The uncorrected error on this rig
    reaches 177 mm, so measuring it inside the 15 mm final window would select
    only the points that already agree and report a small residual for a badly
    placed surface, or find nothing at all and report None. So `before` is
    measured at --assign-tol and `after` at --assign-tol-final, and both windows
    are recorded. The pair is a description of the correction, not a like-for-
    like comparison: a narrower window always flatters the residual.
    """
    tol_before = args.assign_tol if tol is None else tol
    tol_after = args.assign_tol_final if tol is None else tol
    d = depth.astype(np.float64)
    dc = apply_field(d, K, theta)
    out = []
    for pi, plane in enumerate(planes):
        n_c, off_c = plane_in_camera(plane, E)
        z_t, ok, r = target_depth(d.shape, K, n_c, off_c)
        rec = {"plane": pi, "label": plane["label"],
               "window_before_mm": round(tol_before * 1e3, 1),
               "window_after_mm": round(tol_after * 1e3, 1)}
        for tag, z, tol_t in (("before", d, tol_before),
                              ("after", dc, tol_after)):
            near = valid & ok & (np.abs(z - z_t) <= tol_t)
            cnt = int(near.sum())
            if cnt < 100:
                rec[tag] = None
                rec[f"tilt_deg_{tag}"] = None
                continue
            # perpendicular residual, metres
            res = (z[near] - z_t[near]) * r[near]
            rec[tag] = {"n": cnt,
                        "median_mm": round(float(np.median(res)) * 1e3, 2),
                        "rms_mm": round(float(np.sqrt(np.mean(res ** 2)))
                                        * 1e3, 2)}
            rec[f"tilt_deg_{tag}"] = plane_tilt(z, K, E, near, plane, args)
        out.append(rec)
    return out


def plane_tilt(depth_corrected, K, E, near, plane, args):
    """Angle between the surface this camera reconstructs and the true plane."""
    pts = points_camera(depth_corrected, K)[near]
    if len(pts) < 200:
        return None
    n_c, _ = plane_in_camera(plane, E)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    nv = vt[-1]
    cos = abs(float(nv @ (n_c / np.linalg.norm(n_c))))
    return round(float(np.degrees(np.arccos(np.clip(cos, 0, 1)))), 3)


def spatial_structure(depth, K, E, valid, planes, theta, args):
    """p05-p95 range of the per-cell median residual, after correction.

    What a smooth radial field cannot reach. This is the number that says how
    much of the error needs control points rather than a better field, and on
    this rig it is 15 to 21 mm, which is comparable to the inter-view parcel-top
    spread the field exists to remove. Parcels do not sample the whole frame:
    each sits in one region, so its inter-view agreement is governed by this
    figure rather than by the global median residual.
    """
    d = depth.astype(np.float64)
    dc = apply_field(d, K, theta)
    resid = np.full(d.shape, np.nan)
    best = np.full(d.shape, np.inf)
    for plane in planes:
        n_c, off_c = plane_in_camera(plane, E)
        z_t, ok, r = target_depth(d.shape, K, n_c, off_c)
        rr = (dc - z_t) * r
        near = valid & ok & (np.abs(dc - z_t) <= args.assign_tol_final)
        take = near & (np.abs(rr) < best)
        best[take] = np.abs(rr)[take]
        resid[take] = rr[take]
    h, w = d.shape
    ny = nx = max(2, args.residual_cells)
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
        return None
    m = np.asarray(meds) * 1e3
    return round(float(np.percentile(m, 95) - np.percentile(m, 5)), 2)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def capture_correction(cap):
    """What correction, if any, is already baked into depth_{cam}.npy."""
    p = Path(cap) / "capture.json"
    if not p.exists():
        return None
    try:
        meta = (json.loads(p.read_text()) or {}).get("depth_correction") or {}
    except (OSError, ValueError):
        return None
    applied = []
    if meta.get("field") or meta.get("field_source"):
        applied.append(f"a depth field from {meta.get('field_source')}")
    if meta.get("per_camera"):
        applied.append(f"an affine from {meta.get('source')}")
    if meta.get("absolute_applied"):
        applied.append("an absolute anchor")
    if meta.get("auto_align"):
        state = meta.get("auto_align_state") or {}
        applied.append(
            "per-frame auto-align"
            + (f" (model {state.get('model')})" if state.get("model") else "")
            + ", whose per-camera offsets are a property of the FRAME rather "
              "than of the rig and are recorded in the run's "
              "stream_summary.json under auto_align.final_offsets_mm, not per "
              "pixel here")
    return applied or None


def load_capture(cap, names, args=None, solving=True):
    """Load a capture, fitting the UNCORRECTED depth by default.

    da3_stream.py writes depth_{cam}.npy AFTER the correction chain and
    depth_raw_{cam}.npy before it. --load-field applies the field to RAW depth
    at stream time, so a field fitted against the corrected array does not
    compose with the one that gets applied: each camera comes out short by
    whatever the online aligner was doing on the captured frame.

    A capture streamed with no correction at all writes no depth_raw, and its
    depth_{cam}.npy IS raw. That case passes without comment.
    """
    cap = Path(cap)
    mode = getattr(args, "depth_source", "auto") if args is not None else "auto"
    applied = capture_correction(cap)
    data = {}

    for n in names:
        raw = cap / f"depth_raw_{n}.npy"
        cor = cap / f"depth_{n}.npy"

        if mode == "raw":
            if not raw.exists():
                raise SystemExit(
                    f"--depth-source raw, but {raw} does not exist. "
                    f"da3_stream.py writes depth_raw only when a correction "
                    f"was active and --save-raw-depth was on. If this capture "
                    f"was streamed uncorrected, depth_{n}.npy is already raw "
                    f"and --depth-source auto will use it.")
            dpath = raw
        elif mode == "corrected":
            dpath = cor
        elif raw.exists():
            dpath = raw
        else:
            if applied and solving:
                raise SystemExit(
                    f"{cap} has no depth_raw_{n}.npy, and capture.json records "
                    f"that depth_{n}.npy already carries "
                    f"{'; '.join(applied)}.\n\n"
                    f"Fitting a field to it would describe a depth map that no "
                    f"longer exists: da3_stream applies the field to RAW depth "
                    f"and runs the aligner afterwards, so the two would not "
                    f"compose and each camera would come out short by the "
                    f"offset baked in here. Measured on this rig, a top camera "
                    f"carrying +34.5 mm produced a field whose online tracker "
                    f"then sat at +30.6 mm for an entire run undoing it, while "
                    f"parcel-top spread rose from 15.2 to 22.4 mm.\n\n"
                    f"Re-capture with --save-raw-depth on, or pass "
                    f"--depth-source corrected if you are diagnosing rather "
                    f"than producing a field to stream with.")
            dpath = cor

        req = {"K": cap / f"K_{n}.npy", "E": cap / f"E_{n}.npy"}
        missing = [str(p) for p in [dpath, *req.values()] if not p.exists()]
        if missing:
            raise SystemExit(f"missing arrays for {n}: {', '.join(missing)}")

        e = {k: np.load(p) for k, p in req.items()}
        e["depth"] = np.squeeze(np.load(dpath)).astype(np.float64)
        e["depth_source"] = dpath.name
        e["K"] = e["K"].astype(np.float64).reshape(3, 3)

        E = np.asarray(e["E"], np.float64)
        if E.size == 12:
            T = np.eye(4)
            T[:3, :4] = E.reshape(3, 4)
            E = T
        e["E"] = E.reshape(4, 4)

        cp = cap / f"conf_{n}.npy"
        e["conf"] = np.load(cp) if cp.exists() else None
        if e["conf"] is not None and e["conf"].shape != e["depth"].shape:
            e["conf"] = None

        rgb = cap / f"{n}.png"
        e["rgb"] = (cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB)
                    if rgb.exists() and cv2 is not None else None)
        data[n] = e

    srcs = {e["depth_source"] for e in data.values()}
    all_raw = all(s.startswith("depth_raw_") for s in srcs)
    tag = ("UNCORRECTED (depth_raw)" if all_raw
           else "ALREADY CORRECTED" if applied else "uncorrected")
    print(f"[depth] fitting {tag}: "
          + ", ".join(f"{n}/{data[n]['depth_source']}" for n in names))
    if applied and all_raw:
        print(f"        this capture was streamed with {'; '.join(applied)}, "
              f"which is why depth_raw exists and is the array to fit")
    if applied and not all_raw:
        print(f"[WARN] this depth already carries {'; '.join(applied)}. These "
              f"coefficients will NOT compose with --load-field at stream "
              f"time, which applies the field to raw depth. Diagnosis only.")

    return data


def valid_mask(e, args):
    d = e["depth"]
    v = np.isfinite(d) & (d > 0)
    if args.conf_percentile > 0 and e["conf"] is not None and v.any():
        v &= e["conf"] >= float(np.percentile(e["conf"][v], args.conf_percentile))
    if args.edge_thresh > 0:
        v &= edge_mask(d, args.edge_thresh, args.edge_dilate)
    if args.max_incidence < 90:
        ok, _ = incidence_mask(points_camera(d, e["K"]), args.max_incidence)
        v &= ok
    return v


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    names = list(args.cameras)
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} not among {names}")
    out_dir = args.out_dir or (args.capture_dir / "field")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    data = load_capture(args.capture_dir, names, args,
                        solving=(args.load_field is None))
    masks = {n: valid_mask(data[n], args) for n in names}
    planes = gt_planes_world(args.gt, data[args.reference]["E"])

    print(f"[field] d' = d * (s + a*x~ + b*y~) + c,  up to {args.dof} DOF"
          + ("   s HELD AT 1" if args.fix_scale else ""))
    if args.fix_scale:
        print("        the field is tilt plus a constant offset, which is what "
              "was measured:")
        print("        the residual is flat in height to 1-4 mm across the "
              "anchor span, and a")
        print("        218 mm lever arm cannot resolve a scale below that")
    print(f"        stage 1: {args.bootstrap_plane or 'ALL PLANES'} alone, "
          f"window {args.assign_tol * 1e3:.0f} -> "
          f"{args.assign_tol_final * 1e3:.0f} mm")
    if args.bootstrap_plane:
        print(f"        stage 2: every plane, window "
              f"{args.assign_tol_stage2 * 1e3:.0f} -> "
              f"{args.assign_tol_final * 1e3:.0f} mm, started from stage 1")
    print(f"[plane] {len(planes)} measured reference plane(s) from {args.gt}")
    for p in planes:
        h = p.get("height_above_deck_m")
        print(f"        {p['label']:<8} perp {p['perp_ref_m']:.4f} m from "
              f"{args.reference}"
              + (f", {float(h) * 1e3:+.0f} mm above the deck"
                 if isinstance(h, (int, float)) else ""))
    gap = closest_plane_gap_m(planes)
    if gap > 0:
        print(f"        closest pair {gap * 1e3:.1f} mm apart, so any "
              f"assignment window must stay under {gap * 500:.1f} mm")
        if args.assign_tol_stage2 >= gap / 2:
            print(f"[WARN] --assign-tol-stage2 "
                  f"{args.assign_tol_stage2 * 1e3:.0f} mm is at or above half "
                  f"that gap, so a point near one plane will be claimed by "
                  f"both and stage two will be solved on a mixture. Lower it "
                  f"below {gap * 500:.0f} mm.")
        if not args.bootstrap_plane and args.assign_tol >= gap / 2:
            print(f"[WARN] one-stage fitting with a "
                  f"{args.assign_tol * 1e3:.0f} mm window on planes only "
                  f"{gap * 1e3:.0f} mm apart cross-assigns every surface. Use "
                  f"the bootstrap.")
    for n in names:
        print(f"[load ] {n:<8} {int(masks[n].sum()):>8d} of "
              f"{data[n]['depth'].size} pixels "
              f"({100 * masks[n].sum() / data[n]['depth'].size:5.1f}%)")

    report = {"capture_dir": str(args.capture_dir), "gt": str(args.gt),
              "reference": args.reference, "cameras": names,
              "dof": args.dof, "fix_scale": bool(args.fix_scale),
              "depth_source": {n: data[n]["depth_source"] for n in names},
              "capture_correction_already_applied": capture_correction(
                  args.capture_dir),
              "planes": planes,
              "filters": {"conf_percentile": args.conf_percentile,
                          "edge_thresh": args.edge_thresh,
                          "max_incidence_deg": args.max_incidence},
              "per_camera": {}}

    loaded = None
    if args.load_field is not None:
        doc = json.loads(args.load_field.read_text())
        loaded = doc.get("coefficients") or {}
        missing = [n for n in names if n not in loaded]
        if missing:
            raise SystemExit(f"{args.load_field} has no coefficients for "
                             f"{', '.join(missing)}")
        print(f"\n[apply] stored field from {args.load_field}")
        print(f"        solved on {doc.get('source_capture')}")
        print("        NOT re-solved. The residuals below are a DRIFT CHECK: "
              "the field is a")
        print("        property of the cameras and the lens, so if the deck "
              "residual has grown")
        print("        since it was solved, something has moved, been "
              "refocused, or been re-calibrated.")
        src_then = doc.get("depth_source")
        if src_then:
            print(f"        it was solved on {src_then}")
    else:
        print("\n[solve] every camera fitted to the MEASURED planes, "
              "including the reference")

    fields, corrected = {}, {}
    for n in names:
        e, v = data[n], masks[n]
        if loaded is not None:
            c = loaded[n]
            f = {"solved": True, "stage": "loaded",
                 "theta": [float(c["s"]), float(c["a"]), float(c["b"]),
                           float(c["c"])],
                 "s": float(c["s"]), "a": float(c["a"]), "b": float(c["b"]),
                 "c_m": float(c["c"]), "dof_used": "loaded",
                 "guard_notes": [], "per_plane": []}
            gg = gain(e["depth"].shape, e["K"], np.asarray(f["theta"]))
            f["gain_min"], f["gain_max"] = float(gg.min()), float(gg.max())
            f["gain_span_pct"] = float((gg.max() - gg.min()) * 100)
            f["implied_scale_error_pct"] = float((f["s"] - 1.0) * 100)
        else:
            f = solve_bootstrapped(e["depth"], e["K"], e["E"], v, planes, args)
        fields[n] = f
        if not f["solved"]:
            print(f"        {n:<8} NOT SOLVED: {f['reason']}")
            report["per_camera"][n] = f
            corrected[n] = e["depth"]
            continue
        th = np.asarray(f["theta"])
        ev = evaluate(e["depth"], e["K"], e["E"], v, planes, th, args)
        ss = spatial_structure(e["depth"], e["K"], e["E"], v, planes, th, args)
        f["per_plane_residual"] = ev
        f["spatial_structure_mm"] = ss
        f["depth_source"] = e["depth_source"]
        corrected[n] = apply_field(e["depth"], e["K"], th)

        print(f"        {n:<8} s={f['s']:.6f}  a={f['a']:+.6f}  "
              f"b={f['b']:+.6f}  c={f['c_m'] * 1e3:+7.2f} mm   "
              f"gain span {f['gain_span_pct']:.2f}%  "
              + ("[loaded]" if f.get("stage") == "loaded"
                 else f"[{f.get('dof_used')} DOF"
                      + (", s fixed" if f.get("fix_scale") else "")
                      + f", {f.get('stage')}]"))
        for r in ev:
            if not r.get("before") or not r.get("after"):
                continue
            print(f"                 {r['label']:<8} "
                  f"rms {r['before']['rms_mm']:7.2f} -> "
                  f"{r['after']['rms_mm']:6.2f} mm   "
                  f"median {r['before']['median_mm']:+8.2f} -> "
                  f"{r['after']['median_mm']:+6.2f} mm   tilt "
                  f"{r.get('tilt_deg_before')} -> {r.get('tilt_deg_after')} deg")
        if ss is not None:
            print(f"                 spatial structure after correction: "
                  f"{ss:.1f} mm  <- what no radial field can reach")
        # A fit sitting exactly ON the gain ceiling with earlier iterations
        # rejected for exceeding it is being clipped, not converging.
        clipped = [g for g in f.get("guard_notes", []) if "gain span" in g]
        if clipped and abs(f["gain_span_pct"] - args.max_gain_span * 100) < 0.05:
            print(f"                 CLIPPED: the accepted gain span "
                  f"{f['gain_span_pct']:.2f}% sits on the "
                  f"--max-gain-span ceiling with "
                  f"{len(clipped)} earlier iteration(s) rejected above it. "
                  f"This camera wants more tilt than the guard allows; check "
                  f"its post-correction tilt against its rig-mates before "
                  f"trusting it")
        for g in f["guard_notes"]:
            print(f"                 GUARD: {g}")
        report["per_camera"][n] = f

    # ---- inter-camera agreement, the layer_align metric ------------------
    print("\n[layer] inter-camera spread at each measured plane, mm")
    print("        computed against the PHYSICAL plane, so this is accuracy, "
          "not merely agreement")
    layering = []
    for pi, plane in enumerate(planes):
        row = {"plane": pi, "label": plane["label"],
               "before": {}, "after": {}}
        for n in names:
            e, v = data[n], masks[n]
            n_c, off_c = plane_in_camera(plane, e["E"])
            z_t, ok, r = target_depth(e["depth"].shape, e["K"], n_c, off_c)
            for tag, z in (("before", e["depth"]), ("after", corrected[n])):
                near = v & ok & (np.abs(z - z_t) <= args.assign_tol_final)
                if int(near.sum()) < 100:
                    continue
                row[tag][n] = round(float(np.median((z[near] - z_t[near])
                                                    * r[near])) * 1e3, 2)
        for tag in ("before", "after"):
            vals = list(row[tag].values())
            row[f"spread_{tag}_mm"] = (round(max(vals) - min(vals), 2)
                                       if len(vals) > 1 else None)
        layering.append(row)
        cams_a = "  ".join(f"{k} {v:+7.2f}" for k, v in row["after"].items())
        print(f"        {plane['label']:<8} after: {cams_a}")
        print(f"                 spread {row['spread_before_mm']} -> "
              f"{row['spread_after_mm']} mm"
              + ("" if len(row["after"]) == len(names)
                 else f"   absent: "
                      f"{', '.join(n for n in names if n not in row['after'])}"))
    report["layering"] = layering

    # ---- the holdout ----------------------------------------------------
    if args.holdout:
        deck = [p for p in planes if p["label"] == "deck"]
        others = [p for p in planes if p["label"] != "deck"]
        if not deck or not others:
            print("\n[hold ] not run: needs a plane labelled 'deck' and at "
                  "least one other")
        else:
            print("\n[hold ] fitted on the DECK ALONE, evaluated on the "
                  "surface(s) above it")
            print("        This is the only residual here not measured on the "
                  "planes it was fitted to.")
            folds = {}
            for n in names:
                e, v = data[n], masks[n]
                f = solve_field(e["depth"], e["K"], e["E"], v, deck, args)
                if not f["solved"]:
                    continue
                th = np.asarray(f["theta"])
                ev = evaluate(e["depth"], e["K"], e["E"], v, others, th, args)
                folds[n] = {"theta": f["theta"], "held_out": ev}
                for r in ev:
                    if not r.get("before") or not r.get("after"):
                        continue
                    print(f"        {n:<8} {r['label']:<8} "
                          f"median {r['before']['median_mm']:+8.2f} -> "
                          f"{r['after']['median_mm']:+7.2f} mm   "
                          f"rms {r['before']['rms_mm']:7.2f} -> "
                          f"{r['after']['rms_mm']:6.2f} mm")
            report["holdout"] = folds
            print("        A held-out residual no better than the uncorrected "
                  "one means the deck")
            print("        cannot stand in for the parcel tops, whatever the "
                  "deck residual says.")

    # ---- write ----------------------------------------------------------
    if args.load_field is None:
        (out_dir / "depth_field.json").write_text(json.dumps({
            "source_capture": str(args.capture_dir),
            "model": "d' = d * (s + a*x~ + b*y~) + c",
            "reference": args.reference,
            "fix_scale": bool(args.fix_scale),
            # WHICH ARRAY THESE COEFFICIENTS DESCRIBE. They compose with
            # da3_stream's --load-field only if this says depth_raw, because
            # that is the array the field is applied to at stream time.
            "depth_source": {n: data[n]["depth_source"] for n in names},
            "capture_correction_already_applied": report[
                "capture_correction_already_applied"],
            "coefficients": {n: {"s": fields[n].get("s", 1.0),
                                 "a": fields[n].get("a", 0.0),
                                 "b": fields[n].get("b", 0.0),
                                 "c": fields[n].get("c_m", 0.0)}
                             for n in names},
            "planes": planes,
            "filters": report["filters"],
            "layering": layering,
            "detail": fields,
        }, indent=2, default=str))

    if args.write_clouds:
        fused = o3d.geometry.PointCloud()
        tinted = o3d.geometry.PointCloud()
        for i, n in enumerate(names):
            pts = points_camera(corrected[n], data[n]["K"])[masks[n]]
            pw = to_world(pts, data[n]["E"])
            cols = (data[n]["rgb"][masks[n]]
                    if data[n]["rgb"] is not None
                    and data[n]["rgb"].shape[:2] == masks[n].shape else None)
            fused += to_o3d(pw, colors=cols)
            tinted += to_o3d(pw, rgb01=PALETTE[i % len(PALETTE)])
        if args.voxel > 0:
            fused = fused.voxel_down_sample(args.voxel)
        o3d.io.write_point_cloud(str(out_dir / "fused_field.ply"), fused)
        o3d.io.write_point_cloud(str(out_dir / "fused_field_by_camera.ply"),
                                 tinted)
        print(f"\n[write] {out_dir / 'fused_field.ply'}  "
              f"{len(fused.points)} points")
        print(f"        {out_dir / 'fused_field_by_camera.ply'}  "
              f"one colour per camera: open this one to see whether the sheets "
              f"are gone")

    if args.write_capture is not None:
        # A corrected capture directory, byte-compatible with what da3_stream
        # wrote, so box_segment.py, layer_align.py and layer_collapse.py all
        # run on it unchanged. The only difference is that depth_<cam>.npy now
        # holds ANCHORED METRIC depth, which means --seg-plane-distance and
        # --deck-distance become PHYSICAL distances on this directory rather
        # than positions in DA3's unanchored coordinates. That is the whole
        # point: the deck sits where the ChArUco board says it does.
        import shutil
        cd = args.write_capture
        cd.mkdir(parents=True, exist_ok=True)
        for n in names:
            np.save(cd / f"depth_{n}.npy",
                    corrected[n].astype(np.float32))
            # The uncorrected array travels with it, so this directory can be
            # re-fitted later without tripping the depth-source guard.
            np.save(cd / f"depth_raw_{n}.npy",
                    data[n]["depth"].astype(np.float32))
            for stem in (f"K_{n}.npy", f"E_{n}.npy", f"conf_{n}.npy",
                         f"{n}.png"):
                src = args.capture_dir / stem
                if src.exists():
                    shutil.copy2(src, cd / stem)
        for stem in ("capture.json", "meta.json"):
            src = args.capture_dir / stem
            if src.exists():
                shutil.copy2(src, cd / stem)
        (cd / "corrected_by.json").write_text(json.dumps({
            "tool": "depth_field.py",
            "model": "d' = d * (s + a*x~ + b*y~) + c",
            "source_capture": str(args.capture_dir),
            "depth_source": {n: data[n]["depth_source"] for n in names},
            "field_source": (str(args.load_field) if args.load_field
                             else "solved on this capture"),
            "fix_scale": bool(args.fix_scale),
            "coefficients": {n: {"s": fields[n].get("s", 1.0),
                                 "a": fields[n].get("a", 0.0),
                                 "b": fields[n].get("b", 0.0),
                                 "c": fields[n].get("c_m", 0.0)}
                             for n in names},
            "depth_is_metric_and_anchored": True,
            "deck_perp_m": next((p["perp_ref_m"] for p in planes
                                 if p["label"] == "deck"), None),
            "note": ("depth_<cam>.npy holds CORRECTED depth and "
                     "depth_raw_<cam>.npy the array it was computed from. "
                     "Distances in this directory are physical metres from the "
                     "reference camera, so --seg-plane-distance and "
                     "--deck-distance take the MEASURED deck standoff here, "
                     "not a probed value. capture.json is copied verbatim from "
                     "the source and describes THAT capture's correction, not "
                     "this one"),
        }, indent=2, default=str))
        deck_m = next((p["perp_ref_m"] for p in planes
                       if p["label"] == "deck"), None)
        print(f"\n[write] corrected capture -> {cd}")
        print( "        every downstream tool reads this directory unchanged")
        if deck_m:
            print(f"        depth here is ANCHORED, so segment with "
                  f"--seg-plane-distance around {deck_m:.4f} m")

    report["runtime_s"] = round(time.perf_counter() - t0, 3)
    (out_dir / "field_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"report: {out_dir / 'field_report.json'}")
    print("\nRead, in order: the holdout residual, then the spread AFTER at the "
          "upper plane,\nthen the spatial structure. A plane cannot see "
          "in-plane translation or yaw, so a\nclean result here means the depth "
          "field is right, not that the cameras are\nregistered.")
    if not all(s.startswith("depth_raw_")
               for s in report["depth_source"].values()):
        print("\nNOTE: these coefficients were fitted on depth_<cam>.npy. They "
              "describe that\narray, not the raw model output that "
              "da3_stream's --load-field corrects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())