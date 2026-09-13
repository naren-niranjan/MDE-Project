#!/usr/bin/env python3
"""
depth_align.py

Fit a per-view depth correction expressed in HEIGHT ABOVE THE DECK, against
the ChArUco board's own solved pose in each capture, and write
depth_correction.json for the fusion pipeline to consume.

WHAT CHANGED, AND WHY
---------------------
The previous version fitted an affine map in inverse depth,

    1 / z_true = a * (1 / z_measured) + b

pooled over every placement, plus an 8 x 8 residual grid in disparity. Three
things were wrong with that, and all three are visible in its own output.

First, THE ERROR IS NOT AFFINE IN DISPARITY, IT IS AFFINE IN HEIGHT. The
[relief] block already reported the correct model: the excess is a loss of
relief, a rise of h reported as h(1 - k), with k between 0.155 and 0.181 and a
within-board scatter of 4 to 9 mm. Fitting the same relationship in disparity
over a depth span of 0.22 m out of 3.07 m is a badly conditioned
reparameterisation of it. The condition number came out at 1.6e4 and the
leave-one-out test failed by 43 to 86 mm when the deck group was held out,
which is a solve extrapolating outside its own data, not a correction.

Second, THE RESIDUAL GRID WAS DOING REAL DAMAGE. A disparity offset dd moves
depth by roughly z^2 dd. At 3.07 m the largest fitted cells, 0.0086 and 0.0115
per metre, are 81 mm and 108 mm of depth. Coverage was 23 to 31 per cent,
unsupported cells were zero, and the 8 x 8 grid was bilinearly stretched onto
a 504-pixel map, so each of those corrections ramped to zero over about sixty
pixels. That is a fold, and it is why the conveyor looked clean before the
correction and rippled afterwards. The grid is gone. If it returns it needs a
magnitude cap and a neighbourhood support requirement, not a minimum count.

Third, THE GATE WAS AIMED AT THE WRONG TERM. Ninety-three per cent of the
placement scatter is shared by all four views inside one capture, because DA3
solves the four frames jointly. That term moves all four sheets together, so
it cannot cause layering and no per-view correction can reach it. The old file
measured it, correctly called it irreducible, and then refused to write the
one part of the correction that was well determined. Worse, with no per-
capture parameter in the model that shared drift was being absorbed into a
and b, corrupting the part that was recoverable.

THE MODEL
---------
Heights are measured perpendicular to the deck plane, in each camera's OWN
frame, so no extrinsic enters the fit:

    h = z * (n_c . r) - d_c          r = ((u - cx)/fx, (v - cy)/fy, 1)

with (n_c, d_c) the deck plane fitted from the deck run's board faces in that
camera's frame. The FORWARD model, which is the one that is fitted, is

    h_measured = p_v * h_true + q_v

    p_v = p_bar + a_v                sum_v a_v = 0
    q_v = q_bar + b_v                sum_v b_v = 0

so p is the surviving fraction of every rise and 1 - p is the loss of relief
that the [relief] table already reported. The correction applied downstream is
its inverse, h' = alpha_v h + beta_v with alpha_v = 1 / p_v and
beta_v = -q_v / p_v, converted back to a depth along the same ray.

THE DIRECTION OF THE REGRESSION IS NOT A MATTER OF TASTE. The board's height
is known exactly from its own solved pose; every error is in the measurement.
Regressing the known quantity on the noisy one attenuates the slope by
var(h) / (var(h) + var(noise)), and on this rig the per-capture drift is large
enough for that to matter: on synthetic data with the measured drift the
attenuation turned a true compression of 0.17 into 0.06. Fitted forward, with
the exact quantity as the predictor, ordinary least squares is unbiased.

THE FIT IS IN TWO STAGES, AND SO IS THE GATE
--------------------------------------------
Stage A fits alpha_bar and beta_bar over every placement, pooled, with no
per-capture term. This is the mean loss of relief. Its uncertainty is
dominated by the per-capture drift and comes out at tens of millimetres. It is
still worth applying, because it is a mean and applying it halves the absolute
error, but it must be reported with that uncertainty attached.

Stage B fits the per-view deviations WITH a free offset e_f per capture:

    h_true = alpha_v * h_measured + beta_v + e_f      sum_f e_f = 0

The capture term soaks up the shared drift, so a_v and b_v are estimated from
how the four views disagree WITH EACH OTHER, which is the quantity that
actually needs correcting. e_f is never applied downstream; it is unknowable
at run time. It cancels in any spread across views, which is exactly why the
layering can be removed while the absolute error cannot.

IDENTIFIABILITY, AND THE SLOPE GATE
-----------------------------------
Within one capture the four views see the board at the same true height, so
a_v is identified only by how each view's deviation from the capture mean
GROWS with height across captures. On this rig the per-view scatter is 7 to
8 mm and the compression spread is 0.026, worth about 6 mm at 218 mm, so a_v
is marginal while b_v is not. A cluster bootstrap over captures measures both,
and a camera whose slope is not resolved keeps an offset-only correction
rather than a fitted slope that is mostly noise. This mirrors the earlier
finding on this rig that an offset-only correction was equivalent to or better
than a scale-corrected one.

WHAT IS WRITTEN, AND WHAT IS REFUSED
------------------------------------
The delayering gate is on the per-view part, at parcel-top height. If it
fails, nothing is written, because a per-view correction that is not resolved
will move the four sheets apart rather than together. The shared gate is
informational and never blocks: the shared correction is written together with
the measured height uncertainty, and downstream must carry that number rather
than derive one from inter-view agreement.

THE BOARD THICKNESS IS NOT APPLIED HERE
---------------------------------------
The depth model reconstructs the printed face, so the face is the target and
the thickness is zero in the fit. It reappears only in the validation that
converts a board-face standoff into a deck-surface standoff.

INPUTS
------
Capture runs written by capture_gt.py, each a directory holding frame_XXXXX
subdirectories and a manifest.json. Each frame directory must hold, per
camera, the rectified image the board was detected in, and the DA3 arrays
depth_<name>.npy and K_<name>.npy. E_<name>.npy is used only for the reports.

EXAMPLE
-------
python depth_align.py \
    --run runs/gt_deck --run runs/gt_riser1 --run runs/gt_riser \
    --calib-dir /home/jetson/Projects/Calibration_4_5/results \
    --board-thickness-mm 6 --out depth_correction.json

Self-check, no hardware and no captures:
    python depth_align.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

# Rectification, intrinsics and the world transform come from the streamer, so
# there is ONE definition of the rig geometry rather than two that drift apart.
# The import is soft: --self-test must run on a machine with no Arena SDK, and
# da3_stream.py pulls that in. Anything that needs the rig raises with a clear
# message instead of failing at import time.
_IMPORT_ERROR = None
try:
    from da3_stream import (DEFAULT_CALIB, build_rect_plan, camera_centre,
                            load_extrinsics, load_intrinsics, to_world)
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = exc
    DEFAULT_CALIB = Path("/home/jetson/Projects/Calibration_4_5/results")

    def _unavailable(*_a, **_k):
        raise SystemExit(
            f"da3_stream.py must sit beside this file and must import "
            f"cleanly; rectification, intrinsics and the world transform are "
            f"taken from it so there is one definition of the rig geometry. "
            f"The import failed with: {_IMPORT_ERROR!r}")

    build_rect_plan = camera_centre = _unavailable
    load_extrinsics = load_intrinsics = to_world = _unavailable


MIN_CORNERS = 12
IMAGE_PATTERNS = ("rect_{name}.png", "rgb_{name}.png", "image_{name}.png",
                  "{name}.png", "rect_{name}.jpg", "rgb_{name}.jpg",
                  "{name}.jpg")

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# --------------------------------------------------------------------------
# geometry primitives
# --------------------------------------------------------------------------

def pixel_rays(shape, K):
    h, w = shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def ray_dirs(u, v, K):
    """Unnormalised rays for scattered pixel coordinates, r = (x, y, 1)."""
    x = (np.asarray(u, np.float64) - K[0, 2]) / K[0, 0]
    y = (np.asarray(v, np.float64) - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def plane_target_depth(shape, K, n_cam, d_cam):
    """
    Depth each pixel would report if the plane n . x = d, expressed in this
    camera's own frame, were reconstructed perfectly.

    Along a ray x = z r with r = (x, y, 1),  z = d / (n . r). No extrinsics are
    involved: the board pose and the depth map share a frame, so the fit is
    immune to extrinsic error. Extrinsics are used only for the cross-camera
    consistency report and for expressing heights in a common frame.
    """
    rays = pixel_rays(shape, K)
    denom = rays @ n_cam
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(denom) > 1e-9, d_cam / denom, np.nan)
    return np.where(np.isfinite(z) & (z > 0), z, np.nan)


def heights_from_depth(z, u, v, K, n_c, d_c):
    """
    Height above the deck plane for scattered samples.

    h = z (n_c . r) - d_c. With the plane normal oriented back towards the
    camera, d_c is negative and n_c . r is about -1, so h grows as the depth
    falls, which is the physical sense of the word.
    """
    nr = ray_dirs(u, v, K) @ np.asarray(n_c, np.float64)
    return np.asarray(z, np.float64) * nr - float(d_c)


def correct_depth_height(depth, K, n_c, d_c, alpha, beta):
    """
    Apply h' = alpha h + beta to a whole depth map and invert back to depth.

    Returns NaN where the ray is parallel to the deck plane or the correction
    drives the depth non-positive. Those are genuine failures and are left as
    NaN rather than clamped: a clamped pixel back-projects to a plausible
    wrong position and nothing downstream can notice.
    """
    z = np.asarray(depth, np.float64)
    nr = pixel_rays(z.shape, K) @ np.asarray(n_c, np.float64)
    h = z * nr - float(d_c)
    h2 = float(alpha) * h + float(beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(np.abs(nr) > 1e-9, (h2 + float(d_c)) / nr, np.nan)
    return np.where(np.isfinite(z2) & (z2 > 0), z2, np.nan)


def fit_plane_robust(points, iters=6, tukey_c=2.5):
    """One plane through pooled points, reweighted rather than RANSACed."""
    P = np.asarray(points, np.float64)
    if len(P) < 8:
        return None, None, None
    w = np.ones(len(P))
    n = np.array([0.0, 0.0, 1.0])
    c = P.mean(axis=0)
    for _ in range(max(1, iters)):
        ws = w / max(float(w.sum()), 1e-12)
        c = (P * ws[:, None]).sum(axis=0)
        M = (P - c) * np.sqrt(ws)[:, None]
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        n = vt[-1] / np.linalg.norm(vt[-1])
        r = (P - c) @ n
        s = max(1.4826 * float(np.median(np.abs(r))), 1e-9)
        u = np.clip(r / (tukey_c * s), -1.0, 1.0)
        w = (1.0 - u ** 2) ** 2
    return n, float(n @ c), (P - c) @ n


# --------------------------------------------------------------------------
# board, detection and pose
# --------------------------------------------------------------------------
# Conventions copied deliberately from measure_deck.py: same dictionary
# lookup, same legacy-pattern handling, same solvePnPGeneric ambiguity test.
# They are re-stated rather than imported because the board layout is read per
# run from the capture manifest, which may not match any tool's compiled-in
# default.

def build_board(spec, legacy):
    name = spec["dictionary"]
    if not hasattr(cv2.aruco, name):
        raise SystemExit(f"unknown dictionary {name!r}. Available: "
                         + ", ".join(a for a in dir(cv2.aruco)
                                     if a.startswith("DICT_")))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    sx, sy = int(spec["squares"][0]), int(spec["squares"][1])
    n_markers = (sx * sy) // 2
    ids = np.arange(spec.get("id_offset", 0),
                    spec.get("id_offset", 0) + n_markers, dtype=np.int32)
    board = cv2.aruco.CharucoBoard((sx, sy),
                                   float(spec["square_mm"]) / 1000.0,
                                   float(spec["marker_mm"]) / 1000.0,
                                   dictionary, ids)
    if legacy:
        if not hasattr(board, "setLegacyPattern"):
            raise SystemExit("this OpenCV build has no setLegacyPattern, but "
                             "the manifest says the board is a legacy layout")
        board.setLegacyPattern(True)
    return board


def build_detector(board):
    cp = cv2.aruco.CharucoParameters()
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.cornerRefinementWinSize = 5
    dp.cornerRefinementMaxIterations = 50
    dp.cornerRefinementMinAccuracy = 0.01
    dp.adaptiveThreshWinSizeMax = 53
    return cv2.aruco.CharucoDetector(board, cp, dp, cv2.aruco.RefineParameters())


def board_centre_object(board):
    """
    The board's geometric centre in board coordinates.

    The mean of ALL chessboard corners, not of the ones this camera happened
    to detect. Every height below is read at this one point, so a view that
    sees half a tilted board reports the same height as a view that sees all
    of it, and the cross-camera check measures the rig rather than the
    detection.
    """
    return np.asarray(board.getChessboardCorners(),
                      np.float64).reshape(-1, 3).mean(axis=0)


def solve_board_pose(gray, board, detector, K, dist):
    """
    Board pose from one rectified image, returned with the printed-face plane
    in the camera frame.

    The plane normal is oriented back towards the camera, so d = n . centroid
    is negative and z = d / (n . r) is positive.
    """
    best = (None, None, 0)
    for im in (gray, _CLAHE.apply(gray)):
        cc, ids, _, _ = detector.detectBoard(im)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            best = (cc, ids, n)
    cc, ids, n = best
    if ids is None or n < MIN_CORNERS:
        return None, f"only {n} ChArUco corners, below {MIN_CORNERS}"

    obj, imgp = board.matchImagePoints(cc, ids)
    if obj is None or len(obj) < MIN_CORNERS:
        return None, "corner matching failed"

    ambiguity = None
    rvec = tvec = None
    try:
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if n_sol >= 1:
            e = np.asarray(errs, np.float64).ravel()
            order = np.argsort(e)
            rvec, tvec = rvecs[order[0]], tvecs[order[0]]
            if n_sol >= 2 and e[order[0]] > 1e-9:
                ambiguity = float(e[order[1]] / e[order[0]])
    except cv2.error:
        pass
    if rvec is None:
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, "pose solve failed"

    rvec, tvec = cv2.solvePnPRefineLM(obj, imgp, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, np.float64).ravel()
    face_pts = (np.asarray(obj, np.float64).reshape(-1, 3) @ R.T) + t
    centre = face_pts.mean(axis=0)

    normal = R[:, 2].astype(np.float64)
    normal = normal / np.linalg.norm(normal)
    if float(normal @ centre) > 0:
        normal = -normal

    centre_board = R @ board_centre_object(board) + t

    return {
        "n_corners": int(n),
        "rms_px": rms,
        "ambiguity_ratio": ambiguity,
        "normal_cam": normal,
        "d_cam": float(normal @ centre),
        "R": R, "t": t,
        "face_points_cam": face_pts,
        "face_centroid_cam": centre,
        "board_centre_cam": centre_board,
        "tilt_deg": float(np.degrees(np.arccos(
            np.clip(abs(normal[2]), 0.0, 1.0)))),
        "lateral_offset_m": float(np.linalg.norm(centre[:2])),
        "range_m": float(np.linalg.norm(centre)),
    }, None


def choose_legacy(samples, spec):
    """
    Try both marker layouts on a few images and keep whichever measures
    better. A board matching neither still returns interpolated corners, so
    the failure mode is a pose solved against the wrong object points rather
    than a detection error, and corner count alone does not catch it.
    """
    scores = {}
    for legacy in (False, True):
        try:
            board = build_board(spec, legacy)
        except SystemExit:
            continue
        det = build_detector(board)
        corners, rms = 0, []
        for gray, K, dist in samples[:3]:
            rec, _ = solve_board_pose(gray, board, det, K, dist)
            if rec is not None:
                corners += rec["n_corners"]
                rms.append(rec["rms_px"])
        scores[legacy] = (corners, -(float(np.mean(rms)) if rms else 9.99))
    if not scores or max(v[0] for v in scores.values()) == 0:
        raise SystemExit("no ChArUco corners detected under either marker "
                         "layout. Check the manifest board block and that the "
                         "images are the rectified ones.")
    for lg, (c, negrms) in sorted(scores.items()):
        print(f"[board] legacy={str(lg):<5} corners={c:4d}  "
              f"mean rms={-negrms:.3f} px")
    vals = list(scores.values())
    if len(vals) == 2 and vals[0] == vals[1]:
        print("[board] both layouts score identically, so setLegacyPattern is "
              "a no-op on this board or this OpenCV build and the choice does "
              "not matter")
        return None
    return max(scores, key=lambda k: scores[k])


def board_mask(shape, K, face_pts_cam, erode_px):
    """
    Pixels inside the detected board footprint, on the depth grid, together
    with each pixel's distance in pixels from the footprint boundary.

    The hull of the MATCHED corners is used rather than the board rectangle:
    it is guaranteed to lie on the board and to be covered by detection, and
    it carries no assumption about which corner the marker grid starts from.
    Projection goes through the depth-grid intrinsics directly, so no scaling
    convention between the rectified image and the depth map is assumed.

    The distance is computed on the UN-eroded footprint, so it stays a
    physical distance from the board edge however --mask-erode-px is set.
    """
    h, w = shape
    x = face_pts_cam @ K.T
    uv = x[:, :2] / x[:, 2:3]
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
    raw = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(raw, np.round(hull).astype(np.int32), 1)
    dist = cv2.distanceTransform(raw, cv2.DIST_L2, 5)
    mask = raw
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(raw, k)
    return mask.astype(bool), dist


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------

def irls_line(x, y, huber=None, iters=5):
    """Robust least squares for y = alpha x + beta."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x)
    alpha, beta = 1.0, 0.0
    for _ in range(iters if huber else 1):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
        if not huber:
            break
        r = np.abs(alpha * x + beta - y)
        s = max(1.4826 * float(np.median(r)), 1e-12)
        w = np.where(r <= huber * s, 1.0, huber * s / np.maximum(r, 1e-12))
    return alpha, beta


def lstsq_constrained(A, y, C, big=1e5):
    """Least squares subject to C p = 0, imposed by heavy weighting."""
    A2 = np.vstack([A, big * np.asarray(C, np.float64)])
    y2 = np.concatenate([y, np.zeros(len(C))])
    sol, *_ = np.linalg.lstsq(A2, y2, rcond=None)
    return sol


def solve_effects(rows, cams, slope_cams, huber=None):
    """
    The two-stage solve.

    Stage A is the rig-wide loss of relief. Stage B is the per-view deviation
    from it, with one free offset per capture so that the drift shared by all
    four views inside a capture is absorbed rather than smeared into the
    per-view terms.

    Both stages run FORWARD, h_measured against h_true, because the board's
    height is exact and the measurement is not. Fitting the other way round
    puts the capture drift into the predictor and attenuates the slope.

    Gauge: the per-view slopes sum to zero over the cameras that keep one, the
    per-view offsets sum to zero over all cameras, and the capture offsets sum
    to zero. Without those three constraints stage B is rank deficient by
    exactly three and the split between shared and per-view is arbitrary.
    """
    x = np.array([r["h_true"] for r in rows], np.float64)
    y = np.array([r["h_meas"] for r in rows], np.float64)

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

    C = []
    if nS >= 2:
        row = np.zeros(A.shape[1])
        row[:nS] = 1.0
        C.append(row)
    row = np.zeros(A.shape[1])
    row[nS:nS + nC] = 1.0
    C.append(row)
    row = np.zeros(A.shape[1])
    row[nS + nC:] = 1.0
    C.append(row)

    sol = lstsq_constrained(A, resid, np.stack(C))

    a = {c: 0.0 for c in cams}
    for c, i in sidx.items():
        a[c] = float(sol[i])
    b = {c: float(sol[nS + cidx[c]]) for c in cams}
    e = {k: float(sol[nS + nC + kidx[k]]) for k in keys}
    return {"p_bar": p_bar, "q_bar": q_bar, "a": a, "b": b, "e": e}


def bootstrap_effects(rows, cams, slope_cams, huber, draws, seed=0):
    """
    Cluster bootstrap over CAPTURES, not over pixels and not over placements.

    Four views of one capture share the drift that dominates this rig, so they
    are one cluster. Resampling captures is the only resampling that carries
    that dependence into the reported uncertainty. Resampling placements would
    treat four correlated rows as four independent ones and understate every
    sigma by about a factor of two.
    """
    rng = np.random.default_rng(seed)
    keys = sorted({r["key"] for r in rows})
    by_key = {k: [] for k in keys}
    for r in rows:
        by_key[r["key"]].append(r)

    out = {"p_bar": [], "q_bar": [],
           "a": {c: [] for c in cams}, "b": {c: [] for c in cams}}
    for d in range(draws):
        pick = rng.choice(len(keys), len(keys), replace=True)
        sub = []
        for j, p in enumerate(pick):
            for r in by_key[keys[p]]:
                q = dict(r)
                q["key"] = (r["key"], j)
                sub.append(q)
        counts = {c: 0 for c in cams}
        for r in sub:
            counts[r["camera"]] += 1
        if min(counts.values()) < 3:
            continue
        if len({round(r["h_true"], 3) for r in sub}) < 2:
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
    return out


def sd(v):
    v = np.asarray(v, np.float64)
    return float(v.std(ddof=1)) if len(v) > 2 else float("nan")


# --------------------------------------------------------------------------
# capture loading
# --------------------------------------------------------------------------

def resolve_image(frame_dir: Path, name: str, pattern: str | None):
    pats = (pattern,) if pattern else IMAGE_PATTERNS
    for p in pats:
        cand = frame_dir / p.format(name=name)
        if cand.exists():
            return cand
    return None


def load_manifest(run_dir: Path):
    path = run_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{run_dir} has no manifest.json; this must be a "
                         f"capture_gt.py run directory")
    man = json.loads(path.read_text())
    for key in ("label", "nominal_mm", "cameras", "board"):
        if key not in man:
            raise SystemExit(f"{path} is missing {key!r}")
    frames = sorted(d for d in run_dir.iterdir()
                    if d.is_dir() and d.name.startswith("frame_"))
    if not frames:
        raise SystemExit(f"{run_dir} holds no frame_XXXXX directories")
    return man, frames


# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Fit a per-view height-space depth correction against the "
                    "ChArUco board pose solved in each capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", action="append", metavar="DIR",
                    help="a capture_gt.py run directory; repeat once per "
                         "reference height, and include the deck run")
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="override the camera list in the manifests")
    ap.add_argument("--reference", default="center")
    ap.add_argument("--lens", default="",
                    help="lens tag written into the output; empty takes the "
                         "tag from the manifests, which must all agree")
    ap.add_argument("--out", type=Path, default=Path("depth_correction.json"))
    ap.add_argument("--image-pattern", default=None,
                    help="filename of the rectified image inside a frame "
                         "directory, with {name} for the camera; the usual "
                         "candidates are tried when this is not given")

    ap.add_argument("--board-thickness-mm", type=float, default=6.0,
                    help="printed face to the surface underneath. Used ONLY "
                         "to turn a board-face standoff into a deck-surface "
                         "standoff in the validation report; it never enters "
                         "the fit, because the depth model sees the face")
    ap.add_argument("--board-legacy", choices=["auto", "true", "false"],
                    default="auto",
                    help="marker layout; auto scores both and keeps the "
                         "better, and the manifest's own flag is the default "
                         "starting guess")
    ap.add_argument("--max-rms-px", type=float, default=0.60,
                    help="drop a placement whose pose reprojects worse")
    ap.add_argument("--min-ambiguity-ratio", type=float, default=3.0,
                    help="drop a placement whose second PnP solution is within "
                         "this factor of the first; such a pose is not "
                         "determined by the corners")
    ap.add_argument("--min-tilt-deg", type=float, default=0.0,
                    help="warn below this; a near fronto-parallel board makes "
                         "the standoff the least well constrained quantity in "
                         "the solve")

    ap.add_argument("--mask-erode-px", type=int, default=4,
                    help="inward margin on the board footprint, in depth-grid "
                         "pixels, keeping the model's edge bleed out of the fit")
    ap.add_argument("--edge-thresh", type=float, default=0.02,
                    help="relative depth gradient above which a pixel is "
                         "treated as a discontinuity; 0 disables")
    ap.add_argument("--max-residual-m", type=float, default=0.40,
                    help="absolute reject before the robust fit; a pixel this "
                         "far from the board plane is not on the board")
    ap.add_argument("--huber", type=float, default=2.0,
                    help="Huber threshold in robust sigmas for the shared "
                         "stage; 0 for plain least squares")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="per camera per placement, subsampled for the solve")

    ap.add_argument("--bootstrap", type=int, default=400,
                    help="cluster bootstrap draws over captures; 0 disables "
                         "and with it every gate, since the gates are on "
                         "bootstrap sigmas")
    ap.add_argument("--predict-height", type=float, default=0.270,
                    help="height at which the correction and its uncertainty "
                         "are reported, for comparison against parcel tops")
    ap.add_argument("--max-view-sigma-mm", type=float, default=5.0,
                    help="refuse to write when the one-sigma uncertainty of a "
                         "view's DELAYERING correction at --predict-height "
                         "exceeds this. This is the gate that matters: it is "
                         "on the part a per-view correction can actually reach")
    ap.add_argument("--max-shared-sigma-mm", type=float, default=40.0,
                    help="warn, but never refuse, when the shared correction "
                         "is this uncertain at --predict-height. The shared "
                         "term is a mean and is worth applying even when it "
                         "is poorly determined, provided the uncertainty is "
                         "carried downstream")
    ap.add_argument("--slope-sigma-ratio", type=float, default=1.0,
                    help="a view keeps its fitted slope only when the slope's "
                         "contribution at --predict-height exceeds this many "
                         "of its own bootstrap sigmas; otherwise the view is "
                         "corrected by an offset alone")
    ap.add_argument("--no-view-slope", dest="view_slope", action="store_false",
                    default=True,
                    help="never fit per-view slopes, only per-view offsets")

    ap.add_argument("--belt-exclude-px", type=int, default=60,
                    help="margin in depth-grid pixels kept clear around the "
                         "board when sampling the belt beside it. It must "
                         "clear the BLOCK the board stands on, not just the "
                         "board; at 4.3 mm per pixel, 60 is about 250 mm")
    ap.add_argument("--belt-tol-m", type=float, default=0.030,
                    help="half-window about the deck plane for selecting belt "
                         "pixels; wide enough and a 218 mm block's side faces "
                         "are counted as belt")
    ap.add_argument("--belt-edge-thresh", type=float, default=0.01,
                    help="relative depth gradient above which a belt pixel is "
                         "rejected, which removes block sides and the "
                         "smoothing around them")
    ap.add_argument("--belt-report", action="store_true",
                    help="measure the belt beside the board in every capture "
                         "and test whether it could anchor the shared drift "
                         "live; slow, since it reloads every depth array")
    ap.add_argument("--placement-report", action="store_true",
                    help="list every board with its height, its position in "
                         "the frame and its own median excess")
    ap.add_argument("--edge-profile", action="store_true",
                    help="report the raw depth excess in bins of distance "
                         "from the board border, which separates a smoothing "
                         "halo around a raised board from a genuine range "
                         "error. Run it with --mask-erode-px 0 --edge-thresh 0")
    ap.add_argument("--verbose-poses", action="store_true",
                    help="list every placement rather than a summary per "
                         "surface and camera")
    ap.add_argument("--force", action="store_true",
                    help="write the file even when the delayering gate fails")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def collect(args):
    """
    Walk every run, solve a board pose per camera per placement, and return
    the pooled per-pixel samples plus the geometry records.
    """
    runs = []
    for d in args.run:
        run_dir = Path(d)
        man, frames = load_manifest(run_dir)
        runs.append({"dir": run_dir, "manifest": man, "frames": frames})

    names = list(args.cameras) if args.cameras else list(
        runs[0]["manifest"]["cameras"])
    for r in runs:
        if list(r["manifest"]["cameras"]) != names and not args.cameras:
            raise SystemExit(f"{r['dir']} lists cameras "
                             f"{r['manifest']['cameras']}, not {names}")
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} is not among {names}")

    lens_tags = {r["manifest"].get("lens_id") for r in runs}
    if len(lens_tags) > 1:
        raise SystemExit(f"mixed lens tags across the runs: {sorted(lens_tags)}")
    lens = args.lens or (lens_tags.pop() if lens_tags else "")

    # ---- rig geometry, rebuilt exactly as the capture rectified it -------
    intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
    ext = load_extrinsics(args.calib_dir, names, args.reference)
    sizes = {intr[n]["size"] for n in names}
    if len(sizes) > 1:
        raise SystemExit(f"the cameras are calibrated at mixed sizes: {sizes}")
    w_raw, h_raw = sizes.pop()

    inf = runs[0]["manifest"].get("inference", {})
    rect_mode = inf.get("rect_mode", "common")
    undistort = bool(inf.get("undistort", True))
    plan = build_rect_plan(intr, names, (w_raw, h_raw), rect_mode, undistort)

    for r in runs:
        want = r["manifest"].get("rectified_sizes")
        if not want:
            continue
        for n in names:
            cw, ch = plan[n]["crop"][2], plan[n]["crop"][3]
            if [cw, ch] != list(want[n]):
                raise SystemExit(
                    f"{r['dir']}: {n} was captured at {want[n]} but the "
                    f"rectification plan rebuilt here gives [{cw}, {ch}]. The "
                    f"calibration or --rect-mode has changed since the "
                    f"capture; the pose would be solved through the wrong "
                    f"principal point.")

    Krect = {n: plan[n]["K"] for n in names}
    # Rectified images carry no distortion, so the pose is solved with zeros.
    # Passing the raw dist_coeffs here would correct a second time.
    dist = {n: (np.zeros(5) if undistort else intr[n]["dist"]) for n in names}

    print(f"[rig  ] lens {lens or 'untagged'}   cameras {names}   "
          f"reference {args.reference}")
    for n in names:
        print(f"[rect ] {n:<7} {plan[n]['crop'][2]}x{plan[n]['crop'][3]}  "
              f"fx={Krect[n][0, 0]:8.2f}  cx={Krect[n][0, 2]:7.2f}  "
              f"cy={Krect[n][1, 2]:7.2f}")

    # ---- board layout ---------------------------------------------------
    spec = dict(runs[0]["manifest"]["board"])
    if args.board_legacy == "auto":
        samples = []
        for r in runs:
            for fr in r["frames"][:2]:
                p = resolve_image(fr, args.reference, args.image_pattern)
                if p is None:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    samples.append((img, Krect[args.reference],
                                    dist[args.reference]))
        if not samples:
            raise SystemExit(
                "no rectified image found inside the frame directories. Tried "
                + ", ".join(IMAGE_PATTERNS) + "; pass --image-pattern.")
        legacy = choose_legacy(samples, spec)
        if legacy is None:
            legacy = bool(spec.get("legacy", False))
            print(f"[board] keeping the manifest's legacy={legacy}")
    else:
        legacy = args.board_legacy == "true"
    if legacy != bool(spec.get("legacy", legacy)):
        print(f"[board] manifest says legacy={spec.get('legacy')} but "
              f"legacy={legacy} scores better; using {legacy}")
    board = build_board(spec, legacy)
    detector = build_detector(board)

    # ---- walk the placements -------------------------------------------
    samples = {n: [] for n in names}          # per-pixel, pooled over runs
    poses = []                                # one record per camera per frame
    skipped = []

    for r in runs:
        label = r["manifest"]["label"]
        nominal = float(r["manifest"]["nominal_mm"])
        for fr in r["frames"]:
            for n in names:
                img_path = resolve_image(fr, n, args.image_pattern)
                if img_path is None:
                    skipped.append((label, fr.name, n, "no image"))
                    continue
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    skipped.append((label, fr.name, n, "unreadable image"))
                    continue
                if gray.shape[::-1] != (plan[n]["crop"][2], plan[n]["crop"][3]):
                    skipped.append((label, fr.name, n,
                                    f"image is {gray.shape[1]}x{gray.shape[0]},"
                                    f" not the rectified size"))
                    continue

                rec, err = solve_board_pose(gray, board, detector,
                                            Krect[n], dist[n])
                if rec is None:
                    skipped.append((label, fr.name, n, err))
                    continue
                if rec["rms_px"] > args.max_rms_px:
                    skipped.append((label, fr.name, n,
                                    f"rms {rec['rms_px']:.3f} px"))
                    continue
                amb = rec["ambiguity_ratio"]
                if amb is not None and amb < args.min_ambiguity_ratio:
                    skipped.append((label, fr.name, n,
                                    f"ambiguity ratio {amb:.2f}"))
                    continue

                rec.update({"label": label, "nominal_mm": nominal,
                            "frame": fr.name, "camera": n})
                poses.append(rec)

                # ---- the depth arrays, if this run carries them ----------
                dpath = fr / f"depth_{n}.npy"
                kpath = fr / f"K_{n}.npy"
                if not (dpath.exists() and kpath.exists()):
                    continue
                depth = np.load(dpath).astype(np.float64)
                K_d = np.load(kpath).astype(np.float64)

                mask, edge_dist = board_mask(depth.shape, K_d,
                                             rec["face_points_cam"],
                                             args.mask_erode_px)
                z_true = plane_target_depth(depth.shape, K_d,
                                            rec["normal_cam"], rec["d_cam"])
                ok = (mask & np.isfinite(depth) & (depth > 0)
                      & np.isfinite(z_true))
                if args.edge_thresh > 0:
                    gy, gx = np.gradient(depth)
                    grad = np.hypot(gx, gy) / np.maximum(depth, 1e-6)
                    ok &= grad <= args.edge_thresh
                ok &= np.abs(depth - z_true) <= args.max_residual_m
                idx = np.flatnonzero(ok.ravel())
                if len(idx) < 200:
                    skipped.append((label, fr.name, n,
                                    f"only {len(idx)} board pixels in depth"))
                    continue
                if len(idx) > args.max_points:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(idx, args.max_points, replace=False)
                vv, uu = np.unravel_index(idx, depth.shape)
                samples[n].append({
                    "label": label, "nominal_mm": nominal, "frame": fr.name,
                    "camera": n,
                    "z_meas": depth.ravel()[idx],
                    "z_true": z_true.ravel()[idx],
                    "u": uu.astype(np.float64), "v": vv.astype(np.float64),
                    "K": K_d,
                    "edge_px": edge_dist.ravel()[idx].astype(np.float64),
                    "shape": depth.shape,
                })

    return {"names": names, "lens": lens, "runs": runs, "intr": intr,
            "ext": ext, "plan": plan, "Krect": Krect, "board": board,
            "legacy": legacy, "poses": poses, "samples": samples,
            "skipped": skipped}


def deck_planes(ctx, deck_label):
    """
    A deck plane per camera, in that camera's OWN frame, from the deck run's
    board faces.

    Fitting it per camera rather than transforming one world plane through the
    extrinsics keeps every extrinsic error out of the correction. The physical
    plane is the same for all four, so heights measured perpendicular to it
    are the same physical quantity in every frame; any disagreement between
    the four fitted planes is a constant per camera and is absorbed by that
    view's own offset.
    """
    out = {}
    for n in ctx["names"]:
        pts = [p["face_points_cam"] for p in ctx["poses"]
               if p["camera"] == n and p["label"] == deck_label]
        if not pts:
            continue
        n_c, d_c, resid = fit_plane_robust(np.vstack(pts))
        if n_c is None:
            continue
        if d_c > 0:                     # camera at the origin must be above
            n_c, d_c = -n_c, -d_c
        out[n] = {"n_cam": n_c, "d_cam": float(d_c),
                  "rms_mm": float(np.sqrt(np.mean(resid ** 2))) * 1e3,
                  "n_placements": len(pts)}
    return out


def build_rows(ctx, planes):
    """
    One row per placement per camera: the median measured height and the
    median true height over the board footprint.

    A quarter of a million pixels on one board are not a quarter of a million
    measurements. They are one board, seen once. Reducing each placement to a
    single row is what keeps every sigma below honest.
    """
    rows = []
    for n in ctx["names"]:
        pl = planes.get(n)
        if pl is None:
            continue
        for c in ctx["samples"][n]:
            hm = heights_from_depth(c["z_meas"], c["u"], c["v"], c["K"],
                                    pl["n_cam"], pl["d_cam"])
            ht = heights_from_depth(c["z_true"], c["u"], c["v"], c["K"],
                                    pl["n_cam"], pl["d_cam"])
            rows.append({
                "camera": n, "label": c["label"], "frame": c["frame"],
                "key": (c["label"], c["frame"]),
                "h_meas": float(np.median(hm)),
                "h_true": float(np.median(ht)),
                "within_rms_mm": float(np.sqrt(np.mean(
                    ((hm - ht) - np.median(hm - ht)) ** 2))) * 1e3,
                "n": int(len(hm)),
                "u": float(np.median(c["u"])), "v": float(np.median(c["v"])),
            })
    return rows


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def geometry_report(ctx, args):
    """
    What the board says about the rig, in world coordinates, using the nominal
    heights only as something to be checked against.
    """
    names, ext = ctx["names"], ctx["ext"]
    thickness = args.board_thickness_mm / 1000.0

    by_label = {}
    for p in ctx["poses"]:
        by_label.setdefault(p["label"], []).append(p)

    world, centres_w = {}, {}
    for label, recs in by_label.items():
        world[label] = np.vstack(
            [to_world(r["face_points_cam"], ext[r["camera"]]["E"])
             for r in recs])
        centres_w[label] = np.vstack(
            [to_world(r["board_centre_cam"][None, :], ext[r["camera"]]["E"])
             for r in recs])

    labels = sorted(by_label, key=lambda k: by_label[k][0]["nominal_mm"])
    deck_label = labels[0]
    n_w, d_w, resid = fit_plane_robust(world[deck_label])
    if n_w is None:
        raise SystemExit("the deck group has too few corners to fit a plane")
    cam_c = np.stack([camera_centre(ext[n]["E"]) for n in names])
    if float(np.mean(cam_c @ n_w - d_w)) < 0:
        n_w, d_w = -n_w, -d_w          # height is positive towards the cameras

    tilts = [r["tilt_deg"] for r in ctx["poses"]]
    print(f"\n[deck ] '{deck_label}' face plane: normal "
          f"{np.round(n_w, 5)}  offset {d_w:+.5f} m")
    print(f"[deck ] pooled corner residual about it: rms "
          f"{np.sqrt(np.mean(resid ** 2)) * 1e3:.2f} mm  p95 "
          f"{np.percentile(np.abs(resid), 95) * 1e3:.2f} mm")
    if float(np.median(tilts)) > 5.0:
        print(f"[WARN] median placement tilt is {np.median(tilts):.1f} deg. "
              f"The pooled fit above assumes the boards LIE ON the surface, so "
              f"a propped board puts its own tilt into that residual and into "
              f"the deck normal.")
    cres = centres_w[deck_label] @ n_w - d_w
    print(f"[deck ] placement-to-placement scatter at the board centre: rms "
          f"{np.sqrt(np.mean(cres ** 2)) * 1e3:.2f} mm  p95 "
          f"{np.percentile(np.abs(cres), 95) * 1e3:.2f} mm. That is the board "
          f"rocking on the surface, and it is the error the correction cannot "
          f"see past.")

    heights = {}
    print("\n[valid] heights are OUTPUTS here. Nominal is never fed to the "
          "solve.")
    print(f"[valid] {'label':<8} {'n':>3}  {'block top above deck':>21}  "
          f"{'spread p95':>11}  {'nominal':>8}  {'error':>8}")
    for label in labels:
        h_face = (centres_w[label] @ n_w - d_w)
        med = float(np.median(h_face))
        spread = float(np.percentile(np.abs(h_face - med), 95))
        nominal = by_label[label][0]["nominal_mm"] / 1000.0
        err = (med - nominal) * 1e3
        heights[label] = {
            "n_placements": len(by_label[label]),
            "block_top_above_deck_mm": round(med * 1e3, 3),
            "spread_p95_mm": round(spread * 1e3, 3),
            "nominal_mm": by_label[label][0]["nominal_mm"],
            "error_mm": round(err, 3),
        }
        print(f"[valid] {label:<8} {len(by_label[label]):>3}  "
              f"{med * 1e3:>18.3f} mm  {spread * 1e3:>8.2f} mm  "
              f"{nominal * 1e3:>5.0f} mm  {err:>+6.2f} mm")
    print("[valid] the deck row is zero by construction, it defines the plane. "
          "The spread column is the board rocking on the surface, not error "
          "in the fit.")

    ref = args.reference if args.reference in ext else names[0]
    C = camera_centre(ext[ref]["E"])
    face_perp = float(C @ n_w - d_w)
    print(f"\n[stand] {ref} sits {face_perp * 1e3:.1f} mm above the deck-board "
          f"face, so the deck SURFACE is {(face_perp + thickness) * 1e3:.1f} mm "
          f"away perpendicular")

    print()
    cross = {}
    for label, recs in by_label.items():
        per_frame = {}
        for r in recs:
            per_frame.setdefault(r["frame"], []).append(r)
        ang, off = [], []
        for frame, group in per_frame.items():
            if len(group) < 2:
                continue
            ns = []
            for r in group:
                E = ext[r["camera"]]["E"]
                nw = E[:3, :3].T @ r["normal_cam"]
                ns.append(nw / np.linalg.norm(nw))
            ns = np.stack(ns)
            mean_n = ns.mean(axis=0)
            mean_n /= np.linalg.norm(mean_n)
            ang.append(float(np.degrees(np.max(np.arccos(
                np.clip(np.abs(ns @ mean_n), 0, 1))))))
            hs = [float(to_world(r["board_centre_cam"][None, :],
                                 ext[r["camera"]]["E"])[0] @ n_w - d_w)
                  for r in group]
            off.append((max(hs) - min(hs)) * 1e3)
        if ang:
            cross[label] = {"max_normal_disagreement_deg": round(max(ang), 4),
                            "max_height_disagreement_mm": round(max(off), 3)}
            print(f"[cross] {label:<8} views disagree by at most "
                  f"{max(ang):.3f} deg on the normal and {max(off):.2f} mm on "
                  f"the height of the same board")
    print("[cross] that is an extrinsics check. It is bounded below by the "
          "focal spread across the rig and says nothing about DA3.")

    return {"deck_label": deck_label,
            "deck_normal_world": n_w.tolist(),
            "deck_offset_m": float(d_w),
            "reference_camera": ref,
            "deck_face_perp_m": round(face_perp, 5),
            "deck_surface_perp_m": round(face_perp + thickness, 5),
            "deck_pooled_corner_rms_mm": round(
                float(np.sqrt(np.mean(resid ** 2))) * 1e3, 3),
            "deck_placement_scatter_rms_mm": round(
                float(np.sqrt(np.mean(cres ** 2))) * 1e3, 3),
            "median_placement_tilt_deg": round(float(np.median(tilts)), 3),
            "board_thickness_mm": args.board_thickness_mm,
            "heights": heights, "cross_camera": cross}, n_w, d_w


def anchor_report(ctx, labels):
    """
    The absolute range check. A scale error common to all four views is
    invisible to every inter-view analysis and shows up only here.
    """
    print()
    anchor = {}
    for n in ctx["names"]:
        per = {}
        for c in ctx["samples"][n]:
            per.setdefault(c["label"], []).append(c)
        for label in labels:
            cs = per.get(label)
            if not cs:
                continue
            zt = np.concatenate([c["z_true"] for c in cs])
            zm = np.concatenate([c["z_meas"] for c in cs])
            ratio = float(np.median(zm) / np.median(zt))
            anchor.setdefault(n, {})[label] = {
                "charuco_median_m": round(float(np.median(zt)), 4),
                "da3_median_m": round(float(np.median(zm)), 4),
                "excess_mm": round(float(np.median(zm) - np.median(zt)) * 1e3, 2),
                "scale_ratio": round(ratio, 5)}
            print(f"[anchr] {n:<7} {label:<8} ChArUco {np.median(zt):.4f} m   "
                  f"DA3 {np.median(zm):.4f} m   "
                  f"{(np.median(zm) - np.median(zt)) * 1e3:+7.1f} mm   "
                  f"x{ratio:.4f}")
    if anchor:
        allr = [v["scale_ratio"] for c in anchor.values() for v in c.values()]
        print(f"[anchr] common-mode scale across the rig "
              f"{np.mean(allr):.4f} +/- {np.std(allr):.4f}")
    return anchor


def layer_stats(rows, cams, labels, fit=None):
    """
    How far apart the four views place the same board, before and after.

    Per capture, the spread across cameras of the reconstructed height. That
    spread is invariant to the per-capture drift, because the drift is common
    to all four views and cancels in a maximum minus a minimum. It is
    therefore the exact quantity a per-view correction can move, and the only
    one it should be judged on.
    """
    out = {}
    for label in labels:
        pre, post = [], []
        keys = sorted({r["key"] for r in rows if r["label"] == label})
        for k in keys:
            grp = [r for r in rows if r["key"] == k]
            if len(grp) < 2:
                continue
            hb = [r["h_meas"] for r in grp]
            pre.append((max(hb) - min(hb)) * 1e3)
            if fit is not None:
                ha = [fit["alpha"][r["camera"]] * r["h_meas"]
                      + fit["beta"][r["camera"]] for r in grp]
                post.append((max(ha) - min(ha)) * 1e3)
        if not pre:
            continue
        out[label] = {
            "n_captures": len(pre),
            "before_median_mm": round(float(np.median(pre)), 2),
            "before_max_mm": round(float(np.max(pre)), 2),
            "after_median_mm": (round(float(np.median(post)), 2)
                                if post else None),
            "after_max_mm": (round(float(np.max(post)), 2) if post else None),
        }
    return out


def print_layer_stats(stats, corrected):
    tag = "after" if corrected else "before"
    print(f"\n[layer] spread across the four views on the same board, the "
          f"quantity a per-view correction can actually move")
    print(f"[layer] {'label':<8} {'captures':>9} {'before med':>12} "
          f"{'before max':>12}" + (f" {'after med':>11} {'after max':>11}"
                                   if corrected else ""))
    for label, v in stats.items():
        line = (f"[layer] {label:<8} {v['n_captures']:>9} "
                f"{v['before_median_mm']:>9.1f} mm {v['before_max_mm']:>9.1f} mm")
        if corrected and v["after_median_mm"] is not None:
            line += (f" {v['after_median_mm']:>8.1f} mm "
                     f"{v['after_max_mm']:>8.1f} mm")
        print(line)
    if not corrected:
        print(f"[layer] ({tag} correction)")


def common_mode_report(rows, labels):
    """
    Split the placement scatter into the part shared by all views within one
    capture and the part that differs between them.

    DA3 is run once per capture over all four frames together, with shared
    intrinsics and extrinsics, so the four depth maps are one joint solution
    rather than four independent measurements. An error in that solution
    appears in every view at once. If the shared part dominates then no
    inter-view check can see it, agreement between views is not evidence, and
    no per-view correction can remove it.
    """
    print("\n[common] height error shared by all views within a capture, "
          "against the part that differs between them")
    print(f"[common] {'label':<8} {'captures':>9} {'shared sd':>11} "
          f"{'per-view sd':>13} {'shared':>8}")
    out = {}
    for label in labels:
        per_key = {}
        for r in rows:
            if r["label"] != label:
                continue
            per_key.setdefault(r["key"], {})[r["camera"]] = (
                (r["h_meas"] - r["h_true"]) * 1e3)
        grp = [(k, v) for k, v in per_key.items() if len(v) >= 3]
        if len(grp) < 3:
            continue
        means = np.array([float(np.mean(list(v.values()))) for _, v in grp])
        resid = np.concatenate([np.array(list(v.values())) - m
                                for (_, v), m in zip(grp, means)])
        total = np.concatenate([np.array(list(v.values())) for _, v in grp])
        shared = (1.0 - resid.var() / total.var()) if total.var() > 0 else 0.0
        out[label] = {
            "n_captures": len(grp),
            "shared_sd_mm": round(float(means.std(ddof=1)), 2),
            "per_view_sd_mm": round(float(resid.std(ddof=1)), 2),
            "shared_fraction": round(float(shared), 3),
            "per_capture_mean_mm": {str(k[1]): round(float(m), 2)
                                    for (k, _), m in zip(grp, means)},
        }
        print(f"[common] {label:<8} {len(grp):>9} "
              f"{means.std(ddof=1):>8.1f} mm {resid.std(ddof=1):>10.1f} mm "
              f"{shared * 100:>7.0f}%")
        print("[common]   per-capture mean: " + "  ".join(
            f"{k[1].replace('frame_000', '')}:{m:+6.1f}"
            for (k, _), m in zip(grp, means)))
    print("[common] the shared column is what the capture fixed effect absorbs "
          "in stage B and what remains as absolute uncertainty afterwards. The "
          "per-view column is what stage B corrects.")
    return out


def relief_report(ctx, args, n_w, d_w, rows):
    """
    The loss of relief per camera, and whether it varies across the belt.

    Reported in the same currency as the fit: a rise of h is reconstructed as
    h(1 - k). This is a diagnostic rather than the solve, but it is the
    quantity the solve is built on and it belongs in the record.
    """
    poses = {(p["label"], p["frame"], p["camera"]): p for p in ctx["poses"]}
    out, rows_all = {}, []
    for n in ctx["names"]:
        sub = []
        for r in rows:
            if r["camera"] != n:
                continue
            p = poses.get((r["label"], r["frame"], n))
            if p is None:
                continue
            cw = to_world(p["board_centre_cam"][None, :], ctx["ext"][n]["E"])[0]
            sub.append({**r,
                        "height_mm": r["h_true"] * 1e3,
                        "x_m": float(cw[0]), "y_m": float(cw[1]),
                        "deficit_mm": (r["h_true"] - r["h_meas"]) * 1e3})
        if len(sub) < 3:
            continue
        h = np.array([s["height_mm"] for s in sub])
        m = np.array([s["deficit_mm"] for s in sub])
        k, c0 = np.polyfit(h, m, 1)
        between = float(np.sqrt(np.mean((m - (k * h + c0)) ** 2)))
        within = float(np.median([s["within_rms_mm"] for s in sub]))

        x = np.array([s["x_m"] for s in sub])
        y = np.array([s["y_m"] for s in sub])
        field_rms, coef = None, None
        if len(sub) >= 8:
            A = np.stack([np.ones_like(h), h, h * x, h * y], axis=1)
            coef, *_ = np.linalg.lstsq(A, m, rcond=None)
            field_rms = float(np.sqrt(np.mean((m - A @ coef) ** 2)))
        out[n] = {"compression": round(float(k), 5),
                  "offset_at_deck_mm": round(float(c0), 3),
                  "between_placement_rms_mm": round(between, 3),
                  "within_placement_rms_mm": round(within, 3),
                  "field_residual_rms_mm": (None if field_rms is None
                                            else round(field_rms, 3)),
                  "field_coef": (None if coef is None
                                 else [round(float(v), 6) for v in coef]),
                  "n_placements": len(sub)}
        rows_all.extend(sub)

    print("\n[relief] a rise of h is reconstructed as h(1-k). k is fitted over "
          "placements, never assumed.")
    print(f"[relief] {'camera':<7} {'k':>7} {'at deck':>9} "
          f"{'270mm reads':>12} {'within board':>13} {'between boards':>15}")
    for n, r in out.items():
        print(f"[relief] {n:<7} {r['compression']:7.4f} "
              f"{r['offset_at_deck_mm']:+8.1f} mm "
              f"{270 * (1 - r['compression']):9.0f} mm "
              f"{r['within_placement_rms_mm']:10.1f} mm "
              f"{r['between_placement_rms_mm']:12.1f} mm")
    if any(r.get("field_residual_rms_mm") for r in out.values()):
        print("\n[field ] allowing the compression a linear gradient across "
              "the belt: deficit = c0 + h (k0 + kx x + ky y)")
        print(f"[field ] {'camera':<7} {'k0':>8} {'kx /m':>9} {'ky /m':>9} "
              f"{'between boards':>16} {'after the field':>17}")
        for n, r in out.items():
            if not r.get("field_coef"):
                continue
            c0f, k0, kx, ky = r["field_coef"]
            print(f"[field ] {n:<7} {k0:8.4f} {kx:9.4f} {ky:9.4f} "
                  f"{r['between_placement_rms_mm']:13.1f} mm "
                  f"{r['field_residual_rms_mm']:14.1f} mm")
        print("[field ] if the last column collapses towards the within-board "
              "figure, the error is a deterministic function of where the "
              "parcel sits. If it does not, the scatter is not positional and "
              "the capture fixed effect is the right home for it.")

    if args.placement_report:
        print(f"\n[place] {'camera':<7} {'label':<8} {'frame':<12} "
              f"{'height':>9} {'u':>6} {'v':>6} {'deficit':>9} "
              f"{'within':>8} {'n':>7}")
        for r in sorted(rows_all, key=lambda x: (x["camera"], x["height_mm"])):
            print(f"[place] {r['camera']:<7} {r['label']:<8} {r['frame']:<12} "
                  f"{r['height_mm']:8.1f} {r['u']:6.0f} {r['v']:6.0f} "
                  f"{r['deficit_mm']:+8.1f} {r['within_rms_mm']:7.1f} "
                  f"{r['n']:7d}")
    return out


def edge_profile(ctx, labels):
    """
    Raw depth excess as a function of distance from the board's edge.

    A depth model does not reproduce a step, it smooths across it. A board
    raised above the deck therefore carries a halo of intermediate depths
    running inward from its border. If the profile falls and then flattens,
    the plateau is the number to believe and --mask-erode-px should be set
    past the knee. If it never flattens, the error is not an edge effect.
    """
    edges = [0, 4, 8, 12, 16, 24, 1e9]
    print("\n[edge ] median depth excess by distance from the board border, "
          "before any correction")
    print(f"[edge ] {'camera':<7} {'label':<8} " + " ".join(
        f"{f'{lo}-{hi}px':>9}" for lo, hi in zip(edges[:-2], edges[1:-1]))
        + f"{'>24px':>9}")
    out = {}
    for n in ctx["names"]:
        for label in labels:
            cs = [c for c in ctx["samples"][n] if c["label"] == label]
            if not cs:
                continue
            d = np.concatenate([c["edge_px"] for c in cs])
            r = (np.concatenate([c["z_meas"] for c in cs])
                 - np.concatenate([c["z_true"] for c in cs])) * 1e3
            row, cells = [], {}
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (d >= lo) & (d < hi)
                if sel.sum() < 50:
                    row.append("        -")
                    continue
                med = float(np.median(r[sel]))
                cells[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = round(med, 2)
                row.append(f"{med:8.1f} ")
            out.setdefault(n, {})[label] = cells
            print(f"[edge ] {n:<7} {label:<8} " + "".join(row))
    print("[edge ] falling then flat means a halo, and the flat value is the "
          "real excess. Flat throughout means the range error is genuine.")
    return out


def belt_report(ctx, args, n_w, d_w):
    """
    Measure the belt itself in every capture, beside the board, and ask
    whether the per-capture drift seen on the board is visible there too.

    This is the question that decides whether the shared term can be removed
    at run time. It is shared by all four views and is not a function of where
    the board sits, so no static correction reaches it. But the belt is in
    frame on every live capture. If the same capture that puts the board 90 mm
    out also puts the belt out by a related amount, then the belt is a
    per-frame reference. If the belt reads clean while the board does not, the
    error lives only in relief and nothing in the frame reveals it.
    """
    poses = {(p["label"], p["frame"], p["camera"]): p for p in ctx["poses"]}
    board_excess = {}
    for n in ctx["names"]:
        for c in ctx["samples"][n]:
            board_excess[(c["label"], c["frame"], n)] = (
                float(np.median((c["z_meas"] - c["z_true"]) * 1e3)),
                float(np.median(c["z_true"])))

    rows = []
    for r in ctx["runs"]:
        label = r["manifest"]["label"]
        for fr in r["frames"]:
            for n in ctx["names"]:
                p = poses.get((label, fr.name, n))
                if p is None or (label, fr.name, n) not in board_excess:
                    continue
                dpath, kpath = fr / f"depth_{n}.npy", fr / f"K_{n}.npy"
                if not (dpath.exists() and kpath.exists()):
                    continue
                depth = np.load(dpath).astype(np.float64)
                K = np.load(kpath).astype(np.float64)

                E = ctx["ext"][n]["E"]
                R, t = E[:3, :3], E[:3, 3]
                n_c = R @ n_w
                d_c = d_w + float(n_c @ t)
                z_deck = plane_target_depth(depth.shape, K, n_c, d_c)
                if not np.isfinite(z_deck).any():
                    n_c, d_c = -n_c, -d_c
                    z_deck = plane_target_depth(depth.shape, K, n_c, d_c)

                raw, _ = board_mask(depth.shape, K, p["face_points_cam"], 0)
                k = np.ones((2 * args.belt_exclude_px + 1,) * 2, np.uint8)
                near_board = cv2.dilate(raw.astype(np.uint8), k).astype(bool)

                ok = (~near_board & np.isfinite(depth) & (depth > 0)
                      & np.isfinite(z_deck)
                      & (np.abs(depth - z_deck) <= args.belt_tol_m))
                gy, gx = np.gradient(depth)
                grad = np.hypot(gx, gy) / np.maximum(depth, 1e-6)
                ok &= grad <= args.belt_edge_thresh
                if ok.sum() < 500:
                    continue
                be = float(np.median((depth[ok] - z_deck[ok]) * 1e3))
                bz = float(np.median(z_deck[ok]))
                me, mz = board_excess[(label, fr.name, n)]
                rows.append({"label": label, "frame": fr.name, "camera": n,
                             "belt_mm": be, "belt_z": bz,
                             "belt_n": int(ok.sum()),
                             "board_mm": me, "board_z": mz})

    if len(rows) < 6:
        print("\n[belt  ] not enough belt visible beside the board to compare")
        return None

    print("\n[belt  ] the belt measured in the same capture as the board")
    labels_sorted = sorted({r["label"] for r in rows})
    print(f"[belt  ] {'camera':<7} " + " ".join(
        f"{lb:>18}" for lb in labels_sorted))
    consistency = {}
    for n in ctx["names"]:
        cells = []
        for lb in labels_sorted:
            sub = [r for r in rows if r["label"] == lb and r["camera"] == n]
            if not sub:
                cells.append(f"{'-':>18}")
                continue
            v = np.array([r["belt_mm"] for r in sub])
            px = int(np.median([r["belt_n"] for r in sub]))
            consistency.setdefault(n, {})[lb] = {
                "median_mm": round(float(np.median(v)), 2),
                "sd_mm": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 2),
                "median_pixels": px}
            cells.append(f"{np.median(v):+8.1f}+-{v.std(ddof=1):<5.1f}mm"
                         if len(v) > 1 else f"{np.median(v):+8.1f}mm      ")
        print(f"[belt  ] {n:<7} " + " ".join(cells))
    print("[belt  ] the belt never moved between runs, so a run that "
          "disagrees is not measuring the belt. Widen --belt-exclude-px.")

    print(f"\n[belt  ] {'label':<8} {'camera':<7} {'n':>3} {'belt sd':>9} "
          f"{'board sd':>10} {'r':>7} {'offset':>9} {'scale':>8}")
    out = {}
    for label in labels_sorted:
        for n in ctx["names"]:
            sub = [r for r in rows if r["label"] == label and r["camera"] == n]
            if len(sub) < 4:
                continue
            belt = np.array([r["belt_mm"] for r in sub])
            brd = np.array([r["board_mm"] for r in sub])
            ratio = np.array([r["board_z"] / r["belt_z"] for r in sub])
            corr = (float(np.corrcoef(belt, brd)[0, 1])
                    if belt.std() > 1e-9 and brd.std() > 1e-9 else float("nan"))
            off = float((brd - belt).std(ddof=1))
            sca = float((brd - belt * ratio).std(ddof=1))
            out.setdefault(label, {})[n] = {
                "n": len(sub),
                "belt_sd_mm": round(float(belt.std(ddof=1)), 2),
                "board_sd_mm": round(float(brd.std(ddof=1)), 2),
                "correlation": None if np.isnan(corr) else round(corr, 3),
                "after_offset_anchor_sd_mm": round(off, 2),
                "after_scale_anchor_sd_mm": round(sca, 2)}
            print(f"[belt  ] {label:<8} {n:<7} {len(sub):>3} "
                  f"{belt.std(ddof=1):>6.1f} mm {brd.std(ddof=1):>7.1f} mm "
                  f"{corr:>7.3f} {off:>6.1f} mm {sca:>5.1f} mm")
    print("[belt  ] the last two columns are what a per-frame anchor on the "
          "belt would leave. Below the board column, the belt is worth reading "
          "live. At or above it, the drift cannot be anchored away.")
    out["_consistency"] = consistency
    return out


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not args.run:
        raise SystemExit("--run is required; pass one capture run per "
                         "reference height, including the deck")

    ctx = collect(args)
    names = ctx["names"]

    if ctx["skipped"]:
        print(f"\n[skip ] {len(ctx['skipped'])} placement(s) excluded")
        for label, frame, cam, why in ctx["skipped"][:20]:
            print(f"[skip ] {label:<8} {frame} {cam:<7} {why}")
        if len(ctx["skipped"]) > 20:
            print(f"[skip ] ... and {len(ctx['skipped']) - 20} more")
    if not ctx["poses"]:
        raise SystemExit("no board pose survived the gates")

    if args.verbose_poses:
        print(f"\n[pose ] {'label':<8} {'frame':<12} {'camera':<7} {'n':>3} "
              f"{'rms':>6} {'amb':>7} {'tilt':>7} {'lateral':>8} {'range':>7}")
        for p in ctx["poses"]:
            amb = ("n/a" if p["ambiguity_ratio"] is None
                   else f"{p['ambiguity_ratio']:.1f}")
            print(f"[pose ] {p['label']:<8} {p['frame']:<12} {p['camera']:<7} "
                  f"{p['n_corners']:>3} {p['rms_px']:6.3f} {amb:>7} "
                  f"{p['tilt_deg']:6.2f}d {p['lateral_offset_m']:7.3f}m "
                  f"{p['range_m']:6.3f}m")
    else:
        print(f"\n[pose ] {'label':<8} {'camera':<7} {'n':>3} {'corners':>8} "
              f"{'rms px':>14} {'worst amb':>10} {'tilt deg':>16} "
              f"{'lateral m':>16}")
        groups = {}
        for p in ctx["poses"]:
            groups.setdefault((p["label"], p["camera"]), []).append(p)
        for (label, cam), ps in groups.items():
            amb = [p["ambiguity_ratio"] for p in ps
                   if p["ambiguity_ratio"] is not None]
            t = [p["tilt_deg"] for p in ps]
            L = [p["lateral_offset_m"] for p in ps]
            r = [p["rms_px"] for p in ps]
            print(f"[pose ] {label:<8} {cam:<7} {len(ps):>3} "
                  f"{int(np.median([p['n_corners'] for p in ps])):>8} "
                  f"{np.median(r):>7.3f} (max {max(r):.3f}) "
                  f"{min(amb) if amb else float('nan'):>10.1f} "
                  f"{min(t):>7.2f} to {max(t):<6.2f} "
                  f"{min(L):>7.3f} to {max(L):<6.3f}")
        print("[pose ] --verbose-poses lists every placement individually")
    tilts = [p["tilt_deg"] for p in ctx["poses"]]
    if args.min_tilt_deg > 0 and min(tilts) < args.min_tilt_deg:
        print(f"[WARN] the flattest placement is {min(tilts):.2f} deg. Near "
              f"fronto-parallel is where the standoff is least constrained.")

    geom, n_w, d_w = geometry_report(ctx, args)

    labels = sorted({p["label"] for p in ctx["poses"]},
                    key=lambda L: [p["nominal_mm"] for p in ctx["poses"]
                                   if p["label"] == L][0])
    n_heights = len(labels)

    anchor = anchor_report(ctx, labels)

    # ---- height axis, per camera, extrinsic-free ------------------------
    planes = deck_planes(ctx, geom["deck_label"])
    if not planes:
        raise SystemExit("no camera has enough deck placements to define a "
                         "height axis")
    print("\n[plane] deck plane per camera, in that camera's own frame. "
          "Heights are measured perpendicular to it, so no extrinsic enters "
          "the correction.")
    for n in names:
        if n not in planes:
            print(f"[plane] {n:<7} no deck placements, this view cannot be "
                  f"corrected")
            continue
        p = planes[n]
        print(f"[plane] {n:<7} normal {np.round(p['n_cam'], 5)}  standoff "
              f"{-p['d_cam']:.4f} m  corner rms {p['rms_mm']:.2f} mm  from "
              f"{p['n_placements']} placements")

    rows = build_rows(ctx, planes)
    if not rows:
        raise SystemExit("no view could be fitted; none of the runs carried "
                         "depth arrays inside their frame directories")
    cams = [n for n in names if any(r["camera"] == n for r in rows)]

    before = layer_stats(rows, cams, labels)
    print_layer_stats(before, corrected=False)

    relief = relief_report(ctx, args, n_w, d_w, rows)
    common = common_mode_report(rows, labels)
    profile = edge_profile(ctx, labels) if args.edge_profile else None
    belt = belt_report(ctx, args, n_w, d_w) if args.belt_report else None

    if n_heights < 2:
        raise SystemExit("a single reference height cannot separate a "
                         "compression from an offset. Capture at least two, "
                         "and three to make the shared leave-one-out a real "
                         "test.")

    # ---- stage A and stage B -------------------------------------------
    huber = args.huber or None
    H = args.predict_height
    slope_cams = list(cams) if args.view_slope else []
    fit = solve_effects(rows, cams, slope_cams, huber)
    p_bar, q_bar = fit["p_bar"], fit["q_bar"]
    if p_bar < 0.2:
        raise SystemExit(f"the fitted surviving fraction of relief is "
                         f"{p_bar:.4f}. Either the height axis is wrong or the "
                         f"depth maps are not measuring the boards at all; "
                         f"check the [plane] standoffs against the rule.")
    alpha_bar, beta_bar = 1.0 / p_bar, -q_bar / p_bar
    scale = 1.0 / p_bar          # measured-height mm to corrected-height mm

    shared_deck = -q_bar * 1e3
    shared_high = (H - (p_bar * H + q_bar)) * 1e3
    print(f"\n[shared] p={p_bar:.6f}  q={q_bar * 1e3:+.2f} mm, so the rig "
          f"keeps {p_bar * 100:.1f} per cent of every rise and the mean "
          f"compression is k={1 - p_bar:.4f}")
    print(f"[shared] as a correction, alpha={alpha_bar:.6f} "
          f"beta={beta_bar * 1e3:+.2f} mm: it moves a deck point by "
          f"{shared_deck:+.1f} mm and a {H * 1e3:.0f} mm point by "
          f"{shared_high:+.1f} mm")

    loo_shared = {}
    if n_heights >= 3:
        for held in labels:
            tr = [r for r in rows if r["label"] != held]
            te = [r for r in rows if r["label"] == held]
            if not tr or not te:
                continue
            p_l, q_l = irls_line([r["h_true"] for r in tr],
                                 [r["h_meas"] for r in tr], huber)
            err = np.array([((r["h_meas"] - q_l) / p_l - r["h_true"]) * 1e3
                            for r in te])
            loo_shared[held] = {"p": p_l, "q_mm": round(q_l * 1e3, 3),
                                "median_mm": round(float(np.median(err)), 2),
                                "rms_mm": round(
                                    float(np.sqrt(np.mean(err ** 2))), 2)}
            print(f"[loo   ] shared, held out {held:<8} p={p_l:.6f} "
                  f"q={q_l * 1e3:+6.1f} mm  ->  corrected height off by "
                  f"{np.median(err):+7.2f} mm median, rms "
                  f"{np.sqrt(np.mean(err ** 2)):6.2f} mm")
        print("[loo   ] this tests the SHARED term only. Its error is "
              "dominated by the per-capture drift, not by the fit.")

    boot = (bootstrap_effects(rows, cams, slope_cams, huber, args.bootstrap)
            if args.bootstrap > 0 else None)

    if boot and slope_cams:
        print()
        keep = []
        for c in slope_cams:
            s = sd(boot["a"][c]) * H * 1e3 * scale
            v = abs(fit["a"][c]) * H * 1e3 * scale
            resolved = (np.isfinite(s) and s > 0
                        and v >= args.slope_sigma_ratio * s)
            print(f"[slope ] {c:<7} contributes {v:5.1f} mm at "
                  f"{H * 1e3:.0f} mm against its own sigma {s:5.1f} mm  "
                  f"{'kept' if resolved else 'dropped, offset only'}")
            if resolved:
                keep.append(c)
        if len(keep) < 2:
            print("[slope ] fewer than two views resolve a slope, so the "
                  "gauge sum_v a_v = 0 would force the rest to zero anyway. "
                  "Correcting every view by an offset alone.")
            keep = []
        if keep != slope_cams:
            slope_cams = keep
            fit = solve_effects(rows, cams, slope_cams, huber)
            boot = bootstrap_effects(rows, cams, slope_cams, huber,
                                     args.bootstrap)
            p_bar, q_bar = fit["p_bar"], fit["q_bar"]
            alpha_bar, beta_bar = 1.0 / p_bar, -q_bar / p_bar
            scale = 1.0 / p_bar

    p_v = {c: p_bar + fit["a"][c] for c in cams}
    q_v = {c: q_bar + fit["b"][c] for c in cams}
    bad = [c for c in cams if p_v[c] < 0.2]
    if bad:
        raise SystemExit(f"views {bad} fitted a surviving relief fraction "
                         f"below 0.2, which is not a correction but a "
                         f"singularity. Re-run with --no-view-slope.")
    fit["alpha"] = {c: 1.0 / p_v[c] for c in cams}
    fit["beta"] = {c: -q_v[c] / p_v[c] for c in cams}

    sig_shared_high = (float(np.std(
        [H - (p * H + q) for p, q in zip(boot["p_bar"], boot["q_bar"])],
        ddof=1)) * 1e3 if boot and len(boot["p_bar"]) > 2 else None)

    print(f"\n[view  ] per-view deviation from the rig mean, in millimetres of "
          f"reconstructed height. This is the delayering term, and the only "
          f"part a per-view correction reaches.")
    print(f"[view  ] {'camera':<7} {'p_v':>9} {'q_v mm':>9} "
          f"{'at deck':>9} {'at ' + str(int(H * 1e3)) + ' mm':>11} "
          f"{'sigma':>8} {'slope':>7}")
    view_out, gate_failed = {}, []
    for c in cams:
        d_deck = fit["b"][c] * 1e3 * scale
        d_high = (fit["a"][c] * H + fit["b"][c]) * 1e3 * scale
        s_high = None
        if boot and len(boot["b"][c]) > 2:
            s_high = float(np.std(
                [a * H + b for a, b in zip(boot["a"][c], boot["b"][c])],
                ddof=1)) * 1e3 * scale
        view_out[c] = {
            "p_v": p_v[c], "q_v": q_v[c],
            "alpha": fit["alpha"][c], "beta": fit["beta"][c],
            "a_v": fit["a"][c], "b_v": fit["b"][c],
            "slope_fitted": c in slope_cams,
            "n_cam": planes[c]["n_cam"].tolist(),
            "d_cam": planes[c]["d_cam"],
            "deck_plane_rms_mm": round(planes[c]["rms_mm"], 3),
            "delayering_at_deck_mm": round(d_deck, 3),
            "delayering_at_predict_height_mm": round(d_high, 3),
            "delayering_sigma_mm": (None if s_high is None
                                    else round(s_high, 3)),
            "a_v_sigma": (round(sd(boot["a"][c]), 6) if boot else None),
            "b_v_sigma_mm": (round(sd(boot["b"][c]) * 1e3, 3)
                             if boot else None),
            "n_placements": sum(1 for r in rows if r["camera"] == c),
        }
        print(f"[view  ] {c:<7} {p_v[c]:>9.5f} {q_v[c] * 1e3:>+8.2f} "
              f"{d_deck:>+8.2f} {d_high:>+10.2f} "
              + (f"{s_high:>7.2f}" if s_high is not None else f"{'n/a':>7}")
              + f" {'yes' if c in slope_cams else 'no':>7}")
        if s_high is not None and s_high > args.max_view_sigma_mm:
            gate_failed.append(
                f"{c}: the delayering correction at {H * 1e3:.0f} mm carries "
                f"a one-sigma of {s_high:.2f} mm, above --max-view-sigma-mm "
                f"{args.max_view_sigma_mm:.1f}")

    # ---- what it does to the layering ----------------------------------
    after = layer_stats(rows, cams, labels, fit)
    print_layer_stats(after, corrected=True)
    print("[layer] if the after columns are not clearly smaller than the "
          "before columns, the per-view term is not the problem on this rig "
          "and applying it will not help.")

    # ---- residual per plane, per view ----------------------------------
    print(f"\n[resid] height error per view, before and after, in mm. The "
          f"'with capture' column removes the drift the correction cannot "
          f"know about and is the floor a live system could reach with a "
          f"perfect per-frame anchor.")
    print(f"[resid] {'camera':<7} {'label':<8} {'before':>9} {'after':>9} "
          f"{'with capture':>14}")
    resid_out = {}
    for c in cams:
        for label in labels:
            sub = [r for r in rows if r["camera"] == c and r["label"] == label]
            if not sub:
                continue
            b0 = np.array([(r["h_meas"] - r["h_true"]) * 1e3 for r in sub])
            a1 = np.array([(fit["alpha"][c] * r["h_meas"] + fit["beta"][c]
                            - r["h_true"]) * 1e3 for r in sub])
            # The capture effect lives in MEASURED height, so it is subtracted
            # before the inverse map, not added after it. Adding it afterwards
            # doubles the drift instead of removing it.
            a2 = np.array([(fit["alpha"][c]
                            * (r["h_meas"] - fit["e"].get(r["key"], 0.0))
                            + fit["beta"][c] - r["h_true"]) * 1e3
                           for r in sub])
            resid_out.setdefault(c, {})[label] = {
                "n": len(sub),
                "before_rms_mm": round(float(np.sqrt(np.mean(b0 ** 2))), 2),
                "after_rms_mm": round(float(np.sqrt(np.mean(a1 ** 2))), 2),
                "after_with_capture_rms_mm": round(
                    float(np.sqrt(np.mean(a2 ** 2))), 2)}
            print(f"[resid] {c:<7} {label:<8} "
                  f"{np.sqrt(np.mean(b0 ** 2)):>6.2f} mm "
                  f"{np.sqrt(np.mean(a1 ** 2)):>6.2f} mm "
                  f"{np.sqrt(np.mean(a2 ** 2)):>11.2f} mm")

    # ---- the uncertainty the correction does not remove -----------------
    uncertainty = None
    pts = []
    for label, v in common.items():
        h_lab = float(np.median([r["h_true"] for r in rows
                                 if r["label"] == label])) * 1e3
        # Views are not independent, so averaging four of them reduces only
        # the per-view part, and by about two rather than by four. The scatter
        # is measured in the depth model's own compressed heights, so it is
        # divided by p to express it in the corrected heights the pipeline
        # will report.
        pts.append((h_lab, float(np.hypot(v["shared_sd_mm"],
                                          v["per_view_sd_mm"] / 2.0)) * scale))
    if len(pts) >= 2:
        h = np.array([p[0] for p in pts])
        s = np.array([p[1] for p in pts])
        s1, s0 = np.polyfit(h, s, 1)
        uncertainty = {
            "model": "sigma_mm = constant + per_mm_of_height * height_mm",
            "constant_mm": round(float(s0), 3),
            "per_mm_of_height": round(float(s1), 5),
            "measured": {f"{int(a)}": round(float(b), 2) for a, b in pts},
            "at_predict_height_mm": round(float(s0 + s1 * H * 1e3), 2),
            "shared_correction_sigma_mm": (None if sig_shared_high is None
                                           else round(sig_shared_high, 2)),
            "note": ("one sigma on a reconstructed absolute height after this "
                     "correction. It is dominated by the term shared across "
                     "all four views, so it is NOT reduced by view consensus "
                     "and must not be estimated from inter-view agreement. "
                     "The layering, which IS reduced, is reported separately "
                     "under layering."),
        }
        print(f"\n[unc  ] absolute height uncertainty after correction: "
              f"{s0:.1f} mm + {s1 * 1000:.1f} mm per metre of height")
        for a, b in pts:
            print(f"[unc  ]   at {int(a):>3} mm: {b:.1f} mm")
        print(f"[unc  ] at {H * 1e3:.0f} mm: {s0 + s1 * H * 1e3:.1f} mm. Carry "
              f"this into height_uncertainty_mm downstream.")
    if sig_shared_high is not None:
        flag = ("" if sig_shared_high <= args.max_shared_sigma_mm
                else "   ABOVE --max-shared-sigma-mm, informational only")
        print(f"[unc  ] the shared correction itself is uncertain by "
              f"{sig_shared_high:.1f} mm at {H * 1e3:.0f} mm{flag}")

    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lens_id": ctx["lens"],
        "model": "deck_height_linear",
        "formula": ("h = z * (n_cam . r) - d_cam ; "
                    "h' = alpha * h + beta ; "
                    "z' = (h' + d_cam) / (n_cam . r)"),
        "reference": "per-capture ChArUco board pose",
        "runs": [{"dir": str(r["dir"]), "label": r["manifest"]["label"],
                  "nominal_mm": r["manifest"]["nominal_mm"],
                  "frames": len(r["frames"])} for r in ctx["runs"]],
        "board": {**ctx["runs"][0]["manifest"]["board"],
                  "legacy_used": ctx["legacy"]},
        "board_thickness_applied_to_fit": False,
        "rectified_sizes": {n: [ctx["plan"][n]["crop"][2],
                                ctx["plan"][n]["crop"][3]] for n in names},
        "depth_shape": [int(ctx["samples"][cams[0]][0]["shape"][0]),
                        int(ctx["samples"][cams[0]][0]["shape"][1])],
        "predict_height_m": H,
        "shared": {"p": p_bar, "q": q_bar,
                   "alpha": alpha_bar, "beta": beta_bar,
                   "compression": 1.0 - p_bar,
                   "at_deck_mm": round(shared_deck, 3),
                   "at_predict_height_mm": round(shared_high, 3),
                   "sigma_at_predict_height_mm": (
                       None if sig_shared_high is None
                       else round(sig_shared_high, 3)),
                   "p_sigma": (round(sd(boot["p_bar"]), 6)
                               if boot else None),
                   "q_sigma_mm": (round(sd(boot["q_bar"]) * 1e3, 3)
                                  if boot else None),
                   "leave_one_height_out": loo_shared},
        "geometry": geom,
        "anchor": anchor,
        "relief": relief,
        "common_mode": common,
        "edge_profile": profile,
        "belt_anchor": belt,
        "labels": labels,
        "layering": {"before": before, "after": after},
        "residuals": resid_out,
        # In millimetres of MEASURED height, one per capture, mean zero. These
        # are diagnostic only. They are never applied, because nothing at run
        # time reveals which one a live capture has drawn.
        "capture_effects_mm": {f"{k[0]}/{k[1]}": round(v * 1e3, 2)
                               for k, v in fit["e"].items()},
        "height_uncertainty": uncertainty,
        "bootstrap_draws": (len(boot["p_bar"]) if boot else 0),
        "views": view_out,
    }

    if gate_failed:
        print()
        for g in gate_failed:
            print(f"[GATE ] {g}")

    if gate_failed and not args.force:
        raise SystemExit(
            "refusing to write. The per-view part of the correction is not "
            "resolved, and an unresolved per-view term moves the four sheets "
            "apart rather than together. The levers are more captures, which "
            "shrinks these sigmas as the square root, and placements spread "
            "further across the frame. The SHARED term is fine and is not "
            "what failed here; if you want it alone, re-run with "
            "--no-view-slope, or with --force, which writes both and leaves "
            "the per-view part in place.")

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.out}")
    if gate_failed:
        print("written past a failed gate. Watch the [layer] after columns in "
              "the live stream; if the spread does not fall, remove the "
              "per-view part.")
    if n_heights == 2:
        print("REMINDER: two heights make the shared solve exactly determined. "
              "A third turns the shared leave-one-out into a real test.")
    if not slope_cams:
        print("NOTE: no view kept a fitted slope, so every view is corrected "
              "by an offset alone. On this rig that has previously been "
              "equivalent to or better than a slope-corrected fit.")
    return 0


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    """
    End to end on synthetic geometry, no files and no hardware.

    A known per-view correction and a known per-capture drift are imposed on
    perfectly reconstructed boards at three heights, and the solve has to
    separate them. This is the property the whole file turns on: if the drift
    leaks into the per-view terms, the correction will make the layering
    worse rather than better.
    """
    fails = []
    rng = np.random.default_rng(7)

    K = np.array([[700.0, 0.0, 320.0],
                  [0.0, 700.0, 250.0],
                  [0.0, 0.0, 1.0]])
    shape = (500, 640)

    # ---- depth round trip through the height map -----------------------
    n_c = np.array([0.02, -0.013, -1.0])
    n_c /= np.linalg.norm(n_c)
    d_c = -3.07
    alpha, beta = 1.2183, -0.0042
    z_true = plane_target_depth(shape, K, n_c, d_c * 0.93)
    rays = pixel_rays(shape, K)
    nr = rays @ n_c
    h_true = z_true * nr - d_c
    h_meas = (h_true - beta) / alpha
    z_meas = (h_meas + d_c) / nr
    z_back = correct_depth_height(z_meas, K, n_c, d_c, alpha, beta)
    err = float(np.nanmax(np.abs(z_back - z_true)))
    print(f"[test ] height correction round trip max error {err:.3e} m")
    if err > 1e-9:
        fails.append("height correction does not round trip")

    ident = correct_depth_height(z_meas, K, n_c, d_c, 1.0, 0.0)
    if float(np.nanmax(np.abs(ident - z_meas))) > 1e-12:
        fails.append("identity correction is not the identity")

    # ---- separation of per-view terms from per-capture drift -----------
    # The rig's own measured numbers: a mean compression of about 0.17, a
    # per-view compression spread of about 0.026, per-view offsets of about
    # 18 mm peak to peak, and a per-capture drift that grows from 10 mm at the
    # deck to 33 mm at 218 mm. The solve has to pull the first three out from
    # under the fourth.
    cams = ["left", "center", "right", "top"]
    a_true = {"left": 0.0175, "center": -0.0002, "right": 0.0241,
              "top": -0.0020}
    b_true = {"left": 0.0068, "center": -0.0006, "right": 0.0008,
              "top": -0.0070}
    a_true = {c: v - float(np.mean(list(a_true.values())))
              for c, v in a_true.items()}
    b_true = {c: v - float(np.mean(list(b_true.values())))
              for c, v in b_true.items()}
    P0, Q0 = 0.8300, -0.0010

    rows, drift = [], {}
    for label, h in (("deck", 0.000), ("riser1", 0.130), ("riser", 0.218)):
        for f in range(8):
            key = (label, f"frame_{f:05d}")
            e = float(rng.normal(0.0, 0.010 + 0.105 * h))
            drift[key] = e
            for c in cams:
                hm = ((P0 + a_true[c]) * h + (Q0 + b_true[c]) + e
                      + float(rng.normal(0.0, 0.0008)))
                rows.append({"camera": c, "label": label,
                             "frame": key[1], "key": key,
                             "h_meas": hm, "h_true": h,
                             "within_rms_mm": 5.0, "n": 20000,
                             "u": 250.0, "v": 200.0})

    fit = solve_effects(rows, cams, cams, huber=None)
    print(f"[test ] shared p recovered {fit['p_bar']:.4f} (true {P0}), "
          f"q {fit['q_bar'] * 1e3:+.2f} mm (true {Q0 * 1e3:+.1f})")
    if abs(fit["p_bar"] - P0) > 0.02:
        fails.append("the shared compression is biased, which is what fitting "
                     "in the wrong direction causes")

    da = max(abs(fit["a"][c] - a_true[c]) for c in cams)
    db = max(abs(fit["b"][c] - b_true[c]) for c in cams) * 1e3
    print(f"[test ] per-view slopes recovered to {da:.4f}, offsets to "
          f"{db:.2f} mm, under a per-capture drift of 10 to 33 mm")
    if da > 0.008 or db > 1.5:
        fails.append("per-view terms not separated from the capture drift")

    # The capture effects carry the shared fit's own error, by construction:
    # e_hat = (P0 - p_hat)(h - h_mean) + e - e_mean, because the gauge removes
    # the constant. Comparing against the raw drift would therefore fail for a
    # reason that is not a defect. The comparison below removes that term, so
    # what is left is the estimator's error in e alone.
    hs = {r["key"]: r["h_true"] for r in rows}
    h_mean = float(np.mean([hs[k] for k in drift]))
    e_mean = float(np.mean(list(drift.values())))
    de = max(abs(fit["e"][k] - (P0 - fit["p_bar"]) * (hs[k] - h_mean)
                 - (drift[k] - e_mean)) for k in drift) * 1e3
    print(f"[test ] capture effects recovered to {de:.2f} mm once the shared "
          f"fit's own error is accounted for")
    if de > 1.0:
        fails.append("capture effects not recovered")

    # ---- the gauge must hold -------------------------------------------
    ga = abs(sum(fit["a"].values()))
    gb = abs(sum(fit["b"].values()))
    ge = abs(sum(fit["e"].values()))
    print(f"[test ] gauge sums: a {ga:.2e}  b {gb:.2e}  e {ge:.2e}")
    if max(ga, gb, ge) > 1e-6:
        fails.append("gauge constraints not satisfied")

    # ---- the layering must actually fall -------------------------------
    pv = {c: fit["p_bar"] + fit["a"][c] for c in cams}
    qv = {c: fit["q_bar"] + fit["b"][c] for c in cams}
    fit["alpha"] = {c: 1.0 / pv[c] for c in cams}
    fit["beta"] = {c: -qv[c] / pv[c] for c in cams}
    labels = ["deck", "riser1", "riser"]
    st_before = layer_stats(rows, cams, labels)
    st_after = layer_stats(rows, cams, labels, fit)
    b_riser = st_before["riser"]["before_median_mm"]
    a_riser = st_after["riser"]["after_median_mm"]
    print(f"[test ] layering at 218 mm: {b_riser:.1f} mm before, "
          f"{a_riser:.1f} mm after")
    if a_riser > max(0.25 * b_riser, 3.0):
        fails.append("the correction does not collapse the layering")

    # ---- offset-only must still be a valid solve ------------------------
    fit_off = solve_effects(rows, cams, [], huber=None)
    if abs(sum(fit_off["a"].values())) > 1e-12 or any(
            abs(v) > 1e-12 for v in fit_off["a"].values()):
        fails.append("offset-only mode fitted a slope")
    print(f"[test ] offset-only mode holds every slope at zero")

    # ---- the bootstrap must widen when the captures disagree more -------
    tight = [dict(r) for r in rows]
    boot_a = bootstrap_effects(rows, cams, cams, None, 120, seed=1)
    for r in tight:
        r["h_meas"] = r["h_meas"] + float(rng.normal(0.0, 0.004))
    boot_b = bootstrap_effects(tight, cams, cams, None, 120, seed=1)
    s_a = sd(boot_a["b"]["left"]) * 1e3
    s_b = sd(boot_b["b"]["left"]) * 1e3
    print(f"[test ] bootstrap sigma on b_left: {s_a:.2f} mm clean, "
          f"{s_b:.2f} mm with 4 mm of extra per-view noise")
    if not (s_b > s_a):
        fails.append("bootstrap does not track the per-view scatter")

    # ---- plane target is exact by construction --------------------------
    n2 = np.array([0.1, -0.05, -1.0])
    n2 /= np.linalg.norm(n2)
    z = plane_target_depth((40, 50), K, n2, -3.0)
    r = pixel_rays((40, 50), K) * z[..., None]
    perr = float(np.nanmax(np.abs(r @ n2 + 3.0)))
    print(f"[test ] plane target max error {perr:.3e} m")
    if perr > 1e-12:
        fails.append("plane target inconsistent")

    # ---- thickness sign, stated as the report states it -----------------
    face_h, thickness = 3.06706, 0.006
    print(f"[test ] board face at {face_h * 1e3:.0f} mm -> deck surface at "
          f"{(face_h + thickness) * 1e3:.0f} mm")
    if abs((face_h + thickness) - 3.07306) > 1e-9:
        fails.append("thickness sign wrong")

    for f in fails:
        print(f"[test ] FAIL: {f}")
    print("[test ] " + ("FAIL" if fails else "ok"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())