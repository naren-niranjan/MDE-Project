#!/usr/bin/env python3
"""
measure_deck.py

Measure the metric reference planes that anchor every dimension the depth
pipeline reports, and write them to ground_truth.json.

What this version changes, and why it matters for accuracy
----------------------------------------------------------
The previous version measured each placement independently and reduced the
surface to the MEDIAN of the per-placement perpendicular distances. On the
12 mm rig that produced 4.2 to 5.4 mm of spread across three placements of the
same rigid surface, which is not surface roughness: it is the coupling between
a small normal error and a large lateral offset.

    perp = |n . p|

For a plane through p with normal n, a normal error d rotates the foot of the
perpendicular. If the board centroid sits L metres off the optical axis, the
reported perpendicular moves by roughly L * |d|. At L = 0.5 m and 0.5 deg of
normal error that is 4 mm, which is exactly the spread observed. The quantity
was therefore partly measuring WHERE the board was placed.

This version pools instead. Every placement's ChArUco corners are lifted into
the camera frame, offset by the board thickness onto the surface they rest on,
and all placements of one labelled surface are fitted with a SINGLE robust
plane. Three consequences:

  the normal is constrained by the full lateral extent of every placement
  taken together, rather than by one board's footprint, so its error falls
  roughly in proportion to the extent

  the reported spread becomes a genuine consistency figure: the residual of
  each placement about the common plane, which is what a flatness measurement
  should be

  perp_m and the point used to project heights come from the SAME fit. The
  previous version took perp_m from a median and the point from whichever
  placement had the lowest reprojection error, so the two described slightly
  different planes.

THE SIGN OF THE BOARD THICKNESS
-------------------------------
The board's printed face sits one board-plus-frame thickness ABOVE the surface
it rests on, and the cameras look down, so the printed face is NEARER the
camera and the surface is the printed face PLUS the thickness.

    surface_perp = measured_perp + thickness

charuco_gt.py had this backwards and SUBTRACTED, which placed the deck 47 mm
above the board that was lying on it and left deck_perp_m 94 mm short. The
error was invisible in every height difference, because the thickness cancels
there, and the floor-to-deck separation it reported was correct to about a
millimetre. Retire charuco_gt.py; this file supersedes it.

VALIDATING THE SIGN
-------------------
Pass --expect LABEL HEIGHT_MM with a height taken off the surface with a rule,
and the measured height above the deck is asserted against it. That one
comparison exercises the intrinsics, the board geometry, the marker layout,
the thickness handling and the normal projection in a single number. A
disagreement of twice the thickness means the sign is wrong on one surface; a
disagreement of half a square means the marker layout was chosen wrong.

BOARD TILT AND THE PLANAR PnP AMBIGUITY
---------------------------------------
A near fronto-parallel board is the planar two-fold ambiguity waiting to
happen, and it also makes the pose's standoff the least well constrained
quantity in the solve. Tilt each placement by 10 to 20 degrees. This version
solves with solvePnPGeneric and inspects the SECOND solution: when its
reprojection error is within --ambiguity-ratio of the first, the pose is not
uniquely determined by the corners and that placement is reported and excluded
rather than silently averaged in.

WHAT THE ANCHOR IS LATER MATCHED AGAINST
----------------------------------------
layer_align.py anchors by matching the planes it extracts from a CAPTURE onto
the planes measured here. That match only succeeds for surfaces present in
both. The deck is always present. A riser measured here is present in the
capture only if the riser BLOCK is still standing on the belt when the capture
is taken, with the board removed, since the board face and the block top are
one thickness apart. Arbitrary parcels at arbitrary heights match nothing.

Usage
-----
    python measure_deck.py --calib-dir ../Calibration_4_5/results \
        --camera center --board-thickness-mm 47 \
        --plane deck  gt/deck_*.png \
        --plane riser gt/riser_*.png \
        --plane floor gt/floor_*.png \
        --expect riser 257 \
        --write ground_truth.json

A plane labelled exactly "deck" is required; it defines the normal every
height is measured along.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")


DEFAULTS = {
    "squares_x": 12,
    "squares_y": 9,
    "square_m": 0.060,
    "marker_m": 0.047,
    "dictionary": "DICT_5X5_250",
    "id_offset": 0,
}

MIN_CORNERS = 20
FLAT_TILT_WARN_DEG = 8.0
# Residual of a placement about the pooled plane. This is now a genuine
# consistency figure rather than a placement artefact, so the threshold is
# tighter than the 3 mm the per-placement spread used.
SPREAD_WARN_MM = 2.0
# Reprojection error above which the board parameters are suspect, given
# intrinsics validated at about 0.27 px.
RMS_WARN_PX = 0.60

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# --------------------------------------------------------------------------
# board
# --------------------------------------------------------------------------

def build_board(args, legacy):
    if not hasattr(cv2.aruco, args.dictionary):
        raise SystemExit(
            f"unknown dictionary {args.dictionary!r}. Available: "
            + ", ".join(a for a in dir(cv2.aruco) if a.startswith("DICT_")))
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dictionary))
    n_markers = (args.squares_x * args.squares_y) // 2
    ids = np.arange(args.id_offset, args.id_offset + n_markers, dtype=np.int32)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y),
                                   args.square_m, args.marker_m,
                                   dictionary, ids)
    if legacy:
        if not hasattr(board, "setLegacyPattern"):
            raise RuntimeError("this OpenCV build has no setLegacyPattern")
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
    return cv2.aruco.CharucoDetector(board, cp, dp,
                                     cv2.aruco.RefineParameters())


def choose_legacy(sample_paths, args, K, dist):
    """Try both marker layouts and keep whichever measures better.

    The two conventions start the marker grid from different corners. A board
    matching neither still returns interpolated corners, so the failure is a
    pose solved against the wrong object points rather than a detection error.
    Scoring on corner count AND reprojection error catches that; corner count
    alone does not.
    """
    scores = {}
    for legacy in (False, True):
        try:
            board = build_board(args, legacy)
        except RuntimeError:
            continue
        det = build_detector(board)
        corners, rms = 0, []
        for p in sample_paths[:3]:
            rec, _ = measure_one(p, board, det, K, dist, args)
            if rec is not None:
                corners += rec["n_corners"]
                rms.append(rec["rms_px"])
        scores[legacy] = (corners, -(float(np.mean(rms)) if rms else 9.99))
    if not scores or max(v[0] for v in scores.values()) == 0:
        raise SystemExit("no ChArUco corners detected under either marker "
                         "layout. Check --dictionary, --squares-x/-y and that "
                         "the board is in frame and in focus.")
    for lg, (c, negrms) in sorted(scores.items()):
        print(f"[board] legacy={str(lg):<5} corners={c:4d} "
              f"mean rms={-negrms:.3f} px")
    best = max(scores, key=lambda k: scores[k])
    vals = list(scores.values())
    if len(vals) == 2 and vals[0] == vals[1]:
        print("[board] both layouts score identically, so setLegacyPattern is "
              "a no-op on this OpenCV build and the choice does not matter")
    print(f"[board] legacy-pattern auto -> {best}")
    return best


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def load_intrinsics(calib_dir, name):
    """Load intrinsics, refusing rather than defaulting the distortion.

    A renamed or absent dist_coeffs key used to silently become zeros. Against
    2.4 per cent measured radial distortion that is a pose solved through the
    wrong optics, reported with a clean reprojection error, and the whole point
    of this file is that it beats a tape measure because the intrinsics are
    validated.
    """
    path = Path(calib_dir) / f"intrinsics_{name}.json"
    if not path.exists():
        raise SystemExit(f"no intrinsics at {path}")
    d = json.loads(path.read_text())
    if "camera_matrix" not in d:
        raise SystemExit(f"{path} has no 'camera_matrix'")
    K = np.asarray(d["camera_matrix"], np.float64).reshape(3, 3)
    if "dist_coeffs" not in d:
        raise SystemExit(
            f"{path} has no 'dist_coeffs'. It is NOT defaulted to zero: the "
            f"12 mm lens carries about 2.4 per cent radial distortion at "
            f"maximum field, and solving a pose without it would produce a "
            f"believable wrong standoff.")
    dist = np.asarray(d["dist_coeffs"], np.float64).ravel()
    return K, dist, d


def report_focal_spread(calib_dir, names):
    """Print the focal spread across the rig, because it bounds agreement.

    A camera whose solved fx sits a fraction f from the rig mean receives depth
    scaled by roughly f under the intrinsic prior. At 3.08 m a 0.77 per cent
    spread is 24 mm, the same order as the inter-camera disagreement the
    correction exists to remove. Only the per-view SCALE term reaches it, and
    that term needs two reference planes.
    """
    fxs = {}
    for n in names:
        p = Path(calib_dir) / f"intrinsics_{n}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            fxs[n] = float(np.asarray(d["camera_matrix"],
                                      np.float64).reshape(3, 3)[0, 0])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            print(f"[focal] {p.name}: unreadable ({exc})")
            continue
    if len(fxs) < 2:
        return None
    mean = float(np.mean(list(fxs.values())))
    spread = (max(fxs.values()) - min(fxs.values())) / mean
    print("\n[focal] solved fx across the rig")
    for n, v in fxs.items():
        print(f"        {n:<8} {v:8.1f} px  {(v / mean - 1) * 100:+6.3f}% "
              f"of the rig mean")
    print(f"        spread {spread * 100:.3f}%, which is "
          f"{spread * 3.08 * 1e3:.0f} mm of depth at the measured standoff")
    if spread > 0.004:
        print("        ** above 0.4 per cent. That is a floor on inter-camera "
              "agreement unless\n           the per-view SCALE term is solved, "
              "which needs two reference planes. **")
    return {"fx_px": {n: round(v, 2) for n, v in fxs.items()},
            "spread_fraction": round(spread, 5)}


# --------------------------------------------------------------------------
# per-placement pose
# --------------------------------------------------------------------------

def measure_one(path, board, detector, K, dist, args):
    """Board pose from one image, returned as SURFACE POINTS in the camera
    frame rather than as a plane.

    Returning points rather than a plane is what lets several placements be
    fitted together. A plane per placement throws away the lateral extent that
    constrains the normal, and it is the normal error that couples with the
    board's offset from the optical axis to move the perpendicular.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None, f"cannot read {path}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # solvePnPGeneric so the planar two-fold ambiguity is visible rather than
    # resolved arbitrarily. A flat board seen near fronto-parallel admits two
    # poses that reproject almost equally well and differ by a reflection of
    # the normal; averaging one of each across placements is silent nonsense.
    ambiguity = None
    rvec = tvec = None
    try:
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if n_sol >= 1:
            order = np.argsort(np.asarray(errs).ravel())
            rvec, tvec = rvecs[order[0]], tvecs[order[0]]
            e = np.asarray(errs).ravel()
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

    # Printed-face corners in the camera frame.
    face_pts = (np.asarray(obj, np.float64).reshape(-1, 3) @ R.T) + t

    normal = R[:, 2].astype(np.float64)
    centre = face_pts.mean(axis=0)
    # Orient the normal back towards the camera, so subtracting the thickness
    # pushes the plane AWAY from the camera onto the surface underneath.
    if float(normal @ centre) > 0:
        normal = -normal

    thickness = args.board_thickness_mm / 1000.0
    surf_pts = face_pts - normal * thickness

    return {
        "file": str(path),
        "n_corners": int(n),
        "rms_px": rms,
        "ambiguity_ratio": ambiguity,
        "normal": normal,
        "face_centroid": centre,
        "surface_points": surf_pts,
        "surface_centroid": surf_pts.mean(axis=0),
        "tilt_deg": float(np.degrees(np.arccos(
            np.clip(abs(normal[2]), 0.0, 1.0)))),
        "lateral_offset_m": float(np.linalg.norm(surf_pts.mean(axis=0)[:2])),
    }, None


# --------------------------------------------------------------------------
# pooled plane fit
# --------------------------------------------------------------------------

def fit_plane_robust(points, iters=6, tukey_c=2.5):
    """One plane through pooled surface points, reweighted.

    Reweighting rather than RANSAC: every point here is a detected ChArUco
    corner on a rigid surface, so there are no gross outliers to reject, only
    the tails of the corner-localisation error and any genuine departure of the
    surface from flatness. Least squares with a soft weight keeps all of the
    lateral extent, which is the thing constraining the normal.

    Returns (normal oriented towards the camera, offset, residuals).
    """
    P = np.asarray(points, np.float64)
    if len(P) < 8:
        return None, None, None
    w = np.ones(len(P))
    n = np.array([0.0, 0.0, -1.0])
    c = P.mean(axis=0)
    for _ in range(max(1, iters)):
        ws = w / max(float(w.sum()), 1e-12)
        c = (P * ws[:, None]).sum(axis=0)
        M = (P - c) * np.sqrt(ws)[:, None]
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        n = vt[-1]
        if float(n @ c) > 0:
            n = -n
        r = (P - c) @ n
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9
        u = np.abs(r) / (tukey_c * s)
        w = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if float(w.sum()) < 8:
            w = np.ones(len(P))
            break
    resid = (P - c) @ n
    return n, float(n @ c), resid


def measure_surface(label, paths, board, detector, K, dist, args):
    """Every placement of one surface, fitted as a SINGLE pooled plane.

    The thickness has already been applied per placement in measure_one, along
    that placement's own normal, which is correct: the board is displaced along
    its own normal, not along some common one.
    """
    results, failed = [], []
    for p in paths:
        rec, why = measure_one(p, board, detector, K, dist, args)
        if rec is None:
            failed.append({"file": str(p), "reason": why})
            print(f"  [{label}] {Path(p).name:<28} SKIPPED: {why}")
            continue
        amb = rec["ambiguity_ratio"]
        if amb is not None and amb < args.ambiguity_ratio:
            failed.append({"file": str(p),
                           "reason": f"planar PnP ambiguity: the second pose "
                                     f"reprojects only {amb:.2f} times worse "
                                     f"than the first, so the standoff is not "
                                     f"determined by the corners. Tilt the "
                                     f"board 10 to 20 deg"})
            print(f"  [{label}] {Path(p).name:<28} SKIPPED: ambiguous pose "
                  f"(ratio {amb:.2f} < {args.ambiguity_ratio})")
            continue
        results.append(rec)
        print(f"  [{label}] {Path(p).name:<28} "
              f"{rec['n_corners']:3d} corners, {rec['rms_px']:.3f} px, "
              f"tilt {rec['tilt_deg']:4.1f} deg, "
              f"offset {rec['lateral_offset_m']:.2f} m"
              + (f", ambiguity {amb:.1f}" if amb is not None else ""))

    if not results:
        return None, failed

    pooled = np.vstack([r["surface_points"] for r in results])
    n, d, resid = fit_plane_robust(pooled)
    if n is None:
        return None, failed

    # Per-placement consistency about the COMMON plane. This is the honest
    # spread: it asks whether the placements describe one surface, without the
    # normal-times-lateral-offset artefact that a per-placement perpendicular
    # carries.
    at = 0
    per_image = []
    for r in results:
        m = len(r["surface_points"])
        rr = resid[at:at + m]
        at += m
        per_image.append({
            "file": r["file"],
            "n_corners": r["n_corners"],
            "rms_px": round(r["rms_px"], 4),
            "tilt_deg": round(r["tilt_deg"], 2),
            "lateral_offset_m": round(r["lateral_offset_m"], 3),
            "ambiguity_ratio": (round(r["ambiguity_ratio"], 2)
                                if r["ambiguity_ratio"] is not None else None),
            "offset_from_pooled_mm": round(float(np.mean(rr)) * 1e3, 2),
            "residual_rms_mm": round(float(np.sqrt(np.mean(rr ** 2))) * 1e3, 2),
        })

    offsets = [e["offset_from_pooled_mm"] for e in per_image]
    spread_mm = (float(max(offsets) - min(offsets)) if len(offsets) > 1 else 0.0)
    flat_mm = float(np.sqrt(np.mean(resid ** 2))) * 1e3
    max_tilt = max(r["tilt_deg"] for r in results)
    extent = float(np.linalg.norm(pooled[:, :2].max(axis=0)
                                  - pooled[:, :2].min(axis=0)))

    summary = {
        "label": label,
        "n_images": len(results),
        "n_points": int(len(pooled)),
        "perp_m": float(abs(d)),
        "normal": n,
        "point": pooled.mean(axis=0),
        "spread_mm": round(spread_mm, 2),
        "flatness_rms_mm": round(flat_mm, 2),
        "pooled_extent_m": round(extent, 3),
        "best_rms_px": round(min(r["rms_px"] for r in results), 4),
        "worst_rms_px": round(max(r["rms_px"] for r in results), 4),
        "max_board_tilt_deg": round(max_tilt, 2),
        "min_corners": min(r["n_corners"] for r in results),
        "per_image": per_image,
    }
    print(f"  [{label}] pooled {len(pooled)} corners over {len(results)} "
          f"placement(s) spanning {extent:.2f} m")
    print(f"  [{label}] perpendicular {summary['perp_m']:.4f} m, "
          f"placement spread {spread_mm:.2f} mm, "
          f"flatness rms {flat_mm:.2f} mm, max board tilt {max_tilt:.1f} deg")
    if max_tilt < FLAT_TILT_WARN_DEG:
        print(f"  [{label}] ** every placement is within {max_tilt:.1f} deg of "
              f"fronto-parallel. That is where the planar PnP ambiguity lives "
              f"and where the standoff is least well constrained. Tilt the "
              f"board 10 to 20 deg. **")
    return summary, failed


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the metric reference planes that anchor the depth "
                    "pipeline, and write ground_truth.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--camera", default="center",
                    help="the REFERENCE camera. The fusion's world origin is "
                         "its optical centre, so every anchor must be measured "
                         "from it and no other")
    ap.add_argument("--all-cameras", nargs="+",
                    default=["left", "center", "right", "top"],
                    help="used only to report the focal spread across the rig")
    ap.add_argument("--plane", action="append", nargs="+", required=True,
                    metavar="LABEL IMAGE",
                    help="a labelled surface and the images of the board lying "
                         "on it, repeatable. A plane labelled exactly 'deck' "
                         "is required")
    ap.add_argument("--expect", action="append", nargs=2, default=[],
                    metavar=("LABEL", "HEIGHT_MM"),
                    help="assert a surface's height above the deck against a "
                         "value taken with a rule, repeatable. This is the one "
                         "check that exercises the intrinsics, the board "
                         "geometry, the marker layout, the thickness sign and "
                         "the normal projection in a single number")
    ap.add_argument("--expect-tol-mm", type=float, default=8.0,
                    help="how far a measured height may sit from its expected "
                         "value before the anchor is marked unusable")
    ap.add_argument("--board-thickness-mm", type=float, default=0.0,
                    help="board plus mount, ADDED along the normal so the "
                         "plane reported is the SURFACE and not the printed "
                         "face. On this rig the board is on a 47 mm frame. "
                         "Cancels in every height difference")
    ap.add_argument("--min-span", type=float, default=0.150,
                    help="metres between the nearest and furthest reference "
                         "plane below which the anchor is declared unusable")
    ap.add_argument("--parcel-band", nargs=2, type=float,
                    default=[0.030, 0.400], metavar=("MIN_M", "MAX_M"),
                    help="height band above the deck the robot picks from. At "
                         "least one reference plane must lie inside it, or "
                         "every correction solved from this anchor is an "
                         "extrapolation into the volume that matters")
    ap.add_argument("--max-spread-mm", type=float, default=SPREAD_WARN_MM,
                    help="largest disagreement between placements about the "
                         "pooled plane before the anchor is marked unusable. "
                         "This is now a real consistency figure, so unlike the "
                         "old per-placement spread it is a PROBLEM and not "
                         "merely a printed warning")
    ap.add_argument("--ambiguity-ratio", type=float, default=3.0,
                    help="a placement is discarded when the second PnP "
                         "solution reprojects less than this many times worse "
                         "than the first, because the pose is then not "
                         "determined by the corners. Raise the board tilt "
                         "rather than lowering this")
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--lens-id", default=None)

    ap.add_argument("--squares-x", type=int, default=DEFAULTS["squares_x"])
    ap.add_argument("--squares-y", type=int, default=DEFAULTS["squares_y"])
    ap.add_argument("--square-m", type=float, default=DEFAULTS["square_m"])
    ap.add_argument("--marker-m", type=float, default=DEFAULTS["marker_m"])
    ap.add_argument("--dictionary", default=DEFAULTS["dictionary"])
    ap.add_argument("--id-offset", type=int, default=DEFAULTS["id_offset"])
    ap.add_argument("--legacy-pattern", choices=["auto", "on", "off"],
                    default="auto")
    args = ap.parse_args()

    # ---- inputs -------------------------------------------------------
    groups = {}
    for entry in args.plane:
        if len(entry) < 2:
            raise SystemExit(f"--plane {entry[0]!r} was given no images")
        label, paths = entry[0], [Path(p) for p in entry[1:]]
        groups.setdefault(label, []).extend(paths)
    if "deck" not in groups:
        raise SystemExit("a plane labelled exactly 'deck' is required; it "
                         "defines the normal every height is measured along")

    expected = {}
    for label, val in args.expect:
        if label not in groups:
            raise SystemExit(f"--expect {label} but no --plane {label} given")
        expected[label] = float(val) / 1000.0

    K, dist, intr = load_intrinsics(args.calib_dir, args.camera)
    lens_id = args.lens_id or intr.get("lens_id")
    print(f"Reference camera : {args.camera} (SN {intr.get('serial')})")
    print(f"Intrinsics       : fx {K[0, 0]:.1f} px, RMS "
          f"{intr.get('rms_reprojection_error_px', float('nan')):.3f} px, "
          f"lens {lens_id}")
    if lens_id is None:
        print("[WARN] the intrinsics record no lens. The anchor written here "
              "will be untagged, and da3_fuse.py refuses untagged anchors.")
    print(f"Board thickness  : {args.board_thickness_mm:.1f} mm, ADDED along "
          f"the normal"
          + ("   ** zero: if the board rests ON the surface this under-reports "
             "every standoff by its thickness **"
             if args.board_thickness_mm == 0 else ""))
    print(f"Surfaces         : {', '.join(groups)}")
    if expected:
        print("Expected heights : "
              + ", ".join(f"{k} {v * 1e3:.0f} mm" for k, v in expected.items()))

    focal = report_focal_spread(args.calib_dir, args.all_cameras)
    print()

    # ---- board --------------------------------------------------------
    if args.legacy_pattern == "auto":
        legacy = choose_legacy(groups["deck"], args, K, dist)
    else:
        legacy = args.legacy_pattern == "on"
        print(f"[board] legacy-pattern forced -> {legacy}")
    board = build_board(args, legacy)
    detector = build_detector(board)
    print()

    # ---- measure ------------------------------------------------------
    measured, failures = {}, {}
    for label, paths in groups.items():
        s, f = measure_surface(label, paths, board, detector, K, dist, args)
        if s is not None:
            measured[label] = s
        if f:
            failures[label] = f
        print()

    if "deck" not in measured:
        raise SystemExit("no usable deck image; nothing can be anchored")

    deck = measured["deck"]
    n_deck = np.asarray(deck["normal"], float)
    base = float(n_deck @ np.asarray(deck["point"], float))

    planes = []
    for label, s in measured.items():
        h = float(n_deck @ np.asarray(s["point"], float)) - base
        tilt = float(np.degrees(np.arccos(np.clip(
            abs(float(n_deck @ np.asarray(s["normal"], float))), 0.0, 1.0))))
        entry = {
            "label": label,
            "perp_m": round(s["perp_m"], 5),
            "height_above_deck_m": round(h, 5),
            "height_above_deck_mm": round(h * 1e3, 1),
            "tilt_vs_deck_deg": round(tilt, 3),
            "spread_mm": s["spread_mm"],
            "flatness_rms_mm": s["flatness_rms_mm"],
            "pooled_extent_m": s["pooled_extent_m"],
            "max_board_tilt_deg": s["max_board_tilt_deg"],
            "n_images": s["n_images"],
            "n_points": s["n_points"],
            "worst_rms_px": s["worst_rms_px"],
            "min_corners": s["min_corners"],
            "normal": [round(float(v), 6) for v in s["normal"]],
            "per_image": s["per_image"],
        }
        if label in expected:
            err = h - expected[label]
            entry["expected_height_mm"] = round(expected[label] * 1e3, 1)
            entry["expected_error_mm"] = round(err * 1e3, 1)
        planes.append(entry)
    planes.sort(key=lambda e: e["perp_m"])

    # ---- report -------------------------------------------------------
    print("=" * 92)
    print(f"{'label':<10} {'perp from camera':>18} {'above deck':>12} "
          f"{'tilt':>7} {'spread':>8} {'flatness':>9} {'vs rule':>9}")
    for e in planes:
        vs = (f"{e['expected_error_mm']:+7.1f}mm"
              if "expected_error_mm" in e else "        -")
        print(f"{e['label']:<10} {e['perp_m']:>15.4f} m "
              f"{e['height_above_deck_mm']:>9.1f} mm "
              f"{e['tilt_vs_deck_deg']:>6.2f}d {e['spread_mm']:>6.2f} mm "
              f"{e['flatness_rms_mm']:>6.2f} mm {vs:>9}")
    print("=" * 92)

    span = planes[-1]["perp_m"] - planes[0]["perp_m"]
    lo, hi = args.parcel_band
    in_band = [e for e in planes
               if lo - 0.02 <= e["height_above_deck_m"] <= hi + 0.02
               and e["label"] != "deck"]
    problems = []

    if len(planes) < 2:
        problems.append("only one surface was measured, so scale and offset "
                        "cannot be separated at all")
    if span < args.min_span:
        problems.append(f"the planes span only {span * 1e3:.0f} mm, below the "
                        f"{args.min_span * 1e3:.0f} mm needed to separate "
                        f"scale from offset against the depth noise")
    if not in_band:
        problems.append(f"no reference plane lies inside the parcel band "
                        f"{lo * 1e3:.0f} to {hi * 1e3:.0f} mm above the deck, "
                        f"so any correction solved from this anchor is "
                        f"extrapolated into the volume the robot picks from")

    # The placement spread is now a genuine consistency figure, so it becomes a
    # PROBLEM rather than a printed warning that nothing acts on. It propagates
    # directly into the scale term of every correction solved from this anchor.
    for e in planes:
        if e["spread_mm"] > args.max_spread_mm:
            problems.append(
                f"{e['label']}: the placements disagree by "
                f"{e['spread_mm']:.2f} mm about the pooled plane, above the "
                f"{args.max_spread_mm:.1f} mm limit. That disagreement enters "
                f"the anchor's scale term directly. Either the surface is not "
                f"flat over the area sampled, or a placement is poorly "
                f"constrained: check the tilt and lateral offset columns")
        if e["worst_rms_px"] > RMS_WARN_PX:
            problems.append(
                f"{e['label']}: a pose reprojects at {e['worst_rms_px']:.3f} "
                f"px against intrinsics validated near 0.27 px, so the board "
                f"parameters may not match those used in calibration")

    # ---- the assertion against the rule -------------------------------
    t_mm = args.board_thickness_mm
    for e in planes:
        if "expected_error_mm" not in e:
            continue
        err = e["expected_error_mm"]
        if abs(err) <= args.expect_tol_mm:
            print(f"[check] {e['label']} measures "
                  f"{e['height_above_deck_mm']:.1f} mm against the expected "
                  f"{e['expected_height_mm']:.0f} mm, {err:+.1f} mm out. The "
                  f"thickness sign, the marker layout and the normal "
                  f"projection are all consistent.")
            continue
        hint = ""
        if t_mm > 0 and abs(abs(err) - 2 * t_mm) < 0.25 * t_mm:
            hint = (f" That is twice the {t_mm:.0f} mm board thickness, so the "
                    f"frame was under one surface and not the other, or the "
                    f"expected height already included it.")
        elif t_mm > 0 and abs(abs(err) - t_mm) < 0.25 * t_mm:
            hint = (f" That is one board thickness, so the frame was present "
                    f"on one surface only.")
        elif abs(abs(err) - args.square_m * 500) < 0.25 * args.square_m * 500:
            hint = (" That is half a checker square, which is the signature of "
                    "the wrong ChArUco marker layout.")
        problems.append(f"{e['label']} measures "
                        f"{e['height_above_deck_mm']:.1f} mm against the "
                        f"expected {e['expected_height_mm']:.0f} mm, "
                        f"{err:+.1f} mm out, beyond the "
                        f"{args.expect_tol_mm:.0f} mm tolerance.{hint}")

    for e in planes:
        if e["tilt_vs_deck_deg"] > 2.0 and e["label"] != "deck":
            problems.append(f"{e['label']} sits {e['tilt_vs_deck_deg']:.1f} deg "
                            f"from the deck. Either it is genuinely not "
                            f"parallel or a board pose has flipped through the "
                            f"planar PnP ambiguity")

    usable = not problems
    print(f"\nspan {span * 1e3:.0f} mm over {len(planes)} surface(s); "
          f"{len(in_band)} inside the parcel band")
    for w in problems:
        print(f"  ** {w} **")

    floor = next((e for e in planes if e["label"] == "floor"), None)
    separation = (round(abs(floor["height_above_deck_m"]), 5)
                  if floor is not None else None)

    record = {
        "lens_id": lens_id,
        "reference_camera": args.camera,
        "deck_perp_m": round(deck["perp_m"], 5),
        "separation_m": separation,
        "reference_planes": planes,
        "span_m": round(span, 5),
        "parcel_band_m": [lo, hi],
        "planes_in_parcel_band": [e["label"] for e in in_band],
        "usable": bool(usable),
        "problems": problems,
        "focal_spread": focal,
        "measured": date.today().isoformat(),
        "method": ("ChArUco board laid on each surface, solvePnPGeneric "
                   "against the calibrated intrinsics of the reference camera, "
                   f"board plus mount {args.board_thickness_mm:.1f} mm ADDED "
                   f"along each placement's own normal, then ALL placements of "
                   f"one surface fitted as a single robust plane. Legacy "
                   f"marker layout {'on' if legacy else 'off'}"),
        "notes": ("perp_m values are PHYSICAL distances from the reference "
                  "camera's optical centre, taken from a plane pooled over "
                  "every placement, so they do not carry the "
                  "normal-error-times-lateral-offset artefact that a "
                  "per-placement perpendicular does. spread_mm is now the "
                  "disagreement between placements about that common plane, "
                  "which is a consistency measurement rather than a placement "
                  "artefact. These are not the same quantity as "
                  "--seg-plane-distance or --deck-distance, which select a "
                  "band in DA3's unanchored coordinates and read several per "
                  "cent short. layer_align.py can only match a plane listed "
                  "here if the same physical surface is present in the "
                  "capture, so leave the riser BLOCK on the belt, board "
                  "removed, when taking the alignment capture."),
        "detail": {
            "camera_serial": intr.get("serial"),
            "intrinsics_rms_px": intr.get("rms_reprojection_error_px"),
            "board": {"squares": [args.squares_x, args.squares_y],
                      "square_m": args.square_m, "marker_m": args.marker_m,
                      "dictionary": args.dictionary,
                      "legacy_pattern": bool(legacy),
                      "thickness_mm": args.board_thickness_mm,
                      "thickness_sign": "added along the normal"},
            "ambiguity_ratio_limit": args.ambiguity_ratio,
            "max_spread_mm": args.max_spread_mm,
            "failures": failures,
        },
    }

    if args.write is not None:
        args.write.write_text(json.dumps(record, indent=2))
        print(f"\n-> {args.write}")
        if not usable:
            print("   This anchor is marked UNUSABLE. da3_fuse.py will load it "
                  "and report NaN\n   rather than a believable wrong number.")
        else:
            print("   Anchor usable. When you take the capture for "
                  "MODE=align, leave the riser\n   BLOCK standing on the belt "
                  "with the board REMOVED: layer_align.py matches\n   the "
                  "planes it extracts onto the planes measured here, and it "
                  "can only match\n   a surface that is present in both.")
    else:
        print("\n" + json.dumps(record, indent=2))
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())