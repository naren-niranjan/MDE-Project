#!/usr/bin/env python3
"""
board_gt.py

Measure the deck standoff and the reference plane separations from ChArUco
board images, using the calibrated intrinsics of one camera.

What this establishes
---------------------
The absolute perpendicular distance from a camera to the deck, which is the
one number that lets depth_align.py claim accuracy rather than merely
inter-view agreement. Without it the correction makes four views consistent
with each other while leaving a scale error common to all of them invisible.

Why the separations matter more than the absolute distance
----------------------------------------------------------
The board sits in a rigid frame, so its marker plane is some tens of
millimetres above whatever it rests on. That standoff biases every absolute
distance identically, and therefore cancels completely in the difference
between two heights. If the board is placed on the deck and then on a 130 mm
riser, the measured plane separation should read 130 mm whatever the frame
thickness is. A separation that comes back wrong indicts the intrinsics scale,
the pose chain or the riser itself, and it does so independently of any
assumption about the frame.

Consequently: trust the separations, and treat the absolute deck distance as
conditional on --standoff-mm, which should be measured with calipers.

What this does not establish
----------------------------
Nothing about the depth model. These are board poses from a calibrated
camera, not model output. A depth correction must be fitted on depths produced
by the production inference path, because DA3's scale depends on the framing
and on the batch a view was run in.

Example
-------
python board_gt.py \
    --images "gt/deck_*.png:deck:0" \
    --images "gt/riser1_*.png:riser1:130" \
    --images "gt/riser_*.png:riser:218" \
    --camera center --calib-dir /home/jetson/Projects/Calibration_4_5/results \
    --standoff-mm 6.0 --out ground_truth.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")


# --------------------------------------------------------------------------
# board and detection, tolerating both the pre- and post-4.7 aruco APIs
# --------------------------------------------------------------------------

def make_board(args):
    if not hasattr(cv2, "aruco"):
        raise SystemExit("this OpenCV build has no aruco module; install "
                         "opencv-contrib-python")
    dict_id = getattr(cv2.aruco, args.dictionary, None)
    if dict_id is None:
        raise SystemExit(f"unknown dictionary {args.dictionary!r}")
    adict = (cv2.aruco.getPredefinedDictionary(dict_id)
             if hasattr(cv2.aruco, "getPredefinedDictionary")
             else cv2.aruco.Dictionary_get(dict_id))

    size = (args.squares_x, args.squares_y)
    sq, mk = args.square_mm / 1000.0, args.marker_mm / 1000.0
    if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard,
                                                      "generateImage"):
        board = cv2.aruco.CharucoBoard(size, sq, mk, adict)
        if args.legacy and hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
    else:
        board = cv2.aruco.CharucoBoard_create(size[0], size[1], sq, mk, adict)
        if args.legacy:
            print("[warn ] this OpenCV predates setLegacyPattern; a calib.io "
                  "board may not match the generated layout")
    return board, adict


def detect_board(image, board, adict):
    """Return matched object and image points, or (None, None)."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        cc, ci, _, _ = detector.detectBoard(grey)
        if cc is None or ci is None or len(ci) < 4:
            return None, None
        obj, img = board.matchImagePoints(cc, ci)
        if obj is None or len(obj) < 4:
            return None, None
        return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)

    corners, ids, _ = cv2.aruco.detectMarkers(grey, adict)
    if ids is None or len(ids) == 0:
        return None, None
    _, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, grey, board)
    if cc is None or ci is None or len(ci) < 4:
        return None, None
    all_obj = board.chessboardCorners
    obj = np.asarray([all_obj[int(i)] for i in ci.ravel()], np.float64)
    return obj, cc.reshape(-1, 2).astype(np.float64)


# --------------------------------------------------------------------------
# pose
# --------------------------------------------------------------------------

def quad_basis(obj):
    """Quadratic shape basis in centred, scaled board coordinates."""
    xy = obj[:, :2] - obj[:, :2].mean(axis=0)
    scale = max(float(np.abs(xy).max()), 1e-9)
    x, y = xy[:, 0] / scale, xy[:, 1] / scale
    return np.stack([x * x, x * y, y * y], axis=1)


def board_plane(obj, img, K, dist, fit_bow=True, iters=15):
    """
    Solve the board pose and, optionally, its out-of-plane bow.

    The plane is returned as a unit normal pointing away from the camera plus
    the perpendicular distance from the camera centre.

    On bow: a displacement delta along the board normal shifts a corner's
    projection by (f*delta/Z)*(n_x - x_n*n_z) horizontally and the analogous
    quantity vertically. Constant and tilt terms in delta are exactly a pose
    offset and rotation, so only the quadratic shape is identifiable. Fitting
    that shape against a fixed pose badly underestimates it, because solvePnP
    has already absorbed most of the signature; measured against a synthetic
    dome, a fixed-pose fit recovered barely a fifth of the true amplitude.
    Alternating between pose and shape removes that bias, at the cost of
    needing the pose re-solved each iteration.

    The noise floor is roughly 0.2 mm at 0.05 px detection noise, so a
    reported bow below that is not evidence of a bowed board.
    """
    basis = quad_basis(obj) if (fit_bow and len(obj) >= 12) else None
    coef = np.zeros(3)
    delta = np.zeros(len(obj))
    rvec = tvec = None

    for _ in range(iters if basis is not None else 1):
        obj_b = obj + np.outer(delta, [0.0, 0.0, 1.0])
        ok, rvec, tvec = cv2.solvePnP(obj_b, img, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj_b, img, K, dist, rvec, tvec)
        if basis is None:
            break

        R, _ = cv2.Rodrigues(rvec)
        n = R @ np.array([0.0, 0.0, 1.0])
        proj, _ = cv2.projectPoints(obj_b, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        Z = ((obj_b @ R.T) + tvec.ravel())[:, 2]
        if np.any(Z <= 0):
            return None

        f = 0.5 * (K[0, 0] + K[1, 1])
        xn = (proj[:, 0] - K[0, 2]) / K[0, 0]
        yn = (proj[:, 1] - K[1, 2]) / K[1, 1]
        gu = (f / Z) * (n[0] - xn * n[2])
        gv = (f / Z) * (n[1] - yn * n[2])

        A = np.vstack([basis * gu[:, None], basis * gv[:, None]])
        r = np.concatenate([img[:, 0] - proj[:, 0], img[:, 1] - proj[:, 1]])
        step, *_ = np.linalg.lstsq(A, r, rcond=None)
        coef = coef + step
        delta = basis @ coef
        delta -= delta.mean()
        if np.linalg.norm(step) < 1e-9:
            break

    obj_b = obj + np.outer(delta, [0.0, 0.0, 1.0])
    R, _ = cv2.Rodrigues(rvec)
    n = R @ np.array([0.0, 0.0, 1.0])
    pts_cam = (obj_b @ R.T) + tvec.ravel()
    centroid = pts_cam.mean(axis=0)
    if float(n @ centroid) < 0:
        n = -n
    d = float(n @ centroid)

    proj, _ = cv2.projectPoints(obj_b, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)

    out = {"normal": n, "distance_m": d, "centroid": centroid,
           "rms_px": float(np.sqrt(np.mean(err ** 2))),
           "max_px": float(err.max()), "n_corners": int(len(obj))}
    if basis is not None:
        out["bow_pp_mm"] = float((delta.max() - delta.min()) * 1e3)
        out["bow_rms_mm"] = float(np.sqrt(np.mean(delta ** 2)) * 1e3)
    return out


# --------------------------------------------------------------------------

def load_intrinsics(calib_dir: Path, camera: str):
    path = calib_dir / f"intrinsics_{camera}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = json.loads(path.read_text())
    return (np.asarray(data["camera_matrix"], np.float64).reshape(3, 3),
            np.asarray(data.get("dist_coeffs", [0.0] * 5), np.float64).ravel(),
            data.get("lens_id"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ChArUco plane distances and separations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", action="append", required=True,
                    metavar="GLOB:LABEL:NOMINAL_MM",
                    help="image glob, group label and the nominal height of "
                         "the supporting surface in mm; repeat per height")
    ap.add_argument("--camera", required=True,
                    help="which camera's intrinsics the images came from")
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--squares-x", type=int, default=12)
    ap.add_argument("--squares-y", type=int, default=9)
    ap.add_argument("--square-mm", type=float, default=60.0)
    ap.add_argument("--marker-mm", type=float, default=47.0)
    ap.add_argument("--dictionary", default="DICT_5X5_250")
    ap.add_argument("--legacy", dest="legacy", action="store_true", default=True,
                    help="calib.io boards use the legacy marker layout")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")
    ap.add_argument("--no-bow-fit", dest="fit_bow", action="store_false",
                    default=True,
                    help="skip the bow estimate and hold the board rigid")
    ap.add_argument("--bow-snr", type=float, default=2.0,
                    help="a bow is called real only when it exceeds this "
                         "multiple of the noise floor; corner localisation "
                         "noise alone produces an apparent dome, so a fixed "
                         "threshold either passes bowed boards at low noise "
                         "or fails flat ones at high noise")
    ap.add_argument("--bow-floor-per-px", type=float, default=4.8,
                    help="apparent bow in mm produced by 1 px of detection "
                         "noise, measured on a synthetic flat board at 3 m "
                         "with 88 corners; scale it if your geometry differs")
    ap.add_argument("--max-sep-error-mm", type=float, default=3.0)
    ap.add_argument("--standoff-mm", type=float, default=6.0,
                    help="height of the marker plane above the surface the "
                         "board rests on; affects absolute distances only, "
                         "never the separations")
    ap.add_argument("--max-rms-px", type=float, default=1.0,
                    help="reject a pose whose reprojection exceeds this")
    ap.add_argument("--base-label", default="deck",
                    help="group treated as height zero")
    ap.add_argument("--out", type=Path, default=Path("ground_truth.json"))
    args = ap.parse_args()

    K, dist, lens = load_intrinsics(args.calib_dir, args.camera)
    print(f"[calib] {args.camera}  lens {lens}  fx={K[0,0]:.2f}")

    board, adict = make_board(args)
    print(f"[board] {args.squares_x}x{args.squares_y}  square "
          f"{args.square_mm} mm  marker {args.marker_mm} mm  "
          f"{args.dictionary}  legacy={args.legacy}")

    groups = {}
    for spec in args.images:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--images expects GLOB:LABEL:NOMINAL_MM, "
                             f"got {spec!r}")
        pattern, label, nominal = parts[0], parts[1], float(parts[2])
        paths = sorted(Path(p) for p in glob.glob(pattern))
        if not paths:
            raise SystemExit(f"no images matched {pattern!r}")
        groups[label] = {"nominal_mm": nominal, "paths": paths, "poses": []}

    print()
    for label, g in groups.items():
        for path in g["paths"]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[skip ] {path} unreadable")
                continue
            obj, img = detect_board(image, board, adict)
            if obj is None:
                print(f"[skip ] {path} no board detected")
                continue
            pose = board_plane(obj, img, K, dist, fit_bow=args.fit_bow)
            if pose is None:
                print(f"[skip ] {path} pose failed")
                continue
            if pose["rms_px"] > args.max_rms_px:
                print(f"[skip ] {path.name} rms {pose['rms_px']:.3f} px "
                      f"exceeds --max-rms-px")
                continue
            pose["file"] = path.name
            g["poses"].append(pose)
            print(f"[pose ] {label:<7} {path.name:<16} "
                  f"D={pose['distance_m']:.5f} m  corners "
                  f"{pose['n_corners']:>3d}  rms {pose['rms_px']:.3f} px")

    usable = {k: v for k, v in groups.items() if len(v["poses"]) >= 2}
    if args.base_label not in usable:
        raise SystemExit(f"base group {args.base_label!r} has too few usable "
                         f"poses")

    print()
    summary = {}
    for label, g in usable.items():
        d = np.array([p["distance_m"] for p in g["poses"]])
        n = np.stack([p["normal"] for p in g["poses"]])
        n_mean = n.mean(axis=0)
        n_mean /= np.linalg.norm(n_mean)
        tilt = np.degrees(np.arccos(np.clip(n @ n_mean, -1, 1)))
        summary[label] = {
            "n_poses": len(d),
            "distance_mean_m": float(d.mean()),
            "distance_spread_mm": float((d.max() - d.min()) * 1e3),
            "distance_std_mm": float(d.std(ddof=1) * 1e3),
            "normal": n_mean.tolist(),
            "within_group_tilt_max_deg": float(tilt.max()),
            "rms_px_mean": float(np.mean([p["rms_px"] for p in g["poses"]])),
        }
        bows = [p["bow_pp_mm"] for p in g["poses"] if "bow_pp_mm" in p]
        if bows:
            floor = args.bow_floor_per_px * summary[label]["rms_px_mean"]
            med = float(np.median(bows))
            summary[label]["bow_pp_median_mm"] = round(med, 3)
            summary[label]["bow_pp_max_mm"] = round(float(max(bows)), 3)
            summary[label]["bow_noise_floor_mm"] = round(float(floor), 3)
            summary[label]["bow_snr"] = round(med / max(floor, 1e-9), 2)
            summary[label]["bow_significant"] = bool(med > args.bow_snr * floor)
        bow_str = ""
        if "bow_pp_median_mm" in summary[label]:
            bow_str = (f"  bow {summary[label]['bow_pp_median_mm']:5.2f} mm "
                       f"(floor {summary[label]['bow_noise_floor_mm']:4.2f}, "
                       f"snr {summary[label]['bow_snr']:4.2f})")
            if summary[label]["bow_significant"]:
                bow_str += "  BOWED"
        print(f"[group] {label:<7} n={len(d)}  D={d.mean():.5f} m  "
              f"spread {summary[label]['distance_spread_mm']:6.2f} mm  "
              f"tilt {tilt.max():.3f} deg  "
              f"rms {summary[label]['rms_px_mean']:.3f} px{bow_str}")

    # ---- separations, which the standoff cannot touch -------------------
    base = summary[args.base_label]
    n0 = np.asarray(base["normal"])
    print()
    separations = {}
    for label, g in usable.items():
        if label == args.base_label:
            continue
        # Measure along the base plane normal, so a tilted board does not
        # inflate the separation through its lever arm.
        heights = [float(np.dot(n0, p["centroid"])) for p in g["poses"]]
        base_h = [float(np.dot(n0, p["centroid"]))
                  for p in usable[args.base_label]["poses"]]
        sep_mm = (np.mean(base_h) - np.mean(heights)) * 1e3
        nominal = g["nominal_mm"] - usable[args.base_label]["nominal_mm"]
        spread = (max(heights) - min(heights)) * 1e3
        tilt = np.degrees(np.arccos(np.clip(
            np.dot(np.asarray(summary[label]["normal"]), n0), -1, 1)))
        separations[f"{args.base_label}->{label}"] = {
            "measured_mm": round(float(sep_mm), 3),
            "nominal_mm": nominal,
            "error_mm": round(float(sep_mm - nominal), 3),
            "within_group_spread_mm": round(float(spread), 3),
            "tilt_vs_base_deg": round(float(tilt), 4),
        }
        flag = "" if abs(sep_mm - nominal) <= 3.0 else "   CHECK"
        print(f"[sep  ] {args.base_label}->{label:<7} measured "
              f"{sep_mm:8.3f} mm  nominal {nominal:7.1f} mm  error "
              f"{sep_mm - nominal:+7.3f} mm  tilt {tilt:.3f} deg{flag}")

    sep_ok = bool(separations) and all(
        abs(v["error_mm"]) <= args.max_sep_error_mm
        for v in separations.values())
    bow_ok = not any(g.get("bow_significant", False) for g in summary.values())

    deck_board_m = base["distance_mean_m"]
    deck_surface_m = deck_board_m + args.standoff_mm / 1000.0
    print(f"\n[deck ] board marker plane {deck_board_m:.5f} m from "
          f"{args.camera}")
    print(f"[deck ] deck surface {deck_surface_m:.5f} m, assuming a "
          f"{args.standoff_mm:.1f} mm standoff")
    print("[deck ] the standoff is an assumption; the separations above are "
          "not")

    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera": args.camera,
        "lens_id": lens,
        "board": {"squares": [args.squares_x, args.squares_y],
                  "square_mm": args.square_mm, "marker_mm": args.marker_mm,
                  "dictionary": args.dictionary, "legacy": args.legacy},
        "standoff_mm": args.standoff_mm,
        "groups": summary,
        "separations": separations,
        "deck_marker_plane_m": deck_board_m,
        "deck_surface_m": deck_surface_m,
        "separations_ok": sep_ok,
        "bow_ok": bow_ok,
        "ok": sep_ok and bow_ok,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.out}   ok={payload['ok']}  "
          f"(separations {'pass' if sep_ok else 'FAIL'}, "
          f"flatness {'pass' if bow_ok else 'FAIL'})")
    if not sep_ok:
        print("The separations disagree with the risers. Something upstream is "
              "wrong, so the absolute distance is not trustworthy either. Do "
              "not use deck_surface_m.")
    elif not bow_ok:
        print("The separations are good but the board is measurably bowed "
              "beyond the noise floor. deck_surface_m is usable; the board is "
              "a poor per-pixel reference plane until it is clamped flat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())