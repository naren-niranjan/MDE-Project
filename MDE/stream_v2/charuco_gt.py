#!/usr/bin/env python3
"""
charuco_gt.py

Metric ground truth for plane distances using a ChArUco board, replacing tape
measurements in the depth-fusion validation.

What it measures
----------------
Place the board flat on a surface, capture from the reference camera, and the
board pose gives that surface as a plane in the camera frame. Do it twice --
once on the conveyor deck, once on the floor -- and you get:

    camera-to-plane perpendicular distance   for each surface
    camera-to-board-centroid distance        (comparable to plane_separation.py)
    plane-to-plane separation                the floor-to-deck height
    angle between the two planes             non-zero means one is not level

Why this beats a tape measure
-----------------------------
The optical centre sits inside the lens barrel and cannot be marked from
outside, so a tape to the "camera" is guessing to a centimetre or two. The
board pose is solved against the same intrinsics already validated at 0.26 px
reprojection error, so the distance comes with a stated error budget instead
of tape uncertainty.

Board configuration
-------------------
Defaults match the calib.io 9x12 board: 60 mm checkers, 47 mm markers,
DICT_5X5. A 9x12 board carries (9*12)//2 = 54 markers, so the dictionary must
hold at least 54 -- DICT_5X5_50 is ruled out. The bit patterns differ between
_100 / _250 / _1000, so the dictionary must match what calib.io actually used.

LEGACY PATTERN: OpenCV changed the ChArUco marker layout at version 4.7. The
old convention starts the marker grid from a different corner, and with an odd
square count in one dimension the two conventions give visibly different
boards. A board generated under the old convention will detect few or no
corners against a new-convention board object. Hence --legacy-pattern, which
defaults to 'auto' and tries both.

Rather than guessing the dictionary and the pattern convention, run:

    python charuco_gt.py --probe --deck <one image> --floor <one image> \
        --intrinsics-json <...>

which sweeps every plausible dictionary x legacy combination and reports which
one detects the most corners. Lock the winner in with explicit flags after.

Board thickness
---------------
The detected plane is the board's PRINTED FACE, one board thickness above the
surface it rests on.

    camera-to-surface distance : thickness is subtracted (--board-thickness)
    plane-to-plane separation  : thickness CANCELS, provided the same board
                                 lies flat on both surfaces. No correction
                                 applied and none needed.

Example
-------
python charuco_gt.py \
    --deck  gt/deck_01.png  --deck  gt/deck_02.png  --deck gt/deck_03.png \
    --floor gt/floor_01.png --floor gt/floor_02.png --floor gt/floor_03.png \
    --intrinsics-json /home/jetson/Projects/Calibration_4_5/results/intrinsics_center.json \
    --board-thickness 0.003 \
    --out gt_planes.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required: pip install opencv-contrib-python")


# --------------------------------------------------------------------------
# board defaults -- calib.io 9x12, 60 mm checker, 47 mm marker, DICT_5X5
# --------------------------------------------------------------------------

BOARD_SQUARES_X = 12
BOARD_SQUARES_Y = 9
SQUARE_LENGTH = 0.060
MARKER_LENGTH = 0.047
ARUCO_DICT_NAME = "DICT_5X5_250"
MARKER_ID_OFFSET = 0

PROBE_DICTS = ["DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
               "DICT_4X4_250", "DICT_6X6_250"]


# --------------------------------------------------------------------------
# OpenCV aruco compatibility (API changed substantially at 4.7)
# --------------------------------------------------------------------------

def get_dictionary(name):
    if not hasattr(cv2.aruco, name):
        sys.exit(f"unknown dictionary {name!r}. Available: "
                 + ", ".join(a for a in dir(cv2.aruco) if a.startswith("DICT_")))
    key = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(key)
    return cv2.aruco.Dictionary_get(key)


def make_board(sx, sy, sql, mkl, dictionary, legacy=False, id_offset=0):
    """Build a CharucoBoard across old and new OpenCV APIs."""
    n_markers = (sx * sy) // 2
    ids = None
    if id_offset:
        ids = np.arange(n_markers, dtype=np.int32) + int(id_offset)

    board = None
    try:
        if ids is not None:
            board = cv2.aruco.CharucoBoard((sx, sy), sql, mkl, dictionary, ids)
        else:
            board = cv2.aruco.CharucoBoard((sx, sy), sql, mkl, dictionary)
    except Exception:
        board = cv2.aruco.CharucoBoard_create(sx, sy, sql, mkl, dictionary)
        if ids is not None:
            try:
                board.ids = ids
            except Exception:
                print("[warn] cannot apply marker id offset on this OpenCV "
                      "build; results will be wrong if the board is offset")

    if legacy:
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
        else:
            raise RuntimeError("this OpenCV build has no setLegacyPattern; "
                               "legacy boards need OpenCV >= 4.7")
    return board


def detect_charuco(gray, board, dictionary):
    """Return (charuco_corners, charuco_ids) or (None, None)."""
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            det = cv2.aruco.CharucoDetector(board)
            cc, ci, _, _ = det.detectBoard(gray)
            if cc is not None and len(cc) >= 6:
                return cc, ci
        except Exception:
            pass

    try:
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            ad = cv2.aruco.ArucoDetector(dictionary, params)
            mc, mi, _ = ad.detectMarkers(gray)
        else:
            params = cv2.aruco.DetectorParameters_create()
            mc, mi, _ = cv2.aruco.detectMarkers(gray, dictionary,
                                                parameters=params)
        if mi is None or len(mi) == 0:
            return None, None
        _, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board)
        if cc is not None and len(cc) >= 6:
            return cc, ci
    except Exception as e:
        print(f"  [detect] legacy path failed: {e}")

    return None, None


def detect_markers_only(gray, dictionary):
    """Raw marker detection, for the probe sweep."""
    try:
        if hasattr(cv2.aruco, "ArucoDetector"):
            ad = cv2.aruco.ArucoDetector(dictionary,
                                         cv2.aruco.DetectorParameters())
            _, ids, _ = ad.detectMarkers(gray)
        else:
            _, ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary,
                parameters=cv2.aruco.DetectorParameters_create())
        return np.asarray(ids).ravel() if ids is not None else np.array([])
    except Exception:
        return np.array([])


def board_object_points(board, ids):
    if hasattr(board, "getChessboardCorners"):
        allpts = np.asarray(board.getChessboardCorners(), np.float64)
    else:
        allpts = np.asarray(board.chessboardCorners, np.float64)
    return allpts[np.asarray(ids).ravel()]


def solve_board_pose(cc, ci, board, K, dist):
    """Return (R, t, rms_px) with X_cam = R @ X_board + t."""
    obj = board_object_points(board, ci)
    img = np.asarray(cc, np.float64).reshape(-1, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(
        obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel(), float(np.sqrt(np.mean(err ** 2)))


# --------------------------------------------------------------------------
# probe: which dictionary and pattern convention does this board use?
# --------------------------------------------------------------------------

def probe(images, args):
    print("=" * 72)
    print("PROBE: sweeping dictionary x legacy-pattern combinations")
    print(f"board {args.squares_x}x{args.squares_y}, "
          f"square {args.square_length * 1e3:.0f} mm, "
          f"marker {args.marker_length * 1e3:.0f} mm, "
          f"expecting {(args.squares_x * args.squares_y) // 2} markers")
    print("=" * 72)

    grays = []
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"cannot read image: {p}")
        grays.append((p.name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    print(f"\n-- raw marker detection (dictionary only, pattern irrelevant) --")
    best_dict, best_markers = None, -1
    for dname in PROBE_DICTS:
        try:
            d = get_dictionary(dname)
        except SystemExit:
            continue
        counts, id_lo, id_hi = [], [], []
        for _, g in grays:
            ids = detect_markers_only(g, d)
            counts.append(len(ids))
            if len(ids):
                id_lo.append(int(ids.min()))
                id_hi.append(int(ids.max()))
        tot = sum(counts)
        rng = f"ids {min(id_lo)}-{max(id_hi)}" if id_lo else "none"
        print(f"  {dname:<16} markers/img={counts}  {rng}")
        if tot > best_markers:
            best_dict, best_markers = dname, tot

    if best_markers <= 0:
        sys.exit("\nno markers detected under any dictionary. Check that the "
                 "board is in frame, in focus, and well lit.")

    print(f"\n  -> best dictionary: {best_dict}")
    print("  If the lowest detected id is not 0, set --marker-id-offset "
          "to that value.")

    print(f"\n-- ChArUco corner interpolation ({best_dict}) --")
    d = get_dictionary(best_dict)
    results = {}
    for legacy in (False, True):
        try:
            b = make_board(args.squares_x, args.squares_y, args.square_length,
                           args.marker_length, d, legacy, args.marker_id_offset)
        except RuntimeError as e:
            print(f"  legacy={legacy}: unavailable ({e})")
            continue
        counts = []
        for _, g in grays:
            cc, _ = detect_charuco(g, b, d)
            counts.append(0 if cc is None else len(cc))
        results[legacy] = sum(counts)
        print(f"  legacy={str(legacy):<5} corners/img={counts}  "
              f"total={sum(counts)}")

    if results:
        best_legacy = max(results, key=results.get)
        max_corners = (args.squares_x - 1) * (args.squares_y - 1)
        print(f"\n  -> best: --dictionary {best_dict} "
              f"{'--legacy-pattern on' if best_legacy else '--legacy-pattern off'}")
        print(f"  (a fully visible board yields up to {max_corners} corners)")
        if results[best_legacy] == 0:
            print("\n  WARNING: zero corners either way. The dictionary is "
                  "detecting markers but the board geometry does not match. "
                  "Re-check squares_x / squares_y and the id offset.")
    print("=" * 72)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def plane_from_pose(R, t):
    """Board plane in the camera frame; normal points away from the camera."""
    n = R[:, 2].astype(np.float64)
    if np.dot(n, t) < 0:
        n = -n
    d = -float(np.dot(n, t))
    return n, d, abs(d)


def board_centroid(R, t, board, ids):
    obj = board_object_points(board, ids)
    return R @ obj.mean(axis=0) + t


def measure_surface(paths, board, dictionary, K, dist, label):
    results = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"cannot read image: {p}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cc, ci = detect_charuco(gray, board, dictionary)
        if cc is None:
            print(f"  [{label}] {p.name}: no usable ChArUco corners")
            continue

        sol = solve_board_pose(cc, ci, board, K, dist)
        if sol is None:
            print(f"  [{label}] {p.name}: pose solve failed")
            continue
        R, t, rms = sol
        n, d, perp = plane_from_pose(R, t)
        cen = board_centroid(R, t, board, ci)

        results.append({
            "image": p.name, "n_corners": int(len(cc)), "rms_px": round(rms, 4),
            "normal": n.tolist(), "d": float(d),
            "perp_dist_m": round(perp, 5),
            "centroid_dist_m": round(float(np.linalg.norm(cen)), 5),
            "centroid_cam": cen.tolist(),
        })
        print(f"  [{label}] {p.name}: corners={len(cc):>3d} "
              f"rms={rms:.3f}px  perp={perp:.4f} m  "
              f"centroid={np.linalg.norm(cen):.4f} m")
    return results


def aggregate(results, label):
    if not results:
        sys.exit(f"no successful detections for {label}. Run --probe to find "
                 f"the right dictionary and pattern convention.")
    perp = np.array([r["perp_dist_m"] for r in results])
    cent = np.array([r["centroid_dist_m"] for r in results])
    normals = np.array([r["normal"] for r in results])
    ds = np.array([r["d"] for r in results])

    n_mean = normals.mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)

    agg = {
        "n_images": len(results),
        "perp_dist_mean_m": round(float(perp.mean()), 5),
        "perp_dist_spread_mm": round(float(np.ptp(perp) * 1e3), 2),
        "centroid_dist_mean_m": round(float(cent.mean()), 5),
        "normal_mean": n_mean.tolist(),
        "d_mean": float(ds.mean()),
        "rms_px_max": max(r["rms_px"] for r in results),
        "n_corners_min": min(r["n_corners"] for r in results),
    }
    if len(results) > 1:
        cos = normals @ n_mean
        agg["normal_spread_deg"] = round(float(np.degrees(
            np.ptp(np.arccos(np.clip(cos, -1, 1))))), 4)
    return agg


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ChArUco ground truth for plane distances and separation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--deck", action="append", type=Path, required=True,
                    help="board flat on the conveyor deck; repeat for several "
                         "placements")
    ap.add_argument("--floor", action="append", type=Path, required=True,
                    help="board flat on the floor; repeat")
    ap.add_argument("--intrinsics-json", type=Path, required=True)

    g = ap.add_argument_group("board (calib.io 9x12 defaults)")
    g.add_argument("--squares-x", type=int, default=BOARD_SQUARES_X)
    g.add_argument("--squares-y", type=int, default=BOARD_SQUARES_Y)
    g.add_argument("--square-length", type=float, default=SQUARE_LENGTH,
                   help="metres")
    g.add_argument("--marker-length", type=float, default=MARKER_LENGTH,
                   help="metres")
    g.add_argument("--dictionary", default=ARUCO_DICT_NAME)
    g.add_argument("--marker-id-offset", type=int, default=MARKER_ID_OFFSET,
                   help="first marker id on the printed board")
    g.add_argument("--legacy-pattern", choices=["auto", "on", "off"],
                   default="auto",
                   help="OpenCV <4.7 marker layout; 'auto' tries both and "
                        "keeps whichever finds more corners")

    ap.add_argument("--board-thickness", type=float, default=0.0,
                    help="board + mount thickness in metres; subtracted from "
                         "camera-to-surface distances, cancels in separation")
    ap.add_argument("--probe", action="store_true",
                    help="sweep dictionaries and pattern conventions, then exit")
    ap.add_argument("--out", type=Path, default=Path("gt_planes.json"))
    args = ap.parse_args()

    if args.probe:
        probe(list(args.deck) + list(args.floor), args)
        return

    data = json.loads(args.intrinsics_json.read_text())
    K = np.asarray(data["camera_matrix"], np.float64).reshape(3, 3)
    dist = np.asarray(data.get("dist_coeffs", [0, 0, 0, 0, 0]), np.float64)
    print(f"[calib] {args.intrinsics_json.name}  fx={K[0,0]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}  "
          f"rms={data.get('rms_reprojection_error_px', float('nan')):.4f} px")

    n_markers = (args.squares_x * args.squares_y) // 2
    print(f"[board] {args.squares_x}x{args.squares_y}  "
          f"square={args.square_length * 1e3:.0f}mm "
          f"marker={args.marker_length * 1e3:.0f}mm  "
          f"{args.dictionary}  {n_markers} markers  "
          f"id_offset={args.marker_id_offset}")

    dictionary = get_dictionary(args.dictionary)

    # ---- pick the pattern convention ---------------------------------
    def build(legacy):
        return make_board(args.squares_x, args.squares_y, args.square_length,
                          args.marker_length, dictionary, legacy,
                          args.marker_id_offset)

    if args.legacy_pattern == "auto":
        probe_img = cv2.cvtColor(cv2.imread(str(args.deck[0])),
                                 cv2.COLOR_BGR2GRAY)
        scores = {}
        for lg in (False, True):
            try:
                cc, _ = detect_charuco(probe_img, build(lg), dictionary)
                scores[lg] = 0 if cc is None else len(cc)
            except RuntimeError:
                pass
        if not scores or max(scores.values()) == 0:
            sys.exit("no corners detected either way. Run --probe to find the "
                     "correct dictionary and board geometry.")
        legacy = max(scores, key=scores.get)
        print(f"[board] legacy-pattern auto -> {legacy}  "
              f"(corners: off={scores.get(False, 0)}, on={scores.get(True, 0)})")
    else:
        legacy = args.legacy_pattern == "on"
        print(f"[board] legacy-pattern forced -> {legacy}")

    board = build(legacy)
    print()

    deck = measure_surface(args.deck, board, dictionary, K, dist, "deck")
    floor = measure_surface(args.floor, board, dictionary, K, dist, "floor")

    a_deck = aggregate(deck, "deck")
    a_floor = aggregate(floor, "floor")

    n = np.asarray(a_deck["normal_mean"])
    parallel = abs(float(np.dot(n, a_floor["normal_mean"]))) > 0.99
    if parallel:
        sep = abs(a_floor["d_mean"] - a_deck["d_mean"])
    else:
        c_floor = np.mean([r["centroid_cam"] for r in floor], axis=0)
        c_deck = np.mean([r["centroid_cam"] for r in deck], axis=0)
        sep = abs(float(np.dot(c_floor - c_deck, n)))

    angle = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(n, a_floor["normal_mean"])), -1, 1))))

    thick = args.board_thickness
    max_corners = (args.squares_x - 1) * (args.squares_y - 1)
    report = {
        "intrinsics": str(args.intrinsics_json),
        "board": {"squares_x": args.squares_x, "squares_y": args.squares_y,
                  "square_length_m": args.square_length,
                  "marker_length_m": args.marker_length,
                  "dictionary": args.dictionary,
                  "marker_id_offset": args.marker_id_offset,
                  "legacy_pattern": bool(legacy),
                  "thickness_m": thick,
                  "max_corners": max_corners},
        "deck": {"per_image": deck, **a_deck,
                 "surface_perp_dist_m": round(a_deck["perp_dist_mean_m"] - thick, 5),
                 "surface_centroid_dist_m": round(
                     a_deck["centroid_dist_mean_m"] - thick, 5)},
        "floor": {"per_image": floor, **a_floor,
                  "surface_perp_dist_m": round(a_floor["perp_dist_mean_m"] - thick, 5),
                  "surface_centroid_dist_m": round(
                      a_floor["centroid_dist_mean_m"] - thick, 5)},
        "separation_m": round(float(sep), 5),
        "separation_mm": round(float(sep) * 1e3, 2),
        "plane_angle_deg": round(angle, 4),
        "planes_parallel": bool(parallel),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 58)
    print(f"deck  perpendicular : {report['deck']['surface_perp_dist_m']:.4f} m"
          f"   (spread {a_deck['perp_dist_spread_mm']:.2f} mm over "
          f"{a_deck['n_images']} img)")
    print(f"deck  to centroid   : {report['deck']['surface_centroid_dist_m']:.4f} m")
    print(f"floor perpendicular : {report['floor']['surface_perp_dist_m']:.4f} m"
          f"   (spread {a_floor['perp_dist_spread_mm']:.2f} mm over "
          f"{a_floor['n_images']} img)")
    print(f"floor to centroid   : {report['floor']['surface_centroid_dist_m']:.4f} m")
    print(f"\nSEPARATION          : {report['separation_mm']:.2f} mm")
    print(f"plane angle         : {angle:.3f} deg")
    print("=" * 58)

    if angle > 1.5:
        print("\n  WARNING: planes differ by more than 1.5 deg. Either a "
              "surface is not level or a board was not lying flat. A single "
              "separation number does not represent the pair.")
    if max(a_deck["rms_px_max"], a_floor["rms_px_max"]) > 1.0:
        print("\n  WARNING: pose reprojection error above 1 px. Check that the "
              "board parameters match those used in calibration.")
    weakest = min(a_deck["n_corners_min"], a_floor["n_corners_min"])
    if weakest < 0.3 * max_corners:
        print(f"\n  WARNING: as few as {weakest} of {max_corners} corners in "
              f"some views. Partial detections tilt the plane fit; move the "
              f"board closer or improve lighting.")

    print(f"\nFeed these to plane_separation.py:")
    print(f"  --gt-separation {report['separation_m']:.4f} "
          f"--gt-distance {report['deck']['surface_centroid_dist_m']:.4f}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()