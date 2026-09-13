#!/usr/bin/env python3
"""
Intrinsic calibration from ChArUco frames (5-coefficient distortion model).

Board : calib.io ChArUco, 9x12 squares, square=60 mm, marker=47 mm, DICT_5X5_250
Input : grayscale PNGs from capture_charuco.py
Output: intrinsics.json (K, dist, image size, RMS, per-view errors, FOV) + console diagnostics

Notes:
  * Uses the default OpenCV distortion model => 5 coeffs (k1,k2,p1,p2,k3).
    No CALIB_RATIONAL_MODEL: rational poly overfits at typical frame counts.
  * Legacy ChArUco pattern is auto-detected (calib.io usually needs legacy=True on >=4.6).
  * Intrinsics are independent of the absolute square size; getting square/marker
    in meters only affects the reported board pose, not K/dist.
"""
import argparse, glob, json, math, os, sys
import numpy as np
import cv2

SQX, SQY = 12, 9   # squares (cols, rows) — verified against the physical board
SQUARE_M, MARKER_M = 0.060, 0.047
ARUCO_DICT = cv2.aruco.DICT_5X5_250
MAX_CORNERS = (SQX - 1) * (SQY - 1)


def make_board(legacy):
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    b = cv2.aruco.CharucoBoard((SQX, SQY), SQUARE_M, MARKER_M, adict)
    b.setLegacyPattern(legacy)
    return b


def decide_legacy(paths):
    """Sum charuco corners over a sample of images for each pattern; pick the winner."""
    sample = paths[:min(8, len(paths))]
    totals = {}
    for legacy in (False, True):
        det = cv2.aruco.CharucoDetector(make_board(legacy))
        s = 0
        for p in sample:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            cc, ci, mc, mi = det.detectBoard(g)
            s += 0 if cc is None else len(cc)
        totals[legacy] = s
    legacy = totals[True] >= totals[False]
    print(f"legacy auto-detect: corners False={totals[False]} True={totals[True]} -> legacy={legacy}")
    return legacy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='./captures')
    ap.add_argument('--glob', default='*.png')
    ap.add_argument('--out', default='intrinsics.json')
    ap.add_argument('--min-corners', type=int, default=12)
    ap.add_argument('--max-view-rms', type=float, default=1.0,
                    help='drop views above this reprojection error (px) and recalibrate once')
    ap.add_argument('--legacy', choices=['auto', 'true', 'false'], default='auto')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, args.glob)))
    if not paths:
        sys.exit(f"No images at {os.path.join(args.images, args.glob)}")

    legacy = {'true': True, 'false': False}.get(args.legacy) \
        if args.legacy != 'auto' else decide_legacy(paths)
    board = make_board(legacy)
    det = cv2.aruco.CharucoDetector(board)

    obj_pts, img_pts, used, img_size, n_markers0 = [], [], [], None, 0
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"  skip (unreadable): {p}"); continue
        if img_size is None:
            img_size = (g.shape[1], g.shape[0])  # (w, h)
        elif (g.shape[1], g.shape[0]) != img_size:
            print(f"  skip (size mismatch): {p}"); continue
        cc, ci, mc, mi = det.detectBoard(g)
        n = 0 if cc is None else len(cc)
        n_markers0 += 0 if mi is None else len(mi)
        if n < args.min_corners:
            print(f"  skip ({n} corners): {os.path.basename(p)}"); continue
        op, ip = board.matchImagePoints(cc, ci)
        if op is None or len(op) < args.min_corners:
            print(f"  skip (matchImagePoints): {os.path.basename(p)}"); continue
        obj_pts.append(op); img_pts.append(ip); used.append(p)
        print(f"  use  ({n:>2} corners): {os.path.basename(p)}")

    if n_markers0 == 0:
        sys.exit("No ArUco markers detected in any image -> wrong dictionary "
                 "(expected DICT_5X5_250). Verify with a brute-force dict probe.")
    if len(obj_pts) < 6:
        sys.exit(f"Only {len(obj_pts)} usable views; need ~15-25 covering the full frustum.")

    def calibrate(objp, imgp):
        flags = 0  # default 5-coeff model (k1,k2,p1,p2,k3)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, img_size, None, None, flags=flags)
        per = []
        for o, i, r, t in zip(objp, imgp, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, r, t, K, dist)
            d = i.reshape(-1, 2) - proj.reshape(-1, 2)
            per.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))  # per-view RMS (px)
        return rms, K, dist, per

    rms, K, dist, per = calibrate(obj_pts, img_pts)
    print(f"\nInitial: RMS={rms:.4f}px over {len(obj_pts)} views")

    # drop worst views above threshold, recalibrate once
    keep = [k for k in range(len(per)) if per[k] <= args.max_view_rms]
    if 0 < len(keep) < len(per) and len(keep) >= 6:
        dropped = [(os.path.basename(used[k]), round(per[k], 3))
                   for k in range(len(per)) if k not in keep]
        print(f"Dropping {len(dropped)} view(s) > {args.max_view_rms}px: {dropped}")
        obj_pts = [obj_pts[k] for k in keep]
        img_pts = [img_pts[k] for k in keep]
        used = [used[k] for k in keep]
        rms, K, dist, per = calibrate(obj_pts, img_pts)
        print(f"Refit:   RMS={rms:.4f}px over {len(obj_pts)} views")

    w, h = img_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fov_x = math.degrees(2 * math.atan(w / (2 * fx)))   # you condition DA3 on fov_x (deg)
    fov_y = math.degrees(2 * math.atan(h / (2 * fy)))

    print("\nK =\n", np.round(K, 3))
    print("dist (k1 k2 p1 p2 k3) =", np.round(dist.ravel(), 6))
    print(f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"FOV: x={fov_x:.3f} deg  y={fov_y:.3f} deg")
    print("per-view RMS (px):", [round(e, 3) for e in per])

    out = {
        "camera_ip": "172.128.0.3",
        "camera_serial": "260505158",
        "model": "TRI050S-C",
        "opencv_version": cv2.__version__,
        "image_width": w, "image_height": h,
        "board": {"squaresX": SQX, "squaresY": SQY, "square_m": SQUARE_M,
                  "marker_m": MARKER_M, "dict": "DICT_5X5_250", "legacy_pattern": bool(legacy)},
        "distortion_model": "opencv_5coeff_k1k2p1p2k3",
        "rms_reproj_px": float(rms),
        "num_views": len(obj_pts),
        "K": K.tolist(),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "dist": dist.ravel().tolist(),
        "fov_x_deg": fov_x, "fov_y_deg": fov_y,
        "per_view_rms_px": per,
        "images_used": [os.path.basename(p) for p in used],
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()