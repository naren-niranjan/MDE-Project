"""Per-camera intrinsic calibration from captured ChArUco images.

For every camera listed in config.CAMERAS, this script:
  * Detects ChArUco corners in every image in captures/<name>/
  * Builds object/image point lists
  * Calls cv2.calibrateCamera with the rational distortion model (8 coeffs)
  * Writes results/intrinsics_<name>.json

Reprojection-error guidance for a 2448x2048 IMX264 sensor:
  < 0.30 px  great
  0.30-0.5   acceptable
  > 0.5      reshoot — usually too few views, board too small in frame, or motion blur
"""
import os, glob, json
import cv2
import numpy as np
import config


def detect_views(image_dir, charuco_detector, min_corners=8):
    """Return (object_points_list, image_points_list, image_size, used_files)."""
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No PNG images in {image_dir}")

    board = config.get_charuco_board()
    obj_pts_list, img_pts_list, used = [], [], []
    image_size = None

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (w, h)

        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            print(f"  skip {os.path.basename(f)} ({0 if ch_ids is None else len(ch_ids)} corners)")
            continue

        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < min_corners:
            continue

        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)
        used.append(f)

    return obj_pts_list, img_pts_list, image_size, used


def calibrate_camera(name):
    print(f"\n=== {name} ===")
    image_dir = os.path.join(config.CAPTURE_DIR, name)
    detector  = cv2.aruco.CharucoDetector(config.get_charuco_board())

    obj_list, img_list, image_size, used = detect_views(image_dir, detector)
    if len(obj_list) < 8:
        raise RuntimeError(
            f"Only {len(obj_list)} usable views for {name} -- capture more "
            "(aim for at least 20).")
    print(f"  {len(used)} usable views (of {len(used) + 0})")

    # Rational 8-coeff distortion model; useful when there's noticeable lens distortion.
    flags = cv2.CALIB_RATIONAL_MODEL
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, image_size, None, None, flags=flags)

    # Per-view error
    per_view = []
    for i in range(len(obj_list)):
        proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(img_list[i], proj, cv2.NORM_L2) / len(proj)
        per_view.append(float(e))
    per_view = np.array(per_view)

    print(f"  overall RMS reproj error : {rms:.4f} px")
    print(f"  mean per-view error      : {per_view.mean():.4f} px")
    print(f"  max  per-view error      : {per_view.max():.4f} px")
    print(f"  fx, fy = {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"  cx, cy = {K[0,2]:.2f}, {K[1,2]:.2f}")

    return {
        "camera"                     : name,
        "image_size"                 : [int(image_size[0]), int(image_size[1])],
        "camera_matrix"              : K.tolist(),
        "dist_coeffs"                : dist.flatten().tolist(),
        "distortion_model"           : "rational_polynomial_k1k2p1p2k3k4k5k6",
        "rms_reprojection_error_px"  : float(rms),
        "mean_per_view_error_px"     : float(per_view.mean()),
        "max_per_view_error_px"      : float(per_view.max()),
        "n_views_used"               : int(len(obj_list)),
        "views_used"                 : [os.path.basename(p) for p in used],
    }


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for name in config.CAMERAS:
        res = calibrate_camera(name)
        out = os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
