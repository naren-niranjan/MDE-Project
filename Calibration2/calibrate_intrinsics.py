"""Per-camera intrinsic calibration from captured ChArUco images.

For every camera listed in config.CAMERAS, this script:
  * Detects ChArUco corners in every image in captures/<name>/
  * Drops degenerate views (collinear corners) that would crash calibrateCamera
  * Builds object/image point lists
  * Calls cv2.calibrateCamera with the PLAIN 5-coeff distortion model
    (k1,k2,p1,p2,k3) -- the rational 8-coeff model overfits low-distortion
    machine-vision lenses and can produce non-physical (pole/folding) maps
  * Sanity-checks that the recovered distortion is monotonic & pole-free
  * Writes results/intrinsics_<name>.json

Reprojection-error guidance for a 2448x2048 IMX264 sensor:
  < 0.30 px  great
  0.30-0.5   acceptable
  > 0.5      reshoot -- usually too few views, board too small in frame, or motion blur

If the plain model leaves you above ~0.5 px, the fix is MORE / better-spread
views, not more distortion coefficients.
"""
import os, glob, json
import cv2
import numpy as np
import config

MIN_CORNERS = 12   # stricter than before; a view with <12 corners barely helps


def detect_views(image_dir, charuco_detector, min_corners=MIN_CORNERS):
    """Return (object_points_list, image_points_list, image_size, used_files, n_total)."""
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
            print(f"  skip {os.path.basename(f)} "
                  f"({0 if ch_ids is None else len(ch_ids)} corners)")
            continue

        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < min_corners:
            continue

        # calibrateCamera seeds intrinsics via a per-view homography
        # (initIntrinsicParams2D). If a view's corners are (near-)collinear,
        # findHomography returns nothing and you get the cryptic
        # "matH0.size() == Size(3, 3)" assertion. Test it here and skip.
        obj_xy = obj_pts.reshape(-1, 3)[:, :2].astype(np.float32)
        img_xy = img_pts.reshape(-1, 2).astype(np.float32)
        H, _ = cv2.findHomography(obj_xy, img_xy)
        if H is None:
            print(f"  skip {os.path.basename(f)} (degenerate / collinear corners)")
            continue

        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)
        used.append(f)

    return obj_pts_list, img_pts_list, image_size, used, len(files)


def distortion_is_sane(K, dist, image_size, margin=1.10):
    """True if the radial distortion map is monotonic & pole-free out to the
    sensor corner (with a small margin). Catches overfit models that fit the
    calibration corners but are non-physical everywhere else."""
    d = np.asarray(dist).ravel()
    k = list(d[:8]) + [0.0] * max(0, 8 - len(d[:8]))
    k1, k2, p1, p2, k3, k4, k5, k6 = k[:8]
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    w, h = image_size
    rmax = np.hypot(max(cx, w - cx), max(cy, h - cy)) / fx * margin
    r = np.linspace(1e-6, rmax, 500); r2 = r * r
    num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3   # == 1 for the plain model
    rd = r * num / den
    return bool(np.all(den > 0) and np.all(np.diff(rd) > 0)), rmax


def calibrate_camera(name):
    print(f"\n=== {name} ===")
    image_dir = os.path.join(config.CAPTURE_DIR, name)
    detector  = cv2.aruco.CharucoDetector(config.get_charuco_board())

    obj_list, img_list, image_size, used, n_total = detect_views(image_dir, detector)
    if len(obj_list) < 8:
        raise RuntimeError(
            f"Only {len(obj_list)} usable views for {name} -- capture more "
            "(aim for at least 20).")
    print(f"  {len(used)} usable views (of {n_total})")

    # Plain radial-tangential model: k1, k2, p1, p2, k3. No rational denominator
    # to overfit. If you genuinely have strong distortion and this leaves RMS
    # high, capture more/better views rather than re-enabling CALIB_RATIONAL_MODEL.
    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, image_size, None, None, flags=flags)

    # Per-view RMS reprojection error (comparable to the overall RMS):
    # cv2.norm(..., NORM_L2) = sqrt(sum of dx^2+dy^2) over the view, so divide
    # by sqrt(N) -- NOT N -- to get root-mean-square distance per point.
    per_view = []
    for i in range(len(obj_list)):
        proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(img_list[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        per_view.append(float(e))
    per_view = np.array(per_view)

    sane, rmax = distortion_is_sane(K, dist, image_size)

    print(f"  overall RMS reproj error : {rms:.4f} px")
    print(f"  mean per-view error      : {per_view.mean():.4f} px")
    print(f"  max  per-view error      : {per_view.max():.4f} px")
    print(f"  fx, fy = {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"  cx, cy = {K[0,2]:.2f}, {K[1,2]:.2f}")
    print(f"  distortion sane to r={rmax:.2f}? {sane}")
    if not sane:
        print("  ** WARNING: non-monotonic / pole in distortion map -- overfit; "
              "do not trust. Capture more views or reduce the model. **")

    return {
        "camera"                     : name,
        "image_size"                 : [int(image_size[0]), int(image_size[1])],
        "camera_matrix"              : K.tolist(),
        "dist_coeffs"                : dist.flatten().tolist(),
        "distortion_model"           : "radial_tangential_k1k2p1p2k3",
        "rms_reprojection_error_px"  : float(rms),
        "mean_per_view_error_px"     : float(per_view.mean()),
        "max_per_view_error_px"      : float(per_view.max()),
        "distortion_sane"            : bool(sane),
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