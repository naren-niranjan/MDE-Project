"""Per-camera intrinsic calibration from captured ChArUco images.

Runs over every camera in config.CAMERAS (currently four: left, center, top,
right). For each one this script:
  * Detects ChArUco corners (sub-pixel refined) in captures/<name>/*.png,
    trying raw grayscale and CLAHE (matching the capture-time strategies, so
    frames the capture gate accepted don't come back corner-starved offline)
  * Drops degenerate views (collinear corners) that would crash calibrateCamera
  * Calibrates with the PLAIN model, seeded with the KNOWN optics
    (CALIB_USE_INTRINSIC_GUESS) and constrained with CALIB_FIX_ASPECT_RATIO
    (square pixels) and CALIB_FIX_K3 (k3 overfits with few views)
  * GUARDS the focal: if fx lands outside [0.8, 1.25] x the value implied by
    the lens + pixel pitch, the solve has collapsed into the focal/distance
    degeneracy (insufficient tilt diversity) and the result is REJECTED rather
    than written. fx = 13812 on an 8 mm / 3.45 um camera is not a calibration,
    it's a 47 mm lens that doesn't exist.
  * Rejects outlier views (> OUTLIER_MEDIAN_MULT x median per-view error) and
    refits -- a couple of blurred or misdetected frames at 2x the mean error
    routinely drag RMS from ~1.1 to ~1.6
  * Sanity-checks that the recovered distortion is monotonic & pole-free
  * Writes results/intrinsics_<name>.json, tagged with the camera's SERIAL

Reprojection-error guidance for a 2448x2048 IMX264 sensor:
  < 0.30 px  great   |   0.30-0.5 acceptable   |   > 0.5 reshoot

If the plain model leaves you above ~0.5 px, the fix is MORE / better-spread
views (real tilt about BOTH axes, edges of the frame, three distances), not
more distortion coefficients.

SERIAL TAGGING: the friendly names were remapped when the fourth camera went
in -- 261100628 used to be "left" and is now "center", 261100627 used to be
"center" and is now "right". A results/intrinsics_<name>.json written before
that change describes a DIFFERENT physical camera than the one that name now
points at, and silently reusing it would poison every downstream extrinsic and
depth result. Each output now carries its serial, and a mismatch against an
existing file is a hard error telling you to clear results/.

Cross-camera check: four nominally identical Tritons should land within a
couple of percent of each other on fx. A wide spread means the lenses are not
actually interchangeable and per-camera intrinsics are load-bearing (they
already are here), but a spread that appears suddenly usually means one camera
had a weak view set rather than a genuinely different lens. The summary at the
end reports it.

Usage:
  python calibrate_intrinsics.py                       # all four
  python calibrate_intrinsics.py --cameras top         # just the reshot one
  python calibrate_intrinsics.py --cameras left right --force
"""
import os, glob, json, argparse
import cv2
import numpy as np
import config

MIN_CORNERS = 12   # a view with < 12 corners barely helps
MIN_VIEWS   = 8

# === Known optics: the guard rail against focal collapse ====================
LENS_MM       = 8.0       # nominal lens focal length
PIXEL_SIZE_MM = 0.00345   # IMX264: 3.45 um
EXPECTED_FX   = LENS_MM / PIXEL_SIZE_MM          # ~2319 px
FX_RATIO_LO   = 0.8
FX_RATIO_HI   = 1.25

# Views with per-view RMS above this multiple of the median are dropped and
# the calibration re-run once.
OUTLIER_MEDIAN_MULT = 2.5

# Relative fx spread across the rig above which the summary complains.
FX_SPREAD_WARN = 0.03

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def detect_best(gray, detector):
    """Detect on raw grayscale and CLAHE-enhanced; keep whichever finds more
    corners. Mirrors the capture-time multi-strategy detection (the two
    strategies that matter offline) so the offline detector is never weaker
    than the gate that accepted the frame."""
    best = (None, None, 0)
    for im in (gray, _CLAHE.apply(gray)):
        cc, ids, _, _ = detector.detectBoard(im)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            best = (cc, ids, n)
    return best[0], best[1]


def detect_views(image_dir, detector, min_corners=MIN_CORNERS):
    """Return (object_points_list, image_points_list, image_size, used_files, n_total)."""
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No PNG images in {image_dir}")

    board = config.get_charuco_board()
    obj_pts_list, img_pts_list, used = [], [], []
    image_size = None

    for f in files:
        # cv2.imread returns BGR, which is the order auto_capture.py wrote the
        # camera's RGB frames in, so this round-trips exactly.
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (w, h)

        ch_corners, ch_ids = detect_best(gray, detector)
        if ch_ids is None or len(ch_ids) < min_corners:
            print(f"  skip {os.path.basename(f)} "
                  f"({0 if ch_ids is None else len(ch_ids)} corners)")
            continue

        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < min_corners:
            continue

        # calibrateCamera seeds intrinsics via a per-view homography
        # (initIntrinsicParams2D). Near-collinear corners make findHomography
        # return nothing -> cryptic "matH0.size() == Size(3, 3)" assertion.
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


def per_view_errors(obj_list, img_list, rvecs, tvecs, K, dist):
    """Per-view RMS reprojection error: cv2.norm(NORM_L2) = sqrt(sum dx^2+dy^2),
    so divide by sqrt(N) -- NOT N -- to get RMS distance per point."""
    errs = []
    for i in range(len(obj_list)):
        proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(img_list[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        errs.append(float(e))
    return np.array(errs)


def run_calibration(obj_list, img_list, image_size):
    """One calibrateCamera pass with the known-optics seed and the constrained
    plain model. Seeding + CALIB_USE_INTRINSIC_GUESS keeps the optimizer near
    the physical focal instead of letting the focal/distance degeneracy walk it
    to 6x reality on low-tilt data."""
    w, h = image_size
    K0 = np.array([[EXPECTED_FX, 0,           w / 2],
                   [0,           EXPECTED_FX, h / 2],
                   [0,           0,           1]], dtype=np.float64)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS
             | cv2.CALIB_FIX_ASPECT_RATIO   # square-pixel sensor: fx == fy.
                                            # (the 3.5% fx/fy split seen on
                                            # SN 261100627 was another
                                            # collapse symptom -- note that
                                            # camera is now "right", it was
                                            # "center" under the old naming)
             | cv2.CALIB_FIX_K3)            # k3 overfits the distortion tail
                                            # with tens of views
    return cv2.calibrateCamera(obj_list, img_list, image_size,
                               K0, None, flags=flags)


def distortion_is_sane(K, dist, image_size, margin=1.10):
    """True if the radial distortion map is monotonic & pole-free out to the
    sensor corner (with a small margin)."""
    d = np.asarray(dist).ravel()
    k = list(d[:8]) + [0.0] * max(0, 8 - len(d[:8]))
    k1, k2, p1, p2, k3, k4, k5, k6 = k[:8]
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    w, h = image_size
    rmax = np.hypot(max(cx, w - cx), max(cy, h - cy)) / fx * margin
    r = np.linspace(1e-6, rmax, 500); r2 = r * r
    num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
    rd = r * num / den
    return bool(np.all(den > 0) and np.all(np.diff(rd) > 0)), rmax


def check_stale_result(name, serial, force):
    """Refuse to overwrite an intrinsics file that was computed for a DIFFERENT
    physical camera under the same friendly name. Files predating the serial
    tagging carry no serial at all and are equally suspect."""
    path = os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            old = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    old_serial = old.get("serial")
    if old_serial == serial:
        return
    msg = (f"{name}: results/intrinsics_{name}.json was written for serial "
           f"{old_serial or '<untagged, pre-4-camera>'} but '{name}' now means "
           f"serial {serial}. The friendly names were remapped when the fourth "
           f"camera was added, so that file describes a different camera.")
    if force:
        print(f"  ** {msg} Overwriting because --force. **")
    else:
        raise RuntimeError(
            msg + " Archive or delete results/ and recalibrate the whole rig, "
                  "or pass --force if you are certain. Nothing written.")


def calibrate_camera(name, detector, force=False):
    serial = config.CAMERAS[name]
    print(f"\n=== {name}  (SN {serial}) ===")
    check_stale_result(name, serial, force)
    image_dir = os.path.join(config.CAPTURE_DIR, name)

    obj_list, img_list, image_size, used, n_total = detect_views(image_dir, detector)
    if len(obj_list) < MIN_VIEWS:
        raise RuntimeError(
            f"Only {len(obj_list)} usable views for {name} -- capture more "
            "(aim for 40-60 per camera).")
    print(f"  {len(used)} usable views (of {n_total})")
    if len(obj_list) < 40:
        print(f"  NOTE: {len(obj_list)} views is below the 40-60 target -- "
              "expect weaker distortion constraint at the edges.")

    rms, K, dist, rvecs, tvecs = run_calibration(obj_list, img_list, image_size)
    per_view = per_view_errors(obj_list, img_list, rvecs, tvecs, K, dist)

    # ---- outlier rejection: drop views far above the median, refit once ----
    med = np.median(per_view)
    keep = per_view <= OUTLIER_MEDIAN_MULT * med
    n_dropped = int((~keep).sum())
    if n_dropped and keep.sum() >= MIN_VIEWS:
        for f, e, k in zip(used, per_view, keep):
            if not k:
                print(f"  outlier: {os.path.basename(f)} "
                      f"({e:.2f} px > {OUTLIER_MEDIAN_MULT:.1f} x median "
                      f"{med:.2f} px) -- dropped")
        obj_list = [o for o, k in zip(obj_list, keep) if k]
        img_list = [p for p, k in zip(img_list, keep) if k]
        used     = [f for f, k in zip(used, keep) if k]
        rms, K, dist, rvecs, tvecs = run_calibration(obj_list, img_list,
                                                     image_size)
        per_view = per_view_errors(obj_list, img_list, rvecs, tvecs, K, dist)
        print(f"  refit after dropping {n_dropped} outlier view(s)")

    # ---- focal guard rail ----
    fx = K[0, 0]
    ratio = fx / EXPECTED_FX
    print(f"  fx = {fx:.1f} px vs expected {EXPECTED_FX:.0f} px "
          f"({LENS_MM:g} mm / {PIXEL_SIZE_MM*1e3:g} um) -> ratio {ratio:.3f}")
    if not (FX_RATIO_LO < ratio < FX_RATIO_HI):
        raise RuntimeError(
            f"{name}: focal collapsed (fx={fx:.0f}, expected ~{EXPECTED_FX:.0f}). "
            f"This is the planar focal/distance degeneracy -- the views lack "
            f"tilt diversity. Reshoot with deliberate +/-30-45 deg tilts about "
            f"BOTH axes at three distances. Result NOT written.")

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
        "serial"                     : serial,
        "image_size"                 : [int(image_size[0]), int(image_size[1])],
        "camera_matrix"              : K.tolist(),
        "dist_coeffs"                : dist.flatten().tolist(),
        "distortion_model"           : "radial_tangential_k1k2p1p2 (k3 fixed 0)",
        "rms_reprojection_error_px"  : float(rms),
        "mean_per_view_error_px"     : float(per_view.mean()),
        "max_per_view_error_px"      : float(per_view.max()),
        "expected_fx_px"             : float(EXPECTED_FX),
        "fx_over_expected"           : float(ratio),
        "distortion_sane"            : bool(sane),
        "n_views_used"               : int(len(obj_list)),
        "n_outliers_dropped"         : n_dropped,
        "views_used"                 : [os.path.basename(p) for p in used],
    }


def summarize(results):
    """Side-by-side table across the rig, plus the fx-spread check.

    Worth reading every run: four cameras make it easy for one weak view set to
    hide behind three good ones, and the per-camera prints scroll past. A
    camera whose fx sits well off the others while its RMS looks fine is the
    classic partial-collapse signature -- passing the ratio guard but still
    solving a different lens than it has.
    """
    if len(results) < 2:
        return
    print("\n=== rig summary ===")
    print(f"  {'camera':8s} {'serial':11s} {'fx':>9s} {'fy':>9s} "
          f"{'cx':>8s} {'cy':>8s} {'RMS px':>8s} {'views':>6s}")
    for r in results:
        K = np.array(r["camera_matrix"])
        print(f"  {r['camera']:8s} {r['serial']:11s} "
              f"{K[0,0]:9.1f} {K[1,1]:9.1f} {K[0,2]:8.1f} {K[1,2]:8.1f} "
              f"{r['rms_reprojection_error_px']:8.3f} "
              f"{r['n_views_used']:6d}")

    fxs = np.array([r["camera_matrix"][0][0] for r in results])
    spread = float((fxs.max() - fxs.min()) / fxs.mean())
    lo = results[int(np.argmin(fxs))]["camera"]
    hi = results[int(np.argmax(fxs))]["camera"]
    print(f"\n  fx spread across the rig: {spread*100:.1f}% "
          f"({lo} lowest, {hi} highest)")
    if spread > FX_SPREAD_WARN:
        print(f"  ** {spread*100:.1f}% is wide for nominally identical "
              f"cameras. Either the lenses genuinely differ (so per-camera "
              f"intrinsics are essential and cross-view scale needs watching) "
              f"or the outlying camera's views lack tilt diversity and its "
              f"focal is partially collapsed. Compare '{hi}' and '{lo}' view "
              f"counts and per-view errors above before trusting either. **")

    worst = max(results, key=lambda r: r["rms_reprojection_error_px"])
    if worst["rms_reprojection_error_px"] > 0.5:
        print(f"  ** {worst['camera']} is above the 0.5 px reshoot threshold "
              f"({worst['rms_reprojection_error_px']:.3f} px). **")


def main():
    p = argparse.ArgumentParser(
        description="Per-camera intrinsic calibration for the four-Triton rig.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cameras", nargs="+", default=list(config.CAMERAS),
                   choices=list(config.CAMERAS),
                   help="subset to calibrate; the others keep their existing "
                        "results/intrinsics_<name>.json")
    p.add_argument("--force", action="store_true",
                   help="overwrite an intrinsics file whose recorded serial "
                        "does not match the current config mapping")
    args = p.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    detector = config.get_charuco_detector()
    names = [n for n in config.CAMERAS if n in set(args.cameras)]

    results, failed = [], []
    for name in names:
        try:
            res = calibrate_camera(name, detector, args.force)
        except RuntimeError as e:
            print(f"  ** {e} **")
            failed.append(name)
            continue
        out = os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  -> {out}")
        results.append(res)

    # Pull in cameras that weren't recalibrated this run so the summary still
    # describes the whole rig -- but only ones whose serial still matches.
    if len(names) < len(config.CAMERAS):
        for name in config.CAMERAS:
            if name in names:
                continue
            path = os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    old = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if old.get("serial") == config.CAMERAS[name]:
                results.append(old)
            else:
                print(f"\n  NOTE: skipping {name} in the summary -- its stored "
                      f"result is for serial {old.get('serial') or '<untagged>'}"
                      f", not {config.CAMERAS[name]}.")
        results.sort(key=lambda r: list(config.CAMERAS).index(r["camera"]))

    summarize(results)

    if failed:
        print(f"\nFAILED cameras (no intrinsics written): {', '.join(failed)}")
        print("Do NOT run extrinsics until every camera passes.")


if __name__ == "__main__":
    main()