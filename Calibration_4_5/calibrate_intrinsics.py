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
    than written.
  * Rejects outlier views (> OUTLIER_MEDIAN_MULT x median per-view error) and
    refits -- a couple of blurred or misdetected frames at 2x the mean error
    routinely drag RMS from ~1.1 to ~1.6
  * Sanity-checks that the recovered distortion is monotonic & pole-free, and
    compares its magnitude against the lens datasheet
  * Writes results/intrinsics_<name>.json, tagged with the camera's SERIAL and
    the LENS it was shot through

LENS CHANGE (12 mm Edmund Optics C Series, EO 58-001)
-----------------------------------------------------
The optics constants no longer live in this file. LENS_MM, PIXEL_SIZE_MM and
the derived EXPECTED_FX all come from config.py, which is the single place a
lens swap has to be recorded. What changed with the 12 mm lens:

  * EXPECTED_FX moves 2319 px -> 3478 px. The guard is a ratio against that,
    so an old-lens answer near 2319 px is now correctly rejected: 0.67 x
    expected falls outside the band. fx = 13812 on a 12 mm / 3.45 um camera is
    not a calibration, it's a 47 mm lens that doesn't exist.
  * The datasheet quotes < 2.5 % distortion at maximum field, which is modest.
    The solved radial map is now checked against that number: a fit reporting
    8 % on a lens spec'd at 2.5 % is fitting noise or bad correspondences, not
    glass. See distortion_magnitude_pct().
  * The image circle (2/3", 11.0 mm) barely covers the 11.011 mm sensor
    diagonal, so illumination falls off at the frame corners and detections
    there get scarcer. Those are exactly the corners that constrain k1/k2, so
    if the distortion sanity check starts failing, the cause is usually a view
    set with nothing near the edges rather than a genuinely odd lens.
  * EVERY pre-lens-change capture and result is invalid. captures/,
    captures_extrinsic/ and results/ must be archived or deleted before this is
    run again; a mixed-lens view set produces a plausible-looking RMS around a
    focal length that describes neither lens.

Reprojection-error guidance for a 2448x2048 IMX264 sensor:
  < 0.30 px  great   |   0.30-0.5 acceptable   |   > 0.5 reshoot

If the plain model leaves you above ~0.5 px, the fix is MORE / better-spread
views (real tilt about BOTH axes, edges of the frame, three distances between
roughly 1.8 and 3.4 m with this lens), not more distortion coefficients.

SERIAL AND LENS TAGGING: the friendly names were remapped when the fourth
camera went in -- 261100628 used to be "left" and is now "center", 261100627
used to be "center" and is now "right". A results/intrinsics_<name>.json
written before that change describes a DIFFERENT physical camera than the one
that name now points at, and one written before the lens change describes
different glass. Either would silently poison every downstream extrinsic and
depth result. Each output carries its serial AND its lens id, and a mismatch
against an existing file is a hard error telling you to clear results/.

Cross-camera check: four nominally identical Tritons with four nominally
identical lenses should land within a couple of percent of each other on fx.
A wide spread means the lenses are not actually interchangeable and per-camera
intrinsics are load-bearing (they already are here), but a spread that appears
suddenly usually means one camera had a weak view set rather than a genuinely
different lens. With four freshly fitted C Series units, also check that every
lens is focused and locked at the same distance -- focus position shifts the
effective focal length, and an unlocked ring is the most common cause of one
camera drifting away from the other three. The summary at the end reports it.

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
# Sourced from config.py so this script and auto_capture.py cannot disagree
# about which lens is fitted.
LENS_MM       = config.LENS_MM              # 12.0
PIXEL_SIZE_MM = config.PIXEL_SIZE_MM        # 0.00345 (IMX264, 3.45 um)
EXPECTED_FX   = config.EXPECTED_FX          # ~3478 px
FX_RATIO_LO   = config.FX_RATIO_LO
FX_RATIO_HI   = config.FX_RATIO_HI

# Datasheet distortion at maximum field, and the multiple of it above which the
# solved map is treated as suspicious rather than merely different.
DIST_SPEC_PCT      = config.LENS_DISTORTION_SPEC_PCT   # 2.5 %
DIST_SPEC_TOLERANCE = 2.0

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
    than the gate that accepted the frame. CLAHE earns its place with this
    lens: the corners of the frame sit right at the edge of the image circle
    and are measurably dimmer than the centre."""
    best = (None, None, 0)
    for im in (gray, _CLAHE.apply(gray)):
        cc, ids, _, _ = detector.detectBoard(im)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            best = (cc, ids, n)
    return best[0], best[1]


def detect_views(image_dir, detector, min_corners=MIN_CORNERS):
    """Return (object_points_list, image_points_list, image_size, used_files,
    n_total, coverage)."""
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No PNG images in {image_dir}")

    board = config.get_charuco_board()
    obj_pts_list, img_pts_list, used = [], [], []
    image_size = None
    all_pts = []

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
        all_pts.append(img_xy)

    coverage = frame_coverage(all_pts, image_size) if all_pts else 0.0
    return obj_pts_list, img_pts_list, image_size, used, len(files), coverage


def frame_coverage(all_pts, image_size):
    """Fraction of an 8x8 grid over the image that saw at least one corner.

    Worth reporting on its own now. The 12 mm lens' image circle only just
    covers the sensor, the corners of the frame are dimmer, and a shooter who
    keeps the board comfortably in the middle will get a low RMS from a view
    set that never constrained the distortion tail at all. Below ~0.6 the
    distortion coefficients are extrapolating, not fitting.
    """
    if image_size is None:
        return 0.0
    w, h = image_size
    grid = np.zeros((8, 8), bool)
    for pts in all_pts:
        gx = np.clip((pts[:, 0] / w * 8).astype(int), 0, 7)
        gy = np.clip((pts[:, 1] / h * 8).astype(int), 0, 7)
        grid[gy, gx] = True
    return float(grid.mean())


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
    to several times reality on low-tilt data."""
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
                                            # with tens of views, and this lens
                                            # is spec'd under 2.5 % anyway
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


def distortion_magnitude_pct(K, dist, image_size):
    """Radial distortion at maximum field, as a percentage, comparable with
    the lens datasheet figure.

    The C Series 12 mm is quoted at < 2.5 %. This is not an exact like-for-like
    (manufacturers quote optical or TV distortion against slightly different
    references) but it is the right order of magnitude, and it catches the case
    where the solve has absorbed detection error or a mixed-lens view set into
    k1/k2 and produced a distortion curve no physical lens has.
    """
    d = np.asarray(dist).ravel()
    k = list(d[:5]) + [0.0] * max(0, 5 - len(d[:5]))
    k1, k2, p1, p2, k3 = k[:5]
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    w, h = image_size
    r = np.hypot(max(cx, w - cx), max(cy, h - cy)) / fx
    r2 = r * r
    rd = r * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    return float(abs(rd - r) / r * 100.0)


def check_stale_result(name, serial, force):
    """Refuse to overwrite an intrinsics file computed for a DIFFERENT physical
    camera, or through DIFFERENT glass, under the same friendly name. Files
    predating the tagging carry neither field and are equally suspect."""
    path = os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            old = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    problems = []
    old_serial = old.get("serial")
    if old_serial != serial:
        problems.append(
            f"it was written for serial {old_serial or '<untagged, pre-4-camera>'} "
            f"but '{name}' now means serial {serial} (the friendly names were "
            f"remapped when the fourth camera was added)")
    old_lens = old.get("lens_id")
    if old_lens != config.LENS_ID:
        problems.append(
            f"it was shot through lens {old_lens or '<untagged, pre-12mm>'} "
            f"but the rig now carries {config.LENS_ID} "
            f"({config.LENS_MM:g} mm)")
    if not problems:
        return

    msg = (f"{name}: results/intrinsics_{name}.json is stale -- " +
           "; ".join(problems) + ".")
    if force:
        print(f"  ** {msg} Overwriting because --force. **")
    else:
        raise RuntimeError(
            msg + " Archive or delete results/ (and the capture directories, "
                  "which are equally lens-specific) and recalibrate the whole "
                  "rig, or pass --force if you are certain. Nothing written.")


def calibrate_camera(name, detector, force=False):
    serial = config.CAMERAS[name]
    print(f"\n=== {name}  (SN {serial}, {config.LENS_MM:g} mm) ===")
    check_stale_result(name, serial, force)
    image_dir = os.path.join(config.CAPTURE_DIR, name)

    (obj_list, img_list, image_size, used,
     n_total, coverage) = detect_views(image_dir, detector)
    if len(obj_list) < MIN_VIEWS:
        raise RuntimeError(
            f"Only {len(obj_list)} usable views for {name} -- capture more "
            "(aim for 40-60 per camera).")
    print(f"  {len(used)} usable views (of {n_total})")
    if len(obj_list) < 40:
        print(f"  NOTE: {len(obj_list)} views is below the 40-60 target -- "
              "expect weaker distortion constraint at the edges.")
    print(f"  frame coverage: {coverage*100:.0f}% of an 8x8 grid saw a corner")
    if coverage < 0.60:
        print("  ** Thin coverage. With this lens the corners of the frame are "
              "dim and easy to under-sample, and they are what constrains k1/k2. "
              "Reshoot with the board deliberately pushed into all four corners "
              "and along the edges. **")

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
        hint = ""
        old_fx = 8.0 / PIXEL_SIZE_MM
        if abs(fx - old_fx) / old_fx < 0.10:
            hint = (" This is suspiciously close to the OLD 8 mm lens' focal "
                    f"({old_fx:.0f} px) -- are these captures from before the "
                    f"lens change?")
        raise RuntimeError(
            f"{name}: focal outside the guard band (fx={fx:.0f}, expected "
            f"~{EXPECTED_FX:.0f}). Usually the planar focal/distance "
            f"degeneracy -- the views lack tilt diversity. Reshoot with "
            f"deliberate +/-30-45 deg tilts about BOTH axes at three "
            f"distances.{hint} Result NOT written.")

    sane, rmax = distortion_is_sane(K, dist, image_size)
    dist_pct = distortion_magnitude_pct(K, dist, image_size)

    print(f"  overall RMS reproj error : {rms:.4f} px")
    print(f"  mean per-view error      : {per_view.mean():.4f} px")
    print(f"  max  per-view error      : {per_view.max():.4f} px")
    print(f"  fx, fy = {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"  cx, cy = {K[0,2]:.2f}, {K[1,2]:.2f}")
    print(f"  distortion sane to r={rmax:.2f}? {sane}")
    print(f"  radial distortion at max field: {dist_pct:.2f}% "
          f"(datasheet < {DIST_SPEC_PCT:g}%)")
    if not sane:
        print("  ** WARNING: non-monotonic / pole in distortion map -- overfit; "
              "do not trust. Capture more views or reduce the model. **")
    if dist_pct > DIST_SPEC_TOLERANCE * DIST_SPEC_PCT:
        print(f"  ** WARNING: {dist_pct:.1f}% is far above the {DIST_SPEC_PCT:g}% "
              f"the lens is specified at. The fit has absorbed something that "
              f"is not glass -- most often edge corners detected badly in the "
              f"dim frame corners, or captures mixed across two lenses. **")

    return {
        "camera"                     : name,
        "serial"                     : serial,
        "lens_id"                    : config.LENS_ID,
        "lens_name"                  : config.LENS_NAME,
        "lens_mm"                    : float(config.LENS_MM),
        "pixel_size_mm"              : float(PIXEL_SIZE_MM),
        "aperture"                   : config.CALIBRATION_APERTURE,
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
        "distortion_max_field_pct"   : float(dist_pct),
        "frame_coverage"             : float(coverage),
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
    solving a different lens than it has. With four newly fitted C Series
    lenses the other candidate is an unlocked focus ring: focus position moves
    the effective focal length, so a camera that was refocused after the others
    reads differently for an entirely mechanical reason.
    """
    if len(results) < 2:
        return
    print("\n=== rig summary ===")
    print(f"  {'camera':8s} {'serial':11s} {'fx':>9s} {'fy':>9s} "
          f"{'cx':>8s} {'cy':>8s} {'RMS px':>8s} {'dist%':>6s} {'views':>6s}")
    for r in results:
        K = np.array(r["camera_matrix"])
        print(f"  {r['camera']:8s} {r['serial']:11s} "
              f"{K[0,0]:9.1f} {K[1,1]:9.1f} {K[0,2]:8.1f} {K[1,2]:8.1f} "
              f"{r['rms_reprojection_error_px']:8.3f} "
              f"{r.get('distortion_max_field_pct', float('nan')):6.2f} "
              f"{r['n_views_used']:6d}")

    fxs = np.array([r["camera_matrix"][0][0] for r in results])
    spread = float((fxs.max() - fxs.min()) / fxs.mean())
    lo = results[int(np.argmin(fxs))]["camera"]
    hi = results[int(np.argmax(fxs))]["camera"]
    print(f"\n  fx spread across the rig: {spread*100:.1f}% "
          f"({lo} lowest, {hi} highest); expected fx "
          f"{EXPECTED_FX:.0f} px for {LENS_MM:g} mm")
    if spread > FX_SPREAD_WARN:
        print(f"  ** {spread*100:.1f}% is wide for nominally identical "
              f"cameras and lenses. Either the units genuinely differ (so "
              f"per-camera intrinsics are essential and cross-view scale needs "
              f"watching), or one focus ring is set differently, or the "
              f"outlying camera's views lack tilt diversity and its focal is "
              f"partially collapsed. Compare '{hi}' and '{lo}' view counts, "
              f"coverage and per-view errors above before trusting either. **")

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
                   help="overwrite an intrinsics file whose recorded serial or "
                        "lens does not match the current config")
    args = p.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(config.describe_optics())
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
    # describes the whole rig -- but only ones whose serial AND lens match.
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
            if (old.get("serial") == config.CAMERAS[name]
                    and old.get("lens_id") == config.LENS_ID):
                results.append(old)
            else:
                print(f"\n  NOTE: skipping {name} in the summary -- its stored "
                      f"result is for serial {old.get('serial') or '<untagged>'} "
                      f"/ lens {old.get('lens_id') or '<untagged>'}, not "
                      f"{config.CAMERAS[name]} / {config.LENS_ID}.")
        results.sort(key=lambda r: list(config.CAMERAS).index(r["camera"]))

    summarize(results)

    if failed:
        print(f"\nFAILED cameras (no intrinsics written): {', '.join(failed)}")
        print("Do NOT run extrinsics until every camera passes.")


if __name__ == "__main__":
    main()