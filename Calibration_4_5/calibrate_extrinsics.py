"""Multi-camera extrinsic calibration relative to the reference camera.

Each target camera is calibrated against the reference with cv2.stereoCalibrate
and CALIB_FIX_INTRINSIC, which JOINTLY minimizes reprojection error over all
shared frames and can't quietly launder bad intrinsics into a 16 m baseline.
This is strictly better than averaging independent per-frame solvePnP poses.

The per-frame relative-pose spread is still computed -- purely as a data-health
DIAGNOSTIC, and it must be read with care: for a near-fronto-parallel PLANAR
target, solvePnP has two nearly-equal-cost solutions and flips between them
frame to frame (the planar two-fold ambiguity). That flipping produces degrees
of rotation "spread" and tens of cm of translation "spread" on data that is
perfectly synchronized. So a wide spread on LOW-TILT captures means "reshoot
with real tilt", not "your sync is broken". We use SOLVEPNP_IPPE here (the
planar-specific solver) to reduce -- not eliminate -- the flipping.

LENS CHANGE (12 mm Edmund Optics C Series, EO 58-001)
-----------------------------------------------------
Extrinsics do not reference a focal length directly -- they consume whatever
intrinsics were solved -- which is exactly why the lens has to be checked here.
An extrinsic solve fed 8 mm intrinsics on 12 mm imagery will converge, report a
believable RMS, and place the cameras somewhere they are not. load_intrinsics()
therefore rejects any intrinsics file whose recorded lens does not match
config.LENS_ID, the same way it already rejects a serial mismatch.

Two practical consequences of the narrower field of view:

  * The per-pair OVERLAP shrinks by the same 1.5x as the field of view. At the
    3.16 m deck each camera now sees 2.22 x 1.86 m, and the right camera sits
    0.861 m off the reference, so the shared volume is a fraction of what the
    8 mm rig gave. Shoot every pair separately, work the board hard through the
    overlap, and expect to need more attempts to reach 25-40 shared views.
  * The chain-consistency cross-check below needs frames seen by TWO TARGETS at
    once. That was already uncommon; at this focal length it will often be
    impossible for left<->right. If you want the check, deliberately shoot a
    handful of poses in the middle where two targets overlap, or accept that
    the composed transform is unvalidated and treat the per-pair RMS as the
    only evidence you have -- which it is not, and the script says so.

The physical camera mounts did not move, but the extrinsics still change: the
entrance pupil of the new lens sits at a different place along the optical
axis, so the translation between projection centres is genuinely different.
Every pre-lens-change extrinsics.json is invalid, not merely stale.

FOUR CAMERAS
------------
Three in a row (left, center, right) plus one above (top), all solved against
the center reference. Three things worth knowing:

  * Detection is cached. Every image is detected exactly once and the pairs are
    assembled from the cache, instead of re-detecting the reference shoot once
    per target.

  * A failing pair no longer kills the run. With pair-by-pair extrinsic
    shoots it is normal for one pair to be under-captured while the others are
    fine. Failures are collected, the pairs that succeeded are still written,
    and the missing ones are named at the end.

  * Chain consistency is checked where the data allows. Downstream, a
    left->right transform is obtained by composing through the reference, so
    its error is the sum of two independent solves. Wherever enough frames were
    seen by two targets at once, this script also solves that pair DIRECTLY and
    reports how far the composed answer drifts from it. That number, not the
    per-pair RMS, is what limits multi-view point-cloud fusion.

The top camera is worth watching separately: its baseline to the reference is
mostly vertical, so it constrains a different direction than the horizontal
row does, and a board sweep that worked for left/right will be nearly
degenerate for it.

PREREQUISITES
-------------
1. Lenses focused and LOCKED, then good intrinsics (run calibrate_intrinsics.py;
   every camera must pass the focal guard against the 12 mm expectation and
   report "distortion sane? True"). Refocusing after this point invalidates
   both intrinsics and extrinsics.
2. SYNCHRONIZED capture into config.EXTRINSIC_CAPTURE_DIR (run auto_capture.py
   with --mode extrinsics): the SAME index is the SAME instant across cameras,
   AND the board is visible to BOTH the reference and the target. Target 25-40
   shared views per pair, worked through each overlap region.

Convention stored in extrinsics.json:  X_target = R @ X_center + t

Usage:
  python calibrate_extrinsics.py
  python calibrate_extrinsics.py --cameras top        # redo one pair
  python calibrate_extrinsics.py --no-consistency
"""
import os, glob, json, argparse, itertools
import cv2
import numpy as np
import config

# BUGFIX: this previously read getattr(config, "captures", ...), i.e. it looked
# up a config attribute literally named "captures", never found one, and fell
# back to CAPTURE_DIR -- so extrinsics silently ran on the intrinsics shoot.
EXTRINSIC_CAPTURE_DIR = getattr(config, "EXTRINSIC_CAPTURE_DIR",
                                config.CAPTURE_DIR)
REFERENCE   = getattr(config, "REFERENCE_CAMERA", "center")
MIN_CORNERS = 12          # PnP / stereo want a good spread of corners
MAX_FRAMES  = 9999
MIN_PAIR_VIEWS   = 3      # hard floor for stereoCalibrate
GOOD_PAIR_VIEWS  = 25     # below this, warn
MIN_CHAIN_VIEWS  = 8      # floor for the direct target<->target cross-check

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def load_intrinsics(name):
    path = os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")
    if not os.path.exists(path):
        raise RuntimeError(
            f"No intrinsics for '{name}' ({path}). Run calibrate_intrinsics.py "
            "and make sure the camera passes the focal guard first.")
    with open(path) as f:
        d = json.load(f)

    # The friendly names were remapped when the fourth camera was added, so an
    # intrinsics file can carry the right NAME and the wrong CAMERA. Catch it
    # here rather than discovering it as an inexplicable baseline later.
    stored = d.get("serial")
    expected = config.CAMERAS.get(name)
    if stored is None:
        print(f"  WARNING: intrinsics_{name}.json has no serial recorded "
              f"(written before serial tagging). If it predates the "
              f"four-camera rename it describes a different camera -- "
              f"recalibrate intrinsics to be sure.")
    elif expected is not None and stored != expected:
        raise RuntimeError(
            f"intrinsics_{name}.json is for serial {stored}, but '{name}' now "
            f"means serial {expected}. Recalibrate intrinsics before running "
            f"extrinsics -- this mapping changed with the fourth camera.")

    # Same argument for the glass. stereoCalibrate with CALIB_FIX_INTRINSIC
    # takes K and dist as given: feed it the 8 mm solve and it will happily fit
    # a rotation and translation that reconcile 12 mm imagery with an 8 mm
    # model, and report a low RMS for doing it.
    lens = d.get("lens_id")
    if lens is None:
        print(f"  WARNING: intrinsics_{name}.json records no lens (written "
              f"before lens tagging). If it predates the change to "
              f"{config.LENS_ID} it describes different glass -- recalibrate "
              f"intrinsics to be sure.")
    elif lens != config.LENS_ID:
        raise RuntimeError(
            f"intrinsics_{name}.json was solved for lens {lens}, but the rig "
            f"now carries {config.LENS_ID} ({config.LENS_MM:g} mm). Archive "
            f"results/ and the capture directories, then recalibrate "
            f"intrinsics before running extrinsics.")

    fx = float(d["camera_matrix"][0][0])
    ratio = fx / config.EXPECTED_FX
    if not (config.FX_RATIO_LO < ratio < config.FX_RATIO_HI):
        raise RuntimeError(
            f"intrinsics_{name}.json has fx = {fx:.0f} px, {ratio:.2f}x the "
            f"{config.EXPECTED_FX:.0f} px expected for the "
            f"{config.LENS_MM:g} mm lens. That file should never have been "
            f"written -- recalibrate intrinsics.")

    K    = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    size = (int(d["image_size"][0]), int(d["image_size"][1]))   # (w, h)
    return K, dist, size


def frame_index(path):
    return os.path.basename(path)


def detect(img_path, detector):
    """Return (charuco_corners Nx2, charuco_ids N) or (None, None). Tries raw
    grayscale then CLAHE, mirroring capture-time detection."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = (None, None, 0)
    for im in (gray, _CLAHE.apply(gray)):
        cc, ids, _, _ = detector.detectBoard(im)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            best = (cc, ids, n)
    cc, ids, n = best
    if ids is None or n < MIN_CORNERS:
        return None, None
    return cc.reshape(-1, 2), ids.ravel()


def detect_all(names, detector):
    """Detect every image once, keyed by camera then frame name.

    With four cameras the old per-pair detection re-ran the reference shoot
    three times. Detection is by far the expensive part of this script, so the
    cache is most of the runtime win -- and it guarantees every pair sees
    identical corners for the same image, which per-pair detection did not
    strictly promise.
    """
    cache = {}
    print("Detecting boards ...")
    for name in names:
        d = {}
        paths = sorted(glob.glob(os.path.join(EXTRINSIC_CAPTURE_DIR,
                                              name, "*.png")))
        for p in paths:
            cc, ids = detect(p, detector)
            if ids is not None:
                d[frame_index(p)] = (cc, ids)
        cache[name] = d
        print(f"  {name:8s} {len(d):4d} usable / {len(paths):4d} images")
    return cache


def report_coverage(cache, names):
    """Which pairs actually have data, printed before any solving starts."""
    print("\nOverlap with the reference:")
    for name in names:
        if name == REFERENCE:
            continue
        shared = set(cache[REFERENCE]) & set(cache[name])
        flag = ""
        if len(shared) < MIN_PAIR_VIEWS:
            flag = "  <- unusable, reshoot this pair"
        elif len(shared) < GOOD_PAIR_VIEWS:
            flag = f"  <- thin (target {GOOD_PAIR_VIEWS}-40)"
        print(f"  {REFERENCE} <-> {name:8s} {len(shared):4d} frames{flag}")


def board_pose(img_pts, ids, board, K, dist):
    """4x4 T_cam_board from charuco corners, or None. Diagnostic use only.
    SOLVEPNP_IPPE is the planar-specific solver -- it enumerates both solutions
    of the planar ambiguity and returns the lower-error one, so the diagnostic
    spread reflects the data rather than solver flip-flopping. Falls back to
    ITERATIVE if IPPE rejects the configuration."""
    obj = np.array([board.getChessboardCorners()[int(i)] for i in ids], np.float32)
    pts = img_pts.astype(np.float32)
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, pts, K, dist, flags=flag)
        except cv2.error:
            continue
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
            return T
    return None


def angle_between(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def average_rotations(Rs):
    """Chordal (L2) mean of rotation matrices -- diagnostic only."""
    A = sum(Rs)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def build_correspondences(cache, cam_a, cam_b, chess):
    """Point-for-point correspondences over every frame both cameras saw.

    Returns (obj_pts, pts_a, pts_b, frame_names). stereoCalibrate needs the
    SAME board point in both lists, so the charuco IDs are intersected per
    frame -- a corner one camera resolved and the other didn't is dropped
    rather than mismatched.
    """
    shared = sorted(set(cache[cam_a]) & set(cache[cam_b]))[:MAX_FRAMES]
    obj_pts, pts_a, pts_b, frames = [], [], [], []
    for fname in shared:
        cc_a, id_a = cache[cam_a][fname]
        cc_b, id_b = cache[cam_b][fname]
        common = np.intersect1d(id_a, id_b)
        if len(common) < MIN_CORNERS:
            continue
        map_a = {int(i): cc_a[k] for k, i in enumerate(id_a)}
        map_b = {int(i): cc_b[k] for k, i in enumerate(id_b)}
        obj_pts.append(np.array([chess[int(i)] for i in common], np.float32))
        pts_a.append(np.array([map_a[int(i)] for i in common], np.float32))
        pts_b.append(np.array([map_b[int(i)] for i in common], np.float32))
        frames.append(fname)
    return obj_pts, pts_a, pts_b, frames


def stereo_solve(obj_pts, pts_a, pts_b, K_a, dist_a, K_b, dist_b, image_size):
    """stereoCalibrate with intrinsics frozen. Returns (rms, R, t) mapping
    a-frame points into b-frame: X_b = R @ X_a + t."""
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-7)
    rms, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
        obj_pts, pts_a, pts_b, K_a, dist_a, K_b, dist_b, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    return float(rms), np.asarray(R), np.asarray(t).ravel()


def calibrate_pair(name, cache, board, detector, K_C, dist_C, image_size):
    K_A, dist_A, _ = load_intrinsics(name)
    chess = board.getChessboardCorners()

    if not cache[REFERENCE] or not cache[name]:
        raise RuntimeError(
            f"{name}: no usable detections for {REFERENCE} and/or {name} in "
            f"{EXTRINSIC_CAPTURE_DIR}/ -- run auto_capture.py --mode "
            f"extrinsics --cameras {name} first.")

    obj_pts, img_C, img_A, frames = build_correspondences(
        cache, REFERENCE, name, chess)

    if len(obj_pts) < MIN_PAIR_VIEWS:
        raise RuntimeError(
            f"{name}: only {len(obj_pts)} shared board views -- need more "
            f"(target {GOOD_PAIR_VIEWS}-40). Shoot this pair on its own: "
            f"auto_capture.py --mode extrinsics --cameras {name}")
    if len(obj_pts) < GOOD_PAIR_VIEWS:
        print(f"  NOTE: {len(obj_pts)} shared views is below the "
              f"{GOOD_PAIR_VIEWS}-40 target. The {config.LENS_MM:g} mm lens "
              f"makes this pair's overlap volume small -- work the board "
              f"deliberately through it rather than sweeping the whole cell.")

    print(f"\n=== {name}  ({len(obj_pts)} shared board views) ===")

    # ---- data-health: spread of per-frame relative poses (diagnostic) ----
    rels = []
    for o, pC, pA, ids_frame in zip(obj_pts, img_C, img_A, frames):
        # Recover the shared IDs for this frame from the cached detections.
        idC = cache[REFERENCE][ids_frame][1]
        idA = cache[name][ids_frame][1]
        common = np.intersect1d(idC, idA)
        T_C = board_pose(pC, common, board, K_C, dist_C)
        T_A = board_pose(pA, common, board, K_A, dist_A)
        if T_C is not None and T_A is not None:
            rels.append(T_A @ np.linalg.inv(T_C))

    if len(rels) >= 2:
        Rs = [T[:3, :3] for T in rels]
        ts = np.array([T[:3, 3] for T in rels])
        R0 = average_rotations(Rs)
        t0 = np.median(ts, axis=0)
        rot_dev = np.array([angle_between(R0, R) for R in Rs])
        t_dev   = ts - t0
        print(f"  per-frame spread : rot median |dev| = {np.median(rot_dev):.2f} deg "
              f"(max {rot_dev.max():.2f} deg); "
              f"t std = {t_dev[:,0].std()*100:.1f}/{t_dev[:,1].std()*100:.1f}/"
              f"{t_dev[:,2].std()*100:.1f} cm")
        if np.median(rot_dev) > 1.0 or np.linalg.norm(t_dev.std(0)) > 0.02:
            print("  ** WIDE SPREAD. Possible causes, most likely first:\n"
                  "     1. Low-tilt views -> planar PnP two-fold ambiguity\n"
                  "        (per-frame poses flip even on perfect data; reshoot\n"
                  "        with real +/-30-45 deg tilt). The narrower field of\n"
                  "        view makes it tempting to hold the board square on\n"
                  "        just to keep it in both frames -- resist that.\n"
                  "     2. Bad intrinsics (check the focal guard in\n"
                  "        calibrate_intrinsics.py passed for BOTH cameras)\n"
                  "     3. Actual desync / bad correspondences (only if 1 and 2\n"
                  "        are excluded -- with PTP+Action0 sync this is rare)\n"
                  "     stereoCalibrate below is unaffected by cause 1, but\n"
                  "     nothing recovers causes 2-3. **")

    # ---- joint stereo calibration with intrinsics fixed ----
    rms, R, t = stereo_solve(obj_pts, img_C, img_A,
                             K_C, dist_C, K_A, dist_A, image_size)

    C = -R.T @ t                                # target center in center frame
    print(f"  stereo reproj RMS = {rms:.4f} px   baseline = {np.linalg.norm(t):.3f} m")
    print(f"  target center in center frame (x=right,y=down,z=fwd): "
          f"[{C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f}] m")

    return {
        "reference"     : REFERENCE,
        "serial"        : config.CAMERAS.get(name),
        "lens_id"       : config.LENS_ID,
        "lens_mm"       : float(config.LENS_MM),
        "R"             : R.tolist(),
        "t"             : t.tolist(),
        "rms_error_px"  : float(rms),
        "baseline_m"    : float(np.linalg.norm(t)),
        "n_shared_views": len(obj_pts),
    }


def check_chain_consistency(out, cache, board, image_size):
    """Compare composed target<->target transforms against direct solves.

    Downstream a left->right transform comes from composing left->center with
    center->right, so it carries both solves' errors. Wherever two targets saw
    the board in the same frames, the pair can also be solved directly, and the
    gap between the two answers is the honest bound on multi-view fusion
    accuracy. A per-pair RMS of 0.3 px means nothing if composing two of them
    puts the clouds 2 cm apart.

    With the 12 mm lens the frames needed for this check are much rarer, so an
    empty result here is expected rather than alarming -- but it does mean the
    composed transforms are unvalidated, which matters directly for the
    inter-camera disagreement seen at parcel tops.
    """
    chess = board.getChessboardCorners()
    targets = [n for n in out if n != REFERENCE and "R" in out[n]]
    results = []

    print("\n=== chain consistency (composed vs direct) ===")
    any_checked = False
    for a, b in itertools.combinations(targets, 2):
        obj_pts, pts_a, pts_b, _ = build_correspondences(cache, a, b, chess)
        if len(obj_pts) < MIN_CHAIN_VIEWS:
            print(f"  {a} <-> {b}: {len(obj_pts)} shared views -- too few to "
                  f"cross-check (need {MIN_CHAIN_VIEWS})")
            continue
        try:
            K_a, dist_a, _ = load_intrinsics(a)
            K_b, dist_b, _ = load_intrinsics(b)
            rms_direct, R_dir, t_dir = stereo_solve(
                obj_pts, pts_a, pts_b, K_a, dist_a, K_b, dist_b, image_size)
        except Exception as e:
            print(f"  {a} <-> {b}: direct solve failed ({e})")
            continue

        # X_a = R_a X_C + t_a ; X_b = R_b X_C + t_b
        #   => X_b = (R_b R_a^T) X_a + (t_b - R_b R_a^T t_a)
        R_a = np.array(out[a]["R"]); t_a = np.array(out[a]["t"])
        R_b = np.array(out[b]["R"]); t_b = np.array(out[b]["t"])
        R_cmp = R_b @ R_a.T
        t_cmp = t_b - R_cmp @ t_a

        d_rot = float(angle_between(R_cmp, R_dir))
        d_t   = float(np.linalg.norm(t_cmp - t_dir))
        any_checked = True
        results.append({"pair": [a, b], "n_views": len(obj_pts),
                        "direct_rms_px": rms_direct,
                        "rotation_gap_deg": d_rot,
                        "translation_gap_m": d_t,
                        "direct_baseline_m": float(np.linalg.norm(t_dir))})
        flag = ""
        if d_rot > 0.5 or d_t > 0.01:
            flag = "  <- composed chain disagrees with the direct solve"
        print(f"  {a} <-> {b}: {len(obj_pts):3d} views, direct RMS "
              f"{rms_direct:.3f} px | gap: {d_rot:.3f} deg, "
              f"{d_t*1000:.1f} mm{flag}")

    if not any_checked:
        print("  (no target pair shares enough frames -- expected with this "
              "field of view and pair-by-pair shooting; shoot a few poses "
              "visible to two targets at once if you want this check)")
    elif any(r["rotation_gap_deg"] > 0.5 or r["translation_gap_m"] > 0.01
             for r in results):
        print("  ** A gap here means the reference-relative solves are not "
              "mutually consistent. The usual cause is one pair being solved "
              "from a thin or low-tilt view set: its own RMS looks fine "
              "because it fits its own data, but it disagrees with everyone "
              "else. Compare n_shared_views across the pairs above. **")
    return results


def main():
    p = argparse.ArgumentParser(
        description="Extrinsic calibration of the four-Triton rig against the "
                    "reference camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cameras", nargs="+", default=None,
                   choices=[n for n in config.CAMERAS if n != REFERENCE],
                   help="target cameras to solve; default all. Pairs not "
                        "solved keep their existing entry in extrinsics.json")
    p.add_argument("--no-consistency", dest="consistency",
                   action="store_false",
                   help="skip the composed-vs-direct cross-check between "
                        "target cameras")
    args = p.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(config.describe_optics())
    board    = config.get_charuco_board()
    detector = config.get_charuco_detector()
    K_C, dist_C, image_size = load_intrinsics(REFERENCE)

    targets = args.cameras or [n for n in config.CAMERAS if n != REFERENCE]
    names = [REFERENCE] + [n for n in config.CAMERAS
                           if n in set(targets) and n != REFERENCE]

    print(f"\nExtrinsic captures: {EXTRINSIC_CAPTURE_DIR}/  reference: "
          f"{REFERENCE} (SN {config.CAMERAS.get(REFERENCE)})")
    print(f"Targets: {', '.join(n for n in names if n != REFERENCE)}")

    cache = detect_all(names, detector)
    report_coverage(cache, names)

    # Start from whatever is already on disk so a partial run doesn't discard
    # pairs solved earlier -- but drop entries solved through a different lens,
    # which are not merely stale, they are wrong.
    path = os.path.join(config.RESULTS_DIR, "extrinsics.json")
    out = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                out = json.load(f)
        except (json.JSONDecodeError, OSError):
            out = {}
    stale = [n for n, e in out.items()
             if isinstance(e, dict) and "R" in e
             and e.get("lens_id") != config.LENS_ID]
    for n in stale:
        out.pop(n, None)
    if stale:
        print(f"\n  Dropped pre-existing entries solved through a different "
              f"lens: {', '.join(sorted(stale))}. Re-solve those pairs.")

    out[REFERENCE] = {"reference": REFERENCE,
                      "serial": config.CAMERAS.get(REFERENCE),
                      "lens_id": config.LENS_ID,
                      "lens_mm": float(config.LENS_MM),
                      "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
                      "rms_error_px": 0.0, "note": "reference frame"}

    failed = []
    for name in names:
        if name == REFERENCE:
            continue
        try:
            out[name] = calibrate_pair(name, cache, board, detector,
                                       K_C, dist_C, image_size)
        except RuntimeError as e:
            # One under-captured pair must not cost the pairs that worked.
            print(f"\n=== {name} ===\n  ** {e} **")
            failed.append(name)

    solved = [n for n in config.CAMERAS if n in out and n != REFERENCE
              and "R" in out[n]]
    if solved:
        print("\n=== rig summary ===")
        print(f"  {'camera':8s} {'serial':11s} {'baseline m':>11s} "
              f"{'RMS px':>8s} {'views':>6s}")
        for n in solved:
            e = out[n]
            print(f"  {n:8s} {str(e.get('serial')):11s} "
                  f"{e['baseline_m']:11.3f} {e['rms_error_px']:8.3f} "
                  f"{e.get('n_shared_views', 0):6d}")
        print("  Compare the baselines against the tape measure. They will not "
              "match the pre-lens-change values exactly even though nothing "
              "moved: the entrance pupil sits elsewhere in the new lens.")

    consistency = []
    if args.consistency and len(solved) >= 2:
        consistency = check_chain_consistency(out, cache, board, image_size)

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")

    if consistency:
        cpath = os.path.join(config.RESULTS_DIR, "extrinsics_consistency.json")
        with open(cpath, "w") as f:
            json.dump(consistency, f, indent=2)
        print(f"-> {cpath}")

    if failed:
        print(f"\nFAILED pairs (not written): {', '.join(failed)}")
        print("Reshoot those pairs individually: auto_capture.py --mode "
              f"extrinsics --cameras {failed[0]}")


if __name__ == "__main__":
    main()