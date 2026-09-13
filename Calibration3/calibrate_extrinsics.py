"""Multi-camera extrinsic calibration relative to 'center'.

Each target camera is calibrated against the reference with cv2.stereoCalibrate
and CALIB_FIX_INTRINSIC, which JOINTLY minimizes reprojection error over all
shared frames. This is strictly better than averaging independent per-frame
solvePnP poses: no planar pose-ambiguity outliers, no non-robust rotation mean.

The per-frame relative-pose spread is still computed -- purely as a data-health
check. If it is wide, your frames are NOT synchronized (or correspondences are
bad) and NOTHING, stereoCalibrate included, will recover good extrinsics.

PREREQUISITES
-------------
1. Good intrinsics first (run calibrate_intrinsics.py; "distortion sane? True").
2. SYNCHRONIZED capture: EXTRINSIC_CAPTURE_DIR/<cam>/frame_XXXX.png where the
   SAME index is the SAME instant across cameras, AND the board is visible to
   BOTH the reference ('center') and the target in that frame. Reusing the
   per-camera intrinsics captures does NOT work -- the spread check will flag it.

Convention stored in extrinsics.json:  X_target = R @ X_center + t
"""
import os, glob, json
import cv2
import numpy as np
import config

EXTRINSIC_CAPTURE_DIR = getattr(config, "captures", config.CAPTURE_DIR)
REFERENCE   = "center"
MIN_CORNERS = 12          # PnP / stereo want a good spread of corners
MAX_FRAMES  = 9999


def load_intrinsics(name):
    with open(os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")) as f:
        d = json.load(f)
    K    = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    size = (int(d["image_size"][0]), int(d["image_size"][1]))   # (w, h)
    return K, dist, size


def frame_index(path):
    return os.path.basename(path)


def detect(img_path, detector):
    """Return (charuco_corners Nx2, charuco_ids N) or (None, None)."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cc, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None, None
    return cc.reshape(-1, 2), ids.ravel()


def board_pose(img_pts, ids, board, K, dist):
    """4x4 T_cam_board from charuco corners, or None. Diagnostic use only."""
    obj = np.array([board.getChessboardCorners()[int(i)] for i in ids], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts.astype(np.float32), K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
    return T


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


def calibrate_pair(name, board, detector, K_C, dist_C, image_size):
    K_A, dist_A, _ = load_intrinsics(name)
    chess = board.getChessboardCorners()

    ref_files = {frame_index(p) for p in
                 glob.glob(os.path.join(EXTRINSIC_CAPTURE_DIR, REFERENCE, "*.png"))}
    tgt_files = {frame_index(p) for p in
                 glob.glob(os.path.join(EXTRINSIC_CAPTURE_DIR, name, "*.png"))}
    shared = sorted(ref_files & tgt_files)[:MAX_FRAMES]
    if not shared:
        raise RuntimeError(f"No shared frame names between {REFERENCE} and {name}")

    obj_pts, img_C, img_A = [], [], []
    rels = []                                   # per-frame T_A_C, diagnostic only
    for fname in shared:
        ccC, idC = detect(os.path.join(EXTRINSIC_CAPTURE_DIR, REFERENCE, fname), detector)
        ccA, idA = detect(os.path.join(EXTRINSIC_CAPTURE_DIR, name, fname), detector)
        if idC is None or idA is None:
            continue

        # stereoCalibrate needs point-for-point correspondences: intersect the
        # charuco IDs seen by both cameras in this frame.
        common = np.intersect1d(idC, idA)
        if len(common) < MIN_CORNERS:
            continue
        mapC = {int(i): ccC[k] for k, i in enumerate(idC)}
        mapA = {int(i): ccA[k] for k, i in enumerate(idA)}
        o  = np.array([chess[int(i)] for i in common], np.float32)
        pC = np.array([mapC[int(i)]  for i in common], np.float32)
        pA = np.array([mapA[int(i)]  for i in common], np.float32)
        obj_pts.append(o); img_C.append(pC); img_A.append(pA)

        T_C = board_pose(pC, common, board, K_C, dist_C)
        T_A = board_pose(pA, common, board, K_A, dist_A)
        if T_C is not None and T_A is not None:
            rels.append(T_A @ np.linalg.inv(T_C))

    if len(obj_pts) < 3:
        raise RuntimeError(f"{name}: only {len(obj_pts)} shared board views -- need more")

    print(f"\n=== {name}  ({len(obj_pts)} shared board views) ===")

    # ---- data-health: spread of per-frame relative poses ----
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
            print("  ** WIDE SPREAD: frames likely NOT synchronized / bad matches. "
                  "stereoCalibrate cannot fix unsynchronized data -- reshoot. **")

    # ---- joint stereo calibration with intrinsics fixed ----
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-7)
    rms, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
        obj_pts, img_C, img_A, K_C, dist_C, K_A, dist_A, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    R = np.asarray(R); t = np.asarray(t).ravel()

    C = -R.T @ t                                # target center in center frame
    print(f"  stereo reproj RMS = {rms:.4f} px   baseline = {np.linalg.norm(t):.3f} m")
    print(f"  target center in center frame (x=right,y=down,z=fwd): "
          f"[{C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f}] m")

    return {
        "reference"     : REFERENCE,
        "R"             : R.tolist(),
        "t"             : t.tolist(),
        "rms_error_px"  : float(rms),
        "baseline_m"    : float(np.linalg.norm(t)),
        "n_shared_views": len(obj_pts),
    }


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    board    = config.get_charuco_board()
    detector = config.get_charuco_detector()
    K_C, dist_C, image_size = load_intrinsics(REFERENCE)

    out = {REFERENCE: {"reference": REFERENCE,
                       "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
                       "rms_error_px": 0.0, "note": "reference frame"}}
    for name in config.CAMERAS:
        if name == REFERENCE:
            continue
        out[name] = calibrate_pair(name, board, detector, K_C, dist_C, image_size)

    path = os.path.join(config.RESULTS_DIR, "extrinsics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()