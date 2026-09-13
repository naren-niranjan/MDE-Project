"""Multi-camera extrinsic calibration relative to 'center', with diagnostics.

PREREQUISITES
-------------
1. Intrinsics already calibrated and SANE (run the fixed calibrate_intrinsics.py
   first; confirm each camera reports "distortion sane? True").
2. SYNCHRONIZED capture: EXTRINSIC_CAPTURE_DIR/<cam>/frame_XXXX.png where the
   SAME index is the SAME instant across cameras, AND the board is visible to
   the reference camera ('center') and the target camera in that frame.
   If your extrinsic images were shot independently per camera (e.g. you reused
   the per-camera intrinsics captures), STOP -- extrinsics cannot be recovered
   from non-simultaneous views, and this script will (correctly) report a huge
   per-frame spread.

METHOD
------
For each frame shared by center C and target A:
    solvePnP -> T_C_board  (board points -> center cam coords)
    solvePnP -> T_A_board  (board points -> target cam coords)
    T_A_C = T_A_board @ inv(T_C_board)          (center coords -> target coords)
The per-frame T_A_C estimates are then robustly averaged. We store R, t with the
convention  X_target = R @ X_center + t , matching extrinsics.json.

The KEY diagnostic is the spread of the per-frame estimates: tight => good data,
wide => unsynchronized frames or bad correspondences (no averaging will save it).
"""
import os, glob, json
import cv2
import numpy as np
import config

# --- adjust to wherever your SIMULTANEOUS multi-camera captures live ---
EXTRINSIC_CAPTURE_DIR = getattr(config, "EXTRINSIC_CAPTURE_DIR", config.CAPTURE_DIR)
REFERENCE   = "center"
MIN_CORNERS = 12          # be stricter than intrinsics; PnP wants a good spread
MAX_FRAMES  = 9999


def load_intrinsics(name):
    with open(os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def board_pose(img_path, detector, board, K, dist):
    """Return 4x4 T_cam_board (board coords -> cam coords) or None."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < MIN_CORNERS:
        return None
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < MIN_CORNERS:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.ravel()
    return T


def frame_index(path):
    return os.path.basename(path)


def average_rotations(Rs):
    """Chordal (L2) mean of rotation matrices -- valid for clustered inputs."""
    A = sum(Rs)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def angle_between(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def calibrate_pair(name, board, detector, K_C, dist_C):
    K_A, dist_A = load_intrinsics(name)

    ref_files = {frame_index(p) for p in
                 glob.glob(os.path.join(EXTRINSIC_CAPTURE_DIR, REFERENCE, "*.png"))}
    tgt_files = {frame_index(p) for p in
                 glob.glob(os.path.join(EXTRINSIC_CAPTURE_DIR, name, "*.png"))}
    shared = sorted(ref_files & tgt_files)[:MAX_FRAMES]
    if not shared:
        raise RuntimeError(f"No shared frame names between {REFERENCE} and {name}")

    rels, used = [], []
    sq_err, n_pts = 0.0, 0
    for fname in shared:
        T_C = board_pose(os.path.join(EXTRINSIC_CAPTURE_DIR, REFERENCE, fname),
                         detector, board, K_C, dist_C)
        T_A = board_pose(os.path.join(EXTRINSIC_CAPTURE_DIR, name, fname),
                         detector, board, K_A, dist_A)
        if T_C is None or T_A is None:
            continue
        rels.append(T_A @ np.linalg.inv(T_C))      # T_A_C
        used.append(fname)

    if len(rels) < 3:
        raise RuntimeError(f"{name}: only {len(rels)} shared board views -- need more")

    Rs = [T[:3, :3] for T in rels]
    ts = np.array([T[:3, 3] for T in rels])

    R_mean = average_rotations(Rs)
    t_mean = np.median(ts, axis=0)                  # robust to outliers

    # ---- spread diagnostic ----
    rot_dev = np.array([angle_between(R_mean, R) for R in Rs])
    t_dev   = ts - t_mean
    print(f"\n=== {name}  ({len(rels)} shared board views) ===")
    print(f"  rotation spread : median |dev| = {np.median(rot_dev):.2f} deg, "
          f"max = {rot_dev.max():.2f} deg")
    print(f"  translation spread (std): "
          f"x={t_dev[:,0].std()*100:.1f} y={t_dev[:,1].std()*100:.1f} "
          f"z={t_dev[:,2].std()*100:.1f} cm")
    if np.median(rot_dev) > 1.0 or np.linalg.norm(t_dev.std(0)) > 0.02:
        print("  ** WIDE SPREAD: frames likely NOT synchronized / bad matches. "
              "Fix the capture before trusting any averaged result. **")

    # ---- final extrinsic + reprojection RMS ----
    T_A_C = np.eye(4); T_A_C[:3, :3] = R_mean; T_A_C[:3, 3] = t_mean
    for fname in used:
        T_C = board_pose(os.path.join(EXTRINSIC_CAPTURE_DIR, REFERENCE, fname),
                         detector, board, K_C, dist_C)
        img = cv2.imread(os.path.join(EXTRINSIC_CAPTURE_DIR, name, fname))
        cc, ci, _, _ = detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        obj, obs = board.matchImagePoints(cc, ci)
        T_pred = T_A_C @ T_C
        rvec, _ = cv2.Rodrigues(T_pred[:3, :3])
        proj, _ = cv2.projectPoints(obj, rvec, T_pred[:3, 3], K_A, dist_A)
        d = proj.reshape(-1, 2) - obs.reshape(-1, 2)
        sq_err += np.sum(d ** 2); n_pts += len(d)
    rms = float(np.sqrt(sq_err / n_pts))

    C = -R_mean.T @ t_mean                          # target center in center frame
    print(f"  baseline = {np.linalg.norm(t_mean):.3f} m   reproj RMS = {rms:.3f} px")
    print(f"  target center in center frame (x=right,y=down,z=fwd): "
          f"[{C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f}] m")

    return {
        "reference": REFERENCE,
        "R": R_mean.tolist(),
        "t": t_mean.tolist(),
        "rms_error_px": rms,
        "baseline_m": float(np.linalg.norm(t_mean)),
        "n_shared_views": len(used),
    }


def main():
    board    = config.get_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    K_C, dist_C = load_intrinsics(REFERENCE)

    out = {REFERENCE: {"reference": REFERENCE,
                       "R": np.eye(3).tolist(), "t": [0, 0, 0],
                       "rms_error_px": 0.0, "note": "reference frame"}}
    for name in config.CAMERAS:
        if name == REFERENCE:
            continue
        out[name] = calibrate_pair(name, board, detector, K_C, dist_C)

    path = os.path.join(config.RESULTS_DIR, "extrinsics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()