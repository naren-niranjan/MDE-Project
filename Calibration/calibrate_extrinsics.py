"""Multi-camera extrinsics calibration.

Computes the rigid transform of each non-reference camera in the reference
camera's frame, using pairs of synchronized board views (frames where the
ChArUco board is visible in BOTH cameras).

Output written to results/extrinsics.json with one entry per non-reference camera:
    R          - 3x3 rotation
    t          - 3x1 translation in metres
    rms_error  - RMS reprojection error in px

Convention: a point X in the reference camera frame maps to the other camera as
            X_other = R @ X_ref + t
"""
import os, glob, json
import cv2
import numpy as np
import config


def load_intrinsics(name):
    with open(os.path.join(config.RESULTS_DIR, f"intrinsics_{name}.json")) as f:
        d = json.load(f)
    K    = np.array(d["camera_matrix"],  dtype=np.float64)
    dist = np.array(d["dist_coeffs"],    dtype=np.float64).reshape(-1, 1)
    size = tuple(d["image_size"])
    return K, dist, size


def detect_charuco_per_file(image_dir, detector):
    """Return dict: 'frame_XXXX' -> (charuco_corners (N,1,2), charuco_ids (N,1))."""
    out = {}
    for f in sorted(glob.glob(os.path.join(image_dir, "frame_*.png"))):
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            continue
        key = os.path.splitext(os.path.basename(f))[0]   # frame_0007
        out[key] = (c, ids)
    return out


def stereo_calibrate_pair(ref_name, other_name, detector, board_corners_3d):
    """Calibrate extrinsics with ref as the master."""
    K_r, d_r, size_r = load_intrinsics(ref_name)
    K_o, d_o, size_o = load_intrinsics(other_name)
    if size_r != size_o:
        print(f"  warning: image sizes differ ({size_r} vs {size_o})")

    dets_r = detect_charuco_per_file(os.path.join(config.CAPTURE_DIR, ref_name),   detector)
    dets_o = detect_charuco_per_file(os.path.join(config.CAPTURE_DIR, other_name), detector)
    common = sorted(set(dets_r) & set(dets_o))
    print(f"  shared frames with detections: {len(common)}")

    obj_pts_all, pts_r_all, pts_o_all = [], [], []
    for key in common:
        c_r, ids_r = dets_r[key]
        c_o, ids_o = dets_o[key]
        ids_r_flat = ids_r.flatten()
        ids_o_flat = ids_o.flatten()

        shared = np.intersect1d(ids_r_flat, ids_o_flat)
        if len(shared) < 6:
            continue

        # Build aligned points for the shared IDs
        idx_r = {cid: i for i, cid in enumerate(ids_r_flat)}
        idx_o = {cid: i for i, cid in enumerate(ids_o_flat)}

        obj_pts = np.array([board_corners_3d[cid] for cid in shared], dtype=np.float32)
        pts_r   = np.array([c_r[idx_r[cid]][0]    for cid in shared], dtype=np.float32)
        pts_o   = np.array([c_o[idx_o[cid]][0]    for cid in shared], dtype=np.float32)

        obj_pts_all.append(obj_pts)
        pts_r_all.append(pts_r)
        pts_o_all.append(pts_o)

    if len(obj_pts_all) < 5:
        raise RuntimeError(
            f"Only {len(obj_pts_all)} usable shared views between '{ref_name}' "
            f"and '{other_name}'. Capture more board positions in the overlap.")

    print(f"  usable shared views: {len(obj_pts_all)}")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts_all, pts_r_all, pts_o_all,
        K_r, d_r, K_o, d_o, size_r,
        flags=flags, criteria=criteria)

    print(f"  RMS reprojection error: {rms:.4f} px")
    print(f"  translation (m): [{T[0,0]:.4f}, {T[1,0]:.4f}, {T[2,0]:.4f}] "
          f"  (baseline = {np.linalg.norm(T):.4f} m)")
    return rms, R, T


def main():
    board = config.get_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)

    # 3D coords of every ChArUco corner in the board frame (board lies in Z=0 plane).
    # Indexing by ChArUco corner ID matches what detector.detectBoard returns.
    board_corners_3d = board.getChessboardCorners()   # shape (N, 3)

    ref = config.REFERENCE_CAMERA
    if ref not in config.CAMERAS:
        raise RuntimeError(f"REFERENCE_CAMERA '{ref}' not in CAMERAS.")

    extrinsics = {
        ref: {
            "reference"        : ref,
            "R"                : np.eye(3).tolist(),
            "t"                : [0.0, 0.0, 0.0],
            "rms_error_px"     : 0.0,
            "note"             : "reference frame",
        }
    }

    for name in config.CAMERAS:
        if name == ref:
            continue
        print(f"\n=== Extrinsics: {ref} -> {name} ===")
        rms, R, T = stereo_calibrate_pair(ref, name, detector, board_corners_3d)
        extrinsics[name] = {
            "reference"     : ref,
            "R"             : R.tolist(),
            "t"             : T.flatten().tolist(),
            "rms_error_px"  : float(rms),
            "baseline_m"    : float(np.linalg.norm(T)),
        }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "extrinsics.json")
    with open(out, "w") as f:
        json.dump(extrinsics, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
