#!/usr/bin/env python3
"""
Multi-view metric depth fusion using Depth Anything 3 (nested giant-large 1.1).

v3 - why v2 could still merge the wrong boxes:
  The parcels on the conveyor are near-identical and regularly spaced.
  Unconstrained ICP on such repetitive geometry can converge onto the wrong
  box (aliasing by one box pitch), because from ICP's point of view any box
  fits any box. v3 therefore anchors the alignment to *appearance*, not just
  geometry:

  Stage A - SIFT scale-and-shift (unchanged principle):
    Cross-view SIFT matches, epipolar-filtered against calibration, are
    triangulated to metric anchors; a robust z' = a*z + b is fitted and
    applied per camera.

  Stage B - Feature-anchored rigid alignment (NEW):
    The same SIFT matches give direct 3D-3D correspondences: each matched
    pixel is back-projected through its own (corrected) depth in both the
    center and the side camera, both expressed in the reference frame.
    A RANSAC + Umeyama fit then solves the per-side-camera SE(3) correction.
    Because every correspondence is a matched *texture* point on the same
    physical box, this cannot alias to a neighbouring box.

  Stage C - Constrained ICP polish (max distance tightened, default 4 cm):
    Only fine residuals remain after Stage B, so ICP is restricted to small
    correspondences and can no longer jump a box pitch.

  Plus: flying-pixel filtering. Points on strong depth gradients (box edges,
  the streaky "curtain" artefacts) are discarded before fusion and before
  registration.

Example:
  python3 fuse_multiview_da3_v3.py --fp16 --voxel 0.005
Ablation:
  --no-feat / --no-icp / --no-sift disable individual stages.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ["left", "center", "right"]
REFERENCE = "center"
SIDES = [c for c in CAMERAS if c != REFERENCE]


# --------------------------------------------------------------------------- #
# Calibration loading
# --------------------------------------------------------------------------- #

def _find_first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    K = _find_first(data, ["camera_matrix", "K", "intrinsic_matrix", "intrinsics", "mtx"])
    dist = _find_first(data, ["dist_coeffs", "distortion_coefficients", "dist", "D", "distortion"])
    if K is None:
        raise ValueError(f"{path}: no camera matrix found; keys present: {list(data.keys())}")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    return K, dist


def _rt_to_4x4(R, T):
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    M[:3, 3] = np.asarray(T, dtype=np.float64).ravel()
    return M


def _parse_cam_entry(entry):
    if isinstance(entry, dict):
        M = _find_first(entry, ["matrix", "transform", "T_4x4", "pose"])
        if M is not None:
            return np.asarray(M, dtype=np.float64).reshape(4, 4)
        R = _find_first(entry, ["R", "rotation", "rotation_matrix"])
        T = _find_first(entry, ["T", "t", "translation", "translation_vector", "tvec"])
        if R is not None and T is not None:
            return _rt_to_4x4(R, T)
    elif isinstance(entry, (list, tuple)):
        arr = np.asarray(entry, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
    return None


def load_extrinsics(path, direction, log):
    with open(path, "r") as f:
        data = json.load(f)

    mats = {}
    for cam in CAMERAS:
        if cam == REFERENCE:
            mats[cam] = np.eye(4)
            continue
        entry = data.get(cam)
        M = _parse_cam_entry(entry) if entry is not None else None
        if M is None:
            for k in [k for k in data.keys() if cam in k.lower()]:
                M = _parse_cam_entry(data[k])
                if M is not None:
                    break
        if M is None:
            R = _find_first(data, [f"R_{cam}", f"{cam}_R"])
            T = _find_first(data, [f"T_{cam}", f"{cam}_T"])
            if R is not None and T is not None:
                M = _rt_to_4x4(R, T)
        if M is None:
            raise ValueError(f"{path}: could not locate extrinsics for '{cam}'; "
                             f"top-level keys: {list(data.keys())}")
        mats[cam] = M

    t_mags = [np.linalg.norm(mats[c][:3, 3]) for c in SIDES]
    if t_mags and np.median(t_mags) > 5.0:
        log(f"[extrinsics] median |T| = {np.median(t_mags):.1f} -> mm, converting to metres")
        for c in CAMERAS:
            mats[c][:3, 3] /= 1000.0
    else:
        log(f"[extrinsics] median |T| = {np.median(t_mags):.3f} m")

    if direction == "cam_to_ref":
        for c in SIDES:
            mats[c] = np.linalg.inv(mats[c])

    for c in SIDES:
        log(f"[extrinsics] {c}: baseline to {REFERENCE} = "
            f"{np.linalg.norm(np.linalg.inv(mats[c])[:3, 3]):.3f} m")
    return mats


# --------------------------------------------------------------------------- #
# Matching / triangulation
# --------------------------------------------------------------------------- #

def sift_match(img_a, img_b, ratio=0.75, max_feats=8000):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_feats)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    pts_a, pts_b = [], []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            pts_a.append(kp_a[pair[0].queryIdx].pt)
            pts_b.append(kp_b[pair[0].trainIdx].pt)
    return np.asarray(pts_a, dtype=np.float64), np.asarray(pts_b, dtype=np.float64)


def epipolar_filter(pts_ref, pts_cam, K_ref, K_cam, T_ref_to_cam, thresh_px):
    if pts_ref.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    R = T_ref_to_cam[:3, :3]
    t = T_ref_to_cam[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K_cam).T @ (tx @ R) @ np.linalg.inv(K_ref)
    ones = np.ones((pts_ref.shape[0], 1))
    x1 = np.hstack([pts_ref, ones])
    x2 = np.hstack([pts_cam, ones])
    lines = (F @ x1.T).T
    num = np.abs(np.sum(lines * x2, axis=1))
    den = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2) + 1e-12
    return (num / den) < thresh_px


def triangulate_pair(pts_ref, pts_cam, K_ref, K_cam, T_ref_to_cam):
    P1 = K_ref @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_cam @ T_ref_to_cam[:3, :]
    Xh = cv2.triangulatePoints(P1, P2, pts_ref.T, pts_cam.T)
    X = (Xh[:3] / Xh[3]).T
    z_ref = X[:, 2]
    z_cam = (X @ T_ref_to_cam[:3, :3].T + T_ref_to_cam[:3, 3])[:, 2]
    return X, z_ref, z_cam


def robust_scale_shift(z_pred, z_tri, fit_shift=True, iters=5, trim=2.5):
    mask = np.isfinite(z_pred) & np.isfinite(z_tri) & (z_pred > 0.05) & (z_tri > 0.05)
    x, y = z_pred[mask], z_tri[mask]
    if x.size < 20:
        return None
    inl = np.ones(x.size, dtype=bool)
    a, b = 1.0, 0.0
    res = np.zeros_like(x)
    for _ in range(iters):
        xi, yi = x[inl], y[inl]
        if xi.size < 10:
            return None
        if fit_shift:
            A = np.stack([xi, np.ones_like(xi)], axis=1)
            (a, b), *_ = np.linalg.lstsq(A, yi, rcond=None)
        else:
            a, b = float(np.dot(xi, yi) / np.dot(xi, xi)), 0.0
        res = y - (a * x + b)
        sigma = np.std(res[inl]) + 1e-9
        inl = np.abs(res) < trim * sigma
    rms = float(np.sqrt(np.mean(res[inl] ** 2)))
    return float(a), float(b), int(inl.sum()), rms


def sample_depth(depth, pts_xy):
    u = np.clip(np.round(pts_xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pts_xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    return depth[v, u]


# --------------------------------------------------------------------------- #
# Feature-anchored rigid alignment (Stage B)
# --------------------------------------------------------------------------- #

def umeyama(src, dst, with_scale=False):
    """Least-squares similarity/rigid transform mapping src -> dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = dc.T @ sc / src.shape[0]
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    if with_scale:
        var_s = np.mean(np.sum(sc ** 2, axis=1))
        s = np.trace(np.diag(S) @ D) / (var_s + 1e-12)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def ransac_rigid(src, dst, tol, iters=2000, with_scale=False, seed=0):
    """RANSAC over 3-point Umeyama hypotheses. Returns (T 4x4, inlier_mask)."""
    n = src.shape[0]
    if n < 4:
        return None, None
    rng = np.random.default_rng(seed)
    best_inl = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        s3, d3 = src[idx], dst[idx]
        # reject degenerate (near-collinear) samples
        if np.linalg.norm(np.cross(s3[1] - s3[0], s3[2] - s3[0])) < 1e-8:
            continue
        T = umeyama(s3, d3, with_scale=with_scale)
        res = np.linalg.norm(src @ T[:3, :3].T + T[:3, 3] - dst, axis=1)
        inl = res < tol
        if inl.sum() > best_count:
            best_count = int(inl.sum())
            best_inl = inl
    if best_inl is None or best_count < 10:
        return None, None
    T = umeyama(src[best_inl], dst[best_inl], with_scale=with_scale)
    # one refinement pass on the refit inliers
    res = np.linalg.norm(src @ T[:3, :3].T + T[:3, 3] - dst, axis=1)
    inl = res < tol
    if inl.sum() >= 10:
        T = umeyama(src[inl], dst[inl], with_scale=with_scale)
    return T, inl


def backproject_pixels(pts_xy, depth, K, z_min, z_max):
    """Back-project specific pixels through the depth map. Returns (pts Nx3, valid mask)."""
    z = sample_depth(depth, pts_xy)
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (pts_xy[:, 0] - cx) * z / fx
    y = (pts_xy[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=-1), valid


# --------------------------------------------------------------------------- #
# ICP (Stage C polish)
# --------------------------------------------------------------------------- #

def icp_refine_open3d(src_pts, tgt_pts, voxel, max_dist):
    import open3d as o3d
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts))
    src = src.voxel_down_sample(voxel)
    tgt = tgt.voxel_down_sample(voxel)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    return np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)


def icp_refine_numpy(src_pts, tgt_pts, voxel, max_dist, iters=30):
    from scipy.spatial import cKDTree

    def voxel_ds(p, v):
        keys = np.floor(p / v).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return p[idx]

    src = voxel_ds(src_pts, voxel)
    tgt = voxel_ds(tgt_pts, voxel)
    tree = cKDTree(tgt)
    T_total = np.eye(4)
    src_cur = src.copy()
    rmse, fitness = np.inf, 0.0
    for _ in range(iters):
        dist, idx = tree.query(src_cur, k=1)
        mask = dist < max_dist
        if mask.sum() < 100:
            break
        p, q = src_cur[mask], tgt[idx[mask]]
        mu_p, mu_q = p.mean(0), q.mean(0)
        H = (p - mu_p).T @ (q - mu_q)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mu_q - R @ mu_p
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        src_cur = src_cur @ R.T + t
        T_total = step @ T_total
        rmse = float(np.sqrt(np.mean(dist[mask] ** 2)))
        fitness = float(mask.mean())
        if np.linalg.norm(t) < 1e-5 and np.abs(R - np.eye(3)).max() < 1e-6:
            break
    return T_total, fitness, rmse


def icp_refine(src_pts, tgt_pts, voxel, max_dist, log, name):
    try:
        T, fitness, rmse = icp_refine_open3d(src_pts, tgt_pts, voxel, max_dist)
        backend = "open3d_point_to_plane"
    except ImportError:
        try:
            T, fitness, rmse = icp_refine_numpy(src_pts, tgt_pts, voxel, max_dist)
            backend = "numpy_point_to_point"
        except ImportError:
            log(f"[icp] {name}: neither Open3D nor SciPy available, skipping")
            return np.eye(4), {"backend": "none"}
    dt = np.linalg.norm(T[:3, 3])
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    log(f"[icp] {name} ({backend}): |dt| = {dt * 1000:.1f} mm, dR = {ang:.3f} deg, "
        f"fitness {fitness:.3f}, rmse {rmse * 1000:.1f} mm")
    return T, {"backend": backend, "fitness": fitness, "inlier_rmse_m": rmse,
               "translation_correction_mm": dt * 1000, "rotation_correction_deg": ang,
               "transform": T.tolist()}


# --------------------------------------------------------------------------- #
# Geometry / IO
# --------------------------------------------------------------------------- #

def flying_pixel_mask(depth, rel_thresh):
    """
    True where depth is locally smooth. Discards points on strong depth
    discontinuities (box edges), which cause the streaky curtain artefacts.
    rel_thresh: max |dz| per pixel as a fraction of local depth.
    """
    dzdy, dzdx = np.gradient(depth)
    grad = np.sqrt(dzdx ** 2 + dzdy ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = grad / np.maximum(depth, 1e-6)
    return rel < rel_thresh


def backproject_dense(depth, K, stride, z_min, z_max, smooth_mask=None):
    H, W = depth.shape
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)
    if smooth_mask is not None:
        valid &= smooth_mask[vv, uu]
    uu, vv, z = uu[valid], vv[valid], z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=-1)
    return pts, (vv, uu)


def cam_to_reference(pts, T_ref_to_cam):
    R = T_ref_to_cam[:3, :3]
    t = T_ref_to_cam[:3, 3]
    return (pts - t) @ R


def apply_4x4(pts, T):
    return pts @ T[:3, :3].T + T[:3, 3]


def write_ply(path, pts, colors):
    pts = pts.astype(np.float32)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


def voxel_downsample_np(pts, colors, voxel):
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    acc_p = np.zeros((counts.shape[0], 3))
    acc_c = np.zeros((counts.shape[0], 3))
    np.add.at(acc_p, inv, pts)
    np.add.at(acc_c, inv, colors)
    return acc_p / counts[:, None], acc_c / counts[:, None]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def to_numpy(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def run_da3(images_rgb, model_id, device, fp16, log):
    import torch
    from depth_anything_3.api import DepthAnything3

    log(f"[model] loading {model_id} on {device}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()
    log(f"[model] loaded in {time.time() - t0:.1f} s")

    t0 = time.time()
    with torch.no_grad():
        if fp16 and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model.inference(images_rgb)
        else:
            prediction = model.inference(images_rgb)
    infer_s = time.time() - t0
    log(f"[model] inference ({len(images_rgb)} views) in {infer_s:.2f} s")

    depth = to_numpy(getattr(prediction, "depth", None))
    conf = to_numpy(getattr(prediction, "conf", None))
    if depth is None:
        raise RuntimeError("DA3 prediction has no .depth attribute; check package version")
    if depth.ndim == 2:
        depth = depth[None]
    if conf is not None and conf.ndim == 2:
        conf = conf[None]
    return depth, conf, infer_s


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="3-view DA3 fusion: SIFT scale/shift + feature-anchored rigid alignment + constrained ICP")
    ap.add_argument("--snapshot-dir", type=Path,
                    default=Path.home() / "Projects/MDE/DA3_v3/snapshots/20260720_142152_525")
    ap.add_argument("--results-dir", type=Path,
                    default=Path.home() / "Projects/MDE/DA3_v3/results")
    ap.add_argument("--extrinsics", type=Path, default=None)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--extrinsics-direction", choices=["ref_to_cam", "cam_to_ref"],
                    default="ref_to_cam")
    ap.add_argument("--flip-left", dest="flip_left", action="store_true", default=True)
    ap.add_argument("--no-flip-left", dest="flip_left", action="store_false")
    ap.add_argument("--no-undistort", dest="undistort", action="store_false", default=True)
    # Stage toggles (all on by default)
    ap.add_argument("--no-sift", dest="use_sift", action="store_false", default=True,
                    help="Disable Stage A scale-and-shift")
    ap.add_argument("--no-feat", dest="use_feat", action="store_false", default=True,
                    help="Disable Stage B feature-anchored rigid alignment")
    ap.add_argument("--no-icp", dest="use_icp", action="store_false", default=True,
                    help="Disable Stage C constrained ICP polish")
    # Stage parameters
    ap.add_argument("--scale-only", action="store_true",
                    help="Stage A: fit scale only, no shift term")
    ap.add_argument("--epi-thresh", type=float, default=3.0)
    ap.add_argument("--feat-tol", type=float, default=0.03,
                    help="Stage B RANSAC inlier tolerance in metres")
    ap.add_argument("--feat-sim3", action="store_true",
                    help="Stage B: also estimate a scale factor (Sim(3) instead of SE(3))")
    ap.add_argument("--icp-voxel", type=float, default=0.02)
    ap.add_argument("--icp-max-dist", type=float, default=0.04,
                    help="Stage C max correspondence distance in metres "
                         "(kept small deliberately so ICP cannot jump one box pitch)")
    # Cloud controls
    ap.add_argument("--edge-thresh", type=float, default=0.03,
                    help="Flying-pixel filter: max relative depth gradient per pixel (0 disables)")
    ap.add_argument("--global-scale", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--z-min", type=float, default=0.2)
    ap.add_argument("--z-max", type=float, default=8.0)
    ap.add_argument("--conf-min", type=float, default=0.0)
    ap.add_argument("--voxel", type=float, default=0.0)
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    extrinsics_path = args.extrinsics or (args.results_dir / "extrinsics.json")
    output_path = args.output or (args.snapshot_dir / "fused_scene.ply")
    diag_dir = args.snapshot_dir / "fusion_artifacts"
    diag_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    diagnostics = {"cameras": {}, "settings": {k: str(v) for k, v in vars(args).items()}}

    # ---- Calibration ------------------------------------------------------- #
    t0 = time.time()
    intrinsics = {}
    for cam in CAMERAS:
        K, dist = load_intrinsics(args.results_dir / f"intrinsics_{cam}.json")
        intrinsics[cam] = (K, dist)
        log(f"[intrinsics] {cam}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    extrinsics = load_extrinsics(extrinsics_path, args.extrinsics_direction, log)
    timings["load_calibration_s"] = time.time() - t0

    # ---- Images ------------------------------------------------------------- #
    t0 = time.time()
    images_rgb = {}
    infer_inputs = []
    for cam in CAMERAS:
        p = args.snapshot_dir / f"{cam}.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            log(f"ERROR: could not read {p}")
            sys.exit(1)
        if args.undistort:
            K, dist = intrinsics[cam]
            bgr = cv2.undistort(bgr, K, dist)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images_rgb[cam] = rgb
        if cam == "left" and args.flip_left:
            infer_inputs.append(rgb[::-1, ::-1].copy())
            log("[preprocess] left: rotated 180 deg for inference")
        else:
            infer_inputs.append(rgb)
    timings["preprocess_s"] = time.time() - t0

    # ---- Inference ----------------------------------------------------------- #
    depths_raw, confs, infer_s = run_da3(infer_inputs, args.model, args.device, args.fp16, log)
    timings["inference_s"] = infer_s

    # ---- Depth post-processing ------------------------------------------------ #
    depths = {}
    for i, cam in enumerate(CAMERAS):
        H, W = images_rgb[cam].shape[:2]
        d = depths_raw[i].astype(np.float32)
        c = confs[i].astype(np.float32) if confs is not None else None
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
            if c is not None:
                c = cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)
        if cam == "left" and args.flip_left:
            d = d[::-1, ::-1].copy()
            if c is not None:
                c = c[::-1, ::-1].copy()
        d *= args.global_scale
        if c is not None and args.conf_min > 0.0:
            d = np.where(c >= args.conf_min, d, 0.0)
        depths[cam] = d

    # ---- SIFT matching (computed once, reused by Stages A and B) --------------- #
    matches = {}
    if args.use_sift or args.use_feat:
        t0 = time.time()
        for side in SIDES:
            pts_c, pts_s = sift_match(images_rgb[REFERENCE], images_rgb[side])
            n_raw = pts_c.shape[0]
            if n_raw:
                keep = epipolar_filter(pts_c, pts_s, intrinsics[REFERENCE][0],
                                       intrinsics[side][0], extrinsics[side], args.epi_thresh)
                pts_c, pts_s = pts_c[keep], pts_s[keep]
            matches[side] = (pts_c, pts_s)
            log(f"[sift] {REFERENCE}<->{side}: {n_raw} raw -> {pts_c.shape[0]} epipolar-consistent")
        timings["sift_matching_s"] = time.time() - t0

    # ---- Stage A: scale-and-shift ------------------------------------------------ #
    if args.use_sift:
        t0 = time.time()
        anchors = {c: {"pred": [], "tri": []} for c in CAMERAS}
        for side in SIDES:
            pts_c, pts_s = matches[side]
            if pts_c.shape[0] == 0:
                continue
            _, z_ref, z_cam = triangulate_pair(pts_c, pts_s, intrinsics[REFERENCE][0],
                                               intrinsics[side][0], extrinsics[side])
            valid = (z_ref > args.z_min) & (z_ref < args.z_max) & \
                    (z_cam > args.z_min) & (z_cam < args.z_max)
            anchors[REFERENCE]["pred"].append(sample_depth(depths[REFERENCE], pts_c[valid]))
            anchors[REFERENCE]["tri"].append(z_ref[valid])
            anchors[side]["pred"].append(sample_depth(depths[side], pts_s[valid]))
            anchors[side]["tri"].append(z_cam[valid])

        sift_diag = {}
        for cam in CAMERAS:
            if not anchors[cam]["pred"]:
                sift_diag[cam] = {"a": 1.0, "b": 0.0, "note": "no anchors"}
                continue
            z_pred = np.concatenate(anchors[cam]["pred"])
            z_tri = np.concatenate(anchors[cam]["tri"])
            fit = robust_scale_shift(z_pred, z_tri, fit_shift=not args.scale_only)
            if fit is None:
                sift_diag[cam] = {"a": 1.0, "b": 0.0, "anchors": int(z_pred.size),
                                  "note": "fit failed"}
                log(f"[stageA] {cam}: fit failed, identity kept")
                continue
            a, b, n_inl, rms = fit
            depths[cam] = np.where(depths[cam] > 0, a * depths[cam] + b, 0.0).astype(np.float32)
            sift_diag[cam] = {"a": a, "b": b, "anchors": int(z_pred.size),
                              "inliers": n_inl, "inlier_rms_m": rms}
            log(f"[stageA] {cam}: depth' = {a:.4f} * depth {b:+.4f}  "
                f"({n_inl}/{z_pred.size} inliers, rms {rms * 100:.1f} cm)")
        diagnostics["stageA_scale_shift"] = sift_diag
        timings["stageA_s"] = time.time() - t0
        with open(diag_dir / "scale_shift_diagnostics.json", "w") as f:
            json.dump(sift_diag, f, indent=2)

    for cam in CAMERAS:
        np.save(diag_dir / f"depth_{cam}.npy", depths[cam])

    # ---- Flying-pixel masks -------------------------------------------------------- #
    smooth = {}
    for cam in CAMERAS:
        smooth[cam] = flying_pixel_mask(depths[cam], args.edge_thresh) \
            if args.edge_thresh > 0 else None
        if smooth[cam] is not None:
            log(f"[edges] {cam}: keeping {smooth[cam].mean() * 100:.1f}% of pixels "
                f"(rel gradient < {args.edge_thresh})")

    # ---- Back-projection ------------------------------------------------------------ #
    t0 = time.time()
    clouds, colors = {}, {}
    for cam in CAMERAS:
        K, _ = intrinsics[cam]
        pts_cam, (vv, uu) = backproject_dense(depths[cam], K, args.stride,
                                              args.z_min, args.z_max, smooth[cam])
        clouds[cam] = cam_to_reference(pts_cam, extrinsics[cam])
        colors[cam] = images_rgb[cam][vv, uu]
        d_valid = depths[cam][(depths[cam] > args.z_min) & (depths[cam] < args.z_max)]
        diagnostics["cameras"][cam] = {
            "points": int(clouds[cam].shape[0]),
            "depth_median_m": float(np.median(d_valid)) if d_valid.size else None,
        }
        log(f"[cloud] {cam}: {clouds[cam].shape[0]} points, "
            f"median depth {diagnostics['cameras'][cam]['depth_median_m']} m")
    timings["backproject_s"] = time.time() - t0

    # ---- Stage B: feature-anchored rigid alignment ------------------------------------ #
    if args.use_feat:
        t0 = time.time()
        feat_diag = {}
        for side in SIDES:
            pts_c, pts_s = matches[side]
            if pts_c.shape[0] < 10:
                log(f"[stageB] {side}: too few matches, skipping")
                feat_diag[side] = {"note": "too few matches"}
                continue
            # 3D-3D correspondences on the SAME physical surface point:
            p_ref_c, ok_c = backproject_pixels(pts_c, depths[REFERENCE],
                                               intrinsics[REFERENCE][0],
                                               args.z_min, args.z_max)
            p_cam_s, ok_s = backproject_pixels(pts_s, depths[side],
                                               intrinsics[side][0],
                                               args.z_min, args.z_max)
            ok = ok_c & ok_s
            if ok.sum() < 10:
                log(f"[stageB] {side}: too few valid-depth correspondences, skipping")
                feat_diag[side] = {"note": "too few valid correspondences"}
                continue
            src = cam_to_reference(p_cam_s[ok], extrinsics[side])  # side cloud coords
            dst = p_ref_c[ok]                                       # center cloud coords
            T, inl = ransac_rigid(src, dst, tol=args.feat_tol,
                                  with_scale=args.feat_sim3)
            if T is None:
                log(f"[stageB] {side}: RANSAC failed, skipping")
                feat_diag[side] = {"note": "ransac failed",
                                   "correspondences": int(ok.sum())}
                continue
            res = np.linalg.norm(src[inl] @ T[:3, :3].T + T[:3, 3] - dst[inl], axis=1)
            clouds[side] = apply_4x4(clouds[side], T)
            dt = np.linalg.norm(T[:3, 3])
            ang = np.degrees(np.arccos(np.clip(
                (np.trace(T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))) - 1) / 2, -1, 1)))
            feat_diag[side] = {
                "correspondences": int(ok.sum()), "inliers": int(inl.sum()),
                "inlier_rms_mm": float(np.sqrt(np.mean(res ** 2)) * 1000),
                "translation_correction_mm": dt * 1000,
                "rotation_correction_deg": float(ang),
                "transform": T.tolist(),
            }
            log(f"[stageB] {side}: {inl.sum()}/{ok.sum()} inliers, "
                f"|dt| = {dt * 1000:.1f} mm, dR = {ang:.3f} deg, "
                f"rms {feat_diag[side]['inlier_rms_mm']:.1f} mm")
        diagnostics["stageB_feature_alignment"] = feat_diag
        timings["stageB_s"] = time.time() - t0

    # ---- Stage C: constrained ICP polish ----------------------------------------------- #
    if args.use_icp:
        t0 = time.time()
        icp_diag = {}
        for side in SIDES:
            T, info = icp_refine(clouds[side], clouds[REFERENCE],
                                 args.icp_voxel, args.icp_max_dist, log, side)
            clouds[side] = apply_4x4(clouds[side], T)
            icp_diag[side] = info
        diagnostics["stageC_icp"] = icp_diag
        timings["stageC_s"] = time.time() - t0

    # ---- Fuse ---------------------------------------------------------------------------- #
    pts = np.concatenate([clouds[c] for c in CAMERAS], axis=0)
    cols = np.concatenate([colors[c] for c in CAMERAS], axis=0).astype(np.float64)
    log(f"[fuse] concatenated cloud: {pts.shape[0]} points")

    if args.voxel > 0.0:
        t0 = time.time()
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols / 255.0)
            pcd = pcd.voxel_down_sample(args.voxel)
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors) * 255.0
            log(f"[downsample] Open3D voxel {args.voxel} m -> {pts.shape[0]} points")
        except ImportError:
            pts, cols = voxel_downsample_np(pts, cols, args.voxel)
            log(f"[downsample] NumPy voxel {args.voxel} m -> {pts.shape[0]} points")
        timings["downsample_s"] = time.time() - t0

    # ---- Outputs --------------------------------------------------------------------------- #
    t0 = time.time()
    write_ply(output_path, pts, cols)
    timings["write_ply_s"] = time.time() - t0
    diagnostics["fused_points"] = int(pts.shape[0])
    diagnostics["output_ply"] = str(output_path)

    with open(diag_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2)
    with open(diag_dir / "fusion_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    log(f"[done] PLY written to {output_path}")
    log(f"[done] artifacts in {diag_dir}")


if __name__ == "__main__":
    main()