#!/usr/bin/env python3
"""
Multi-camera monocular depth estimation (MDE) fusion using MoGe-2.

Pipeline:
    1. Load per-camera intrinsics + multi-camera extrinsics.
    2. Undistort each raw image with cv2.undistort using the full 14-element
       distortion vector; carry a validity mask.
    3. Run MoGe-2 per view -> metric depth + per-view validity mask, resized
       to input resolution.
    4. SIFT-triangulation cross-view scale+shift alignment per camera
       (z_true ≈ s * z_mde + t, IRLS-reweighted least squares).
    5. Back-project corrected depth with calibrated K, applying MoGe mask +
       undistort-validity mask + depth-range clamp + edge filter.
    6. Transform per-camera clouds into the center (world) frame.
    7. Optional point-to-plane ICP refinement of side cameras -> center.
    8. Optional TSDF volumetric fusion via Open3D ScalableTSDFVolume.
    9. Per-stage timings + scale-shift diagnostics persisted to JSON.

Ablation matrix (flag combos):
    baseline                : (no flags)
    + scale-shift alignment : --align-depths
    + ICP                   : --align-depths --icp
    + TSDF                  : --align-depths --tsdf
    full                    : --align-depths --icp --tsdf

Calibration conventions:
    intrinsics_<cam>.json   : { camera_matrix(3x3), dist_coeffs(14),
                                image_size[W,H], distortion_model }
    extrinsics.json         : { <cam>: { R(3x3), t(3), reference: "center" } }
        Convention: P_cam = R @ P_center + t  (OpenCV stereoCalibrate output,
        i.e. world-to-camera with world = center). Use --inverse-extrinsics
        if your file stores cam-to-world instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch


# =============================================================================
# Stage timing
# =============================================================================

TIMINGS: Dict[str, float] = {}


@contextmanager
def stage(name: str):
    """Time a pipeline stage; log to stdout AND persist to TIMINGS dict."""
    print(f"[stage] {name} ...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        TIMINGS[name] = dt
        print(f"[stage] {name} done in {dt:.3f} s", flush=True)


# =============================================================================
# Calibration loading
# =============================================================================

@dataclass
class Intrinsics:
    name: str
    K: np.ndarray         # 3x3 float64
    dist: np.ndarray      # 14-vector: k1 k2 p1 p2 k3 k4 k5 k6 [+ thin prism + tilted, zeros for us]
    size: Tuple[int, int] # (W, H)


def load_intrinsics(path: Path, expected_name: Optional[str] = None) -> Intrinsics:
    with open(path) as f:
        data = json.load(f)
    K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    if dist.size < 14:
        dist = np.pad(dist, (0, 14 - dist.size))
    elif dist.size > 14:
        dist = dist[:14]
    size = tuple(int(v) for v in data["image_size"])
    name = data.get("camera", expected_name or path.stem)
    return Intrinsics(name=name, K=K, dist=dist, size=size)


def load_extrinsics(path: Path, cameras: List[str],
                    inverse: bool = False) -> Dict[str, np.ndarray]:
    """
    Returns T_w2c per camera as 4x4. world = center, so center is identity.
    Convention: x_cam = R @ x_world + t.
    """
    with open(path) as f:
        data = json.load(f)
    extrinsics: Dict[str, np.ndarray] = {}
    for cam in cameras:
        if cam == "center":
            T = np.eye(4)
        else:
            if cam not in data:
                raise KeyError(f"Camera '{cam}' not in {path}")
            entry = data[cam]
            R = np.asarray(entry["R"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(entry["t"], dtype=np.float64).reshape(3)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
        if inverse:
            T = np.linalg.inv(T)
        extrinsics[cam] = T
    return extrinsics


# =============================================================================
# Image I/O + undistortion
# =============================================================================

def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img


def undistort_with_validity(img: np.ndarray, K: np.ndarray, dist: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    cv2.undistort with the full 14-element distortion vector. The new camera
    matrix is kept identical to K, so calibrated K can be reused for
    back-projection of the undistorted image.

    Validity mask: True where the undistorted pixel was sampled from inside
    the original image bounds. A 3x3 erosion suppresses interpolation
    artifacts at the validity boundary.
    """
    H, W = img.shape[:2]
    und = cv2.undistort(img, K, dist, None, K)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, K, (W, H), cv2.CV_32FC1
    )
    valid = ((map1 >= 0) & (map1 <= W - 1)
             & (map2 >= 0) & (map2 <= H - 1))
    valid = cv2.erode(valid.astype(np.uint8),
                      np.ones((3, 3), np.uint8), iterations=1)
    return und, valid.astype(bool)


# =============================================================================
# MoGe-2 inference
# =============================================================================

def load_moge_model(checkpoint: str, device: torch.device):
    """Load MoGe-2 metric model. Falls back to v1 import path if needed."""
    try:
        from moge.model.v2 import MoGeModel
    except ImportError:
        from moge.model import MoGeModel  # type: ignore
    model = MoGeModel.from_pretrained(checkpoint)
    model = model.to(device).eval()
    return model


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def run_moge_inference(model, img_rgb_u8: np.ndarray, device: torch.device,
                       fp16: bool = False,
                       fov_x_deg: Optional[float] = None,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (depth_m: HxW float32, mask: HxW bool) at input resolution.

    Notes:
      - On Jetson we never call model.half(); LayerNorm in the ViT backbone
        loses precision in pure fp16. torch.autocast handles mixed precision
        safely. We also pass use_fp16=False to MoGe's infer() so it doesn't
        cast modules internally.
      - MoGe's depth may be at a downsampled internal resolution; we resize
        with INTER_LINEAR (mask with INTER_NEAREST) before any filtering.
      - MoGe's predicted intrinsics are ignored on purpose for back-projection
        — we use the calibrated K. But we DO pass the calibrated FOV into
        infer() so MoGe doesn't have to guess (per-view FOV guesses are the
        biggest source of cross-view metric-scale drift).
    """
    H, W = img_rgb_u8.shape[:2]
    img_t = (torch.from_numpy(img_rgb_u8)
             .permute(2, 0, 1).float().div(255.0).to(device))

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if fp16 and device.type == "cuda" else nullcontext()
    )

    infer_kwargs = {"use_fp16": False, "apply_mask": False}
    if fov_x_deg is not None:
        infer_kwargs["fov_x"] = fov_x_deg

    with torch.inference_mode(), autocast_ctx:
        try:
            out = model.infer(img_t, **infer_kwargs)
        except TypeError:
            # Older MoGe API: may not accept use_fp16/apply_mask kwargs.
            # Still try to pass fov_x if we have one.
            try:
                if fov_x_deg is not None:
                    out = model.infer(img_t, fov_x=fov_x_deg)
                else:
                    out = model.infer(img_t)
            except TypeError:
                out = model.infer(img_t)

    depth = _to_numpy(out["depth"])
    if "mask" in out:
        mask = _to_numpy(out["mask"]).astype(bool)
    else:
        mask = np.ones((H, W), dtype=bool)

    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    bad = ~np.isfinite(depth)
    if bad.any():
        depth = np.where(bad, 0.0, depth)
        mask = mask & ~bad
    return depth.astype(np.float32), mask


# =============================================================================
# SIFT + matching
# =============================================================================

def detect_sift(img_bgr: np.ndarray, n_features: int = 5000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_lowe(des1, des2, ratio: float = 0.75):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


# =============================================================================
# Triangulation + scale-shift alignment
# =============================================================================

def projection_matrix(K: np.ndarray, T_w2c: np.ndarray) -> np.ndarray:
    """3x4 P = K @ [R | t] in world (center) frame."""
    return K @ T_w2c[:3, :4]


def triangulate_pair(kp_a, kp_b, matches, P_a: np.ndarray, P_b: np.ndarray,
                     max_reproj_px: float = 2.0
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DLT-triangulate matched keypoints between two views. Reject by
    reprojection error > max_reproj_px in either camera. Returns
    (X_world Nx3, pts_a Nx2, pts_b Nx2) after filtering.
    """
    if not matches:
        return (np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))

    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], dtype=np.float64)

    Xh = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)  # 4xN
    w = Xh[3]
    valid = np.abs(w) > 1e-9
    X = np.full((Xh.shape[1], 3), np.nan)
    X[valid] = (Xh[:3, valid] / w[valid]).T

    def reproj(P, X_):
        Xh_ = np.hstack([X_, np.ones((X_.shape[0], 1))])
        x = (P @ Xh_.T).T                       # (N, 3)
        z = x[:, 2]                             # (N,)
        good = np.abs(z) > 1e-9                 # (N,) 1-D mask
        out = np.full((X_.shape[0], 2), np.nan)
        out[good] = x[good, :2] / z[good, None]
        return out, good

    rp_a, good_a = reproj(P_a, X)
    rp_b, good_b = reproj(P_b, X)
    with np.errstate(invalid="ignore"):
        err_a = np.linalg.norm(rp_a - pts_a, axis=1)
        err_b = np.linalg.norm(rp_b - pts_b, axis=1)
        keep = (valid & good_a & good_b
                & (err_a < max_reproj_px)
                & (err_b < max_reproj_px))
    return X[keep], pts_a[keep], pts_b[keep]


def sample_depth_bilinear(depth: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Bilinear depth sampling. pts_xy is Nx2 (x, y) in image pixels."""
    if pts_xy.size == 0:
        return np.empty(0, dtype=np.float32)
    map_x = pts_xy[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = pts_xy[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(depth.astype(np.float32), map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return sampled.flatten()


def transform_points_world_to_cam(X_world: np.ndarray, T_w2c: np.ndarray
                                  ) -> np.ndarray:
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]
    return X_world @ R.T + t


def irls_scale_shift(z_mde: np.ndarray, z_true: np.ndarray,
                     iters: int = 4
                     ) -> Tuple[float, float, np.ndarray]:
    """
    Robust affine fit  z_true ≈ s * z_mde + t  with IRLS soft weights
        sigma = median(|residual|)
        w_i   = sigma / (sigma + |residual_i|)
    Initialised with ordinary least squares.
    """
    A = np.stack([z_mde, np.ones_like(z_mde)], axis=1)
    coef, *_ = np.linalg.lstsq(A, z_true, rcond=None)
    s, t = float(coef[0]), float(coef[1])

    w = np.ones_like(z_mde)
    for _ in range(iters):
        resid = z_true - (s * z_mde + t)
        sigma = float(np.median(np.abs(resid)) + 1e-6)
        w = sigma / (sigma + np.abs(resid))
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        bw = z_true * sw
        coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        s, t = float(coef[0]), float(coef[1])
    return s, t, w


def compute_scale_shift_alignment(
    cameras: List[str],
    sift_results: Dict[str, Tuple[list, np.ndarray]],
    depths: Dict[str, np.ndarray],
    sample_masks: Dict[str, np.ndarray],
    Ks: Dict[str, np.ndarray],
    extrinsics: Dict[str, np.ndarray],
    ratio: float = 0.75,
    max_reproj_px: float = 2.0,
    irls_iters: int = 4,
) -> Dict[str, dict]:
    """
    Cross-view (z_mde, z_true) collection + robust per-camera affine fit.

    For each ordered pair (i, j), Lowe-match SIFT, DLT-triangulate in the
    world (center) frame using calibrated P_i, P_j, drop reprojection
    outliers, then attribute each surviving 3D point to both cameras:
        z_true_i = (R_w2c_i @ X_world + t_w2c_i).z
        z_mde_i  = bilinear-sampled depth at feature pixel u_i
    The IRLS affine fit is then run independently per camera.

    Returns a diagnostics dict { cam: { s, t, inlier_count, total_pairs,
    rmse_before_mm, rmse_after_mm } }.
    """
    pairs: Dict[str, List[Tuple[float, float]]] = {c: [] for c in cameras}
    Ps = {c: projection_matrix(Ks[c], extrinsics[c]) for c in cameras}

    for i, ci in enumerate(cameras):
        kp_i, des_i = sift_results[ci]
        for cj in cameras[i + 1:]:
            kp_j, des_j = sift_results[cj]
            matches = match_lowe(des_i, des_j, ratio=ratio)
            if not matches:
                continue
            X_world, pts_i, pts_j = triangulate_pair(
                kp_i, kp_j, matches, Ps[ci], Ps[cj], max_reproj_px=max_reproj_px
            )
            if X_world.shape[0] == 0:
                continue

            X_in_i = transform_points_world_to_cam(X_world, extrinsics[ci])
            X_in_j = transform_points_world_to_cam(X_world, extrinsics[cj])
            z_true_i = X_in_i[:, 2]
            z_true_j = X_in_j[:, 2]
            z_mde_i = sample_depth_bilinear(depths[ci], pts_i)
            z_mde_j = sample_depth_bilinear(depths[cj], pts_j)

            def keep(pts, mask, zm, zt):
                ix = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)
                iy = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1)
                in_mask = mask[iy, ix]
                return (in_mask & (zm > 1e-3) & (zt > 1e-3)
                        & np.isfinite(zm) & np.isfinite(zt))

            keep_i = keep(pts_i, sample_masks[ci], z_mde_i, z_true_i)
            keep_j = keep(pts_j, sample_masks[cj], z_mde_j, z_true_j)

            for k in np.nonzero(keep_i)[0]:
                pairs[ci].append((float(z_mde_i[k]), float(z_true_i[k])))
            for k in np.nonzero(keep_j)[0]:
                pairs[cj].append((float(z_mde_j[k]), float(z_true_j[k])))

            print(f"[align] pair ({ci}, {cj}): {len(matches)} matches -> "
                  f"{X_world.shape[0]} triangulated; "
                  f"contribs {int(keep_i.sum())}/{int(keep_j.sum())}")

    diagnostics: Dict[str, dict] = {}
    for c in cameras:
        plist = pairs[c]
        if len(plist) < 10:
            print(f"[align] WARNING: only {len(plist)} pairs for {c}; "
                  f"falling back to identity (s=1, t=0).")
            diagnostics[c] = {
                "s": 1.0, "t": 0.0,
                "inlier_count": len(plist),
                "total_pairs": len(plist),
                "rmse_before_mm": None,
                "rmse_after_mm": None,
            }
            continue
        arr = np.asarray(plist, dtype=np.float64)
        z_mde, z_true = arr[:, 0], arr[:, 1]
        s, t, w = irls_scale_shift(z_mde, z_true, iters=irls_iters)
        inl = w > 0.5
        resid_before = z_true - z_mde
        resid_after = z_true - (s * z_mde + t)
        rmse_before_mm = (float(np.sqrt(np.mean(resid_before[inl] ** 2)) * 1000.0)
                          if inl.any() else None)
        rmse_after_mm = (float(np.sqrt(np.mean(resid_after[inl] ** 2)) * 1000.0)
                         if inl.any() else None)
        diagnostics[c] = {
            "s": s,
            "t": t,
            "inlier_count": int(inl.sum()),
            "total_pairs": int(arr.shape[0]),
            "rmse_before_mm": rmse_before_mm,
            "rmse_after_mm": rmse_after_mm,
        }
        print(f"[align] {c}: s={s:.4f}, t={t:+.4f} m, "
              f"inliers={int(inl.sum())}/{arr.shape[0]}, "
              f"RMSE {rmse_before_mm:.1f} -> {rmse_after_mm:.1f} mm")
    return diagnostics


# =============================================================================
# Back-projection + filtering
# =============================================================================

def depth_edge_mask(depth: np.ndarray, rel_threshold: float = 0.05) -> np.ndarray:
    """Keep pixels with |grad(depth)| / depth below threshold (no edges)."""
    d = depth.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    rel = mag / np.maximum(d, 1e-6)
    return rel < rel_threshold


def backproject(depth: np.ndarray, color_rgb: np.ndarray, K: np.ndarray,
                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project masked pixels with calibrated K. Returns (Nx3, Nx3) in cam frame."""
    vs, us = np.where(mask)
    z = depth[vs, us].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    cols = color_rgb[vs, us].astype(np.float32) / 255.0
    return pts, cols


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T + t).astype(np.float32)


# =============================================================================
# Open3D helpers (cloud / frustum / ICP / TSDF)
# =============================================================================

def to_o3d_cloud(pts: np.ndarray, cols: Optional[np.ndarray] = None
                 ) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if cols is not None and cols.size:
        pc.colors = o3d.utility.Vector3dVector(
            np.clip(cols, 0.0, 1.0).astype(np.float64)
        )
    return pc


def make_frustum(K: np.ndarray, T_w2c: np.ndarray,
                 image_size: Tuple[int, int],
                 depth: float = 0.2,
                 color=(1.0, 0.6, 0.0)) -> o3d.geometry.LineSet:
    W, H = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_px = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    corners_cam = np.stack(
        [(corners_px[:, 0] - cx) * depth / fx,
         (corners_px[:, 1] - cy) * depth / fy,
         np.full(4, depth)], axis=1,
    )
    apex_cam = np.zeros((1, 3))
    pts_cam = np.vstack([apex_cam, corners_cam])

    # cam -> world for row-vector points: x_w = (x_c - t) @ R.
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]
    pts_world = (pts_cam - t) @ R

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def icp_align(src: o3d.geometry.PointCloud,
              dst: o3d.geometry.PointCloud,
              threshold: float = 0.05
              ) -> Tuple[np.ndarray, float, float]:
    """Point-to-plane ICP, identity init. Returns (T, fitness, inlier_rmse)."""
    if len(dst.normals) == 0:
        dst.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
    res = o3d.pipelines.registration.registration_icp(
        src, dst, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return np.asarray(res.transformation), float(res.fitness), float(res.inlier_rmse)


def tsdf_fuse(per_cam: Dict[str, dict], voxel: float, trunc: float,
              depth_min: float, depth_max: float
              ) -> o3d.geometry.PointCloud:
    """
    Volumetric fusion via Open3D ScalableTSDFVolume. Per-camera depth maps
    are masked + range-clamped (invalid pixels set to 0, which Open3D
    treats as no measurement). T_w2c is the post-ICP camera pose.
    """
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for cam, d in per_cam.items():
        depth = d["depth"].astype(np.float32).copy()
        mask = d["mask"]
        depth[~mask] = 0.0
        depth[(depth < depth_min) | (depth > depth_max)] = 0.0

        H, W = depth.shape
        K = d["K"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2],
        )
        color_img = o3d.geometry.Image(np.ascontiguousarray(d["color_rgb"]))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        vol.integrate(rgbd, intrinsic, d["T_w2c"])
    return vol.extract_point_cloud()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-camera MoGe-2 depth fusion (thesis pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--snapshot",
        default="/home/jetson/Projects/Image Capture/snapshots/20260522_093514_634",
        help="Directory containing per-camera <cam>.png images.",
    )
    p.add_argument(
        "--calib-dir",
        default="/home/jetson/Projects/Calibration/results",
        help="Directory for intrinsics_*.json + extrinsics.json. "
             "Falls back to --snapshot, then CWD if not found here.",
    )
    p.add_argument("--cameras", nargs="+",
                   default=["center", "left", "right"],
                   help="Ordered list of cameras. 'center' is the world frame.")
    p.add_argument("--output-dir", default="./fused_output")
    p.add_argument("--moge-checkpoint", default="Ruicheng/moge-2-vitl-normal",
                   help="HuggingFace ID of the MoGe-2 metric checkpoint.")

    # Ablation knobs
    p.add_argument("--align-depths", action="store_true",
                   help="SIFT-triangulation cross-view scale+shift alignment.")
    p.add_argument("--icp", action="store_true",
                   help="Point-to-plane ICP refinement of side cameras -> center.")
    p.add_argument("--tsdf", action="store_true",
                   help="Use Open3D ScalableTSDFVolume instead of point concatenation.")

    # Fusion knobs
    p.add_argument("--tsdf-voxel", type=float, default=0.005)
    p.add_argument("--tsdf-trunc", type=float, default=0.04)
    p.add_argument("--edge-rel-threshold", type=float, default=0.05)
    p.add_argument("--depth-min", type=float, default=0.3)
    p.add_argument("--depth-max", type=float, default=15.0)
    p.add_argument("--icp-threshold", type=float, default=0.05)

    # Alignment knobs
    p.add_argument("--sift-features", type=int, default=5000)
    p.add_argument("--lowe-ratio", type=float, default=0.75)
    p.add_argument("--reproj-thresh", type=float, default=2.0)
    p.add_argument("--irls-iters", type=int, default=4)

    # I/O & runtime
    p.add_argument("--inverse-extrinsics", action="store_true",
                   help="Treat extrinsics file as cam-to-world (default is world-to-cam).")
    p.add_argument("--save-per-camera", action="store_true",
                   help="Save per-camera clouds + frustums for visual debugging.")
    p.add_argument("--fp16", action="store_true",
                   help="torch.autocast(fp16). Never calls model.half().")
    p.add_argument("--no-moge-fov", action="store_true",
                   help="Don't pass calibrated FOV to MoGe; let it self-estimate. "
                        "Default is to pass it — per-view FOV mismatch is the "
                        "single biggest source of cross-view scale drift.")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def _resolve_calib_path(calib_dir: Optional[Path], snapshot_dir: Path,
                        filename: str) -> Path:
    candidates: List[Path] = []
    if calib_dir is not None:
        candidates.append(calib_dir / filename)
    candidates.append(snapshot_dir / filename)
    candidates.append(Path.cwd() / filename)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {[str(c) for c in candidates]}"
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  torch={torch.__version__}  "
          f"cv2={cv2.__version__}  o3d={o3d.__version__}")

    cameras: List[str] = list(args.cameras)
    if "center" not in cameras:
        print(f"[init] WARNING: 'center' not in --cameras; using {cameras[0]} "
              f"as the world frame.")
    snapshot_dir = Path(args.snapshot)
    calib_dir = Path(args.calib_dir) if args.calib_dir else None

    # 1. Calibration --------------------------------------------------------
    with stage("load_calibration"):
        intrinsics: Dict[str, Intrinsics] = {}
        for cam in cameras:
            path = _resolve_calib_path(calib_dir, snapshot_dir, f"intrinsics_{cam}.json")
            intrinsics[cam] = load_intrinsics(path, expected_name=cam)
            print(f"[calib] {cam}: K=\n{intrinsics[cam].K}\n"
                  f"        dist[0:8]={intrinsics[cam].dist[:8]}  size={intrinsics[cam].size}")
        extr_path = _resolve_calib_path(calib_dir, snapshot_dir, "extrinsics.json")
        extrinsics = load_extrinsics(extr_path, cameras,
                                     inverse=args.inverse_extrinsics)

    # 2. Undistortion -------------------------------------------------------
    images_bgr: Dict[str, np.ndarray] = {}
    valid_masks: Dict[str, np.ndarray] = {}
    with stage("undistort"):
        for cam in cameras:
            img = load_image_bgr(snapshot_dir / f"{cam}.png")
            und, valid = undistort_with_validity(
                img, intrinsics[cam].K, intrinsics[cam].dist
            )
            images_bgr[cam] = und
            valid_masks[cam] = valid
            print(f"[undistort] {cam}: shape={und.shape}, "
                  f"valid_frac={valid.mean():.3f}")

    # 3. MoGe-2 inference --------------------------------------------------
    depths: Dict[str, np.ndarray] = {}
    moge_masks: Dict[str, np.ndarray] = {}
    with stage("moge_inference"):
        model = load_moge_model(args.moge_checkpoint, device)
        for cam in cameras:
            img_rgb = cv2.cvtColor(images_bgr[cam], cv2.COLOR_BGR2RGB)
            fov_x_deg = None
            if not args.no_moge_fov:
                K_cam = intrinsics[cam].K
                W_cam = float(intrinsics[cam].size[0])
                fov_x_rad = 2.0 * np.arctan(W_cam / (2.0 * K_cam[0, 0]))
                fov_x_deg = float(np.degrees(fov_x_rad))
                print(f"[moge] {cam}: passing fov_x={fov_x_deg:.2f} deg")
            depth, mask = run_moge_inference(model, img_rgb, device,
                                             fp16=args.fp16,
                                             fov_x_deg=fov_x_deg)
            depths[cam] = depth
            moge_masks[cam] = mask
            in_mask = depth[mask]
            if in_mask.size:
                print(f"[moge] {cam}: depth median={np.median(in_mask):.3f} m "
                      f"min={in_mask.min():.3f} max={in_mask.max():.3f}  "
                      f"valid={mask.mean():.3f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Cross-view scale+shift alignment ----------------------------------
    diagnostics: Dict[str, dict] = {}
    if args.align_depths:
        with stage("sift_detect"):
            sift_results = {
                cam: detect_sift(images_bgr[cam], n_features=args.sift_features)
                for cam in cameras
            }
            for cam, (kp, _) in sift_results.items():
                print(f"[sift] {cam}: {len(kp)} keypoints")

        with stage("scale_shift_alignment"):
            sample_masks = {c: (moge_masks[c] & valid_masks[c]) for c in cameras}
            Ks = {c: intrinsics[c].K for c in cameras}
            diagnostics = compute_scale_shift_alignment(
                cameras, sift_results, depths, sample_masks, Ks, extrinsics,
                ratio=args.lowe_ratio, max_reproj_px=args.reproj_thresh,
                irls_iters=args.irls_iters,
            )
            for cam, d in diagnostics.items():
                depths[cam] = (d["s"] * depths[cam] + d["t"]).astype(np.float32)
            with open(output_dir / "scale_shift_diagnostics.json", "w") as f:
                json.dump(diagnostics, f, indent=2)

    # 5. Back-projection ---------------------------------------------------
    per_cam_buffers: Dict[str, dict] = {}
    with stage("backproject"):
        for cam in cameras:
            depth = depths[cam]
            color_rgb = cv2.cvtColor(images_bgr[cam], cv2.COLOR_BGR2RGB)
            range_mask = (depth >= args.depth_min) & (depth <= args.depth_max)
            edge_ok = depth_edge_mask(depth, rel_threshold=args.edge_rel_threshold)
            final_mask = (moge_masks[cam] & valid_masks[cam] & range_mask
                          & edge_ok & np.isfinite(depth))
            pts_cam, cols = backproject(depth, color_rgb,
                                        intrinsics[cam].K, final_mask)
            per_cam_buffers[cam] = {
                "pts_cam": pts_cam,
                "cols": cols,
                "depth": depth,
                "mask": final_mask,
                "color_rgb": color_rgb,
                "K": intrinsics[cam].K,
                "T_w2c": extrinsics[cam].copy(),
            }
            print(f"[bp] {cam}: {final_mask.sum():,}/{final_mask.size:,} "
                  f"valid ({100.0 * final_mask.mean():.1f}%)")

    # 6. Transform to center frame -----------------------------------------
    with stage("transform_to_center"):
        for cam in cameras:
            T_c2w = np.linalg.inv(per_cam_buffers[cam]["T_w2c"])
            per_cam_buffers[cam]["pts_world"] = transform_points(
                per_cam_buffers[cam]["pts_cam"], T_c2w
            )

    # 7. Optional ICP -------------------------------------------------------
    icp_results: Dict[str, dict] = {}
    if args.icp:
        with stage("icp"):
            ref_cam = "center" if "center" in cameras else cameras[0]
            ref_pc = to_o3d_cloud(per_cam_buffers[ref_cam]["pts_world"],
                                  per_cam_buffers[ref_cam]["cols"])
            ref_pc.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )
            for cam in cameras:
                if cam == ref_cam:
                    icp_results[cam] = {
                        "fitness": 1.0, "inlier_rmse_mm": 0.0,
                        "transform": np.eye(4).tolist(),
                    }
                    continue
                src = to_o3d_cloud(per_cam_buffers[cam]["pts_world"],
                                   per_cam_buffers[cam]["cols"])
                T, fitness, rmse = icp_align(
                    src, ref_pc, threshold=args.icp_threshold
                )
                per_cam_buffers[cam]["pts_world"] = transform_points(
                    per_cam_buffers[cam]["pts_world"], T
                )
                # Re-derive T_w2c so TSDF integration uses the post-ICP pose:
                # cam-to-world_new = T @ cam-to-world_old  =>
                # T_w2c_new = T_w2c_old @ inv(T)
                per_cam_buffers[cam]["T_w2c"] = (
                    per_cam_buffers[cam]["T_w2c"] @ np.linalg.inv(T)
                )
                icp_results[cam] = {
                    "fitness": fitness,
                    "inlier_rmse_mm": rmse * 1000.0,
                    "transform": T.tolist(),
                }
                print(f"[icp] {cam}: fitness={fitness:.3f}, "
                      f"inlier_rmse={rmse * 1000:.2f} mm")
            with open(output_dir / "icp_diagnostics.json", "w") as f:
                json.dump(icp_results, f, indent=2)

    # 8. Fuse ---------------------------------------------------------------
    if args.tsdf:
        with stage("tsdf_fuse"):
            fused = tsdf_fuse(
                per_cam_buffers,
                voxel=args.tsdf_voxel, trunc=args.tsdf_trunc,
                depth_min=args.depth_min, depth_max=args.depth_max,
            )
            out_ply = output_dir / "fused_tsdf.ply"
            o3d.io.write_point_cloud(str(out_ply), fused)
            print(f"[fuse] TSDF -> {out_ply}  ({len(fused.points):,} pts)")
    else:
        with stage("concat_clouds"):
            all_pts = np.concatenate(
                [per_cam_buffers[c]["pts_world"] for c in cameras], axis=0
            )
            all_cols = np.concatenate(
                [per_cam_buffers[c]["cols"] for c in cameras], axis=0
            )
            fused = to_o3d_cloud(all_pts, all_cols)
            out_ply = output_dir / "fused.ply"
            o3d.io.write_point_cloud(str(out_ply), fused)
            print(f"[fuse] concat -> {out_ply}  ({len(fused.points):,} pts)")

    # 9. Per-camera diagnostics --------------------------------------------
    if args.save_per_camera:
        with stage("save_per_camera"):
            pc_dir = output_dir / "per_camera"
            pc_dir.mkdir(exist_ok=True)
            for cam in cameras:
                pc = to_o3d_cloud(per_cam_buffers[cam]["pts_world"],
                                  per_cam_buffers[cam]["cols"])
                o3d.io.write_point_cloud(str(pc_dir / f"{cam}.ply"), pc)
            color_map = {
                "center": (0.2, 0.8, 0.2),
                "left":   (0.2, 0.6, 1.0),
                "right":  (1.0, 0.4, 0.4),
            }
            frustums = o3d.geometry.LineSet()
            for cam in cameras:
                fr = make_frustum(
                    intrinsics[cam].K,
                    per_cam_buffers[cam]["T_w2c"],
                    intrinsics[cam].size,
                    depth=0.2,
                    color=color_map.get(cam, (0.7, 0.7, 0.7)),
                )
                frustums += fr
            o3d.io.write_line_set(str(pc_dir / "frustums.ply"), frustums)
            print(f"[per_cam] wrote per-camera clouds + frustums to {pc_dir}")

    with open(output_dir / "timings.json", "w") as f:
        json.dump(TIMINGS, f, indent=2)
    print(f"[done] outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())