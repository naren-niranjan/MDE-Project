#!/usr/bin/env python3
"""
Multi-view metric point-cloud fusion from 3 calibrated cameras.

Pipeline (per image):
  1. Load raw image + calibration (intrinsics + extrinsics)
  2. Undistort the image using the rational-polynomial distortion model
  3. Run Depth Anything 3 (nested-giant-large) -> metric depth (meters)
  4. Back-project depth with calibrated K -> point cloud in camera frame
  5. Transform point cloud into the CENTER camera frame using extrinsics
  6. Concatenate all three clouds, optionally voxel-downsample, save as .ply

Every stage is timed; timings are written to <out_dir>/timings.json.

EXTRINSICS CONVENTION
---------------------
This script assumes the JSON gives R, t that maps a point FROM the reference
(center) frame TO the camera frame -- i.e. OpenCV's stereoCalibrate output:
        P_cam = R @ P_center + t
Hence camera -> center is:
        P_center = R.T @ (P_cam - t)
If your file follows the opposite (cam-to-world) convention, pass
`--inverse-extrinsics`.

MULTI-VIEW MODE
---------------
By default each image is processed independently (mono metric). DA3 also
supports joint multi-view inference, which produces scale-consistent depth
across views. Enable with `--multi-view` if you trust the calibration enough
to ignore the extrinsics from the model's own pose head.

Usage:
    python fuse_multiview_depth.py
    python fuse_multiview_depth.py --fp16 --voxel-size 0.01 --depth-max 12.0
"""

import argparse
import json
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

# ---------- Defaults ---------------------------------------------------------
SNAPSHOT_DIR = "/home/jetson/Projects/Image Capture/snapshots/20260522_093514_634"
CALIB_DIR    = "/home/jetson/Projects/Calibration/results"

DEFAULTS = {
    "images": {
        "center": f"{SNAPSHOT_DIR}/center.png",
        "left":   f"{SNAPSHOT_DIR}/left.png",
        "right":  f"{SNAPSHOT_DIR}/right.png",
    },
    "intrinsics": {
        "center": f"{CALIB_DIR}/intrinsics_center.json",
        "left":   f"{CALIB_DIR}/intrinsics_left.json",
        "right":  f"{CALIB_DIR}/intrinsics_right.json",
    },
    "extrinsics": f"{CALIB_DIR}/extrinsics.json",
    "output_dir": "./fused_output",
    "model":      "depth-anything/da3nested-giant-large",
    "depth_min":  0.3,
    "depth_max":  15.0,
    "voxel_size": 0.01,
}

CAMERAS = ("center", "left", "right")


# ---------- Timing -----------------------------------------------------------
class Stopwatch:
    """Logs each stage's wall-clock duration with absolute timestamps."""
    def __init__(self):
        self.records = []
        self.t_global = time.perf_counter()

    @contextmanager
    def stage(self, name):
        t_start = time.perf_counter()
        ts_start = time.time()
        print(f"[{time.strftime('%H:%M:%S', time.localtime(ts_start))}] >>> {name}")
        try:
            yield
        finally:
            dt = time.perf_counter() - t_start
            ts_end = time.time()
            self.records.append({
                "stage": name,
                "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(ts_start)),
                "end_iso":   time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(ts_end)),
                "duration_s": round(dt, 4),
            })
            print(f"[{time.strftime('%H:%M:%S', time.localtime(ts_end))}] "
                  f"<<< {name}  ({dt:.3f}s)")

    def save(self, path):
        total = time.perf_counter() - self.t_global
        with open(path, "w") as f:
            json.dump({"total_s": round(total, 4),
                       "stages": self.records}, f, indent=2)
        print(f"\nTotal: {total:.2f}s  |  timings -> {path}")


# ---------- Calibration ------------------------------------------------------
def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1)
    w, h = int(d["image_size"][0]), int(d["image_size"][1])
    return K, dist, (w, h)


def load_extrinsics(path, invert=False):
    """Returns {cam_name: 4x4 transform mapping cam_frame -> center_frame}."""
    with open(path) as f:
        ext = json.load(f)
    T_cam_to_center = {}
    for cam, d in ext.items():
        R = np.array(d["R"], dtype=np.float64)
        t = np.array(d["t"], dtype=np.float64).reshape(3)
        T = np.eye(4)
        if invert:
            # JSON gives camera pose in world (camera-to-world)
            T[:3, :3] = R
            T[:3,  3] = t
        else:
            # JSON gives world-to-camera (OpenCV stereoCalibrate)
            # cam -> center is the inverse
            T[:3, :3] = R.T
            T[:3,  3] = -R.T @ t
        T_cam_to_center[cam] = T
    return T_cam_to_center


# ---------- Geometry ---------------------------------------------------------
def undistort_image(img_bgr, K, dist):
    """OpenCV picks the right distortion model from vector length (8/12/14).
       Returns the undistorted image, the same K, and a validity mask of
       pixels that came from inside the original image."""
    h, w = img_bgr.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, K, (w, h), cv2.CV_32FC1)
    und = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = (map1 >= 0) & (map1 < w - 1) & (map2 >= 0) & (map2 < h - 1)
    return und, K, valid


def detect_depth_edges(depth, rel_threshold):
    """Pixels where the local depth gradient is large relative to the depth
    value -- these are silhouette/discontinuity artifacts that produce
    'spike' or 'veil' points after back-projection."""
    if rel_threshold <= 0:
        return np.zeros_like(depth, dtype=bool)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx * dx + dy * dy)
    return grad > rel_threshold * np.maximum(np.abs(depth), 1e-6)


def depth_to_pointcloud(depth, color_rgb, K, valid_mask, depth_min, depth_max,
                        conf=None, conf_threshold=0.0, edge_rel_threshold=0.0):
    """Back-project depth -> Nx3 metric points in the camera frame.
       Optional filtering: confidence threshold + depth-edge rejection."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth.astype(np.float32)

    m = np.isfinite(Z) & (Z > depth_min) & (Z < depth_max) & valid_mask
    if conf is not None and conf_threshold > 0:
        m &= conf >= conf_threshold
    if edge_rel_threshold > 0:
        m &= ~detect_depth_edges(Z, edge_rel_threshold)
    u, v, Z = u[m], v[m], Z[m]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts  = np.stack([X, Y, Z], axis=1).astype(np.float32)
    cols = (color_rgb[m].astype(np.float32) / 255.0)
    return pts, cols


def transform_points(pts, T):
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=pts.dtype)])
    return (pts_h @ T.T)[:, :3].astype(np.float32)


def camera_frustum_lineset(T_cam_to_center, K, image_size, scale=0.15, color=(1, 0, 0)):
    """Build an Open3D LineSet for a single camera's frustum, expressed in the
    center-camera frame. Useful for sanity-checking extrinsics visually."""
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # 4 image corners back-projected at depth `scale`
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_cam = np.zeros((4, 3))
    corners_cam[:, 0] = (corners_px[:, 0] - cx) * scale / fx
    corners_cam[:, 1] = (corners_px[:, 1] - cy) * scale / fy
    corners_cam[:, 2] = scale
    apex_cam = np.zeros((1, 3))
    pts_cam = np.vstack([apex_cam, corners_cam])  # 5 points: apex + 4 corners
    pts_center = transform_points(pts_cam.astype(np.float32), T_cam_to_center)
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],   # apex -> corners
             [1, 2], [2, 3], [3, 4], [4, 1]]   # image plane rectangle
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_center.astype(np.float64)),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def icp_refine(target_pts, source_pts, threshold=0.05, voxel=0.02):
    """Point-to-plane ICP, returning a 4x4 transform that maps source -> target.
       Both inputs are Nx3 numpy arrays already in the same (center) frame, so
       we start from identity and ICP only corrects residual misalignment."""
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))
    if voxel > 0:
        tgt = tgt.voxel_down_sample(voxel)
        src = src.voxel_down_sample(voxel)
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 3, 0.05),
                                             max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=60, relative_fitness=1e-7, relative_rmse=1e-7))
    return (np.asarray(result.transformation, dtype=np.float32),
            float(result.fitness), float(result.inlier_rmse))


# ---------- Scale-shift depth alignment via multi-view triangulation ----------
#
# Idea: the MDE depths are smooth and well-shaped but each view has its own
# (small) systematic scale/shift error. Multi-view geometry, on the other hand,
# is *metrically true* by construction. Feature-match between pairs of views,
# triangulate using the calibrated projection matrices, and you get a sparse
# set of "ground-truth" depth samples per camera. A robust LS fit
#     z_true ≈ s * z_mde + t
# per camera then bends the MDE depths into metric agreement. This is the key
# step that takes layered fusion down from cm-scale to mm-scale.

def detect_sift_features(img_rgb, max_features=5000):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    kps, descs = sift.detectAndCompute(gray, None)
    if descs is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    return pts, descs


def match_features_lowe(desc_a, desc_b, ratio=0.75):
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.array([], int), np.array([], int)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    ia, ib = [], []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            ia.append(m[0].queryIdx)
            ib.append(m[0].trainIdx)
    return np.asarray(ia, int), np.asarray(ib, int)


def triangulate(pts_a, pts_b, P_a, P_b):
    """DLT triangulation. pts_*: (N,2), P_*: (3,4). Returns (N,3) in P_a's world."""
    X = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
    return (X[:3] / X[3:]).T


def robust_scale_shift(z_mde, z_true, n_iter=4):
    """IRLS for z_true = s * z_mde + t with soft (Geman-McClure-ish) weights."""
    z_mde = np.asarray(z_mde, np.float64)
    z_true = np.asarray(z_true, np.float64)
    w = np.ones_like(z_mde)
    for _ in range(n_iter):
        A = np.stack([z_mde * w, w], axis=1)
        b = z_true * w
        (s, t), *_ = np.linalg.lstsq(A, b, rcond=None)
        r = np.abs(s * z_mde + t - z_true)
        sigma = np.median(r) + 1e-6
        w = sigma / (sigma + r)        # bounded in (0, 1]
    return float(s), float(t), w


def align_depths_via_triangulation(images_rgb, depths, K_und, T_cam_to_center,
                                   cameras, depth_max,
                                   max_features=5000,
                                   reproj_threshold_px=2.0):
    """Compute per-camera (s, t) corrections by triangulating SIFT matches.
       Returns {cam: (s, t)} and a dict of per-camera diagnostic stats."""
    # World-to-camera and projection matrices in the center (world) frame
    T_w2c = {}
    P = {}
    for cam in cameras:
        T_w2c[cam] = np.linalg.inv(T_cam_to_center[cam])
        Rt = T_w2c[cam][:3, :]
        P[cam] = K_und[cam] @ Rt

    # SIFT features per image
    feats = {}
    for cam in cameras:
        pts, descs = detect_sift_features(images_rgb[cam], max_features)
        feats[cam] = (pts, descs)
        print(f"  {cam}: {len(pts)} SIFT features")

    # All distinct unordered pairs
    pairs = [(a, b) for i, a in enumerate(cameras) for b in cameras[i + 1:]]
    per_cam = {c: {"zm": [], "zt": []} for c in cameras}

    for ca, cb in pairs:
        pts_a, desc_a = feats[ca]
        pts_b, desc_b = feats[cb]
        ia, ib = match_features_lowe(desc_a, desc_b)
        if len(ia) < 20:
            print(f"  {ca}<->{cb}: only {len(ia)} matches, skipping")
            continue
        pa, pb = pts_a[ia], pts_b[ib]

        # Triangulate in the world (=center) frame
        X_world = triangulate(pa, pb, P[ca], P[cb])

        # Reproject to each camera and reject geometric outliers
        def reproject(X, cam):
            Xc = (T_w2c[cam][:3, :3] @ X.T + T_w2c[cam][:3, 3:]).T
            z = Xc[:, 2]
            uv = (K_und[cam] @ Xc.T).T
            uv = uv[:, :2] / np.maximum(uv[:, 2:], 1e-9)
            return uv, z

        uv_a_rep, z_true_a = reproject(X_world, ca)
        uv_b_rep, z_true_b = reproject(X_world, cb)
        rep_err = np.maximum(
            np.linalg.norm(uv_a_rep - pa, axis=1),
            np.linalg.norm(uv_b_rep - pb, axis=1))

        ok = ((rep_err < reproj_threshold_px)
              & (z_true_a > 0) & (z_true_a < depth_max)
              & (z_true_b > 0) & (z_true_b < depth_max))

        # Sample MDE depths at the feature pixels (nearest)
        def sample_depth(d, uv):
            u = np.clip(uv[:, 0].astype(int), 0, d.shape[1] - 1)
            v = np.clip(uv[:, 1].astype(int), 0, d.shape[0] - 1)
            return d[v, u]

        z_mde_a = sample_depth(depths[ca], pa)
        z_mde_b = sample_depth(depths[cb], pb)
        ok &= (z_mde_a > 0) & (z_mde_b > 0) \
              & np.isfinite(z_mde_a) & np.isfinite(z_mde_b)

        per_cam[ca]["zm"].extend(z_mde_a[ok].tolist())
        per_cam[ca]["zt"].extend(z_true_a[ok].tolist())
        per_cam[cb]["zm"].extend(z_mde_b[ok].tolist())
        per_cam[cb]["zt"].extend(z_true_b[ok].tolist())
        print(f"  {ca}<->{cb}: {len(ia)} matches -> {int(ok.sum())} inliers "
              f"(reproj < {reproj_threshold_px}px)")

    corrections, diag = {}, {}
    for cam in cameras:
        zm, zt = per_cam[cam]["zm"], per_cam[cam]["zt"]
        if len(zm) < 30:
            print(f"  {cam}: too few inliers ({len(zm)}), no correction applied")
            corrections[cam] = (1.0, 0.0)
            diag[cam] = {"n": len(zm), "s": 1.0, "t": 0.0,
                         "rmse_before_mm": None, "rmse_after_mm": None}
            continue
        s, t, _ = robust_scale_shift(zm, zt)
        zm_arr = np.asarray(zm); zt_arr = np.asarray(zt)
        rmse_before = float(np.sqrt(np.mean((zm_arr - zt_arr) ** 2)))
        rmse_after  = float(np.sqrt(np.mean((s * zm_arr + t - zt_arr) ** 2)))
        corrections[cam] = (s, t)
        diag[cam] = {"n": len(zm), "s": s, "t": t,
                     "rmse_before_mm": rmse_before * 1000,
                     "rmse_after_mm":  rmse_after  * 1000}
        print(f"  {cam}: s={s:.4f}  t={t*1000:+.1f}mm  "
              f"(n={len(zm)})  RMSE {rmse_before*1000:.1f}mm -> "
              f"{rmse_after*1000:.1f}mm")
    return corrections, diag


def tsdf_fuse(depths, colors_rgb, K_und, T_cam_to_center, cameras,
              voxel_size=0.005, sdf_trunc=0.04, depth_trunc=10.0):
    """Volumetric TSDF fusion. Returns an Open3D PointCloud (extracted from
       the volume) in the center frame."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for cam in cameras:
        d = depths[cam]
        H, W = d.shape
        K = K_und[cam]
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        T_w2c = np.linalg.inv(T_cam_to_center[cam])

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(colors_rgb[cam]))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(d.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, T_w2c)
    return vol.extract_point_cloud()


# ---------- Main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--depth-min",  type=float, default=DEFAULTS["depth_min"])
    parser.add_argument("--depth-max",  type=float, default=DEFAULTS["depth_max"])
    parser.add_argument("--voxel-size", type=float, default=DEFAULTS["voxel_size"],
                        help="Voxel downsample size (m). 0 to disable.")
    parser.add_argument("--inverse-extrinsics", action="store_true",
                        help="Flip extrinsics convention to cam-to-world.")
    parser.add_argument("--model",   default=DEFAULTS["model"])
    parser.add_argument("--fp16",    action="store_true",
                        help="Run DA3 in half precision (recommended on Jetson).")
    parser.add_argument("--multi-view", action="store_true",
                        help="Run DA3 on all 3 views jointly (scale-consistent).")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                        help="Drop pixels with DA3 confidence below this "
                             "(0 disables; 0.3 is a sane default).")
    parser.add_argument("--edge-rel-threshold", type=float, default=0.05,
                        help="Drop pixels where |grad(depth)| / depth exceeds "
                             "this (kills silhouette spikes; 0 disables).")
    parser.add_argument("--icp", action="store_true",
                        help="Run point-to-plane ICP to refine left/right "
                             "alignment to center after back-projection.")
    parser.add_argument("--icp-threshold", type=float, default=0.05,
                        help="ICP correspondence distance threshold in meters.")
    parser.add_argument("--align-depths", action="store_true",
                        help="Estimate per-camera scale+shift via SIFT "
                             "feature triangulation and apply to MDE depths "
                             "BEFORE back-projection. Strongly recommended.")
    parser.add_argument("--tsdf", action="store_true",
                        help="Volumetric TSDF fusion instead of point "
                             "concatenation. Produces cleaner surfaces.")
    parser.add_argument("--tsdf-voxel", type=float, default=0.005,
                        help="TSDF voxel size in meters (default 5 mm).")
    parser.add_argument("--tsdf-trunc", type=float, default=0.04,
                        help="TSDF truncation distance in meters "
                             "(default 4x voxel).")
    parser.add_argument("--save-per-camera", action="store_true",
                        help="Also save unfused per-camera clouds.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sw = Stopwatch()

    # ---- Calibration ----
    intrinsics = {}
    sizes      = {}
    with sw.stage("Load calibration JSONs"):
        for cam, p in DEFAULTS["intrinsics"].items():
            K, dist, size = load_intrinsics(p)
            intrinsics[cam] = (K, dist)
            sizes[cam]      = size
            print(f"  {cam:6s}  size={size}  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}"
                  f"  dist_len={len(dist)}")
        T_cam_to_center = load_extrinsics(
            DEFAULTS["extrinsics"], invert=args.inverse_extrinsics)
        for cam, T in T_cam_to_center.items():
            print(f"  T_{cam}->center translation = {T[:3,3]}")

    # ---- Model ----
    with sw.stage(f"Load model: {args.model}"):
        from depth_anything_3.api import DepthAnything3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained(args.model).to(device).eval()
        # NOTE: do NOT call `model.half()`. The DA3 depth head's LayerNorm
        # mixes fp16 inputs with fp32 weights and raises
        # "expected scalar type Float but found Half". Use torch.autocast
        # instead: weights stay fp32, matmul/conv run in fp16, LayerNorm
        # is auto-promoted -- almost all the speed, none of the bug.
        use_amp = bool(args.fp16) and device.type == "cuda"
        amp_ctx = (lambda: torch.autocast(device_type="cuda",
                                          dtype=torch.float16)) \
                  if use_amp else nullcontext
        print(f"  device={device}  autocast_fp16={use_amp}")

    # ---- Load + undistort all images first (needed for multi-view too) ----
    und_imgs_rgb  = {}
    valid_masks   = {}
    K_und         = {}
    for cam in CAMERAS:
        with sw.stage(f"[{cam}] Load image"):
            img_bgr = cv2.imread(DEFAULTS["images"][cam], cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(DEFAULTS["images"][cam])
            print(f"  shape={img_bgr.shape}")

        with sw.stage(f"[{cam}] Undistort"):
            K, dist = intrinsics[cam]
            und_bgr, K_u, valid = undistort_image(img_bgr, K, dist)
            und_imgs_rgb[cam] = cv2.cvtColor(und_bgr, cv2.COLOR_BGR2RGB)
            valid_masks[cam]  = valid
            K_und[cam]        = K_u

    # ---- Depth inference ----
    depths, confs = {}, {}
    if args.multi_view:
        with sw.stage("DA3 inference (multi-view, pose-conditioned)"):
            pils = [Image.fromarray(und_imgs_rgb[c]) for c in CAMERAS]

            # ---- Build pose-conditioning tensors for DA3 ----
            #   extrinsics: (N, 4, 4) world-to-camera, with world == center
            #   intrinsics: (N, 3, 3) at the (undistorted) input resolution
            N = len(CAMERAS)
            ext_np = np.zeros((N, 4, 4), dtype=np.float32)
            int_np = np.zeros((N, 3, 3), dtype=np.float32)
            with open(DEFAULTS["extrinsics"]) as f:
                raw_ext = json.load(f)
            for i, cam in enumerate(CAMERAS):
                int_np[i] = K_und[cam]
                R = np.array(raw_ext[cam]["R"], dtype=np.float32)
                t = np.array(raw_ext[cam]["t"], dtype=np.float32)
                ext_np[i, :3, :3] = R
                ext_np[i, :3,  3] = t
                ext_np[i,  3,  3] = 1.0
            ext_t = ext_np                       # DA3 expects numpy arrays
            int_t = int_np                       # (it calls .copy() internally)
            print("  Pose-conditioning: passed calibrated (R, t) and K for "
                  "all 3 views; world frame = center.")

            with torch.inference_mode(), amp_ctx():
                pred = model.inference(
                    pils,
                    extrinsics=ext_t,
                    intrinsics=int_t,
                    align_to_input_ext_scale=True,
                )
            has_conf = hasattr(pred, "conf") and pred.conf is not None
            for i, cam in enumerate(CAMERAS):
                depths[cam] = np.asarray(pred.depth[i], dtype=np.float32)
                if has_conf:
                    confs[cam] = np.asarray(pred.conf[i], dtype=np.float32)
                print(f"  {cam:6s}  depth shape={depths[cam].shape}  "
                      f"min={depths[cam].min():.2f}m  "
                      f"max={depths[cam].max():.2f}m"
                      + (f"  conf range=[{confs[cam].min():.2f},"
                         f" {confs[cam].max():.2f}]" if has_conf else ""))
    else:
        for cam in CAMERAS:
            with sw.stage(f"[{cam}] DA3 inference (mono)"):
                pil = Image.fromarray(und_imgs_rgb[cam])
                with tempfile.TemporaryDirectory() as tmp, \
                     torch.inference_mode(), amp_ctx():
                    pred = model.inference([pil], export_dir=tmp,
                                           export_format="npz")
                depths[cam] = np.asarray(pred.depth[0], dtype=np.float32)
                if hasattr(pred, "conf") and pred.conf is not None:
                    confs[cam] = np.asarray(pred.conf[0], dtype=np.float32)
                print(f"  depth shape={depths[cam].shape}  "
                      f"min={depths[cam].min():.2f}m  "
                      f"max={depths[cam].max():.2f}m")

    # ---- Resize depths/confidences to image resolution (one pass for all) ----
    with sw.stage("Resize depths/confs to image resolution"):
        for cam in CAMERAS:
            H_img, W_img = und_imgs_rgb[cam].shape[:2]
            d = depths[cam]
            if d.shape != (H_img, W_img):
                d = cv2.resize(d, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            depths[cam] = d
            pct = np.percentile(d[np.isfinite(d)], [5, 50, 95])
            print(f"  {cam:6s}  depth percentiles (5/50/95): "
                  f"{pct[0]:.2f} / {pct[1]:.2f} / {pct[2]:.2f}  m")
            c = confs.get(cam)
            if c is not None and c.shape != (H_img, W_img):
                confs[cam] = cv2.resize(c, (W_img, H_img),
                                        interpolation=cv2.INTER_LINEAR)

    # ---- Optional: scale+shift alignment of MDE depths via triangulation ----
    if args.align_depths:
        with sw.stage("Align MDE depths via SIFT triangulation"):
            corrections, align_diag = align_depths_via_triangulation(
                und_imgs_rgb, depths, K_und, T_cam_to_center,
                list(CAMERAS), args.depth_max)
            for cam in CAMERAS:
                s, t = corrections[cam]
                depths[cam] = (s * depths[cam] + t).astype(np.float32)
            # Persist diagnostics for the thesis writeup
            with open(out_dir / "scale_shift_diagnostics.json", "w") as f:
                json.dump(align_diag, f, indent=2)
            print(f"  diagnostics -> {(out_dir / 'scale_shift_diagnostics.json').resolve()}")

    # ---- Back-project + transform ----
    per_cam = {}                       # cam -> (pts_in_center_frame, colors)
    cam_colors = {"center": (1, 0, 0), "left": (0, 1, 0), "right": (0, 0, 1)}
    for cam in CAMERAS:
        with sw.stage(f"[{cam}] Back-project to point cloud (camera frame)"):
            pts_cam, cols = depth_to_pointcloud(
                depths[cam], und_imgs_rgb[cam], K_und[cam], valid_masks[cam],
                args.depth_min, args.depth_max,
                conf=confs.get(cam),
                conf_threshold=args.conf_threshold,
                edge_rel_threshold=args.edge_rel_threshold)
            print(f"  points after filtering: {len(pts_cam):,} "
                  f"(conf>={args.conf_threshold}, "
                  f"edge<{args.edge_rel_threshold})")

        with sw.stage(f"[{cam}] Transform -> center frame"):
            pts_center = transform_points(pts_cam, T_cam_to_center[cam])

        per_cam[cam] = (pts_center, cols)

    # ---- Optional ICP refinement (left/right -> center) ----
    if args.icp:
        center_pts, _ = per_cam["center"]
        for cam in ("left", "right"):
            with sw.stage(f"[{cam}] ICP refinement -> center"):
                src_pts, src_cols = per_cam[cam]
                T_refine, fit, rmse = icp_refine(
                    center_pts, src_pts, threshold=args.icp_threshold)
                print(f"  fitness={fit:.4f}  inlier RMSE={rmse*1000:.2f} mm")
                refined = transform_points(src_pts, T_refine)
                per_cam[cam] = (refined, src_cols)

    # ---- Save per-camera (after any ICP refinement) ----
    if args.save_per_camera:
        with sw.stage("Save per-camera .ply + frustums"):
            for cam in CAMERAS:
                pts_center, cols = per_cam[cam]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_center)
                pcd.colors = o3d.utility.Vector3dVector(cols)
                o3d.io.write_point_cloud(
                    str(out_dir / f"cloud_{cam}.ply"), pcd)
                ls = camera_frustum_lineset(
                    T_cam_to_center[cam], K_und[cam], sizes[cam],
                    scale=0.2, color=cam_colors[cam])
                o3d.io.write_line_set(
                    str(out_dir / f"frustum_{cam}.ply"), ls)

    # Reassemble flat lists for the fusion step
    all_pts  = [per_cam[c][0] for c in CAMERAS]
    all_cols = [per_cam[c][1] for c in CAMERAS]

    # ---- Fuse ----
    if args.tsdf:
        with sw.stage(f"TSDF fusion (voxel={args.tsdf_voxel} m, "
                      f"trunc={args.tsdf_trunc} m)"):
            # TSDF needs the depth maps (possibly already corrected by
            # --align-depths); it weights and averages across views internally.
            pcd = tsdf_fuse(depths, und_imgs_rgb, K_und, T_cam_to_center,
                            list(CAMERAS),
                            voxel_size=args.tsdf_voxel,
                            sdf_trunc=args.tsdf_trunc,
                            depth_trunc=args.depth_max)
            print(f"  TSDF extracted points: {len(pcd.points):,}")
    else:
        with sw.stage("Fuse clouds (concatenate)"):
            pts_all  = np.concatenate(all_pts,  axis=0)
            cols_all = np.concatenate(all_cols, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_all)
            pcd.colors = o3d.utility.Vector3dVector(cols_all)
            print(f"  total points: {len(pts_all):,}")

        if args.voxel_size > 0:
            with sw.stage(f"Voxel downsample ({args.voxel_size} m)"):
                pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                print(f"  after downsample: {len(pcd.points):,}")

    with sw.stage("Save fused .ply"):
        out_ply = out_dir / ("fused_tsdf.ply" if args.tsdf else "fused.ply")
        o3d.io.write_point_cloud(str(out_ply), pcd)
        print(f"  -> {out_ply.resolve()}")

    sw.save(out_dir / "timings.json")


if __name__ == "__main__":
    main()