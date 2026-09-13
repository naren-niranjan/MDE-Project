#!/usr/bin/env python3
"""
da3_multicam_fusion.py
======================

Four-camera point cloud generation and fusion using Depth Anything 3
(default: depth-anything/da3nested-giant-large-1.1).

Two comparison modes are produced from the same snapshot so that the results
can be inspected side by side:

  MODE "prior"    The calibrated intrinsics AND the calibrated extrinsics are
                  handed to DA3 as known camera parameters. The returned
                  per-view depth maps are back-projected with the calibrated
                  intrinsics and placed into the reference frame with the
                  calibrated extrinsics.

  MODE "noprior"  DA3 receives the images only (optionally the intrinsics, see
                  --noprior-give-intrinsics). It solves the multi-view problem
                  itself. The returned per-view depth maps are then fused
                  EXTERNALLY using the calibrated extrinsics, after a single
                  global metric scale is recovered by aligning DA3's predicted
                  camera centres to the calibrated camera centres.

Outputs, per mode, into <out>/<mode>/:
    cloud_<cam>.ply        per-camera cloud, already in the reference frame
    fused.ply              concatenation of all cameras (optionally voxelised)
    depth_<cam>.npy        raw depth map as returned by DA3 (de-rotated)
    diagnostics.json       timings, memory, scale, pose-agreement metrics

Nothing is compared automatically. The two fused.ply files are written so the
comparison can be done by hand.

Example
-------
python da3_multicam_fusion.py \
  --images-dir /home/jetson/Projects/Image_Capture_4_1/snapshots/20260804_092615_018 \
  --calib-dir  /home/jetson/Projects/Calibration_4_1/results \
  --out        ~/Projects/MDE/DA3_4cam/runs/20260804_092615_018 \
  --mode both --amp
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    print("[fatal] OpenCV (cv2) is required.", file=sys.stderr)
    raise

try:
    import torch
except ImportError:  # pragma: no cover
    print("[fatal] PyTorch is required.", file=sys.stderr)
    raise

try:
    import psutil
except ImportError:
    psutil = None


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[da3] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


class Timer:
    """Per-stage wall-clock timing with CUDA synchronisation."""

    def __init__(self, device: str):
        self.device = device
        self.stages: Dict[str, float] = {}

    def _sync(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                timer._sync()
                self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                timer._sync()
                dt = (time.perf_counter() - self_inner.t0) * 1000.0
                timer.stages[name] = timer.stages.get(name, 0.0) + dt
                return False

        return _Ctx()


def memory_snapshot(device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if psutil is not None:
        vm = psutil.virtual_memory()
        out["cpu_ram_used_gb"] = round((vm.total - vm.available) / 1e9, 3)
        out["cpu_ram_total_gb"] = round(vm.total / 1e9, 3)
    if device.startswith("cuda") and torch.cuda.is_available():
        out["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        out["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
        out["gpu_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return out


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# --------------------------------------------------------------------------- #
# calibration IO
# --------------------------------------------------------------------------- #

class Intrinsics:
    def __init__(self, path: Path):
        d = json.loads(Path(path).read_text())
        self.path = str(path)
        self.camera = d.get("camera")
        self.serial = str(d.get("serial", ""))
        self.K = np.asarray(d["camera_matrix"], dtype=np.float64)
        self.dist = np.asarray(d.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()
        w, h = d["image_size"]
        self.size = (int(w), int(h))  # (W, H)
        self.rms = float(d.get("rms_reprojection_error_px", float("nan")))

    def __repr__(self) -> str:
        return (f"Intrinsics({self.camera}, sn={self.serial}, "
                f"fx={self.K[0, 0]:.1f}, size={self.size}, rms={self.rms:.3f}px)")


class Extrinsic:
    """Stored as world->camera:  x_cam = R @ x_world + t, world == reference camera."""

    def __init__(self, name: str, R: np.ndarray, t: np.ndarray, serial: str = "",
                 rms: float = float("nan")):
        self.name = name
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        self.serial = str(serial)
        self.rms = rms

    @property
    def C(self) -> np.ndarray:
        """Camera centre expressed in the reference frame."""
        return -self.R.T @ self.t

    def cam_to_ref(self, pts: np.ndarray) -> np.ndarray:
        """pts: (N,3) in this camera's frame -> reference frame."""
        return (pts - self.t[None, :]) @ self.R

    def as_matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M


def load_extrinsics(path: Path, convention: str) -> Tuple[Dict[str, Extrinsic], str]:
    d = json.loads(Path(path).read_text())
    ext: Dict[str, Extrinsic] = {}
    ref_name = None
    for name, entry in d.items():
        R = np.asarray(entry["R"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(entry["t"], dtype=np.float64).reshape(3)
        if convention == "c2w":
            R_w2c = R.T
            t_w2c = -R.T @ t
            R, t = R_w2c, t_w2c
        ext[name] = Extrinsic(name, R, t, entry.get("serial", ""),
                              float(entry.get("rms_error_px", float("nan"))))
        if entry.get("reference") is not None and name == entry.get("reference"):
            ref_name = name
        if str(entry.get("note", "")).lower().startswith("reference"):
            ref_name = name
    if ref_name is None:
        # fall back: the camera whose R is identity and t is zero
        for name, e in ext.items():
            if np.allclose(e.R, np.eye(3), atol=1e-9) and np.allclose(e.t, 0.0, atol=1e-12):
                ref_name = name
                break
    return ext, ref_name


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def roll_deg_from_R(R: np.ndarray) -> float:
    """In-plane (optical axis) rotation of this camera relative to the reference."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def rot90_K(K: np.ndarray, W: int, H: int, k: int) -> Tuple[np.ndarray, int, int]:
    """
    Adjust an intrinsic matrix for k counter-clockwise 90 degree rotations of the
    image (np.rot90 semantics). Returns (K_rot, W_rot, H_rot).

    For one CCW rotation:  x' = y,  y' = W-1-x   =>  fx'=fy, fy'=fx,
                           cx'=cy,  cy'=W-1-cx,  new size (H, W).
    """
    K = K.copy()
    k = int(k) % 4
    for _ in range(k):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        K_new = np.eye(3)
        K_new[0, 0] = fy
        K_new[1, 1] = fx
        K_new[0, 2] = cy
        K_new[1, 2] = (W - 1) - cx
        K = K_new
        W, H = H, W
    return K, W, H


def scale_K(K: np.ndarray, src_wh: Tuple[int, int], dst_wh: Tuple[int, int]) -> np.ndarray:
    """Rescale intrinsics from src (W,H) to dst (W,H). Independent sx, sy."""
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H,W) depth + K -> (H,W,3) points in the camera frame."""
    H, W = depth.shape
    u = np.arange(W, dtype=np.float64)[None, :]
    v = np.arange(H, dtype=np.float64)[:, None]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth.astype(np.float64)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x, y, z], axis=-1)


def umeyama_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Least-squares similarity that maps A -> B:  B ~= s * R @ A + t.
    A, B are (N,3). Returns (s, R, t, rmse).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    var_a = (Ac ** 2).sum() / n
    Sigma = (Bc.T @ Ac) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_a) if var_a > 1e-18 else 1.0
    t = mu_b - s * R @ mu_a
    resid = B - (s * (R @ A.T).T + t)
    rmse = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return s, R, t, rmse


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# point cloud IO / filtering
# --------------------------------------------------------------------------- #

def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    if voxel is None or voxel <= 0 or pts.shape[0] == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return pts[idx], cols[idx]


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    """Binary little-endian PLY with uint8 colour. No Open3D dependency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(pts.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    c = np.clip(cols, 0, 255).astype(np.uint8)
    arr["red"] = c[:, 0]
    arr["green"] = c[:, 1]
    arr["blue"] = c[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


# --------------------------------------------------------------------------- #
# DA3 adapter
# --------------------------------------------------------------------------- #

INFER_METHOD_CANDIDATES = ("inference", "infer", "predict", "__call__")
INTRINSIC_KWARGS = ("intrinsics", "known_intrinsics", "gt_intrinsics",
                    "camera_intrinsics", "Ks", "K")
EXTRINSIC_KWARGS = ("extrinsics", "known_extrinsics", "gt_extrinsics",
                    "camera_poses", "poses", "c2ws", "w2cs")
DEPTH_KEYS = ("depth", "depths", "depth_map", "metric_depth", "pred_depth")
CONF_KEYS = ("conf", "confidence", "depth_conf", "conf_map")
INTR_KEYS = ("intrinsics", "pred_intrinsics", "camera_intrinsics")
EXTR_KEYS = ("extrinsics", "pred_extrinsics", "poses", "camera_poses")
POINT_KEYS = ("points", "pointmap", "point_map", "world_points", "xyz", "pts3d")


def _get_field(obj: Any, names: Sequence[str]) -> Optional[Any]:
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            v = getattr(obj, n)
            if not callable(v):
                return v
    return None


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    if isinstance(x, (list, tuple)):
        try:
            return np.stack([_to_numpy(e) for e in x], axis=0)
        except Exception:
            return None
    return None


def load_model(model_id: str, device: str):
    try:
        from depth_anything_3.api import DepthAnything3  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import depth_anything_3.api.DepthAnything3. "
            "Activate the DA3 venv, or set PYTHONPATH to the DA3 checkout."
        ) from exc
    log(f"loading {model_id} ...")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def resolve_infer(model):
    for name in INFER_METHOD_CANDIDATES:
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                sig = None
            return name, fn, sig
    raise RuntimeError("No usable inference method found on the DA3 model object.")


def run_da3(model, images: List[np.ndarray], K_list: Optional[List[np.ndarray]],
            E_list: Optional[List[np.ndarray]], amp: bool, device: str,
            extra_kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Call DA3 with whatever camera-prior keyword arguments the installed build
    actually accepts. The resolved call is reported in the diagnostics so that
    it is unambiguous which priors were used.
    """
    name, fn, sig = resolve_infer(model)
    params = set(sig.parameters.keys()) if sig is not None else set()
    kwargs: Dict[str, Any] = {}
    used = {"method": name, "signature": str(sig) if sig else "unavailable",
            "intrinsics_kwarg": None, "extrinsics_kwarg": None}

    if K_list is not None:
        for cand in INTRINSIC_KWARGS:
            if cand in params:
                kwargs[cand] = np.stack(K_list).astype(np.float32)
                used["intrinsics_kwarg"] = cand
                break
        if used["intrinsics_kwarg"] is None:
            warn("model.inference() accepts no intrinsics argument; intrinsics not passed.")

    if E_list is not None:
        for cand in EXTRINSIC_KWARGS:
            if cand in params:
                kwargs[cand] = np.stack(E_list).astype(np.float32)
                used["extrinsics_kwarg"] = cand
                break
        if used["extrinsics_kwarg"] is None:
            warn("model.inference() accepts no extrinsics argument; extrinsics not passed. "
                 "The 'prior' mode will then differ from 'noprior' only in post-processing.")

    for k, v in extra_kwargs.items():
        if sig is None or k in params:
            kwargs[k] = v

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if (amp and device.startswith("cuda")) else _NullCtx())

    with torch.inference_mode():
        with autocast_ctx:
            try:
                pred = fn(images, **kwargs)
            except TypeError:
                # some builds expect PIL images or a stacked array
                try:
                    from PIL import Image
                    pil = [Image.fromarray(im) for im in images]
                    pred = fn(pil, **kwargs)
                except Exception:
                    pred = fn(np.stack(images), **kwargs)
    return pred, used


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def parse_prediction(pred: Any, n_views: int) -> Dict[str, Optional[np.ndarray]]:
    """Normalise the DA3 output into depth / conf / intrinsics / extrinsics / points."""
    depth = _to_numpy(_get_field(pred, DEPTH_KEYS))
    conf = _to_numpy(_get_field(pred, CONF_KEYS))
    intr = _to_numpy(_get_field(pred, INTR_KEYS))
    extr = _to_numpy(_get_field(pred, EXTR_KEYS))
    pts = _to_numpy(_get_field(pred, POINT_KEYS))

    if depth is None:
        raise RuntimeError(
            "Could not locate a depth field in the DA3 prediction. "
            f"Available attributes: {sorted(a for a in dir(pred) if not a.startswith('_'))[:40]}"
        )
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 2:
        depth = depth[None, ...]
    if depth.shape[0] != n_views:
        warn(f"depth batch dimension is {depth.shape[0]}, expected {n_views}.")

    if conf is not None:
        conf = np.asarray(conf)
        if conf.ndim == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.ndim == 2:
            conf = conf[None, ...]

    if extr is not None:
        extr = np.asarray(extr)
        if extr.ndim == 3 and extr.shape[1:] == (3, 4):
            pad = np.zeros((extr.shape[0], 1, 4))
            pad[:, 0, 3] = 1.0
            extr = np.concatenate([extr, pad], axis=1)

    return {"depth": depth, "conf": conf, "intrinsics": intr,
            "extrinsics": extr, "points": pts}


# --------------------------------------------------------------------------- #
# per-camera preparation
# --------------------------------------------------------------------------- #

class CamData:
    def __init__(self, name: str, img_path: Path, intr: Intrinsics, ext: Extrinsic):
        self.name = name
        self.img_path = img_path
        self.intr = intr
        self.ext = ext
        self.rot_k = 0
        self.roll_deg = 0.0
        self.residual_roll_deg = 0.0
        self.img_bgr: Optional[np.ndarray] = None       # undistorted, original orientation
        self.img_rot_rgb: Optional[np.ndarray] = None   # rotated, RGB, fed to DA3
        self.K_rot: Optional[np.ndarray] = None
        self.size_rot: Tuple[int, int] = (0, 0)


def prepare_camera(cam: CamData, undistort: bool, upright: bool,
                   manual_k: Optional[int]) -> None:
    img = cv2.imread(str(cam.img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {cam.img_path}")
    H, W = img.shape[:2]
    if (W, H) != tuple(cam.intr.size):
        warn(f"{cam.name}: image is {W}x{H} but intrinsics declare "
             f"{cam.intr.size[0]}x{cam.intr.size[1]}; rescaling intrinsics.")
        cam.intr.K = scale_K(cam.intr.K, cam.intr.size, (W, H))
        cam.intr.size = (W, H)

    if undistort and np.any(np.abs(cam.intr.dist) > 1e-12):
        img = cv2.undistort(img, cam.intr.K, cam.intr.dist, None, cam.intr.K)

    cam.img_bgr = img
    cam.roll_deg = roll_deg_from_R(cam.ext.R)

    if manual_k is not None:
        k = int(manual_k) % 4
    elif upright:
        k = int(round(cam.roll_deg / 90.0)) % 4
    else:
        k = 0
    cam.rot_k = k
    cam.residual_roll_deg = cam.roll_deg - 90.0 * int(round(cam.roll_deg / 90.0)) if upright else cam.roll_deg

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cam.img_rot_rgb = np.ascontiguousarray(np.rot90(rgb, k=k))
    K_rot, W_rot, H_rot = rot90_K(cam.intr.K, cam.intr.size[0], cam.intr.size[1], k)
    cam.K_rot = K_rot
    cam.size_rot = (W_rot, H_rot)


# --------------------------------------------------------------------------- #
# cloud construction
# --------------------------------------------------------------------------- #

def build_cloud(cam: CamData, depth_rot: np.ndarray, conf_rot: Optional[np.ndarray],
                scale: float, args) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    depth_rot / conf_rot are in the ROTATED orientation as returned by DA3.
    They are de-rotated back to the original image orientation, so the original
    (calibrated) intrinsics and extrinsics apply unchanged.
    """
    k = cam.rot_k
    depth = np.ascontiguousarray(np.rot90(depth_rot, k=-k)).astype(np.float64) * float(scale)
    conf = None
    if conf_rot is not None:
        conf = np.ascontiguousarray(np.rot90(conf_rot, k=-k)).astype(np.float64)

    h, w = depth.shape
    # colour, resized in the rotated frame then de-rotated so the grids line up
    col_rot = cv2.resize(cam.img_rot_rgb, (depth_rot.shape[1], depth_rot.shape[0]),
                         interpolation=cv2.INTER_AREA)
    col = np.ascontiguousarray(np.rot90(col_rot, k=-k))

    K_d = scale_K(cam.intr.K, cam.intr.size, (w, h))
    pts_cam = backproject(depth, K_d).reshape(-1, 3)
    cols = col.reshape(-1, 3)

    valid = np.isfinite(pts_cam).all(axis=1)
    z = pts_cam[:, 2]
    valid &= (z > args.min_depth) & (z < args.max_depth)
    if conf is not None and args.conf_percentile > 0.0:
        cflat = conf.reshape(-1)
        thr = np.percentile(cflat[np.isfinite(cflat)], args.conf_percentile)
        valid &= cflat >= thr

    pts_cam = pts_cam[valid]
    cols = cols[valid]
    pts_ref = cam.ext.cam_to_ref(pts_cam)

    stats = {
        "depth_shape": [int(h), int(w)],
        "rot_k": int(k),
        "median_depth_m": float(np.median(z[np.isfinite(z) & (z > 0)])) if np.any(z > 0) else None,
        "n_points_valid": int(pts_ref.shape[0]),
        "n_points_total": int(valid.size),
        "K_depth_fx": float(K_d[0, 0]),
    }
    return pts_ref, cols, stats


# --------------------------------------------------------------------------- #
# mode runners
# --------------------------------------------------------------------------- #

def relative_model_poses(extr_model: np.ndarray, ref_idx: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert DA3's world->cam matrices into poses relative to the reference view."""
    out = []
    R_ref = extr_model[ref_idx][:3, :3]
    t_ref = extr_model[ref_idx][:3, 3]
    for M in extr_model:
        R_i, t_i = M[:3, :3], M[:3, 3]
        R_rel = R_i @ R_ref.T
        t_rel = t_i - R_rel @ t_ref
        out.append((R_rel, t_rel))
    return out


def solve_global_scale(cams: List[CamData], parsed: Dict[str, Any], ref_idx: int
                       ) -> Tuple[float, Dict[str, Any]]:
    """
    Recover the metric scale of a DA3 reconstruction by aligning its predicted
    camera centres to the calibrated camera centres.
    """
    info: Dict[str, Any] = {"method": None}
    extr = parsed.get("extrinsics")
    if extr is None or len(extr) < 2:
        info["method"] = "unavailable"
        return 1.0, info

    rel = relative_model_poses(np.asarray(extr, dtype=np.float64), ref_idx)
    C_model = np.stack([-R.T @ t for (R, t) in rel])
    C_calib = np.stack([c.ext.C for c in cams])

    s, R_align, t_align, rmse = umeyama_similarity(C_model, C_calib)
    info.update({
        "method": "umeyama_camera_centres",
        "scale": float(s),
        "alignment_rmse_m": float(rmse),
        "model_centres": C_model,
        "calib_centres": C_calib,
    })

    per_cam = []
    for cam, (R_rel, t_rel) in zip(cams, rel):
        R_err = rotation_angle_deg(R_rel @ cam.ext.R.T)
        C_m = s * (R_align @ (-R_rel.T @ t_rel)) + t_align
        per_cam.append({
            "camera": cam.name,
            "rotation_error_deg": float(R_err),
            "centre_error_m": float(np.linalg.norm(C_m - cam.ext.C)),
            "calib_baseline_m": float(np.linalg.norm(cam.ext.C)),
            "model_baseline_scaled_m": float(np.linalg.norm(s * (-R_rel.T @ t_rel))),
        })
    info["per_camera_pose_agreement"] = per_cam
    return float(s), info


def run_mode(mode: str, cams: List[CamData], model, args, ref_idx: int,
             out_root: Path) -> Dict[str, Any]:
    device = args.device
    timer = Timer(device)
    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    give_intr = True if mode == "prior" else bool(args.noprior_give_intrinsics)
    give_extr = (mode == "prior")

    images = [c.img_rot_rgb for c in cams]
    K_list = [c.K_rot.astype(np.float32) for c in cams] if give_intr else None
    E_list = [c.ext.as_matrix().astype(np.float32) for c in cams] if give_extr else None

    extra = {}
    if args.export_dir_kwarg:
        extra["export_dir"] = str(out_dir / "da3_export")

    log(f"[{mode}] running DA3 on {len(images)} views "
        f"(intrinsics={'yes' if give_intr else 'no'}, extrinsics={'yes' if give_extr else 'no'})")
    with timer("inference_ms"):
        pred, call_info = run_da3(model, images, K_list, E_list, args.amp, device, extra)

    with timer("parse_ms"):
        parsed = parse_prediction(pred, len(cams))

    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "model": args.model,
        "call": call_info,
        "reference_camera": cams[ref_idx].name,
        "cameras": [c.name for c in cams],
        "gave_intrinsics": give_intr,
        "gave_extrinsics": give_extr,
    }

    # ---- metric scale ----------------------------------------------------- #
    scale = 1.0
    if mode == "noprior" and args.anchor != "none":
        with timer("scale_solve_ms"):
            scale, scale_info = solve_global_scale(cams, parsed, ref_idx)
        if scale_info["method"] == "unavailable":
            warn("DA3 returned no extrinsics; cannot recover metric scale from camera "
                 "centres. Falling back to --noprior-scale.")
            scale = float(args.noprior_scale)
            scale_info = {"method": "manual", "scale": scale}
        diagnostics["scale_recovery"] = json_safe(scale_info)
    elif mode == "noprior":
        scale = float(args.noprior_scale)
        diagnostics["scale_recovery"] = {"method": "manual", "scale": scale}
    else:
        # still report how DA3's own poses compare with the calibration
        _, agree = solve_global_scale(cams, parsed, ref_idx)
        diagnostics["pose_agreement_report"] = json_safe(agree)
    diagnostics["applied_depth_scale"] = float(scale)

    # ---- clouds ----------------------------------------------------------- #
    depth = parsed["depth"]
    conf = parsed["conf"]
    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    per_cam_stats = []

    with timer("backproject_ms"):
        for i, cam in enumerate(cams):
            d_i = np.asarray(depth[i], dtype=np.float64)
            c_i = np.asarray(conf[i], dtype=np.float64) if conf is not None else None
            pts, cols, st = build_cloud(cam, d_i, c_i, scale, args)
            st["camera"] = cam.name
            per_cam_stats.append(st)
            if args.save_depth:
                np.save(out_dir / f"depth_{cam.name}.npy",
                        np.rot90(d_i, k=-cam.rot_k) * scale)
            if args.save_per_camera:
                p_d, c_d = voxel_downsample(pts, cols, args.voxel)
                write_ply(out_dir / f"cloud_{cam.name}.ply", p_d, c_d)
            all_pts.append(pts)
            all_cols.append(cols)

    diagnostics["per_camera"] = json_safe(per_cam_stats)

    with timer("fuse_ms"):
        pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
        cols = np.concatenate(all_cols, axis=0) if all_cols else np.zeros((0, 3))
        n_before = int(pts.shape[0])
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        if args.max_points > 0 and pts.shape[0] > args.max_points:
            sel = np.random.default_rng(0).choice(pts.shape[0], args.max_points, replace=False)
            pts, cols = pts[sel], cols[sel]

    with timer("write_ms"):
        write_ply(out_dir / "fused.ply", pts, cols)

    diagnostics["fused"] = {
        "n_points_before_voxel": n_before,
        "n_points_written": int(pts.shape[0]),
        "voxel_m": args.voxel,
        "bbox_min": pts.min(axis=0).tolist() if pts.size else None,
        "bbox_max": pts.max(axis=0).tolist() if pts.size else None,
    }
    diagnostics["timings_ms"] = {k: round(v, 3) for k, v in timer.stages.items()}
    diagnostics["timings_ms"]["total_ms"] = round(sum(timer.stages.values()), 3)
    diagnostics["memory"] = memory_snapshot(device)

    (out_dir / "diagnostics.json").write_text(json.dumps(json_safe(diagnostics), indent=2))
    log(f"[{mode}] fused cloud: {pts.shape[0]} points -> {out_dir / 'fused.ply'}")
    log(f"[{mode}] inference {timer.stages.get('inference_ms', 0):.1f} ms, "
        f"total {sum(timer.stages.values()):.1f} ms")
    return diagnostics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_kv(values: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Four-camera DA3 depth and fusion, with and without extrinsic priors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--cameras", default="left,center,right,top",
                   help="comma-separated camera names, must match the calibration keys")
    p.add_argument("--images-dir", type=Path,
                   default=Path("/home/jetson/Projects/Image_Capture_4_1/snapshots/20260804_092615_018"),
                   help="directory containing <name>.png for each camera")
    p.add_argument("--image", action="append", metavar="NAME=PATH",
                   help="override an individual image path")
    p.add_argument("--image-ext", default="png")

    p.add_argument("--calib-dir", type=Path,
                   default=Path("/home/jetson/Projects/Calibration_4_1/results"),
                   help="directory containing intrinsics_<name>.json and extrinsics.json")
    p.add_argument("--intrinsics", action="append", metavar="NAME=PATH",
                   help="override an individual intrinsics file")
    p.add_argument("--extrinsics-file", type=Path, default=None)
    p.add_argument("--extrinsics-convention", choices=["w2c", "c2w"], default="w2c",
                   help="w2c means x_cam = R @ x_world + t (OpenCV stereoCalibrate output)")
    p.add_argument("--ref", default=None,
                   help="reference camera; default is the one flagged in extrinsics.json")

    p.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true",
                   help="run inference under torch.autocast float16 (never model.half())")
    p.add_argument("--mode", choices=["prior", "noprior", "both"], default="both")
    p.add_argument("--noprior-give-intrinsics", action="store_true",
                   help="in noprior mode, still pass the calibrated intrinsics to DA3")
    p.add_argument("--anchor", choices=["camera-centres", "none"], default="camera-centres",
                   help="how to recover metric scale in noprior mode")
    p.add_argument("--noprior-scale", type=float, default=1.0,
                   help="fallback / manual depth scale for noprior mode")
    p.add_argument("--export-dir-kwarg", action="store_true",
                   help="pass export_dir to DA3 if the build accepts it")

    p.add_argument("--undistort", dest="undistort", action="store_true", default=True)
    p.add_argument("--no-undistort", dest="undistort", action="store_false")
    p.add_argument("--upright", dest="upright", action="store_true", default=True,
                   help="rotate each image by the nearest multiple of 90 deg to remove "
                        "the mounting roll before inference, then de-rotate the depth map")
    p.add_argument("--no-upright", dest="upright", action="store_false")
    p.add_argument("--rot", action="append", metavar="NAME=K",
                   help="manual override of the CCW 90-degree rotation count for a camera")

    p.add_argument("--min-depth", type=float, default=0.3)
    p.add_argument("--max-depth", type=float, default=12.0)
    p.add_argument("--conf-percentile", type=float, default=0.0,
                   help="drop the lowest N%% of pixels by DA3 confidence, if confidence exists")
    p.add_argument("--voxel", type=float, default=0.004,
                   help="voxel size in metres for downsampling; 0 disables")
    p.add_argument("--max-points", type=int, default=0,
                   help="random cap on fused point count; 0 disables")
    p.add_argument("--save-per-camera", dest="save_per_camera", action="store_true", default=True)
    p.add_argument("--no-save-per-camera", dest="save_per_camera", action="store_false")
    p.add_argument("--save-depth", action="store_true", default=True)

    p.add_argument("--out", type=Path, default=Path("./da3_4cam_out"))
    return p


def main() -> int:
    args = build_argparser().parse_args()
    names = [n.strip() for n in args.cameras.split(",") if n.strip()]
    img_over = parse_kv(args.image)
    intr_over = parse_kv(args.intrinsics)
    rot_over = {k: int(v) for k, v in parse_kv(args.rot).items()}

    ext_path = args.extrinsics_file or (args.calib_dir / "extrinsics.json")
    ext_map, ref_from_file = load_extrinsics(ext_path, args.extrinsics_convention)
    ref = args.ref or ref_from_file
    if ref is None:
        raise SystemExit("could not determine the reference camera; pass --ref")
    if ref not in names:
        raise SystemExit(f"reference camera '{ref}' is not in --cameras")

    cams: List[CamData] = []
    for name in names:
        ipath = Path(img_over.get(name, args.images_dir / f"{name}.{args.image_ext}"))
        kpath = Path(intr_over.get(name, args.calib_dir / f"intrinsics_{name}.json"))
        if name not in ext_map:
            raise SystemExit(f"camera '{name}' absent from {ext_path}")
        intr = Intrinsics(kpath)
        ext = ext_map[name]
        if intr.serial and ext.serial and intr.serial != ext.serial:
            warn(f"{name}: serial mismatch, intrinsics={intr.serial} extrinsics={ext.serial}")
        cams.append(CamData(name, ipath, intr, ext))

    ref_idx = names.index(ref)
    log(f"reference camera: {ref}")

    for cam in cams:
        prepare_camera(cam, args.undistort, args.upright, rot_over.get(cam.name))
        log(f"{cam.name}: fx={cam.intr.K[0,0]:.1f} size={cam.intr.size} "
            f"roll={cam.roll_deg:+.2f} deg -> rot_k={cam.rot_k} "
            f"(residual {cam.residual_roll_deg:+.2f} deg), "
            f"baseline={np.linalg.norm(cam.ext.C):.4f} m")

    out_root = Path(os.path.expanduser(str(args.out)))
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_model(args.model, args.device)
    _, _, sig = resolve_infer(model)
    log(f"inference signature: {sig}")

    modes = ["prior", "noprior"] if args.mode == "both" else [args.mode]
    summary = {
        "snapshot": str(args.images_dir),
        "calibration": str(args.calib_dir),
        "extrinsics_file": str(ext_path),
        "reference": ref,
        "cameras": [
            {"name": c.name, "serial": c.intr.serial, "image": str(c.img_path),
             "roll_deg": c.roll_deg, "rot_k": c.rot_k,
             "residual_roll_deg": c.residual_roll_deg,
             "baseline_m": float(np.linalg.norm(c.ext.C)),
             "intrinsics_rms_px": c.intr.rms,
             "extrinsics_rms_px": c.ext.rms}
            for c in cams
        ],
        "runs": {},
    }
    for mode in modes:
        summary["runs"][mode] = run_mode(mode, cams, model, args, ref_idx, out_root)

    (out_root / "summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    log(f"summary -> {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())