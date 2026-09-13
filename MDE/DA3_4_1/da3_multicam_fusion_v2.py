#!/usr/bin/env python3
"""
da3_multicam_fusion_v2.py
=========================

Four-camera point cloud generation and fusion using Depth Anything 3
(default: depth-anything/da3nested-giant-large-1.1).

WHAT CHANGED FROM v1
--------------------
v1 derived the depth-grid intrinsics by rescaling the calibrated K from the
full image size to the depth map size. That is wrong whenever DA3 crops.
With a mixed portrait/landscape batch DA3 resizes each view to fit
process_res, then centre-crops the whole batch to the smallest common size,
which changes the principal point but NOT the focal length. v1 therefore used
a focal 504/420 = 1.2x too short in one axis, differently per camera, which
fanned each view into a skewed sheet.

v2 instead:
  1. back-projects with the intrinsics DA3 RETURNS (exact, whatever
     preprocessing happened internally),
  2. produces points in the rotated ("virtual") camera frame and rotates them
     into the true camera frame with Rz(k*90 deg) before applying extrinsics,
     rather than de-rotating the depth map,
  3. reconstructs DA3's resize-and-crop transform from the returned intrinsics
     so colour sampling lands on the right pixels,
  4. offers --pad-square, which pads the rotated images to square at full
     resolution so the batch is uniform and DA3 never crops, recovering the
     field of view the centre-crop discarded. Padded regions are masked out
     geometrically.

Derivation of the point rotation, for the record. np.rot90(img, k=1) maps a
scene point at image (x, y) to (y, W-1-x). In normalised coordinates that is
(u', v') = (v, -u), so the ray direction transforms as d' = Rz(-90 deg) d,
hence X_rot = Rz(-k*90) X_cam and X_cam = Rz(+k*90) X_rot.

MODES
-----
  prior    calibrated intrinsics AND extrinsics passed to DA3, then
           back-projected and placed with the same calibration.
  noprior  images only, DA3 solves the multi-view problem itself, then fused
           externally with the calibrated extrinsics after a global metric
           scale is recovered from camera-centre alignment.

Note that in noprior mode DA3 also predicts its own intrinsics, and these are
typically ~16% short of truth on this rig. --noprior-intrinsics controls
whether back-projection uses the model's estimate (self-consistent with its
depth) or the calibrated values (consistent with reality). Both are worth
running; the FOV discrepancy is reported in the diagnostics either way.

Example
-------
python da3_multicam_fusion_v2.py \
  --images-dir /home/jetson/Projects/Image_Capture_4_1/snapshots/20260804_092615_018 \
  --calib-dir  /home/jetson/Projects/Calibration_4_1/results \
  --out        ~/Projects/MDE/DA3_4_1/runs/20260804_092615_018_v2 \
  --mode both --amp --pad-square
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
# utilities
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[da3] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


class Timer:
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
        self.size = (int(w), int(h))
        self.rms = float(d.get("rms_reprojection_error_px", float("nan")))


class Extrinsic:
    """world->camera: x_cam = R @ x_world + t, world == reference camera."""

    def __init__(self, name: str, R: np.ndarray, t: np.ndarray, serial: str = "",
                 rms: float = float("nan")):
        self.name = name
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        self.serial = str(serial)
        self.rms = rms

    @property
    def C(self) -> np.ndarray:
        return -self.R.T @ self.t

    def cam_to_ref(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.t[None, :]) @ self.R

    def as_matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M


def load_extrinsics(path: Path, convention: str) -> Tuple[Dict[str, Extrinsic], Optional[str]]:
    d = json.loads(Path(path).read_text())
    ext: Dict[str, Extrinsic] = {}
    ref_name = None
    for name, entry in d.items():
        R = np.asarray(entry["R"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(entry["t"], dtype=np.float64).reshape(3)
        if convention == "c2w":
            R, t = R.T, -R.T @ t
        ext[name] = Extrinsic(name, R, t, entry.get("serial", ""),
                              float(entry.get("rms_error_px", float("nan"))))
        if str(entry.get("note", "")).lower().startswith("reference"):
            ref_name = name
    if ref_name is None:
        for name, e in ext.items():
            if np.allclose(e.R, np.eye(3), atol=1e-9) and np.allclose(e.t, 0.0, atol=1e-12):
                ref_name = name
                break
    return ext, ref_name


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def roll_deg_from_R(R: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot90_K(K: np.ndarray, W: int, H: int, k: int) -> Tuple[np.ndarray, int, int]:
    K = K.copy()
    for _ in range(int(k) % 4):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Kn = np.eye(3)
        Kn[0, 0], Kn[1, 1] = fy, fx
        Kn[0, 2], Kn[1, 2] = cy, (W - 1) - cx
        K = Kn
        W, H = H, W
    return K, W, H


def scale_K(K: np.ndarray, src_wh: Tuple[int, int], dst_wh: Tuple[int, int]) -> np.ndarray:
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    u = np.arange(W, dtype=np.float64)[None, :]
    v = np.arange(H, dtype=np.float64)[:, None]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.astype(np.float64)
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)


def umeyama_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    var_a = (Ac ** 2).sum() / n
    U, D, Vt = np.linalg.svd((Bc.T @ Ac) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_a) if var_a > 1e-18 else 1.0
    t = mu_b - s * R @ mu_a
    rmse = float(np.sqrt((((B - (s * (R @ A.T).T + t)) ** 2).sum(axis=1)).mean()))
    return s, R, t, rmse


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# cloud IO
# --------------------------------------------------------------------------- #

def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float):
    if voxel is None or voxel <= 0 or pts.shape[0] == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return pts[idx], cols[idx]


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(pts.shape[0])
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    c = np.clip(cols, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


# --------------------------------------------------------------------------- #
# DA3 adapter
# --------------------------------------------------------------------------- #

DEPTH_KEYS = ("depth", "depths", "depth_map", "metric_depth", "pred_depth")
CONF_KEYS = ("conf", "confidence", "depth_conf", "conf_map")
INTR_KEYS = ("intrinsics", "pred_intrinsics", "camera_intrinsics")
EXTR_KEYS = ("extrinsics", "pred_extrinsics", "poses", "camera_poses")


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


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load_model(model_id: str, device: str):
    from depth_anything_3.api import DepthAnything3  # type: ignore
    log(f"loading {model_id} ...")
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()
    return model


def run_da3(model, images, K_list, E_list, amp, device, extra) -> Tuple[Any, Dict[str, Any]]:
    sig = inspect.signature(model.inference)
    params = set(sig.parameters)
    kwargs: Dict[str, Any] = {}
    used = {"method": "inference", "signature": str(sig),
            "intrinsics_kwarg": None, "extrinsics_kwarg": None}
    if K_list is not None and "intrinsics" in params:
        kwargs["intrinsics"] = np.stack(K_list).astype(np.float32)
        used["intrinsics_kwarg"] = "intrinsics"
    if E_list is not None and "extrinsics" in params:
        kwargs["extrinsics"] = np.stack(E_list).astype(np.float32)
        used["extrinsics_kwarg"] = "extrinsics"
    for k, v in extra.items():
        if k in params:
            kwargs[k] = v
    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
           if (amp and device.startswith("cuda")) else _NullCtx())
    with torch.inference_mode(), ctx:
        pred = model.inference(images, **kwargs)
    return pred, used


def parse_prediction(pred: Any, n_views: int) -> Dict[str, Optional[np.ndarray]]:
    depth = _to_numpy(_get_field(pred, DEPTH_KEYS))
    conf = _to_numpy(_get_field(pred, CONF_KEYS))
    intr = _to_numpy(_get_field(pred, INTR_KEYS))
    extr = _to_numpy(_get_field(pred, EXTR_KEYS))
    if depth is None:
        raise RuntimeError("no depth field in DA3 prediction")
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[:, 0] if depth.shape[1] == 1 else depth[..., 0]
    if depth.ndim == 2:
        depth = depth[None, ...]
    if conf is not None:
        conf = np.asarray(conf)
        if conf.ndim == 4:
            conf = conf[:, 0] if conf.shape[1] == 1 else conf[..., 0]
        if conf.ndim == 2:
            conf = conf[None, ...]
    if extr is not None:
        extr = np.asarray(extr)
        if extr.ndim == 3 and extr.shape[1:] == (3, 4):
            pad = np.zeros((extr.shape[0], 1, 4))
            pad[:, 0, 3] = 1.0
            extr = np.concatenate([extr, pad], axis=1)
    if intr is not None:
        intr = np.asarray(intr)
    return {"depth": depth, "conf": conf, "intrinsics": intr, "extrinsics": extr}


# --------------------------------------------------------------------------- #
# camera preparation
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
        self.img_fed: Optional[np.ndarray] = None    # exactly what DA3 receives
        self.K_fed: Optional[np.ndarray] = None      # intrinsics of img_fed
        self.size_fed: Tuple[int, int] = (0, 0)      # (W, H)
        self.pad_x0 = 0                              # valid-region origin inside img_fed
        self.pad_y0 = 0
        self.valid_wh: Tuple[int, int] = (0, 0)      # valid region size inside img_fed


def prepare_camera(cam: CamData, undistort: bool, upright: bool,
                   manual_k: Optional[int], pad_square: bool, pad_mode: str) -> None:
    img = cv2.imread(str(cam.img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {cam.img_path}")
    H, W = img.shape[:2]
    if (W, H) != tuple(cam.intr.size):
        warn(f"{cam.name}: image {W}x{H} vs intrinsics {cam.intr.size}; rescaling K")
        cam.intr.K = scale_K(cam.intr.K, cam.intr.size, (W, H))
        cam.intr.size = (W, H)

    if undistort and np.any(np.abs(cam.intr.dist) > 1e-12):
        img = cv2.undistort(img, cam.intr.K, cam.intr.dist, None, cam.intr.K)

    cam.roll_deg = roll_deg_from_R(cam.ext.R)
    nearest = int(round(cam.roll_deg / 90.0))
    if manual_k is not None:
        k = int(manual_k) % 4
    elif upright:
        k = nearest % 4
    else:
        k = 0
    cam.rot_k = k
    cam.residual_roll_deg = cam.roll_deg - 90.0 * nearest if upright else cam.roll_deg

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rot = np.ascontiguousarray(np.rot90(rgb, k=k))
    K_rot, W_rot, H_rot = rot90_K(cam.intr.K, cam.intr.size[0], cam.intr.size[1], k)

    if pad_square and W_rot != H_rot:
        side = max(W_rot, H_rot)
        px = (side - W_rot) // 2
        py = (side - H_rot) // 2
        border = {"replicate": cv2.BORDER_REPLICATE,
                  "reflect": cv2.BORDER_REFLECT_101,
                  "zero": cv2.BORDER_CONSTANT}[pad_mode]
        rot = cv2.copyMakeBorder(rot, py, side - H_rot - py, px, side - W_rot - px,
                                 border, value=(0, 0, 0))
        K_rot = K_rot.copy()
        K_rot[0, 2] += px
        K_rot[1, 2] += py
        cam.pad_x0, cam.pad_y0 = px, py
        cam.valid_wh = (W_rot, H_rot)
        W_rot = H_rot = side
    else:
        cam.pad_x0 = cam.pad_y0 = 0
        cam.valid_wh = (W_rot, H_rot)

    cam.img_fed = np.ascontiguousarray(rot)
    cam.K_fed = K_rot
    cam.size_fed = (W_rot, H_rot)


# --------------------------------------------------------------------------- #
# cloud construction
# --------------------------------------------------------------------------- #

def recover_transform(K_fed: np.ndarray, K_out: np.ndarray, size_fed: Tuple[int, int],
                      out_hw: Tuple[int, int]) -> Dict[str, float]:
    """
    Recover the resize-and-crop DA3 applied, from the intrinsics it returned.
    Focal ratio gives the resize factor; the principal-point discrepancy gives
    the crop origin. Works whether or not a crop occurred.
    """
    W_f, H_f = size_fed
    h_o, w_o = out_hw
    sx = float(K_out[0, 0] / K_fed[0, 0])
    sy = float(K_out[1, 1] / K_fed[1, 1])
    w_r = max(w_o, int(round(W_f * sx)))
    h_r = max(h_o, int(round(H_f * sy)))
    x0 = int(round(float(K_fed[0, 2]) * sx - float(K_out[0, 2])))
    y0 = int(round(float(K_fed[1, 2]) * sy - float(K_out[1, 2])))
    x0 = int(np.clip(x0, 0, max(0, w_r - w_o)))
    y0 = int(np.clip(y0, 0, max(0, h_r - h_o)))
    return {"sx": sx, "sy": sy, "w_r": w_r, "h_r": h_r, "x0": x0, "y0": y0}


def sample_colors(cam: CamData, tf: Dict[str, float], out_hw: Tuple[int, int]) -> np.ndarray:
    h_o, w_o = out_hw
    resized = cv2.resize(cam.img_fed, (int(tf["w_r"]), int(tf["h_r"])),
                         interpolation=cv2.INTER_AREA)
    x0, y0 = int(tf["x0"]), int(tf["y0"])
    return resized[y0:y0 + h_o, x0:x0 + w_o]


def padding_mask(cam: CamData, tf: Dict[str, float], out_hw: Tuple[int, int]) -> np.ndarray:
    """True where a depth pixel corresponds to real image content, not padding."""
    h_o, w_o = out_hw
    if cam.pad_x0 == 0 and cam.pad_y0 == 0 and cam.valid_wh == cam.size_fed:
        return np.ones((h_o, w_o), dtype=bool)
    u = np.arange(w_o)[None, :] + tf["x0"]
    v = np.arange(h_o)[:, None] + tf["y0"]
    x_fed = u / tf["sx"]
    y_fed = v / tf["sy"]
    vw, vh = cam.valid_wh
    return ((x_fed >= cam.pad_x0) & (x_fed < cam.pad_x0 + vw) &
            (y_fed >= cam.pad_y0) & (y_fed < cam.pad_y0 + vh))


def build_cloud(cam: CamData, depth: np.ndarray, conf: Optional[np.ndarray],
                K_out: Optional[np.ndarray], scale: float, args
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    h, w = depth.shape[-2:]
    if K_out is None:
        warn(f"{cam.name}: DA3 returned no intrinsics; falling back to derived rescale "
             "(assumes pure anisotropic resize, unsafe if DA3 cropped)")
        K_use = scale_K(cam.K_fed, cam.size_fed, (w, h))
        tf = {"sx": w / cam.size_fed[0], "sy": h / cam.size_fed[1],
              "w_r": w, "h_r": h, "x0": 0, "y0": 0}
    else:
        K_use = np.asarray(K_out, dtype=np.float64)
        tf = recover_transform(cam.K_fed, K_use, cam.size_fed, (h, w))

    d = depth.astype(np.float64) * float(scale)
    pts_rot = backproject(d, K_use).reshape(-1, 3)

    # rotated ("virtual") camera frame -> true camera frame
    Rk = Rz(90.0 * cam.rot_k)
    pts_cam = pts_rot @ Rk.T

    cols = sample_colors(cam, tf, (h, w)).reshape(-1, 3)

    valid = np.isfinite(pts_cam).all(axis=1)
    z = d.reshape(-1)
    valid &= (z > args.min_depth) & (z < args.max_depth)
    valid &= padding_mask(cam, tf, (h, w)).reshape(-1)
    if conf is not None and args.conf_percentile > 0.0:
        cf = conf.reshape(-1).astype(np.float64)
        thr = np.percentile(cf[np.isfinite(cf)], args.conf_percentile)
        valid &= cf >= thr

    pts_ref = cam.ext.cam_to_ref(pts_cam[valid])
    stats = {
        "camera": cam.name,
        "depth_shape": [int(h), int(w)],
        "rot_k": int(cam.rot_k),
        "fed_size": list(cam.size_fed),
        "K_used_fx": float(K_use[0, 0]),
        "K_used_fy": float(K_use[1, 1]),
        "K_used_cx": float(K_use[0, 2]),
        "K_used_cy": float(K_use[1, 2]),
        "intrinsics_source": "da3_returned" if K_out is not None else "derived_fallback",
        "recovered_transform": tf,
        "fov_ratio_vs_calibrated": float(K_use[0, 0] / (cam.K_fed[0, 0] * tf["sx"])),
        "median_depth_m": float(np.median(z[np.isfinite(z) & (z > 0)])) if np.any(z > 0) else None,
        "n_points_valid": int(pts_ref.shape[0]),
        "n_points_total": int(valid.size),
    }
    return pts_ref, cols[valid], stats


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

def relative_model_poses(extr: np.ndarray, ref_idx: int):
    R_ref, t_ref = extr[ref_idx][:3, :3], extr[ref_idx][:3, 3]
    out = []
    for M in extr:
        R_rel = M[:3, :3] @ R_ref.T
        out.append((R_rel, M[:3, 3] - R_rel @ t_ref))
    return out


def solve_global_scale(cams, parsed, ref_idx):
    info: Dict[str, Any] = {"method": None}
    extr = parsed.get("extrinsics")
    if extr is None or len(extr) < 2:
        return 1.0, {"method": "unavailable"}
    rel = relative_model_poses(np.asarray(extr, dtype=np.float64), ref_idx)
    C_model = np.stack([-R.T @ t for (R, t) in rel])
    C_calib = np.stack([c.ext.C for c in cams])
    s, R_a, t_a, rmse = umeyama_similarity(C_model, C_calib)
    info.update({"method": "umeyama_camera_centres", "scale": float(s),
                 "alignment_rmse_m": float(rmse),
                 "model_centres": C_model, "calib_centres": C_calib})
    info["per_camera_pose_agreement"] = [{
        "camera": cam.name,
        "rotation_error_deg": float(rotation_angle_deg(R_rel @ cam.ext.R.T)),
        "centre_error_m": float(np.linalg.norm(
            (s * (R_a @ (-R_rel.T @ t_rel)) + t_a) - cam.ext.C)),
        "calib_baseline_m": float(np.linalg.norm(cam.ext.C)),
        "model_baseline_scaled_m": float(np.linalg.norm(s * (-R_rel.T @ t_rel))),
    } for cam, (R_rel, t_rel) in zip(cams, rel)]
    return float(s), info


def run_mode(mode: str, cams: List[CamData], model, args, ref_idx: int, out_root: Path):
    device = args.device
    timer = Timer(device)
    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    give_intr = True if mode == "prior" else bool(args.noprior_give_intrinsics)
    give_extr = (mode == "prior")

    images = [c.img_fed for c in cams]
    K_list = [c.K_fed.astype(np.float32) for c in cams] if give_intr else None
    E_list = [c.ext.as_matrix().astype(np.float32) for c in cams] if give_extr else None

    extra = {}
    if args.process_res:
        extra["process_res"] = int(args.process_res)
    if args.export_dir_kwarg:
        extra["export_dir"] = str(out_dir / "da3_export")

    log(f"[{mode}] DA3 on {len(images)} views "
        f"(intrinsics={'yes' if give_intr else 'no'}, extrinsics={'yes' if give_extr else 'no'})")
    with timer("inference_ms"):
        pred, call_info = run_da3(model, images, K_list, E_list, args.amp, device, extra)
    with timer("parse_ms"):
        parsed = parse_prediction(pred, len(cams))

    diagnostics: Dict[str, Any] = {
        "mode": mode, "model": args.model, "call": call_info,
        "reference_camera": cams[ref_idx].name,
        "cameras": [c.name for c in cams],
        "gave_intrinsics": give_intr, "gave_extrinsics": give_extr,
        "pad_square": bool(args.pad_square),
    }

    scale = 1.0
    if mode == "noprior":
        if args.anchor != "none":
            with timer("scale_solve_ms"):
                scale, sinfo = solve_global_scale(cams, parsed, ref_idx)
            if sinfo["method"] == "unavailable":
                warn("no model extrinsics; using --noprior-scale")
                scale = float(args.noprior_scale)
                sinfo = {"method": "manual", "scale": scale}
        else:
            scale = float(args.noprior_scale)
            sinfo = {"method": "manual", "scale": scale}
        diagnostics["scale_recovery"] = json_safe(sinfo)
    else:
        _, agree = solve_global_scale(cams, parsed, ref_idx)
        agree["note"] = ("tautological in prior mode: DA3 echoes the supplied extrinsics, "
                         "so near-zero error confirms conditioning was accepted, nothing more")
        diagnostics["pose_agreement_report"] = json_safe(agree)
    diagnostics["applied_depth_scale"] = float(scale)

    # which intrinsics to back-project with
    intr_model = parsed.get("intrinsics")
    use_model_intr = True
    if mode == "noprior" and args.noprior_intrinsics == "calibrated":
        use_model_intr = False

    depth, conf = parsed["depth"], parsed["conf"]
    all_pts, all_cols, stats = [], [], []

    with timer("backproject_ms"):
        for i, cam in enumerate(cams):
            h, w = np.asarray(depth[i]).shape[-2:]
            if use_model_intr and intr_model is not None:
                K_out = intr_model[i] if intr_model.ndim == 3 else intr_model
            elif intr_model is not None:
                # calibrated, mapped onto DA3's grid via the recovered transform
                tf = recover_transform(cam.K_fed,
                                       intr_model[i] if intr_model.ndim == 3 else intr_model,
                                       cam.size_fed, (h, w))
                K_out = cam.K_fed.copy()
                K_out[0, 0] *= tf["sx"]; K_out[0, 2] = K_out[0, 2] * tf["sx"] - tf["x0"]
                K_out[1, 1] *= tf["sy"]; K_out[1, 2] = K_out[1, 2] * tf["sy"] - tf["y0"]
            else:
                K_out = None
            d_i = np.asarray(depth[i], dtype=np.float64)
            c_i = np.asarray(conf[i], dtype=np.float64) if conf is not None else None
            pts, cols, st = build_cloud(cam, d_i, c_i, K_out, scale, args)
            stats.append(st)
            if args.save_depth:
                np.save(out_dir / f"depth_{cam.name}.npy", d_i * scale)
                if K_out is not None:
                    np.save(out_dir / f"K_{cam.name}.npy", np.asarray(K_out))
            if args.save_per_camera:
                p_d, c_d = voxel_downsample(pts, cols, args.voxel)
                write_ply(out_dir / f"cloud_{cam.name}.ply", p_d, c_d)
            all_pts.append(pts)
            all_cols.append(cols)

    diagnostics["per_camera"] = json_safe(stats)

    with timer("fuse_ms"):
        pts = np.concatenate(all_pts, 0) if all_pts else np.zeros((0, 3))
        cols = np.concatenate(all_cols, 0) if all_cols else np.zeros((0, 3))
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
        "bbox_min": pts.min(0).tolist() if pts.size else None,
        "bbox_max": pts.max(0).tolist() if pts.size else None,
    }
    diagnostics["timings_ms"] = {k: round(v, 3) for k, v in timer.stages.items()}
    diagnostics["timings_ms"]["total_ms"] = round(sum(timer.stages.values()), 3)
    diagnostics["memory"] = memory_snapshot(device)
    (out_dir / "diagnostics.json").write_text(json.dumps(json_safe(diagnostics), indent=2))

    log(f"[{mode}] fused {pts.shape[0]} points -> {out_dir / 'fused.ply'}  "
        f"({timer.stages.get('inference_ms', 0):.0f} ms inference)")
    return diagnostics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_kv(values):
    out = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_argparser():
    p = argparse.ArgumentParser(
        description="Four-camera DA3 depth and fusion, with and without extrinsic priors (v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cameras", default="left,center,right,top")
    p.add_argument("--images-dir", type=Path,
                   default=Path("/home/jetson/Projects/Image_Capture_4_1/snapshots/20260804_092615_018"))
    p.add_argument("--image", action="append", metavar="NAME=PATH")
    p.add_argument("--image-ext", default="png")
    p.add_argument("--calib-dir", type=Path,
                   default=Path("/home/jetson/Projects/Calibration_4_1/results"))
    p.add_argument("--intrinsics", action="append", metavar="NAME=PATH")
    p.add_argument("--extrinsics-file", type=Path, default=None)
    p.add_argument("--extrinsics-convention", choices=["w2c", "c2w"], default="w2c")
    p.add_argument("--ref", default=None)

    p.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--mode", choices=["prior", "noprior", "both"], default="both")
    p.add_argument("--process-res", type=int, default=0,
                   help="override DA3 process_res; 0 leaves the default (504)")
    p.add_argument("--noprior-give-intrinsics", action="store_true")
    p.add_argument("--noprior-intrinsics", choices=["model", "calibrated"], default="model",
                   help="back-project noprior depth with DA3's own intrinsics estimate "
                        "(self-consistent) or the calibrated ones (physically correct)")
    p.add_argument("--anchor", choices=["camera-centres", "none"], default="camera-centres")
    p.add_argument("--noprior-scale", type=float, default=1.0)
    p.add_argument("--export-dir-kwarg", action="store_true")

    p.add_argument("--undistort", dest="undistort", action="store_true", default=True)
    p.add_argument("--no-undistort", dest="undistort", action="store_false")
    p.add_argument("--upright", dest="upright", action="store_true", default=True)
    p.add_argument("--no-upright", dest="upright", action="store_false")
    p.add_argument("--rot", action="append", metavar="NAME=K")
    p.add_argument("--pad-square", action="store_true",
                   help="pad rotated images to square so DA3 never centre-crops the batch")
    p.add_argument("--pad-mode", choices=["replicate", "reflect", "zero"], default="replicate")

    p.add_argument("--min-depth", type=float, default=0.3)
    p.add_argument("--max-depth", type=float, default=12.0)
    p.add_argument("--conf-percentile", type=float, default=0.0)
    p.add_argument("--voxel", type=float, default=0.004)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--save-per-camera", dest="save_per_camera", action="store_true", default=True)
    p.add_argument("--no-save-per-camera", dest="save_per_camera", action="store_false")
    p.add_argument("--save-depth", action="store_true", default=True)
    p.add_argument("--out", type=Path, default=Path("./da3_4cam_out_v2"))
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
        intr, ext = Intrinsics(kpath), ext_map[name]
        if intr.serial and ext.serial and intr.serial != ext.serial:
            warn(f"{name}: serial mismatch, intrinsics={intr.serial} extrinsics={ext.serial}")
        cams.append(CamData(name, ipath, intr, ext))

    ref_idx = names.index(ref)
    log(f"reference camera: {ref}")

    for cam in cams:
        prepare_camera(cam, args.undistort, args.upright, rot_over.get(cam.name),
                       args.pad_square, args.pad_mode)
        log(f"{cam.name}: roll={cam.roll_deg:+7.2f} deg -> rot_k={cam.rot_k} "
            f"(residual {cam.residual_roll_deg:+6.2f}), fed {cam.size_fed[0]}x{cam.size_fed[1]}, "
            f"baseline={np.linalg.norm(cam.ext.C):.4f} m")

    fed_sizes = {c.size_fed for c in cams}
    if len(fed_sizes) > 1:
        warn(f"mixed input sizes {sorted(fed_sizes)}: DA3 will centre-crop the batch to the "
             "smallest common size, discarding field of view. Consider --pad-square.")

    out_root = Path(os.path.expanduser(str(args.out)))
    out_root.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_model(args.model, args.device)
    modes = ["prior", "noprior"] if args.mode == "both" else [args.mode]
    summary = {
        "snapshot": str(args.images_dir), "calibration": str(args.calib_dir),
        "extrinsics_file": str(ext_path), "reference": ref,
        "pad_square": bool(args.pad_square), "upright": bool(args.upright),
        "cameras": [{"name": c.name, "serial": c.intr.serial, "image": str(c.img_path),
                     "roll_deg": c.roll_deg, "rot_k": c.rot_k,
                     "residual_roll_deg": c.residual_roll_deg,
                     "fed_size": list(c.size_fed),
                     "baseline_m": float(np.linalg.norm(c.ext.C)),
                     "intrinsics_rms_px": c.intr.rms,
                     "extrinsics_rms_px": c.ext.rms} for c in cams],
        "runs": {},
    }
    for mode in modes:
        summary["runs"][mode] = run_mode(mode, cams, model, args, ref_idx, out_root)

    (out_root / "summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    log(f"summary -> {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())