#!/usr/bin/env python3
"""
MoGe-2 multi-camera metric depth fusion.

Runs MoGe-2 on left / center / right frames, undistorts each frame with its
calibrated intrinsics, back-projects the metric z-depth with the (undistorted)
camera matrix, transforms every cloud into the reference (center) frame using
the extrinsics, and writes a single fused binary PLY.

Design choices (matching the established pipeline):
  - Each frame is undistorted (dist_coeffs are non-trivial here) before MoGe.
  - MoGe-2 is fed fov_x (degrees) derived from fx; metric points come from
    back-projecting MoGe's z-depth with the calibrated matrix.
  - Binary little-endian PLY is hand-written (Open3D unavailable on aarch64/py3.12).

Run:
  python moge2_multicam_fusion.py --root /home/jetson/Projects/MultiCamera/Moge2 \
      --save-individual
"""

import argparse
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import cv2
import torch

try:
    from moge.model.v2 import MoGeModel
except Exception:  # pragma: no cover
    from moge.model import MoGeModel


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #
try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None


@contextmanager
def step(name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        msg = f"[{name}] {dt:.3f}s"
        if _PROC is not None:
            msg += f" | RSS {_PROC.memory_info().rss / 1e9:.2f}GB"
        if torch.cuda.is_available():
            msg += (f" | CUDA {torch.cuda.memory_allocated() / 1e9:.2f}"
                    f"/{torch.cuda.max_memory_allocated() / 1e9:.2f}GB peak")
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Intrinsics: return full K (3x3), dist coeffs, and image size
# --------------------------------------------------------------------------- #
def _find_K(obj):
    if isinstance(obj, dict):
        low = {k.lower(): k for k in obj}
        for mk in ("camera_matrix", "k", "intrinsic_matrix", "intrinsics", "matrix"):
            if mk in low:
                a = np.asarray(obj[low[mk]], dtype=float)
                if a.size == 9:
                    return a.reshape(3, 3)
        if all(k in low for k in ("fx", "fy", "cx", "cy")):
            K = np.eye(3)
            K[0, 0] = obj[low["fx"]]; K[1, 1] = obj[low["fy"]]
            K[0, 2] = obj[low["cx"]]; K[1, 2] = obj[low["cy"]]
            return K
        for v in obj.values():
            r = _find_K(v)
            if r is not None:
                return r
    return None


def load_camera(path):
    with open(path) as f:
        d = json.load(f)
    K = _find_K(d)
    if K is None:
        raise ValueError(f"Could not parse camera matrix from {path}")
    dist = None
    for dk in ("dist_coeffs", "distortion_coefficients", "dist", "D"):
        if dk in d:
            dist = np.asarray(d[dk], dtype=float).reshape(-1)
            break
    size = d.get("image_size")  # [W, H] if present
    return K, dist, size


# --------------------------------------------------------------------------- #
# Extrinsics: 4x4 camera->center, robust to R+t dicts or matrices
# --------------------------------------------------------------------------- #
def _to_4x4(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        a = np.asarray(obj, dtype=float)
        if a.shape == (4, 4):
            return a
        if a.shape == (16,):
            return a.reshape(4, 4)
        if a.shape == (3, 4):
            T = np.eye(4); T[:3, :] = a; return T
        return None
    if isinstance(obj, dict):
        low = {k.lower(): k for k in obj}
        R = t = None
        for rk in ("r", "rotation", "rot", "rmat"):
            if rk in low:
                R = np.asarray(obj[low[rk]], dtype=float); break
        for tk in ("t", "translation", "tvec", "trans", "position"):
            if tk in low:
                t = np.asarray(obj[low[tk]], dtype=float).reshape(-1); break
        if R is not None:
            if R.shape == (9,):
                R = R.reshape(3, 3)
            if R.shape == (3, 3):
                T = np.eye(4); T[:3, :3] = R
                if t is not None and t.size >= 3:
                    T[:3, 3] = t[:3]
                return T
        for mk in ("matrix", "transform", "pose", "extrinsic", "extrinsics"):
            if mk in low:
                r = _to_4x4(obj[low[mk]])
                if r is not None:
                    return r
    return None


def _find_transform_for(d, cam):
    cam = cam.lower()
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() == cam:
                T = _to_4x4(v)
                if T is not None:
                    return T
        for k, v in d.items():
            if cam in k.lower():
                T = _to_4x4(v)
                if T is not None:
                    return T
        for v in d.values():
            if isinstance(v, (dict, list)):
                T = _find_transform_for(v, cam)
                if T is not None:
                    return T
    elif isinstance(d, list):
        for v in d:
            T = _find_transform_for(v, cam)
            if T is not None:
                return T
    return None


def load_extrinsics(path, cam_names, direction):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for cam in cam_names:
        T = _find_transform_for(d, cam)
        if T is None:
            print(f"  WARNING: no extrinsic for '{cam}', using identity")
            T = np.eye(4)
        elif direction == "world2cam":
            T = np.linalg.inv(T)
        out[cam] = T
    return out


# --------------------------------------------------------------------------- #
# Geometry / IO
# --------------------------------------------------------------------------- #
def backproject(depth, mask, fx, fy, cx, cy):
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    valid = mask & np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (us[valid] - cx) / fx * z
    y = (vs[valid] - cy) / fy * z
    return np.stack([x, y, z], axis=1).astype(np.float32), valid


def voxel_downsample(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def write_ply(path, xyz, rgb):
    n = xyz.shape[0]
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts = np.empty(n, dtype=dt)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jetson/Projects/MultiCamera/Moge2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--ref", default="center")
    ap.add_argument("--extrinsics-direction", choices=["cam2world", "world2cam"],
                    default="cam2world")
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--save-individual", action="store_true")
    args = ap.parse_args()

    root = args.root
    cams = ["left", "center", "right"]
    out_path = args.out or os.path.join(root, "fused.ply")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ext_path = os.path.join(root, "calibration", "extrinsics.json")
    print("Loading extrinsics...")
    T_cw = load_extrinsics(ext_path, cams, args.extrinsics_direction)
    ref_inv = np.linalg.inv(T_cw[args.ref])

    with step("load_model"):
        model = MoGeModel.from_pretrained(args.model).to(device).eval()

    all_xyz, all_rgb = [], []

    for cam in cams:
        img_path = os.path.join(root, "snapshots", f"{cam}.png")
        intr_path = os.path.join(root, "calibration", f"intrinsics_{cam}.json")

        K, dist, _ = load_camera(intr_path)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        H0, W0 = bgr.shape[:2]

        # undistort -> pinhole model with newK holds on the result
        if (not args.no_undistort) and dist is not None and np.any(np.abs(dist) > 1e-8):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W0, H0), alpha=0)
            bgr = cv2.undistort(bgr, K, dist, None, newK)
            x, y, w, h = roi
            if w > 0 and h > 0:
                bgr = bgr[y:y + h, x:x + w]
                newK = newK.copy()
                newK[0, 2] -= x
                newK[1, 2] -= y
            K = newK

        if args.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_AREA)
            K = K.copy() * args.scale
            K[2, 2] = 1.0

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fov_x_deg = float(np.degrees(2.0 * np.arctan(W / (2.0 * fx))))
        print(f"\n[{cam}] {W}x{H} | fx={fx:.2f} fy={fy:.2f} "
              f"cx={cx:.2f} cy={cy:.2f} | fov_x={fov_x_deg:.2f} deg")

        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
        with step(f"infer_{cam}"), torch.inference_mode():
            out = model.infer(img_t, fov_x=fov_x_deg)

        depth = out["depth"].detach().cpu().numpy().astype(np.float32)
        if "mask" in out and out["mask"] is not None:
            mask = out["mask"].detach().cpu().numpy().astype(bool)
        else:
            mask = np.isfinite(depth)

        pts_cam, valid = backproject(depth, mask, fx, fy, cx, cy)
        cols = rgb[valid]

        M = ref_inv @ T_cw[cam]            # camera -> reference frame
        pts_ref = (pts_cam @ M[:3, :3].T + M[:3, 3]).astype(np.float32)

        c = M[:3, 3]
        print(f"  valid: {pts_ref.shape[0]:,} | center in {args.ref} frame "
              f"[{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}] m | "
              f"depth median {np.median(pts_cam[:, 2]):.3f} m")

        if args.save_individual:
            ip = os.path.join(os.path.dirname(out_path), f"cloud_{cam}.ply")
            write_ply(ip, pts_ref, cols)
            print(f"  wrote {ip}")

        all_xyz.append(pts_ref)
        all_rgb.append(cols)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    print(f"\nFused (raw): {xyz.shape[0]:,} points")

    if args.voxel > 0:
        with step("voxel_downsample"):
            xyz, rgb = voxel_downsample(xyz, rgb, args.voxel)
        print(f"Fused (voxel {args.voxel} m): {xyz.shape[0]:,} points")

    with step("write_ply"):
        write_ply(out_path, xyz, rgb)
    print(f"\nWrote fused cloud -> {out_path}")


if __name__ == "__main__":
    main()