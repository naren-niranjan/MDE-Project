#!/usr/bin/env python3
"""
fuse_da3.py  —  Fused metric point cloud from 3 cameras using Depth Anything 3
                (model: depth-anything/da3nested-giant-large-1.1)

Pipeline
--------
1. Load the 3 images (center / left / right) plus per-camera intrinsics and the
   shared extrinsics.
2. Run ONE DA3 inference over all three views at once. Passing all views in a
   single call is what activates DA3's multi-view attention, so the three depth
   maps are mutually consistent. Passing your known intrinsics + extrinsics with
   `align_to_input_ext_scale=True` locks DA3 to YOUR poses and rescales its
   depth to YOUR metric scale (the nested-giant-large model outputs meters).
3. Back-project each view's metric depth to 3D using that view's intrinsics,
   transform into the common world frame using that view's extrinsics, colour
   each point from the image, filter by confidence, and concatenate.
4. Write a single fused, coloured, metric .ply in world coordinates.

Run from the DA3_v2 folder:
    python fuse_da3.py

Everything below CONFIG has sensible defaults matching the paths you gave.
The one thing you may need to flip is EXTRINSICS_CONVENTION (see notes there).
"""

import os
import json
import argparse
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# CONFIG — defaults match the paths you provided
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    snapshot="/home/jetson/Projects/MDE/DA3_v2/snapshots/20260707_105753_062",
    intr_center="/home/jetson/Projects/MDE/DA3_v2/results/intrinsics_center.json",
    intr_left="/home/jetson/Projects/MDE/DA3_v2/results/intrinsics_left.json",
    intr_right="/home/jetson/Projects/MDE/DA3_v2/results/intrinsics_right.json",
    extrinsics="/home/jetson/Projects/MDE/DA3_v2/results/extrinsics.json",
    out="/home/jetson/Projects/MDE/DA3_v2/results/fused_pointcloud.ply",
    model="depth-anything/da3nested-giant-large-1.1",
)

# Order in which the three cameras are stacked. Intrinsic files and the keys in
# extrinsics.json are matched to these names.
CAM_ORDER = ["center", "left", "right"]

# Are the extrinsics in extrinsics.json stored as world->camera (w2c) or
# camera->world (c2w)?  DA3 wants w2c. If your fused cloud comes out mirrored /
# cameras end up in the wrong places, flip this. There is a sanity print of the
# inter-camera baselines below to help you tell.
EXTRINSICS_CONVENTION = "w2c"          # "w2c" or "c2w"

# Confidence filtering: drop the lowest CONF_PERCENTILE% least-confident pixels
# per view (0 disables). DA3's own GLB export uses 40.0 by default.
CONF_PERCENTILE = 40.0

# Discard points beyond this range in metres (kills sky / blown-out depth).
MAX_DEPTH_M = 80.0
MIN_DEPTH_M = 0.1

# Optional final voxel downsample in metres (needs open3d; 0 disables).
VOXEL_SIZE_M = 0.0

# Hard cap on exported points (random subsample if exceeded). 0 disables.
MAX_POINTS = 0


# ----------------------------------------------------------------------------
# JSON loaders (robust to several common layouts)
# ----------------------------------------------------------------------------
def _as_matrix(x):
    """Coerce nested lists / OpenCV {data,rows,cols} into a float ndarray."""
    if isinstance(x, dict):
        if "data" in x and "rows" in x and "cols" in x:
            return np.asarray(x["data"], float).reshape(x["rows"], x["cols"])
        for k in ("data", "matrix", "value"):
            if k in x:
                return np.asarray(x[k], float)
    return np.asarray(x, float)


def load_intrinsics(path):
    """Return a 3x3 K matrix from many possible JSON shapes."""
    with open(path) as f:
        d = json.load(f)

    # raw 3x3 list
    if isinstance(d, list):
        K = _as_matrix(d)
        if K.shape == (3, 3):
            return K

    if isinstance(d, dict):
        for key in ("K", "camera_matrix", "intrinsic_matrix", "intrinsics", "intrinsic"):
            if key in d:
                K = _as_matrix(d[key])
                if K.size == 9:
                    return K.reshape(3, 3)
        # fx/fy/cx/cy style (possibly nested)
        src = d
        for key in ("intrinsics", "intrinsic", "params"):
            if key in d and isinstance(d[key], dict):
                src = d[key]
                break
        if all(k in src for k in ("fx", "fy", "cx", "cy")):
            return np.array([[src["fx"], 0, src["cx"]],
                             [0, src["fy"], src["cy"]],
                             [0, 0, 1]], float)

    raise ValueError(f"Could not parse a 3x3 intrinsic matrix from {path}. "
                     f"Top-level keys: {list(d) if isinstance(d, dict) else type(d)}")


def _pose_to_4x4(v):
    """Coerce a pose entry (4x4, 3x4, or {R,T}) into a 4x4 matrix."""
    if isinstance(v, dict):
        if "R" in v and ("T" in v or "t" in v):
            R = _as_matrix(v["R"]).reshape(3, 3)
            t = np.asarray(v.get("T", v.get("t")), float).reshape(3)
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = t
            return M
        for k in ("matrix", "extrinsic", "extrinsics", "pose", "T", "transform"):
            if k in v:
                return _pose_to_4x4(v[k])
    M = _as_matrix(v)
    if M.shape == (4, 4):
        return M
    if M.shape == (3, 4):
        out = np.eye(4)
        out[:3, :4] = M
        return out
    raise ValueError(f"Unrecognised pose shape {M.shape}")


def load_extrinsics(path, cam_order):
    """Return dict cam_name -> 4x4 world->camera (w2c) matrix."""
    with open(path) as f:
        d = json.load(f)

    # unwrap a common outer key
    if isinstance(d, dict) and "extrinsics" in d and isinstance(d["extrinsics"], dict):
        d = d["extrinsics"]

    poses = {}
    if isinstance(d, dict) and all(name in d for name in cam_order):
        for name in cam_order:
            poses[name] = _pose_to_4x4(d[name])
    elif isinstance(d, list) and len(d) == len(cam_order):
        for name, v in zip(cam_order, d):
            poses[name] = _pose_to_4x4(v)
    else:
        raise ValueError(
            f"extrinsics.json does not map cleanly to {cam_order}. "
            f"Found: {list(d) if isinstance(d, dict) else f'list len {len(d)}'}. "
            f"Edit load_extrinsics() to match your file's layout.")

    # convention -> w2c
    if EXTRINSICS_CONVENTION.lower() == "c2w":
        poses = {k: np.linalg.inv(v) for k, v in poses.items()}
    elif EXTRINSICS_CONVENTION.lower() != "w2c":
        raise ValueError("EXTRINSICS_CONVENTION must be 'w2c' or 'c2w'")
    return poses


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def scale_K(K, w_orig, h_orig, w_new, h_new):
    """Rescale intrinsics from original image size to DA3's processed size."""
    sx, sy = w_new / w_orig, h_new / h_orig
    Ks = K.copy().astype(float)
    Ks[0, 0] *= sx; Ks[0, 2] *= sx
    Ks[1, 1] *= sy; Ks[1, 2] *= sy
    return Ks


def unproject_to_world(depth, K, w2c, rgb, conf, conf_pct):
    """depth:(H,W) metres, K:(3,3) at depth res, w2c:(4,4), rgb:(H,W,3) uint8.
    Returns (points_world (M,3), colors (M,3) uint8)."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32)

    valid = np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
    if conf is not None and conf_pct > 0:
        thr = np.percentile(conf[np.isfinite(conf)], conf_pct)
        valid &= conf >= thr

    us, vs, z = us[valid], vs[valid], z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=1)                     # (M,3) camera coords

    c2w = np.linalg.inv(w2c)
    R, t = c2w[:3, :3], c2w[:3, 3]
    pts_world = pts_cam @ R.T + t                             # (M,3) world coords

    colors = rgb[valid]
    return pts_world.astype(np.float32), colors.astype(np.uint8)


# ----------------------------------------------------------------------------
# PLY writer (binary little-endian, coloured — no open3d needed)
# ----------------------------------------------------------------------------
def write_ply(path, pts, colors):
    n = len(pts)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fuse 3 cameras into a metric point cloud with DA3.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    ap.add_argument("--also-da3-glb", action="store_true",
                    help="Also write DA3's own GLB export (handy for a quick look; "
                         "note its GLB path may scene-normalise, so treat the PLY as "
                         "the authoritative metric output).")
    args = ap.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    # --- gather inputs in CAM_ORDER ---
    intr_paths = {"center": args.intr_center, "left": args.intr_left, "right": args.intr_right}
    img_paths, K_orig, orig_size = [], [], []
    for name in CAM_ORDER:
        p = os.path.join(args.snapshot, f"{name}.png")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        img_paths.append(p)
        K_orig.append(load_intrinsics(intr_paths[name]))
        with Image.open(p) as im:
            orig_size.append(im.size)  # (W, H)

    poses = load_extrinsics(args.extrinsics, CAM_ORDER)
    extr_w2c = np.stack([poses[n] for n in CAM_ORDER], axis=0)        # (N,4,4) w2c
    intr_in = np.stack(K_orig, axis=0)                               # (N,3,3)

    # --- sanity: camera baselines (metres) from the poses ---
    cam_centers = {n: (np.linalg.inv(poses[n])[:3, 3]) for n in CAM_ORDER}
    print("Camera centres (world, m):")
    for n in CAM_ORDER:
        print(f"  {n:>6}: {cam_centers[n].round(4)}")
    print("Baselines (m):")
    for a, b in (("center", "left"), ("center", "right"), ("left", "right")):
        print(f"  {a}-{b}: {np.linalg.norm(cam_centers[a] - cam_centers[b]):.4f}")
    print("  ^ if these don't match your physical rig, flip EXTRINSICS_CONVENTION.\n")

    # --- DA3 inference over all 3 views at once ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    model = DepthAnything3.from_pretrained(args.model).to(device)

    print("Running multi-view metric inference ...")
    export_dir = os.path.join(os.path.dirname(args.out), "da3_export") if args.also_da3_glb else None
    pred = model.inference(
        image=img_paths,
        extrinsics=extr_w2c,               # (N,4,4) world->camera, YOUR poses
        intrinsics=intr_in,                # (N,3,3) YOUR intrinsics
        align_to_input_ext_scale=True,     # lock to your poses + rescale depth to metric
        export_dir=export_dir,
        export_format="glb" if args.also_da3_glb else "mini_npz",
        conf_thresh_percentile=CONF_PERCENTILE,
    )

    depth = np.asarray(pred.depth)                     # (N,Hp,Wp) metres
    conf = np.asarray(pred.conf) if getattr(pred, "conf", None) is not None else [None] * len(depth)
    imgs = np.asarray(pred.processed_images)           # (N,Hp,Wp,3) uint8 RGB
    Hp, Wp = depth.shape[1], depth.shape[2]
    print(f"Depth maps: {depth.shape} (metres). Processed res {Wp}x{Hp}.")

    # --- back-project each view and fuse in the world frame ---
    all_pts, all_col = [], []
    for i, name in enumerate(CAM_ORDER):
        w_o, h_o = orig_size[i]
        K_i = scale_K(K_orig[i], w_o, h_o, Wp, Hp)     # intrinsics at processed res
        c_i = conf[i] if conf is not None else None
        pts, col = unproject_to_world(depth[i], K_i, poses[name], imgs[i], c_i, CONF_PERCENTILE)
        print(f"  {name:>6}: {len(pts):>8d} points")
        all_pts.append(pts); all_col.append(col)

    pts = np.concatenate(all_pts, 0)
    col = np.concatenate(all_col, 0)
    print(f"Fused total: {len(pts)} points")

    # --- optional voxel downsample ---
    if VOXEL_SIZE_M > 0:
        try:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pc.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)
            pc = pc.voxel_down_sample(VOXEL_SIZE_M)
            pts = np.asarray(pc.points, np.float32)
            col = (np.asarray(pc.colors) * 255).astype(np.uint8)
            print(f"After {VOXEL_SIZE_M} m voxel downsample: {len(pts)} points")
        except ImportError:
            print("open3d not installed; skipping voxel downsample.")

    # --- optional cap ---
    if MAX_POINTS and len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts, col = pts[idx], col[idx]
        print(f"Subsampled to {len(pts)} points")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_ply(args.out, pts, col)
    print(f"\nWrote fused metric point cloud -> {args.out}")
    if export_dir:
        print(f"DA3 GLB (viewing only) -> {export_dir}")


if __name__ == "__main__":
    main()