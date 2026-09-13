#!/usr/bin/env python3
"""
da3_fuse.py

Multi-camera depth fusion using Depth Anything 3 conditioned on calibrated
intrinsics and extrinsics.

What changed from the previous pipeline
---------------------------------------
1. pad_square REMOVED. Padding 2448x2048 to 2448x2448 while passing the
   unmodified cy told the model the vertical FOV was 55.1 deg when it is
   actually 47.5 deg. The model is conditioned on intrinsics, believes them,
   and rescales depth to reconcile the contradiction. This was the entire
   1.5x depth error: probe measured 3.4341 m native vs 2.2791 m padded,
   a ratio of 1.507.

2. rot_k REMOVED. The calibrated extrinsics ALREADY encode each camera's
   physical roll (left ~109 deg, right ~180 deg, top ~90 deg, read off the
   R matrices). Rotating the image upright and then passing the unmodified
   extrinsic makes the pose disagree with the pixels.

3. Manual K rescaling REMOVED. pred.intrinsics comes back already scaled to
   the depth grid and pred.extrinsics alongside it. They match pred.depth by
   construction, so no resize-factor inference, no pad offset arithmetic,
   no recovered_transform.

4. Umeyama scale recovery and applied_depth_scale REMOVED. Both existed only
   to compensate for the padding bug. pred.is_metric is 1 and pred.depth is
   already in metres.

5. Lens distortion is now undistorted before inference (--undistort, default
   on). DA3 assumes a pinhole model; k1 = -0.16 is roughly 1.5% displacement
   at the image edge, which is not negligible at this working distance.

Verification after running
--------------------------
    median_depth_m near 3.4 for every camera
    n_points_valid DIFFERING between cameras, not pinned at a constant
        (a constant means confidence filtering is doing nothing)
    plane_separation.py reporting near 770 mm

Example
-------
python da3_fuse.py \
    --snapshot-dir /home/jetson/Projects/Image_Capture_4_2/snapshots/20260805_100601_125 \
    --calib-dir    /home/jetson/Projects/Calibration_4_1/results \
    --out-dir      runs/$(date +%Y%m%d_%H%M%S)/prior \
    --mode prior
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def load_intrinsics(calib_dir, name):
    path = calib_dir / f"intrinsics_{name}.json"
    data = json.loads(path.read_text())
    K = np.asarray(data["camera_matrix"], np.float64).reshape(3, 3)
    dist = np.asarray(data.get("dist_coeffs", [0, 0, 0, 0, 0]), np.float64).ravel()
    size = tuple(int(v) for v in data["image_size"])  # (w, h)
    return {"K": K, "dist": dist, "size": size,
            "rms_px": data.get("rms_reprojection_error_px")}


def load_extrinsics(calib_dir, names, reference):
    data = json.loads((calib_dir / "extrinsics.json").read_text())
    out = {}
    for n in names:
        if n not in data:
            raise SystemExit(f"camera {n!r} absent from extrinsics.json")
        e = data[n]
        E = np.eye(4)
        E[:3, :3] = np.asarray(e["R"], np.float64).reshape(3, 3)
        E[:3, 3] = np.asarray(e["t"], np.float64).ravel()
        out[n] = E
        ref = e.get("reference")
        if ref not in (None, reference):
            print(f"[warn] {n} is referenced to {ref!r}, not {reference!r}")
    return out


def camera_centre(E):
    return -E[:3, :3].T @ E[:3, 3]


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def undistort(img, K, dist):
    """Undistort to a pinhole model. Returns the image and its new K."""
    if cv2 is None:
        raise SystemExit("opencv is required for --undistort; "
                         "pip install opencv-python or pass --no-undistort")
    h, w = img.shape[:2]
    # alpha=0 crops to the all-valid region, which keeps DA3 from seeing
    # black wedges that it would otherwise try to assign depth to.
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    out = cv2.undistort(img, K, dist, None, newK)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        out = out[y:y + rh, x:x + rw]
        newK = newK.copy()
        newK[0, 2] -= x
        newK[1, 2] -= y
    return out, newK


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(model_id, device):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained(model_id)
    if hasattr(m, "to"):
        m = m.to(device)
    return m


def as_numpy(v):
    if torch is not None and isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


# --------------------------------------------------------------------------
# back-projection
# --------------------------------------------------------------------------

def backproject(depth, K, E, colors=None, valid=None):
    """
    depth : (H, W)      metres
    K     : (3, 3)      matching the depth grid
    E     : (3, 4) or (4, 4)   world-to-camera, OpenCV convention
    Returns (N, 3) world points and (N, 3) uint8 colours.
    """
    h, w = depth.shape
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)

    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us.astype(np.float64) - cx) / fx * z
    y = (vs.astype(np.float64) - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=1)

    R = E[:3, :3]
    t = E[:3, 3]
    pts_world = (pts_cam - t) @ R  # R^T @ (X - t), vectorised

    cols = None
    if colors is not None:
        cols = colors[vs, us]
    return pts_world, cols


def to_o3d(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, np.float64) / 255.0)
    return pcd


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="DA3 multi-camera fusion with calibrated conditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--snapshot-dir", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--ext", default="png")
    ap.add_argument("--model",
                    default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--conf-percentile", type=float, default=40.0,
                    help="drop points below this percentile of confidence, "
                         "computed per camera; 0 disables")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--undistort", dest="undistort", action="store_true",
                    default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if o3d is None:
        raise SystemExit("open3d is required")

    names = list(args.cameras)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ---- inputs -------------------------------------------------------
    intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
    ext = load_extrinsics(args.calib_dir, names, args.reference)

    images, Ks_full = [], []
    for n in names:
        p = args.snapshot_dir / f"{n}.{args.ext}"
        if not p.exists():
            raise SystemExit(f"missing image: {p}")
        img = load_image(p)
        K = intr[n]["K"]
        if args.undistort:
            img, K = undistort(img, K, intr[n]["dist"])
        images.append(img)
        Ks_full.append(K)
        print(f"[input] {n:<8} {img.shape[1]}x{img.shape[0]}  "
              f"fx={K[0,0]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}  "
              f"rms={intr[n]['rms_px']:.3f}px")

    Es = np.stack([ext[n] for n in names])
    print()
    for n, E in zip(names, Es):
        print(f"[pose ] {n:<8} C={camera_centre(E).round(4)}  "
              f"|C|={np.linalg.norm(camera_centre(E)):.4f} m")
    print()

    # ---- inference ----------------------------------------------------
    model = load_model(args.model, args.device)

    t0 = time.perf_counter()
    if args.mode == "prior":
        pred = model.inference(
            images,
            intrinsics=np.stack(Ks_full),
            extrinsics=Es,
            align_to_input_ext_scale=True,
            process_res=args.process_res,
        )
    else:
        pred = model.inference(images, process_res=args.process_res)
    infer_ms = (time.perf_counter() - t0) * 1e3

    depth = as_numpy(pred.depth)              # (N, h, w) metres
    conf = as_numpy(pred.conf)                # (N, h, w)
    K_out = as_numpy(pred.intrinsics)         # (N, 3, 3) on the depth grid
    E_out = as_numpy(pred.extrinsics)         # (N, 3, 4) world-to-camera
    rgb = as_numpy(pred.processed_images)     # (N, h, w, 3) uint8
    is_metric = int(getattr(pred, "is_metric", 0))
    scale_factor = float(getattr(pred, "scale_factor", float("nan")))

    print(f"[pred ] depth {depth.shape} metric={is_metric} "
          f"scale_factor={scale_factor:.4f}")

    if not is_metric:
        print("[warn] pred.is_metric is 0 - depth is NOT in metres. "
              "Do not trust the output cloud.")

    # ---- back-projection ----------------------------------------------
    t0 = time.perf_counter()
    clouds, per_cam = {}, []

    for i, n in enumerate(names):
        d, c = depth[i], conf[i]
        finite = np.isfinite(d) & (d > 0)

        if args.conf_percentile > 0:
            thr = float(np.percentile(c[finite], args.conf_percentile))
            valid = finite & (c >= thr)
        else:
            thr = None
            valid = finite

        pts, cols = backproject(d, K_out[i], E_out[i], colors=rgb[i],
                                valid=valid)
        clouds[n] = to_o3d(pts, cols)

        # Sanity: does the returned K match the calibrated one, rescaled?
        h_out, w_out = d.shape
        s = w_out / images[i].shape[1]
        fx_expected = Ks_full[i][0, 0] * s

        rec = {
            "camera": n,
            "depth_shape": list(d.shape),
            "input_size": [int(images[i].shape[1]), int(images[i].shape[0])],
            "K_out_fx": float(K_out[i][0, 0]),
            "K_out_cx": float(K_out[i][0, 2]),
            "K_out_cy": float(K_out[i][1, 2]),
            "fx_expected_from_calib": float(fx_expected),
            "fx_ratio": float(K_out[i][0, 0] / fx_expected),
            "conf_threshold": thr,
            "median_depth_m": float(np.median(d[valid])),
            "depth_p05_m": float(np.percentile(d[valid], 5)),
            "depth_p95_m": float(np.percentile(d[valid], 95)),
            "n_points_valid": int(valid.sum()),
            "n_points_total": int(d.size),
        }
        per_cam.append(rec)
        print(f"[cloud] {n:<8} median={rec['median_depth_m']:.4f} m  "
              f"pts={rec['n_points_valid']:>7d}/{rec['n_points_total']}  "
              f"fx_ratio={rec['fx_ratio']:.4f}")

    backproject_ms = (time.perf_counter() - t0) * 1e3

    # ---- fuse ---------------------------------------------------------
    t0 = time.perf_counter()
    fused = o3d.geometry.PointCloud()
    for n in names:
        fused += clouds[n]
    n_before = len(fused.points)
    if args.voxel > 0:
        fused = fused.voxel_down_sample(args.voxel)
    fuse_ms = (time.perf_counter() - t0) * 1e3

    # ---- write --------------------------------------------------------
    t0 = time.perf_counter()
    for i, n in enumerate(names):
        o3d.io.write_point_cloud(str(args.out_dir / f"cloud_{n}.ply"), clouds[n])
        np.save(args.out_dir / f"depth_{n}.npy", depth[i])
        np.save(args.out_dir / f"K_{n}.npy", K_out[i])
        np.save(args.out_dir / f"E_{n}.npy", E_out[i])
    o3d.io.write_point_cloud(str(args.out_dir / "fused.ply"), fused)
    write_ms = (time.perf_counter() - t0) * 1e3

    bbox = fused.get_axis_aligned_bounding_box()
    diagnostics = {
        "mode": args.mode,
        "model": args.model,
        "snapshot_dir": str(args.snapshot_dir),
        "calib_dir": str(args.calib_dir),
        "reference_camera": args.reference,
        "cameras": names,
        "pad_square": False,
        "rot_k_applied": False,
        "undistorted": args.undistort,
        "gave_intrinsics": args.mode == "prior",
        "gave_extrinsics": args.mode == "prior",
        "is_metric": is_metric,
        "scale_factor": scale_factor,
        "conf_percentile": args.conf_percentile,
        "per_camera": per_cam,
        "fused": {
            "n_points_before_voxel": int(n_before),
            "n_points_written": int(len(fused.points)),
            "voxel_m": args.voxel,
            "bbox_min": np.asarray(bbox.min_bound).tolist(),
            "bbox_max": np.asarray(bbox.max_bound).tolist(),
        },
        "timings_ms": {
            "inference_ms": round(infer_ms, 3),
            "backproject_ms": round(backproject_ms, 3),
            "fuse_ms": round(fuse_ms, 3),
            "write_ms": round(write_ms, 3),
            "total_ms": round((time.perf_counter() - t_total) * 1e3, 3),
        },
    }
    if torch is not None and torch.cuda.is_available():
        diagnostics["memory"] = {
            "gpu_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }

    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2))

    # ---- checks -------------------------------------------------------
    print()
    meds = [r["median_depth_m"] for r in per_cam]
    counts = [r["n_points_valid"] for r in per_cam]
    ratios = [r["fx_ratio"] for r in per_cam]

    print(f"median depth spread : {min(meds):.4f} - {max(meds):.4f} m")
    if max(meds) < 3.0:
        print("  FAIL  still shallow. Check that padding is really gone.")
    else:
        print("  ok    consistent with a ~3.15 m standoff.")

    print(f"fx_ratio range      : {min(ratios):.4f} - {max(ratios):.4f}")
    if not all(0.98 <= r <= 1.02 for r in ratios):
        print("  FAIL  returned K disagrees with calibrated K rescaled. "
              "The model is not honouring the supplied intrinsics.")
    else:
        print("  ok    returned K matches calibration.")

    print(f"valid point counts  : {counts}")
    if len(set(counts)) == 1:
        print("  FAIL  identical across cameras - confidence filtering is "
              "doing nothing.")
    else:
        print("  ok    varies per camera as expected.")

    print(f"\nwritten: {args.out_dir}")
    print("next: python plane_separation.py --cloud left=... --cloud center=... "
          "--gt-separation 0.77 --gt-distance 3.15")


if __name__ == "__main__":
    main()