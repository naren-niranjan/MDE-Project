#!/usr/bin/env python3
"""
da3_fuse.py

Multi-camera depth fusion using Depth Anything 3 conditioned on calibrated
intrinsics and extrinsics. Takes four snapshot images, runs DA3 once, and
writes per-camera and fused point clouds.

Settled findings baked in
-------------------------
1. NO pad_square. Padding 2448x2048 to square while passing the unmodified cy
   told the model the vertical FOV was 55.1 deg when it is 47.5 deg. The model
   believes the intrinsics it is given and rescales depth to reconcile the
   contradiction -- this alone was a 1.507x depth error.

2. NO rot_k. The calibrated extrinsics already encode each camera's physical
   roll (left ~109 deg, right ~180 deg, top ~90 deg). Rotating the image and
   passing the unmodified extrinsic makes the pose disagree with the pixels.

3. NO manual K rescaling. pred.intrinsics and pred.extrinsics come back
   already matched to the depth grid. Use them directly.

4. NO Umeyama scale recovery. pred.is_metric is 1 and pred.depth is in metres.
   The old 1.56 factor existed only to compensate for the padding bug.

5. Lens distortion is removed before inference. DA3 assumes a pinhole model;
   k1 = -0.16 is roughly 1.5% displacement at the image edge.

Artefact filtering
------------------
Two rejections run on the depth map BEFORE back-projection, which is more
effective than cleaning the cloud afterwards:

  edge     Depth discontinuities produce "flying pixels" -- samples straddling
           an object silhouette return an interpolated depth and back-project
           into a curtain hanging off the edge. Rejected by relative depth
           gradient, then dilated one pixel to catch the neighbours.

  grazing  Surfaces seen near-tangentially are sampled by widely separated
           rays and stretch into visible combs. Rejected by the angle between
           the local surface normal and the view ray.

Note that raising --process-res gives FINER flying pixels, not fewer; the edge
filter is the fix for those, and resolution is the fix for comb spacing.

Ground truth for validation (ChArUco, 0.14-0.21 px reprojection):
    floor-to-deck separation   0.8375 m
    camera-to-deck perpendicular 3.0717 m

Example
-------
python da3_fuse.py \
    --snapshot-dir /home/jetson/Projects/Image_Capture_4_3/snapshots/20260806_121000_650 \
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




PALETTE = [(0.90, 0.30, 0.25), (0.25, 0.65, 0.90),
           (0.35, 0.75, 0.40), (0.95, 0.75, 0.20),
           (0.70, 0.45, 0.85), (0.95, 0.55, 0.75)]

GT_SEPARATION_M = 0.8375
GT_DECK_PERP_M = 3.0717


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def load_intrinsics(calib_dir, name):
    data = json.loads((calib_dir / f"intrinsics_{name}.json").read_text())
    return {
        "K": np.asarray(data["camera_matrix"], np.float64).reshape(3, 3),
        "dist": np.asarray(data.get("dist_coeffs", [0] * 5), np.float64).ravel(),
        "size": tuple(int(v) for v in data["image_size"]),
        "rms_px": data.get("rms_reprojection_error_px", float("nan")),
    }


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
        if e.get("reference") not in (None, reference):
            print(f"[warn] {n} referenced to {e.get('reference')!r}, "
                  f"not {reference!r}")
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
    if cv2 is None:
        raise SystemExit("opencv required for undistortion; "
                         "install it or pass --no-undistort")
    h, w = img.shape[:2]
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
# depth-map geometry
# --------------------------------------------------------------------------

def pixel_rays(shape, K):
    """Unit-z camera-frame directions for every pixel."""
    h, w = shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def points_camera(depth, K):
    return pixel_rays(depth.shape, K) * depth[..., None]


def edge_mask(depth, rel_thresh, dilate=1):
    """
    True where the depth map is locally smooth.
    Relative gradient because absolute gradient scales with distance.
    """
    gy, gx = np.gradient(depth.astype(np.float64))
    grad = np.hypot(gx, gy) / np.maximum(depth, 1e-6)
    smooth = grad <= rel_thresh
    if dilate > 0 and cv2 is not None:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        smooth = cv2.erode(smooth.astype(np.uint8), k).astype(bool)
    return smooth


def normals_from_points(pts_cam):
    """Surface normals via cross products of neighbour differences."""
    dv = np.zeros_like(pts_cam)
    du = np.zeros_like(pts_cam)
    du[:, 1:-1] = pts_cam[:, 2:] - pts_cam[:, :-2]
    dv[1:-1, :] = pts_cam[2:, :] - pts_cam[:-2, :]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def incidence_mask(pts_cam, max_deg):
    """True where the surface faces the camera within max_deg of the ray."""
    n = normals_from_points(pts_cam)
    rays = pts_cam / np.maximum(
        np.linalg.norm(pts_cam, axis=-1, keepdims=True), 1e-12)
    cos = np.abs(np.sum(n * rays, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    ang[~np.isfinite(ang)] = 90.0
    return ang <= max_deg, ang


def to_world(pts_cam, E):
    R, t = E[:3, :3], E[:3, 3]
    return (pts_cam - t) @ R          # R^T @ (X - t)


def to_o3d(points, colors=None, rgb01=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb01 is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(rgb01, (len(points), 1)))
    elif colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, np.float64) / 255.0)
    return pcd


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(model_id, device):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained(model_id)
    return m.to(device) if hasattr(m, "to") else m


def as_numpy(v):
    if torch is not None and isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


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
    ap.add_argument("--process-res", type=int, default=504,
                    help="must be a multiple of the ViT patch size 14; "
                         "504=14x36, 1008=14x72. Higher quarters the point "
                         "spacing at ~4x the inference cost.")
    ap.add_argument("--conf-percentile", type=float, default=40.0,
                    help="drop points below this confidence percentile, "
                         "computed per camera; 0 disables")
    ap.add_argument("--edge-thresh", type=float, default=0.02,
                    help="relative depth gradient above which a pixel is "
                         "treated as a discontinuity; 0 disables")
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=85.0,
                    help="drop samples whose surface normal exceeds this "
                         "angle from the view ray; 90 disables")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--undistort", dest="undistort", action="store_true",
                    default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    ap.add_argument("--colour-by-camera", action="store_true",
                    help="also write fused_by_camera.ply with one colour per "
                         "source camera, for spotting per-camera layering")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if o3d is None:
        raise SystemExit("open3d is required")
    if args.process_res % 14:
        print(f"[warn] process_res {args.process_res} is not a multiple of 14; "
              f"DA3 will round it")

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
              f"calib_rms={intr[n]['rms_px']:.3f}px")

    Es = np.stack([ext[n] for n in names])
    print()
    for n, E in zip(names, Es):
        c = camera_centre(E)
        print(f"[pose ] {n:<8} C={c.round(4)}  |C|={np.linalg.norm(c):.4f} m")
    print()

    # ---- inference ----------------------------------------------------
    model = load_model(args.model, args.device)
    t0 = time.perf_counter()
    if args.mode == "prior":
        pred = model.inference(images, intrinsics=np.stack(Ks_full),
                               extrinsics=Es, align_to_input_ext_scale=True,
                               process_res=args.process_res)
    else:
        pred = model.inference(images, process_res=args.process_res)
    infer_ms = (time.perf_counter() - t0) * 1e3

    depth = as_numpy(pred.depth)
    conf = as_numpy(pred.conf)
    K_out = as_numpy(pred.intrinsics)
    E_out = as_numpy(pred.extrinsics)
    rgb = as_numpy(pred.processed_images)
    is_metric = int(getattr(pred, "is_metric", 0))
    scale_factor = float(getattr(pred, "scale_factor", float("nan")))

    print(f"[pred ] depth {depth.shape}  metric={is_metric}  "
          f"scale_factor={scale_factor:.4f}")
    if not is_metric:
        print("[warn] pred.is_metric is 0 -- depth is NOT in metres.")

    # ---- back-projection ----------------------------------------------
    t0 = time.perf_counter()
    clouds, tinted, per_cam = {}, [], []

    for i, n in enumerate(names):
        d = depth[i].astype(np.float64)
        c = conf[i]
        K_i = K_out[i]

        valid = np.isfinite(d) & (d > 0)
        n_start = int(valid.sum())
        drops = {}

        if args.conf_percentile > 0:
            thr = float(np.percentile(c[valid], args.conf_percentile))
            before = int(valid.sum())
            valid &= c >= thr
            drops["conf"] = before - int(valid.sum())
        else:
            thr = None

        if args.edge_thresh > 0:
            before = int(valid.sum())
            valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
            drops["edge"] = before - int(valid.sum())

        pts_cam = points_camera(d, K_i)

        inc_med = None
        if args.max_incidence < 90:
            ok, ang = incidence_mask(pts_cam, args.max_incidence)
            inc_med = float(np.median(ang[valid])) if valid.any() else None
            before = int(valid.sum())
            valid &= ok
            drops["grazing"] = before - int(valid.sum())

        pts_world = to_world(pts_cam[valid], E_out[i])
        cols = rgb[i][valid]
        clouds[n] = to_o3d(pts_world, colors=cols)
        if args.colour_by_camera:
            tinted.append(to_o3d(pts_world, rgb01=PALETTE[i % len(PALETTE)]))

        s = d.shape[1] / images[i].shape[1]
        fx_expected = Ks_full[i][0, 0] * s
        dv = d[valid]

        rec = {
            "camera": n,
            "depth_shape": list(d.shape),
            "input_size": [int(images[i].shape[1]), int(images[i].shape[0])],
            "K_out_fx": float(K_i[0, 0]),
            "fx_expected_from_calib": float(fx_expected),
            "fx_ratio": float(K_i[0, 0] / fx_expected),
            "conf_threshold": thr,
            "incidence_median_deg": inc_med,
            "median_depth_m": float(np.median(dv)),
            "depth_p05_m": float(np.percentile(dv, 5)),
            "depth_p95_m": float(np.percentile(dv, 95)),
            "n_points_initial": n_start,
            "n_points_valid": int(valid.sum()),
            "n_points_total": int(d.size),
            "dropped": drops,
        }
        per_cam.append(rec)
        dstr = " ".join(f"{k}={v}" for k, v in drops.items())
        print(f"[cloud] {n:<8} median={rec['median_depth_m']:.4f} m  "
              f"pts={rec['n_points_valid']:>7d}  fx_ratio={rec['fx_ratio']:.4f}"
              + (f"  dropped[{dstr}]" if dstr else ""))

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
        np.save(args.out_dir / f"conf_{n}.npy", conf[i])
        np.save(args.out_dir / f"K_{n}.npy", K_out[i])
        np.save(args.out_dir / f"E_{n}.npy", E_out[i])
    o3d.io.write_point_cloud(str(args.out_dir / "fused.ply"), fused)
    if tinted:
        merged = o3d.geometry.PointCloud()
        for p in tinted:
            merged += p
        o3d.io.write_point_cloud(str(args.out_dir / "fused_by_camera.ply"),
                                 merged)
    write_ms = (time.perf_counter() - t0) * 1e3

    bbox = fused.get_axis_aligned_bounding_box()
    diagnostics = {
        "mode": args.mode, "model": args.model,
        "snapshot_dir": str(args.snapshot_dir),
        "calib_dir": str(args.calib_dir),
        "reference_camera": args.reference, "cameras": names,
        "pad_square": False, "rot_k_applied": False,
        "undistorted": args.undistort,
        "process_res": args.process_res,
        "gave_intrinsics": args.mode == "prior",
        "gave_extrinsics": args.mode == "prior",
        "is_metric": is_metric, "scale_factor": scale_factor,
        "filters": {"conf_percentile": args.conf_percentile,
                    "edge_thresh": args.edge_thresh,
                    "edge_dilate": args.edge_dilate,
                    "max_incidence_deg": args.max_incidence},
        "ground_truth": {"separation_m": GT_SEPARATION_M,
                         "deck_perp_m": GT_DECK_PERP_M,
                         "source": "charuco, 0.14-0.21 px reprojection"},
        "per_camera": per_cam,
        "fused": {"n_points_before_voxel": int(n_before),
                  "n_points_written": int(len(fused.points)),
                  "voxel_m": args.voxel,
                  "bbox_min": np.asarray(bbox.min_bound).tolist(),
                  "bbox_max": np.asarray(bbox.max_bound).tolist()},
        "timings_ms": {"inference_ms": round(infer_ms, 3),
                       "backproject_ms": round(backproject_ms, 3),
                       "fuse_ms": round(fuse_ms, 3),
                       "write_ms": round(write_ms, 3),
                       "total_ms": round((time.perf_counter() - t_total) * 1e3, 3)},
    }
    if torch is not None and torch.cuda.is_available():
        diagnostics["memory"] = {
            "gpu_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
    (args.out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    # ---- self-checks --------------------------------------------------
    print()
    meds = np.array([r["median_depth_m"] for r in per_cam])
    counts = [r["n_points_valid"] for r in per_cam]
    ratios = np.array([r["fx_ratio"] for r in per_cam])

    print(f"median depth        : {meds.min():.4f} - {meds.max():.4f} m")
    if meds.max() < 3.0:
        print("  FAIL  still shallow -- check that padding is really gone.")
    else:
        print(f"  ok    consistent with a {GT_DECK_PERP_M:.2f} m deck standoff.")

    print(f"fx_ratio            : {ratios.min():.4f} - {ratios.max():.4f}")
    if not np.all((ratios > 0.98) & (ratios < 1.02)):
        print("  FAIL  returned K disagrees with calibrated K rescaled.")
    else:
        print("  ok    returned K matches calibration.")

    print(f"valid points        : {counts}")
    if len(set(counts)) == 1:
        print("  FAIL  identical across cameras -- filtering is doing nothing.")
    else:
        print("  ok    varies per camera as expected.")

    print(f"\nwritten: {args.out_dir}")
    print(f"next:\n  python plane_separation.py \\\n"
          f"    --cloud left={args.out_dir}/cloud_left.ply "
          f"--cloud center={args.out_dir}/cloud_center.ply \\\n"
          f"    --cloud right={args.out_dir}/cloud_right.ply "
          f"--cloud top={args.out_dir}/cloud_top.ply \\\n"
          f"    --gt-separation {GT_SEPARATION_M} "
          f"--gt-distance {GT_DECK_PERP_M}")


if __name__ == "__main__":
    main()