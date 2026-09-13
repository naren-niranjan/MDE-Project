#!/usr/bin/env python3
"""Multi-camera fusion of DA3 monocular depth into a single, refined point cloud.

Per camera:
    image -> (optional 180 deg flip for the physically rolled left camera)
          -> DA3 metric depth
          -> unproject with the calibrated intrinsics (distortion-aware)
          -> transform into the CENTER reference frame with the calibrated extrinsics

Refinement (center is the fixed reference; each side is aligned to it):
    SIFT match side<->center -> fundamental-matrix RANSAC (2D outlier reject)
      -> back-project matches to 3D -> RANSAC + Umeyama similarity (scale+R+t)
      -> optional point-to-plane ICP polish
Then statistical-outlier removal + voxel downsample -> one clean PLY.

The similarity step corrects residual pose AND monocular scale bias, but it locks
the sides to the CENTER camera's scale; it does not fix absolute metric scale
(validate that separately against the rc_viscore ground truth).

Only load_da3() / da3_infer() are tied to your DA3_v5 environment. If they do not
match your install, edit those two functions, or skip inference entirely with
--depth-dir pointing at precomputed depth arrays (see --depth-pattern).

Example:
    python merge_da3_pointclouds.py \
        --snapshot-dir /home/jetson/Projects/MDE/DA3_v5/Snapshots/20260721_121110_179 \
        --results-dir  /home/jetson/Projects/MDE/DA3_v5/results \
        --output merged.ply --save-depth --per-camera-ply
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

CAMS = ["center", "left", "right"]


# --------------------------------------------------------------------------- #
# Calibration I/O
# --------------------------------------------------------------------------- #
def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    K = np.asarray(d["camera_matrix"], dtype=np.float64)
    dist = np.asarray(d["dist_coeffs"], dtype=np.float64).reshape(-1)
    W, H = d["image_size"]  # stored as [width, height]
    return K, dist, (int(W), int(H))


def load_extrinsics(path):
    with open(path) as f:
        d = json.load(f)
    ext = {}
    for cam in CAMS:
        e = d[cam]
        R = np.asarray(e["R"], dtype=np.float64)
        t = np.asarray(e["t"], dtype=np.float64).reshape(3)
        ext[cam] = (R, t)
    return ext


# --------------------------------------------------------------------------- #
# DA3 inference  (ALIGN THESE TWO FUNCTIONS WITH YOUR DA3_v5 SETUP IF NEEDED)
# --------------------------------------------------------------------------- #
def load_da3(model_id, device):
    import torch  # noqa: F401
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    return model


def da3_infer(model, image_rgb, device):
    """Return (depth_HxW_metres float32, conf_HxW or None).

    image_rgb: HxWx3 uint8. The nested giant-large model returns metric depth
    directly (prediction.depth [N,H,W] float32, metres), so no scale conversion
    is applied here. One image is fed per call, i.e. per-camera monocular depth;
    DA3's own estimated extrinsics are ignored because we fuse with the calibrated
    rig geometry instead. Never call model.half() on Jetson; use autocast.
    """
    import torch

    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.inference([image_rgb])
        else:
            pred = model.inference([image_rgb])

    # pred may be an object with attributes or a dict; be defensive.
    def _get(obj, name):
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
        return None

    depth = _get(pred, "depth")
    if depth is None:
        raise RuntimeError(
            "DA3 prediction has no 'depth'. Edit da3_infer() to match your "
            "DA3_v5 inference, or use --depth-dir with precomputed depth."
        )

    if hasattr(depth, "detach"):
        depth = depth.detach().float().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise RuntimeError(f"Expected HxW depth, got shape {depth.shape}")

    conf = _get(pred, "conf")  # may be None
    if conf is not None and hasattr(conf, "detach"):
        conf = conf.detach().float().cpu().numpy()
    if conf is not None:
        conf = np.squeeze(np.asarray(conf, dtype=np.float32))
        if conf.shape != depth.shape:
            conf = None
    return depth, conf


def load_depth_file(depth_dir, pattern, cam):
    path = os.path.join(depth_dir, pattern.format(cam=cam))
    depth = np.load(path).astype(np.float32)
    depth = np.squeeze(depth)
    return depth, None


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def unproject(depth, K, dist, color_rgb, stride, dmin, dmax, conf, conf_thr,
              depth_is_ray=False):
    """Distortion-aware unprojection.

    depth_is_ray=False: depth is perpendicular z-depth (standard).
    depth_is_ray=True : depth is Euclidean distance along the pixel ray; the
                        perpendicular component is recovered as z = r / |ray|.
    """
    H, W = depth.shape
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)  # (h, w)

    Z = depth[vv, uu].astype(np.float32)
    col = color_rgb[vv, uu]

    mask = np.isfinite(Z) & (Z > dmin) & (Z < dmax)
    if conf is not None and conf_thr > 0.0:
        mask &= conf[vv, uu] >= conf_thr
    if not np.any(mask):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    pix = np.stack([uu[mask], vv[mask]], axis=1).astype(np.float32).reshape(-1, 1, 2)
    Zf = Z[mask]
    norm = cv2.undistortPoints(pix, K, dist).reshape(-1, 2)  # normalised (x', y')
    if depth_is_ray:
        # Zf holds ray length r; convert to perpendicular z-depth.
        ray_norm = np.sqrt(norm[:, 0] ** 2 + norm[:, 1] ** 2 + 1.0)
        Zf = Zf / ray_norm
    X = norm[:, 0] * Zf
    Y = norm[:, 1] * Zf
    pts = np.stack([X, Y, Zf], axis=1).astype(np.float32)
    cols = col[mask].reshape(-1, 3).astype(np.uint8)
    return pts, cols


def to_center(pts, R, t, invert):
    """Bring points from a camera frame into the center reference frame.

    Default convention: R, t is the camera->center pose  ->  Xc = R @ Xcam + t.
    --invert-extrinsics flips it to center->camera           ->  Xc = R^T (Xcam - t).
    Toggle if the merged cloud comes out split/mirrored.
    """
    if pts.shape[0] == 0:
        return pts
    if invert:
        return ((R.T @ (pts - t).T).T).astype(np.float32)
    return ((R @ pts.T).T + t).astype(np.float32)


def camera_center(R, t, invert):
    """Camera origin expressed in the center frame (for a sanity print)."""
    if invert:
        return -R.T @ t
    return t.copy()


# --------------------------------------------------------------------------- #
# PLY writer (binary little-endian, xyz float32 + rgb uint8)
# --------------------------------------------------------------------------- #
def write_ply(path, pts, cols):
    n = int(pts.shape[0])
    dt = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["red"], arr["green"], arr["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def maybe_voxel_downsample(pts, cols, voxel):
    if voxel <= 0.0:
        return pts, cols
    try:
        import open3d as o3d
    except Exception:
        print("  [warn] open3d unavailable; skipping voxel downsample")
        return pts, cols
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    pc = pc.voxel_down_sample(voxel)
    p = np.asarray(pc.points, np.float32)
    c = np.clip(np.asarray(pc.colors) * 255.0, 0, 255).astype(np.uint8)
    return p, c


# --------------------------------------------------------------------------- #
# Refinement: SIFT -> 2D RANSAC -> 3D correspondences -> RANSAC similarity -> ICP
# --------------------------------------------------------------------------- #
def backproject_pixels(uv, depth, K, dist, depth_is_ray):
    """uv: (N,2) float pixels -> (pts_cam (N,3) with NaN where invalid, valid (N,))."""
    H, W = depth.shape
    pts = np.full((len(uv), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(uv), dtype=bool)
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(inb)[0]
    if idx.size == 0:
        return pts, valid
    Z = depth[v[idx], u[idx]].astype(np.float64)
    good = np.isfinite(Z) & (Z > 0)
    idx = idx[good]
    Z = Z[good]
    if idx.size == 0:
        return pts, valid
    norm = cv2.undistortPoints(uv[idx].astype(np.float32).reshape(-1, 1, 2),
                               K, dist).reshape(-1, 2)
    if depth_is_ray:
        Z = Z / np.sqrt(norm[:, 0] ** 2 + norm[:, 1] ** 2 + 1.0)
    pts[idx, 0] = norm[:, 0] * Z
    pts[idx, 1] = norm[:, 1] * Z
    pts[idx, 2] = Z
    valid[idx] = True
    return pts, valid


def detect_and_match(gray_src, gray_dst, ratio, max_feats):
    """SIFT (fallback ORB) + Lowe-ratio matching. Returns matched pixels src,dst."""
    try:
        det = cv2.SIFT_create(nfeatures=max_feats)
        norm_type = cv2.NORM_L2
    except Exception:
        det = cv2.ORB_create(nfeatures=max_feats)
        norm_type = cv2.NORM_HAMMING
    ks, ds = det.detectAndCompute(gray_src, None)
    kd, dd = det.detectAndCompute(gray_dst, None)
    if ds is None or dd is None or len(ks) < 4 or len(kd) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(ds, dd, k=2)
    ps, pd = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            ps.append(ks[m.queryIdx].pt)
            pd.append(kd[m.trainIdx].pt)
    return np.asarray(ps, float).reshape(-1, 2), np.asarray(pd, float).reshape(-1, 2)


def ransac_2d_filter(ps, pd, thresh):
    """Epipolar (fundamental-matrix) RANSAC to drop mismatches."""
    if len(ps) < 8:
        return ps, pd
    try:
        F, mask = cv2.findFundamentalMat(
            ps.astype(np.float32), pd.astype(np.float32),
            cv2.FM_RANSAC, thresh, 0.999)
    except cv2.error:
        return ps, pd  # degenerate geometry; leave to the 3D RANSAC stage
    if mask is None or F is None:
        return ps, pd
    m = mask.ravel().astype(bool)
    if m.sum() < 8:
        return ps, pd
    return ps[m], pd[m]


def umeyama(src, dst, with_scale):
    """Closed-form similarity: returns s, R, t with  dst ~= s * R @ src + t."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    Sigma = (Dc.T @ Sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (Sc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def ransac_similarity(src, dst, thresh, iters, with_scale, min_inliers, seed=0):
    """RANSAC over 3-point minimal sets. Returns (s,R,t), inlier_mask, rmse."""
    n = len(src)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, bool)
    for _ in range(iters):
        sel = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[sel], dst[sel], with_scale)
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        mask = err < thresh
        if mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask.sum() < max(min_inliers, 3):
        return None
    s, R, t = umeyama(src[best_mask], dst[best_mask], with_scale)
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    best_mask = err < thresh
    rmse = float(np.sqrt((err[best_mask] ** 2).mean())) if best_mask.any() else float("nan")
    return (s, R, t), best_mask, rmse


def apply_sim(pts, sim):
    if pts.shape[0] == 0 or sim is None:
        return pts
    s, R, t = sim
    return ((s * (R @ pts.T)).T + t).astype(np.float32)


def icp_polish(src_pts, dst_pts, voxel, threshold, max_iter):
    """Point-to-plane ICP from identity init. Returns (4x4 T, fitness, rmse)."""
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(dst_pts.astype(np.float64))
    if voxel > 0:
        src = src.voxel_down_sample(voxel)
        dst = dst.voxel_down_sample(voxel)
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(threshold * 4.0, 0.05), max_nn=30))
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return np.asarray(reg.transformation), float(reg.fitness), float(reg.inlier_rmse)


def apply_rigid_T(pts, T):
    if pts.shape[0] == 0:
        return pts
    return ((T[:3, :3] @ pts.T).T + T[:3, 3]).astype(np.float32)


def clean_cloud(pts, cols, voxel, sor_nb, sor_std):
    try:
        import open3d as o3d
    except Exception:
        print("  [warn] open3d unavailable; skipping cleanup")
        return maybe_voxel_downsample(pts, cols, voxel)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    if sor_nb > 0:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=sor_nb, std_ratio=sor_std)
    p = np.asarray(pc.points, np.float32)
    c = np.clip(np.asarray(pc.colors) * 255.0, 0, 255).astype(np.uint8)
    return p, c


def solve_depth_scale(cam_data, args):
    """Recover the absolute depth scale from the calibrated baseline alone.

    The extrinsic baseline is a metric ruler. If DA3 depth is biased by a factor
    k, each camera's cloud shrinks toward its OWN optical centre while the
    baselines stay fixed, so the three clouds no longer agree: you get ghosting.
    The k that minimises cross-camera disagreement is therefore the depth scale
    that makes the rig self-consistent, recovered WITHOUT any ground truth.

    Returns (best_k, list of {k, score} probes).
    """
    from scipy.spatial import cKDTree

    stride = max(args.stride * args.scale_solve_stride, 1)

    def cloud_at(cam, k):
        d = cam_data[cam]
        pts, _ = unproject(d["depth"] * k, d["K"], d["dist"],
                           d["rgb"], stride, args.min_depth, args.max_depth,
                           d["conf"], args.conf_threshold,
                           depth_is_ray=args.depth_is_ray)
        return to_center(pts, d["R"], d["t"], args.invert_extrinsics)

    def normals_of(pts):
        """Surface normals of the reference cloud (for point-to-plane scoring)."""
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.scale_solve_normal_radius, max_nn=30))
        return np.asarray(pc.normals)

    def score(k):
        """RELATIVE point-to-plane disagreement between the sides and center.

        Two traps this avoids:
        (1) Plain nearest-neighbour distance is dominated by point spacing, so
            the residual is projected onto the local surface normal; tangential
            sampling offset cancels, normal-direction misalignment survives.
        (2) ABSOLUTE error has a false minimum at k=0: shrinking depth collapses
            every cloud toward its own optical centre, and since those centres
            are only a baseline apart, absolute distances shrink trivially. The
            score is therefore normalised by mean scene depth, which is flat
            under pure collapse and only dips at genuine agreement.
        """
        ctr = cloud_at("center", k)
        if ctr.shape[0] < 100:
            return float("inf")
        ctr64 = ctr.astype(np.float64)
        nrm = normals_of(ctr64)
        tree = cKDTree(ctr64)
        depth_scale = float(np.median(np.linalg.norm(ctr64, axis=1)))
        if not np.isfinite(depth_scale) or depth_scale < 1e-6:
            return float("inf")
        errs = []
        for cam in ("left", "right"):
            side = cloud_at(cam, k)
            if side.shape[0] < 100:
                continue
            dist, idx = tree.query(side.astype(np.float64), k=1,
                                   distance_upper_bound=args.scale_solve_cap * k)
            ok = np.isfinite(dist)
            if ok.sum() < 100:
                continue
            diff = side.astype(np.float64)[ok] - ctr64[idx[ok]]
            p2pl = np.abs(np.einsum("ij,ij->i", diff, nrm[idx[ok]]))
            errs.append(float(np.median(p2pl)) / depth_scale)
        return float(np.mean(errs)) if errs else float("inf")

    lo, hi = args.scale_solve_range
    probes = []
    ks = np.linspace(lo, hi, args.scale_solve_steps)
    for k in ks:
        sc = score(float(k))
        probes.append({"k": round(float(k), 5), "score_rel": round(sc, 6)})
        print(f"    k={k:6.4f}  relative disagreement {sc * 1000:8.3f} per-mille")
    scores = np.array([p["score_rel"] for p in probes], dtype=float)
    best_i = int(np.argmin(scores))
    best_k = probes[best_i]["k"]

    # Degeneracy check: a flat curve means the geometry cannot constrain scale
    # (e.g. a single fronto-parallel plane, where scaling just slides points
    # along the surface). Contrast is min-vs-median of the probe scores.
    finite = scores[np.isfinite(scores)]
    contrast = 0.0
    if finite.size >= 3 and np.median(finite) > 0:
        contrast = float((np.median(finite) - finite.min()) / np.median(finite))
    degenerate = contrast < args.scale_solve_min_contrast
    if degenerate:
        print(f"\n    [warn] scale curve is flat (contrast {contrast:.2f} < "
              f"{args.scale_solve_min_contrast}). The overlap is probably "
              f"plane-dominated and cannot pin down scale. Treat k as "
              f"unreliable and fall back to ground-truth alignment.")
    if best_i in (0, len(probes) - 1):
        print(f"\n    [warn] minimum sits at the edge of the search range; "
              f"widen --scale-solve-range.")

    # Golden-section refine within the neighbouring bracket.
    a = probes[max(best_i - 1, 0)]["k"]
    b = probes[min(best_i + 1, len(probes) - 1)]["k"]
    if b > a:
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = score(c), score(d)
        for _ in range(args.scale_solve_refine):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = score(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = score(d)
        best_k = float((a + b) / 2)
    return best_k, {"probes": probes, "contrast": round(contrast, 4),
                    "degenerate": bool(degenerate),
                    "best_score_rel": round(float(finite.min()), 6)
                    if finite.size else None}


def refine_side_to_center(side, center, args):
    """Estimate a corrective transform bringing a side cloud onto center.

    side/center are dicts with keys: gray, depth, K, dist, pts_c (calibrated
    center-frame points), R, t. Returns (refined_pts, info_dict).
    """
    info = {"n_matches": 0, "n_2d_inliers": 0, "n_3d_inliers": 0,
            "scale": 1.0, "sim_rmse_m": None, "icp_fitness": None,
            "icp_rmse_m": None, "status": "calibration_only"}

    ps, pd = detect_and_match(side["gray"], center["gray"],
                              args.lowe_ratio, args.max_features)
    info["n_matches"] = int(len(ps))
    if len(ps) < args.min_matches:
        return side["pts_c"], info
    ps, pd = ransac_2d_filter(ps, pd, args.epipolar_thresh)
    info["n_2d_inliers"] = int(len(ps))
    if len(ps) < args.min_matches:
        return side["pts_c"], info

    src_cam, vs = backproject_pixels(ps, side["depth"], side["K"], side["dist"],
                                     args.depth_is_ray)
    dst_cam, vd = backproject_pixels(pd, center["depth"], center["K"],
                                     center["dist"], args.depth_is_ray)
    both = vs & vd
    if both.sum() < args.min_matches:
        return side["pts_c"], info
    # Source into center frame via calibration; target already in center frame.
    src = to_center(src_cam[both], side["R"], side["t"], args.invert_extrinsics)
    dst = to_center(dst_cam[both], center["R"], center["t"], args.invert_extrinsics)

    res = ransac_similarity(src.astype(np.float64), dst.astype(np.float64),
                            args.ransac_thresh, args.ransac_iters,
                            not args.no_refine_scale, args.min_matches)
    if res is None:
        return side["pts_c"], info
    sim, inl, rmse = res
    info.update(n_3d_inliers=int(inl.sum()), scale=round(float(sim[0]), 5),
                sim_rmse_m=round(rmse, 5), status="sift_similarity")
    refined = apply_sim(side["pts_c"], sim)

    if args.refine == "sift+icp":
        try:
            T, fit, irmse = icp_polish(refined, center["pts_c"],
                                       args.icp_voxel, args.icp_threshold,
                                       args.icp_iters)
            refined = apply_rigid_T(refined, T)
            info.update(icp_fitness=round(fit, 4), icp_rmse_m=round(irmse, 5),
                        status="sift_similarity+icp")
        except Exception as e:
            print(f"  [warn] ICP skipped ({e})")
    return refined, info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", required=True,
                    help="Directory containing center.png / left.png / right.png")
    ap.add_argument("--results-dir",
                    help="Directory with extrinsics.json and intrinsics_{cam}.json")
    ap.add_argument("--extrinsics", help="Override path to extrinsics.json")
    ap.add_argument("--intrinsics-center", help="Override intrinsics_center.json")
    ap.add_argument("--intrinsics-left", help="Override intrinsics_left.json")
    ap.add_argument("--intrinsics-right", help="Override intrinsics_right.json")

    ap.add_argument("--output", default="merged.ply")
    ap.add_argument("--per-camera-ply", action="store_true",
                    help="Also write center.ply / left.ply / right.ply in center frame")

    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1",
                    help="DA3 checkpoint id (ignored when --depth-dir is used). "
                         "Nested giant-large is metric-scale but CC-BY-NC 4.0.")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--depth-dir",
                    help="Load precomputed depth instead of running DA3")
    ap.add_argument("--depth-pattern", default="{cam}_depth.npy",
                    help="Filename pattern under --depth-dir, e.g. depth_{cam}.npy")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="Multiply depth by this factor (units -> metres if needed)")
    ap.add_argument("--save-depth", action="store_true",
                    help="Save each depth map as .npy + a colourised preview .png")

    ap.add_argument("--no-flip-left", dest="flip_left", action="store_false",
                    help="Do NOT 180-deg-flip the left image before inference")
    ap.set_defaults(flip_left=True)

    ap.add_argument("--min-depth", type=float, default=0.2)
    ap.add_argument("--max-depth", type=float, default=8.0)
    ap.add_argument("--stride", type=int, default=2,
                    help="Pixel subsampling step (2 -> quarter of the points)")
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="Drop points below this DA3 confidence (0 disables)")
    ap.add_argument("--voxel-size", type=float, default=0.0,
                    help="Voxel downsample size in metres (needs open3d; 0 disables)")
    ap.add_argument("--depth-is-ray", action="store_true",
                    help="Treat DA3 depth as distance-along-ray, not z-depth "
                         "(flip if a flat wall bows toward the frame edges)")
    ap.add_argument("--invert-extrinsics", action="store_true",
                    help="Use the other R,t convention if the merge looks wrong")

    # --- refinement (SIFT scale-and-shift + ICP) ---
    ap.add_argument("--refine", choices=["none", "sift", "sift+icp"],
                    default="sift+icp",
                    help="Align each side cloud to center before merging")
    ap.add_argument("--no-refine-scale", action="store_true",
                    help="Rigid-only correction (6-DOF) instead of similarity (7-DOF); "
                         "keeps DA3 scale, does not correct scale bias")
    ap.add_argument("--lowe-ratio", type=float, default=0.75)
    ap.add_argument("--max-features", type=int, default=8000)
    ap.add_argument("--min-matches", type=int, default=12,
                    help="Fall back to calibration-only if fewer survive")
    ap.add_argument("--epipolar-thresh", type=float, default=3.0,
                    help="2D fundamental-matrix RANSAC threshold (px)")
    ap.add_argument("--ransac-thresh", type=float, default=0.03,
                    help="3D similarity RANSAC inlier threshold (m)")
    ap.add_argument("--ransac-iters", type=int, default=3000)
    ap.add_argument("--icp-voxel", type=float, default=0.01,
                    help="Voxel size for ICP registration (m)")
    ap.add_argument("--icp-threshold", type=float, default=0.02,
                    help="ICP max correspondence distance (m)")
    ap.add_argument("--icp-iters", type=int, default=60)

    # --- absolute depth scale from the calibrated baseline (no GT needed) ---
    ap.add_argument("--solve-depth-scale", action="store_true",
                    help="Recover the DA3 depth scale by minimising cross-camera "
                         "disagreement. Uses the calibrated baseline as the "
                         "metric ruler; extrinsics are left untouched")
    ap.add_argument("--scale-solve-range", type=float, nargs=2,
                    default=[0.6, 1.6], metavar=("LO", "HI"))
    ap.add_argument("--scale-solve-steps", type=int, default=13)
    ap.add_argument("--scale-solve-refine", type=int, default=8,
                    help="Golden-section refinement iterations")
    ap.add_argument("--scale-solve-stride", type=int, default=2,
                    help="Extra pixel-stride multiplier during the solve (speed)")
    ap.add_argument("--scale-solve-cap", type=float, default=0.25,
                    help="Ignore NN distances beyond this when scoring (m)")
    ap.add_argument("--scale-solve-normal-radius", type=float, default=0.05,
                    help="Normal-estimation radius for point-to-plane scoring (m)")
    ap.add_argument("--scale-solve-min-contrast", type=float, default=0.25,
                    help="Warn if the scale curve is flatter than this "
                         "(scene cannot constrain scale)")

    # --- output cleanup ---
    ap.add_argument("--sor-neighbors", type=int, default=20,
                    help="Statistical outlier removal neighbours (0 disables)")
    ap.add_argument("--sor-std", type=float, default=2.0,
                    help="Statistical outlier removal std-ratio")
    args = ap.parse_args()

    rdir = args.results_dir or args.snapshot_dir
    ext_path = args.extrinsics or os.path.join(rdir, "extrinsics.json")
    intr_paths = {
        "center": args.intrinsics_center or os.path.join(rdir, "intrinsics_center.json"),
        "left":   args.intrinsics_left   or os.path.join(rdir, "intrinsics_left.json"),
        "right":  args.intrinsics_right  or os.path.join(rdir, "intrinsics_right.json"),
    }

    extrinsics = load_extrinsics(ext_path)
    intrinsics = {c: load_intrinsics(p) for c, p in intr_paths.items()}

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    model = None
    if args.depth_dir is None:
        print(f"Loading DA3 model: {args.model}")
        model = load_da3(args.model, args.device)

    cam_data = {}
    report = {"cameras": {}, "refinement": {}, "args": vars(args)}
    t_start = time.time()

    for cam in CAMS:
        K, dist, (W, H) = intrinsics[cam]
        R, t = extrinsics[cam]

        img_path = os.path.join(args.snapshot_dir, f"{cam}.png")
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        flipped = cam == "left" and args.flip_left
        rgb_inf = cv2.rotate(rgb, cv2.ROTATE_180) if flipped else rgb

        t0 = time.time()
        if args.depth_dir is not None:
            depth, conf = load_depth_file(args.depth_dir, args.depth_pattern, cam)
        else:
            depth, conf = da3_infer(model, rgb_inf, args.device)
            if flipped:
                depth = cv2.rotate(depth, cv2.ROTATE_180)
                if conf is not None:
                    conf = cv2.rotate(conf, cv2.ROTATE_180)
        infer_ms = (time.time() - t0) * 1000.0

        depth = depth.astype(np.float32) * args.depth_scale
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            if conf is not None:
                conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_NEAREST)

        if args.save_depth:
            np.save(os.path.join(out_dir, f"{cam}_depth.npy"), depth)
            valid = np.isfinite(depth) & (depth > 0)
            vis = np.zeros_like(depth)
            if np.any(valid):
                lo, hi = np.percentile(depth[valid], [2, 98])
                vis = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            vis8 = (vis * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{cam}_depth.png"),
                        cv2.applyColorMap(vis8, cv2.COLORMAP_TURBO))

        pts_cam, cols = unproject(depth, K, dist, rgb, args.stride,
                                  args.min_depth, args.max_depth,
                                  conf, args.conf_threshold,
                                  depth_is_ray=args.depth_is_ray)
        pts_c = to_center(pts_cam, R, t, args.invert_extrinsics)

        cam_data[cam] = {"gray": gray, "rgb": rgb, "depth": depth, "conf": conf,
                         "K": K, "dist": dist, "R": R, "t": t,
                         "pts_c": pts_c, "cols": cols}

        cc = camera_center(R, t, args.invert_extrinsics)
        report["cameras"][cam] = {
            "points": int(pts_c.shape[0]),
            "infer_ms": round(infer_ms, 1),
            "center_in_center_frame_m": [round(float(v), 4) for v in cc],
            "flipped_for_inference": bool(flipped),
        }
        print(f"{cam:>6}: {pts_c.shape[0]:>9,d} pts | infer {infer_ms:7.1f} ms | "
              f"cam center {np.round(cc, 3).tolist()}")

    # --- solve absolute depth scale from the calibrated baseline ------------ #
    if args.solve_depth_scale:
        print("\nSolving depth scale from cross-camera consistency "
              "(baseline as metric ruler)...")
        try:
            k, diag = solve_depth_scale(cam_data, args)
            report["depth_scale_solve"] = {
                "solved_k": round(k, 5),
                "note": "multiply DA3 depth by k; extrinsics stay unchanged",
                **diag}
            print(f"\n  solved depth scale k = {k:.4f} "
                  f"(depth was {abs(1 - k) * 100:.1f}% "
                  f"{'under' if k > 1 else 'over'}-estimated)")
            if diag["degenerate"]:
                print("  [warn] flat curve; k is unreliable for this scene")
            print("  rebuilding clouds at the solved scale...")
            for cam in CAMS:
                d = cam_data[cam]
                d["depth"] = d["depth"] * k
                pts_cam, cols_k = unproject(
                    d["depth"], d["K"], d["dist"], d["rgb"], args.stride,
                    args.min_depth, args.max_depth, d["conf"],
                    args.conf_threshold, depth_is_ray=args.depth_is_ray)
                d["pts_c"] = to_center(pts_cam, d["R"], d["t"],
                                       args.invert_extrinsics)
                d["cols"] = cols_k
                report["cameras"][cam]["points"] = int(d["pts_c"].shape[0])
        except Exception as e:
            print(f"  [warn] scale solve failed ({e}); continuing unscaled")
            report["depth_scale_solve"] = {"status": f"failed: {e}"}

    # --- refinement: align each side cloud onto center ---------------------- #
    refined = {"center": cam_data["center"]["pts_c"]}
    if args.refine == "sift+icp":
        try:
            import open3d  # noqa: F401
        except Exception:
            print("[warn] open3d unavailable; downgrading --refine to 'sift'")
            args.refine = "sift"

    if args.refine == "none":
        for cam in ("left", "right"):
            refined[cam] = cam_data[cam]["pts_c"]
            report["refinement"][cam] = {"status": "disabled"}
    else:
        print(f"\nRefining side clouds to center ({args.refine}, "
              f"{'rigid' if args.no_refine_scale else 'similarity'})...")
        for cam in ("left", "right"):
            pts_ref, info = refine_side_to_center(cam_data[cam],
                                                  cam_data["center"], args)
            refined[cam] = pts_ref
            report["refinement"][cam] = info
            print(f"{cam:>6}: matches {info['n_matches']:>4d} -> "
                  f"2D {info['n_2d_inliers']:>4d} -> 3D {info['n_3d_inliers']:>4d} | "
                  f"scale {info['scale']:.4f} | sim_rmse "
                  f"{info['sim_rmse_m']} | icp_rmse {info['icp_rmse_m']} | "
                  f"{info['status']}")

    if args.per_camera_ply:
        for cam in CAMS:
            if refined[cam].shape[0] > 0:
                write_ply(os.path.join(out_dir, f"{cam}.ply"),
                          refined[cam], cam_data[cam]["cols"])

    pts = np.concatenate([refined[c] for c in CAMS], axis=0)
    cols = np.concatenate([cam_data[c]["cols"] for c in CAMS], axis=0)

    n_raw = int(pts.shape[0])
    pts, cols = clean_cloud(pts, cols, args.voxel_size,
                            args.sor_neighbors, args.sor_std)

    write_ply(args.output, pts, cols)

    lc = np.asarray(report["cameras"]["left"]["center_in_center_frame_m"])
    rc = np.asarray(report["cameras"]["right"]["center_in_center_frame_m"])
    report["side_camera_separation_m"] = round(float(np.linalg.norm(lc - rc)), 4)
    report["points_before_cleanup"] = n_raw
    report["total_points"] = int(pts.shape[0])
    report["total_time_s"] = round(time.time() - t_start, 2)
    report["output"] = os.path.abspath(args.output)

    with open(os.path.join(out_dir, "merge_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nMerged {n_raw:,d} -> {pts.shape[0]:,d} points after cleanup "
          f"-> {args.output}")
    print(f"Side-camera separation: {report['side_camera_separation_m']} m "
          f"(expect ~1.68 m)")
    if args.refine != "none":
        print("Check merge_report.json: scale near 1.0 and sim_rmse of a few mm "
              "means a clean lock. If a side fell back to calibration_only, "
              "loosen --lowe-ratio or --ransac-thresh.")
    print("If the three clouds do not overlap at all, re-run with "
          "--invert-extrinsics.")


if __name__ == "__main__":
    main()