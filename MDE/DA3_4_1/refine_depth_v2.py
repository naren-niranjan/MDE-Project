#!/usr/bin/env python3
"""
refine_depth_v2.py
==================

Metric refinement of DA3 depth anchored on TRIANGULATED sparse correspondences,
with the calibrated poses held fixed.

WHY v1 FAILED
-------------
v1 minimised the disagreement between depth maps: take view i's point, push it
into view j with the calibrated pose, compare against view j's depth there.
That cost never looks at the images. Two depth maps that are both constant and
far away agree almost perfectly, because parallax vanishes with distance, so
the global minimum sits at infinity. v1 duly found it: a -> 0, b -> const,
every view flattened to 12-22 m.

Fixing the poses does NOT remove the scale freedom. Triangulation determines
scale only via correspondences between physical points, and those come from
image appearance.

WHAT v2 DOES
------------
1. Reconstruct the exact image each view's depth map corresponds to, by
   replaying the rotation, padding, resize and crop recorded in the v2 fusion
   diagnostics.
2. SIFT match every camera pair on those images.
3. Triangulate each match with the CALIBRATED intrinsics and extrinsics. The
   resulting 3D point is metric and owes nothing to the network. Reject matches
   with low parallax, high reprojection error, or out-of-range depth.
4. For each view, the triangulated points give hard measurements of true depth
   at known pixels. Fit the inverse-depth affine robustly:

       1 / Z_true  =  a * (1 / Z_raw)  +  b

   solved per view by IRLS with a Huber weight. Anchored to real geometry, so
   there is no collapse to infinity.
5. Optionally (--joint) refine all views together afterwards, combining the
   dense cross-view term with the anchor term. The anchors pin the scale; the
   dense term tightens agreement between overlaps.

The extrinsics are never modified.

Usage
-----
    python refine_depth_v2.py \
        --run ~/Projects/MDE/DA3_4_1/runs/20260804_092615_018_v2_pad \
        --mode prior
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV is required")

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

CAM_COLOURS = {"left": (230, 60, 60), "center": (60, 200, 90),
               "right": (70, 120, 240), "top": (240, 190, 50)}
EPS = 1e-9


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #

def Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(pts.shape[0])
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n").encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.empty(n, dtype=dt)
    a["x"], a["y"], a["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    c = np.clip(cols, 0, 255).astype(np.uint8)
    a["red"], a["green"], a["blue"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(a.tobytes())


def ransac_plane(pts: np.ndarray, thresh: float = 0.02, iters: int = 400,
                 seed: int = 0) -> Optional[Dict]:
    if pts.shape[0] < 100:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if pts.shape[0] <= 200000 else pts[rng.choice(pts.shape[0], 200000, False)]
    best = None
    for _ in range(iters):
        p0, p1, p2 = sub[rng.choice(sub.shape[0], 3, replace=False)]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -nrm @ p0
        n_in = int((np.abs(sub @ nrm + d) < thresh).sum())
        if best is None or n_in > best[0]:
            best = (n_in, nrm, d)
    if best is None:
        return None
    _, nrm, d = best
    inl = sub[np.abs(sub @ nrm + d) < thresh]
    if inl.shape[0] < 50:
        return None
    cen = inl.mean(0)
    _, _, Vt = np.linalg.svd(inl - cen, full_matrices=False)
    nrm = Vt[-1] / np.linalg.norm(Vt[-1])
    if nrm[2] < 0:
        nrm = -nrm
    d = -nrm @ cen
    return {"normal": nrm.tolist(), "offset_m": float(d),
            "rms_m": float(np.sqrt((((inl @ nrm + d) ** 2)).mean())),
            "n_inliers": int(inl.shape[0])}


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    out = np.full(np.shape(u), np.nan, dtype=np.float64)
    ok = (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
    if not np.any(ok):
        return out
    uu, vv = np.asarray(u)[ok], np.asarray(v)[ok]
    x0, y0 = np.floor(uu).astype(int), np.floor(vv).astype(int)
    fx, fy = uu - x0, vv - y0
    x1, y1 = x0 + 1, y0 + 1
    out[ok] = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
               img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return out


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #

class View:
    def __init__(self, name, depth, K, rot_k, R, t, image):
        self.name = name
        self.depth_raw = depth
        self.inv_raw = np.where(depth > EPS, 1.0 / np.maximum(depth, EPS), np.nan)
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.rot_k = int(rot_k)
        self.Rk = Rz(90.0 * self.rot_k)
        self.R = R
        self.t = t
        self.image = image            # aligned with depth, RGB uint8
        self.a = 1.0
        self.b = 0.0

    # --- geometry ------------------------------------------------------- #
    @property
    def C(self) -> np.ndarray:
        return -self.R.T @ self.t

    def rays_ref(self, uv: np.ndarray) -> np.ndarray:
        """Pixel(s) in the processed image -> unit ray direction(s) in ref frame."""
        pix = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        d_rot = pix @ self.Kinv.T
        d_cam = d_rot @ self.Rk.T          # rotated frame -> true camera frame
        d_ref = d_cam @ self.R             # (R^T d_cam)^T
        return d_ref / np.linalg.norm(d_ref, axis=1, keepdims=True)

    def depth_of_ref_point(self, P: np.ndarray) -> np.ndarray:
        """Z along the optical axis for reference-frame point(s). Rz preserves z."""
        return (P @ self.R.T + self.t[None, :])[:, 2]

    def project_ref(self, P: np.ndarray):
        pc = P @ self.R.T + self.t[None, :]
        p = pc @ self.Rk
        z = p[:, 2]
        u = (p[:, 0] / np.maximum(z, EPS)) * self.K[0, 0] + self.K[0, 2]
        v = (p[:, 1] / np.maximum(z, EPS)) * self.K[1, 1] + self.K[1, 2]
        return u, v, z

    # --- depth correction ------------------------------------------------ #
    def corrected_depth(self, a=None, b=None) -> np.ndarray:
        a = self.a if a is None else a
        b = self.b if b is None else b
        inv = a * self.inv_raw + b
        return np.where(inv > EPS, 1.0 / np.maximum(inv, EPS), np.nan)

    def cloud_ref(self, zmin: float, zmax: float):
        d = self.corrected_depth()
        h, w = d.shape
        u = np.arange(w, dtype=np.float64)[None, :]
        v = np.arange(h, dtype=np.float64)[:, None]
        xn = (u - self.K[0, 2]) / self.K[0, 0]
        yn = (v - self.K[1, 2]) / self.K[1, 1]
        pts = np.stack([xn * d, yn * d, d], axis=-1).reshape(-1, 3) @ self.Rk.T
        z = pts[:, 2]
        keep = np.isfinite(pts).all(1) & (z > zmin) & (z < zmax)
        return (pts[keep] - self.t[None, :]) @ self.R, keep


# --------------------------------------------------------------------------- #
# rebuilding the processed image
# --------------------------------------------------------------------------- #

def rebuild_processed_image(entry: Dict, summary: Dict, calib_dir: Path,
                            depth_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Replay rotation, padding, resize and crop so the image matches the depth."""
    name = entry["camera"]
    cam_sum = next((c for c in summary["cameras"] if c["name"] == name), None)
    if cam_sum is None:
        return None
    img = cv2.imread(cam_sum["image"], cv2.IMREAD_COLOR)
    if img is None:
        return None
    kj = json.loads((calib_dir / f"intrinsics_{name}.json").read_text())
    K = np.asarray(kj["camera_matrix"], dtype=np.float64)
    dist = np.asarray(kj.get("dist_coeffs", [0] * 5), dtype=np.float64).ravel()
    if np.any(np.abs(dist) > 1e-12):
        img = cv2.undistort(img, K, dist, None, K)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rot = np.ascontiguousarray(np.rot90(rgb, k=int(entry["rot_k"])))

    fed_w, fed_h = entry["fed_size"]
    if (rot.shape[1], rot.shape[0]) != (fed_w, fed_h):
        # padding was applied by the fusion script
        side = max(rot.shape[0], rot.shape[1])
        px = (side - rot.shape[1]) // 2
        py = (side - rot.shape[0]) // 2
        rot = cv2.copyMakeBorder(rot, py, side - rot.shape[0] - py,
                                 px, side - rot.shape[1] - px, cv2.BORDER_REPLICATE)

    tf = entry["recovered_transform"]
    resized = cv2.resize(rot, (int(tf["w_r"]), int(tf["h_r"])), interpolation=cv2.INTER_AREA)
    x0, y0 = int(tf["x0"]), int(tf["y0"])
    h_o, w_o = depth_hw
    crop = resized[y0:y0 + h_o, x0:x0 + w_o]
    if crop.shape[:2] != (h_o, w_o):
        return None
    return np.ascontiguousarray(crop)


def load_views(run: Path, mode: str, calib_dir: Path) -> List[View]:
    diag = json.loads((run / mode / "diagnostics.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    ext_all = json.loads((calib_dir / "extrinsics.json").read_text())
    views = []
    for e in diag["per_camera"]:
        name = e["camera"]
        depth = np.load(run / mode / f"depth_{name}.npy").astype(np.float64)
        kpath = run / mode / f"K_{name}.npy"
        K = (np.load(kpath).astype(np.float64) if kpath.exists() else
             np.array([[e["K_used_fx"], 0, e["K_used_cx"]],
                       [0, e["K_used_fy"], e["K_used_cy"]], [0, 0, 1]]))
        img = rebuild_processed_image(e, summary, calib_dir, depth.shape)
        if img is None:
            raise SystemExit(f"could not rebuild the processed image for {name}")
        views.append(View(name, depth, K, e["rot_k"],
                          np.asarray(ext_all[name]["R"], dtype=np.float64),
                          np.asarray(ext_all[name]["t"], dtype=np.float64), img))
    return views


# --------------------------------------------------------------------------- #
# sparse anchors
# --------------------------------------------------------------------------- #

def sift_matches(img_a: np.ndarray, img_b: np.ndarray, n_feat: int,
                 ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=n_feat)
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw = matcher.knnMatch(da, db, k=2)
    pa, pb = [], []
    for m_n in raw:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            pa.append(ka[m.queryIdx].pt)
            pb.append(kb[m.trainIdx].pt)
    return np.asarray(pa, dtype=np.float64), np.asarray(pb, dtype=np.float64)


def triangulate(va: View, vb: View, pa: np.ndarray, pb: np.ndarray):
    """Midpoint triangulation of two ray bundles in the reference frame."""
    Ca, Cb = va.C, vb.C
    da, db = va.rays_ref(pa), vb.rays_ref(pb)
    b = Cb - Ca
    d1d1 = np.ones(len(da))
    d1d2 = np.einsum("ij,ij->i", da, db)
    d2d2 = np.ones(len(db))
    bd1 = da @ b
    bd2 = db @ b
    den = d1d1 * d2d2 - d1d2 ** 2
    ok = np.abs(den) > 1e-8
    s = np.zeros(len(da))
    tt = np.zeros(len(da))
    s[ok] = (bd1[ok] * d2d2[ok] - bd2[ok] * d1d2[ok]) / den[ok]
    tt[ok] = (bd1[ok] * d1d2[ok] - bd2[ok] * d1d1[ok]) / den[ok]
    Pa = Ca[None, :] + s[:, None] * da
    Pb = Cb[None, :] + tt[:, None] * db
    P = 0.5 * (Pa + Pb)
    gap = np.linalg.norm(Pa - Pb, axis=1)
    parallax = np.degrees(np.arccos(np.clip(d1d2, -1, 1)))
    return P, gap, parallax, ok


def collect_anchors(views: List[View], args) -> Dict[int, Dict[str, np.ndarray]]:
    """Metric depth measurements per view from triangulated correspondences."""
    store = {i: {"u": [], "v": [], "z": []} for i in range(len(views))}
    print("\nsparse correspondences")
    for i, j in itertools.combinations(range(len(views)), 2):
        va, vb = views[i], views[j]
        pa, pb = sift_matches(va.image, vb.image, args.n_features, args.ratio)
        if len(pa) < 8:
            print(f"  {va.name:>7s} <-> {vb.name:<7s}  too few matches ({len(pa)})")
            continue
        P, gap, parallax, ok = triangulate(va, vb, pa, pb)
        za = va.depth_of_ref_point(P)
        zb = vb.depth_of_ref_point(P)
        ua, vv_a, _ = va.project_ref(P)
        ub, vv_b, _ = vb.project_ref(P)
        rep_a = np.hypot(ua - pa[:, 0], vv_a - pa[:, 1])
        rep_b = np.hypot(ub - pb[:, 0], vv_b - pb[:, 1])
        good = (ok & (parallax > args.min_parallax) & (gap < args.max_ray_gap) &
                (za > args.min_depth) & (za < args.max_depth) &
                (zb > args.min_depth) & (zb < args.max_depth) &
                (rep_a < args.max_reproj) & (rep_b < args.max_reproj))
        print(f"  {va.name:>7s} <-> {vb.name:<7s}  {len(pa):5d} matches, "
              f"{int(good.sum()):5d} triangulated "
              f"(median parallax {np.median(parallax[good]) if good.any() else float('nan'):.1f} deg)")
        if not good.any():
            continue
        store[i]["u"].append(pa[good, 0]); store[i]["v"].append(pa[good, 1])
        store[i]["z"].append(za[good])
        store[j]["u"].append(pb[good, 0]); store[j]["v"].append(pb[good, 1])
        store[j]["z"].append(zb[good])

    out = {}
    for i in store:
        if not store[i]["u"]:
            out[i] = {"u": np.zeros(0), "v": np.zeros(0), "z": np.zeros(0),
                      "inv_raw": np.zeros(0)}
            continue
        u = np.concatenate(store[i]["u"])
        v = np.concatenate(store[i]["v"])
        z = np.concatenate(store[i]["z"])
        inv_raw = bilinear(views[i].inv_raw, u, v)
        m = np.isfinite(inv_raw) & (inv_raw > EPS) & np.isfinite(z) & (z > EPS)
        out[i] = {"u": u[m], "v": v[m], "z": z[m], "inv_raw": inv_raw[m]}
    return out


def fit_affine_irls(inv_raw: np.ndarray, z_true: np.ndarray, huber: float,
                    iters: int = 30) -> Tuple[float, float, Dict]:
    """Robust fit of  1/z_true = a * inv_raw + b."""
    y = 1.0 / np.maximum(z_true, EPS)
    X = np.stack([inv_raw, np.ones_like(inv_raw)], axis=1)
    w = np.ones_like(y)
    a, b = 1.0, 0.0
    for _ in range(iters):
        W = w[:, None]
        try:
            sol, *_ = np.linalg.lstsq(X * W, y * w, rcond=None)
        except np.linalg.LinAlgError:
            break
        a, b = float(sol[0]), float(sol[1])
        r = X @ sol - y
        s = max(huber, 1.4826 * np.median(np.abs(r - np.median(r))))
        w = np.where(np.abs(r) <= s, 1.0, s / np.maximum(np.abs(r), EPS))
    resid_m = np.abs(1.0 / np.maximum(a * inv_raw + b, EPS) - z_true)
    stats = {"n": int(len(y)),
             "median_abs_depth_error_mm": float(np.median(resid_m) * 1000),
             "p90_abs_depth_error_mm": float(np.percentile(resid_m, 90) * 1000)}
    return a, b, stats


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def evaluate(views: List[View], anchors, zmin: float, zmax: float, stride: int) -> Dict:
    planes, clouds = [], []
    for v in views:
        pr, _ = v.cloud_ref(zmin, zmax)
        clouds.append(pr)
        pl = ransac_plane(pr)
        if pl:
            pl["camera"] = v.name
            planes.append(pl)
    out: Dict = {"planes": planes}
    if len(planes) >= 2:
        nrm = np.array([p["normal"] for p in planes])
        off = np.array([p["offset_m"] for p in planes])
        angs = [float(np.degrees(np.arccos(np.clip(abs(nrm[a] @ nrm[b]), -1, 1))))
                for a, b in itertools.combinations(range(len(planes)), 2)]
        out["max_pairwise_normal_angle_deg"] = float(max(angs))
        out["offset_spread_m"] = float(off.max() - off.min())
        out["mean_plane_rms_m"] = float(np.mean([p["rms_m"] for p in planes]))

    anch = []
    for i, v in enumerate(views):
        A = anchors.get(i)
        if A is None or len(A["z"]) == 0:
            continue
        z_pred = 1.0 / np.maximum(v.a * A["inv_raw"] + v.b, EPS)
        err = np.abs(z_pred - A["z"])
        anch.append({"camera": v.name, "n": int(len(err)),
                     "median_abs_depth_error_mm": float(np.median(err) * 1000),
                     "p90_abs_depth_error_mm": float(np.percentile(err, 90) * 1000)})
    out["anchor_error"] = anch
    if anch:
        out["overall_anchor_median_mm"] = float(
            np.median([a["median_abs_depth_error_mm"] for a in anch]))

    pair = []
    for i, j in itertools.combinations(range(len(views)), 2):
        vi, vj = views[i], views[j]
        pr, _ = vi.cloud_ref(zmin, zmax)
        pr = pr[::stride]
        u, v, z = vj.project_ref(pr)
        front = z > zmin
        inv_raw = bilinear(vj.inv_raw, u[front], v[front])
        zj = 1.0 / np.maximum(vj.a * inv_raw + vj.b, EPS)
        diff = np.abs(z[front] - zj)
        g = np.isfinite(diff)
        if g.sum() > 50:
            pair.append({"pair": [vi.name, vj.name], "n": int(g.sum()),
                         "median_abs_residual_mm": float(np.median(diff[g]) * 1000)})
    out["cross_view"] = pair
    if pair:
        out["overall_median_residual_mm"] = float(
            np.median([p["median_abs_residual_mm"] for p in pair]))
    return out


def show(tag: str, m: Dict, ref: Optional[Dict] = None) -> None:
    print(f"\n{tag}")
    for k in ("max_pairwise_normal_angle_deg", "offset_spread_m", "mean_plane_rms_m",
              "overall_anchor_median_mm", "overall_median_residual_mm"):
        if k in m:
            if ref and k in ref:
                print(f"  {k:35s} {m[k]:10.4f}   (was {ref[k]:.4f})")
            else:
                print(f"  {k:35s} {m[k]:10.4f}")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Triangulation-anchored per-view depth refinement, poses frozen.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mode", default="prior", choices=["prior", "noprior"])
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_1/results"))
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=12.0)
    ap.add_argument("--n-features", type=int, default=20000)
    ap.add_argument("--ratio", type=float, default=0.8, help="Lowe ratio test")
    ap.add_argument("--min-parallax", type=float, default=1.0,
                    help="degrees; below this triangulation is ill-conditioned")
    ap.add_argument("--max-ray-gap", type=float, default=0.05,
                    help="metres between the two rays at closest approach")
    ap.add_argument("--max-reproj", type=float, default=3.0, help="pixels")
    ap.add_argument("--huber", type=float, default=0.005)
    ap.add_argument("--eval-stride", type=int, default=17)
    ap.add_argument("--scale-only", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (args.run / args.mode / "refined_v2")
    out.mkdir(parents=True, exist_ok=True)

    views = load_views(args.run, args.mode, args.calib_dir)
    print(f"loaded {len(views)} views from {args.run / args.mode}")
    for v in views:
        print(f"  {v.name:7s} depth {v.depth_raw.shape} image {v.image.shape} "
              f"rot_k={v.rot_k} median={np.nanmedian(v.depth_raw):.3f} m")

    anchors = collect_anchors(views, args)
    n_ok = sum(1 for i in anchors if len(anchors[i]["z"]) >= 20)
    if n_ok == 0:
        print("\nno usable triangulated anchors. Loosen --ratio, --max-reproj or "
              "--max-ray-gap, or check that the rebuilt images match the depth maps.")
        return 1

    before = evaluate(views, anchors, args.min_depth, args.max_depth, args.eval_stride)
    show("BEFORE", before)

    print("\nper-view robust affine fit against triangulated depth")
    params = []
    for i, v in enumerate(views):
        A = anchors[i]
        if len(A["z"]) < 20:
            print(f"  {v.name:7s} only {len(A['z'])} anchors, leaving uncorrected")
            params.append({"camera": v.name, "a": 1.0, "b": 0.0, "n_anchors": len(A["z"])})
            continue
        if args.scale_only:
            y = 1.0 / np.maximum(A["z"], EPS)
            a = float(np.median(y / np.maximum(A["inv_raw"], EPS)))
            b = 0.0
            resid = np.abs(1.0 / np.maximum(a * A["inv_raw"], EPS) - A["z"])
            st = {"n": int(len(y)),
                  "median_abs_depth_error_mm": float(np.median(resid) * 1000)}
        else:
            a, b, st = fit_affine_irls(A["inv_raw"], A["z"], args.huber)
        v.a, v.b = a, b
        med_raw = float(np.nanmedian(v.depth_raw))
        med_new = float(np.nanmedian(v.corrected_depth()))
        print(f"  {v.name:7s} a={a:8.5f} b={b:+9.6f}  n={st['n']:5d}  "
              f"anchor err {st['median_abs_depth_error_mm']:7.1f} mm  "
              f"median depth {med_raw:6.3f} -> {med_new:6.3f} m")
        params.append({"camera": v.name, "a": a, "b": b, "fit": st,
                       "n_anchors": int(st["n"])})

    after = evaluate(views, anchors, args.min_depth, args.max_depth, args.eval_stride)
    show("AFTER", after, before)

    print("\n  per-pair cross-view residual (median, mm)")
    bmap = {tuple(r["pair"]): r for r in before.get("cross_view", [])}
    for r in after.get("cross_view", []):
        was = bmap.get(tuple(r["pair"]), {}).get("median_abs_residual_mm", float("nan"))
        print(f"    {r['pair'][0]:>7s} <-> {r['pair'][1]:<7s} "
              f"{r['median_abs_residual_mm']:8.1f}   (was {was:8.1f})")

    all_pts, all_cols = [], []
    for v in views:
        pr, keep = v.cloud_ref(args.min_depth, args.max_depth)
        np.save(out / f"depth_{v.name}_refined.npy", v.corrected_depth())
        rgb = v.image.reshape(-1, 3)[keep]
        write_ply(out / f"cam_{v.name}_refined.ply", pr, rgb)
        all_pts.append(pr)
        all_cols.append(np.tile(np.array(CAM_COLOURS.get(v.name, (200, 200, 200)),
                                         np.uint8), (pr.shape[0], 1)))
        if len(all_cols) == 1:
            pass
    pts = np.concatenate(all_pts, 0)
    cc = np.concatenate(all_cols, 0)
    if args.voxel > 0:
        keys = np.floor(pts / args.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx.sort()
        pts, cc = pts[idx], cc[idx]
    write_ply(out / "fused_refined_colourcoded.ply", pts, cc)

    (out / "refine_report.json").write_text(json.dumps({
        "run": str(args.run), "mode": args.mode,
        "method": "triangulation_anchored_affine_inverse_depth",
        "parameters": params, "before": before, "after": after,
    }, indent=2, default=float))

    print(f"\nwrote {out}/fused_refined_colourcoded.ply")
    print(f"      per-camera cam_<name>_refined.ply with true colour")
    print(f"      {out}/refine_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())