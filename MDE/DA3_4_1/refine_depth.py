#!/usr/bin/env python3
"""
refine_depth.py
===============

Frozen-pose metric refinement of DA3 depth against known calibrated baselines.

MOTIVATION
----------
diagnose_layers.py showed, for this rig:
    within-view floor flatness   ~10 mm RMS   (per-view depth is locally fine)
    between-view floor offset    ~230-350 mm  (each view sits at the wrong range)
so the dominant error is a per-view range error, not a spatial warp.

It also showed that DA3's absolute scale moves ~24% purely from input framing,
so the network cannot supply the metric anchor. The calibration can.

METHOD
------
For each view i, correct depth with an affine transform in INVERSE depth,
which is the space MDE errors are actually structured in:

    1 / Z_i(u,v)  =  a_i * (1 / Z_i_raw(u,v))  +  b_i

Solve all {a_i, b_i} jointly by requiring the views to agree geometrically,
with the calibrated extrinsics held FIXED. For a sampled pixel in view i,
back-project with the current parameters, transform into view j with the
calibrated pose, project, and compare the predicted inverse depth against
view j's own corrected inverse depth at that pixel. Residuals are formed in
inverse depth and minimised under a Huber loss.

Because the poses are metric and fixed, this is triangulation against a known
baseline, not mere mutual consistency: the absolute scale is determined by the
calibration rather than by the network. There is no gauge freedom to fix.

The extrinsics are never adjusted. Nothing is moved to make the render agree;
the geometry you trust constrains the geometry you do not.

Usage
-----
    python refine_depth.py \
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
    from scipy.optimize import least_squares
except ImportError:
    raise SystemExit("scipy is required:  pip install scipy")

try:
    import cv2
except ImportError:
    cv2 = None

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
        nrm /= ln
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
    """Bilinear sample with NaN outside the image."""
    h, w = img.shape
    out = np.full(u.shape, np.nan, dtype=np.float64)
    ok = (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
    if not np.any(ok):
        return out
    uu, vv = u[ok], v[ok]
    x0 = np.floor(uu).astype(int)
    y0 = np.floor(vv).astype(int)
    fx, fy = uu - x0, vv - y0
    x1, y1 = x0 + 1, y0 + 1
    val = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
           img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    out[ok] = val
    return out


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #

class View:
    def __init__(self, name, depth, K, rot_k, R, t):
        self.name = name
        self.depth_raw = depth
        self.inv_raw = np.where(depth > EPS, 1.0 / np.maximum(depth, EPS), np.nan)
        self.K = K
        self.rot_k = int(rot_k)
        self.Rk = Rz(90.0 * self.rot_k)
        self.R = R
        self.t = t
        self.a = 1.0
        self.b = 0.0

    def corrected_depth(self, a: float, b: float) -> np.ndarray:
        inv = a * self.inv_raw + b
        return np.where(inv > EPS, 1.0 / np.maximum(inv, EPS), np.nan)

    def cam_points(self, a: float, b: float, mask=None) -> np.ndarray:
        d = self.corrected_depth(a, b)
        h, w = d.shape
        u = np.arange(w, dtype=np.float64)[None, :]
        v = np.arange(h, dtype=np.float64)[:, None]
        xn = (u - self.K[0, 2]) / self.K[0, 0]
        yn = (v - self.K[1, 2]) / self.K[1, 1]
        pts = np.stack([xn * d, yn * d, d], axis=-1).reshape(-1, 3)
        if mask is not None:
            pts = pts[mask]
        return pts @ self.Rk.T

    def to_ref(self, pts_cam: np.ndarray) -> np.ndarray:
        return (pts_cam - self.t[None, :]) @ self.R

    def from_ref(self, pts_ref: np.ndarray) -> np.ndarray:
        return pts_ref @ self.R.T + self.t[None, :]

    def project(self, pts_cam: np.ndarray):
        p = pts_cam @ self.Rk                      # true cam -> rotated frame
        z = p[:, 2]
        u = (p[:, 0] / np.maximum(z, EPS)) * self.K[0, 0] + self.K[0, 2]
        v = (p[:, 1] / np.maximum(z, EPS)) * self.K[1, 1] + self.K[1, 2]
        return u, v, z


def load_views(run: Path, mode: str, calib_dir: Path) -> List[View]:
    diag = json.loads((run / mode / "diagnostics.json").read_text())
    ext_all = json.loads((calib_dir / "extrinsics.json").read_text())
    views = []
    for e in diag["per_camera"]:
        name = e["camera"]
        depth = np.load(run / mode / f"depth_{name}.npy").astype(np.float64)
        kpath = run / mode / f"K_{name}.npy"
        K = (np.load(kpath).astype(np.float64) if kpath.exists() else
             np.array([[e["K_used_fx"], 0, e["K_used_cx"]],
                       [0, e["K_used_fy"], e["K_used_cy"]], [0, 0, 1]]))
        views.append(View(name, depth, K,
                          e["rot_k"],
                          np.asarray(ext_all[name]["R"], dtype=np.float64),
                          np.asarray(ext_all[name]["t"], dtype=np.float64)))
    return views


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def evaluate(views: List[View], params: np.ndarray, zmin: float, zmax: float,
             stride: int) -> Dict:
    ab = params.reshape(-1, 2)
    planes, pts_ref_all = [], []
    for i, v in enumerate(views):
        a, b = float(np.exp(ab[i, 0])), float(ab[i, 1])
        pc = v.cam_points(a, b)
        z = pc[:, 2]
        keep = np.isfinite(pc).all(1) & (z > zmin) & (z < zmax)
        pr = v.to_ref(pc[keep])
        pts_ref_all.append(pr)
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

    pair_res = []
    for i, j in itertools.combinations(range(len(views)), 2):
        vi, vj = views[i], views[j]
        ai, bi = float(np.exp(ab[i, 0])), float(ab[i, 1])
        aj, bj = float(np.exp(ab[j, 0])), float(ab[j, 1])
        pc = vi.cam_points(ai, bi)[::stride]
        z = pc[:, 2]
        keep = np.isfinite(pc).all(1) & (z > zmin) & (z < zmax)
        pj = vj.from_ref(vi.to_ref(pc[keep]))
        u, v, zj = vj.project(pj)
        front = zj > zmin
        inv_raw = bilinear(vj.inv_raw, u[front], v[front])
        inv_j = aj * inv_raw + bj
        pred = 1.0 / np.maximum(zj[front], EPS)
        diff_m = np.abs(1.0 / np.maximum(pred, EPS) - 1.0 / np.maximum(inv_j, EPS))
        good = np.isfinite(diff_m) & (inv_j > EPS)
        if good.sum() > 50:
            pair_res.append({"pair": [vi.name, vj.name], "n": int(good.sum()),
                             "median_abs_residual_mm": float(np.median(diff_m[good]) * 1000),
                             "p90_abs_residual_mm": float(np.percentile(diff_m[good], 90) * 1000)})
    out["cross_view"] = pair_res
    if pair_res:
        out["overall_median_residual_mm"] = float(
            np.median([p["median_abs_residual_mm"] for p in pair_res]))
    return out, pts_ref_all


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #

def make_residual_fn(views, zmin, zmax, stride, prior_w):
    pairs = list(itertools.combinations(range(len(views)), 2))
    # pre-sample source pixels once so the problem is fixed across iterations
    samples = {}
    for i, v in enumerate(views):
        h, w = v.inv_raw.shape
        uu, vv = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
        uu, vv = uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)
        inv = v.inv_raw[vv.astype(int), uu.astype(int)]
        ok = np.isfinite(inv) & (inv > EPS)
        samples[i] = (uu[ok], vv[ok], inv[ok])

    # normalised image coordinates, fixed across iterations
    norm = {}
    for i, v in enumerate(views):
        uu, vv, _ = samples[i]
        norm[i] = ((uu - v.K[0, 2]) / v.K[0, 0], (vv - v.K[1, 2]) / v.K[1, 1])

    def residuals(p):
        # NOTE: the residual vector must have CONSTANT length across iterations,
        # so invalid samples are zeroed rather than removed. Dropping them makes
        # the length parameter-dependent and least_squares fails.
        ab = p.reshape(-1, 2)
        res = []
        for i, j in pairs:
            vi, vj = views[i], views[j]
            ai, bi = np.exp(ab[i, 0]), ab[i, 1]
            aj, bj = np.exp(ab[j, 0]), ab[j, 1]
            uu, vv, inv_raw_i = samples[i]
            xn, yn = norm[i]
            inv_i = ai * inv_raw_i + bi
            zi = np.where(inv_i > EPS, 1.0 / np.maximum(inv_i, EPS), np.nan)
            pc = np.stack([xn * zi, yn * zi, zi], axis=-1) @ vi.Rk.T
            pj = vj.from_ref(np.nan_to_num(vi.to_ref(pc), nan=0.0))
            u2, v2, z2 = vj.project(pj)
            inv_raw_j = bilinear(vj.inv_raw, u2, v2)
            inv_j = aj * inv_raw_j + bj
            r = 1.0 / np.maximum(z2, EPS) - inv_j
            ok = (np.isfinite(r) & np.isfinite(zi) & (zi > zmin) & (zi < zmax)
                  & (z2 > zmin) & (inv_j > EPS))
            res.append(np.where(ok, r, 0.0))
        r = np.concatenate(res)
        # weak prior keeping the solution near the raw parameterisation
        prior = prior_w * np.concatenate([ab[:, 0], ab[:, 1]])
        return np.nan_to_num(np.concatenate([r, prior]),
                             nan=0.0, posinf=0.0, neginf=0.0)

    return residuals


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Frozen-pose per-view inverse-depth affine refinement.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mode", default="prior", choices=["prior", "noprior"])
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_1/results"))
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=12.0)
    ap.add_argument("--stride", type=int, default=6, help="pixel stride for the solve")
    ap.add_argument("--eval-stride", type=int, default=4)
    ap.add_argument("--huber", type=float, default=0.01,
                    help="Huber scale in inverse-depth units")
    ap.add_argument("--prior-weight", type=float, default=1e-3)
    ap.add_argument("--scale-only", action="store_true",
                    help="solve scale only, hold shift at zero")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (args.run / args.mode / "refined")
    out.mkdir(parents=True, exist_ok=True)

    views = load_views(args.run, args.mode, args.calib_dir)
    print(f"loaded {len(views)} views from {args.run / args.mode}")
    for v in views:
        print(f"  {v.name:7s} {v.depth_raw.shape} rot_k={v.rot_k} "
              f"median={np.nanmedian(v.depth_raw):.3f} m")

    p0 = np.zeros(2 * len(views))       # log a = 0, b = 0

    print("\nBEFORE")
    before, _ = evaluate(views, p0, args.min_depth, args.max_depth, args.eval_stride)
    for k in ("max_pairwise_normal_angle_deg", "offset_spread_m",
              "mean_plane_rms_m", "overall_median_residual_mm"):
        if k in before:
            print(f"  {k:35s} {before[k]:.4f}")

    fn = make_residual_fn(views, args.min_depth, args.max_depth,
                          args.stride, args.prior_weight)
    if args.scale_only:
        mask = np.zeros_like(p0, dtype=bool)
        mask[0::2] = True

        def fn_scaled(q):
            p = np.zeros_like(p0)
            p[mask] = q
            return fn(p)

        print("\nsolving (scale only) ...")
        sol = least_squares(fn_scaled, p0[mask], loss="huber", f_scale=args.huber,
                            max_nfev=200, verbose=1)
        p = np.zeros_like(p0)
        p[mask] = sol.x
    else:
        print("\nsolving (scale + shift, inverse depth) ...")
        sol = least_squares(fn, p0, loss="huber", f_scale=args.huber,
                            max_nfev=300, verbose=1)
        p = sol.x

    ab = p.reshape(-1, 2)
    print("\nsolved parameters   1/Z = a*(1/Z_raw) + b")
    for i, v in enumerate(views):
        a, b = float(np.exp(ab[i, 0])), float(ab[i, 1])
        med_raw = float(np.nanmedian(v.depth_raw))
        med_new = float(np.nanmedian(v.corrected_depth(a, b)))
        print(f"  {v.name:7s} a={a:8.5f}  b={b:+9.6f}   "
              f"median depth {med_raw:6.3f} -> {med_new:6.3f} m")

    print("\nAFTER")
    after, pts_ref_all = evaluate(views, p, args.min_depth, args.max_depth,
                                  args.eval_stride)
    for k in ("max_pairwise_normal_angle_deg", "offset_spread_m",
              "mean_plane_rms_m", "overall_median_residual_mm"):
        if k in after:
            b_v = before.get(k, float("nan"))
            print(f"  {k:35s} {after[k]:.4f}   (was {b_v:.4f})")

    print("\n  per-pair cross-view residual (median, mm)")
    bmap = {tuple(r["pair"]): r for r in before.get("cross_view", [])}
    for r in after.get("cross_view", []):
        was = bmap.get(tuple(r["pair"]), {}).get("median_abs_residual_mm", float("nan"))
        print(f"    {r['pair'][0]:>7s} <-> {r['pair'][1]:<7s} "
              f"{r['median_abs_residual_mm']:8.1f}   (was {was:8.1f})")

    # ---- write refined artefacts ------------------------------------------ #
    all_pts, all_cols = [], []
    for i, v in enumerate(views):
        a, b = float(np.exp(ab[i, 0])), float(ab[i, 1])
        np.save(out / f"depth_{v.name}_refined.npy", v.corrected_depth(a, b))
        pr = pts_ref_all[i]
        col = np.tile(np.array(CAM_COLOURS.get(v.name, (200, 200, 200)), np.uint8),
                      (pr.shape[0], 1))
        write_ply(out / f"cam_{v.name}_refined.ply", pr, col)
        all_pts.append(pr)
        all_cols.append(col)

    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)
    if args.voxel > 0:
        keys = np.floor(pts / args.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx.sort()
        pts, cols = pts[idx], cols[idx]
    write_ply(out / "fused_refined_colourcoded.ply", pts, cols)

    (out / "refine_report.json").write_text(json.dumps({
        "run": str(args.run), "mode": args.mode,
        "scale_only": bool(args.scale_only),
        "parameters": [{"camera": v.name,
                        "a": float(np.exp(ab[i, 0])), "b": float(ab[i, 1])}
                       for i, v in enumerate(views)],
        "before": before, "after": after,
    }, indent=2, default=float))

    print(f"\nwrote {out}/fused_refined_colourcoded.ply")
    print(f"      {out}/refine_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())