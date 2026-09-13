#!/usr/bin/env python3
"""
refine_depth_v3.py
==================

Two jobs.

(1) FIX THE PLANE METRIC. v2 fitted a plane independently per camera and
    compared them. Once the depths were corrected the largest plane by inlier
    count became a different physical surface in different cameras, producing a
    meaningless 89 degree "disagreement". v3 instead fits one plane on the
    reference camera and measures every camera's points against THAT plane, so
    the comparison is always about the same surface.

(2) ANSWER WHETHER THE REMAINING ERROR IS SPATIALLY STRUCTURED, by
    cross-validation rather than assertion. Three correction models are fitted
    per view, all linear in their parameters:

        constant   1/Z = a*(1/Z_raw) + b                        (2 params)
        linear     a and b vary linearly over the image         (6 params)
        quadratic  a and b vary quadratically over the image     (12 params)

    Each is fitted on a random 70% of the triangulated anchors and scored on
    the held-out 30%. With only a few hundred anchors per view, a richer model
    will always fit the training set better; only the held-out score says
    whether the structure is real. If quadratic does not beat constant on test
    error, the residual is not smoothly structured and a finer grid will not
    help either.

Requires refine_depth_v2.py in the same directory (imported for the view
loading, image rebuilding and triangulated anchor collection).

Usage
-----
    python refine_depth_v3.py \
        --run ~/Projects/MDE/DA3_4_1/runs/20260804_092615_018_v2_pad \
        --mode prior
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from refine_depth_v2 import (View, load_views, collect_anchors, bilinear,
                                 ransac_plane, write_ply, CAM_COLOURS, EPS)
except ImportError as exc:
    raise SystemExit(
        "refine_depth_v2.py must sit in the same directory as this script "
        f"({exc})")


# --------------------------------------------------------------------------- #
# correction models, all linear in their coefficients
# --------------------------------------------------------------------------- #

def basis(u: np.ndarray, v: np.ndarray, order: int, w: int, h: int) -> np.ndarray:
    """Polynomial basis over normalised image coordinates."""
    x = 2.0 * u / max(w - 1, 1) - 1.0
    y = 2.0 * v / max(h - 1, 1) - 1.0
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack(cols, axis=1)


def design(inv_raw: np.ndarray, u: np.ndarray, v: np.ndarray, order: int,
           w: int, h: int) -> np.ndarray:
    """
    Model:  inv_true = A(u,v) * inv_raw + B(u,v)
    with A and B expanded in the same polynomial basis. Linear in coefficients,
    so this is an ordinary least-squares design matrix.
    """
    B = basis(u, v, order, w, h)
    return np.concatenate([B * inv_raw[:, None], B], axis=1)


def fit_irls(X: np.ndarray, y: np.ndarray, huber: float, iters: int = 25,
             ridge: float = 0.0, n_basis: int = 1) -> np.ndarray:
    """
    Robust least squares with optional ridge on the NON-constant coefficients.

    Column layout is [A-basis * inv_raw | B-basis], so index 0 and index
    n_basis are the constant terms of A and B. Penalising everything else
    pulls the fit back toward a plain global affine wherever the anchors do
    not constrain the higher-order terms, which is what stops the surface
    exploding away from the anchor clusters.
    """
    wt = np.ones_like(y)
    coef = np.zeros(X.shape[1])
    pen = np.ones(X.shape[1])
    pen[0] = 0.0
    if n_basis < X.shape[1]:
        pen[n_basis] = 0.0
    P = np.diag(np.sqrt(ridge) * pen) if ridge > 0 else None
    for _ in range(iters):
        W = wt[:, None]
        Xa, ya = X * W, y * wt
        if P is not None:
            Xa = np.concatenate([Xa, P], axis=0)
            ya = np.concatenate([ya, np.zeros(X.shape[1])])
        try:
            coef, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        except np.linalg.LinAlgError:
            break
        r = X @ coef - y
        s = max(huber, 1.4826 * np.median(np.abs(r - np.median(r))))
        wt = np.where(np.abs(r) <= s, 1.0, s / np.maximum(np.abs(r), EPS))
    return coef


def apply_model(view: View, coef: np.ndarray, order: int) -> np.ndarray:
    h, w = view.inv_raw.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    X = design(view.inv_raw.reshape(-1), uu.reshape(-1), vv.reshape(-1),
               order, w, h)
    inv = (X @ coef).reshape(h, w)
    return np.where(inv > EPS, 1.0 / np.maximum(inv, EPS), np.nan)


def depth_error_mm(X: np.ndarray, coef: np.ndarray, z_true: np.ndarray
                   ) -> Tuple[float, float]:
    inv = X @ coef
    z = 1.0 / np.maximum(inv, EPS)
    bad = ~np.isfinite(z) | (inv <= EPS)
    err = np.abs(z - z_true)
    err[bad] = np.inf
    finite = np.isfinite(err)
    if finite.sum() == 0:
        return float("inf"), float("inf")
    return (float(np.median(err[finite]) * 1000),
            float(np.percentile(err[finite], 90) * 1000))


# --------------------------------------------------------------------------- #
# properly anchored surface metric
# --------------------------------------------------------------------------- #

def common_surface_metric(views: List[View], ref_idx: int, zmin: float,
                          zmax: float, band: float = 0.10) -> Dict:
    """
    Fit one plane on the reference camera, then measure how every camera's
    points sit relative to that single plane. Points further than `band` from
    it are ignored, so the comparison stays on one physical surface.
    """
    pr_ref, _ = views[ref_idx].cloud_ref(zmin, zmax)
    pl = ransac_plane(pr_ref)
    if pl is None:
        return {"error": "no plane on the reference camera"}
    n = np.asarray(pl["normal"], dtype=np.float64)
    d = float(pl["offset_m"])

    per_cam = []
    for v in views:
        pr, _ = v.cloud_ref(zmin, zmax)
        if pr.shape[0] < 100:
            continue
        sd = pr @ n + d
        near = np.abs(sd) < band
        if near.sum() < 100:
            per_cam.append({"camera": v.name, "n_near": int(near.sum()),
                            "note": "too few points near the reference plane"})
            continue
        per_cam.append({
            "camera": v.name,
            "n_near": int(near.sum()),
            "median_signed_offset_mm": float(np.median(sd[near]) * 1000),
            "rms_about_plane_mm": float(np.sqrt((sd[near] ** 2).mean()) * 1000),
        })
    out = {"reference": views[ref_idx].name, "plane_normal": n.tolist(),
           "plane_offset_m": d, "per_camera": per_cam}
    offs = [c["median_signed_offset_mm"] for c in per_cam
            if "median_signed_offset_mm" in c]
    if len(offs) >= 2:
        out["offset_spread_mm"] = float(max(offs) - min(offs))
    rmss = [c["rms_about_plane_mm"] for c in per_cam if "rms_about_plane_mm" in c]
    if rmss:
        out["mean_rms_about_plane_mm"] = float(np.mean(rmss))
    return out


def cross_view(views: List[View], zmin: float, zmax: float, stride: int) -> Dict:
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
            pair.append({"pair": [vi.name, vj.name],
                         "median_abs_residual_mm": float(np.median(diff[g]) * 1000)})
    out = {"pairs": pair}
    if pair:
        out["overall_median_mm"] = float(
            np.median([p["median_abs_residual_mm"] for p in pair]))
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-validated spatially-varying depth correction.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mode", default="prior", choices=["prior", "noprior"])
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_1/results"))
    ap.add_argument("--ref", default="center", help="camera for the shared plane")
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=12.0)
    ap.add_argument("--n-features", type=int, default=40000)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--min-parallax", type=float, default=1.0)
    ap.add_argument("--max-ray-gap", type=float, default=0.03)
    ap.add_argument("--max-reproj", type=float, default=2.0)
    ap.add_argument("--huber", type=float, default=0.005)
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="penalty on non-constant coefficients; higher decays the "
                         "fit toward a plain global affine where anchors are thin")
    ap.add_argument("--cv-blocks", type=int, default=3,
                    help="grid size for the spatial holdout (3 -> 9 blocks)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--n-splits", type=int, default=5,
                    help="repeat the split this many times and average")
    ap.add_argument("--eval-stride", type=int, default=17)
    ap.add_argument("--plane-band", type=float, default=0.10)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--model", choices=["auto", "constant", "linear", "quadratic"],
                    default="auto", help="which model to apply for the output")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (args.run / args.mode / "refined_v3")
    out.mkdir(parents=True, exist_ok=True)

    views = load_views(args.run, args.mode, args.calib_dir)
    names = [v.name for v in views]
    ref_idx = names.index(args.ref) if args.ref in names else 0
    print(f"loaded {len(views)} views; shared-plane reference = {names[ref_idx]}")

    anchor_args = SimpleNamespace(
        n_features=args.n_features, ratio=args.ratio,
        min_parallax=args.min_parallax, max_ray_gap=args.max_ray_gap,
        max_reproj=args.max_reproj, min_depth=args.min_depth,
        max_depth=args.max_depth)
    anchors = collect_anchors(views, anchor_args)

    # ---- model selection by held-out error --------------------------------- #
    orders = {"constant": 0, "linear": 1, "quadratic": 2}
    n_basis_of = {0: 1, 1: 3, 2: 6}
    rng = np.random.default_rng(0)
    print("\nheld-out anchor error, median mm")
    print("  RANDOM split measures interpolation between anchor clusters.")
    print("  SPATIAL split holds out whole image blocks and measures whether the")
    print("  model generalises to regions with no anchors, which is what applying")
    print("  it image-wide actually requires. Trust the spatial column.")
    print(f"\n  {'camera':>8s} {'n':>6s} | {'--- random ---':>32s} | "
          f"{'--- spatial ---':>32s}")
    print(f"  {'':>8s} {'':>6s} | {'const':>10s} {'linear':>10s} {'quad':>10s} | "
          f"{'const':>10s} {'linear':>10s} {'quad':>10s}")

    cv_table: Dict[str, Dict[str, float]] = {}
    for i, v in enumerate(views):
        A = anchors[i]
        n = len(A["z"])
        if n < 40:
            print(f"  {v.name:>8s} {n:6d}   too few anchors to cross-validate")
            continue
        h, w = v.inv_raw.shape
        y = 1.0 / np.maximum(A["z"], EPS)

        def score(tr, te, order):
            Xtr = design(A["inv_raw"][tr], A["u"][tr], A["v"][tr], order, w, h)
            Xte = design(A["inv_raw"][te], A["u"][te], A["v"][te], order, w, h)
            if Xtr.shape[0] <= Xtr.shape[1] + 2 or len(te) < 5:
                return float("inf")
            coef = fit_irls(Xtr, y[tr], args.huber, ridge=args.ridge,
                            n_basis=n_basis_of[order])
            med, _ = depth_error_mm(Xte, coef, A["z"][te])
            return med

        rand_s = {k: [] for k in orders}
        for _ in range(args.n_splits):
            perm = rng.permutation(n)
            n_test = max(10, int(round(args.test_frac * n)))
            te, tr = perm[:n_test], perm[n_test:]
            for key, order in orders.items():
                rand_s[key].append(score(tr, te, order))

        nb = args.cv_blocks
        bx = np.clip((A["u"] / (w / nb)).astype(int), 0, nb - 1)
        by = np.clip((A["v"] / (h / nb)).astype(int), 0, nb - 1)
        blk = by * nb + bx
        spat_s = {k: [] for k in orders}
        for b in np.unique(blk):
            te = np.where(blk == b)[0]
            tr = np.where(blk != b)[0]
            if len(te) < 5 or len(tr) < 40:
                continue
            for key, order in orders.items():
                spat_s[key].append(score(tr, te, order))

        cv = {f"random_{k}": float(np.median(vv)) for k, vv in rand_s.items()}
        cv.update({f"spatial_{k}": (float(np.median(vv)) if vv else float("nan"))
                   for k, vv in spat_s.items()})
        cv_table[v.name] = cv
        print(f"  {v.name:>8s} {n:6d} | {cv['random_constant']:10.1f} "
              f"{cv['random_linear']:10.1f} {cv['random_quadratic']:10.1f} | "
              f"{cv['spatial_constant']:10.1f} {cv['spatial_linear']:10.1f} "
              f"{cv['spatial_quadratic']:10.1f}")

    if cv_table:
        agg = {k: float(np.nanmean([cv[f"spatial_{k}"] for cv in cv_table.values()]))
               for k in orders}
        agg_r = {k: float(np.mean([cv[f"random_{k}"] for cv in cv_table.values()]))
                 for k in orders}
        print(f"  {'MEAN':>8s} {'':6s} | {agg_r['constant']:10.1f} "
              f"{agg_r['linear']:10.1f} {agg_r['quadratic']:10.1f} | "
              f"{agg['constant']:10.1f} {agg['linear']:10.1f} {agg['quadratic']:10.1f}")
        chosen = min(agg, key=agg.get) if args.model == "auto" else args.model
        print(f"\nselected model (on SPATIAL cv): {chosen}")
        if args.model == "auto" and agg["quadratic"] >= agg["constant"] * 0.95:
            print("  NOTE: the richer models do not beat a constant correction when whole\n"
                  "        image regions are held out. The structure visible under a random\n"
                  "        split is local to the anchor clusters and does not generalise, so\n"
                  "        no post-hoc fit over these anchors will fix the image as a whole.\n"
                  "        Either densify the correspondences or change the depth source.")
    else:
        chosen = "constant"

    # ---- baseline metrics before any correction ---------------------------- #
    for v in views:
        v.a, v.b = 1.0, 0.0
    before_plane = common_surface_metric(views, ref_idx, args.min_depth,
                                         args.max_depth, args.plane_band)
    before_cv = cross_view(views, args.min_depth, args.max_depth, args.eval_stride)

    # ---- fit the chosen model on all anchors and apply --------------------- #
    order = orders[chosen]
    params, refined_depth = [], {}
    print(f"\nfitting {chosen} model on all anchors")
    for i, v in enumerate(views):
        A = anchors[i]
        if len(A["z"]) < 40:
            params.append({"camera": v.name, "n_anchors": len(A["z"]),
                           "note": "uncorrected"})
            refined_depth[v.name] = v.depth_raw.copy()
            continue
        h, w = v.inv_raw.shape
        y = 1.0 / np.maximum(A["z"], EPS)
        X = design(A["inv_raw"], A["u"], A["v"], order, w, h)
        coef = fit_irls(X, y, args.huber, ridge=args.ridge,
                        n_basis={0: 1, 1: 3, 2: 6}[order])
        med, p90 = depth_error_mm(X, coef, A["z"])
        d_new = apply_model(v, coef, order)
        refined_depth[v.name] = d_new
        # keep the constant-equivalent a,b so the evaluation helpers still work
        v.a, v.b = 1.0, 0.0
        v.inv_raw = np.where(d_new > EPS, 1.0 / np.maximum(d_new, EPS), np.nan)
        v.depth_raw = d_new
        print(f"  {v.name:>8s} n={len(A['z']):5d}  train anchor err "
              f"{med:7.1f} mm (p90 {p90:7.1f})  "
              f"median depth -> {np.nanmedian(d_new):6.3f} m")
        params.append({"camera": v.name, "model": chosen, "order": order,
                       "coefficients": coef.tolist(), "n_anchors": int(len(A["z"])),
                       "train_median_mm": med, "train_p90_mm": p90})

    after_plane = common_surface_metric(views, ref_idx, args.min_depth,
                                        args.max_depth, args.plane_band)
    after_cv = cross_view(views, args.min_depth, args.max_depth, args.eval_stride)

    # ---- report ------------------------------------------------------------ #
    print(f"\nshared-plane metric, all cameras against {names[ref_idx]}'s plane")
    print(f"  {'camera':>8s} {'offset mm':>12s} {'rms mm':>10s}   (before -> after)")
    bmap = {c["camera"]: c for c in before_plane.get("per_camera", [])}
    for c in after_plane.get("per_camera", []):
        b = bmap.get(c["camera"], {})
        if "median_signed_offset_mm" in c and "median_signed_offset_mm" in b:
            print(f"  {c['camera']:>8s} {b['median_signed_offset_mm']:+8.1f} -> "
                  f"{c['median_signed_offset_mm']:+8.1f} "
                  f"{b['rms_about_plane_mm']:8.1f} -> {c['rms_about_plane_mm']:6.1f}")
    if "offset_spread_mm" in after_plane:
        print(f"  offset spread {before_plane.get('offset_spread_mm', float('nan')):.1f}"
              f" -> {after_plane['offset_spread_mm']:.1f} mm")
    if "mean_rms_about_plane_mm" in after_plane:
        print(f"  mean rms about plane "
              f"{before_plane.get('mean_rms_about_plane_mm', float('nan')):.1f}"
              f" -> {after_plane['mean_rms_about_plane_mm']:.1f} mm")

    print("\ncross-view residual, median mm")
    bp = {tuple(p["pair"]): p for p in before_cv.get("pairs", [])}
    for p in after_cv.get("pairs", []):
        was = bp.get(tuple(p["pair"]), {}).get("median_abs_residual_mm", float("nan"))
        print(f"  {p['pair'][0]:>7s} <-> {p['pair'][1]:<7s} "
              f"{p['median_abs_residual_mm']:8.1f}   (was {was:8.1f})")
    if "overall_median_mm" in after_cv:
        print(f"  overall {before_cv.get('overall_median_mm', float('nan')):.1f}"
              f" -> {after_cv['overall_median_mm']:.1f} mm")

    # ---- write ------------------------------------------------------------- #
    all_pts, all_cc = [], []
    for v in views:
        np.save(out / f"depth_{v.name}_refined.npy", refined_depth[v.name])
        pr, keep = v.cloud_ref(args.min_depth, args.max_depth)
        write_ply(out / f"cam_{v.name}_refined.ply", pr, v.image.reshape(-1, 3)[keep])
        all_pts.append(pr)
        all_cc.append(np.tile(np.array(CAM_COLOURS.get(v.name, (200, 200, 200)),
                                       np.uint8), (pr.shape[0], 1)))
    pts = np.concatenate(all_pts, 0)
    cc = np.concatenate(all_cc, 0)
    if args.voxel > 0:
        keys = np.floor(pts / args.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx.sort()
        pts, cc = pts[idx], cc[idx]
    write_ply(out / "fused_refined_colourcoded.ply", pts, cc)

    (out / "refine_report.json").write_text(json.dumps({
        "run": str(args.run), "mode": args.mode, "chosen_model": chosen,
        "cross_validation_median_mm": cv_table,
        "parameters": params,
        "before": {"plane": before_plane, "cross_view": before_cv},
        "after": {"plane": after_plane, "cross_view": after_cv},
    }, indent=2, default=float))

    print(f"\nwrote {out}/fused_refined_colourcoded.ply")
    print(f"      {out}/refine_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())