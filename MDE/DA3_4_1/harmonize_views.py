#!/usr/bin/env python3
"""
harmonize_views.py
==================

Make the four cameras' depth maps AGREE WITH EACH OTHER, densely, with the
calibrated extrinsics held fixed.

This optimises consistency, not accuracy. Views that agree can still share a
bias, and this procedure will not remove one. What it does remove is the
per-pixel disagreement that makes the fused cloud look layered and that breaks
clustering and plane fitting downstream.

WHY THE EARLIER ATTEMPTS FAILED
-------------------------------
  v1   dense consistency alone, no image evidence -> collapsed to infinity,
       because distant depth maps agree trivially.
  v2   global affine from sparse anchors -> fixed the range, left ~100 mm.
  v3   low-order polynomial from sparse anchors -> failed spatial holdout on
       two of four cameras. A few hundred clustered points cannot constrain a
       correction over a 504x504 grid.

Consistency is a dense per-pixel property, so the correction must be dense.

METHOD
------
Initialise with the global affine fitted to triangulated anchors, then iterate:

  1. RENDER. For each target view i, project every other view's current cloud
     into i and z-buffer it, giving that view's opinion of depth at each pixel
     of i, with occlusion handled.
  2. CONSENSUS. Combine those opinions with view i's own depth using a robust
     weighted mean in inverse depth. Pixels seen by more cameras carry more
     weight.
  3. UPDATE. Step part-way toward the consensus (--step), never all the way,
     so the system relaxes rather than oscillates.
  4. SMOOTH. Gaussian-smooth the correction field via normalised convolution,
     which both regularises it and carries corrections into regions no other
     camera observes. This is the part a sparse fit cannot do.
  5. RE-ANCHOR. Refit two parameters (scale and shift in inverse depth) against
     the triangulated anchors. This holds the absolute scale and is what stops
     the consensus drifting to infinity. Two parameters against several hundred
     anchors cannot overfit.

Per-iteration cross-view residual and anchor error are printed, so the run is
falsifiable: if consistency improves while anchor error stays flat, it is
working; if anchor error climbs, the scale is drifting and --anchor-weight
should go up.

Requires refine_depth_v2.py alongside it.

Usage
-----
    python harmonize_views.py \
        --run ~/Projects/MDE/DA3_4_1/runs/20260804_092615_018_v2_pad \
        --mode prior --iters 20
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
    import cv2
except ImportError:
    raise SystemExit("OpenCV is required")

try:
    from refine_depth_v2 import (View, load_views, collect_anchors, bilinear,
                                 ransac_plane, write_ply, fit_affine_irls,
                                 CAM_COLOURS, EPS)
except ImportError as exc:
    raise SystemExit(f"refine_depth_v2.py must sit in the same directory ({exc})")


# --------------------------------------------------------------------------- #
# geometry on explicit depth arrays
# --------------------------------------------------------------------------- #

def backproject_ref(view: View, depth: np.ndarray) -> np.ndarray:
    """(H,W) depth -> (H*W,3) points in the reference frame. NaN where invalid."""
    h, w = depth.shape
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    xn = (u - view.K[0, 2]) / view.K[0, 0]
    yn = (v - view.K[1, 2]) / view.K[1, 1]
    pts_rot = np.stack([xn * depth, yn * depth, depth], axis=-1).reshape(-1, 3)
    pts_cam = pts_rot @ view.Rk.T
    return (pts_cam - view.t[None, :]) @ view.R


def render_into(src: View, dst: View, depth_src: np.ndarray,
                zmin: float, zmax: float) -> np.ndarray:
    """
    Render src's depth into dst's image grid with z-buffering.
    Returns a (H,W) depth map in dst's frame, NaN where nothing landed.
    """
    pts = backproject_ref(src, depth_src)
    d = depth_src.reshape(-1)
    ok = np.isfinite(pts).all(1) & np.isfinite(d) & (d > zmin) & (d < zmax)
    if ok.sum() == 0:
        return np.full(dst.depth_raw.shape, np.nan)
    u, v, z = dst.project_ref(pts[ok])
    h, w = dst.depth_raw.shape
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > zmin) & (z < zmax)
    if inb.sum() == 0:
        return np.full((h, w), np.nan)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    # farthest first, so the nearest surface wins the pixel
    order = np.argsort(-z)
    buf = np.full(h * w, np.nan)
    buf[vi[order] * w + ui[order]] = z[order]
    return buf.reshape(h, w)


def smooth_nan(field: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: Gaussian smoothing that ignores and fills NaN."""
    if sigma <= 0:
        return np.where(np.isfinite(field), field, 0.0)
    m = np.isfinite(field).astype(np.float64)
    x = np.where(np.isfinite(field), field, 0.0)
    k = int(max(3, round(sigma * 4)) | 1)
    num = cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(m, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return out


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int,
                  eps: float) -> np.ndarray:
    """
    He et al. guided filter, single-channel guide, implemented with box filters
    so no opencv-contrib is needed.

    Smooths `src` while respecting edges present in `guide`. Used here so the
    depth correction does not bleed across parcel boundaries: an isotropic
    Gaussian spreads the large corrections that belong at an edge over
    everything nearby, which lifts box faces off their true plane.
    """
    if radius < 1:
        return src
    d = 2 * radius + 1
    box = lambda x: cv2.boxFilter(x, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    I = guide.astype(np.float64)
    p = src.astype(np.float64)
    mI, mp = box(I), box(p)
    mIp = box(I * p)
    cov = mIp - mI * mp
    mII = box(I * I)
    var = mII - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def flying_pixel_mask(depth: np.ndarray, thresh: float) -> np.ndarray:
    """
    True where a pixel is NOT a flying pixel. Monocular depth interpolates
    across depth discontinuities, producing sheets of points spanning the gap
    between a parcel edge and the surface behind it. These are artefacts of
    the prior, not measurements, and should be discarded rather than corrected.
    """
    if thresh <= 0:
        return np.ones_like(depth, dtype=bool)
    d = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    local_range = cv2.dilate(d, k) - cv2.erode(d, k)
    return np.isfinite(depth) & (local_range < thresh)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def anchor_error(view: View, depth: np.ndarray, A: Dict) -> Optional[float]:
    if len(A["z"]) == 0:
        return None
    inv = np.where(depth > EPS, 1.0 / np.maximum(depth, EPS), np.nan)
    z = 1.0 / np.maximum(bilinear(inv, A["u"], A["v"]), EPS)
    err = np.abs(z - A["z"])
    g = np.isfinite(err)
    return float(np.median(err[g]) * 1000) if g.any() else None


def cross_view_stats(views: List[View], cur: Dict[str, np.ndarray],
                     zmin: float, zmax: float, stride: int) -> Tuple[float, List[Dict]]:
    pairs = []
    for i, j in itertools.combinations(range(len(views)), 2):
        vi, vj = views[i], views[j]
        pts = backproject_ref(vi, cur[vi.name])
        d = cur[vi.name].reshape(-1)
        ok = np.isfinite(pts).all(1) & (d > zmin) & (d < zmax)
        pts = pts[ok][::stride]
        if pts.shape[0] < 50:
            continue
        u, v, z = vj.project_ref(pts)
        front = z > zmin
        inv_j = np.where(cur[vj.name] > EPS,
                         1.0 / np.maximum(cur[vj.name], EPS), np.nan)
        zj = 1.0 / np.maximum(bilinear(inv_j, u[front], v[front]), EPS)
        diff = np.abs(z[front] - zj)
        g = np.isfinite(diff)
        if g.sum() > 50:
            pairs.append({"pair": [vi.name, vj.name],
                          "median_mm": float(np.median(diff[g]) * 1000),
                          "p90_mm": float(np.percentile(diff[g], 90) * 1000)})
    overall = float(np.median([p["median_mm"] for p in pairs])) if pairs else float("nan")
    return overall, pairs


def reanchor(depth: np.ndarray, A: Dict, huber: float, weight: float,
             prev: np.ndarray) -> np.ndarray:
    """Refit scale and shift in inverse depth against the triangulated anchors."""
    if len(A["z"]) < 20:
        return depth
    inv = np.where(depth > EPS, 1.0 / np.maximum(depth, EPS), np.nan)
    inv_at = bilinear(inv, A["u"], A["v"])
    m = np.isfinite(inv_at) & (inv_at > EPS)
    if m.sum() < 20:
        return depth
    a, b, _ = fit_affine_irls(inv_at[m], A["z"][m], huber)
    inv_new = a * inv + b
    fixed = np.where(inv_new > EPS, 1.0 / np.maximum(inv_new, EPS), np.nan)
    if weight >= 1.0:
        return fixed
    inv_blend = (weight * np.where(np.isfinite(inv_new), inv_new, 0.0) +
                 (1 - weight) * np.where(np.isfinite(inv), inv, 0.0))
    return np.where(inv_blend > EPS, 1.0 / np.maximum(inv_blend, EPS), np.nan)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dense multi-view consistency refinement, poses frozen.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mode", default="prior", choices=["prior", "noprior"])
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_1/results"))
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=12.0)

    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--step", type=float, default=0.5,
                    help="fraction of the way to the consensus per iteration")
    ap.add_argument("--smooth-sigma", type=float, default=12.0,
                    help="pixels; larger spreads corrections further into "
                         "unobserved regions and resists noise")
    ap.add_argument("--max-delta", type=float, default=0.15,
                    help="clamp on the inverse-depth correction per iteration")
    ap.add_argument("--own-weight", type=float, default=1.0,
                    help="weight of a view's own depth in a mean consensus")
    ap.add_argument("--consensus", choices=["median", "mean"], default="median",
                    help="median resists occlusion contamination; mean does not")
    ap.add_argument("--max-disagree", type=float, default=0.20,
                    help="metres; contributions differing from a view's own depth "
                         "by more than this are treated as occlusion and rejected")
    ap.add_argument("--min-support", type=int, default=1,
                    help="other views that must agree before a pixel is corrected")
    ap.add_argument("--guided-radius", type=int, default=8,
                    help="guided-filter radius in pixels; 0 disables and falls "
                         "back to plain Gaussian smoothing of the correction")
    ap.add_argument("--guided-eps", type=float, default=1e-4,
                    help="guided-filter regularisation; smaller follows image "
                         "edges more aggressively")
    ap.add_argument("--edge-thresh", type=float, default=0.05,
                    help="metres of local depth range above which a pixel is "
                         "discarded as a flying pixel; 0 keeps everything")
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="1.0 fully re-imposes the anchor scale each iteration")
    ap.add_argument("--anchor-every", type=int, default=1)

    ap.add_argument("--n-features", type=int, default=40000)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--min-parallax", type=float, default=1.0)
    ap.add_argument("--max-ray-gap", type=float, default=0.03)
    ap.add_argument("--max-reproj", type=float, default=2.0)
    ap.add_argument("--huber", type=float, default=0.005)

    ap.add_argument("--eval-stride", type=int, default=17)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (args.run / args.mode / "harmonized")
    out.mkdir(parents=True, exist_ok=True)

    views = load_views(args.run, args.mode, args.calib_dir)
    print(f"loaded {len(views)} views from {args.run / args.mode}")

    anchors = collect_anchors(views, SimpleNamespace(
        n_features=args.n_features, ratio=args.ratio,
        min_parallax=args.min_parallax, max_ray_gap=args.max_ray_gap,
        max_reproj=args.max_reproj, min_depth=args.min_depth,
        max_depth=args.max_depth))

    # ---- initialise with the anchor-fitted global affine ------------------- #
    cur: Dict[str, np.ndarray] = {}
    print("\ninitial global affine from anchors")
    for i, v in enumerate(views):
        A = anchors[i]
        if len(A["z"]) >= 20:
            a, b, st = fit_affine_irls(A["inv_raw"], A["z"], args.huber)
            inv = a * v.inv_raw + b
            d = np.where(inv > EPS, 1.0 / np.maximum(inv, EPS), np.nan)
            print(f"  {v.name:>8s} a={a:7.4f} b={b:+8.5f}  n={st['n']:4d}  "
                  f"anchor err {st['median_abs_depth_error_mm']:6.1f} mm  "
                  f"median {np.nanmedian(d):.3f} m")
        else:
            d = v.depth_raw.copy()
            print(f"  {v.name:>8s} too few anchors, left as is")
        cur[v.name] = d

    base = {k: val.copy() for k, val in cur.items()}
    ov, pairs0 = cross_view_stats(views, cur, args.min_depth, args.max_depth,
                                  args.eval_stride)
    a0 = {v.name: anchor_error(v, cur[v.name], anchors[i])
          for i, v in enumerate(views)}
    print(f"\nstart   cross-view {ov:7.1f} mm    anchor " +
          "  ".join(f"{k}={vv:.0f}" for k, vv in a0.items() if vv is not None))

    # ---- iterate ----------------------------------------------------------- #
    history = []
    for it in range(1, args.iters + 1):
        rendered = {v.name: [] for v in views}
        for src, dst in itertools.permutations(range(len(views)), 2):
            r = render_into(views[src], views[dst], cur[views[src].name],
                            args.min_depth, args.max_depth)
            rendered[views[dst].name].append(r)

        for v in views:
            own = cur[v.name]
            inv_own = np.where(own > EPS, 1.0 / np.maximum(own, EPS), np.nan)

            # Reject contributions that disagree wildly with this view's own
            # depth: at a parcel edge, another camera sees the surface BEHIND
            # the box, and averaging that in drags the box face off its plane.
            # Large disagreement means occlusion, not error.
            stack_z = np.stack(rendered[v.name], axis=0) if rendered[v.name] \
                else np.zeros((0,) + own.shape)
            if stack_z.shape[0]:
                agree = (np.isfinite(stack_z) &
                         (np.abs(stack_z - own[None, ...]) < args.max_disagree))
                inv_stack = np.where(agree, 1.0 / np.maximum(stack_z, EPS), np.nan)
                support = np.isfinite(inv_stack).sum(axis=0)
            else:
                inv_stack = np.zeros((0,) + own.shape)
                support = np.zeros_like(own, dtype=int)

            allinv = np.concatenate([inv_own[None, ...], inv_stack], axis=0)
            with np.errstate(invalid="ignore"):
                if args.consensus == "median":
                    consensus = np.nanmedian(allinv, axis=0)
                else:
                    consensus = np.nanmean(allinv, axis=0)

            delta = consensus - inv_own
            delta = np.where(support >= args.min_support, delta, np.nan)

            # fill holes, then re-sharpen against the image so the correction
            # does not cross parcel boundaries
            delta = smooth_nan(delta, args.smooth_sigma)
            if args.guided_radius > 0:
                gray = cv2.cvtColor(v.image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
                delta = guided_filter(gray, delta, args.guided_radius, args.guided_eps)
            delta = np.clip(delta, -args.max_delta, args.max_delta)

            inv_new = inv_own + args.step * delta
            d_new = np.where(inv_new > EPS, 1.0 / np.maximum(inv_new, EPS), np.nan)
            d_new = np.where((d_new > args.min_depth) & (d_new < args.max_depth),
                             d_new, np.nan)
            cur[v.name] = d_new

        if args.anchor_every > 0 and it % args.anchor_every == 0:
            for i, v in enumerate(views):
                cur[v.name] = reanchor(cur[v.name], anchors[i], args.huber,
                                       args.anchor_weight, base[v.name])

        ov, pairs = cross_view_stats(views, cur, args.min_depth, args.max_depth,
                                     args.eval_stride)
        ae = {v.name: anchor_error(v, cur[v.name], anchors[i])
              for i, v in enumerate(views)}
        ae_med = np.median([x for x in ae.values() if x is not None])
        history.append({"iter": it, "cross_view_median_mm": ov,
                        "anchor_median_mm": float(ae_med), "pairs": pairs})
        print(f"iter {it:3d}   cross-view {ov:7.1f} mm    anchor median "
              f"{ae_med:6.1f} mm")

    # ---- final report ------------------------------------------------------ #
    print("\nper-pair cross-view residual, median mm")
    p0 = {tuple(p["pair"]): p for p in pairs0}
    for p in history[-1]["pairs"] if history else []:
        was = p0.get(tuple(p["pair"]), {}).get("median_mm", float("nan"))
        print(f"  {p['pair'][0]:>7s} <-> {p['pair'][1]:<7s} "
              f"{p['median_mm']:8.1f}   (was {was:8.1f})")

    print("\nfinal per-camera state")
    for i, v in enumerate(views):
        print(f"  {v.name:>8s} median depth {np.nanmedian(cur[v.name]):6.3f} m   "
              f"anchor err {anchor_error(v, cur[v.name], anchors[i]):6.1f} mm")

    # ---- write ------------------------------------------------------------- #
    all_pts, all_cc, all_rgb = [], [], []
    n_dropped = 0
    for v in views:
        d = cur[v.name]
        np.save(out / f"depth_{v.name}_harmonized.npy", d)
        pts = backproject_ref(v, d)
        dflat = d.reshape(-1)
        keep = (np.isfinite(pts).all(1) & np.isfinite(dflat) &
                (dflat > args.min_depth) & (dflat < args.max_depth))
        fp = flying_pixel_mask(d, args.edge_thresh).reshape(-1)
        n_dropped += int((keep & ~fp).sum())
        keep = keep & fp
        pr = pts[keep]
        rgb = v.image.reshape(-1, 3)[keep]
        write_ply(out / f"cam_{v.name}_harmonized.ply", pr, rgb)
        all_pts.append(pr)
        all_rgb.append(rgb)
        all_cc.append(np.tile(np.array(CAM_COLOURS.get(v.name, (200, 200, 200)),
                                       np.uint8), (pr.shape[0], 1)))
    if args.edge_thresh > 0:
        print(f"\ndiscarded {n_dropped} flying pixels at depth discontinuities")

    pts = np.concatenate(all_pts, 0)
    rgb = np.concatenate(all_rgb, 0)
    cc = np.concatenate(all_cc, 0)
    if args.voxel > 0:
        keys = np.floor(pts / args.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx.sort()
        pts, rgb, cc = pts[idx], rgb[idx], cc[idx]
    write_ply(out / "fused_harmonized.ply", pts, rgb)
    write_ply(out / "fused_harmonized_colourcoded.ply", pts, cc)

    (out / "harmonize_report.json").write_text(json.dumps({
        "run": str(args.run), "mode": args.mode,
        "settings": {"iters": args.iters, "step": args.step,
                     "smooth_sigma": args.smooth_sigma,
                     "anchor_weight": args.anchor_weight},
        "start": {"cross_view_median_mm": float(ov) if not history else None,
                  "pairs": pairs0},
        "history": history,
    }, indent=2, default=float))

    print(f"\nwrote {out}/fused_harmonized.ply  (true colour)")
    print(f"      {out}/fused_harmonized_colourcoded.ply  (per-camera colour)")
    print(f"      {out}/harmonize_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())