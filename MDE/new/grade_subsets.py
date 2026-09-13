#!/usr/bin/env python3
"""
grade_subsets.py

Score every camera-subset / process-res run against one ground-truth cloud, in
the currency the pipeline itself uses: height above the deck.

WHAT THIS MEASURES, AND WHY IN THIS FORM
----------------------------------------
A chamfer distance between two clouds answers the wrong question. The pipeline
does not consume points, it consumes a deck plane and the height of parcel tops
above it, and its error is known to be a LOSS OF RELIEF: a rise of h comes back
as p*h + q with p near 0.85 on this rig. So the primary metric here is that same
line, refitted per run against the ground truth:

    h_recon = p * h_gt + q

p is the surviving fraction of relief, q is the offset at the deck, and the
residual about the line is what neither a scale nor an offset can remove. Those
three numbers are directly comparable to the `shared` block of
depth_correction.json, so a subset can be judged against the four-camera
correction that was actually fitted rather than against an abstract distance.

Chamfer accuracy and completeness are reported too, but as secondary columns.
They conflate an in-plane registration error with a height error and cannot tell
a compressed reconstruction from a noisy one.

THE REGISTRATION IS SOLVED ONCE AND FROZEN
------------------------------------------
Registering each run to the ground truth separately would let a rigid offset --
which is part of the error -- be absorbed into the alignment, and every subset
would score the same. The transform is therefore solved on ONE reference run
(the quad at the highest resolution, by default), written to gt_align.json, and
reused unchanged for every other run. Re-solving it is an explicit flag.

The alignment is rigid, 6-DoF, never Sim(3). Allowing scale in the alignment
would remove exactly the quantity p is there to measure.

INPUTS
------
Run directories written by da3_offline.py, each holding depth_<cam>.npy,
K_<cam>.npy, E_<cam>.npy and fused.ply, and optionally cloud_<cam>.ply, which is
what makes the per-view layering column possible.

The ground-truth cloud may be in its own frame. It is registered by fitting the
deck plane in both, matching the belt rectangle on that plane, and refining with
point-to-plane ICP over four orientation hypotheses. Pass --gt-in-rig-frame if
it was captured through this rig's calibration and only needs the refine.

EXAMPLE
-------
    python3 grade_subsets.py \
        --gt gt/pointcloud_20260826_095616.ply \
        --belt belt.json \
        --runs-root runs_da3 \
        --align-from runs_da3/center+left+top+right_res1008 \
        --out grades

Keep this file beside rigkit.py, da3_stream.py and belt.json.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")

# Soft: the layering measure is reused from the streamer so there is one
# definition of it, but the streamer pulls in torch and open3d and a rig that
# cannot import it should still be able to grade.
try:
    from da3_stream import layer_report as _layer_report
except Exception:  # noqa: BLE001
    _layer_report = None


CAMS = ["center", "left", "top", "right"]


# --------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------

def parse_run(path: Path):
    """Subset and process-res from a directory name like center+left_res504.

    The resolution is confirmed against the depth arrays afterwards, because the
    directory name is a label and the array is the measurement.
    """
    name = path.name
    m = re.search(r"_res(\d+)", name)
    res = int(m.group(1)) if m else None
    subset = re.sub(r"_res\d+$", "", name)
    cams = [c for c in subset.replace(",", "+").split("+") if c]
    unknown = [c for c in cams if c not in CAMS]
    if unknown:
        return None
    return {"dir": path, "subset": rigkit.canon(subset), "cams": cams,
            "res_label": res}


def depth_grid(run):
    for c in run["cams"]:
        p = run["dir"] / f"depth_{c}.npy"
        if p.exists():
            a = np.load(p, mmap_mode="r")
            return int(a.shape[-2]), int(a.shape[-1])
    return None, None


def conditioning(cams, centres):
    if len(cams) < 3:
        return 0.0
    P = np.array([centres[c] for c in cams])
    P = P - P.mean(0)
    s = np.linalg.svd(P, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


# --------------------------------------------------------------------------
# deck plane and belt frame
# --------------------------------------------------------------------------

def belt_frame(belt: dict):
    """The frozen belt as an orthonormal frame: along, across, up towards the
    cameras. Using the frozen footprint rather than refitting per run is the
    same argument --seg-belt-file makes: the conveyor does not move, so a
    footprint that moves between runs moves every measurement with it."""
    ex = np.asarray(belt["x_axis"], float)
    ey = np.asarray(belt["y_axis"], float)
    ex /= np.linalg.norm(ex)
    ey = ey - ex * float(ey @ ex)
    ey /= np.linalg.norm(ey)
    n = np.cross(ex, ey)
    n /= np.linalg.norm(n)
    centre = np.asarray(belt["centre_m"], float)
    if float(n @ (np.zeros(3) - centre)) < 0:
        n = -n                     # positive height is towards the cameras
        ey = -ey
    return centre, ex, ey, n


def in_belt(P, centre, ex, ey, belt, margin=0.05):
    rel = P - centre
    return ((np.abs(rel @ ex) <= belt["length_m"] / 2 + margin)
            & (np.abs(rel @ ey) <= belt["width_m"] / 2 + margin))


def largest_plane(P, tol=0.008, trials=400, index=0, seed=0):
    """The index-th largest plane, peeled one at a time.

    A ground-truth scan of a cell usually contains a floor larger than the
    conveyor, so --gt-plane-index exists for the same reason
    --seg-plane-distance does.
    """
    rng = np.random.default_rng(seed)
    Q = P
    keep = np.arange(len(P))
    n = d = None
    for _ in range(index + 1):
        if len(keep) < 200:
            break
        n, d = rigkit.fit_plane_ransac(Q[keep], tol=tol, trials=trials,
                                       seed=int(rng.integers(1 << 30)))
        inl = np.abs(Q[keep] @ n + d) < tol
        if _ == index:
            return n, d, keep[inl]
        keep = keep[~inl]
    return n, d, np.empty(0, dtype=int)


def plane_patch_rect(P, n, d, tol, cell=0.010, close_m=0.30):
    """Rectangle of the largest CONNECTED patch of a plane, not of every point
    that happens to be coplanar with it.

    A ground-truth scan of a cell contains floor, racking and framing that lie
    in or near the conveyor's plane but metres away from it, and a minAreaRect
    over all of them returns a rectangle spanning the room. This is the same
    problem box_segment.belt_from_raster solves for the live belt, and the same
    treatment: rasterise the inliers on the plane, close by close_m so the gaps
    the parcels leave in the deck do not sever it, take the largest connected
    component, and fit the rectangle to the ORIGINAL occupancy inside that
    component so the closing bridges gaps without inflating the extent.
    """
    inl = P[np.abs(P @ n + d) < tol]
    if len(inl) < 200:
        return None
    ex = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(ex) < 1e-6:
        ex = np.cross(n, [1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    u, v = inl @ ex, inl @ ey
    u0, v0 = float(u.min()), float(v.min())
    W = int((u.max() - u0) / cell) + 1
    H = int((v.max() - v0) / cell) + 1
    while W * H > 40_000_000 and cell < 0.1:
        cell *= 2.0
        W = int((u.max() - u0) / cell) + 1
        H = int((v.max() - v0) / cell) + 1
    if W < 3 or H < 3:
        return None
    img = np.zeros((H, W), np.uint8)
    img[np.clip(((v - v0) / cell).astype(np.int64), 0, H - 1),
        np.clip(((u - u0) / cell).astype(np.int64), 0, W - 1)] = 255
    k = max(3, int(round(close_m / cell)))
    k += 1 - (k % 2)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n_lab <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero((lab == biggest) & (img > 0))
    if len(xs) < 100:
        return None
    uv = np.stack([xs * cell + u0, ys * cell + v0], 1) * 1000.0
    (cu, cvv), (w, h), ang = cv2.minAreaRect(uv.astype(np.float32))
    a = np.radians(ang)
    bx = ex * np.cos(a) + ey * np.sin(a)
    by = -ex * np.sin(a) + ey * np.cos(a)
    du, dv = w / 1000.0, h / 1000.0
    if du < dv:
        bx, by, du, dv = by, -bx, dv, du
    centre = ex * (cu / 1000.0) + ey * (cvv / 1000.0) - n * d
    return {"centre": centre, "x": bx, "y": by, "n": n,
            "length": float(du), "width": float(dv),
            "n_cells": int(len(xs)), "n_components": int(n_lab - 1)}


def choose_gt_plane(gt, belt, args):
    """Search the first few planes and keep the one whose largest connected
    patch best matches the frozen belt footprint.

    Scoring against a known rectangle rather than taking the largest plane is
    what stops the floor being fitted as the conveyor, which is the same failure
    --seg-plane-distance exists to prevent in the live segmentation.
    """
    want_L, want_W = belt["length_m"], belt["width_m"]
    Q = gt
    keep = np.arange(len(gt))
    table, best, best_score = [], None, np.inf
    for i in range(max(1, args.gt_plane_search)):
        if len(keep) < 500:
            break
        n, d = rigkit.fit_plane_ransac(Q[keep], tol=args.gt_plane_tol,
                                       trials=400, seed=i)
        inl = np.abs(Q[keep] @ n + d) < args.gt_plane_tol
        rect = plane_patch_rect(Q[keep], n, d, args.gt_plane_tol,
                               close_m=args.gt_close)
        if rect is not None:
            score = abs(rect["length"] - want_L) + abs(rect["width"] - want_W)
            table.append((i, int(inl.sum()), rect, score))
            if score < best_score and args.gt_plane_index is None:
                best, best_score = rect, score
            if args.gt_plane_index == i:
                best, best_score = rect, score
        else:
            table.append((i, int(inl.sum()), None, np.inf))
        keep = keep[~inl]

    print(f"[gt   ] plane search, scored against the frozen belt "
          f"{want_L * 1e3:.0f} x {want_W * 1e3:.0f} mm")
    print(f"[gt   ] {'#':>2s} {'inliers':>9s} {'patch cells':>12s} "
          f"{'length mm':>10s} {'width mm':>10s} {'score m':>9s}")
    for i, ninl, rect, score in table:
        if rect is None:
            print(f"[gt   ] {i:>2d} {ninl:>9d}   no connected patch")
            continue
        mark = " <-- taken" if rect is best else ""
        print(f"[gt   ] {i:>2d} {ninl:>9d} {rect['n_cells']:>12d} "
              f"{rect['length'] * 1e3:>10.0f} {rect['width'] * 1e3:>10.0f} "
              f"{score:>9.3f}{mark}")
    if best is None:
        raise SystemExit("no plane in the ground truth produced a connected "
                         "patch. Raise --gt-plane-search, or loosen "
                         "--gt-plane-tol: a scan resolves the rollers as "
                         "cylinders, so the deck plane is only tangent to them "
                         "and 8 mm may be too tight")
    if best_score > args.gt_plane_accept:
        print(f"[WARN] the best match is still {best_score:.3f} m from the "
              f"frozen belt. Either the ground truth does not contain the "
              f"conveyor, or the belt footprint is stale. Check by eye before "
              f"trusting anything below, or align by hand with --gt-pairs.")
    return best


def rect_transform(src, dst, flip_yaw=False, flip_n=False):
    """Rigid transform taking one plane rectangle onto another."""
    sx, sy, sn = src["x"], src["y"], src["n"]
    if flip_yaw:
        sx, sy = -sx, -sy
    if flip_n:
        sn, sy = -sn, -sy
    A = np.column_stack([sx, sy, sn])
    B = np.column_stack([dst["x"], dst["y"], dst["n"]])
    R = B @ A.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dst["centre"] - R @ src["centre"]
    return T


def kabsch(src, dst):
    """Rigid transform taking src onto dst. No scale, deliberately: a
    similarity fit here would absorb the relief compression the study measures."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def register(gt, ref, belt, args):
    """Ground truth into the rig frame: plane, rectangle, then ICP.

    Rigid only. A Sim(3) fit would absorb the relief compression the whole
    study is about.
    """
    ex_c, ex_x, ex_y, ex_n = belt_frame(belt)
    dst = {"centre": ex_c, "x": ex_x, "y": ex_y, "n": ex_n,
           "length": belt["length_m"], "width": belt["width_m"]}

    if args.gt_pairs:
        pairs = json.loads(Path(args.gt_pairs).read_text())
        a, b = np.asarray(pairs["gt"], float), np.asarray(pairs["rig"], float)
        if len(a) < 3 or len(a) != len(b):
            raise SystemExit("--gt-pairs needs at least three matched points "
                             "in both frames")
        T = kabsch(a, b)
        print(f"[gt   ] hand alignment from {len(a)} point pairs, residual "
              f"{np.sqrt(((apply(a, T) - b) ** 2).sum(1).mean()) * 1e3:.1f} mm")
        return icp_refine(gt, ref, T, args) if o3d is not None else T

    if args.gt_in_rig_frame:
        best = np.eye(4)
    else:
        src = choose_gt_plane(gt, belt, args)
        # One tree, one subsample. Rebuilding it per hypothesis over a
        # million-point scan is what made this appear to hang.
        rng = np.random.default_rng(0)
        ref_s = ref[rng.choice(len(ref), min(len(ref), args.icp_sample),
                               replace=False)]
        gt_s = gt[rng.choice(len(gt), min(len(gt), args.icp_sample),
                             replace=False)]
        tree = cKDTree(ref_s) if cKDTree is not None else None
        best, best_rms = None, np.inf
        for fy, fn in itertools.product((False, True), (False, True)):
            T = rect_transform(src, dst, fy, fn)
            rms = icp_rms(apply(gt_s, T), tree, args)
            print(f"[gt   ]   yaw{'+180' if fy else '   '} "
                  f"normal{'-' if fn else '+'}  {rms * 1e3:8.2f} mm")
            if rms < best_rms:
                best, best_rms = T, rms
        print(f"[gt   ] best of four orientation hypotheses: "
              f"{best_rms * 1e3:.2f} mm before refinement")

    if o3d is not None:
        best = icp_refine(gt, ref, best, args)
    return best


def apply(P, T):
    return P @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def _pcd(P, normals=False, radius=0.03):
    c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P)))
    if normals:
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, 30))
    return c


def icp_rms(P, tree, args):
    if tree is None:
        return np.inf
    d, _ = tree.query(P, k=1)
    d = d[d < args.icp_max_corr]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else np.inf


def icp_refine(gt, ref, T0, args):
    src, dst = _pcd(gt), _pcd(ref, normals=True)
    T = np.asarray(T0, float)
    for tol in (args.icp_max_corr, args.icp_max_corr / 3.0):
        res = o3d.pipelines.registration.registration_icp(
            src, dst, tol, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=args.icp_iters))
        T = res.transformation
    print(f"[gt   ] ICP fitness {res.fitness:.3f}  inlier rms "
          f"{res.inlier_rmse * 1e3:.2f} mm")
    return np.asarray(T, float)


# --------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------

def height_raster(P, centre, ex, ey, n, belt, cell, margin=0.05):
    """Median height above the deck plane per belt cell.

    A raster rather than a nearest-neighbour comparison, because the two clouds
    have different densities and a nearest-neighbour distance between a dense
    scan and a sparse reconstruction reports the density difference as an error.
    A cell median is the same physical quantity in both.
    """
    rel = P - centre
    u, v, h = rel @ ex, rel @ ey, rel @ n
    L, W = belt["length_m"] / 2 + margin, belt["width_m"] / 2 + margin
    ok = (np.abs(u) <= L) & (np.abs(v) <= W)
    if not ok.any():
        return None, None, (0, 0)
    nu, nv = int(2 * L / cell) + 1, int(2 * W / cell) + 1
    iu = np.clip(((u[ok] + L) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(((v[ok] + W) / cell).astype(np.int64), 0, nv - 1)
    key = iu * nv + iv
    order = np.argsort(key, kind="stable")
    key, hs = key[order], h[ok][order]
    edges = np.flatnonzero(np.diff(key)) + 1
    med = np.full(nu * nv, np.nan)
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(key)]):
        med[key[a]] = np.median(hs[a:b])
    return med, ~np.isnan(med), (nu, nv)


def robust_line(x, y, huber=2.0, iters=6):
    """h_recon = p h_gt + q, forward, with the exact quantity as predictor.

    The direction matters for the same reason it does in depth_align.py:
    regressing the known height on the noisy one attenuates the slope, and the
    attenuation on this rig is large enough to turn a compression of 0.17 into
    one of 0.06.
    """
    A = np.stack([x, np.ones_like(x)], 1)
    w = np.ones_like(x)
    p, q = 1.0, 0.0
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        p, q = float(coef[0]), float(coef[1])
        r = np.abs(p * x + q - y)
        s = max(1.4826 * float(np.median(r)), 1e-9)
        w = np.where(r <= huber * s, 1.0, huber * s / np.maximum(r, 1e-12))
    return p, q


def chamfer(P, Q, cap=0.20):
    if cKDTree is None or not len(P) or not len(Q):
        return None, None, None
    d_ac, _ = cKDTree(Q).query(P, k=1)
    d_co, _ = cKDTree(P).query(Q, k=1)
    return (float(np.median(d_ac)), float(np.median(d_co)),
            float(np.mean(d_co < 0.010)))


def multi_view_coverage(run, cams, centre, ex, ey, n, belt, args,
                        gt_top_cells=None):
    """Belt area reached by two or more cameras, not by any one.

    grade_subsets' coverage_frac counts a cell as covered if ANY view puts a
    point in it, which is the wrong test for this pipeline: box_segment needs
    --seg-min-views 2 before it will call a parcel pickable, so a cell only one
    camera can see contributes nothing to a pick. The two figures differ a
    great deal here -- single-view coverage saturates near 99 per cent while
    the projection geometry caps two-view coverage at about 87 per cent of the
    belt -- so quoting the first as though it were the second overstates what
    any configuration delivers.

    Counted per camera from cloud_calib_<cam>.ply, and ONLY over cells where
    the ground truth has a parcel. Counting every cell with a point in the
    height band instead rewards exactly the wrong thing: a run whose deck is
    tilted sweeps that deck up through the band and scores as though it had
    parcels everywhere. On this rig left+top has the worst deck tilt of any
    subset, 1.96 degrees, and by the unrestricted count it looked like the best
    two-view coverage in the table. Restricted to real parcels it cannot.
    """
    grids = []
    for c in cams:
        f = run / f"cloud_calib_{c}.ply"
        if not f.exists():
            return None
        P = rigkit.load_cloud(str(f))
        rel = P - centre
        u, v, h = rel @ ex, rel @ ey, rel @ n
        L, W = belt["length_m"] / 2, belt["width_m"] / 2
        sel = ((np.abs(u) <= L) & (np.abs(v) <= W)
               & (h > args.mv_top_min) & (h < args.mv_top_max))
        if not sel.any():
            grids.append(None)
            continue
        nu = int(2 * L / args.cell) + 1
        nv = int(2 * W / args.cell) + 1
        g = np.zeros(nu * nv, bool)
        iu = np.clip(((u[sel] + L) / args.cell).astype(np.int64), 0, nu - 1)
        iv = np.clip(((v[sel] + W) / args.cell).astype(np.int64), 0, nv - 1)
        g[iu * nv + iv] = True
        grids.append(g)
    nu_ = int(2 * (belt["length_m"] / 2) / args.cell) + 1
    nv_ = int(2 * (belt["width_m"] / 2) / args.cell) + 1
    grids = [g for g in grids if g is not None]
    if not grids:
        return None
    counts = np.sum(np.stack(grids), axis=0)
    cell_m2 = args.cell ** 2
    if gt_top_cells is not None:
        dom = np.zeros_like(counts, dtype=bool)
        idx = gt_top_cells[gt_top_cells < len(dom)]
        dom[idx] = True
        counts = np.where(dom, counts, 0)
        denom = float(int(dom.sum())) * cell_m2
    else:
        denom = round(belt["length_m"] * belt["width_m"], 3)
    return {"mv1_m2": round(float(np.count_nonzero(counts >= 1)) * cell_m2, 3),
            "mv2_m2": round(float(np.count_nonzero(counts >= 2)) * cell_m2, 3),
            "mv3_m2": round(float(np.count_nonzero(counts >= 3)) * cell_m2, 3),
            "belt_m2": round(max(denom, 1e-9), 3),
            "restricted_to_gt_parcels": gt_top_cells is not None,
            "n_views": len(grids)}


def per_view_layering(run, centres, belt, args):
    """Spread across views of the same surface, from cloud_<cam>.ply.

    This is the only error term a per-view correction can move, and the only one
    that should improve with camera count. Reported separately from the height
    error for exactly that reason.
    """
    if _layer_report is None:
        return None
    pts, cc = {}, []
    for c in run["cams"]:
        p = run["dir"] / f"cloud_{c}.ply"
        if not p.exists():
            return None
        pts[c] = rigkit.load_cloud(str(p))
        cc.append(centres[c])
    if len(pts) < 2:
        return None
    rep = _layer_report(pts, np.stack(cc), args.layer_band_mm / 1e3,
                        args.layer_above_mm / 1e3)
    return rep


def grade(run, gt_r, gt_ras, gt_mask, centres, belt, frame, args):
    centre, ex, ey, n = frame
    fused = run["dir"] / args.fused_name
    if not fused.exists():
        print(f"[skip ] {run['dir'].name}: no fused.ply")
        return None
    P = rigkit.load_cloud(str(fused))
    dh, dw = depth_grid(run)

    row = {
        "subset": run["subset"],
        "n_cams": len(run["cams"]),
        "res": dw or run["res_label"],
        "depth_grid": f"{dh}x{dw}" if dh else "",
        "cond": round(conditioning(run["cams"], centres), 4),
        "n_points": len(P),
    }

    # ---- deck plane, in the frozen belt frame -------------------------
    rel = P - centre
    h = rel @ n
    on_belt = in_belt(P, centre, ex, ey, belt)
    deck = on_belt & (np.abs(h) <= args.deck_band)
    if deck.sum() > 500:
        nd, dd = rigkit.fit_plane_svd(P[deck])
        tilt = np.degrees(np.arccos(np.clip(abs(float(nd @ n)), 0, 1)))
        row["deck_offset_mm"] = round(float(np.median(h[deck])) * 1e3, 2)
        row["deck_rms_mm"] = round(
            float(np.sqrt(np.mean(h[deck] ** 2))) * 1e3, 2)
        row["deck_tilt_deg"] = round(float(tilt), 3)

    # ---- height raster against the ground truth ------------------------
    ras, mask, _ = height_raster(P, centre, ex, ey, n, belt, args.cell)
    if ras is None:
        return row
    both = mask & gt_mask
    row["cells_recon"] = int(mask.sum())
    row["coverage_frac"] = round(float(both.sum() / max(gt_mask.sum(), 1)), 4)
    row["coverage_m2"] = round(float(mask.sum()) * args.cell ** 2, 3)
    if both.sum() < 200:
        print(f"[warn ] {run['dir'].name}: only {int(both.sum())} shared cells")
        return row

    hg, hr = gt_ras[both], ras[both]
    err = (hr - hg) * 1e3
    p, q = robust_line(hg, hr, args.huber)
    resid = (hr - (p * hg + q)) * 1e3

    row["p_relief"] = round(float(p), 5)
    row["q_mm"] = round(float(q) * 1e3, 2)
    row["line_resid_rms_mm"] = round(float(np.sqrt(np.mean(resid ** 2))), 2)
    row["err_median_mm"] = round(float(np.median(err)), 2)
    row["err_rms_mm"] = round(float(np.sqrt(np.mean(err ** 2))), 2)
    row["err_p95_mm"] = round(float(np.percentile(np.abs(err), 95)), 2)

    # Parcel tops are the number that matters; the deck is largely definitional
    # once the plane has been fitted to it.
    tops = hg > args.top_height
    if tops.sum() > 50:
        e = err[tops]
        row["n_cells_tops"] = int(tops.sum())
        row["top_err_median_mm"] = round(float(np.median(e)), 2)
        row["top_err_rms_mm"] = round(float(np.sqrt(np.mean(e ** 2))), 2)
        row["top_err_p95_mm"] = round(
            float(np.percentile(np.abs(e), 95)), 2)

    # ---- chamfer, secondary -------------------------------------------
    sel = on_belt
    acc, comp, comp10 = chamfer(P[sel], gt_r[in_belt(gt_r, centre, ex, ey, belt)])
    if acc is not None:
        row["accuracy_mm"] = round(acc * 1e3, 2)
        row["completeness_mm"] = round(comp * 1e3, 2)
        row["complete_at_10mm"] = round(comp10, 4)

    # ---- layering ------------------------------------------------------
    gt_cells = None
    if gt_ras is not None:
        L, W = belt["length_m"] / 2, belt["width_m"] / 2
        nv_m = int(2 * W / args.cell) + 1
        nu_m = int(2 * L / args.cell) + 1
        nv_g = int(2 * (W + args.margin_cell) / args.cell) + 1
        keep = []
        for k in np.flatnonzero(gt_mask & (gt_ras > args.mv_top_min)):
            iu, iv = divmod(int(k), nv_g)
            iu -= int(args.margin_cell / args.cell)
            iv -= int(args.margin_cell / args.cell)
            if 0 <= iu < nu_m and 0 <= iv < nv_m:
                keep.append(iu * nv_m + iv)
        gt_cells = np.asarray(sorted(set(keep)), dtype=np.int64)
    mv = multi_view_coverage(run["dir"], run["cams"], centre, ex, ey, n,
                             belt, args, gt_cells)
    if mv:
        row["mv1_m2"] = mv["mv1_m2"]
        row["mv2_m2"] = mv["mv2_m2"]
        row["mv3_m2"] = mv["mv3_m2"]
        row["mv2_frac"] = round(mv["mv2_m2"] / mv["belt_m2"], 4)
        row["mv1_frac"] = round(mv["mv1_m2"] / mv["belt_m2"], 4)

    # Per-view offsets against the FROZEN belt plane, not against a plane
    # fitted to each run. A per-run fit is only as good as that run's deck, and
    # a run whose deck holds 20 per cent of its points produces a bad plane and
    # therefore bad offsets: the same subset at the same resolution reported
    # 12.9 mm in one arm and 209 mm in the other purely from this. Against a
    # fixed plane the two are comparable.
    off = {}
    for cam in run["cams"]:
        f = run["dir"] / f"cloud_calib_{cam}.ply"
        if not f.exists():
            continue
        Q = rigkit.load_cloud(str(f))
        rel = Q - centre
        on = in_belt(Q, centre, ex, ey, belt)
        hh = rel @ n
        sel = on & (hh > args.mv_top_min) & (hh < args.mv_top_max)
        if int(sel.sum()) >= args.layer_min_points:
            off[cam] = float(np.median(hh[sel]))
    if len(off) > 1:
        v = np.array(list(off.values())) * 1e3
        row["layer_belt_frame_sd_mm"] = round(float(np.std(v, ddof=1)), 2)
        row["layer_belt_frame_range_mm"] = round(float(v.max() - v.min()), 2)
        row["layer_n_views_used"] = len(off)
    row["layer_thin_views"] = len(run["cams"]) - len(off)

    rep = per_view_layering(run, centres, belt, args)
    if rep:
        row["layer_belt_spread_mm"] = rep.get("belt_spread_mm")
        row["layer_above_spread_mm"] = rep.get("above_spread_mm")
        # Max minus min is biased by camera count: the expected RANGE of a
        # sample grows with its size, so a four-camera run is marked down
        # against a two-camera one for having more views rather than for being
        # worse. Layering is the one quantity that should improve with camera
        # count, so it needs a statistic that does not move with n. The
        # standard deviation of the per-view offsets does not.
        for key, tag in (("belt", "belt"), ("above", "above")):
            v = [x for x in (rep.get(key) or {}).values()]
            if len(v) > 1:
                row[f"layer_{tag}_sd_mm"] = round(
                    float(np.std(v, ddof=1)), 2)
                row[f"layer_{tag}_n_views"] = len(v)
    return row


# --------------------------------------------------------------------------

FIELDS = ["subset", "n_cams", "res", "depth_grid", "cond", "n_points",
          "coverage_m2", "coverage_frac", "cells_recon",
          "p_relief", "q_mm", "line_resid_rms_mm",
          "err_median_mm", "err_rms_mm", "err_p95_mm",
          "n_cells_tops", "top_err_median_mm", "top_err_rms_mm",
          "top_err_p95_mm",
          "deck_offset_mm", "deck_rms_mm", "deck_tilt_deg",
          "accuracy_mm", "completeness_mm", "complete_at_10mm",
          "layer_belt_spread_mm", "layer_above_spread_mm",
          "layer_belt_sd_mm", "layer_above_sd_mm", "layer_above_n_views",
          "mv1_m2", "mv2_m2", "mv3_m2", "mv1_frac", "mv2_frac",
          "layer_belt_frame_sd_mm", "layer_belt_frame_range_mm",
          "layer_n_views_used", "layer_thin_views"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grade camera-subset and resolution runs against a "
                    "ground-truth cloud, in height above the deck.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--belt", type=Path, default=Path("belt.json"),
                    help="the frozen conveyor footprint. Every run is measured "
                         "inside the same footprint, so a run that under-"
                         "detects the belt is not also given a smaller area to "
                         "be right about")
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--run", type=Path, action="append", default=None)
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="grade every subdirectory that parses as a subset")
    ap.add_argument("--align-from", type=Path, default=None,
                    help="the run the ground-truth alignment is solved on. "
                         "Defaults to the run with the most cameras at the "
                         "finest grid")
    ap.add_argument("--align-file", type=Path, default=Path("gt_align.json"))
    ap.add_argument("--realign", action="store_true",
                    help="re-solve the alignment and overwrite --align-file")
    ap.add_argument("--gt-in-rig-frame", action="store_true",
                    help="the ground truth was captured through this rig's "
                         "calibration, so only an ICP refine is needed")
    ap.add_argument("--gt-plane-index", type=int, default=None,
                    help="force a plane by rank instead of taking the best "
                         "match to the frozen belt footprint")
    ap.add_argument("--gt-plane-search", type=int, default=6,
                    help="how many planes to peel and score")
    ap.add_argument("--gt-plane-tol", type=float, default=0.012,
                    help="inlier distance for the ground-truth deck plane. A "
                         "scan resolves the rollers as cylinders, so the deck "
                         "plane is tangent to them and a tolerance set for a "
                         "flat belt starves the fit")
    ap.add_argument("--gt-close", type=float, default=0.30,
                    help="gap bridged when finding the deck patch; must exceed "
                         "the widest parcel standing on it")
    ap.add_argument("--gt-plane-accept", type=float, default=0.30,
                    help="total footprint mismatch above which the chosen "
                         "plane is reported as doubtful")
    ap.add_argument("--gt-pairs", type=Path, default=None,
                    help="a json holding {\"gt\": [[x,y,z],...], \"rig\": "
                         "[[x,y,z],...]} of at least three matched points, for "
                         "when the automatic search cannot find the conveyor")
    ap.add_argument("--icp-sample", type=int, default=60000,
                    help="points used when scoring the orientation hypotheses")
    ap.add_argument("--gt-voxel", type=float, default=0.003)
    ap.add_argument("--plane-tol", type=float, default=0.008)
    ap.add_argument("--icp-max-corr", type=float, default=0.05)
    ap.add_argument("--icp-iters", type=int, default=80)
    ap.add_argument("--cell", type=float, default=0.010,
                    help="belt raster cell for the height comparison")
    ap.add_argument("--deck-band", type=float, default=0.015)
    ap.add_argument("--layer-min-points", type=int, default=2000,
                    help="a view needs this many points in the band before its "
                         "median counts. Below it the median is noise: one arm "
                         "had a camera with 483 points near the deck and its "
                         "offset was quoted as though it were a measurement")
    ap.add_argument("--margin-cell", type=float, default=0.05,
                    help="the margin height_raster adds around the belt, "
                         "needed to map its cells onto the unmargined grid the "
                         "two-view count uses")
    ap.add_argument("--mv-top-min", type=float, default=0.060,
                    help="lower height bound for the two-view coverage count. "
                         "Parcel TOPS are what must be seen twice, so the "
                         "count is taken above the deck rather than on it")
    ap.add_argument("--mv-top-max", type=float, default=0.600)
    ap.add_argument("--top-height", type=float, default=0.060,
                    help="ground-truth height above which a cell counts as a "
                         "parcel top rather than deck")
    ap.add_argument("--huber", type=float, default=2.0)
    ap.add_argument("--layer-band-mm", type=float, default=40.0)
    ap.add_argument("--layer-above-mm", type=float, default=60.0)
    ap.add_argument("--fused-name", default="fused.ply",
                    help="the cloud to grade inside each run. Use "
                         "fused_calib.ply after check_fusion.py --rewrite")
    ap.add_argument("--out", type=Path, default=Path("grades"))
    args = ap.parse_args()

    belt = json.loads(args.belt.read_text())
    frame = belt_frame(belt)
    rig = rigkit.Rig(str(args.calib_dir))
    centres = {c: rig.centre(c) for c in rig.cams}

    runs = []
    for d in (args.run or []):
        r = parse_run(Path(d))
        if r:
            runs.append(r)
    if args.runs_root:
        for d in sorted(Path(args.runs_root).iterdir()):
            if d.is_dir():
                r = parse_run(d)
                if r and (d / args.fused_name).exists():
                    runs.append(r)
    if not runs:
        raise SystemExit("no runs found; pass --run or --runs-root")
    print(f"[runs ] {len(runs)} run(s) to grade")

    gt = rigkit.load_cloud(str(args.gt))
    print(f"[gt   ] {len(gt)} points from {args.gt}")
    if args.gt_voxel > 0 and o3d is not None:
        gt = np.asarray(_pcd(gt).voxel_down_sample(args.gt_voxel).points)
        print(f"[gt   ] {len(gt)} after {args.gt_voxel * 1e3:.0f} mm voxel")

    # ---- one alignment, frozen ----------------------------------------
    if args.align_file.exists() and not args.realign:
        T = np.asarray(json.loads(args.align_file.read_text())["T"], float)
        print(f"[gt   ] reusing the alignment in {args.align_file}")
    else:
        ref_run = None
        if args.align_from:
            ref_run = parse_run(Path(args.align_from))
        if ref_run is None:
            ref_run = sorted(runs, key=lambda r: (len(r["cams"]),
                                                  r["res_label"] or 0))[-1]
        print(f"[gt   ] solving the alignment on {ref_run['dir'].name}")
        ref = rigkit.load_cloud(str(ref_run["dir"] / args.fused_name))
        T = register(gt, ref, belt, args)
        args.align_file.write_text(json.dumps(
            {"T": np.asarray(T).tolist(), "gt": str(args.gt),
             "solved_on": str(ref_run["dir"]), "rigid_only": True}, indent=2))
        print(f"[gt   ] written {args.align_file}. Every run below is scored "
              f"through this same transform, so a rigid error stays an error")

    gt_r = apply(gt, T)
    gt_ras, gt_mask, _ = height_raster(gt_r, *frame, belt, args.cell)
    if gt_ras is None:
        raise SystemExit("no ground-truth points landed inside the belt "
                         "footprint; the alignment is wrong")
    print(f"[gt   ] {int(gt_mask.sum())} belt cells occupied, "
          f"{float(gt_mask.sum()) * args.cell ** 2:.2f} m2, heights to "
          f"{np.nanmax(gt_ras) * 1e3:.0f} mm above the deck")

    rows = []
    for r in sorted(runs, key=lambda r: (len(r["cams"]), r["subset"],
                                         r["res_label"] or 0)):
        row = grade(r, gt_r, gt_ras, gt_mask, centres, belt, frame, args)
        if row:
            rows.append(row)
            print(f"[grade] {r['dir'].name:32s} p={row.get('p_relief', 0):.4f} "
                  f"tops rms {row.get('top_err_rms_mm', float('nan')):6.1f} mm "
                  f"cover {row.get('coverage_frac', 0) * 100:5.1f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "grades.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (args.out / "grades.json").write_text(json.dumps(rows, indent=2))

    # ---- the two tables the question actually asks for -----------------
    print("\nby subset, at each grid. p is the surviving fraction of relief, "
          "1.0 is perfect.")
    print(f"{'subset':24s} {'grid':>9s} {'cond':>7s} {'p':>7s} {'q mm':>7s} "
          f"{'tops rms':>9s} {'tops p95':>9s} {'cover':>7s} {'2-view':>8s} "
          f"{'layer':>7s}")
    for r in rows:
        print(f"{r['subset']:24s} {r.get('depth_grid', ''):>9s} "
              f"{r['cond']:>7.4f} {r.get('p_relief', float('nan')):>7.4f} "
              f"{r.get('q_mm', float('nan')):>+7.1f} "
              f"{r.get('top_err_rms_mm', float('nan')):>9.1f} "
              f"{r.get('top_err_p95_mm', float('nan')):>9.1f} "
              f"{r.get('coverage_frac', float('nan')) * 100:>6.1f}% "
              f"{r.get('mv2_frac', float('nan')) * 100:>7.1f}% "
              f"{r.get('layer_above_spread_mm') or float('nan'):>7.1f}")

    print("\nRead it this way. p and q are removable by a correction fitted "
          "once, so a subset with a stable p is not disqualified by it. The "
          "residual about that line, and tops p95, are what no correction "
          "reaches. The layer column is the only one that should improve with "
          "camera count; if it does not, the extra cameras are buying coverage "
          "and nothing else.")
    print(f"\nwritten: {args.out / 'grades.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())