#!/usr/bin/env python3
"""
deck_flatness.py

Separate the GLOBAL warp of a depth field from the LOCAL flatness at the scale
of a parcel, and decide whether that warp is low-order enough to be corrected.

Why this exists
---------------
layer_align.py reports the reference camera's departure from its own dominant
plane as a single rms over the whole frame. On the 12 mm rig that is 6.8 mm rms
with a -12.5 to +10.4 mm spread, and it does not improve when the processing
resolution is raised from 504 to 728, so it is not a sampling artefact.

Taken at face value that number rules out centimetre-level work. But it is the
wrong number for the question actually being asked. A parcel top face occupies
perhaps a fifth of the frame width. If the warp is a smooth low-order field,
the deviation across a footprint that size is a small fraction of the deviation
across the whole frame; if the warp is high-frequency, it is not. Those two
cases demand completely different responses, and one global rms cannot tell
them apart.

This script measures three quantities on the same surface:

  GLOBAL rms        departure from one plane fitted across the whole deck.
                    This is the figure layer_align.py already prints.

  LOCAL rms         the deck is tiled into patches of --patch-m, a plane is
                    fitted independently in each, and the residual is taken
                    within the patch. This is what a parcel top face actually
                    experiences, and it is the number that decides whether
                    footprint, tilt and pick pose are usable.

  MODEL rms         a low-order 2D polynomial in the image plane is fitted to
                    the global residual field and subtracted. What it removes
                    is the correctable part; what remains is not reachable by
                    any smooth per-view correction.

The interpretation is mechanical:

  local small, model removes most of the global
      the warp is a smooth field. Parcels are measured correctly in isolation,
      the error is a slowly varying bias in WHERE they sit, and a per-view
      spatial correction fitted on the deck removes it. The deck is visible in
      every frame, so that correction can be solved online.

  local small, model removes little
      the warp is smooth at parcel scale but not describable by a low-order
      surface. Parcels are still measured correctly; the positional bias needs
      a denser control-point field rather than a polynomial.

  local comparable to global
      the depth field is rough at parcel scale. Nothing downstream is
      recoverable by correction, and the honest result is that this model at
      this framing does not support the accuracy target.

Input is a capture directory holding depth_<cam>.npy, conf_<cam>.npy,
K_<cam>.npy and E_<cam>.npy, as written by da3_stream.py, da3_fuse.py or
res_sweep.py --dump-arrays.

The filters default to the values the pipeline runs, because a flatness
measured on a different population of samples from the one the pipeline uses
is not a measurement of the pipeline.

Example
-------
    python deck_flatness.py --capture-dir runs/ressweep_.../res_0504 \\
        --patch-m 0.4 --out-dir /tmp/flatness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d is required")

try:
    from da3_fuse import points_camera, edge_mask, incidence_mask, to_world
except ImportError as exc:
    raise SystemExit(f"cannot import da3_fuse.py: {exc}")


# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Separate the global warp of a depth field from its local "
                    "flatness at parcel scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center",
                    help="the camera whose cloud defines the deck plane every "
                         "other camera is measured against")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <capture-dir>/flatness")

    ap.add_argument("--patch-m", type=float, default=0.40,
                    help="side of the square patch a local plane is fitted in, "
                         "in metres. Set it to the footprint of a typical "
                         "parcel: this is the scale at which the question is "
                         "being asked")
    ap.add_argument("--min-patch-points", type=int, default=200,
                    help="points a patch must hold before its local fit is "
                         "trusted")
    ap.add_argument("--poly-order", type=int, default=2, choices=[1, 2, 3],
                    help="order of the 2D polynomial fitted to the global "
                         "residual field. 2 is a saddle or a bowl, which is "
                         "what a smooth depth warp usually looks like")

    ap.add_argument("--plane-thresh", type=float, default=0.006,
                    help="RANSAC inlier distance for the deck plane")
    ap.add_argument("--plane-iters", type=int, default=2000)
    ap.add_argument("--assign-tol", type=float, default=0.015,
                    help="a point belongs to the deck if it lies within this "
                         "distance of the fitted plane. Keep it TIGHT: at "
                         "50 mm this admits parcel bases and belt frame, and "
                         "the flatness then measures the scene rather than the "
                         "depth field")
    ap.add_argument("--seed", type=int, default=0)

    # filters, matching da3_stream.py defaults exactly
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--conf-min", type=float, default=0.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)

    ap.add_argument("--no-maps", dest="maps", action="store_false", default=True,
                    help="skip writing the residual field images")
    return ap.parse_args()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_capture(capture_dir: Path, names):
    data = {}
    for n in names:
        req = {k: capture_dir / f"{k}_{n}.npy" for k in ("depth", "K", "E")}
        if not all(p.exists() for p in req.values()):
            continue
        entry = {k: np.load(p) for k, p in req.items()}
        conf_p = capture_dir / f"conf_{n}.npy"
        entry["conf"] = np.load(conf_p) if conf_p.exists() else None
        data[n] = entry
    if not data:
        raise SystemExit(f"no depth arrays found in {capture_dir}")
    return data


def valid_mask(depth, conf, K, args):
    """The same filter chain the pipeline runs, so the population matches."""
    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    if args.conf_percentile > 0 and conf is not None and valid.any():
        valid &= conf >= float(np.percentile(conf[valid], args.conf_percentile))
    if args.conf_min > 0 and conf is not None:
        valid &= conf >= args.conf_min
    if args.edge_thresh > 0:
        valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
    if args.max_incidence < 90:
        ok, _ = incidence_mask(points_camera(d, K), args.max_incidence)
        valid &= ok
    return valid


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def fit_plane(points, thresh, iters, seed=0):
    o3d.utility.random.seed(seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, inliers = pcd.segment_plane(thresh, 3, iters)
    a, b, c, d = model
    n = np.array([a, b, c], float)
    nrm = np.linalg.norm(n)
    if nrm < 1e-9:
        return None, None
    n, off = n / nrm, float(-d / nrm)
    if off < 0:
        n, off = -n, -off
    # Least-squares refinement over the inliers, so the plane is not defined by
    # the three points RANSAC happened to sample.
    for _ in range(3):
        sd = points @ n - off
        inl = points[np.abs(sd) <= thresh]
        if len(inl) < 10:
            break
        c0 = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
        n_new = vt[-1]
        if float(n_new @ n) < 0:
            n_new = -n_new
        n, off = n_new, float(n_new @ c0)
    return n, off


def basis_from_normal(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    h = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = h - n * float(h @ n)
    ex /= np.linalg.norm(ex)
    return ex, np.cross(n, ex)


def local_flatness(P, resid, ex, ey, patch_m, min_points):
    """Fit a plane independently inside each patch and report the residual.

    A parcel top face is measured against its own local geometry, not against a
    plane fitted a metre away, so this is the quantity that governs footprint,
    tilt and the pick pose. The global figure governs only where that parcel is
    reported to be.
    """
    u, v = P @ ex, P @ ey
    iu = np.floor((u - u.min()) / patch_m).astype(np.int64)
    iv = np.floor((v - v.min()) / patch_m).astype(np.int64)
    key = iu * (iv.max() + 1) + iv
    out = []
    for k in np.unique(key):
        sel = key == k
        n_pts = int(sel.sum())
        if n_pts < min_points:
            continue
        Q = P[sel]
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        nl = vt[-1]
        r = (Q - c) @ nl
        out.append({
            "n_points": n_pts,
            "centre_uv_m": [round(float(u[sel].mean()), 4),
                            round(float(v[sel].mean()), 4)],
            "rms_mm": float(np.sqrt(np.mean(r ** 2))) * 1e3,
            "p2p_mm": float(np.percentile(r, 97) - np.percentile(r, 3)) * 1e3,
            # how far this patch's own plane tilts from the global one, which
            # is the error a footprint inherits as a tilt rather than as a bow
            "tilt_deg": float(np.degrees(np.arccos(np.clip(
                abs(float(nl @ np.cross(ex, ey))), 0.0, 1.0)))),
            "offset_mm": float(np.mean(resid[sel])) * 1e3,
        })
    return out


def poly_design(u, v, order):
    cols = [np.ones_like(u), u, v]
    if order >= 2:
        cols += [u * u, u * v, v * v]
    if order >= 3:
        cols += [u ** 3, u * u * v, u * v * v, v ** 3]
    return np.stack(cols, axis=1)


def polynomial_model(u, v, resid, order):
    """How much of the global warp a smooth low-order surface accounts for.

    If a quadratic removes most of it, the warp is a bowl or a saddle and a
    per-view spatial correction fitted on the deck will transfer, because the
    deck is visible in every frame. If it removes little, the field is smooth
    at parcel scale but not globally describable, and the correction has to be
    a denser control-point grid.
    """
    A = poly_design(u, v, order)
    sol, *_ = np.linalg.lstsq(A, resid, rcond=None)
    fit = A @ sol
    left = resid - fit
    var0 = float(np.var(resid))
    return {
        "order": order,
        "coeffs": [float(c) for c in sol],
        "rms_before_mm": float(np.sqrt(np.mean(resid ** 2))) * 1e3,
        "rms_after_mm": float(np.sqrt(np.mean(left ** 2))) * 1e3,
        "variance_explained": (round(1.0 - float(np.var(left)) / var0, 4)
                               if var0 > 0 else None),
    }, left


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_field(u, v, resid, path, title, cell=0.02, lim_mm=None):
    """Top-down median residual per cell, in millimetres."""
    if cv2 is None:
        return None
    u0, v0 = float(u.min()), float(v.min())
    W = int((u.max() - u0) / cell) + 1
    H = int((v.max() - v0) / cell) + 1
    if W < 4 or H < 4 or W * H > 4_000_000:
        return None
    iu = np.clip(((u - u0) / cell).astype(np.int64), 0, W - 1)
    iv = np.clip(((v - v0) / cell).astype(np.int64), 0, H - 1)
    flat = iv * W + iu
    tot = np.bincount(flat, weights=resid * 1e3, minlength=W * H)
    cnt = np.bincount(flat, minlength=W * H)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    img = mean.reshape(H, W)

    lim = lim_mm if lim_mm else max(1.0, float(np.nanpercentile(np.abs(img), 98)))
    norm = np.clip((np.nan_to_num(img, nan=0.0) / lim + 1.0) / 2.0, 0, 1)
    canvas = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    canvas[cnt.reshape(H, W) == 0] = (25, 25, 25)

    scale = max(1, int(700 / max(W, H)))
    canvas = cv2.resize(canvas, (W * scale, H * scale),
                        interpolation=cv2.INTER_NEAREST)
    bar = np.full((30, canvas.shape[1], 3), 18, np.uint8)
    cv2.putText(bar, f"{title}   +/-{lim:.1f} mm full scale", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([bar, canvas]))
    return path


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (args.capture_dir / "flatness")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_capture(args.capture_dir, args.cameras)
    names = [n for n in args.cameras if n in data]
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} not among {names}")

    print(f"[filt ] conf p{args.conf_percentile:.0f}, edge {args.edge_thresh}, "
          f"incidence {args.max_incidence:.0f} deg, assign tol "
          f"{args.assign_tol * 1e3:.0f} mm")
    print(f"[patch] {args.patch_m * 1e3:.0f} mm squares, minimum "
          f"{args.min_patch_points} points, polynomial order {args.poly_order}")

    clouds = {}
    for n in names:
        m = valid_mask(data[n]["depth"], data[n]["conf"], data[n]["K"], args)
        pts = points_camera(data[n]["depth"].astype(np.float64), data[n]["K"])
        clouds[n] = to_world(pts[m], data[n]["E"])
        print(f"[load ] {n:<8} {len(clouds[n]):>8d} points")

    n_plane, off = fit_plane(clouds[args.reference], args.plane_thresh,
                             args.plane_iters, args.seed)
    if n_plane is None:
        raise SystemExit("the deck plane fit failed")
    ex, ey = basis_from_normal(n_plane)
    print(f"[plane] normal {np.round(n_plane, 4)}  offset {off:.4f} m "
          f"(DA3's unanchored coordinates)")

    report = {"capture_dir": str(args.capture_dir),
              "reference": args.reference,
              "patch_m": args.patch_m,
              "plane": {"normal": [round(float(v), 6) for v in n_plane],
                        "offset_m": round(float(off), 5)},
              "filters": {"conf_percentile": args.conf_percentile,
                          "edge_thresh": args.edge_thresh,
                          "max_incidence_deg": args.max_incidence,
                          "assign_tol_m": args.assign_tol},
              "cameras": {}}

    print(f"\n{'=' * 92}")
    print(f"{'camera':<9}{'points':>9}{'GLOBAL rms':>12}{'p05-p95':>11}"
          f"{'LOCAL rms':>12}{'local p2p':>11}{'MODEL rms':>11}{'var expl':>10}")
    print(f"{'-' * 92}")

    for n in names:
        P = clouds[n]
        sd = P @ n_plane - off
        near = np.abs(sd) <= args.assign_tol
        if int(near.sum()) < 2000:
            print(f"{n:<9}{int(near.sum()):>9}   too few deck points")
            continue
        Q, r = P[near], sd[near]
        u, v = Q @ ex, Q @ ey

        g_rms = float(np.sqrt(np.mean(r ** 2))) * 1e3
        g_p05 = float(np.percentile(r, 5)) * 1e3
        g_p95 = float(np.percentile(r, 95)) * 1e3

        patches = local_flatness(Q, r, ex, ey, args.patch_m,
                                 args.min_patch_points)
        l_rms = (float(np.median([p["rms_mm"] for p in patches]))
                 if patches else float("nan"))
        l_p2p = (float(np.median([p["p2p_mm"] for p in patches]))
                 if patches else float("nan"))
        l_tilt = (float(np.median([p["tilt_deg"] for p in patches]))
                  if patches else float("nan"))

        model, left = polynomial_model(u - u.mean(), v - v.mean(), r,
                                       args.poly_order)

        print(f"{n:<9}{int(near.sum()):>9}{g_rms:>12.2f}"
              f"{g_p05:>+6.1f}/{g_p95:>+5.1f}{l_rms:>12.2f}{l_p2p:>11.2f}"
              f"{model['rms_after_mm']:>11.2f}"
              f"{(model['variance_explained'] or 0) * 100:>9.0f}%")

        report["cameras"][n] = {
            "n_deck_points": int(near.sum()),
            "global": {"rms_mm": round(g_rms, 3),
                       "p05_mm": round(g_p05, 2),
                       "p95_mm": round(g_p95, 2)},
            "local": {"n_patches": len(patches),
                      "median_rms_mm": round(l_rms, 3),
                      "median_p2p_mm": round(l_p2p, 3),
                      "median_tilt_deg": round(l_tilt, 4),
                      "worst_rms_mm": (round(max(p["rms_mm"] for p in patches), 3)
                                       if patches else None),
                      "patch_offset_spread_mm": (
                          round(max(p["offset_mm"] for p in patches)
                                - min(p["offset_mm"] for p in patches), 2)
                          if len(patches) > 1 else None),
                      "patches": patches},
            "polynomial": {k: v for k, v in model.items() if k != "coeffs"},
            "polynomial_coeffs": model["coeffs"],
        }

        if args.maps and cv2 is not None:
            render_field(u, v, r, out_dir / f"residual_{n}.png",
                         f"{n}: residual about the global plane")
            render_field(u, v, left, out_dir / f"residual_{n}_after_poly.png",
                         f"{n}: after an order-{args.poly_order} surface")

    print(f"{'=' * 92}")

    ref = report["cameras"].get(args.reference)
    if ref:
        g = ref["global"]["rms_mm"]
        l = ref["local"]["median_rms_mm"]
        m = ref["polynomial"]["rms_after_mm"]
        ve = ref["polynomial"]["variance_explained"] or 0.0
        ratio = l / g if g > 0 else float("nan")
        print(f"\nreference camera {args.reference}: global {g:.2f} mm, "
              f"local {l:.2f} mm at {args.patch_m * 1e3:.0f} mm, "
              f"{ve * 100:.0f}% of the warp explained by an order-"
              f"{args.poly_order} surface")
        print(f"local is {ratio * 100:.0f} per cent of global\n")

        if ratio < 0.4 and ve > 0.6:
            verdict = (
                "SMOOTH AND CORRECTABLE. The depth field is far flatter over a "
                "parcel footprint than\nover the whole frame, and most of the "
                "warp is a low-order surface. Parcel\ndimensions, tilt and pick "
                "poses are governed by the LOCAL figure and are usable.\nThe "
                "global warp is a positional bias, and because it is low-order "
                "and the deck is\nvisible in every frame, a per-view spatial "
                "correction fitted on the deck will\ntransfer. That is the next "
                "change to make, and it is now justified by\nmeasurement rather "
                "than by assumption.")
        elif ratio < 0.4:
            verdict = (
                "SMOOTH BUT NOT LOW-ORDER. Parcels are measured correctly in "
                "isolation, so\ndimensions and pick poses stand on the LOCAL "
                f"figure of {l:.2f} mm. The global warp\nis real but is not a "
                "bowl or a saddle, so a polynomial will not remove it; that\n"
                "needs a denser control-point grid, anchored at known "
                "positions across the deck.")
        else:
            verdict = (
                "ROUGH AT PARCEL SCALE. The depth field is nearly as bad over a "
                "footprint as over\nthe whole frame, so no correction recovers "
                "it: there is no smooth field to\nsubtract. Dimensions and top-"
                "face residuals inherit this directly. The honest\nresult is "
                "that this model at this framing does not support the accuracy "
                "target,\nand the next lever is the model or the optics rather "
                "than the software.")
        print(verdict)
        report["verdict"] = verdict.replace("\n", " ")
        report["local_over_global"] = round(ratio, 4)

    print(f"\nA parcel top face is measured against its own local geometry, so "
          f"the LOCAL column\ngoverns footprint, tilt and pick pose. The GLOBAL "
          f"column governs only where that\nparcel is reported to be.")

    (out_dir / "flatness.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'flatness.json'}")
    if args.maps and cv2 is not None:
        print(f"fields : {out_dir}/residual_<cam>.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())