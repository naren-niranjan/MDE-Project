#!/usr/bin/env python3
"""
layer_audit.py

Decompose the layering that survives --affine into the three terms that have
three different causes and three different fixes.

WHY THIS EXISTS
---------------
layer_report() in da3_stream.py reports each view's MEDIAN signed offset from
the fused plane. A cloud in which four views place the belt on four planes
that scissor through one another reports a median spread of nearly zero, so
the metric is blind to exactly the failure the clouds are showing. Tuning
against it cannot converge.

This tool reports, per view:

  offset_mm         median height above the fused plane. This is what --affine
                    corrects and what layer_report already sees.

  tilt_deg          angle between this view's own belt plane and the fused
                    plane. A per-view inverse-depth affine CANNOT produce or
                    remove a tilt: 1/z' = a/z + b is a function of z alone,
                    so it moves a surface, it does not rotate it. A tilt is a
                    pose error, an intrinsic error, or a depth warp.

  warp_rms_mm       rms residual of this view's belt points about its OWN best
                    plane. Non-planarity. Neither the scalar affine nor a pose
                    fix touches this.

and then the question that decides everything:

  IS THE NON-PLANARITY SHARED OR PER-VIEW?

The belt height field is rasterised onto a common grid in plane coordinates
and split into the part all views agree on and the part they do not:

  shared_sd_mm      spatial standard deviation of the across-view mean field.
                    A real wave in the deck, or a common-mode DA3 error, puts
                    its signal here. It is NOT layering and no per-view
                    correction reaches it.

  per_view_rms_mm   rms of each view's departure from that mean field, after
                    its own offset is removed. THIS is the layering, and it is
                    the only part a per-view correction can be held
                    responsible for.

A wavy belt with a low per_view_rms_mm is a wavy belt. A wavy belt with a high
per_view_rms_mm is four wavy belts, which is a different problem.

POSE SUBSTITUTION
-----------------
da3_stream.py back-projects through E_out and K_out, which are DA3's RETURNED
extrinsics and intrinsics, not the calibrated ones:

    E_out = as_numpy(getattr(pred, "extrinsics", None))
    if E_out is None:
        E_out = Es              # only a fallback

depth_align.py fits its correction in each camera's OWN frame and says so:
"No extrinsics are involved ... the fit is immune to extrinsic error." So a
pose error survives the correction completely intact. Four metrically correct
depth maps pushed through four drifted poses land in four places.

--pose both runs the whole audit twice, once through the poses saved in the
capture and once through calibration, and prints both tables. If the spread
collapses under 'calib', the layering was never a depth problem.

INPUT
-----
A frame directory written by da3_stream.py with --save-npy, holding per
camera: depth_<name>.npy, depth_raw_<name>.npy, K_<name>.npy, E_<name>.npy
and optionally conf_<name>.npy.

    python layer_audit.py --capture /home/jetson/Projects/MDE/new/runs/before/frame_00002 \
        --calib-dir /home/jetson/Projects/Calibration_4_5/results \
        --pose both --json layer_audit.json

Add --use-raw to audit the depth BEFORE the affine was applied, which gives
you the before/after pair on identical geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401  (imported for parity with the pipeline's filters)
except ImportError:
    raise SystemExit("opencv is required")

# One definition of the rig geometry, taken from the streamer, exactly as
# depth_align.py does. da3_stream.py imports arena_api lazily, so this is safe
# on a machine with no camera SDK.
try:
    from da3_stream import (DEFAULT_CALIB, camera_centre, edge_mask,
                            fit_plane_ransac, incidence_mask,
                            load_extrinsics, load_intrinsics, points_camera,
                            to_world)
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"da3_stream.py must sit beside this file and import "
                     f"cleanly. The import failed with: {exc!r}")


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def plane_basis(n):
    """Two orthonormal in-plane axes for a unit normal."""
    n = np.asarray(n, np.float64)
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def fit_plane_tukey(P, iters=6, tukey_c=2.5):
    """Reweighted plane fit. Returns (normal, offset, signed residuals)."""
    P = np.asarray(P, np.float64)
    if len(P) < 8:
        return None, None, None
    w = np.ones(len(P))
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(max(1, iters)):
        ws = w / max(float(w.sum()), 1e-12)
        c = (P * ws[:, None]).sum(axis=0)
        M = (P - c) * np.sqrt(ws)[:, None]
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        n = vt[-1] / np.linalg.norm(vt[-1])
        r = (P - c) @ n
        s = max(1.4826 * float(np.median(np.abs(r))), 1e-9)
        t = np.clip(r / (tukey_c * s), -1.0, 1.0)
        w = (1.0 - t ** 2) ** 2
    return n, float(n @ c), (P - c) @ n


def rotation_angle_deg(Ra, Rb):
    """Angle of the relative rotation between two rotation matrices."""
    R = np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T
    c = (float(np.trace(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_view(frame_dir: Path, name: str, use_raw: bool):
    stem = "depth_raw" if use_raw else "depth"
    dpath = frame_dir / f"{stem}_{name}.npy"
    kpath = frame_dir / f"K_{name}.npy"
    epath = frame_dir / f"E_{name}.npy"
    for p in (dpath, kpath):
        if not p.exists():
            raise SystemExit(f"missing {p}. Re-capture with --save-npy"
                             + ("; --use-raw needs depth_raw_*.npy, which is "
                                "only written when --affine was in use"
                                if use_raw else ""))
    cpath = frame_dir / f"conf_{name}.npy"
    return {
        "depth": np.load(dpath).astype(np.float64),
        "K": np.load(kpath).astype(np.float64),
        "E": np.load(epath).astype(np.float64) if epath.exists() else None,
        "conf": np.load(cpath) if cpath.exists() else None,
    }


def filtered_world(view, E, args):
    """Back-project one view with the same rejection the streamer applies."""
    d = view["depth"]
    c = view["conf"]
    K = view["K"]

    valid = np.isfinite(d) & (d > 0)
    if args.conf_percentile > 0 and c is not None and valid.any():
        thr = float(np.percentile(c[valid], args.conf_percentile))
        valid &= c >= thr
    if args.edge_thresh > 0:
        valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)

    safe = np.where(np.isfinite(d), d, 0.0)
    pts_cam = points_camera(safe, K)
    if args.max_incidence < 90:
        ok, _ = incidence_mask(pts_cam, args.max_incidence)
        valid &= ok
    return to_world(pts_cam[valid], E)


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------

def raster_field(s, t, h, cell_m, min_count):
    """Median height per square cell in plane coordinates."""
    gi = np.floor(s / cell_m).astype(np.int64)
    gj = np.floor(t / cell_m).astype(np.int64)
    key = {}
    order = np.lexsort((gj, gi))
    gi, gj, h = gi[order], gj[order], h[order]
    # boundaries where (gi, gj) changes
    change = np.ones(len(gi), bool)
    change[1:] = (gi[1:] != gi[:-1]) | (gj[1:] != gj[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(gi))
    for a, b in zip(starts, ends):
        if b - a >= min_count:
            key[(int(gi[a]), int(gj[a]))] = float(np.median(h[a:b]))
    return key


def decompose(fields, min_views):
    """
    Split the rasterised height fields into the part the views agree on and
    the part they do not, after removing each view's own mean offset.
    """
    names = list(fields)
    if len(names) < min_views:
        return None
    counts = {}
    for n in names:
        for k in fields[n]:
            counts[k] = counts.get(k, 0) + 1
    common = [k for k, v in counts.items() if v >= min_views]
    if len(common) < 20:
        return None

    M = np.full((len(names), len(common)), np.nan)
    for i, n in enumerate(names):
        f = fields[n]
        for j, k in enumerate(common):
            if k in f:
                M[i, j] = f[k]

    # Remove each view's own offset so the decomposition measures shape, not
    # the term --affine already handles.
    offsets = np.nanmedian(M, axis=1, keepdims=True)
    D = M - offsets

    shared = np.nanmean(D, axis=0)
    resid = D - shared[None, :]

    per_view = {}
    for i, n in enumerate(names):
        r = resid[i][np.isfinite(resid[i])]
        per_view[n] = round(float(np.sqrt(np.mean(r ** 2))) * 1e3, 2)

    pair = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = np.isfinite(D[i]) & np.isfinite(D[j])
            if both.sum() < 20:
                continue
            a, b = D[i][both], D[j][both]
            corr = (float(np.corrcoef(a, b)[0, 1])
                    if a.std() > 1e-9 and b.std() > 1e-9 else float("nan"))
            pair[f"{names[i]}|{names[j]}"] = {
                "n_cells": int(both.sum()),
                "correlation": None if np.isnan(corr) else round(corr, 3),
                "difference_rms_mm": round(
                    float(np.sqrt(np.mean((a - b) ** 2))) * 1e3, 2),
            }

    allres = resid[np.isfinite(resid)]
    return {
        "n_cells": len(common),
        "shared_sd_mm": round(float(np.nanstd(shared)) * 1e3, 2),
        "per_view_rms_mm": round(float(np.sqrt(np.mean(allres ** 2))) * 1e3, 2),
        "per_view_rms_by_camera_mm": per_view,
        "pairwise": pair,
    }


def audit(world, centres, args, tag):
    allpts = np.vstack([p for p in world.values() if len(p)])
    if len(allpts) < 1000:
        raise SystemExit(f"{tag}: only {len(allpts)} points survived the "
                         f"filters; nothing to fit")

    n, d = fit_plane_ransac(allpts, thresh=0.010)
    if n is None:
        raise SystemExit(f"{tag}: no dominant plane found")
    if float(np.mean(centres @ n - d)) < 0:
        n, d = -n, -d
    u, v = plane_basis(n)

    band = args.band_mm / 1e3
    above = args.above_mm / 1e3

    per_view, belt_fields, above_fields = {}, {}, {}
    for name, pts in world.items():
        if not len(pts):
            continue
        h = pts @ n - d
        sel = np.abs(h) <= band
        rec = {"n_points": int(len(pts)), "n_belt_points": int(sel.sum())}
        if sel.sum() >= 500:
            nv, dv, res = fit_plane_tukey(pts[sel])
            rec.update({
                "offset_mm": round(float(np.median(h[sel])) * 1e3, 2),
                "tilt_deg": round(float(np.degrees(np.arccos(
                    np.clip(abs(float(nv @ n)), 0.0, 1.0)))), 4),
                "warp_rms_mm": round(
                    float(np.sqrt(np.mean(res ** 2))) * 1e3, 2),
                "warp_p95_mm": round(
                    float(np.percentile(np.abs(res), 95)) * 1e3, 2),
            })
            belt_fields[name] = raster_field(
                pts[sel] @ u, pts[sel] @ v, h[sel],
                args.cell_mm / 1e3, args.min_per_cell)
        hi = h > above
        if hi.sum() >= 500:
            rec["above_offset_mm"] = round(float(np.median(h[hi])) * 1e3, 2)
            nv, _, res = fit_plane_tukey(pts[hi])
            rec["above_warp_rms_mm"] = round(
                float(np.sqrt(np.mean(res ** 2))) * 1e3, 2)
            above_fields[name] = raster_field(
                pts[hi] @ u, pts[hi] @ v, h[hi],
                args.cell_mm / 1e3, args.min_per_cell)
        per_view[name] = rec

    offs = [r["offset_mm"] for r in per_view.values() if "offset_mm" in r]
    tilts = [r["tilt_deg"] for r in per_view.values() if "tilt_deg" in r]
    aoffs = [r["above_offset_mm"] for r in per_view.values()
             if "above_offset_mm" in r]

    return {
        "pose_source": tag,
        "plane_normal": n.tolist(),
        "plane_offset_m": round(float(d), 6),
        "per_view": per_view,
        "belt_offset_spread_mm": (round(max(offs) - min(offs), 2)
                                  if len(offs) > 1 else None),
        "belt_tilt_spread_deg": (round(max(tilts) - min(tilts), 4)
                                 if len(tilts) > 1 else None),
        "above_offset_spread_mm": (round(max(aoffs) - min(aoffs), 2)
                                   if len(aoffs) > 1 else None),
        "belt_field": decompose(belt_fields, args.min_views),
        "above_field": decompose(above_fields, args.min_views),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_poses(deltas):
    print("\n[pose ] DA3's returned geometry against the calibration")
    print(f"[pose ] {'camera':<8} {'rot err':>9} {'centre err':>12} "
          f"{'fx ratio':>10}")
    for name, r in deltas.items():
        if r.get("rotation_deg") is None:
            print(f"[pose ] {name:<8} {'no E_*.npy saved':>33}")
            continue
        flag = ""
        if r["rotation_deg"] > 0.05 or r["centre_err_mm"] > 2.0:
            flag = "   <-- pose substitution is moving this view"
        print(f"[pose ] {name:<8} {r['rotation_deg']:8.4f}d "
              f"{r['centre_err_mm']:9.2f} mm {r['fx_ratio']:10.5f}{flag}")
    print("[pose ] a rotation of 0.1 deg is about 2 mm across a 1.2 m belt, "
          "and it tilts the view rather than shifting it, so no per-view "
          "inverse-depth affine can remove it.")


def print_audit(rep):
    tag = rep["pose_source"]
    print(f"\n[{tag:<5}] per-view belt geometry about the fused plane")
    print(f"[{tag:<5}] {'camera':<8} {'offset':>9} {'tilt':>9} "
          f"{'warp rms':>10} {'warp p95':>10} {'top offset':>11}")
    for name, r in rep["per_view"].items():
        if "offset_mm" not in r:
            print(f"[{tag:<5}] {name:<8} too few belt points")
            continue
        print(f"[{tag:<5}] {name:<8} {r['offset_mm']:+8.2f} "
              f"{r['tilt_deg']:8.4f}d {r['warp_rms_mm']:7.2f} mm "
              f"{r['warp_p95_mm']:7.2f} mm "
              + (f"{r['above_offset_mm']:+8.2f} mm"
                 if "above_offset_mm" in r else f"{'-':>11}"))
    print(f"[{tag:<5}] spread: offset "
          f"{rep['belt_offset_spread_mm']} mm   tilt "
          f"{rep['belt_tilt_spread_deg']} deg   parcel top "
          f"{rep['above_offset_spread_mm']} mm")

    for key, label in (("belt_field", "belt"), ("above_field", "parcel tops")):
        f = rep.get(key)
        if not f:
            print(f"[{tag:<5}] {label}: too little overlap between views to "
                  f"decompose the height field")
            continue
        ratio = (f["per_view_rms_mm"] / f["shared_sd_mm"]
                 if f["shared_sd_mm"] > 1e-9 else float("inf"))
        print(f"[{tag:<5}] {label} height field over {f['n_cells']} cells: "
              f"shared shape {f['shared_sd_mm']:.2f} mm sd, per-view departure "
              f"{f['per_view_rms_mm']:.2f} mm rms  (ratio {ratio:.2f})")
        print(f"[{tag:<5}]   per view: " + "  ".join(
            f"{k}:{v:.1f}" for k, v in
            f["per_view_rms_by_camera_mm"].items()))
        for pair, p in f["pairwise"].items():
            print(f"[{tag:<5}]   {pair:<16} r={p['correlation']} "
                  f"difference {p['difference_rms_mm']:.2f} mm rms over "
                  f"{p['n_cells']} cells")


def verdict(rep, deltas):
    print("\n[verdict]")
    lines = []
    pose_bad = any(r.get("rotation_deg") is not None
                   and (r["rotation_deg"] > 0.05 or r["centre_err_mm"] > 2.0)
                   for r in deltas.values())
    tilt = rep.get("belt_tilt_spread_deg") or 0.0
    off = rep.get("belt_offset_spread_mm") or 0.0
    f = rep.get("belt_field") or {}
    shared = f.get("shared_sd_mm", 0.0)
    perview = f.get("per_view_rms_mm", 0.0)

    if pose_bad:
        lines.append(
            "DA3 is returning geometry that differs from the calibration, and "
            "the streamer back-projects through it. Re-run the audit with "
            "--pose calib and compare the spreads; if they fall, pin E_out "
            "and K_out to the calibrated values in run_inference.")
    if tilt > 0.2:
        lines.append(
            f"The views disagree by {tilt:.3f} deg on the belt normal. A "
            f"per-view inverse-depth affine cannot rotate a surface, so no "
            f"depth_affine.json will fix this. Look at pose and intrinsics "
            f"first, then at a depth warp with a linear gradient.")
    if perview > 6.0 and perview > 0.6 * max(shared, 1e-9):
        lines.append(
            f"The belt is not one surface seen four times: each view departs "
            f"from the across-view mean shape by {perview:.1f} mm rms against "
            f"a shared shape of only {shared:.1f} mm sd. That is genuine "
            f"per-view spatial warp. The scalar (a, b) cannot reach it by "
            f"construction; only a residual field can, and it needs board "
            f"coverage across the whole frame.")
    if shared > 6.0 and perview < 0.4 * shared:
        lines.append(
            f"The non-planarity is SHARED: {shared:.1f} mm sd of shape that "
            f"all views agree on against {perview:.1f} mm rms of "
            f"disagreement. That is either a real deck wave or a common-mode "
            f"DA3 error. It is not layering, no per-view correction touches "
            f"it, and it must not be chased with --affine.")
    if off > 6.0 and tilt < 0.2 and perview < 6.0:
        lines.append(
            f"What is left is a clean per-view offset of {off:.1f} mm with "
            f"flat, agreeing surfaces. That IS what the affine corrects, so "
            f"suspect the correction file: check grid_coverage_fraction, "
            f"depth_shape against this run, and that it is a real "
            f"depth_align.py solve.")
    if not lines:
        lines.append(
            "No term exceeds its threshold. The residual layering in this "
            "capture is within the per-view noise floor; re-run on a capture "
            "that shows the problem, or lower the thresholds deliberately.")
    for line in lines:
        print("  - " + line)


# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Decompose residual layering into offset, tilt and warp, "
                    "and separate shared surface shape from per-view warp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True,
                    help="a frame_XXXXX directory written with --save-npy")
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--pose", choices=["da3", "calib", "both"], default="both",
                    help="which extrinsics to back-project through. 'da3' is "
                         "what da3_stream.py actually uses")
    ap.add_argument("--use-raw", action="store_true",
                    help="audit depth_raw_*.npy, the depth before --affine")

    ap.add_argument("--band-mm", type=float, default=40.0,
                    help="half-thickness of the belt band about the fused plane")
    ap.add_argument("--above-mm", type=float, default=60.0,
                    help="points this far above the plane are the parcel tops")
    ap.add_argument("--cell-mm", type=float, default=40.0,
                    help="raster cell for the height field, in mm on the belt")
    ap.add_argument("--min-per-cell", type=int, default=15)
    ap.add_argument("--min-views", type=int, default=3,
                    help="views that must populate a cell for it to enter the "
                         "shared/per-view decomposition")

    # Defaults deliberately mirror da3_stream.py so the audit sees the same
    # point set the fusion does.
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=2)
    ap.add_argument("--max-incidence", type=float, default=70.0)

    ap.add_argument("--json", default="layer_audit.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    names = list(args.cameras)
    if not args.capture.is_dir():
        raise SystemExit(f"{args.capture} is not a directory")

    views = {n: load_view(args.capture, n, args.use_raw) for n in names}
    intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
    ext = load_extrinsics(args.calib_dir, names, args.reference)
    centres = np.stack([camera_centre(ext[n]["E"]) for n in names])

    print(f"[audit] {args.capture}")
    print(f"[audit] depth source: "
          f"{'depth_raw (before affine)' if args.use_raw else 'depth (after affine)'}")
    dh, dw = views[names[0]]["depth"].shape
    print(f"[audit] depth grid {dh}x{dw}")

    # ---- how far DA3's geometry has drifted from the calibration --------
    deltas = {}
    for n in names:
        Ecal = ext[n]["E"]
        Eda3 = views[n]["E"]
        # The rectified fx expected on this depth grid, from calibration.
        fx_expect = intr[n]["K"][0, 0] * 0.0  # placeholder, replaced below
        # Rebuilding the rectification plan here would duplicate the streamer;
        # the ratio that matters is DA3's fx against the fx implied by the
        # saved K, scaled to the depth grid. Use the saved K's own fx against
        # the calibrated raw fx scaled by the grid width.
        fx_expect = intr[n]["K"][0, 0] * dw / intr[n]["size"][0]
        rec = {"fx_ratio": round(float(views[n]["K"][0, 0] / fx_expect), 5)}
        if Eda3 is not None:
            rec["rotation_deg"] = round(
                rotation_angle_deg(Eda3[:3, :3], Ecal[:3, :3]), 4)
            rec["centre_err_mm"] = round(float(np.linalg.norm(
                camera_centre(Eda3) - camera_centre(Ecal))) * 1e3, 2)
        else:
            rec["rotation_deg"] = None
            rec["centre_err_mm"] = None
        deltas[n] = rec
    print_poses(deltas)
    print("[pose ] fx_ratio compares DA3's returned fx against the calibrated "
          "fx scaled to this grid. It ignores the rectification crop, so read "
          "it as a coarse check only; the [warm ] line in da3_stream.py is "
          "the exact one.")

    sources = (["da3", "calib"] if args.pose == "both" else [args.pose])
    reports = []
    for src in sources:
        world = {}
        for n in names:
            E = ext[n]["E"] if src == "calib" else views[n]["E"]
            if E is None:
                print(f"[{src:<5}] {n}: no E_{n}.npy saved, using calibration")
                E = ext[n]["E"]
            world[n] = filtered_world(views[n], E, args)
        rep = audit(world, centres, args, src)
        print_audit(rep)
        reports.append(rep)

    if len(reports) == 2:
        a, b = reports[0], reports[1]
        print("\n[compare] da3 poses -> calibrated poses")
        for k in ("belt_offset_spread_mm", "belt_tilt_spread_deg",
                  "above_offset_spread_mm"):
            print(f"[compare] {k:<26} {a.get(k)}  ->  {b.get(k)}")
        print("[compare] if these fall substantially under the calibrated "
              "poses, the layering is a pose-substitution artefact and the "
              "fix is in run_inference, not in depth_affine.json.")

    verdict(reports[-1], deltas)

    out = {"capture": str(args.capture),
           "depth_source": "raw" if args.use_raw else "corrected",
           "depth_shape": [int(dh), int(dw)],
           "pose_deltas": deltas,
           "reports": reports,
           "settings": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()}}
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())