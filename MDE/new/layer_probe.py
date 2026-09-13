#!/usr/bin/env python3
"""
layer_probe.py

Decompose the layering seen in a fused cloud into its separable causes, so
that the correction applied is the one the data actually calls for.

Four mechanisms produce visually similar layering in a multi-view MDE fusion,
and they need different fixes:

  reflections     Points below the deck plane. The powered roller deck is
                  polished steel, so parcels reflect in it and the model
                  reconstructs the virtual image as geometry. Fixed by a
                  height cut, not by depth correction.

  per-view offset Each view puts the deck itself at a different height. A
                  constant additive bias per view, correctable by a single
                  scalar shift per camera.

  scale error     The views agree at the deck but diverge at parcel-top
                  height. The bias is proportional to depth, so a shift
                  fitted at the deck does not transfer upward. Correctable
                  in inverse-depth space, not in depth space.

  radial bias     Within one view the deck is reconstructed as a bowl, with
                  the residual varying systematically with field angle. No
                  scalar per view can flatten this; it needs a spatially
                  varying correction over a control-point grid.

The probe reports all four and states which dominates. It reads the
per-camera clouds written by da3_stream.py --save-per-camera, so run that
first on a settled scene.

Only numpy is required. Open3D is used for PLY reading when present, and a
built-in reader covers the binary and ASCII formats otherwise.

Example
-------
python layer_probe.py \
    --run runs/live_20260818_142300/frame_00000 \
    --calib-dir /home/jetson/Projects/Calibration_4_5/results
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


# --------------------------------------------------------------------------
# PLY reading
# --------------------------------------------------------------------------

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_builtin(path: Path) -> np.ndarray:
    """Minimal vertex reader for ascii and binary_little_endian PLY."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        count = None
        props = []
        in_vertex = False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            text = line.decode("ascii", "replace").strip()
            if text.startswith("format"):
                fmt = text.split()[1]
            elif text.startswith("element"):
                parts = text.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif text.startswith("property") and in_vertex:
                parts = text.split()
                if parts[1] == "list":
                    raise ValueError("list properties on vertices unsupported")
                props.append((parts[2], PLY_TYPES[parts[1]]))
            elif text == "end_header":
                break

        if count is None:
            raise ValueError(f"{path} declares no vertex element")
        names = [p[0] for p in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path} has no {axis} property")

        if fmt == "ascii":
            raw = np.loadtxt(fh, max_rows=count, ndmin=2)
            idx = [names.index(a) for a in ("x", "y", "z")]
            return raw[:, idx].astype(np.float64)

        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt!r}")
        dtype = np.dtype([(n, "<" + t) for n, t in props])
        data = np.frombuffer(fh.read(dtype.itemsize * count), dtype=dtype,
                             count=count)
        return np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)


def read_cloud(path: Path) -> np.ndarray:
    if o3d is not None:
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if pts.size:
            return pts.astype(np.float64)
    return read_ply_builtin(path)


# --------------------------------------------------------------------------
# plane fitting
# --------------------------------------------------------------------------

def fit_plane_ransac(points, thresh=0.010, iters=800, seed=0, subsample=200000):
    """
    Robust dominant plane. Returns (normal, offset, inlier_mask) with the
    plane defined as n . x = d and |n| = 1.
    """
    rng = np.random.default_rng(seed)
    pts = points
    if len(pts) > subsample:
        pts = pts[rng.choice(len(pts), subsample, replace=False)]

    best_n, best_d, best_count = None, None, -1
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        d = float(n @ a)
        count = int((np.abs(pts @ n - d) <= thresh).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count

    if best_n is None:
        raise SystemExit("plane fitting failed; the cloud may be empty")

    # Least-squares refit on the inliers of the full cloud.
    for _ in range(3):
        inl = np.abs(points @ best_n - best_d) <= thresh
        if inl.sum() < 3:
            break
        sel = points[inl]
        centroid = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
        best_n = vt[-1] / np.linalg.norm(vt[-1])
        best_d = float(best_n @ centroid)

    return best_n, best_d, np.abs(points @ best_n - best_d) <= thresh


def plane_fit_ls(points):
    """Least-squares plane through a small point set."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return n, float(n @ centroid)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def load_camera_frames(calib_dir: Path, names):
    """Camera centre and world-frame optical axis for each view."""
    path = calib_dir / "extrinsics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out = {}
    for n in names:
        if n not in data:
            continue
        R = np.asarray(data[n]["R"], np.float64).reshape(3, 3)
        t = np.asarray(data[n]["t"], np.float64).ravel()
        out[n] = {"C": -R.T @ t, "axis": R.T @ np.array([0.0, 0.0, 1.0])}
    return out or None


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def height_frame(clouds, thresh):
    """
    Fit the deck on the combined cloud and return a function mapping points to
    signed height above it, positive away from the cameras.
    """
    combined = np.vstack(list(clouds.values()))
    n, d, _ = fit_plane_ransac(combined, thresh=thresh)

    # Orient so that the majority of the cloud sits on the positive side of
    # the deck. Parcels are above the deck; reflections and the floor below.
    h = combined @ n - d
    if np.median(h[np.abs(h) > thresh]) < 0:
        n, d = -n, -d
    return n, d


def cell_index(points, size):
    """Integer XY cell keys after projecting onto the deck plane basis."""
    return np.floor(points / size).astype(np.int64)


def analyse(clouds, frames, args):
    n_deck, d_deck = height_frame(clouds, args.plane_thresh)
    print(f"[deck ] normal {np.round(n_deck, 4)}  offset {d_deck:.4f} m")

    # Build an in-plane basis so cells are measured along the deck, not along
    # world axes that may be tilted relative to it.
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(n_deck @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_deck, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_deck, e1)

    report = {"per_view": {}, "deck_normal": n_deck.tolist(),
              "deck_offset_m": float(d_deck)}
    heights, uv = {}, {}

    for name, pts in clouds.items():
        h = pts @ n_deck - d_deck
        heights[name] = h
        uv[name] = np.stack([pts @ e1, pts @ e2], axis=-1)

        below = h < -args.below_tol
        deck_band = np.abs(h) <= args.deck_band
        parcel_band = (h > args.min_parcel_height) & (h < args.max_parcel_height)

        rec = {
            "n_points": int(len(pts)),
            "below_deck_fraction": float(below.mean()),
            "deck_band_points": int(deck_band.sum()),
            "parcel_band_points": int(parcel_band.sum()),
        }

        if deck_band.sum() >= 100:
            hd = h[deck_band]
            rec["deck_bias_mm"] = float(np.median(hd) * 1e3)
            rec["deck_rms_mm"] = float(np.sqrt(np.mean(hd ** 2)) * 1e3)
            rec["deck_p95_spread_mm"] = float(
                (np.percentile(hd, 95) - np.percentile(hd, 5)) * 1e3)
            nv, dv = plane_fit_ls(pts[deck_band])
            if nv @ n_deck < 0:
                nv = -nv
            rec["deck_tilt_deg"] = float(
                np.degrees(np.arccos(np.clip(nv @ n_deck, -1, 1))))

            # Radial profile: median deck residual against field angle. A
            # bowl shows up here and cannot be removed by a scalar shift.
            if frames and name in frames:
                C, axis = frames[name]["C"], frames[name]["axis"]
                v = pts[deck_band] - C
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                ang = np.degrees(np.arccos(np.clip(v @ axis, -1, 1)))
                edges = np.linspace(0, max(ang.max(), 1e-6), args.radial_bins + 1)
                prof = []
                for i in range(args.radial_bins):
                    sel = (ang >= edges[i]) & (ang < edges[i + 1])
                    prof.append(float(np.median(hd[sel]) * 1e3)
                                if sel.sum() >= 30 else None)
                valid = [p for p in prof if p is not None]
                rec["radial_profile_mm"] = [None if p is None else round(p, 2)
                                            for p in prof]
                rec["radial_amplitude_mm"] = (float(max(valid) - min(valid))
                                              if len(valid) >= 2 else None)

        if parcel_band.sum() >= 100:
            hp = h[parcel_band]
            rec["parcel_median_height_mm"] = float(np.median(hp) * 1e3)

        report["per_view"][name] = rec

    # ---- inter-view comparison on shared cells -------------------------
    names = list(clouds)
    cells = {}
    for name in names:
        sel = ((heights[name] > args.min_parcel_height)
               & (heights[name] < args.max_parcel_height))
        if sel.sum() == 0:
            continue
        keys = cell_index(uv[name][sel], args.cell)
        h = heights[name][sel]
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys, h = keys[order], h[order]
        _, starts = np.unique(keys, axis=0, return_index=True)
        starts = np.sort(starts)
        bounds = np.append(starts, len(h))
        for i in range(len(starts)):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < args.min_cell_points:
                continue
            key = (int(keys[lo, 0]), int(keys[lo, 1]))
            cells.setdefault(key, {})[name] = float(np.median(h[lo:hi]))

    pairwise = {}
    for key, per_view in cells.items():
        vs = sorted(per_view)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                pairwise.setdefault(f"{vs[i]}|{vs[j]}", []).append(
                    (per_view[vs[i]] - per_view[vs[j]]) * 1e3)

    report["parcel_top_disagreement_mm"] = {
        pair: {"n_cells": len(v),
               "median_mm": round(float(np.median(v)), 2),
               "abs_median_mm": round(float(np.median(np.abs(v))), 2),
               "p95_abs_mm": round(float(np.percentile(np.abs(v), 95)), 2)}
        for pair, v in sorted(pairwise.items()) if len(v) >= args.min_cells}

    multi = sum(1 for v in cells.values() if len(v) >= 2)
    agree = sum(1 for v in cells.values() if len(v) >= 2
                and (max(v.values()) - min(v.values())) * 1e3 <= args.agree_tol_mm)
    report["cells"] = {
        "total": len(cells),
        "single_view": len(cells) - multi,
        "multi_view": multi,
        "multi_view_agreeing": agree,
        "single_view_fraction": round((len(cells) - multi) / max(len(cells), 1), 3),
        "agreement_tolerance_mm": args.agree_tol_mm,
    }
    return report


def verdict(report, args):
    """State which mechanism dominates and what to do about it."""
    pv = report["per_view"]
    lines = []

    below = {k: v["below_deck_fraction"] for k, v in pv.items()}
    worst_below = max(below.values()) if below else 0.0

    biases = [v["deck_bias_mm"] for v in pv.values() if "deck_bias_mm" in v]
    bias_spread = (max(biases) - min(biases)) if len(biases) >= 2 else 0.0

    tops = [v["parcel_median_height_mm"] for v in pv.values()
            if "parcel_median_height_mm" in v]
    top_spread = (max(tops) - min(tops)) if len(tops) >= 2 else 0.0

    amps = [v["radial_amplitude_mm"] for v in pv.values()
            if v.get("radial_amplitude_mm") is not None]
    worst_amp = max(amps) if amps else 0.0

    pair = report["parcel_top_disagreement_mm"]
    pair_med = (float(np.median([p["abs_median_mm"] for p in pair.values()]))
                if pair else 0.0)

    lines.append(f"reflections      {worst_below * 100:5.2f} % of the worst "
                 f"view lies below the deck")
    lines.append(f"per-view offset  deck bias spans {bias_spread:6.2f} mm "
                 f"across views")
    lines.append(f"scale error      parcel tops span {top_spread:6.2f} mm, "
                 f"pairwise cell disagreement {pair_med:6.2f} mm")
    lines.append(f"radial bias      worst within-view deck bow "
                 f"{worst_amp:6.2f} mm")

    actions = []
    if worst_below > 0.01:
        actions.append(
            "Clip below the deck. The roller surface is specular and the "
            "reflected parcels are being reconstructed as real geometry. A "
            "height cut at the deck plane removes them at no cost.")
    if bias_spread > 5.0:
        actions.append(
            f"Apply a per-view shift. The views disagree by {bias_spread:.1f} mm "
            f"at the deck itself, which a single scalar per camera removes.")
    if pair_med > 15.0 and pair_med > 2.0 * max(bias_spread, 1e-6):
        actions.append(
            f"Correct in inverse-depth space. Disagreement at parcel tops "
            f"({pair_med:.1f} mm) is far larger than at the deck "
            f"({bias_spread:.1f} mm), so the error scales with depth and a "
            f"shift fitted on the deck will not transfer to the tops.")
    if worst_amp > 10.0:
        actions.append(
            f"Fit a spatially varying correction. One view bows by "
            f"{worst_amp:.1f} mm across its field, so no per-view scalar can "
            f"flatten it. A control-point grid over the image plane, fitted "
            f"on the empty deck, is the minimum that will work.")
    if report["cells"]["multi_view"] and (
            report["cells"]["multi_view_agreeing"]
            / report["cells"]["multi_view"] < 0.5):
        actions.append(
            "Fuse by consensus rather than union. Fewer than half the "
            "multi-view cells agree within tolerance, so concatenating the "
            "clouds preserves every disagreement as a visible layer. Take a "
            "per-cell median across agreeing views instead.")
    if not actions:
        actions.append("No single mechanism dominates at the thresholds used. "
                       "Lower --agree-tol-mm and re-run, or check that the "
                       "scene really was settled when captured.")
    return lines, actions


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose layering in a multi-view fused cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", type=Path,
                    help="directory holding cloud_<name>.ply")
    ap.add_argument("--cloud", action="append", default=[],
                    metavar="NAME=PATH",
                    help="explicit cloud, repeatable; overrides --run")
    ap.add_argument("--calib-dir", type=Path,
                    help="extrinsics.json, enabling the radial profile")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])

    ap.add_argument("--plane-thresh", type=float, default=0.010,
                    help="RANSAC inlier distance for the deck plane, metres")
    ap.add_argument("--deck-band", type=float, default=0.015,
                    help="points within this height of the deck are treated "
                         "as deck samples, metres")
    ap.add_argument("--below-tol", type=float, default=0.020,
                    help="points this far below the deck count as reflections")
    ap.add_argument("--min-parcel-height", type=float, default=0.030)
    ap.add_argument("--max-parcel-height", type=float, default=1.000)
    ap.add_argument("--cell", type=float, default=0.020,
                    help="XY cell size for inter-view comparison, metres")
    ap.add_argument("--min-cell-points", type=int, default=8)
    ap.add_argument("--min-cells", type=int, default=20,
                    help="cells a pair needs before it is reported")
    ap.add_argument("--agree-tol-mm", type=float, default=10.0)
    ap.add_argument("--radial-bins", type=int, default=8)
    ap.add_argument("--json", type=Path, default=Path("layer_probe.json"))
    args = ap.parse_args()

    paths = {}
    if args.cloud:
        for spec in args.cloud:
            if "=" not in spec:
                raise SystemExit(f"--cloud expects NAME=PATH, got {spec!r}")
            name, p = spec.split("=", 1)
            paths[name] = Path(p)
    elif args.run:
        for n in args.cameras:
            p = args.run / f"cloud_{n}.ply"
            if p.exists():
                paths[n] = p
    else:
        raise SystemExit("pass --run or at least two --cloud arguments")

    if len(paths) < 2:
        raise SystemExit(f"need at least two clouds, found {sorted(paths)}; "
                         f"re-run da3_stream.py with --save-per-camera")

    clouds = {}
    for name, p in paths.items():
        pts = read_cloud(p)
        if len(pts) < 1000:
            print(f"[warn ] {name} has only {len(pts)} points")
        clouds[name] = pts
        print(f"[load ] {name:<7} {len(pts):>8d} points  {p}")

    frames = load_camera_frames(args.calib_dir, list(clouds)) \
        if args.calib_dir else None
    if frames is None:
        print("[note ] no --calib-dir, skipping the radial profile")

    print()
    report = analyse(clouds, frames, args)

    print()
    for name, rec in report["per_view"].items():
        bias = rec.get("deck_bias_mm")
        rms = rec.get("deck_rms_mm")
        tilt = rec.get("deck_tilt_deg")
        amp = rec.get("radial_amplitude_mm")
        top = rec.get("parcel_median_height_mm")
        print(f"[view ] {name:<7} deck bias "
              + (f"{bias:7.2f} mm" if bias is not None else "     n/a")
              + "  rms " + (f"{rms:6.2f} mm" if rms is not None else "   n/a")
              + "  tilt " + (f"{tilt:5.3f} deg" if tilt is not None else "  n/a")
              + "  bow " + (f"{amp:6.2f} mm" if amp is not None else "   n/a")
              + "  top " + (f"{top:7.2f} mm" if top is not None else "    n/a")
              + f"  below deck {rec['below_deck_fraction'] * 100:5.2f} %")

    print()
    for pair, rec in report["parcel_top_disagreement_mm"].items():
        print(f"[pair ] {pair:<16} cells {rec['n_cells']:>5d}  median "
              f"{rec['median_mm']:7.2f} mm  |median| {rec['abs_median_mm']:6.2f} mm"
              f"  p95 {rec['p95_abs_mm']:7.2f} mm")

    c = report["cells"]
    print(f"\n[cells] {c['total']} occupied, {c['multi_view']} seen by two or "
          f"more views, {c['multi_view_agreeing']} of those agreeing within "
          f"{c['agreement_tolerance_mm']:.0f} mm  "
          f"(single-view fraction {c['single_view_fraction']:.1%})")

    lines, actions = verdict(report, args)
    print("\n--- mechanism ---")
    for line in lines:
        print("  " + line)
    print("\n--- what to do ---")
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action}")

    report["verdict"] = {"summary": lines, "actions": actions}
    args.json.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())