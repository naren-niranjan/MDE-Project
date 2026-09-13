#!/usr/bin/env python3
"""
inplane_offset.py

Measures the three degrees of freedom that a deck-plane anchor is structurally
blind to.

Anchoring each camera to the ChArUco deck constrains exactly three DOF: the
standoff and the two tilt axes. A plane is invariant under translation within
itself and under rotation about its own normal, so in-plane translation (dx,
dy) and yaw are completely unconstrained by it. layer_collapse can drive the
deck RMS to 4 mm and still leave the parcels displaced by hundreds of mm
sideways -- which is what "53-57 % of above-deck points have no neighbour
within 250 mm" means. That is not sheet separation and no voxel setting
touches it.

This script projects each camera's above-deck points into the deck plane,
rasterises them as occupancy, and recovers each camera's (dx, dy, yaw)
relative to the reference by yaw search plus FFT phase correlation.

Read the result as a MEASUREMENT of extrinsic error, not as a correction to
apply. If the offsets are large and structured, the fix belongs in
Calibration_4_5, not in a fusion-time nudge -- start with
extrinsics_consistency.json. Baking these numbers into the pipeline would hide
a calibration fault behind a plausible-looking cloud.

Input
-----
Per-camera world-frame clouds plus the deck plane, as written by
layer_collapse.py --dump-per-camera.

Example
-------
  python inplane_offset.py \
      --dir runs/live_20260817_102732/capture_00002/percam \
      --reference center
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d not found. Activate the da3 environment.")


def plane_basis(n):
    """Orthonormal (e1, e2) spanning the plane with normal n."""
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(n @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def rasterise(uv, origin, cell, shape):
    """Binary occupancy grid of in-plane coordinates."""
    g = np.zeros(shape, dtype=np.float64)
    ij = np.floor((uv - origin) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & \
         (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    ij = ij[ok]
    if len(ij):
        g[ij[:, 0], ij[:, 1]] = 1.0
    return g


def phase_correlate(a, b):
    """Shift (in cells) that best aligns b onto a, plus the peak score."""
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    cross = A * np.conj(B)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    r = np.fft.irfft2(cross / mag, s=a.shape)
    peak = np.unravel_index(np.argmax(r), r.shape)
    shift = np.array(peak, dtype=float)
    for k in (0, 1):
        if shift[k] > a.shape[k] / 2:
            shift[k] -= a.shape[k]
    return shift, float(r[peak])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory holding cloud_<cam>.ply and deck_plane.json")
    ap.add_argument("--reference", default=None,
                    help="defaults to the reference recorded in deck_plane.json")
    ap.add_argument("--above-min", type=float, default=0.03)
    ap.add_argument("--above-max", type=float, default=0.40)
    ap.add_argument("--cell", type=float, default=0.005, help="grid cell (m)")
    ap.add_argument("--yaw-range", type=float, default=6.0, help="+/- deg")
    ap.add_argument("--yaw-step", type=float, default=0.5, help="deg")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    meta = json.loads((d / "deck_plane.json").read_text())
    n_w = np.asarray(meta["normal"], dtype=float)
    d_w = float(meta["d"])
    ref = args.reference or meta.get("reference")

    e1, e2 = plane_basis(n_w)

    uvs = {}
    for p in sorted(d.glob("cloud_*.ply")):
        cam = p.stem[len("cloud_"):]
        P = np.asarray(o3d.io.read_point_cloud(str(p)).points)
        if len(P) == 0:
            continue
        s = P @ n_w + d_w
        sel = (s > args.above_min) & (s < args.above_max)
        if sel.sum() < 200:
            print(f"{cam:<7} only {int(sel.sum())} above-deck points, skipped")
            continue
        Q = P[sel]
        uvs[cam] = np.stack([Q @ e1, Q @ e2], axis=1)
        print(f"{cam:<7} {len(Q)} above-deck points")

    if ref not in uvs:
        raise SystemExit(f"reference {ref!r} has no usable cloud; have {sorted(uvs)}")

    allp = np.concatenate(list(uvs.values()), axis=0)
    lo = allp.min(axis=0) - 0.5
    hi = allp.max(axis=0) + 0.5
    shape = tuple(int(np.ceil((hi[k] - lo[k]) / args.cell)) for k in (0, 1))
    if max(shape) > 3000:
        raise SystemExit(f"grid {shape} too large; raise --cell")
    print(f"\ngrid {shape} at {args.cell * 1000:.0f} mm, extent "
          f"{hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} m\n")

    ref_grid = rasterise(uvs[ref], lo, args.cell, shape)
    ref_c = uvs[ref].mean(axis=0)

    out = {"reference": ref, "cell_m": args.cell, "cameras": {}}
    yaws = np.arange(-args.yaw_range, args.yaw_range + 1e-9, args.yaw_step)

    for cam, uv in sorted(uvs.items()):
        if cam == ref:
            continue
        best = None
        for yaw in yaws:
            th = np.radians(yaw)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            rot = (uv - ref_c) @ R.T + ref_c
            g = rasterise(rot, lo, args.cell, shape)
            shift, score = phase_correlate(ref_grid, g)
            if best is None or score > best[2]:
                best = (yaw, shift, score)
        yaw, shift, score = best
        dx, dy = shift * args.cell
        mag = float(np.hypot(dx, dy))
        out["cameras"][cam] = {
            "dx_mm": float(dx * 1000.0), "dy_mm": float(dy * 1000.0),
            "magnitude_mm": mag * 1000.0, "yaw_deg": float(yaw),
            "peak_score": score,
            "yaw_at_search_edge": bool(abs(abs(yaw) - args.yaw_range) < 1e-6),
        }
        flag = "  <-- yaw at search edge, widen --yaw-range" \
            if out["cameras"][cam]["yaw_at_search_edge"] else ""
        print(f"{cam:<7} dx {dx * 1000:+8.1f} mm  dy {dy * 1000:+8.1f} mm  "
              f"|d| {mag * 1000:7.1f} mm  yaw {yaw:+.2f} deg{flag}")

    mags = [v["magnitude_mm"] for v in out["cameras"].values()]
    if mags:
        worst = max(mags)
        out["worst_inplane_mm"] = worst
        print()
        if worst < 10.0:
            print(f"Worst in-plane offset {worst:.1f} mm. The residual is not "
                  f"in-plane;\nlook at coverage and at genuinely non-overlapping "
                  f"object faces instead.")
        else:
            print(f"Worst in-plane offset {worst:.1f} mm. This is extrinsic error "
                  f"that the deck\nanchor cannot see and ICP with a 20 mm "
                  f"correspondence cap cannot reach.\nFix it in Calibration_4_5 "
                  f"(start with extrinsics_consistency.json) rather\nthan "
                  f"correcting it at fusion time.")

    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=2))
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()