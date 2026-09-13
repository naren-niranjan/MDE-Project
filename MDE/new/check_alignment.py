#!/usr/bin/env python3
"""
check_alignment.py

Is the ground truth actually sitting on the conveyor?

WHY THIS EXISTS
---------------
grade_subsets.py compares heights above the deck, so everything it reports rests
on one assumption: that the transform in gt_align.json puts the ground truth's
conveyor surface on the frozen belt plane. If it instead put the ground truth's
FLOOR there, every height is out by the deck-to-floor gap, the fitted line
collapses to p near zero and q near that gap, and the table reads as though
every subset failed identically. A constant q across runs whose gauges differ by
a factor of two is the signature: a per-run fault cannot produce a shared
offset.

The deck column being empty in every row says the same thing more directly.
grade_subsets fills it only when more than 500 reconstruction points lie within
the deck band, so an empty column means nothing is near the plane it is
measuring from.

WHAT THIS PRINTS
----------------
For the ground truth and for one reconstruction, in the frozen belt frame:

  * the height distribution inside the belt footprint. The deck surface should
    pile up at zero and the parcels should sit above it, with nothing far below
    except floor seen past the belt edge.
  * the dominant plane inside that footprint and its offset from the belt plane.
    Zero is right. Something near your deck-to-floor gap means the wrong surface
    was matched.
  * how much of each cloud lies within the deck band, which is the number
    grade_subsets needs and is not getting.

It also writes gt_in_rig.ply so the two can be opened together. Do that before
reading another grade: a misalignment of this size is obvious by eye in ten
seconds and invisible in a table of summary statistics.

EXAMPLE
-------
    python3 check_alignment.py --gt gt/pointcloud_20260826_095616.ply \\
        --align gt_align.json --belt belt.json \\
        --run runs_da3/scene_b/center+left+top+right_res1008

Keep this file beside rigkit.py and belt.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")


def belt_frame(belt):
    ex = np.asarray(belt["x_axis"], float)
    ey = np.asarray(belt["y_axis"], float)
    ex /= np.linalg.norm(ex)
    ey = ey - ex * float(ey @ ex)
    ey /= np.linalg.norm(ey)
    n = np.cross(ex, ey)
    n /= np.linalg.norm(n)
    centre = np.asarray(belt["centre_m"], float)
    if float(n @ (np.zeros(3) - centre)) < 0:
        n, ey = -n, -ey
    return centre, ex, ey, n


def report(name, P, frame, belt, args):
    centre, ex, ey, n = frame
    rel = P - centre
    u, v, h = rel @ ex, rel @ ey, rel @ n
    inside = ((np.abs(u) <= belt["length_m"] / 2 + args.margin)
              & (np.abs(v) <= belt["width_m"] / 2 + args.margin))
    print(f"\n[{name}] {len(P)} points, {int(inside.sum())} inside the belt "
          f"footprint")
    if inside.sum() < 200:
        print(f"[{name}] almost nothing lands on the belt in plan. The "
              f"transform is wrong in X or Y, not just in height.")
        return
    hi = h[inside] * 1e3
    qs = np.percentile(hi, [1, 5, 25, 50, 75, 95, 99])
    print(f"[{name}] height above the belt plane, mm: "
          + "  ".join(f"p{p}={q:+.0f}" for p, q in
                      zip((1, 5, 25, 50, 75, 95, 99), qs)))
    band = int(np.count_nonzero(np.abs(hi) <= args.deck_band_mm))
    print(f"[{name}] within +/-{args.deck_band_mm:.0f} mm of the belt plane: "
          f"{band} points, {100 * band / inside.sum():.1f} per cent")

    lo, hiv = -1200.0, 700.0
    nb = 32
    counts, edges = np.histogram(np.clip(hi, lo, hiv), bins=nb,
                                 range=(lo, hiv))
    peak = max(counts.max(), 1)
    print(f"[{name}] distribution (each row 60 mm):")
    for c, e in zip(counts, edges[:-1]):
        if c == 0:
            continue
        mark = " <- belt plane" if abs(e) < 60 else ""
        print(f"[{name}]   {e:+6.0f} {'#' * int(40 * c / peak):<40s}"
              f" {c:8d}{mark}")

    Q = P[inside]
    nd, dd = rigkit.fit_plane_ransac(Q, tol=args.plane_tol, trials=400)
    inl = np.abs(Q @ nd + dd) < args.plane_tol
    off = float(np.median((Q[inl] - centre) @ n)) * 1e3
    tilt = np.degrees(np.arccos(np.clip(abs(float(nd @ n)), 0, 1)))
    print(f"[{name}] dominant plane inside the footprint holds "
          f"{100 * inl.sum() / len(Q):.0f} per cent of those points, sits "
          f"{off:+.0f} mm from the belt plane, tilted {tilt:.2f} deg")
    return off


def raster_heights(P, frame, belt, cell, margin):
    centre, ex, ey, n = frame
    rel = P - centre
    u, v, h = rel @ ex, rel @ ey, rel @ n
    L, W = belt["length_m"] / 2 + margin, belt["width_m"] / 2 + margin
    ok = (np.abs(u) <= L) & (np.abs(v) <= W)
    nv = int(2 * W / cell) + 1
    iu = np.clip(((u[ok] + L) / cell).astype(np.int64), 0,
                 int(2 * L / cell))
    iv = np.clip(((v[ok] + W) / cell).astype(np.int64), 0, nv - 1)
    key = iu * nv + iv
    order = np.argsort(key, kind="stable")
    key, hs = key[order], h[ok][order]
    edges = np.flatnonzero(np.diff(key)) + 1
    out = {}
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(key)]):
        out[int(key[a])] = float(np.median(hs[a:b]))
    return out


def yaw_sweep(gt_r, P, frame, belt, args):
    """Try the four orientations the registration had to choose between.

    A conveyor is a long rectangle and is very nearly symmetric under a 180
    degree yaw, so a wrong hypothesis survives ICP: the footprints still
    overlap, the deck still matches the deck, and every parcel sits on top of a
    different parcel. Plan-view overlap therefore cannot tell the four apart.
    Correlation of HEIGHT can, because it is the parcels that break the
    symmetry. Whichever orientation drives p towards 1 is the right one.
    """
    centre, ex, ey, n = frame
    B = raster_heights(P, frame, belt, args.cell, args.margin)
    print(f"\n{'orientation':22s} {'shared cells':>12s} {'p':>8s} "
          f"{'q mm':>9s} {'resid rms mm':>13s}")
    best, best_p = None, -np.inf
    for label, Ryaw, flip_n in (("as registered", 0.0, False),
                                ("yaw +180", np.pi, False),
                                ("normal flipped", 0.0, True),
                                ("yaw +180, normal", np.pi, True)):
        axis = n if not flip_n else n
        c, s = np.cos(Ryaw), np.sin(Ryaw)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        Q = (gt_r - centre) @ R.T + centre
        if flip_n:
            rel = Q - centre
            Q = centre + rel - 2.0 * np.outer(rel @ n, n)
        A = raster_heights(Q, frame, belt, args.cell, args.margin)
        keys = sorted(set(A) & set(B))
        if len(keys) < 200:
            print(f"{label:22s} {len(keys):>12d}   too few shared cells")
            continue
        x = np.array([A[k] for k in keys])
        y = np.array([B[k] for k in keys])
        M = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        resid = y - M @ coef
        print(f"{label:22s} {len(keys):>12d} {coef[0]:>8.4f} "
              f"{coef[1] * 1e3:>+9.1f} "
              f"{np.sqrt(np.mean(resid ** 2)) * 1e3:>13.1f}")
        if coef[0] > best_p:
            best, best_p = label, coef[0]
    print(f"\nbest orientation by height correlation: {best} (p = {best_p:.4f})")
    if best != "as registered":
        print("The frozen transform took the wrong one. Delete gt_align.json "
              "and re-solve, or pin it with --gt-pairs; every grade computed "
              "against it is meaningless.")
    elif best_p < 0.4:
        print("Even the best orientation barely correlates, so the fault is "
              "not a yaw flip. Open the two clouds together before going on.")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify that the ground-truth transform puts the conveyor "
                    "on the conveyor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--align", type=Path, default=Path("gt_align.json"))
    ap.add_argument("--belt", type=Path, default=Path("belt.json"))
    ap.add_argument("--run", type=Path, default=None,
                    help="a run directory, for the reconstruction side")
    ap.add_argument("--fused-name", default="fused_calib.ply")
    ap.add_argument("--deck-floor-gap-mm", type=float, default=800.0)
    ap.add_argument("--deck-band-mm", type=float, default=15.0)
    ap.add_argument("--plane-tol", type=float, default=0.012)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--cell", type=float, default=0.010,
                    help="belt raster cell, matching grade_subsets")
    ap.add_argument("--no-yaw-sweep", dest="yaw_sweep", action="store_false",
                    default=True)
    ap.add_argument("--out", type=Path, default=Path("gt_in_rig.ply"))
    args = ap.parse_args()

    belt = json.loads(args.belt.read_text())
    frame = belt_frame(belt)
    print(f"belt {belt['length_m'] * 1e3:.0f} x {belt['width_m'] * 1e3:.0f} mm, "
          f"centre {np.round(frame[0], 4)}, normal {np.round(frame[3], 4)}")

    gt = rigkit.load_cloud(str(args.gt))
    if args.align.exists():
        T = np.asarray(json.loads(args.align.read_text())["T"], float)
        gt_r = gt @ T[:3, :3].T + T[:3, 3]
        print(f"applied the transform in {args.align}")
    else:
        gt_r = gt
        print(f"[warn ] {args.align} not found, reporting the ground truth in "
              f"its own frame")

    off_gt = report("gt   ", gt_r, frame, belt, args)
    rigkit.save_ply(str(args.out), gt_r)
    print(f"\nwritten {args.out}")

    off_re = None
    if args.run:
        p = args.run / args.fused_name
        if p.exists():
            off_re = report("recon", rigkit.load_cloud(str(p)), frame, belt,
                            args)
        else:
            print(f"[warn ] {p} not found")

    if args.run and args.yaw_sweep:
        p_ = args.run / args.fused_name
        if p_.exists():
            yaw_sweep(gt_r, rigkit.load_cloud(str(p_)), frame, belt, args)

    print("\nVERDICT")
    for name, off in (("ground truth", off_gt), ("reconstruction", off_re)):
        if off is None:
            continue
        g = args.deck_floor_gap_mm
        if abs(off) < 40:
            print(f"  {name}: dominant surface is on the belt plane. Good.")
        elif abs(abs(off) - g) < 0.25 * g:
            print(f"  {name}: dominant surface sits {off:+.0f} mm off, which is "
                  f"the deck-to-floor gap. The FLOOR was matched to the belt. "
                  f"Re-solve with --realign and a different --gt-plane-index, "
                  f"or pin it by hand with --gt-pairs.")
        else:
            print(f"  {name}: dominant surface sits {off:+.0f} mm from the belt "
                  f"plane, which is neither zero nor the floor gap. Open "
                  f"{args.out} beside the cloud and look before going further.")
    if off_gt is not None and off_re is not None:
        print(f"  the two differ by {off_re - off_gt:+.0f} mm; that difference "
              f"is what grade_subsets reports as q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())