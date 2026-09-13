#!/usr/bin/env python3
"""
extract_gt.py — turn a reference-scanner point cloud into gt_scene.json.

Run this once per parcel layout. gt_scene.json for the 20260707 scene is
already in this folder; you need this script when you collect new layouts.

    python3 extract_gt.py --ply pointcloud_20260707_122811.ply --out gt_scene.json

The cloud must already be in the RIG frame (center camera). If it is raw
scanner output, run scan_register.py first -- and rerun scan_register.py after
any lens change, because the center camera almost certainly moved.

What it writes:
  deck_plane, floor_plane  -- the alpha/beta fit surfaces for grade_subsets.py
  parcels                  -- id, deck-plane centre, L/W/H, top-face corners

Parcels are found in the 30..700 mm band above the deck. Long thin components
(> 800 mm on the long axis, or < 150 mm on the short) are dropped as rails and
frame members. Check the printed table before trusting the file.
"""
import argparse
import json
import numpy as np
from scipy import ndimage

import rigkit as rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default="gt_scene.json")
    ap.add_argument("--band", type=float, nargs=2, default=(0.03, 0.70),
                    help="height band above deck to search for parcels (m)")
    ap.add_argument("--min-pts", type=int, default=5000)
    ap.add_argument("--max-long-mm", type=float, default=800)
    ap.add_argument("--min-short-mm", type=float, default=150)
    ap.add_argument("--res", type=float, default=0.01)
    a = ap.parse_args()

    P = rk.load_cloud(a.ply)
    print(f"{len(P)} points   x[{P[:,0].min():+.2f},{P[:,0].max():+.2f}] "
          f"y[{P[:,1].min():+.2f},{P[:,1].max():+.2f}] "
          f"z[{P[:,2].min():+.2f},{P[:,2].max():+.2f}]")

    # deck: densest depth band
    h_, e_ = np.histogram(P[:, 2], bins=200)
    k = int(h_.argmax())
    band = (P[:, 2] > e_[k] - 0.08) & (P[:, 2] < e_[k + 1] + 0.08)
    dn, dd = rk.fit_plane_ransac(P[band])
    print(f"deck   n={np.round(dn,5)} d={dd:.4f}  "
          f"tilt={np.degrees(np.arccos(abs(dn[2]))):.2f} deg  "
          f"inliers={(np.abs(P@dn+dd)<0.006).sum()}")

    hgt = -(P @ dn + dd)
    lo = P[hgt < -0.5]
    if len(lo) < 5000:
        raise SystemExit("no surface found below the deck -- floor missing from "
                         "the scan? grade_subsets.py needs it for the scale fit.")
    fn, fd = rk.fit_plane_svd(lo)
    sep = 1000 * abs(fd - dd)
    tilt = np.degrees(np.arccos(abs(fn @ dn)))
    print(f"floor  n={np.round(fn,5)} d={fd:.4f}  sep={sep:.1f} mm  "
          f"tilt_vs_deck={tilt:.2f} deg")
    if tilt > 3.0:
        print("  WARNING: floor tilt > 3 deg. A shared linear depth model cannot "
              "express it; consider deck-only fitting (weaker but honest).")

    gt0 = {"deck_plane": {"n": dn.tolist(), "d": float(dd)}}
    n, d, ex, ey = rk.deck_frame(gt0)
    Q = rk.to_deck(P, gt0)
    m = (Q[:, 2] > a.band[0]) & (Q[:, 2] < a.band[1])
    B = Q[m]
    u = ((B[:, 0] - B[:, 0].min()) / a.res).astype(int)
    v = ((B[:, 1] - B[:, 1].min()) / a.res).astype(int)
    g = np.zeros((u.max() + 3, v.max() + 3), bool)
    g[u, v] = True
    g = ndimage.binary_closing(g, np.ones((3, 3)))
    lab, nl = ndimage.label(g, structure=np.ones((3, 3)))
    cl = lab[u, v]

    par, rejected = [], []
    for kk in range(1, nl + 1):
        pts = B[cl == kk]
        if len(pts) < a.min_pts:
            continue
        c2 = pts[:, :2].mean(0)
        A = pts[:, :2] - c2
        _, V = np.linalg.eigh(np.cov(A.T))
        pr = A @ V
        L = np.percentile(pr[:, 1], 99) - np.percentile(pr[:, 1], 1)
        W = np.percentile(pr[:, 0], 99) - np.percentile(pr[:, 0], 1)
        H = np.percentile(pts[:, 2], 97)
        if L * 1000 > a.max_long_mm or W * 1000 < a.min_short_mm:
            rejected.append((c2, L, W, len(pts)))
            continue
        cor = np.array([[-W / 2, -L / 2], [W / 2, -L / 2],
                        [W / 2, L / 2], [-W / 2, L / 2]]) @ V.T + c2
        par.append(dict(
            x=float(c2[0]), y=float(c2[1]),
            L_mm=float(L * 1000), W_mm=float(W * 1000), H_mm=float(H * 1000),
            n_pts=int(len(pts)),
            corners_xyz=rk.from_deck(cor[:, 0], cor[:, 1],
                                     np.full(4, H), gt0).tolist(),
            top_centre_xyz=rk.from_deck([c2[0]], [c2[1]], [H], gt0)[0].tolist()))
    par.sort(key=lambda p: -p["x"])
    for i, p in enumerate(par, 1):
        p["id"] = f"P{i}"

    print(f"\n{len(par)} parcels ({len(rejected)} components rejected as rails/frame)")
    print(f"  {'id':4s} {'x':>7s} {'y':>7s} {'L':>7s} {'W':>7s} {'H':>7s}  pts")
    for p in par:
        print(f"  {p['id']:4s} {p['x']:+7.3f} {p['y']:+7.3f} "
              f"{p['L_mm']:7.1f} {p['W_mm']:7.1f} {p['H_mm']:7.1f}  {p['n_pts']}")
    for c2, L, W, n_ in rejected:
        print(f"  [rej] at ({c2[0]:+.2f},{c2[1]:+.2f}) {L*1000:.0f}x{W*1000:.0f} mm, {n_} pts")

    json.dump(dict(source=a.ply, frame="rig (center camera)",
                   deck_plane=dict(n=dn.tolist(), d=float(dd),
                                   tilt_deg=float(np.degrees(np.arccos(abs(dn[2]))))),
                   floor_plane=dict(n=fn.tolist(), d=float(fd),
                                    sep_mm=float(sep), tilt_vs_deck_deg=float(tilt)),
                   parcels=par),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("Next: coverage.py to find which parcels are clipped, then "
          "verify_gt_alignment.py to confirm the scan matches the frames.")


if __name__ == "__main__":
    main()