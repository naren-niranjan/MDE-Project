#!/usr/bin/env python3
"""
belt_extent.py — the deck's footprint in rig x/y, for rig_geometry.py --extent.

    python3 belt_extent.py --ply <registered scan>.ply

The cloud must already be in the RIG frame, same as extract_gt.py expects. If it
is raw scanner output, run scan_register.py first.

Finds the deck the same way extract_gt.py does (densest depth band, RANSAC
plane), takes the inliers, and reports their extent. It prints both a tight
percentile box and the hard min/max, because the hard box usually includes rails
and frame members sitting at deck height. Use the percentile box unless you have
checked that the extra is real belt.
"""
import argparse

import numpy as np
from scipy import ndimage

import rigkit as rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--deck", type=float, default=3.19,
                    help="approximate deck distance (m). The densest band is "
                         "NOT reliably the deck -- the floor is often flatter "
                         "and wins. Seed it. Resolution-dependent: 3.08 at "
                         "process-res 504, 3.19 at 1008.")
    ap.add_argument("--tol", type=float, default=0.008,
                    help="plane inlier tolerance (m)")
    ap.add_argument("--pct", type=float, default=1.0,
                    help="percentile trimmed off each end for the tight box")
    ap.add_argument("--cell", type=float, default=0.02,
                    help="mask cell size (m)")
    ap.add_argument("--no-fill", action="store_true",
                    help="do NOT fill parcel-shaped holes in the deck. The "
                         "default fills them, because a parcel occludes the "
                         "belt it stands on and you need coverage exactly "
                         "there.")
    ap.add_argument("--component", type=int, default=None,
                    help="which deck component to keep (default: largest)")
    ap.add_argument("--mask-out", default="belt_mask.npz",
                    help="write the chosen component as a mask for "
                         "rig_geometry.py --belt-mask")
    a = ap.parse_args()

    P = rk.load_cloud(a.ply)
    print(f"{len(P)} points")

    h_, e_ = np.histogram(P[:, 2], bins=200)
    c_ = 0.5 * (e_[:-1] + e_[1:])
    cand = np.flatnonzero(h_ > 0.15 * h_.max())
    if not len(cand):
        raise SystemExit("no dense band found")
    # fold adjacent bins into bands so the listing is surfaces, not bins
    bands, run = [], [cand[0]]
    for i in cand[1:]:
        (run.append(i) if i == run[-1] + 1 else (bands.append(run), run := [i]))
    bands.append(run)
    bz = [float(np.average(c_[b], weights=h_[b])) for b in bands]
    bn = [int(h_[b].sum()) for b in bands]
    k = int(bands[int(np.argmin(np.abs(np.array(bz) - a.deck)))]
            [len(bands[int(np.argmin(np.abs(np.array(bz) - a.deck)))]) // 2])
    pick = int(np.argmin(np.abs(np.array(bz) - a.deck)))
    print("dense bands (z, points):")
    for i, (z_, n_) in enumerate(zip(bz, bn)):
        print(f"  {z_:7.3f}  {n_:9d}" + ("   <- taken" if i == pick else ""))
    if len(bz) > 1:
        print("  separations (mm): "
              + "  ".join(f"{1000*(bz[i+1]-bz[i]):.0f}" for i in range(len(bz)-1)))
        print("  the deck and floor are the pair ~780 mm apart; anything nearer "
              "than the deck\n  is a parcel lid. Re-seed --deck if the wrong "
              "band was taken.")
    band = (P[:, 2] > e_[k] - 0.08) & (P[:, 2] < e_[k + 1] + 0.08)
    n, d = rk.fit_plane_ransac(P[band])
    inl = P[np.abs(P @ n + d) < a.tol]
    print(f"deck  d={d:.4f}  tilt={np.degrees(np.arccos(abs(n[2]))):.2f} deg  "
          f"{len(inl)} inliers at {1000*a.tol:.0f} mm")
    if len(inl) < 5000:
        raise SystemExit("too few deck inliers -- is the cloud registered?")

    # ---- the deck is a surface, not a rectangle -------------------------
    cell = a.cell
    x0 = np.floor(inl[:, 0].min() / cell) * cell
    y0 = np.floor(inl[:, 1].min() / cell) * cell
    iu = ((inl[:, 0] - x0) / cell).astype(int)
    iv = ((inl[:, 1] - y0) / cell).astype(int)
    g = np.zeros((iu.max() + 3, iv.max() + 3), bool)
    g[iu, iv] = True
    g = ndimage.binary_closing(g, np.ones((5, 5)))
    raw = g.copy()
    if not a.no_fill:
        g = ndimage.binary_fill_holes(g)
        print(f"\nfilled {(g & ~raw).sum()*cell*cell:.2f} m2 of holes "
              "(parcels occluding the deck they stand on)")
    lab, nl = ndimage.label(g, structure=np.ones((3, 3)))
    sizes = ndimage.sum(g, lab, range(1, nl + 1))
    order = np.argsort(-sizes)
    print(f"\n{nl} surfaces at deck height:")
    for r, ci in enumerate(order[:6]):
        m = lab == ci + 1
        u, v = np.nonzero(m)
        bx = (x0 + u.min() * cell, x0 + (u.max() + 1) * cell)
        by = (y0 + v.min() * cell, y0 + (v.max() + 1) * cell)
        fill = m.sum() / max((np.ptp(u) + 1) * (np.ptp(v) + 1), 1)
        print(f"  [{r}] {sizes[ci]*cell*cell:6.2f} m2   "
              f"x[{bx[0]:+.2f},{bx[1]:+.2f}] y[{by[0]:+.2f},{by[1]:+.2f}]   "
              f"fills {100*fill:3.0f}% of its own box")
    pick = order[0] if a.component is None else order[a.component]
    mask = lab == pick + 1
    u, v = np.nonzero(mask)
    lo = (x0 + u.min() * cell, y0 + v.min() * cell)
    hi = (x0 + (u.max() + 1) * cell, y0 + (v.max() + 1) * cell)
    print(f"\nkeeping component "
          f"{0 if a.component is None else a.component}: "
          f"{mask.sum()*cell*cell:.2f} m2, "
          f"box x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}]")
    if len(order) > 1 and sizes[order[1]] > 0.3 * sizes[order[0]]:
        print("  *** a second surface of comparable size exists at deck height. "
              "Check the\n      list above and pass --component if the wrong "
              "one was taken.")

    np.savez(a.mask_out, mask=mask, x0=x0, y0=y0, cell=cell,
             n=n, d=d, deck_z=float(-d / n[2] if abs(n[2]) > 1e-6 else 0.0))
    print(f"\nwrote {a.mask_out}")
    print(f"\n  python3 rig_geometry.py --calib <calib> "
          f"--belt-mask {a.mask_out}")


if __name__ == "__main__":
    main()