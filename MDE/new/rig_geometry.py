#!/usr/bin/env python3
"""
rig_geometry.py — everything about a camera subset that is fixed by the
calibration, for any number of cameras. No inference, no scan, no GPU.

    python3 rig_geometry.py --calib /home/jetson/Projects/Calibration_4_5/results
    python3 rig_geometry.py --calib ... --deck 3.19 --heights 0 0.1 0.2 0.3 0.4 0.5

Reports, per subset:
  * pose-prior conditioning (s2/s1 of the camera centres). DA3's
    _align_to_input_extrinsics_intrinsics runs Umeyama Sim(3) on the centres and
    needs rank >= 2, so a collinear set cannot carry scale no matter how many
    cameras are in it. This is the number that decides whether a subset can be
    run with --pose-prior always at all.
  * coverage at a sequence of parcel-top heights, under the production
    >= 2-view rule (>= 1 view for single-camera subsets).
  * baseline decomposition into lateral and axial components. An axial baseline
    is near-useless for triangulation but is what breaks collinearity, so the
    two columns pull in opposite directions and must be read together.

Per camera it also reports roll about the optical axis, because gravity
canonicalisation (--upright) is worth several resolved layers on a non-upright
camera, and a subset ranking run without it is partly a ranking of mount angles.
"""
import argparse
import itertools
import json
import os

import cv2
import numpy as np


def load(calib_dir, cams=None):
    ext = json.load(open(os.path.join(calib_dir, "extrinsics.json")))
    cams = cams or [c for c in ext]
    R, t, K, D, size = {}, {}, {}, {}, {}
    for c in cams:
        p = os.path.join(calib_dir, f"intrinsics_{c}.json")
        if not os.path.exists(p):
            p = os.path.join(calib_dir, f"{c}_intrinsics.json")
        j = json.load(open(p))
        K[c] = np.array(j["camera_matrix"], float)
        D[c] = np.array(j["dist_coeffs"], float).ravel()
        size[c] = tuple(j.get("image_size", (2448, 2048)))
        R[c] = np.array(ext[c]["R"], float)
        t[c] = np.array(ext[c]["t"], float).ravel()
    return cams, R, t, K, D, size


def centres(cams, R, t):
    return {c: -R[c].T @ t[c] for c in cams}


def conditioning(sub, C):
    """s2/s1 of the mean-centred camera centres. 0 == collinear == the pose
    prior cannot recover scale. Two cameras are always collinear."""
    P = np.array([C[c] for c in sub])
    if len(P) < 3:
        return 0.0
    P = P - P.mean(0)
    s = np.linalg.svd(P, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def sees(c, X, R, t, K, D, size):
    W, H = size[c]
    Xc = X @ R[c].T + t[c]
    px = cv2.projectPoints(X.astype(np.float64), cv2.Rodrigues(R[c])[0],
                           t[c].astype(np.float64), K[c], D[c])[0].reshape(-1, 2)
    return ((Xc[:, 2] > 0.1) & (px[:, 0] >= 0) & (px[:, 0] < W)
            & (px[:, 1] >= 0) & (px[:, 1] < H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cams", nargs="+")
    ap.add_argument("--deck", type=float, default=3.19,
                    help="deck distance along the reference optical axis (m). "
                         "This is resolution-dependent -- 3.08 at process-res "
                         "504, 3.19 at 1008. Pass the one you are running.")
    ap.add_argument("--heights", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--extent", type=float, nargs=4,
                    default=[-1.6, 1.6, -2.2, 2.2],
                    help="xmin xmax ymin ymax of the belt region to test")
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--belt-mask",
                    help="belt_mask.npz from belt_extent.py. Samples the actual "
                         "deck surface on its true tilted plane, and overrides "
                         "--deck and --extent.")
    a = ap.parse_args()

    cams, R, t, K, D, size = load(a.calib, a.cams)
    C = centres(cams, R, t)

    print("cameras (rig frame = reference camera)")
    for c in cams:
        ax = R[c][2, :]
        tilt = np.degrees(np.arccos(np.clip(ax @ np.array([0, 0, 1.0]), -1, 1)))
        roll = np.degrees(np.arctan2(R[c][0, 1], R[c][0, 0]))
        print(f"  {c:8s} C=({C[c][0]:+.4f},{C[c][1]:+.4f},{C[c][2]:+.4f})  "
              f"axis {tilt:5.2f} deg off ref   roll {roll:+7.1f} deg   "
              f"fx={K[c][0,0]:8.1f}")
    nonup = [c for c in cams if abs(np.degrees(np.arctan2(R[c][0, 1], R[c][0, 0]))) > 20]
    if nonup:
        print(f"  *** non-upright mounts: {', '.join(nonup)}. Run every subset "
              "with --upright or the\n      ranking is partly a ranking of roll "
              "angle.")

    print("\nbaselines (lateral component triangulates; axial component "
          "conditions the pose prior)")
    for x, y in itertools.combinations(cams, 2):
        b = C[x] - C[y]
        ax_, lat = abs(b[2]), float(np.hypot(b[0], b[1]))
        print(f"  {x:8s}-{y:8s} |b|={np.linalg.norm(b):.4f}  lateral={lat:.4f}"
              f"  axial={ax_:.4f}  ({100*ax_/np.linalg.norm(b):4.1f}% axial)")

    if a.belt_mask:
        z = np.load(a.belt_mask)
        m, c0 = z["mask"], float(z["cell"])
        u, v = np.nonzero(m)
        px = z["x0"] + (u + 0.5) * c0
        py = z["y0"] + (v + 0.5) * c0
        dn, dd = z["n"], float(z["d"])
        pz = -(dd + dn[0] * px + dn[1] * py) / dn[2]
        base = np.stack([px, py, pz], 1)
        area_cell = c0 * c0
        print(f"\nbelt mask: {m.sum()} cells, {m.sum()*area_cell:.2f} m2, "
              f"deck tilt {np.degrees(np.arccos(abs(dn[2]))):.2f} deg")
    else:
        xs = np.arange(a.extent[0], a.extent[1] + a.cell, a.cell)
        ys = np.arange(a.extent[2], a.extent[3] + a.cell, a.cell)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        base = np.stack([XX.ravel(), YY.ravel(),
                         np.full(XX.size, a.deck)], 1)
        dn = np.array([0.0, 0.0, 1.0])
        area_cell = a.cell * a.cell
    subs = [s for k in range(1, len(cams) + 1)
            for s in itertools.combinations(cams, k)]

    where = (f"belt mask {a.belt_mask}" if a.belt_mask
             else f"rectangle at z={a.deck:.2f}")
    print(f"\ncoverage (m2) over {where}, >=2 views (>=1 for singles)")
    hdr = f"{'subset':30s} {'cond':>6s} " + "".join(
        f"{'h='+str(int(h*1000)):>8s}" for h in a.heights)
    print(hdr)
    print("-" * len(hdr))
    for sub in subs:
        row = []
        for h in a.heights:
            X = base - h * dn                     # h above the deck, along n
            vis = np.stack([sees(c, X, R, t, K, D, size) for c in sub], 0)
            row.append((vis.sum(0) >= (1 if len(sub) == 1 else 2)).sum()
                       * area_cell)
        cd = conditioning(sub, C)
        cs = f"{cd:.4f}" if len(sub) > 2 else "   -  "
        flag = ""
        if len(sub) > 2 and cd < 0.05:
            flag = "  <- collinear: pose prior cannot carry scale"
        drop = 100 * (row[-1] / row[0] - 1) if row[0] > 0 else 0
        print(f"{'+'.join(sub):30s} {cs:>6s} "
              + "".join(f"{v:8.2f}" for v in row)
              + f"   {drop:+5.0f}% over height{flag}")


if __name__ == "__main__":
    main()