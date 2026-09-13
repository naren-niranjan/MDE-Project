#!/usr/bin/env python3
"""
probe_band.py

Write out the belt cells whose ground-truth height falls in a given band, so
they can be looked at rather than reasoned about.

WHY
---
On this rig the 110 to 160 mm band holds about 2200 cells in every camera and
reconstructs at roughly 254 mm — a 119 mm over-estimate on the shortest
parcels, where every other band under-estimates by 50 to 135 mm as ordinary
relief compression. Three explanations were tried and each failed against the
data: the conveyor side rails (excluding them changed 6 cells), parcel edge
cells straddling a vertical face (excluding them changed 45), and an outlier-
driven fit (robust weighting did not move it).

Rather than propose a fourth, this writes the cells out. Opened beside the
ground truth they are either a recognisable object or they are not, and that
settles it in a way no summary statistic has.

WHAT IT WRITES
--------------
  band_gt.ply      the ground-truth points in those cells, in the rig frame
  band_recon.ply   the reconstruction's points in the same cells
  band_cells.json  each cell's position on the belt, its ground-truth and
                   reconstructed height, and the deficit

Both clouds keep their original colour, so a distinctly coloured object shows
up immediately. The cell positions in the json say whether they cluster in one
place on the belt or are spread across it, which distinguishes one bad object
from a systematic effect.

EXAMPLE
-------
    python3 probe_band.py --gt gt/pointcloud_20260826_095616.ply \\
        --align gt_align.json --belt belt.json \\
        --run runs_da3/resweep_prior/left+top+right_res1008 \\
        --cam left --band 0.110 0.160

Keep this file beside rigkit.py.
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


def load_rgb(path):
    """xyz and rgb, falling back to xyz alone when the ply carries no colour."""
    try:
        import pin_markers
        return pin_markers.load_cloud_rgb(str(path))
    except Exception:  # noqa: BLE001
        P = rigkit.load_cloud(str(path))
        return P, np.full((len(P), 3), 200, np.uint8)


def write_ply(path, P, rgb):
    P = np.asarray(P, float)
    rgb = np.asarray(rgb, np.uint8)
    with open(path, "wb") as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n")
        fh.write(f"element vertex {len(P)}\n".encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\n")
        fh.write(b"property uchar red\nproperty uchar green\n"
                 b"property uchar blue\nend_header\n")
        rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        fh.write(rec.tobytes())


def belt_frame(belt):
    ex = np.asarray(belt["x_axis"], float)
    ey = np.asarray(belt["y_axis"], float)
    ex /= np.linalg.norm(ex)
    ey = ey - ex * float(ey @ ex)
    ey /= np.linalg.norm(ey)
    n = np.cross(ex, ey)
    n /= np.linalg.norm(n)
    c = np.asarray(belt["centre_m"], float)
    if float(n @ (np.zeros(3) - c)) < 0:
        n, ey = -n, -ey
    return c, ex, ey, n


def cell_ids(P, frame, belt, cell, cross):
    c, ex, ey, n = frame
    rel = P - c
    u, v, h = rel @ ex, rel @ ey, rel @ n
    L = belt["length_m"] / 2
    ok = (np.abs(u) <= L) & (np.abs(v) <= cross)
    nv = int(2 * cross / cell) + 1
    iu = np.clip(((u + L) / cell).astype(np.int64), 0, int(2 * L / cell))
    iv = np.clip(((v + cross) / cell).astype(np.int64), 0, nv - 1)
    return iu * nv + iv, h, u, v, ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write out the cells in a ground-truth height band.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--align", type=Path, default=Path("gt_align.json"))
    ap.add_argument("--belt", type=Path, default=Path("belt.json"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--band", nargs=2, type=float, default=[0.110, 0.160],
                    metavar=("LO", "HI"))
    ap.add_argument("--cell", type=float, default=0.010)
    ap.add_argument("--cross", type=float, default=0.36)
    ap.add_argument("--out-prefix", default="band")
    args = ap.parse_args()

    belt = json.loads(args.belt.read_text())
    frame = belt_frame(belt)
    T = np.asarray(json.loads(args.align.read_text())["T"], float)
    gt, gt_rgb = load_rgb(args.gt)
    gt = gt @ T[:3, :3].T + T[:3, 3]
    rp = args.run / f"cloud_calib_{args.cam}.ply"
    rec, rec_rgb = load_rgb(rp)

    gk, gh, gu, gv, gok = cell_ids(gt, frame, belt, args.cell, args.cross)
    rk, rh, _, _, rok = cell_ids(rec, frame, belt, args.cell, args.cross)

    # median height per cell, both clouds
    def medians(keys, h, ok):
        k, hh = keys[ok], h[ok]
        o = np.argsort(k, kind="stable")
        k, hh = k[o], hh[o]
        e = np.flatnonzero(np.diff(k)) + 1
        return {int(k[a]): float(np.median(hh[a:b]))
                for a, b in zip(np.r_[0, e], np.r_[e, len(k)])}

    G, R = medians(gk, gh, gok), medians(rk, rh, rok)
    lo, hi = args.band
    sel = {k for k in G if lo <= G[k] < hi and k in R}
    print(f"[band ] {len(sel)} cells with ground-truth height in "
          f"{lo * 1e3:.0f}-{hi * 1e3:.0f} mm")
    if not sel:
        raise SystemExit("no cells in that band")

    d = np.array([G[k] - R[k] for k in sel])
    print(f"[band ] reconstructed height median "
          f"{np.median([R[k] for k in sel]) * 1e3:.0f} mm, deficit median "
          f"{np.median(d) * 1e3:+.0f} mm")

    gm = np.isin(gk, list(sel)) & gok
    rm = np.isin(rk, list(sel)) & rok
    write_ply(f"{args.out_prefix}_gt.ply", gt[gm], gt_rgb[gm])
    write_ply(f"{args.out_prefix}_recon.ply", rec[rm], rec_rgb[rm])
    print(f"[band ] wrote {args.out_prefix}_gt.ply ({int(gm.sum())} points) "
          f"and {args.out_prefix}_recon.ply ({int(rm.sum())} points)")

    # where they sit on the belt: one object, or spread everywhere?
    c, ex, ey, n = frame
    rel = gt[gm] - c
    u, v = rel @ ex, rel @ ey
    print(f"[band ] along the belt {u.min():+.2f} to {u.max():+.2f} m, "
          f"across {v.min():+.2f} to {v.max():+.2f} m")
    hist, edges = np.histogram(u, bins=12)
    print("[band ] spread along the belt, 12 bins:")
    print("        " + "  ".join(f"{h:5d}" for h in hist))
    occupied = int(np.count_nonzero(hist))
    print(f"        {occupied} of 12 bins occupied — "
          + ("clustered, so likely ONE object" if occupied <= 4
             else "spread, so a systematic effect rather than one object"))

    Path(f"{args.out_prefix}_cells.json").write_text(json.dumps(
        {"band_mm": [lo * 1e3, hi * 1e3], "camera": args.cam,
         "n_cells": len(sel),
         "median_recon_mm": round(float(np.median([R[k] for k in sel])) * 1e3, 1),
         "median_deficit_mm": round(float(np.median(d)) * 1e3, 1)}, indent=2))
    print(f"[band ] wrote {args.out_prefix}_cells.json")
    print("\nOpen band_gt.ply beside band_recon.ply, both coloured. If the "
          "ground-truth points are a recognisable object — the white wrapped "
          "items rather than the cardboard boxes, say — that is the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())