#!/usr/bin/env python3
"""
fit_correction.py

Fit a per-camera depth correction against the ground truth, and test it on belt
area it was not fitted on.

WHAT IT FITS
------------
Per camera, the height deficit between the reconstruction and the ground truth,
modelled as

    deficit = c0 + h * (k0 + kx * x + ky * y)

with h the ground-truth height above the deck and (x, y) the position along and
across the belt. Four parameters: a constant, a relief term, and two positional
gradients. Three nested models are fitted so each term can be priced:

    constant          c0                      what a deck anchor already does
    relief            c0 + k0*h               a single scale per camera
    positional        c0 + h*(k0+kx*x+ky*y)   the full model

WHY A SPATIAL HOLD-OUT
----------------------
Fitting four parameters to 6000 cells will always reduce the residual on those
cells. The question is whether the positional terms describe the camera or the
scene. Splitting the belt along its length and fitting on one half, testing on
the other, answers that: a real field-varying error extrapolates across the
belt, one that is absorbing this scene's parcel placement does not.

This is the same guard that three separate parcel arrangements would give, from
one capture, and it is available today.

WHAT TO EXPECT, AND THE RISK
----------------------------
The original ChArUco work measured 62 to 93 per cent of the height error as
COMMON MODE across views within a capture. A per-view correction cannot touch a
common-mode term. So the honest possible outcome is that the correction removes
the per-view part and leaves most of the error standing. The report separates
the two: the common-mode component is the mean deficit across cameras, and what
each camera's correction removes is the rest.

INPUTS
------
An aligned ground-truth cloud (gt_align.json from the pinning), a run directory
with cloud_calib_<cam>.ply, and belt.json for the frame.

EXAMPLE
-------
    python3 fit_correction.py --gt gt/pointcloud_20260826_095616.ply \\
        --align gt_align.json --belt belt.json \\
        --run runs_da3/cfg1008/center+left+top+right_res1008 \\
        --out depth_correction_fitted.json

Keep this file beside rigkit.py.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")

CAMS = ["center", "left", "top", "right"]

# --- deployed correction, loaded not fitted -------------------------------
# depth_correction.json carries the shared map h' = alpha*h + beta that
# da3_stream.py applies at run time. Scoring it here on the same held-out
# cells as the fitted arms is what makes the plane-vs-belt comparison a
# comparison of holdout designs rather than of models.
_DEP_PATH = Path(__file__).with_name("depth_correction.json")


def _load_deployed():
    """Return (alpha, beta_metres, label). Falls back to identity if absent."""
    try:
        with open(_DEP_PATH) as fh:
            shared = json.load(fh)["shared"]
        return (float(shared["alpha"]), float(shared["beta"]),
                f"deployed(alpha={shared['alpha']:.4f}, "
                f"beta={shared['beta'] * 1e3:+.2f}mm)")
    except (OSError, KeyError, ValueError) as exc:
        print(f"[deployed] {_DEP_PATH.name} unusable ({exc}); "
              f"the deployed arm will read as identity and is meaningless",
              file=sys.stderr)
        return 1.0, 0.0, "deployed(UNAVAILABLE)"


_DEP_ALPHA, _DEP_BETA, _DEP_LABEL = _load_deployed()
_DEP_SIGN = -1.0  # +1: d = recon - gt; -1: d = gt - recon


def deployed_residual(d, h, sign=+1.0):
    """Residual after applying h' = alpha*h + beta to cells with deficit d.

    The correction acts on the MEASURED height, so it scales the deficit as
    well as shifting it. With h_meas = h + d (sign=+1) the corrected residual
    is alpha*d + (alpha-1)*h + beta. Dropping the alpha*d term is wrong: it is
    what makes the correction self-cancelling when alpha exactly undoes the
    compression present in d.

    sign=+1 when d = reconstruction - ground truth.
    sign=-1 when d = ground truth - reconstruction.
    """
    return (_DEP_ALPHA * d
            + sign * ((_DEP_ALPHA - 1.0) * h + _DEP_BETA))
# --------------------------------------------------------------------------



def cams_of(run):
    name = Path(run).name
    if "_res" in name:
        name = name.rsplit("_res", 1)[0]
    return [c for c in name.replace(",", "+").split("+") if c in CAMS]


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


def raster(P, frame, belt, cell, margin=0.0, cross=None):
    """Median height per belt cell, and the cell's own (x, y).

    A cell median rather than a nearest-neighbour distance, because the ground
    truth is far denser than the reconstruction and a nearest-neighbour figure
    would report that density difference as error.

    The fourth value is the cell's own height SPREAD, the 16th to 84th
    percentile. It exists to find cells that straddle a parcel edge: the
    ground-truth sensor sees the vertical side face and fills such a cell with
    points from deck level to the top, so its median lands near half the parcel
    height while the reconstruction, looking down, gives the top. The deficit
    then reads about -120 mm and has nothing to do with depth error. On this
    rig those cells were 40 to 50 per cent of the sample and they are what
    pushed the relief coefficient above 1.

    `cross` bounds the half-width admitted. It must be set to the DECK, not to
    belt.json's footprint: that footprint is the conveyor FRAME at 900 mm while
    the deck surface is 791 mm, so the outer 55 mm each side is structure. The
    side rails stand about 110 to 160 mm above the deck and the ground truth
    holds tens of thousands of points on them, which the reconstruction does
    not represent the same way. Left in, they form a second population with the
    OPPOSITE sign of deficit -- around -120 mm where parcels give +50 to +125 --
    and a linear fit through both returns a relief coefficient above 1, which
    would mean a parcel reading shorter as it grows taller.
    """
    c, ex, ey, n = frame
    rel = P - c
    u, v, h = rel @ ex, rel @ ey, rel @ n
    L = belt["length_m"] / 2 + margin
    W = (cross if cross is not None else belt["width_m"] / 2 + margin)
    ok = (np.abs(u) <= L) & (np.abs(v) <= W)
    if not ok.any():
        return {}
    nv = int(2 * W / cell) + 1
    iu = np.clip(((u[ok] + L) / cell).astype(np.int64), 0, int(2 * L / cell))
    iv = np.clip(((v[ok] + W) / cell).astype(np.int64), 0, nv - 1)
    key = iu * nv + iv
    order = np.argsort(key, kind="stable")
    key, hs = key[order], h[ok][order]
    uu, vv = u[ok][order], v[ok][order]
    edges = np.flatnonzero(np.diff(key)) + 1
    out = {}
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(key)]):
        seg = hs[a:b]
        spread = (float(np.percentile(seg, 84) - np.percentile(seg, 16))
                  if len(seg) >= 8 else 0.0)
        out[int(key[a])] = (float(np.median(seg)), float(np.median(uu[a:b])),
                            float(np.median(vv[a:b])), spread)
    return out


def fit(design, y, robust=True, huber=2.0, iters=8):
    """Least squares, Huber-weighted by default.

    Plain least squares chases the outliers, and this data has a specific kind:
    a cell where the reconstruction has a hole, or where a parcel edge lets the
    deck show through, gives a large negative deficit that is not relief. Fitted
    unweighted, those cells drag the relief coefficient above 1, which would
    mean a parcel reading SHORTER as it gets taller. grade_subsets fits the same
    quantity with a Huber weight and gets p = 0.52 on the same run, so the
    difference is the estimator, not the scene.
    """
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    if not robust:
        return coef
    w = np.ones_like(y)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        r = np.abs(design @ coef - y)
        s = max(1.4826 * float(np.median(r)), 1e-9)
        w = np.where(r <= huber * s, 1.0, huber * s / np.maximum(r, 1e-12))
    return coef


def deficit_by_height(h, d, edges):
    """Median deficit in height bands, with no model imposed.

    If the deficit is linear in height the medians fall on a line and the
    relief coefficient is its slope. If they do not, no linear model is the
    right one and the fitted coefficient is meaningless whatever the estimator.
    """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h < hi)
        if int(m.sum()) >= 50:
            out.append((lo, hi, int(m.sum()), float(np.median(d[m]))))
    return out


def rms(v):
    return float(np.sqrt(np.mean(np.asarray(v) ** 2))) if len(v) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-camera depth correction with a spatial hold-out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--align", type=Path, default=Path("gt_align.json"))
    ap.add_argument("--belt", type=Path, default=Path("belt.json"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cell", type=float, default=0.010)
    ap.add_argument("--top-min", type=float, default=0.060,
                    help="ground-truth height above which a cell counts as a "
                         "parcel top. The deck carries no relief and would "
                         "dominate the fit")
    ap.add_argument("--top-max", type=float, default=0.600)
    ap.add_argument("--split", choices=("along", "across", "random"),
                    default="along",
                    help="how to hold out. 'along' splits the belt at its "
                         "midpoint, which is the honest test for a positional "
                         "model: the held-out half is territory the fit never "
                         "saw. 'random' interleaves cells and will flatter the "
                         "positional terms")
    ap.add_argument("--max-cell-spread", type=float, default=0.030,
                    help="a cell whose own height spans more than this is "
                         "straddling a parcel edge, not sitting on a top, and "
                         "its median is not comparable between the two clouds")
    ap.add_argument("--cross", type=float, default=0.36,
                    help="half-width across the belt admitted, metres. Set to "
                         "the DECK (791 mm wide, so 0.396 minus a margin), not "
                         "to belt.json's 900 mm frame: the rails are a second "
                         "population with the opposite sign of deficit")
    ap.add_argument("--min-cells", type=int, default=200)
    ap.add_argument("--no-robust", dest="robust", action="store_false",
                    default=True,
                    help="fit unweighted. Only to reproduce the outlier-driven "
                         "coefficients above 1")
    ap.add_argument("--out", type=Path,
                    default=Path("depth_correction_fitted.json"))
    args = ap.parse_args()

    belt = json.loads(args.belt.read_text())
    frame = belt_frame(belt)
    T = np.asarray(json.loads(args.align.read_text())["T"], float)
    gt = rigkit.load_cloud(str(args.gt))
    gt = gt @ T[:3, :3].T + T[:3, 3]
    G = raster(gt, frame, belt, args.cell, cross=args.cross)
    print(f"[gt   ] {len(G)} belt cells occupied")

    cams = cams_of(args.run)
    rows, out = {}, {}
    for cam in cams:
        f = args.run / f"cloud_calib_{cam}.ply"
        if not f.exists():
            print(f"[{cam:>6s}] no cloud_calib_{cam}.ply")
            continue
        R = raster(rigkit.load_cloud(str(f)), frame, belt,
                   args.cell, cross=args.cross)
        keys = [k for k in R if k in G
                and args.top_min < G[k][0] < args.top_max
                and G[k][3] <= args.max_cell_spread
                and R[k][3] <= args.max_cell_spread]
        dropped = sum(1 for k in R if k in G
                      and args.top_min < G[k][0] < args.top_max
                      and (G[k][3] > args.max_cell_spread
                           or R[k][3] > args.max_cell_spread))
        if len(keys) < args.min_cells:
            print(f"[{cam:>6s}] only {len(keys)} shared parcel cells, skipped")
            continue
        h = np.array([G[k][0] for k in keys])
        x = np.array([G[k][1] for k in keys])
        y = np.array([G[k][2] for k in keys])
        d = np.array([G[k][0] - R[k][0] for k in keys])   # deficit, metres
        rows[cam] = (h, x, y, d)
        print(f"[{cam:>6s}] {dropped} cells dropped as not flat "
              f"(spread over {args.max_cell_spread * 1e3:.0f} mm)")
        print(f"[{cam:>6s}] {len(keys)} shared parcel cells, deficit "
              f"{np.median(d) * 1e3:+.1f} mm median, {rms(d) * 1e3:.1f} mm rms")

    if not rows:
        raise SystemExit("no camera had enough shared parcel cells")

    # ---- how much of the deficit is common mode? ------------------------
    shared = set.intersection(*[set(range(len(v[0]))) for v in rows.values()]) \
        if False else None
    med = {c: float(np.median(v[3])) for c, v in rows.items()}
    common = float(np.median(list(med.values())))
    print(f"\n[common] median deficit per camera: "
          + "  ".join(f"{c} {m * 1e3:+.0f}mm" for c, m in med.items()))
    print(f"[common] common-mode component {common * 1e3:+.0f} mm; the spread "
          f"about it, {np.std(list(med.values())) * 1e3:.0f} mm, is all a "
          f"per-view correction can reach")

    # ---- fit, with a spatial hold-out -----------------------------------
    print(f"\nheld out by {args.split}; residual on HELD-OUT cells is the "
          f"number that counts")
    print(f"  {'camera':>7s} {'fit n':>7s} {'test n':>7s} "
          f"{'none':>9s} {'deployed':>10s} {'constant':>10s} {'relief':>9s} {'positional':>11s}")
    for cam, (h, x, y, d) in rows.items():
        if args.split == "along":
            m = x < np.median(x)
        elif args.split == "across":
            m = y < np.median(y)
        else:
            m = np.arange(len(x)) % 2 == 0
        models = {}
        for name, cols in (("constant", [np.ones_like(h)]),
                           ("relief", [np.ones_like(h), h]),
                           ("positional", [np.ones_like(h), h, h * x, h * y])):
            A = np.stack(cols, 1)
            coef = fit(A[m], d[m], args.robust)
            models[name] = (coef, rms(d[~m] - A[~m] @ coef))
        bands = deficit_by_height(h, d, np.arange(args.top_min,
                                                   args.top_max + 1e-9, 0.05))
        if bands:
            print(f"  {cam:>7s} deficit by height band, no model imposed:")
            print("          " + "  ".join(
                f"{lo * 1e3:.0f}-{hi * 1e3:.0f}mm n={k} {v * 1e3:+.0f}"
                for lo, hi, k, v in bands))
            if len(bands) > 1:
                hs = np.array([(a + b) / 2 for a, b, _, _ in bands])
                ds = np.array([v for _, _, _, v in bands])
                sl = np.polyfit(hs, ds, 1)[0]
                print(f"          slope through the band medians "
                      f"{sl:+.3f}  -> implied p = {1 - sl:+.3f}")
        print(f"  {cam:>7s} {int(m.sum()):>7d} {int((~m).sum()):>7d} "
              f"{rms(d[~m]) * 1e3:>8.1f}mm "
              + "".join(f"{models[k][1] * 1e3:>9.1f}mm"
                        for k in ("constant", "relief", "positional")))
        A_all = np.stack([np.ones_like(h), h, h * x, h * y], 1)
        full = fit(A_all, d, args.robust)
        out[cam] = {
            # The model-free shape, carried into the file because it is the
            # diagnostic that decides whether ANY linear model is appropriate,
            # and a console table does not survive being pasted.
            "deficit_by_height_mm": [
                {"band_mm": [round(lo * 1e3), round(hi * 1e3)], "n": k,
                 "median_deficit_mm": round(v * 1e3, 1)}
                for lo, hi, k, v in bands],
            "band_slope": (round(float(np.polyfit(
                [(a + b) / 2 for a, b, _, _ in bands],
                [v for _, _, _, v in bands], 1)[0]), 4)
                if len(bands) > 1 else None),
            "c0_mm": round(float(full[0]) * 1e3, 3),
            "k0": round(float(full[1]), 5),
            "kx_per_m": round(float(full[2]), 5),
            "ky_per_m": round(float(full[3]), 5),
            "n_cells": int(len(h)),
            "heldout_rms_mm": {k: round(models[k][1] * 1e3, 2)
                               for k in models},
            "uncorrected_heldout_rms_mm": round(rms(d[~m]) * 1e3, 2),
            "deployed_heldout_rms_mm": round(
                rms(deployed_residual(d, h, _DEP_SIGN)[~m]) * 1e3, 2),
            "deployed_coefficients": {"alpha": _DEP_ALPHA,
                                      "beta_mm": round(_DEP_BETA * 1e3, 3)}}

    args.out.write_text(json.dumps(
        {"run": args.run.name, "split": args.split,
         "args": {k: (str(v) if hasattr(v, "__fspath__") else v)
                  for k, v in vars(args).items()},
         "gt_file": str(args.gt), "gt_mtime": __import__("os").path
                    .getmtime(args.gt),
         "common_mode_mm": round(common * 1e3, 2),
         "model": "deficit = c0 + h*(k0 + kx*x + ky*y), metres",
         "note": "held-out residuals are the honest figure; the coefficients "
                 "are refitted on all cells for use",
         "cameras": out}, indent=2))
    print(f"\nwritten {args.out}")
    print("\nRead the held-out columns left to right. If 'positional' is no "
          "better than 'relief', the error is not field-varying and the two "
          "gradients are fitting this scene rather than the camera. If none "
          "of them improves much on 'none', the error is common mode and a "
          "per-view correction cannot reach it — which the common-mode line "
          "above predicts in advance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())