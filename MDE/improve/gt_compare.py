#!/usr/bin/env python3
"""
gt_compare.py

Grade a fused DA3 cloud against the reference scan, and report WHAT KIND of
error separates them rather than only how large it is.

Why this rather than a cloud-to-cloud distance
----------------------------------------------
"Mean distance 24 mm" is not actionable: it mixes a rigid frame offset, a gain
error in reconstructed height, a lateral shrinkage of every top face, and
ordinary noise. Each has a different fix and three of the four are invisible in
the scalar. This separates them, in the conveyor frame, in the units the pick
record already uses:

    z_fused = gain * z_true + offset        the relief error
    length, width per parcel                the footprint error
    per-cell surface thickness              the layering and noise

The registration is RIGID. Any scale that survives a rigid fit is depth-model
error, so letting the registration absorb it would hide the number worth
reading. Pass --transform to use the frozen scan_to_rig.json instead of
re-solving, which is what makes two runs comparable.

    python gt_compare.py --scan scans/cell.ply --fused runs/.../fused.ply \
        --transform scan_to_rig.json --deck-rig 3.08 --out-dir compare

Read the gain line first. After a correction fitted by scan_align.py it should
come back to 1.00 within a few thousandths; anything else means the correction
is not doing what it was fitted to do.

Keep this file beside scan_frame.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

import scan_frame as sf


# --------------------------------------------------------------------------
# error model
# --------------------------------------------------------------------------

def robust_lstsq(X, y, iters=8):
    w = np.ones(len(y))
    c = np.zeros(X.shape[1])
    for _ in range(iters):
        c = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        r = np.abs(y - X @ c)
        s = max(1.4826 * float(np.median(r)), 1e-9)
        w = np.clip(2.0 * s / np.maximum(r, 1e-9), 0.0, 1.0)
    return c, float(np.median(np.abs(y - X @ c)) * 1.4826)


def height_model(Fl, Gl, ext, cell=0.02, flat=0.012, h_range=None):
    """Fused height against reference height, cell by cell.

    Only cells where the reference is locally flat are used. A cell straddling
    a parcel edge differs by the parcel's whole height for a millimetre of
    lateral misregistration, and a handful of those dominate any least squares
    that admits them.
    """
    g = sf.Grid(ext, cell)
    mF, sF = g.surface(Fl)
    mG, sG = g.surface(Gl)
    bad = cv2.dilate(((~np.isfinite(mG)) | (sG > flat)).astype(np.uint8),
                     np.ones((3, 3), np.uint8)) > 0
    ok = np.isfinite(mF) & np.isfinite(mG) & ~bad
    if h_range is not None:
        ok &= (mG >= h_range[0]) & (mG <= h_range[1])
    ys, xs = np.nonzero(ok)
    h = mG[ok]
    e = (mF - mG)[ok]
    u = xs * cell + g.u0
    v = ys * cell + g.v0
    one = np.ones_like(h)
    models = {}
    for name, X in (("offset only", np.stack([one], 1)),
                    ("offset + gain*h", np.stack([one, h], 1)),
                    ("offset + h*(gain + gu*u + gv*v)",
                     np.stack([one, h, h * u, h * v], 1))):
        models[name] = robust_lstsq(X, e)
    return models, (mF, sF, mG, sG, g), (h, e)


# --------------------------------------------------------------------------
# parcels
# --------------------------------------------------------------------------

def parcel_table(mF, mG, g, min_area_m2=0.02, erode_mm=85, face_tol=0.020):
    """One row per parcel found in the reference, measured in both clouds.

    The footprint is taken at the same tolerance about each cloud's OWN face
    height, so a cloud that reconstructs the face low is not also penalised
    laterally for it. The height is the median over an eroded core, so an edge
    that is smoothed in one cloud and sharp in the other does not move it.
    """
    cell = g.cell
    occ = (np.nan_to_num(mG, nan=-1.0) > 0.05).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(occ, 8)

    def kernel_for(mask):
        """Erosion sized to the parcel, not fixed.

        A fixed 85 mm inward margin is right for a 400 mm parcel and fatal for
        a 104 mm one: nothing survives, the parcel vanishes from the reference,
        and a correct detection is then reported as a false one. The distance
        transform gives half the parcel's minor dimension, so the margin can be
        a fraction of it and stay a margin at any size.
        """
        half = float(cv2.distanceTransform(mask, cv2.DIST_L2, 5).max()) * cell
        k = int(round(min(erode_mm / 1000.0, 0.5 * half) / cell))
        return max(1, k) | 1

    def rect(mask, m, href):
        sel = (mask > 0) & np.isfinite(m) & (np.abs(m - href) < face_tol)
        ys, xs = np.nonzero(sel)
        if len(xs) < 200:
            return None
        pts = np.stack([xs * cell + g.u0, ys * cell + g.v0], 1) * 1000.0
        (cu, cv_), (w, hh), _ = cv2.minAreaRect(pts.astype(np.float32))
        return max(w, hh), min(w, hh), cu, cv_, len(xs)

    rows = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * cell ** 2 < min_area_m2:
            continue
        m = (lab == i).astype(np.uint8)
        ker = kernel_for(m)
        core = cv2.erode(m, np.ones((ker, ker), np.uint8))
        if core.sum() < 60:
            continue
        vg, vf = mG[core > 0], mF[core > 0]
        if not np.isfinite(vg).any() or not np.isfinite(vf).any():
            continue
        hg = float(np.nanmedian(vg))
        hf = float(np.nanmedian(vf))
        if not (np.isfinite(hg) and np.isfinite(hf)):
            continue
        dil = cv2.dilate(m, np.ones((ker + 6, ker + 6), np.uint8))
        rg, rf = rect(dil, mG, hg), rect(dil, mF, hf)
        if rg is None or rf is None:
            continue
        rows.append({"u": cent[i][0] * cell + g.u0,
                     "v": cent[i][1] * cell + g.v0,
                     "gt_h": hg * 1e3, "fused_h": hf * 1e3,
                     "dh": (hf - hg) * 1e3,
                     "gt_l": rg[0], "fused_l": rf[0], "dl": rf[0] - rg[0],
                     "gt_w": rg[1], "fused_w": rf[1], "dw": rf[1] - rg[1]})
    return rows


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grade a fused DA3 cloud against a reference scan and "
                    "separate the error into a relief gain, a footprint bias "
                    "and noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scan", "--gt", dest="scan", type=Path, required=True)
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--transform", type=Path, default=None,
                    help="frozen scan_to_rig.json. Given, the scan is placed "
                         "by it and only a small refinement is run, which is "
                         "what makes two gradings comparable. Omitted, the "
                         "registration is solved from scratch")
    ap.add_argument("--deck-scan", type=float, default=None,
                    help="rough scanner-to-deck distance, m")
    ap.add_argument("--deck-rig", type=float, default=None,
                    help="rough reference-camera-to-deck distance, m")
    ap.add_argument("--voxel", type=float, default=0.006)
    ap.add_argument("--no-refine", action="store_true",
                    help="with --transform, do not refine at all. Use it to "
                         "see the absolute placement error as well as the "
                         "shape error")
    ap.add_argument("--range", nargs=2, type=float, default=[-0.06, 0.45],
                    metavar=("LO_M", "HI_M"),
                    help="height range the gain is fitted over. It must match "
                         "the valid_range_m of the correction being graded: "
                         "outside that range da3_stream.py holds the "
                         "correction constant, so those cells are uncorrected "
                         "and mixing them in measures a blend of two different "
                         "things. On this rig the floor is more than half the "
                         "cells, so the blend is mostly the uncorrected half")
    ap.add_argument("--gross-mm", type=float, default=60.0,
                    help="a parcel whose height differs by more than this is "
                         "reported as a failure and left out of the averages, "
                         "because it is a detection fault rather than a bias")
    ap.add_argument("--out-dir", type=Path, default=Path("compare"))
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the figures, for tracking them across "
                         "runs rather than reading them off a terminal")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    S = sf.voxel(sf.read_ply(args.scan), args.voxel)
    F = sf.voxel(sf.read_ply(args.fused), args.voxel)
    print(f"[load ] scan {len(S)} points   fused {len(F)} points "
          f"(after {args.voxel * 1e3:.0f} mm voxel)")

    if args.transform:
        # ONE frame for both clouds, built from the scan. Framing each cloud by
        # its own belt would undo the registration along the belt, because the
        # two sensors see different lengths of it and their rectangle centres
        # therefore sit in different places. It also makes the frame
        # metrology-defined: heights below are measured against the scanner's
        # deck rather than against DA3's own fit of it, so an error in that fit
        # shows up as a residual instead of being absorbed into the frame.
        T, tmeta = sf.load_transform(args.transform)
        S = sf.apply_T(T, S)
        print(f"[frame] placed by {args.transform}, and both clouds are "
              f"expressed in the scan's conveyor frame")
        nS, dS = sf.deck_plane(S, args.deck_rig)
        bS = sf.belt_frame(S, nS, dS)
        nF, dF = sf.deck_plane(F, args.deck_rig)
        bF = sf.belt_frame(F, nF, dF)
        Gl = sf.to_frame(S, bS)
        Fl = sf.to_frame(F, bS)
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(nF @ nS)), 0, 1))))
        print(f"[deck ] scan  d={dS:.4f} m   belt {bS['length_m'] * 1e3:.0f} x "
              f"{bS['width_m'] * 1e3:.0f} mm visible")
        print(f"[deck ] fused d={dF:.4f} m   belt {bF['length_m'] * 1e3:.0f} x "
              f"{bF['width_m'] * 1e3:.0f} mm visible; its own deck sits "
              f"{(dF - dS) * 1e3:+.1f} mm from the scan's and {tilt:.2f} deg "
              f"off it")
    else:
        print("[frame] no --transform, solving the registration here")
        nS, dS = sf.deck_plane(S, args.deck_scan)
        nF, dF = sf.deck_plane(F, args.deck_rig)
        bS = sf.belt_frame(S, nS, dS)
        bF = sf.belt_frame(F, nF, dF)
        print(f"[deck ] scan  d={dS:.4f} m   belt {bS['length_m'] * 1e3:.0f} x "
              f"{bS['width_m'] * 1e3:.0f} mm visible")
        print(f"[deck ] fused d={dF:.4f} m   belt {bF['length_m'] * 1e3:.0f} x "
              f"{bF['width_m'] * 1e3:.0f} mm visible")
        Gl = sf.to_frame(S, bS)
        Fl = sf.to_frame(F, bF)
        Tc, score, _yaw, _n = sf.coarse_align(Fl, Gl)
        print(f"[align] silhouette correlation {score:.3f}")
        if score < 0.55:
            print("[WARN ] the parcel silhouettes barely correlate; either "
                  "these are not the same scene or one cloud is missing its "
                  "parcels, and everything below is then meaningless")
        Fl = sf.apply_T(Tc, Fl)

    def crop(P):
        return P[(np.abs(P[:, 0]) < 1.6) & (np.abs(P[:, 1]) < 0.62)
                 & (P[:, 2] > -0.06) & (P[:, 2] < 0.75)]

    if not args.no_refine:
        Ti, d = sf.icp(crop(Fl)[::3], crop(Gl))
        Fl = sf.apply_T(Ti, Fl)
        ang = float(np.degrees(np.arccos(
            np.clip((np.trace(Ti[:3, :3]) - 1) / 2, -1, 1))))
        print(f"[icp  ] rigid refinement {ang:.2f} deg, "
              f"{np.round(Ti[:3, 3] * 1e3, 1)} mm; residual median "
              f"{np.median(d) * 1e3:.1f} mm, rms "
              f"{np.sqrt(float((d ** 2).mean())) * 1e3:.1f} mm")

    ext = (float(min(Fl[:, 0].min(), Gl[:, 0].min())),
           float(max(Fl[:, 0].max(), Gl[:, 0].max())),
           float(min(Fl[:, 1].min(), Gl[:, 1].min())),
           float(max(Fl[:, 1].max(), Gl[:, 1].max())))
    full, (mF2, sF2, mG2, sG2, g2), (hf, ef) = height_model(Fl, Gl, ext)
    models, _, (h, e) = height_model(Fl, Gl, ext, h_range=tuple(args.range))

    cf, _ = full["offset + gain*h"]
    print(f"\n[model] over the whole cloud, {len(hf)} flat cells spanning "
          f"{hf.min() * 1e3:.0f} to {hf.max() * 1e3:.0f} mm: "
          f"fused_h = {1 + cf[1]:.4f} * true {cf[0] * 1e3:+.1f} mm")
    print(f"[model] that figure mixes corrected and uncorrected heights and is "
          f"reported only so it is not missing; judge on the next block")

    print(f"\n[model] inside the corrected range {args.range[0] * 1e3:.0f} to "
          f"{args.range[1] * 1e3:.0f} mm, {len(h)} flat cells spanning "
          f"{h.min() * 1e3:.0f} to {h.max() * 1e3:.0f} mm")
    for name, (c, sig) in models.items():
        print(f"[model] {name:<32s} robust sigma {sig * 1e3:6.1f} mm   "
              f"{np.round(c, 4)}")
    c, _ = models["offset + gain*h"]
    gain, off = 1.0 + float(c[1]), float(c[0])
    print(f"[model] fused_h = {gain:.4f} * true {off * 1e3:+.1f} mm")
    if abs(gain - 1.0) > 0.01:
        print(f"[model] that is {abs(gain - 1) * 100:.1f} per cent of every "
              f"height. Divide the applied alpha by {gain:.4f}, or re-fit with "
              f"scan_align.py, which does it from the data")
    else:
        print("[model] the relief gain is within one per cent; what is left is "
              "noise and footprint, not a systematic height error")

    fine = sf.Grid(ext, 0.02)
    _, spF = fine.surface(Fl, 8)
    mGf, spG = fine.surface(Gl, 8)
    noise = {}
    for lbl, lo, hi in (("deck", -0.03, 0.03), ("parcel tops", 0.08, 0.45)):
        m = np.isfinite(spF) & np.isfinite(spG) & (mGf > lo) & (mGf < hi)
        if int(m.sum()) > 100:
            noise[lbl] = (float(np.median(spG[m]) * 1e3),
                          float(np.median(spF[m]) * 1e3))
            print(f"[noise] {lbl:<12s} surface thickness p10-p90: scan "
                  f"{noise[lbl][0]:5.1f} mm   fused {noise[lbl][1]:5.1f} mm")

    fineg = sf.Grid(ext, 0.005)
    mFp, _ = fineg.surface(Fl, 2)
    mGp, _ = fineg.surface(Gl, 2)
    rows = parcel_table(mFp, mGp, fineg)
    print(f"\n[parcel] {'u_m':>6} {'scan LxWxh':>24} {'fused LxWxh':>24} "
          f"{'dL':>6} {'dW':>6} {'dh':>6}")
    for r in rows:
        flag = "  GROSS" if abs(r["dh"]) > args.gross_mm else ""
        print(f"[parcel] {r['u']:>6.2f} {r['gt_l']:>8.0f} x{r['gt_w']:>7.0f} x"
              f"{r['gt_h']:>6.0f} {r['fused_l']:>11.0f} x{r['fused_w']:>7.0f} x"
              f"{r['fused_h']:>6.0f} {r['dl']:>+6.0f} {r['dw']:>+6.0f} "
              f"{r['dh']:>+6.0f}{flag}")

    summary = {"gain": gain, "offset_mm": off * 1e3, "noise": noise,
               "n_parcels": len(rows)}
    good = [r for r in rows if abs(r["dh"]) <= args.gross_mm]
    gross = [r for r in rows if abs(r["dh"]) > args.gross_mm]
    if good:
        arr = lambda k: np.array([r[k] for r in good], dtype=float)
        print(f"[parcel] {len(good)} of {len(rows)} parcels within "
              f"{args.gross_mm:.0f} mm:")
        for k, lbl in (("dh", "height"), ("dl", "length"), ("dw", "width")):
            a = arr(k)
            summary[lbl] = {"mean_mm": float(a.mean()),
                            "sd_mm": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                            "worst_mm": float(a[np.argmax(np.abs(a))])}
            print(f"[parcel]   {lbl:<7s} error mean {a.mean():+6.1f} mm  sd "
                  f"{summary[lbl]['sd_mm']:5.1f} mm  worst "
                  f"{summary[lbl]['worst_mm']:+6.1f} mm")
        gh, dh = arr("gt_h"), arr("dh")
        pred = (gh + dh) / gain - off * 1e3 / gain - gh
        print(f"[parcel]   the same parcels with heights divided by "
              f"{gain:.4f}: mean {pred.mean():+.1f} mm  sd "
              f"{pred.std(ddof=1) if len(pred) > 1 else 0.0:.1f} mm, which is "
              f"the floor the gain fix alone can reach")
    if gross:
        print(f"[parcel] {len(gross)} gross failure(s). These are detection "
              f"faults, not bias: check view_consensus in boxes.json for the "
              f"parcel at u = "
              + ", ".join(f"{r['u']:+.2f}" for r in gross))

    d2 = mF2 - mG2
    img = np.clip((np.nan_to_num(d2, nan=0.0) + 0.08) / 0.16, 0, 1)
    col = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[~np.isfinite(d2)] = (30, 30, 30)
    col = cv2.resize(col, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.putText(col, "fused - scan height   blue -80 mm   green 0   red +80 mm",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "height_error.png"), col)

    ov = sf.Grid(ext, 0.005)
    o = np.zeros((ov.H, ov.W, 3), np.uint8)
    for P, ch in ((Fl, 2), (Gl, 1)):
        s_ = (P[:, 2] > 0.04) & (P[:, 2] < 0.6)
        iu, iv, ok = ov.index(P[s_])
        o[iv[ok], iu[ok], ch] = 255
    cv2.putText(o, "red fused   green scan   yellow both", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "footprint_overlay.png"), o)
    print(f"\nwritten: {args.out_dir / 'height_error.png'}, "
          f"{args.out_dir / 'footprint_overlay.png'}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())