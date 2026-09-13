#!/usr/bin/env python3
"""
probe_gauge.py

Recover the global scale of a subset run from the deck plane in each camera's
own frame, and set it against the scale recovered from the calibrated baselines.

WHY TWO ANCHORS
---------------
With --pose-prior never, DA3 returns depth and poses in one arbitrary gauge, and
that gauge has to be pinned down before anything is metric. There are two things
in the cell that can pin it:

  BASELINES. The calibrated camera centres are known, so a Umeyama fit of the
  predicted centres onto them yields the scale. This is what check_fusion.py
  does. It needs two cameras, and on this rig it leans on the centre-to-top
  baseline, which is 92 per cent axial and is the one measurement DA3 has been
  shown to get wrong by 25 to 50 per cent. So the anchor is contaminated by
  exactly the error it is supposed to be independent of.

  THE DECK. The perpendicular standoff from each camera to the conveyor is
  known from the calibration and does not move: 3.067 m for centre, left and
  right, 3.606 m for top. Fitting the dominant plane in a camera's own
  back-projection and comparing its standoff with that figure gives the scale
  from ONE camera, with no baseline involved and no dependence on the axial
  geometry. It also works for a single-camera subset, where no baseline exists.

The two should agree. Where they do not, the deck is the one to believe, because
its input is a fixed distance measured by calibration rather than a quantity
DA3 predicted.

WHAT ANCHORING ON THE DECK COSTS
--------------------------------
It fixes the deck by construction, so the offset q that grade_subsets.py fits
comes out near zero and stops being a measurement. What remains measurable is p,
the surviving fraction of relief, which is the quantity that actually matters
for parcel heights and the one depth_align.py fits. This is the same trade
depth_align.py makes when it fits a deck plane per camera: the deck is spent to
buy a height axis. Say so in the write-up rather than quoting q as a result.

A NOTE ON PICKING THE PLANE
---------------------------
The dominant plane in a four-camera cell is often the floor, roughly 0.8 m past
the deck, which is the same trap --seg-plane-distance exists for. Three planes
are peeled per camera and the assignment chosen is the one whose implied scales
agree best ACROSS cameras, since all views of one run share a single gauge and
the floor will not reproduce that agreement by accident. For a single camera
there is nothing to agree with, so all three candidates are printed and the
choice is yours.

EXAMPLE
-------
    python3 probe_gauge.py --runs-root runs_da3/scene_b \\
        --fusion-check fusion_check.json

Keep this file beside rigkit.py and depth_correction.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")


CAMS = ["center", "left", "top", "right"]


def parse_run(path: Path):
    m = re.search(r"_res(\d+)", path.name)
    subset = re.sub(r"_res\d+$", "", path.name)
    cams = [c for c in subset.replace(",", "+").split("+") if c]
    if not cams or any(c not in CAMS for c in cams):
        return None
    return {"dir": path, "subset": subset, "cams": cams,
            "res": int(m.group(1)) if m else None}


def deck_standoffs(path: Path, thickness_mm: float):
    """Perpendicular camera-to-deck distances, from the correction file.

    d_cam there is the standoff to the board FACE, so the deck surface sits one
    board thickness further. Six millimetres on three metres is 0.2 per cent and
    will not decide anything here, but it is carried rather than dropped.
    """
    data = json.loads(Path(path).read_text())
    out = {}
    for c, rec in (data.get("views") or {}).items():
        out[c] = abs(float(rec["d_cam"])) + thickness_mm / 1e3
    if not out:
        raise SystemExit(f"{path} carries no per-view deck planes")
    return out


def backproject_camera(depth, K, stride):
    """Camera-frame points, subsampled on the grid rather than at random.

    A stride keeps the sampling uniform over the image, so a plane fit is not
    biased towards whichever region happens to carry the most valid pixels.
    """
    d = np.asarray(depth, float)[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    m = np.isfinite(d) & (d > 0)
    return np.stack([x[m] * d[m], y[m] * d[m], d[m]], 1)


def peel_planes(P, n_planes, tol, min_frac):
    """The largest few planes, each with its standoff, tilt and inlier points.

    The inliers are kept because the standoff alone cannot say whether a plane
    is the conveyor or the floor. Its FOOTPRINT can, and that needs the points.
    """
    out, keep = [], np.arange(len(P))
    for _ in range(n_planes):
        if len(keep) < max(300, min_frac * len(P)):
            break
        n, d = rigkit.fit_plane_ransac(P[keep], tol=tol, trials=300)
        inl = np.abs(P[keep] @ n + d) < tol
        frac = float(inl.sum() / len(P))
        if inl.sum() < 300:
            break
        nn, dd = (n, d) if d < 0 else (-n, -d)
        if frac >= min_frac:
            out.append({"n": nn, "d": float(dd), "standoff": abs(float(dd)),
                        "frac": frac,
                        "tilt_deg": round(float(np.degrees(np.arccos(
                            np.clip(abs(nn[2]), 0, 1)))), 2),
                        "points": P[keep][inl]})
        keep = keep[~inl]
    return out


def patch_width(plane, cell, close_gauge):
    """Short side of the plane's largest connected patch, in the run's gauge.

    The conveyor is 0.900 m across and the floor is metres across, so this
    separates them outright, and it does so with a LATERAL measurement, which
    scales with depth exactly as the standoff does. Two independent readings of
    one gauge that must agree is a far stronger test than either alone.

    Closing bridges the gaps the parcels leave in the deck, exactly as
    box_segment.belt_from_raster does, or the deck comes back as strips and its
    width is measured across one strip.
    """
    Q, n = plane["points"], plane["n"]
    ex = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(ex) < 1e-6:
        ex = np.cross(n, [1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    u, v = Q @ ex, Q @ ey
    u0, v0 = float(u.min()), float(v.min())
    W = int((u.max() - u0) / cell) + 1
    H = int((v.max() - v0) / cell) + 1
    if W < 4 or H < 4 or W * H > 20_000_000:
        return None
    img = np.zeros((H, W), np.uint8)
    img[np.clip(((v - v0) / cell).astype(np.int64), 0, H - 1),
        np.clip(((u - u0) / cell).astype(np.int64), 0, W - 1)] = 255
    k = max(3, int(round(close_gauge / cell)))
    k += 1 - (k % 2)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n_lab <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero((lab == big) & (img > 0))
    if len(xs) < 150:
        return None
    uv = np.stack([xs * cell + u0, ys * cell + v0], 1) * 1000.0
    (_, _), (w, h), _ = cv2.minAreaRect(uv.astype(np.float32))
    du, dv = max(w, h) / 1000.0, min(w, h) / 1000.0
    return {"length": float(du), "width": float(dv), "cells": int(len(xs))}


def identify_deck(cands, expect_standoff, belt_width, args):
    """Which of a camera's planes is the conveyor.

    Two tests, neither of which can be satisfied by agreeing with the other
    cameras, which is what the previous version got wrong:

      WIDTH. g_standoff = expected / measured standoff, and g_width =
      0.900 m / measured patch width, are two readings of the same gauge taken
      from a distance and from a lateral extent. On the deck they agree. On the
      floor the patch is metres wide and they do not.

      PAIR RATIO. The ratio of two planes' standoffs within one camera is
      independent of the gauge entirely. Deck over floor is 3.067 / 3.867 =
      0.793 whatever DA3 did to the scale, so a pair sitting at that ratio
      names the deck with no scale knowledge at all.
    """
    for c in cands:
        g_s = expect_standoff / c["standoff"]
        pw = patch_width(c, cell=c["standoff"] * args.cell_frac,
                         close_gauge=args.belt_close / max(g_s, 1e-6))
        c["patch"] = pw
        c["g_standoff"] = float(g_s)
        c["g_width"] = (None if pw is None or pw["width"] < 1e-6
                        else float(belt_width / pw["width"]))
        c["width_disagree"] = (None if c["g_width"] is None
                               else abs(c["g_width"] - g_s) / g_s)

    ratio = None
    want = expect_standoff / (expect_standoff + args.deck_floor_gap)
    for i, a in enumerate(cands):
        for j, b in enumerate(cands):
            if i == j:
                continue
            r = a["standoff"] / b["standoff"]
            if abs(r - want) <= args.pair_tol:
                ratio = {"deck": i, "floor": j, "ratio": round(float(r), 4)}
                break
        if ratio:
            break

    ok = [c for c in cands if c["width_disagree"] is not None
          and c["width_disagree"] <= args.width_tol]
    if ok:
        best = min(ok, key=lambda c: c["width_disagree"])
        return best, "width", ratio
    # The pair test is CORROBORATION, not a selector. Measured on this rig its
    # ratios scatter from 0.78 to 0.84 against a predicted 0.7935, wide enough
    # that unrelated pairs fall inside any useful window, and the planes it
    # then names carry patches twice the width of the conveyor. It is offered
    # behind a flag so the failure can be reproduced, not relied upon.
    if ratio is not None and args.pair_fallback:
        return cands[ratio["deck"]], "pair ratio (UNCORROBORATED)", ratio
    return None, "unidentified", ratio


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recover each run's global scale from the conveyor plane, "
                    "identified by its footprint rather than by its size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--run", type=Path, action="append", default=None)
    ap.add_argument("--correction", type=Path,
                    default=Path("depth_correction.json"))
    ap.add_argument("--belt", type=Path, default=Path("belt.json"),
                    help="the frozen footprint; its WIDTH is the lateral "
                         "reading that identifies the deck")
    ap.add_argument("--board-thickness-mm", type=float, default=6.0)
    ap.add_argument("--fusion-check", type=Path, default=None)
    ap.add_argument("--planes", type=int, default=7,
                    help="planes peeled per camera. Where the width test found\n                         nothing, the conveyor was simply not among the candidates: every\n                         plane offered had a patch metres wide")
    ap.add_argument("--plane-tol", type=float, default=0.010,
                    help="inlier distance in the run's own gauge")
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="a candidate must hold this fraction of the "
                         "points. This is now only a guard against slivers, "
                         "not the selector: the width test is what rejects a "
                         "wrong plane, so this can sit low")
    ap.add_argument("--max-tilt-deg", type=float, default=15.0,
                    help="the deck is nearly fronto-parallel to every camera "
                         "on this rig, so a steeply leaning candidate is "
                         "structure, not the belt")
    ap.add_argument("--width-tol", type=float, default=0.15,
                    help="allowed disagreement between the standoff reading "
                         "and the width reading of the gauge")
    ap.add_argument("--belt-close", type=float, default=0.30,
                    help="metric gap bridged across the parcels standing on "
                         "the deck; converted into the run's gauge internally")
    ap.add_argument("--cell-frac", type=float, default=0.004,
                    help="raster cell as a fraction of the standoff, so the "
                         "raster is the same angular size whatever the gauge")
    ap.add_argument("--deck-floor-gap", type=float, default=0.80,
                    help="height of the deck above the floor, for the "
                         "gauge-free pair test. Measure it once if you can")
    ap.add_argument("--pair-tol", type=float, default=0.02)
    ap.add_argument("--pair-fallback", action="store_true",
                    help="fall back to the gauge-free pair test where the "
                         "width test finds nothing. Off by default: on this "
                         "rig it selected floor planes about half the time and "
                         "the runs it rescued were worse than the runs it "
                         "dropped")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("gauge_probe.json"))
    args = ap.parse_args()

    expect = deck_standoffs(args.correction, args.board_thickness_mm)
    belt = json.loads(args.belt.read_text())
    belt_w = float(belt["width_m"])
    print(f"conveyor width {belt_w * 1e3:.0f} mm, deck standoffs "
          + "  ".join(f"{c} {expect[c]:.3f}" for c in CAMS if c in expect))
    print(f"deck over floor standoff ratio, gauge free: "
          f"{expect['center'] / (expect['center'] + args.deck_floor_gap):.4f}")

    base = {}
    if args.fusion_check and args.fusion_check.exists():
        for r in json.loads(args.fusion_check.read_text()):
            base[r["run"]] = r.get("scale")

    runs = [Path(d) for d in (args.run or [])]
    if args.runs_root:
        runs += [d for d in sorted(Path(args.runs_root).iterdir())
                 if d.is_dir()]

    print(f"\n{'run':30s} {'cam':>6s} {'stand':>7s} {'g_stand':>8s} "
          f"{'width':>7s} {'g_width':>8s} {'dis':>6s} {'by':>11s}")
    print("-" * 92)
    records = []
    for path in runs:
        run = parse_run(path)
        if run is None:
            continue
        per, scales, first = {}, [], True
        for c in run["cams"]:
            dp, kp = path / f"depth_{c}.npy", path / f"K_{c}.npy"
            if not (dp.exists() and kp.exists()):
                continue
            P = backproject_camera(np.load(dp).astype(float),
                                   np.load(kp).astype(float), args.stride)
            cands = [x for x in peel_planes(P, args.planes, args.plane_tol,
                                            args.min_frac)
                     if x["tilt_deg"] <= args.max_tilt_deg]
            if not cands:
                continue
            best, how, ratio = identify_deck(cands, expect[c], belt_w, args)
            head = path.name if first else ""
            first = False
            if best is None:
                print(f"{head:30s} {c:>6s} {'-':>7s} {'-':>8s} {'-':>7s} "
                      f"{'-':>8s} {'-':>6s} {'UNIDENTIFIED':>11s}")
                per[c] = {"identified": False}
                continue
            w = best["patch"]["width"] if best["patch"] else float("nan")
            print(f"{head:30s} {c:>6s} {best['standoff']:>6.3f}m "
                  f"{best['g_standoff']:>8.4f} {w:>6.3f}m "
                  + (f"{best['g_width']:>8.4f}" if best['g_width'] else f"{'-':>8s}")
                  + (f" {best['width_disagree'] * 100:>5.1f}%"
                     if best['width_disagree'] is not None else f" {'-':>6s}")
                  + f" {how:>11s}")
            per[c] = {"identified": True, "how": how,
                      "standoff_gauge_m": round(best["standoff"], 4),
                      "g_standoff": round(best["g_standoff"], 5),
                      "patch_width_gauge_m": (None if not best["patch"] else
                                              round(best["patch"]["width"], 4)),
                      "g_width": (None if best["g_width"] is None
                                  else round(best["g_width"], 5)),
                      "width_disagree": (None if best["width_disagree"] is None
                                         else round(best["width_disagree"], 4)),
                      "inlier_frac": round(best["frac"], 4),
                      "tilt_deg": best["tilt_deg"],
                      "pair": ratio}
            scales.append(best["g_standoff"])
        if not scales:
            print(f"{path.name:30s} {'':>6s} no camera identified the "
                  f"conveyor; dropped")
            records.append({"run": path.name, "subset": run["subset"],
                            "res": run["res"], "scale_deck": None,
                            "n_identified": 0, "n_cams": len(run["cams"]),
                            "per_camera": per})
            continue
        rec = {"run": path.name, "subset": run["subset"], "res": run["res"],
               "scale_deck": round(float(np.median(scales)), 5),
               "scale_deck_spread": round(float(max(scales) - min(scales)), 5),
               "n_identified": len(scales), "n_cams": len(run["cams"]),
               "scale_baseline": base.get(path.name), "per_camera": per}
        records.append(rec)
        if len(scales) < len(run["cams"]):
            print(f"{'':30s} {'':>6s} only {len(scales)} of "
                  f"{len(run['cams'])} cameras identified the conveyor")

    args.out.write_text(json.dumps(records, indent=2))
    print(f"\nwritten: {args.out}")
    full = [r for r in records if r["n_identified"] == r["n_cams"]]
    print(f"{len(full)} of {len(records)} runs identified the conveyor in "
          f"every camera")
    if full:
        sp = [r["scale_deck_spread"] for r in full if r["n_cams"] > 1]
        if sp:
            print(f"cross-camera spread of the gauge, now a CHECK rather than "
                  f"the selector: median {np.median(sp):.4f}, worst "
                  f"{max(sp):.4f}")
        print("A small spread here means something now: the cameras were not "
              "chosen to agree, they were each identified against the "
              "conveyor's own width and then happened to agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())