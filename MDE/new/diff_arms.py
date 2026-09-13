#!/usr/bin/env python3
"""
diff_arms.py

Put two runs of the same camera subset side by side, per camera, to find why
their layering differs.

WHY layer_above_sd_mm IS NOT ENOUGH
-----------------------------------
It is one number over all views, so it cannot say which camera is the outlier
or whether the spread comes from a bad scale, from real depth error, or from
too few points to take a stable median over. On this rig the same subset at the
same resolution reported 12.9 mm in the prior-free arm and 209 mm in the prior
arm, with half the points and lower coverage, and the summary statistic cannot
distinguish the three explanations:

  a bad per-camera gauge      one camera's tag scale is wrong, so its deck sits
                              high and everything else follows
  real residual depth error   the prior-free arm's gauge had to absorb the
                              focal error AND the depth error per camera, so it
                              corrected more; with the prior, focal is already
                              right and only the depth error is left to show
  a sampling artefact         too few points above the deck for a per-view
                              median to mean anything

This reports, per camera, the gauge applied, the point count, how many points
land near the deck and above it, and the median height of each — so the three
separate.

WHAT TO LOOK FOR
----------------
If one camera's deck offset is large and its gauge is the odd one out, it is
the first case and the gauge is wrong. If the offsets are spread but the gauges
agree, it is the second and the spread is real. If a camera has only a few
hundred points above the deck, its median is noise and neither conclusion
holds.

EXAMPLE
-------
    python3 diff_arms.py \\
        --a runs_da3/cfg1008/left+top+right_res1008 \\
        --b runs_da3/resweep_prior/left+top+right_res1008 \\
        --gauge-a metric_fit.json --gauge-b metric_fit_prior.json

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

CAMS = ["center", "left", "top", "right"]


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


def gauge_for(path, run_name, cam):
    if path is None or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    rec = d.get(run_name) if isinstance(d, dict) else None
    if not rec:
        return None
    per = rec.get("per_camera_scale") or {}
    return per.get(cam, rec.get("scale"))


def report(run, gauge_path, belt, frame, args, label):
    c, ex, ey, n = frame
    cams = cams_of(run)
    print(f"\n[{label}] {Path(run).name}")
    print(f"  {'camera':>7s} {'gauge':>8s} {'points':>10s} {'near deck':>10s} "
          f"{'deck md':>9s} {'above':>9s} {'above md':>9s}")
    decks, aboves = {}, {}
    for cam in cams:
        f = Path(run) / f"cloud_calib_{cam}.ply"
        if not f.exists():
            print(f"  {cam:>7s}   no cloud_calib_{cam}.ply")
            continue
        P = rigkit.load_cloud(str(f))
        rel = P - c
        u, v, h = rel @ ex, rel @ ey, rel @ n
        on = ((np.abs(u) <= belt["length_m"] / 2)
              & (np.abs(v) <= belt["width_m"] / 2))
        near = on & (np.abs(h) <= args.deck_band)
        above = on & (h > args.above_min) & (h < args.above_max)
        g = gauge_for(gauge_path, Path(run).name, cam)
        dmd = float(np.median(h[near])) * 1e3 if near.sum() > 200 else np.nan
        amd = float(np.median(h[above])) * 1e3 if above.sum() > 200 else np.nan
        decks[cam], aboves[cam] = dmd, amd
        print(f"  {cam:>7s} " + (f"{g:>8.4f}" if g else f"{'-':>8s}")
              + f" {len(P):>10d} {int(near.sum()):>10d} {dmd:>8.1f}mm "
              f"{int(above.sum()):>9d} {amd:>8.1f}mm")
    for what, d in (("deck", decks), ("above deck", aboves)):
        v = [x for x in d.values() if np.isfinite(x)]
        if len(v) > 1:
            print(f"  spread {what:11s} range {max(v) - min(v):7.1f} mm   "
                  f"sd {np.std(v, ddof=1):6.1f} mm")
    return decks, aboves


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-camera deck placement, two runs side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--gauge-a", type=Path, default=Path("metric_fit.json"))
    ap.add_argument("--gauge-b", type=Path,
                    default=Path("metric_fit_prior.json"))
    ap.add_argument("--belt", type=Path, default=Path("belt.json"))
    ap.add_argument("--deck-band", type=float, default=0.030)
    ap.add_argument("--above-min", type=float, default=0.060)
    ap.add_argument("--above-max", type=float, default=0.600)
    ap.add_argument("--min-points", type=int, default=2000,
                    help="below this above the deck, a per-view median is not "
                         "a measurement and the comparison says so")
    args = ap.parse_args()

    belt = json.loads(args.belt.read_text())
    frame = belt_frame(belt)
    da, aa = report(args.a, args.gauge_a, belt, frame, args, "A")
    db, ab = report(args.b, args.gauge_b, belt, frame, args, "B")

    print("\nVERDICT")
    for what, x, y in (("deck", da, db), ("above deck", aa, ab)):
        xv = [v for v in x.values() if np.isfinite(v)]
        yv = [v for v in y.values() if np.isfinite(v)]
        if len(xv) > 1 and len(yv) > 1:
            print(f"  {what:11s} A spread {max(xv) - min(xv):7.1f} mm   "
                  f"B spread {max(yv) - min(yv):7.1f} mm")
    shared = [c for c in da if c in db]
    if shared:
        worst = max(shared, key=lambda c: abs(
            (db[c] if np.isfinite(db[c]) else 0)
            - (da[c] if np.isfinite(da[c]) else 0)))
        print(f"  the camera that moves most between the arms is {worst}: "
              f"{da[worst]:.1f} mm in A, {db[worst]:.1f} mm in B")
    print("\n  one camera out of line AND its gauge the odd one out -> the "
          "gauge is wrong, and the layering is an artefact.")
    print("  offsets spread but gauges in agreement -> the spread is real "
          "per-view depth error, which is what depth_correction.json is for.")
    print("  a camera with only a few hundred points above the deck -> its "
          "median is noise and neither conclusion follows from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())