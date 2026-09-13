#!/usr/bin/env python3
"""
mask_audit.py

Why a camera has no observations of a reference plane.

depth_field.py reports n_assigned per plane, which conflates three different
failures into one small number:

  1. the plane is not in this camera's frustum at all
  2. the plane is in view, but the pixels are removed by conf/edge/incidence
  3. the plane is in view and unmasked, but the depth error exceeds the
     assignment window, so nothing lands inside it

Only (2) is fixable by tuning. (1) needs a new ground-truth capture. (3) needs a
wider window or a segmentation mask rather than a proximity window.

This prints all three, per camera per plane, using EXACTLY depth_field's masks.

  python mask_audit.py \
      --capture-dir runs/live_20260817_144844/capture_00001 \
      --gt ground_truth.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from da3_fuse import points_camera, edge_mask, incidence_mask
from depth_field import (gt_planes_world, plane_in_camera, target_depth,
                         load_capture)


def build_masks(e, a):
    """The four masks of depth_field.valid_mask, returned separately."""
    d, K = e["depth"], e["K"]
    finite = np.isfinite(d) & (d > 0)

    m_conf = np.ones_like(finite)
    if a.conf_percentile > 0 and e["conf"] is not None and finite.any():
        thr = float(np.percentile(e["conf"][finite], a.conf_percentile))
        m_conf = e["conf"] >= thr

    m_edge = (edge_mask(d, a.edge_thresh, a.edge_dilate)
              if a.edge_thresh > 0 else np.ones_like(finite))

    m_inc = np.ones_like(finite)
    if a.max_incidence < 90:
        m_inc, _ = incidence_mask(points_camera(d, K), a.max_incidence)

    return finite, {"confidence": m_conf, "edge": m_edge, "incidence": m_inc}


def main():
    ap = argparse.ArgumentParser(
        description="Separate 'not in view' from 'masked out' from 'outside "
                    "the window', per camera per reference plane.")
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("ground_truth.json"))
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--band-mm", type=float, default=150.0,
                    help="half-width of the band around a plane counted as "
                         "'a pixel that could plausibly belong to it'. Keep "
                         "this below the shortest parcel height or the band "
                         "swallows the parcel tops")
    # identical defaults to depth_field.py
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--min-plane-points", type=int, default=500)
    a = ap.parse_args()

    data = load_capture(a.capture_dir, list(a.cameras))
    planes = gt_planes_world(a.gt, data[a.reference]["E"])
    band = a.band_mm * 1e-3

    print(f"[audit] {len(planes)} plane(s) from {a.gt}, band +-{a.band_mm:.0f} mm")

    for n in a.cameras:
        e = data[n]
        d, K = e["depth"], e["K"]
        finite, masks = build_masks(e, a)
        valid = finite.copy()
        for m in masks.values():
            valid &= m

        print(f"\n=== {n}   {d.shape[1]}x{d.shape[0]}   {d.size} px")
        print(f"    finite depth        {int(finite.sum()):9d} "
              f"({100 * finite.mean():5.1f}% of frame)")
        for tag, m in masks.items():
            killed = int((finite & ~m).sum())
            print(f"    {tag:<12} removes {killed:9d} "
                  f"({100 * killed / max(int(finite.sum()), 1):5.1f}% of finite)")
        print(f"    all masks applied   {int(valid.sum()):9d} "
              f"({100 * valid.mean():5.1f}% of frame)")

        for plane in planes:
            n_c, off_c = plane_in_camera(plane, e["E"])
            z_t, ok, r = target_depth(d.shape, K, n_c, off_c)
            res = (d - z_t) * r                      # perpendicular, metres

            in_band_raw = finite & ok & (np.abs(res) <= band)
            in_band_val = valid & ok & (np.abs(res) <= band)

            print(f"\n    plane {plane['label']:<8} "
                  f"perp {plane['perp_ref_m']:.4f} m from {a.reference}")
            print(f"      rays meeting it in front of the camera "
                  f"{int(ok.sum()):9d}   <- FRUSTUM. Near zero here means the "
                  f"plane is not in this camera's view and no tuning helps")
            print(f"      within the band, before masks           "
                  f"{int(in_band_raw.sum()):9d}")
            print(f"      within the band, after masks            "
                  f"{int(in_band_val.sum()):9d}")

            if int(in_band_raw.sum()) > 0:
                print("      which mask costs the band:", end="")
                for tag, m in masks.items():
                    kept = int((in_band_raw & m).sum())
                    print(f"  {tag} keeps {kept}", end="")
                print()

            # (3): is it the WINDOW, not the masks?
            sel = valid & ok
            wide = sel & (np.abs(res) <= 4 * band)
            if int(wide.sum()) >= 200:
                p = np.percentile(res[wide] * 1e3, [5, 25, 50, 75, 95])
                print("      residual over the wide band, mm  "
                      "p5 %+8.1f  p25 %+8.1f  p50 %+8.1f  p75 %+8.1f  p95 %+8.1f"
                      % tuple(p))
            counts = []
            for w in (15, 25, 50, 100, 200, 400):
                counts.append((w, int((sel & (np.abs(res) <= w * 1e-3)).sum())))
            print("      cumulative count by window: "
                  + "  ".join(f"{w}mm {c}" for w, c in counts))
            need = a.min_plane_points
            passes = [w for w, c in counts if c >= need]
            print(f"      smallest window reaching --min-plane-points {need}: "
                  + (f"{min(passes)} mm" if passes
                     else f"NONE up to 400 mm  <- this plane cannot enter the "
                          f"fit for {n} at any usable window"))

            # coverage: what fraction of the surface the fit would actually see
            if int(in_band_val.sum()) > 0:
                got = int((valid & ok & (np.abs(res) <= 0.015)).sum())
                print(f"      coverage at the 15 mm final window: {got} of "
                      f"{int(in_band_val.sum())} in-band pixels "
                      f"({100 * got / int(in_band_val.sum()):5.1f}%)  <- a low "
                      f"figure means the fit is solved on a BAND of the "
                      f"surface and its tilt terms are extrapolated across the "
                      f"rest of the frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())