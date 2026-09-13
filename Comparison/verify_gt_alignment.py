#!/usr/bin/env python3
"""
verify_gt_alignment.py — draw the reference-scan parcel outlines onto the
captured frames. Run this BEFORE any DA3 work; every number downstream is
conditional on this transform being right.

    python3 verify_gt_alignment.py \
        --calib /home/jetson/Projects/Calibration4/results \
        --snap  /home/jetson/Projects/Calibration4/snapshots/20260707_105753_062 \
        --gt gt_scene.json --out overlays

Read the result:
  outlines land on the parcels          -> scene and registration both good
  outlines uniformly offset, all 3 cams -> registration offset; rerun scan_register.py
  outlines offset differently per cam   -> extrinsics wrong, not registration
  one parcel moved, the rest fine       -> the scene changed between capture and scan

For the 20260707 pair specifically there are ~90 minutes between the snapshot
directory (10:57:53) and the scan (12:28:11), so the last case is live.
"""
import argparse
import json
import os
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("needs opencv:  pip install opencv-python")

import rigkit as rk

COLOUR = [(0, 0, 255), (0, 220, 0), (255, 140, 0), (255, 0, 255),
          (0, 220, 220), (200, 200, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--gt", default="gt_scene.json")
    ap.add_argument("--out", default="overlays")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="downscale factor for the written overlays")
    a = ap.parse_args()

    rig = rk.Rig(a.calib)
    gt = json.load(open(a.gt))
    os.makedirs(a.out, exist_ok=True)

    for c in rig.cams:
        path = os.path.join(a.snap, f"{c}.png")
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] {path} not found")
            continue
        h, w = img.shape[:2]
        for i, p in enumerate(gt["parcels"]):
            col = COLOUR[i % len(COLOUR)]
            uv = rig.project(np.array(p["corners_xyz"]), c)
            if np.isnan(uv).any():
                print(f"{c:7s} {p['id']}  behind camera")
                continue
            cv2.polylines(img, [np.round(uv).astype(int)], True, col, 4)
            cu, cv_ = rig.project(np.array([p["top_centre_xyz"]]), c)[0]
            cv2.drawMarker(img, (int(round(cu)), int(round(cv_))), col,
                           cv2.MARKER_CROSS, 40, 4)
            inside = int(((uv[:, 0] >= 0) & (uv[:, 0] < w) &
                          (uv[:, 1] >= 0) & (uv[:, 1] < h)).sum())
            cv2.putText(img, f"{p['id']} {inside}/4",
                        (int(uv[0][0]), max(30, int(uv[0][1]) - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            print(f"{c:7s} {p['id']}  corners in frame {inside}/4   "
                  f"scan {p['L_mm']:.0f}x{p['W_mm']:.0f}x{p['H_mm']:.0f} mm")
        ring = rk.from_deck([-1.6, 1.6, 1.6, -1.6], [-1.2, -1.2, 1.2, 1.2],
                            [0, 0, 0, 0], gt)
        uv = rig.project(ring, c)
        if not np.isnan(uv).any():
            cv2.polylines(img, [np.round(uv).astype(int)], True, (190, 190, 190), 2)
        if a.scale != 1.0:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale)
        out = os.path.join(a.out, f"overlay_{c}.png")
        cv2.imwrite(out, img)
        print(f"  -> {out}\n")


if __name__ == "__main__":
    main()