#!/usr/bin/env python3
"""
overlay_check.py — draw each parcel's BASE and TOP quad on every camera.

    python3 overlay_check.py --calib $CAL --snap captures/scene_c \\
        --gt gt_scene_c.json --out check_c

verify_gt_alignment.py counts corners inside the frame. That is a visibility
check and it cannot tell you whether the quad landed on the box. This draws two
quads per parcel and lets you read the failure mode straight off the image:

  BASE  (thin, on the deck at h=0)   TOP  (thick, at h=H)

  * base on the box, top beside it        -> the HEIGHT is wrong. The top face
                                             projects radially away from the
                                             camera centre in proportion to the
                                             height error, so the displacement
                                             grows with H and points away from
                                             the principal point.
  * both displaced the same way, in ALL   -> the GROUND TRUTH is wrong: either
    four cameras                             the scan-to-rig registration, or
                                             extract_gt's component.
  * both displaced in ONE camera only     -> that camera's extrinsics.
  * quad spans two objects                -> extract_gt merged them. Check the
                                             printed L/W against what you see.

It also prints, per parcel, how far the top quad sits from the base quad in
pixels, which is the number to compare across cameras.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import rigkit as rk

PAL = [(60, 60, 240), (60, 220, 60), (240, 160, 40), (220, 60, 220),
       (40, 220, 220), (240, 240, 60), (150, 100, 250), (100, 200, 150)]


def quad(gt, x, y, L, W, h):
    return np.array([rk.from_deck(x + du, y + dv, h, gt)
                     for du, dv in ((-L/2, -W/2), (-L/2, W/2),
                                    (L/2, W/2), (L/2, -W/2))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cams", nargs="+")
    ap.add_argument("--pattern", default="{cam}.png")
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    rig = rk.Rig(a.calib, cams=a.cams)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for c in rig.cams:
        p = Path(a.snap) / a.pattern.format(cam=c)
        if not p.exists():
            print(f"{c}: {p} not found, skipped")
            continue
        img = cv2.imread(str(p))
        print(f"\n{c}")
        for i, par in enumerate(gt["parcels"]):
            L = par["L_mm"] / 1000
            W = par["W_mm"] / 1000
            H = par["H_mm"] / 1000
            col = PAL[i % len(PAL)]
            uv0 = rig.project(quad(gt, par["x"], par["y"], L, W, 0.0), c)
            uv1 = rig.project(quad(gt, par["x"], par["y"], L, W, H), c)
            if np.isnan(uv0).any() or np.isnan(uv1).any():
                print(f"  {par.get('id', i)}  behind the camera")
                continue
            cv2.polylines(img, [uv0.astype(np.int32)], True, col, 2)
            cv2.polylines(img, [uv1.astype(np.int32)], True, col, 6)
            for k in range(4):                       # verticals
                cv2.line(img, tuple(uv0[k].astype(int)),
                         tuple(uv1[k].astype(int)), col, 1)
            lab = uv1.mean(0)
            cv2.putText(img, f"{par.get('id', i)} {par['H_mm']:.0f}mm",
                        tuple((lab + [8, -8]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
            shift = float(np.linalg.norm(uv1.mean(0) - uv0.mean(0)))
            edge = float(np.linalg.norm(uv1[0] - uv1[1]))
            print(f"  {par.get('id', i):4s} H={par['H_mm']:6.1f} mm   "
                  f"top sits {shift:6.1f} px from base   "
                  f"({shift/max(edge,1e-6):5.2f} of a short edge)   "
                  f"scan {par['L_mm']:.0f}x{par['W_mm']:.0f} mm")
        f = out / f"check_{c}.png"
        cv2.imwrite(str(f), img)
        print(f"  -> {f}")

    print("\nthin = base on the deck, thick = top face.")
    print("If the thin quad is on the box and the thick one is not, the HEIGHT "
          "is wrong,\nnot the position. If both are off in all four cameras, "
          "the ground truth is wrong.")


if __name__ == "__main__":
    main()