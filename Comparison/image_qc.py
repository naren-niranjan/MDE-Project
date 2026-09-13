#!/usr/bin/env python3
"""
image_qc.py — compare the source images per camera.

Left degrades every subset it enters, resolves fewer depth layers than center
or right, and reconstructs half the deck area. Roll explained part of it. This
checks whether the rest is in the image itself: focus, exposure, contrast,
clipping. A soft or flat image gives DA3 less to work with and shows up
exactly as a poorly resolved ground plane.

    python3 image_qc.py --images /path/to/snapshot_dir
    python3 image_qc.py --images DIR --roi 0.3 0.7 0.3 0.7   # centre crop only

Read it as a comparison, not against absolutes. Cameras viewing the same scene
from similar distances should land within a factor of about two on sharpness.
"""
import argparse
import os
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("needs opencv:  pip install opencv-python")

CAMS = ["center", "left", "right", "top"]


def stats(g, name):
    lap = cv2.Laplacian(g, cv2.CV_64F)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # high-frequency energy fraction: robust to exposure, unlike raw variance
    f = np.fft.fftshift(np.abs(np.fft.fft2(g.astype(np.float32))))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    yy, xx = np.ogrid[:h, :w]
    hi = ((yy - cy) ** 2 + (xx - cx) ** 2) > r * r
    return dict(
        name=name,
        mean=float(g.mean()),
        std=float(g.std()),
        p1=float(np.percentile(g, 1)),
        p99=float(np.percentile(g, 99)),
        clip_lo=float((g <= 2).mean() * 100),
        clip_hi=float((g >= 253).mean() * 100),
        lapvar=float(lap.var()),
        edge=float((mag > 30).mean() * 100),
        hf=float(f[hi].sum() / f.sum() * 100),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--cameras", nargs="+", default=CAMS)
    ap.add_argument("--map", action="store_true",
                    help="print a coarse map of where the clipping falls. "
                         "Blown deck is different from blown parcel tops: the "
                         "first costs you the plane fit, the second costs you "
                         "the thing you are measuring.")
    ap.add_argument("--map-cells", type=int, default=12)
    ap.add_argument("--roi", type=float, nargs=4,
                    metavar=("X0", "X1", "Y0", "Y1"),
                    help="fractional crop, e.g. 0.3 0.7 0.3 0.7")
    a = ap.parse_args()

    rows = []
    for c in a.cameras:
        p = None
        for nm in (f"{c}.png", f"proc_{c}.png", f"{c}.jpg"):
            q = os.path.join(a.images, nm)
            if os.path.exists(q):
                p = q
                break
        if p is None:
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] cannot read {p}")
            continue
        if a.roi:
            h, w = img.shape
            x0, x1, y0, y1 = a.roi
            img = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        rows.append(stats(img, c))
        rows[-1]["shape"] = f"{img.shape[1]}x{img.shape[0]}"
        rows[-1]["img"] = img

    if not rows:
        raise SystemExit(f"no camera images found in {a.images}")

    print(f"{'camera':8s} {'size':11s} {'mean':>6s} {'std':>6s} {'p1':>5s} "
          f"{'p99':>5s} {'clip%lo':>8s} {'clip%hi':>8s} {'lapvar':>9s} "
          f"{'edge%':>7s} {'HF%':>6s}")
    for r in rows:
        print(f"{r['name']:8s} {r['shape']:11s} {r['mean']:6.1f} {r['std']:6.1f} "
              f"{r['p1']:5.0f} {r['p99']:5.0f} {r['clip_lo']:8.2f} "
              f"{r['clip_hi']:8.2f} {r['lapvar']:9.1f} {r['edge']:7.2f} "
              f"{r['hf']:6.2f}")

    if a.map:
        key = " . <1%   : <5%   o <15%  O <35%  # >=35% blown"
        for r in rows:
            g = r["img"]
            n = a.map_cells
            h, w = g.shape
            print(f"\n  {r['name']} — clipped-highlight map ({key})")
            for iy in range(n):
                row = ""
                for ix in range(int(n * w / h)):
                    blk = g[iy * h // n:(iy + 1) * h // n,
                            ix * w // int(n * w / h):
                            (ix + 1) * w // int(n * w / h)]
                    f = (blk >= 253).mean()
                    row += (" ." if f < .01 else " :" if f < .05 else
                            " o" if f < .15 else " O" if f < .35 else " #")
                print("   " + row)

    lv = {r["name"]: r["lapvar"] for r in rows}
    best = max(lv.values())
    print()
    for n, v in sorted(lv.items(), key=lambda kv: kv[1]):
        if v < best / 2:
            print(f"  *** {n} sharpness is {best/v:.1f}x below the best camera. "
                  f"Soft focus or motion blur gives DA3 less structure to fit, "
                  f"which reads as a poorly resolved ground plane.")
    for r in rows:
        if r["clip_hi"] > 2:
            print(f"  *** {r['name']}: {r['clip_hi']:.1f}% of pixels blown out")
        if r["clip_lo"] > 5:
            print(f"  *** {r['name']}: {r['clip_lo']:.1f}% of pixels crushed to black")
        if r["std"] < 0.6 * max(x["std"] for x in rows):
            print(f"  *** {r['name']}: contrast {r['std']:.1f} vs "
                  f"{max(x['std'] for x in rows):.1f} on the best camera")


if __name__ == "__main__":
    main()