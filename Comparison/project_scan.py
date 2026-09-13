#!/usr/bin/env python3
"""
project_scan.py — project the registered scan into each camera and blend it
over the photo. The decisive test of scan_register.py.

    python3 project_scan.py --calib $CAL --snap captures/scene_c \\
        --ply captures/scene_c/pointcloud_20260825_144042.ply \\
        --gt gt_scene_c.json --out proj_c

The yellow rectangle is the scan's own deck footprint. Compare its long axis
with the belt in the photo -- that comparison alone settles it.

If the registration is right, the coloured points land ON the objects: the
belt edges follow the belt, box lids sit on box lids. If the transform carries
a yaw error about the deck normal, the scan appears rotated in the frame while
still having the correct internal shape -- which is invisible to every plane
and separation check, because all of those are rotation-invariant.

There is deliberately no automatic yaw estimator here. Scoring candidate
rotations by how many points land in frame does not discriminate -- tried, and
it returns a different answer per camera. Read the overlay instead, and fix
scan_register.py rather than patching the transform downstream, or everything
else that reads the scan will disagree with this.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import rigkit as rk


def yaw_about(n, deg):
    """Rotation by `deg` about the unit axis n (the deck normal)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    t = np.radians(deg)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", help="gt json, for the deck plane (else fitted)")
    ap.add_argument("--cams", nargs="+")
    ap.add_argument("--pattern", default="{cam}.png")
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="TEST ONLY: rotate the scan by this many degrees "
                         "about the deck normal before projecting. Use it to "
                         "find which correction lands, then fix "
                         "scan_register.py. Never leave a yaw applied here.")
    ap.add_argument("--dot", type=int, default=9,
                    help="dot size in px; raise if the wash is still faint")
    a = ap.parse_args()

    global RAD
    RAD = max(1, a.dot)
    rig = rk.Rig(a.calib, cams=a.cams)
    P = rk.load_cloud(a.ply)
    if len(P) > a.max_points:
        P = P[np.random.default_rng(0).choice(len(P), a.max_points, False)]

    if a.gt:
        g = json.load(open(a.gt))
        n = np.array(g["deck_plane"]["n"], float)
        d = float(g["deck_plane"]["d"])
    else:
        h_, e_ = np.histogram(P[:, 2], bins=200)
        k = int(h_.argmax())
        band = (P[:, 2] > e_[k] - 0.1) & (P[:, 2] < e_[k + 1] + 0.1)
        n, d = rk.fit_plane_ransac(P[band])
    hgt = -(P @ n + d)
    keep = (hgt > -0.05) & (hgt < 0.60)          # deck and what stands on it
    Q, hq = P[keep], hgt[keep]
    print(f"{len(Q)} scan points from the deck up to 600 mm")

    if a.yaw:
        cen = Q.mean(0)
        Q = (Q - cen) @ yaw_about(n, a.yaw).T + cen
        print(f"*** TEST YAW {a.yaw:+.1f} deg applied. This is a probe, not a "
              "fix -- the\n    correction belongs in scan_register.py.")
    centroid = Q.mean(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for c in rig.cams:
        p = Path(a.snap) / a.pattern.format(cam=c)
        if not p.exists():
            print(f"{c}: {p} not found")
            continue
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        uv = rig.project(Q, c)
        ok = (~np.isnan(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
              & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        u = uv[ok].astype(np.int32)
        col = cv2.applyColorMap(
            np.clip(hq[ok] / 0.6 * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO).reshape(-1, 3)

        # Single pixels are invisible against a 5 Mpx frame. Paint into a
        # canvas, dilate, and blend ONLY where painted, so the scan reads as a
        # solid wash rather than a 2 per cent speckle.
        lay = np.zeros_like(img)
        hit = np.zeros(img.shape[:2], np.uint8)
        lay[u[:, 1], u[:, 0]] = col
        hit[u[:, 1], u[:, 0]] = 255
        k = np.ones((RAD, RAD), np.uint8)
        lay = cv2.dilate(lay, k)
        hit = cv2.dilate(hit, k)
        m = hit.astype(bool)
        img[m] = (0.72 * lay[m] + 0.28 * img[m]).astype(np.uint8)

        # The decisive mark: the scan's belt footprint as a thick rectangle.
        # If this crosses the photo's belt at 90 degrees, the transform has a
        # yaw error about the deck normal.
        du = rk.to_deck(Q[np.abs(hq) < 0.02], {"deck_plane": {"n": n.tolist(),
                                                              "d": d}})
        if len(du) > 500:
            rect = cv2.minAreaRect(du[:, :2].astype(np.float32))
            box = cv2.boxPoints(rect)
            cor = rk.from_deck(box[:, 0], box[:, 1], np.zeros(4),
                               {"deck_plane": {"n": n.tolist(), "d": d}})
            p4 = rig.project(cor, c)
            if not np.isnan(p4).any():
                cv2.polylines(img, [p4.astype(np.int32)], True,
                              (0, 255, 255), 10)
                cv2.putText(img, f"scan deck {rect[1][0]:.2f} x "
                                 f"{rect[1][1]:.2f} m",
                            tuple(p4.mean(0).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5,
                            cv2.LINE_AA)
                print(f"{c}: {ok.sum():7d} pts in frame   scan deck rect "
                      f"{rect[1][0]:.2f} x {rect[1][1]:.2f} m")
            else:
                print(f"{c}: {ok.sum():7d} pts in frame   deck rect off-frame")
        f = out / f"proj_{c}.png"
        cv2.imwrite(str(f), img)
        print(f"   -> {f}")

    print("\nIf the coloured points do not lie on the objects, the scan-to-rig "
          "transform is\nwrong. A yaw error about the deck normal keeps every "
          "plane fit, tilt and\ndeck-floor separation correct, so only a "
          "projection like this can see it.")


if __name__ == "__main__":
    main()