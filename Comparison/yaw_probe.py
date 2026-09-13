#!/usr/bin/env python3
"""
yaw_probe.py — measure the scan-to-rig yaw error instead of eyeballing it.

    python3 yaw_probe.py --calib $CAL --snap captures/scene_c \\
        --ply captures/scene_c/pointcloud_20260825_144042.ply \\
        --gt gt_scene_c.json

A correctly registered 3D point samples the SAME colour in every camera that
sees it. A misregistered one samples unrelated pixels, so the cross-camera
colour disagreement rises. Rotating the scan about the deck normal and scoring
that disagreement gives a curve with a clear minimum at the true correction.

This is the standard multi-view photo-consistency argument, and unlike counting
in-frame points it actually discriminates -- in-frame counts do not, which is
why the earlier version of this idea was removed.

Reports a coarse sweep, then refines. Prints the correction to APPLY IN
scan_register.py; it deliberately writes nothing, because a transform patched
downstream would disagree with everything else that reads the scan.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import rigkit as rk


def yaw_about(n, deg):
    n = np.asarray(n, float) / np.linalg.norm(n)
    t = np.radians(deg)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def score(rig, imgs, Q, cen, n, deg):
    """Mean cross-camera colour spread for points seen by >= 2 cameras."""
    Qr = (Q - cen) @ yaw_about(n, deg).T + cen
    cols, seen = [], []
    for c in rig.cams:
        im = imgs[c]
        H, W = im.shape[:2]
        uv = rig.project(Qr, c)
        ok = (~np.isnan(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1))
        s = np.zeros((len(Qr), 3), np.float32)
        u = uv[ok].astype(np.int32)
        s[ok] = im[u[:, 1], u[:, 0]].astype(np.float32)
        cols.append(s)
        seen.append(ok)
    C = np.stack(cols)                       # (cam, N, 3)
    S = np.stack(seen)                       # (cam, N)
    k = S.sum(0)
    use = k >= 2
    if use.sum() < 500:
        return np.nan, int(use.sum())
    m = S[:, use][..., None].astype(np.float32)
    X = C[:, use] * m
    mu = X.sum(0) / m.sum(0)
    var = (((C[:, use] - mu) ** 2) * m).sum(0) / m.sum(0)
    return float(np.sqrt(var.mean(1)).mean()), int(use.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--cams", nargs="+")
    ap.add_argument("--pattern", default="{cam}.png")
    ap.add_argument("--max-points", type=int, default=120000)
    a = ap.parse_args()

    rig = rk.Rig(a.calib, cams=a.cams)
    imgs = {}
    for c in rig.cams:
        p = Path(a.snap) / a.pattern.format(cam=c)
        if not p.exists():
            raise SystemExit(f"{p} not found")
        imgs[c] = cv2.imread(str(p))
    print(f"{len(imgs)} images, {' '.join(rig.cams)}")

    P = rk.load_cloud(a.ply)
    if a.gt:
        g = json.load(open(a.gt))
        n = np.array(g["deck_plane"]["n"], float)
        d = float(g["deck_plane"]["d"])
    else:
        h_, e_ = np.histogram(P[:, 2], bins=200)
        k = int(h_.argmax())
        b = (P[:, 2] > e_[k] - 0.1) & (P[:, 2] < e_[k + 1] + 0.1)
        n, d = rk.fit_plane_ransac(P[b])
    hgt = -(P @ n + d)
    Q = P[(hgt > 0.02) & (hgt < 0.60)]        # things ON the deck carry texture
    if len(Q) > a.max_points:
        Q = Q[np.random.default_rng(0).choice(len(Q), a.max_points, False)]
    cen = Q.mean(0)
    print(f"{len(Q)} textured points standing on the deck\n")

    print(f"{'yaw deg':>8s} {'colour spread':>14s} {'pts >=2 views':>14s}")
    coarse = [(score(rig, imgs, Q, cen, n, g_), g_)
              for g_ in range(-180, 180, 15)]
    for (s, k), g_ in [((s, k), g_) for (s, k), g_ in coarse]:
        bar = "#" * int(40 * (1 - min(s, 120) / 120)) if s == s else ""
        print(f"{g_:8d} {s:14.2f} {k:14d}  {bar}")
    best = min((c for c in coarse if c[0][0] == c[0][0]), key=lambda c: c[0][0])
    g0 = best[1]
    fine = [(score(rig, imgs, Q, cen, n, g0 + dg)[0], g0 + dg)
            for dg in np.arange(-12, 12.5, 1.5)]
    fine = [f for f in fine if f[0] == f[0]]
    bg = min(fine)[1]
    print(f"\nbest yaw {bg:+.1f} deg   colour spread {min(fine)[0]:.2f} "
          f"(vs {dict((g_, s) for (s, _), g_ in coarse).get(0, float('nan')):.2f} "
          f"at 0)")
    flat = min(fine)[0] > 0.9 * max(f[0] for f in fine)
    if flat:
        print("*** the curve is flat -- not enough texture to decide. Do not "
              "trust this.")
    elif abs(bg) < 4:
        print("The registration is already correct in yaw. Look elsewhere.")
    else:
        print(f"Rotate the scan by {bg:+.1f} deg about the deck normal.\n"
              f"APPLY THIS IN scan_register.py, not downstream.")


if __name__ == "__main__":
    main()