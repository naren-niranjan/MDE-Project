#!/usr/bin/env python3
"""
Refine a three-camera MoGe merge.

Placement from extrinsics gets you close; MoGe's per-camera scale drift leaves
left/right slightly offset from center. This snaps each side cloud onto the
(fixed) center cloud with a scaled ICP, then voxel-dedups the union so
overlapping regions collapse to a single surface instead of stacking copies.

Inputs are the per-camera clouds written by the MoGe script with
--save-individual (already placed in the center frame):
    cloud_left.ply  cloud_center.ply  cloud_right.ply

Usage:
    python refine_merge.py --dir /home/jetson/Projects/MultiCamera/Moge2 \
        --voxel 0.004
    # rigid-only (no scale) if you trust MoGe's metric scale:
    python refine_merge.py --dir ... --rigid
"""

import argparse
import os
import numpy as np
from scipy.spatial import cKDTree


# ---- minimal PLY read/write with color ------------------------------------ #
def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().split()[1]
        n, props = None, []
        tm = {b"float": "f4", b"float32": "f4", b"double": "f8",
              b"uchar": "u1", b"uint8": "u1"}
        while True:
            ln = f.readline().split()
            if ln and ln[0] == b"element" and ln[1] == b"vertex":
                n = int(ln[2])
            elif ln and ln[0] == b"property" and ln[1] != b"list":
                props.append((ln[2].decode(), tm[ln[1]]))
            elif ln and ln[0] == b"end_header":
                break
        endian = "<" if b"little" in fmt else ">"
        dt = np.dtype([(nm, endian + tc) for nm, tc in props])
        rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    if "red" in rec.dtype.names:
        rgb = np.stack([rec["red"], rec["green"], rec["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 200, np.uint8)
    ok = np.isfinite(xyz).all(1)
    return xyz[ok], rgb[ok]


def write_ply(path, xyz, rgb):
    n = xyz.shape[0]
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.empty(n, dt)
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(v.tobytes())


# ---- Umeyama (scale optional) + scaled ICP -------------------------------- #
def umeyama(src, dst, with_scale=True):
    ms, md = src.mean(0), dst.mean(0)
    Sc, Dc = src - ms, dst - md
    U, S, Vt = np.linalg.svd((Dc.T @ Sc) / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (np.trace(np.diag(S) @ D) / ((Sc ** 2).sum() / len(src))) if with_scale else 1.0
    t = md - s * R @ ms
    return s, R, t


def icp(src, dst, with_scale=True, iters=60, trim=0.8):
    tree = cKDTree(dst)
    span = np.linalg.norm(dst.max(0) - dst.min(0))
    thresh = 0.05 * span
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        d, idx = tree.query(cur, k=1)
        order = np.argsort(d)
        keep = order[: int(trim * len(order))]
        keep = keep[d[keep] < thresh]
        if len(keep) < 50:
            break
        s, R, t = umeyama(src[keep], dst[idx[keep]], with_scale)
        cur = (s * (R @ src.T)).T + t
        thresh = max(3.0 * np.median(d[keep]), 0.002)
    d, _ = tree.query(cur, k=1)
    return s, R, t, np.sqrt((d ** 2).mean())


def voxel_dedup(xyz, rgb, voxel):
    if voxel <= 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--voxel", type=float, default=0.004, help="dedup voxel (m)")
    ap.add_argument("--rigid", action="store_true", help="disable scale in ICP")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with_scale = not args.rigid
    C_xyz, C_rgb = read_ply(os.path.join(args.dir, "cloud_center.ply"))
    print(f"center: {len(C_xyz):,} pts (fixed reference)")

    all_xyz, all_rgb = [C_xyz], [C_rgb]
    for cam in ("left", "right"):
        p = os.path.join(args.dir, f"cloud_{cam}.ply")
        xyz, rgb = read_ply(p)
        # subsample for the ICP solve (apply result to full cloud)
        rng = np.random.default_rng(0)
        src = xyz if len(xyz) <= 120000 else xyz[rng.choice(len(xyz), 120000, False)]
        dst = C_xyz if len(C_xyz) <= 120000 else C_xyz[rng.choice(len(C_xyz), 120000, False)]
        s, R, t, rms = icp(src, dst, with_scale)
        xyz_ref = (s * (R @ xyz.T)).T + t
        print(f"{cam:>6}: scale={s:.4f}  |t|={np.linalg.norm(t)*1000:5.1f}mm  "
              f"post-ICP RMS={rms*1000:5.1f}mm  ({len(xyz):,} pts)")
        all_xyz.append(xyz_ref); all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    xyz, rgb = voxel_dedup(xyz, rgb, args.voxel)
    out = args.out or os.path.join(args.dir, "fused_refined.ply")
    write_ply(out, xyz.astype(np.float32), rgb)
    print(f"\nwrote {out}  ({len(xyz):,} pts, voxel {args.voxel}m)")


if __name__ == "__main__":
    main()