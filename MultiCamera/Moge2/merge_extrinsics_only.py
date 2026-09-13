#!/usr/bin/env python3
"""
Merge the three already-placed per-camera clouds using the calibrated
extrinsics ONLY (no ICP). Also prints placement geometry so you can tell
whether the extrinsic direction was correct.

If placement is right, all three clouds land on the SAME physical scene, so
their centroids sit close together (tens of cm apart at most -- the offset is
just view coverage, not the ~0.8m camera baseline). If left/right centroids
are ~0.8m+ from center, the MoGe run used the wrong --extrinsics-direction and
you must re-export before any merge will look right.

Usage:
    python merge_extrinsics_only.py --dir /home/jetson/Projects/MultiCamera/Moge2 \
        --voxel 0.004
"""

import argparse
import os
import numpy as np


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
    rgb = (np.stack([rec["red"], rec["green"], rec["blue"]], 1).astype(np.uint8)
           if "red" in rec.dtype.names else np.full((n, 3), 200, np.uint8))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clouds = {}
    for cam in ("center", "left", "right"):
        xyz, rgb = read_ply(os.path.join(args.dir, f"cloud_{cam}.ply"))
        clouds[cam] = (xyz, rgb)
        c = xyz.mean(0)
        lo, hi = xyz.min(0), xyz.max(0)
        print(f"{cam:>6}: {len(xyz):>9,} pts | centroid "
              f"[{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}] | "
              f"bbox X[{lo[0]:+.2f},{hi[0]:+.2f}] "
              f"Y[{lo[1]:+.2f},{hi[1]:+.2f}] Z[{lo[2]:+.2f},{hi[2]:+.2f}]")

    cc = clouds["center"][0].mean(0)
    print("\nCentroid distance from center:")
    for cam in ("left", "right"):
        d = np.linalg.norm(clouds[cam][0].mean(0) - cc)
        flag = "  <-- LARGE: likely wrong extrinsic direction" if d > 0.4 else ""
        print(f"  {cam:>6}: {d*1000:6.1f} mm{flag}")

    xyz = np.concatenate([clouds[c][0] for c in clouds])
    rgb = np.concatenate([clouds[c][1] for c in clouds])
    if args.voxel > 0:
        keys = np.floor(xyz / args.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        xyz, rgb = xyz[idx], rgb[idx]

    out = args.out or os.path.join(args.dir, "fused_extrinsic.ply")
    write_ply(out, xyz.astype(np.float32), rgb)
    print(f"\nwrote {out}  ({len(xyz):,} pts, voxel {args.voxel}m)")


if __name__ == "__main__":
    main()