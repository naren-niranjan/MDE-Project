#!/usr/bin/env python3
"""
Manual merge of the three extrinsic-placed clouds.

Two ways to correct each side cloud onto center, chosen per-camera in a small
JSON config:

  1. "correspondences" -- pick >=3 (ideally 4-6) matching points on the side
     cloud and the SAME features on the center cloud in your viewer, paste the
     XYZ coordinates. The script solves the best-fit similarity transform
     (scale + rotation + translation) and reports the fit residual.

  2. "manual" -- type in a nudge directly: tx/ty/tz in metres, rx/ry/rz in
     degrees (applied in the center frame, XYZ order), and a scale. Re-run,
     look at the result, adjust the numbers, repeat.

The side clouds (cloud_left.ply / cloud_right.ply) are already placed in the
center frame by the MoGe script, so these transforms are *corrections* on top
of that placement. center is always the fixed reference.

Quick start:
    python manual_merge.py --dir DIR --make-template      # writes merge_config.json
    # edit merge_config.json (see modes below), then:
    python manual_merge.py --dir DIR --config merge_config.json --voxel 0.004
"""

import argparse
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation


# ---- PLY io (with color) -------------------------------------------------- #
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


# ---- transforms ----------------------------------------------------------- #
def umeyama(src, dst, with_scale=True):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
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


def manual_srt(m):
    R = Rotation.from_euler("xyz",
                            [m.get("rx", 0), m.get("ry", 0), m.get("rz", 0)],
                            degrees=True).as_matrix()
    t = np.array([m.get("tx", 0), m.get("ty", 0), m.get("tz", 0)], float)
    s = float(m.get("scale", 1.0))
    return s, R, t


def apply(xyz, s, R, t):
    return (s * (R @ xyz.T)).T + t


TEMPLATE = {
    "left": {
        "mode": "correspondences",
        "_comment": "pick the SAME features on both clouds; >=3 pairs, 4-6 better",
        "src": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "dst": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "rigid": False
    },
    "right": {
        "mode": "manual",
        "_comment": "nudge in metres / degrees until it lines up, then re-run",
        "tx": 0.0, "ty": 0.0, "tz": 0.0,
        "rx": 0.0, "ry": 0.0, "rz": 0.0, "scale": 1.0
    }
}


def solve_side(cam, cfg):
    mode = cfg.get("mode", "manual")
    if mode == "correspondences":
        src, dst = np.asarray(cfg["src"], float), np.asarray(cfg["dst"], float)
        if len(src) < 3 or len(src) != len(dst):
            raise SystemExit(f"[{cam}] need >=3 matched src/dst pairs of equal length")
        s, R, t = umeyama(src, dst, with_scale=not cfg.get("rigid", False))
        res = np.linalg.norm(apply(src, s, R, t) - dst, axis=1)
        ang = Rotation.from_matrix(R).magnitude() * 180 / np.pi
        print(f"[{cam}] corr fit: scale={s:.4f}  rot={ang:.2f}deg  "
              f"|t|={np.linalg.norm(t)*1000:.1f}mm  "
              f"pair-residual RMS={np.sqrt((res**2).mean())*1000:.1f}mm ({len(src)} pairs)")
        return s, R, t
    else:
        s, R, t = manual_srt(cfg)
        print(f"[{cam}] manual: scale={s:.4f}  "
              f"t=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]m  "
              f"rot=[{cfg.get('rx',0)} {cfg.get('ry',0)} {cfg.get('rz',0)}]deg")
        return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.make_template:
        p = os.path.join(args.dir, "merge_config.json")
        with open(p, "w") as f:
            json.dump(TEMPLATE, f, indent=2)
        print(f"wrote template -> {p}\nEdit it, then rerun with --config {p}")
        return

    if not args.config:
        raise SystemExit("pass --config merge_config.json (or --make-template first)")
    with open(args.config) as f:
        cfg = json.load(f)

    C_xyz, C_rgb = read_ply(os.path.join(args.dir, "cloud_center.ply"))
    all_xyz, all_rgb = [C_xyz], [C_rgb]
    print(f"center: {len(C_xyz):,} pts (fixed)")

    for cam in ("left", "right"):
        xyz, rgb = read_ply(os.path.join(args.dir, f"cloud_{cam}.ply"))
        if cam in cfg:
            s, R, t = solve_side(cam, cfg[cam])
            xyz = apply(xyz, s, R, t)
        else:
            print(f"[{cam}] no config entry -> left as-placed")
        all_xyz.append(xyz); all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    if args.voxel > 0:
        keys = np.floor(xyz / args.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        xyz, rgb = xyz[idx], rgb[idx]

    out = args.out or os.path.join(args.dir, "fused_manual.ply")
    write_ply(out, xyz.astype(np.float32), rgb)
    print(f"\nwrote {out}  ({len(xyz):,} pts, voxel {args.voxel}m)")


if __name__ == "__main__":
    main()