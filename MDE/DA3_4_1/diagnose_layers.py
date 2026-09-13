#!/usr/bin/env python3
"""
diagnose_layers.py
==================

Runs on the artefacts da3_multicam_fusion_v2.py already wrote. No re-inference.

Answers four questions:

  Q1  Is the warp inside a single view, or between views?
      -> writes cam_<name>_only.ply per camera, and colour_coded.ply where each
         camera is a distinct solid colour. Open colour_coded.ply: if the
         separate sheets are DIFFERENT colours the problem is registration
         (extrinsics / scale). If a SINGLE colour contains the sheets and the
         curvature, the problem is inside the depth map.

  Q2  Is DA3's depth Z-along-optical-axis or Euclidean distance along the ray?
      -> rebuilds under both interpretations and reports floor-plane flatness
         for each. Ray depth misread as Z bows every view into a dome, roughly
         12% error at the frame edge for a 55 deg FOV.

  Q3  Is the extrinsics convention right?
      -> rebuilds under w2c and c2w and reports the spread of the per-camera
         floor planes. The correct convention makes all four cameras agree on
         one floor; the wrong one scatters them.

  Q4  How large is the residual disagreement, quantitatively?
      -> per-camera floor plane normal and offset in the reference frame,
         pairwise cross-view depth residual, and plane-fit RMS.

Usage
-----
    python diagnose_layers.py --run ~/Projects/MDE/DA3_4_1/runs/20260804_092615_018_v2_pad \
                              --mode prior \
                              --calib-dir /home/jetson/Projects/Calibration_4_1/results
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# --------------------------------------------------------------------------- #

def Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(pts.shape[0])
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n").encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.empty(n, dtype=dt)
    a["x"], a["y"], a["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    c = np.clip(cols, 0, 255).astype(np.uint8)
    a["red"], a["green"], a["blue"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(a.tobytes())


def ransac_plane(pts: np.ndarray, thresh: float = 0.02, iters: int = 400,
                 seed: int = 0) -> Optional[Dict]:
    """Largest plane by inlier count. Returns normal, offset, rms, inlier count."""
    if pts.shape[0] < 100:
        return None
    rng = np.random.default_rng(seed)
    n_pts = pts.shape[0]
    sub = pts if n_pts <= 200000 else pts[rng.choice(n_pts, 200000, replace=False)]
    best = None
    for _ in range(iters):
        idx = rng.choice(sub.shape[0], 3, replace=False)
        p0, p1, p2 = sub[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -nrm @ p0
        dist = np.abs(sub @ nrm + d)
        n_in = int((dist < thresh).sum())
        if best is None or n_in > best[0]:
            best = (n_in, nrm, d)
    if best is None:
        return None
    _, nrm, d = best
    dist = np.abs(sub @ nrm + d)
    inl = sub[dist < thresh]
    if inl.shape[0] < 50:
        return None
    # least-squares refit on inliers
    cen = inl.mean(0)
    _, _, Vt = np.linalg.svd(inl - cen, full_matrices=False)
    nrm = Vt[-1] / np.linalg.norm(Vt[-1])
    if nrm[2] < 0:
        nrm = -nrm
    d = -nrm @ cen
    rms = float(np.sqrt((((inl @ nrm + d) ** 2)).mean()))
    return {"normal": nrm.tolist(), "offset_m": float(d),
            "rms_m": rms, "n_inliers": int(inl.shape[0]),
            "inlier_fraction": float(inl.shape[0] / sub.shape[0])}


def backproject(depth: np.ndarray, K: np.ndarray, ray: bool) -> np.ndarray:
    h, w = depth.shape
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    d = depth.astype(np.float64)
    if ray:
        d = d / np.sqrt(1.0 + xn ** 2 + yn ** 2)   # ray length -> Z
    return np.stack([xn * d, yn * d, d], axis=-1)


# --------------------------------------------------------------------------- #

CAM_COLOURS = {
    "left":   (230, 60, 60),
    "center": (60, 200, 90),
    "right":  (70, 120, 240),
    "top":    (240, 190, 50),
}


class View:
    def __init__(self, name: str, depth: np.ndarray, K: np.ndarray, rot_k: int,
                 R: np.ndarray, t: np.ndarray):
        self.name = name
        self.depth = depth
        self.K = K
        self.rot_k = rot_k
        self.R = R
        self.t = t

    def cam_points(self, ray: bool) -> np.ndarray:
        pts_rot = backproject(self.depth, self.K, ray).reshape(-1, 3)
        return pts_rot @ Rz(90.0 * self.rot_k).T

    def to_ref(self, pts_cam: np.ndarray, convention: str) -> np.ndarray:
        if convention == "w2c":
            return (pts_cam - self.t[None, :]) @ self.R
        # treat stored R,t as camera-to-world
        return pts_cam @ self.R.T + self.t[None, :]

    def from_ref(self, pts_ref: np.ndarray, convention: str) -> np.ndarray:
        if convention == "w2c":
            return pts_ref @ self.R.T + self.t[None, :]
        return (pts_ref - self.t[None, :]) @ self.R


def load_views(run: Path, mode: str, calib_dir: Path) -> List[View]:
    diag = json.loads((run / mode / "diagnostics.json").read_text())
    ext_all = json.loads((calib_dir / "extrinsics.json").read_text())
    views = []
    for entry in diag["per_camera"]:
        name = entry["camera"]
        dpath = run / mode / f"depth_{name}.npy"
        kpath = run / mode / f"K_{name}.npy"
        if not dpath.exists():
            raise SystemExit(f"missing {dpath}; rerun v2 with --save-depth")
        depth = np.load(dpath).astype(np.float64)
        if kpath.exists():
            K = np.load(kpath).astype(np.float64)
        else:
            K = np.array([[entry["K_used_fx"], 0, entry["K_used_cx"]],
                          [0, entry["K_used_fy"], entry["K_used_cy"]],
                          [0, 0, 1]], dtype=np.float64)
        R = np.asarray(ext_all[name]["R"], dtype=np.float64)
        t = np.asarray(ext_all[name]["t"], dtype=np.float64)
        views.append(View(name, depth, K, int(entry["rot_k"]), R, t))
    return views


# --------------------------------------------------------------------------- #

def plane_spread(views: List[View], ray: bool, convention: str,
                 zmin: float, zmax: float) -> Dict:
    """Per-camera floor plane in the reference frame, and how much they disagree."""
    planes = []
    for v in views:
        pc = v.cam_points(ray)
        z = pc[:, 2]
        keep = np.isfinite(pc).all(1) & (z > zmin) & (z < zmax)
        pr = v.to_ref(pc[keep], convention)
        pl = ransac_plane(pr)
        if pl is not None:
            pl["camera"] = v.name
            planes.append(pl)
    out: Dict = {"planes": planes}
    if len(planes) >= 2:
        normals = np.array([p["normal"] for p in planes])
        offsets = np.array([p["offset_m"] for p in planes])
        angs = []
        for a, b in itertools.combinations(range(len(planes)), 2):
            c = float(np.clip(abs(normals[a] @ normals[b]), -1, 1))
            angs.append(float(np.degrees(np.arccos(c))))
        out["max_pairwise_normal_angle_deg"] = float(max(angs))
        out["offset_spread_m"] = float(offsets.max() - offsets.min())
        out["mean_plane_rms_m"] = float(np.mean([p["rms_m"] for p in planes]))
    return out


def cross_view_residual(views: List[View], ray: bool, convention: str,
                        zmin: float, zmax: float, stride: int = 4) -> List[Dict]:
    """Project view A's points into view B and compare against B's own depth."""
    res = []
    for a, b in itertools.combinations(range(len(views)), 2):
        va, vb = views[a], views[b]
        pc = va.cam_points(ray)[::stride]
        z = pc[:, 2]
        keep = np.isfinite(pc).all(1) & (z > zmin) & (z < zmax)
        pr = va.to_ref(pc[keep], convention)
        pb = vb.from_ref(pr, convention)
        pb_rot = pb @ Rz(90.0 * vb.rot_k)          # true cam -> rotated frame
        zb = pb_rot[:, 2]
        front = zb > 1e-6
        pb_rot, zb = pb_rot[front], zb[front]
        u = (pb_rot[:, 0] / zb) * vb.K[0, 0] + vb.K[0, 2]
        v = (pb_rot[:, 1] / zb) * vb.K[1, 1] + vb.K[1, 2]
        h, w = vb.depth.shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if inb.sum() < 100:
            res.append({"pair": [va.name, vb.name], "n_overlap": int(inb.sum()),
                        "note": "insufficient overlap"})
            continue
        d_b = vb.depth[vi[inb], ui[inb]]
        if ray:
            xn = (ui[inb] - vb.K[0, 2]) / vb.K[0, 0]
            yn = (vi[inb] - vb.K[1, 2]) / vb.K[1, 1]
            d_b = d_b / np.sqrt(1.0 + xn ** 2 + yn ** 2)
        diff = zb[inb] - d_b
        good = np.isfinite(diff)
        res.append({
            "pair": [va.name, vb.name],
            "n_overlap": int(good.sum()),
            "median_abs_residual_m": float(np.median(np.abs(diff[good]))),
            "p90_abs_residual_m": float(np.percentile(np.abs(diff[good]), 90)),
            "median_signed_residual_m": float(np.median(diff[good])),
        })
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Isolate per-view warp from misregistration.")
    ap.add_argument("--run", type=Path, required=True, help="run directory from v2")
    ap.add_argument("--mode", default="prior", choices=["prior", "noprior"])
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_1/results"))
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=12.0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (args.run / args.mode / "diagnose")
    out.mkdir(parents=True, exist_ok=True)

    views = load_views(args.run, args.mode, args.calib_dir)
    print(f"loaded {len(views)} views from {args.run / args.mode}")
    for v in views:
        print(f"  {v.name:7s} depth {v.depth.shape} rot_k={v.rot_k} "
              f"fx={v.K[0,0]:.2f} cx={v.K[0,2]:.2f} "
              f"median_depth={np.nanmedian(v.depth):.3f} m")

    report: Dict = {"run": str(args.run), "mode": args.mode, "tests": {}}

    # ---- Q2 / Q3 sweep ---------------------------------------------------- #
    print("\n" + "=" * 70)
    print("convention sweep: floor-plane agreement across the four cameras")
    print("=" * 70)
    print(f"{'depth':>6} {'extr':>5} | {'max normal ang':>15} {'offset spread':>14} "
          f"{'mean plane rms':>15}")
    best = None
    for ray in (False, True):
        for conv in ("w2c", "c2w"):
            r = plane_spread(views, ray, conv, args.min_depth, args.max_depth)
            key = f"{'ray' if ray else 'z'}_{conv}"
            report["tests"][key] = r
            if "max_pairwise_normal_angle_deg" in r:
                score = r["offset_spread_m"] + r["mean_plane_rms_m"]
                print(f"{'ray' if ray else 'z':>6} {conv:>5} | "
                      f"{r['max_pairwise_normal_angle_deg']:12.3f} deg "
                      f"{r['offset_spread_m']:11.4f} m "
                      f"{r['mean_plane_rms_m']:12.4f} m")
                if best is None or score < best[0]:
                    best = (score, ray, conv, key)
            else:
                print(f"{'ray' if ray else 'z':>6} {conv:>5} |  no plane found")

    if best is None:
        print("\nno usable plane in any configuration; check --min-depth/--max-depth")
        (out / "diagnose.json").write_text(json.dumps(report, indent=2))
        return 1

    _, ray_b, conv_b, key_b = best
    print(f"\nbest configuration: depth={'ray' if ray_b else 'z'}, extrinsics={conv_b}")
    report["best"] = {"depth_interpretation": "ray" if ray_b else "z",
                      "extrinsics_convention": conv_b}

    # ---- Q4 cross-view residual ------------------------------------------ #
    print("\n" + "=" * 70)
    print("cross-view depth residual, best configuration")
    print("=" * 70)
    cv = cross_view_residual(views, ray_b, conv_b, args.min_depth, args.max_depth,
                             args.stride)
    report["cross_view_residual"] = cv
    for r in cv:
        if "median_abs_residual_m" in r:
            print(f"  {r['pair'][0]:>7s} <-> {r['pair'][1]:<7s}  "
                  f"median {r['median_abs_residual_m']*1000:8.1f} mm   "
                  f"p90 {r['p90_abs_residual_m']*1000:8.1f} mm   "
                  f"n={r['n_overlap']}")
        else:
            print(f"  {r['pair'][0]:>7s} <-> {r['pair'][1]:<7s}  {r['note']}")

    # ---- Q1 colour-coded clouds ------------------------------------------ #
    all_pts, all_cols = [], []
    for v in views:
        pc = v.cam_points(ray_b)
        z = pc[:, 2]
        keep = np.isfinite(pc).all(1) & (z > args.min_depth) & (z < args.max_depth)
        pr = v.to_ref(pc[keep], conv_b)
        col = np.tile(np.array(CAM_COLOURS.get(v.name, (200, 200, 200)),
                               dtype=np.uint8), (pr.shape[0], 1))
        write_ply(out / f"cam_{v.name}_only.ply", pr, col)
        all_pts.append(pr)
        all_cols.append(col)
    write_ply(out / "colour_coded.ply",
              np.concatenate(all_pts, 0), np.concatenate(all_cols, 0))

    (out / "diagnose.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print("WHAT TO DO WITH THIS")
    print("=" * 70)
    print(f"  wrote {out}/colour_coded.ply  (left=red, center=green, right=blue, top=yellow)")
    print("  separate sheets in DIFFERENT colours -> registration problem")
    print("  sheets and curvature within ONE colour -> per-view depth problem")
    print(f"  also wrote cam_<name>_only.ply for isolated inspection")
    print(f"  full numbers in {out}/diagnose.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())