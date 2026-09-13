#!/usr/bin/env python3
"""
reproject_check.py

Settles whether the in-plane offsets reported by inplane_offset.py are real
extrinsic error or an artefact of that measurement.

Why this is needed
------------------
inplane_offset.py estimates (dx, dy, yaw) by phase-correlating sparse binary
occupancy grids. That method is fragile exactly where it was applied: 51-61 %
of above-deck points had no counterpart at all, the point counts differ by 8x
between cameras (center 20171, top 2507), and phase correlation whitens
magnitude, which makes a sparse, partially-overlapping pair prone to locking
onto a spurious peak. A reported 1512 mm shift on a 3.58 x 4.45 m grid is well
inside the range where that can happen. So treat those numbers as "large",
not as calibrated values, until this check agrees with them.

What this does instead
---------------------
Two independent tests, neither relying on correlation:

  1. Pure extrinsics, no depth at all. Camera centres are recovered from E and
     all pairwise distances printed. If those do not match the physically
     measured baselines, the extrinsics are wrong and nothing downstream can
     be trusted. This test cannot be fooled by DA3, by the depth field, or by
     coverage.

  2. Cross-projection overlay. The reference camera's corrected above-deck
     points are projected into every other camera's image and drawn in red;
     that camera's own above-deck points are drawn in green. If red lands on
     the same parcels as green, the cameras are registered and the phase
     correlation lied. If red lands on different parcels, the extrinsics are
     wrong and the pixel displacement shows by how much.

Input
-----
The stream_v2 capture directory, plus the per-camera clouds written by
layer_collapse.py --dump-per-camera.

Example
-------
  python reproject_check.py \
      --capture runs/live_20260817_102732/capture_00002 \
      --percam  runs/live_20260817_102732/capture_00002/percam \
      --baselines left=0.292,right=0.861,top=0.567 \
      --outdir   runs/live_20260817_102732/capture_00002/reproject
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d not found. Activate the da3 environment.")
try:
    import cv2
except ImportError:
    raise SystemExit("opencv not found. Activate the da3 environment.")


def to_4x4(M):
    M = np.asarray(M, dtype=np.float64)
    if M.size == 12:
        T = np.eye(4)
        T[:3, :4] = M.reshape(3, 4)
        return T
    if M.size == 16:
        return M.reshape(4, 4)
    raise ValueError(f"extrinsic has {M.size} elements, expected 12 or 16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--percam", required=True,
                    help="layer_collapse.py --dump-per-camera output")
    ap.add_argument("--cameras", default="left,center,right,top")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--extrinsic-convention", default="cam_from_world",
                    choices=["world_from_cam", "cam_from_world"],
                    help="must match what layer_collapse reported")
    ap.add_argument("--baselines", default=None,
                    help="physically measured distances from the reference "
                         "camera, e.g. left=0.292,right=0.861,top=0.567")
    ap.add_argument("--image", default="proc", choices=["proc", "raw"],
                    help="overlay onto proc_<cam>.png or <cam>.png")
    ap.add_argument("--dot", type=int, default=1, help="marker radius in px")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cap = Path(args.capture)
    pc = Path(args.percam)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((pc / "deck_plane.json").read_text())
    n_w = np.asarray(meta["normal"], float)
    d_w = float(meta["d"])
    band = meta.get("above_deck_band_m", [0.03, 0.40])
    ref = args.reference or meta.get("reference")
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    Es, Ks, shapes = {}, {}, {}
    for c in cams:
        Es[c] = to_4x4(np.load(cap / f"E_{c}.npy"))
        Ks[c] = np.load(cap / f"K_{c}.npy").astype(np.float64).reshape(3, 3)
        shapes[c] = np.squeeze(np.load(cap / f"depth_{c}.npy")).shape

    def T_cam_from_world(c):
        return Es[c] if args.extrinsic_convention == "cam_from_world" \
            else np.linalg.inv(Es[c])

    report = {"reference": ref,
              "extrinsic_convention": args.extrinsic_convention,
              "camera_centres": {}, "pairwise_distances_m": {},
              "baseline_check": {}}

    # ---- test 1: camera centres, no depth involved ------------------------
    print("camera centres from E (world frame)")
    centres = {}
    for c in cams:
        T = np.linalg.inv(T_cam_from_world(c))
        centres[c] = T[:3, 3]
        report["camera_centres"][c] = centres[c].tolist()
        print(f"  {c:<7} {np.round(centres[c], 4)}")

    print("\npairwise centre distances (m)")
    for a, b in itertools.combinations(cams, 2):
        dist = float(np.linalg.norm(centres[a] - centres[b]))
        report["pairwise_distances_m"][f"{a}-{b}"] = dist
        print(f"  {a:<7} {b:<7} {dist:.4f}")

    if args.baselines:
        print("\nagainst measured baselines from the reference")
        worst = 0.0
        for tok in args.baselines.split(","):
            k, v = tok.split("=")
            k, v = k.strip(), float(v)
            got = float(np.linalg.norm(centres[k] - centres[ref]))
            err = (got - v) * 1000.0
            worst = max(worst, abs(err))
            report["baseline_check"][k] = {"measured_m": v, "from_E_m": got,
                                          "error_mm": err}
            print(f"  {k:<7} measured {v:.4f}  from E {got:.4f}  "
                  f"error {err:+.1f} mm")
        report["baseline_worst_error_mm"] = worst
        print()
        if worst > 50.0:
            print(f"Worst baseline error {worst:.0f} mm. The extrinsics do not "
                  f"describe the physical\nrig. This is upstream of every depth "
                  f"question - fix Calibration_4_5 first.")
        else:
            print(f"Worst baseline error {worst:.0f} mm. The rig geometry in E is "
                  f"consistent with\nthe tape, so a large in-plane offset is more "
                  f"likely a measurement artefact\nthan a wrong extrinsic. Look at "
                  f"the overlays.")

    # ---- test 2: cross-projection overlays --------------------------------
    def above_deck(cam):
        P = np.asarray(o3d.io.read_point_cloud(str(pc / f"cloud_{cam}.ply")).points)
        if len(P) == 0:
            return P
        s = P @ n_w + d_w
        return P[(s > band[0]) & (s < band[1])]

    ref_pts = above_deck(ref)
    print(f"\nreference above-deck points: {len(ref_pts)}")

    def project(P, cam):
        T = T_cam_from_world(cam)
        X = (T[:3, :3] @ P.T).T + T[:3, 3]
        z = X[:, 2]
        ok = z > 1e-6
        X, z = X[ok], z[ok]
        K = Ks[cam]
        u = K[0, 0] * X[:, 0] / z + K[0, 2]
        v = K[1, 1] * X[:, 1] / z + K[1, 2]
        h, w = shapes[cam]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return np.stack([u[inside], v[inside]], axis=1), int(inside.sum()), int(len(P))

    report["overlays"] = {}
    for c in cams:
        if c == ref:
            continue
        h, w = shapes[c]
        stem = f"{args.image}_{c}.png" if args.image == "proc" else f"{c}.png"
        img_path = cap / stem
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            print(f"  {c}: {stem} missing, drawing on black")
            img = np.zeros((h, w, 3), np.uint8)

        own_uv, own_in, own_tot = project(above_deck(c), c)
        ref_uv, ref_in, ref_tot = project(ref_pts, c)

        for uv, colour in ((own_uv, (0, 255, 0)), (ref_uv, (0, 0, 255))):
            for u, v in uv.astype(int):
                cv2.circle(img, (u, v), args.dot, colour, -1)

        cv2.putText(img, f"green = {c} own   red = {ref} reprojected",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        dst = out / f"overlay_{c}.png"
        cv2.imwrite(str(dst), img)

        cen_own = own_uv.mean(axis=0) if len(own_uv) else None
        cen_ref = ref_uv.mean(axis=0) if len(ref_uv) else None
        gap = float(np.linalg.norm(cen_own - cen_ref)) if cen_own is not None \
            and cen_ref is not None else None
        report["overlays"][c] = {
            "own_points_in_frame": own_in, "own_points_total": own_tot,
            "ref_points_in_frame": ref_in, "ref_points_total": ref_tot,
            "ref_in_frame_fraction": ref_in / ref_tot if ref_tot else 0.0,
            "centroid_gap_px": gap, "file": str(dst)}
        print(f"  {c:<7} ref points landing in frame {ref_in}/{ref_tot} "
              f"({100.0 * ref_in / max(ref_tot, 1):.1f} %)  "
              f"centroid gap {gap:.1f} px" if gap is not None else
              f"  {c:<7} ref points landing in frame {ref_in}/{ref_tot}")

    print(f"\noverlays -> {out}")
    print("If red sits on the same parcels as green, the extrinsics are fine and\n"
          "the in-plane numbers were an artefact. If red sits on different\n"
          "parcels, the extrinsics are wrong by the displacement you can see.\n"
          "A very low ref-in-frame fraction means the two cameras barely share a\n"
          "view, which alone explains the saturation without any error at all.")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()