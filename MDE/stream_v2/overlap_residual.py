#!/usr/bin/env python3
"""
overlap_residual.py  (v2)

Inter-camera depth disagreement measured only where two cameras have a sample on
the same surface.

What changed from v1
--------------------
v1 reported a figure for any camera with at least 100 shared points, and let the
worst camera set the headline. On the affine-corrected run that meant top's 100
shared points (with 14.5 % gated out) produced an 87.6 mm p90 and became the
reported worst case, while right's 991 points gave 4.6 mm. A p90 from 100 points
is a couple of dozen samples in the tail; it is not a measurement.

v2 adds a reliability floor: cameras below --min-shared are reported but marked
unreliable and excluded from the headline, and the gated-out fraction is
promoted to a first-class warning rather than a footnote. It also reports the
count directly so an under-sampled camera can never look like a clean result.

Interpretation
--------------
  bias        median signed difference. A consistent sign across cameras is an
              unabsorbed offset degree of freedom -- see depth_affine.py.
  |d| median  the honest centre of the disagreement.
  spread      p90 and RMS. Near-zero bias with a large spread is NOT an offset;
              it is noise, and if depth_affine reported a large a0 the
              correction is amplifying it (dz'/dz = A at the anchor).

Example
-------
  python overlap_residual.py \
      --capture runs/live_20260817_102732/capture_00002 \
      --percam  runs/live_20260817_102732/capture_00002/percam_affine \
      --report  runs/live_20260817_102732/capture_00002/overlap_residual_affine.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d not found. Activate the da3 environment.")


def to_4x4(M):
    M = np.asarray(M, dtype=np.float64)
    if M.size == 12:
        T = np.eye(4)
        T[:3, :4] = M.reshape(3, 4)
        return T
    if M.size == 16:
        return M.reshape(4, 4)
    raise ValueError(f"extrinsic has {M.size} elements, expected 12 or 16")


def project(P_cam, K, shape):
    h, w = shape
    z = P_cam[:, 2]
    ok = z > 1e-6
    u = np.full(len(P_cam), -1.0)
    v = np.full(len(P_cam), -1.0)
    u[ok] = K[0, 0] * P_cam[ok, 0] / z[ok] + K[0, 2]
    v[ok] = K[1, 1] * P_cam[ok, 1] / z[ok] + K[1, 2]
    ui = np.floor(u).astype(int)
    vi = np.floor(v).astype(int)
    inside = ok & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return ui, vi, z, inside


def zbuffer(P_cam, K, shape):
    h, w = shape
    buf = np.full((h, w), np.inf)
    ui, vi, z, inside = project(P_cam, K, shape)
    np.minimum.at(buf, (vi[inside], ui[inside]), z[inside])
    return buf


def stats(a):
    if len(a) == 0:
        return None
    a = np.asarray(a, dtype=float)
    return {"n": int(len(a)),
            "median_mm": float(np.median(a) * 1000.0),
            "abs_median_mm": float(np.median(np.abs(a)) * 1000.0),
            "abs_p90_mm": float(np.percentile(np.abs(a), 90) * 1000.0),
            "rms_mm": float(np.sqrt((a ** 2).mean()) * 1000.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--percam", required=True)
    ap.add_argument("--cameras", default="left,center,right,top")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--extrinsic-convention", default="cam_from_world",
                    choices=["world_from_cam", "cam_from_world"])
    ap.add_argument("--gate", type=float, default=0.15,
                    help="max |depth difference| still counted as the same "
                         "surface (m); loose on purpose")
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--min-shared", type=int, default=500,
                    help="below this, a camera's figures are marked unreliable "
                         "and excluded from the headline")
    ap.add_argument("--max-gated-fraction", type=float, default=0.10,
                    help="above this, occlusion dominates and the figures are "
                         "marked unreliable")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cap = Path(args.capture)
    pc = Path(args.percam)
    meta = json.loads((pc / "deck_plane.json").read_text())
    n_w = np.asarray(meta["normal"], float)
    d_w = float(meta["d"])
    band = meta.get("above_deck_band_m", [0.03, 0.40])
    ref = args.reference or meta.get("reference")
    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    if meta.get("all_cameras_corrected") is False:
        print("WARNING: the dump reports that not every camera was corrected.\n"
              "         An uncorrected deck falls inside the above-deck band, so\n"
              "         these figures compare decks, not parcels.\n")

    Es, Ks, shapes, clouds = {}, {}, {}, {}
    for c in cams:
        Es[c] = to_4x4(np.load(cap / f"E_{c}.npy"))
        Ks[c] = np.load(cap / f"K_{c}.npy").astype(np.float64).reshape(3, 3)
        shapes[c] = np.squeeze(np.load(cap / f"depth_{c}.npy")).shape
        clouds[c] = np.asarray(
            o3d.io.read_point_cloud(str(pc / f"cloud_{c}.ply")).points)

    def T_cw(c):
        return Es[c] if args.extrinsic_convention == "cam_from_world" \
            else np.linalg.inv(Es[c])

    def above(P):
        s = P @ n_w + d_w
        return P[(s > band[0]) & (s < band[1])]

    def to_cam(P, c):
        T = T_cw(c)
        return (T[:3, :3] @ P.T).T + T[:3, 3]

    ref_world = above(clouds[ref])
    report = {"reference": ref, "gate_m": args.gate, "above_deck_band_m": band,
              "min_shared": args.min_shared,
              "reference_above_deck_points": int(len(ref_world)),
              "cameras": {}}
    print(f"reference {ref}: {len(ref_world)} above-deck points "
          f"(reliability floor {args.min_shared} shared)\n")

    r = args.dilate
    for c in cams:
        if c == ref:
            continue
        shape = shapes[c]
        buf = zbuffer(to_cam(above(clouds[c]), c), Ks[c], shape)
        ref_cam = to_cam(ref_world, c)
        ui, vi, z, inside = project(ref_cam, Ks[c], shape)
        idx = np.flatnonzero(inside)

        best = np.full(len(idx), np.inf)
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                uu = np.clip(ui[idx] + du, 0, shape[1] - 1)
                vv = np.clip(vi[idx] + dv, 0, shape[0] - 1)
                candv = buf[vv, uu]
                take = np.abs(candv - z[idx]) < np.abs(best - z[idx])
                best[take] = candv[take]

        have = np.isfinite(best)
        dz = z[idx][have] - best[have]
        gated = np.abs(dz) < args.gate
        n_shared = int(gated.sum())
        gated_out = float(1.0 - n_shared / max(have.sum(), 1))

        reasons = []
        if n_shared < args.min_shared:
            reasons.append(f"only {n_shared} shared points")
        if gated_out > args.max_gated_fraction:
            reasons.append(f"{gated_out * 100:.1f} % gated out")

        info = {"ref_points_in_frame": int(inside.sum()),
                "in_frame_fraction": float(inside.sum() / max(len(ref_world), 1)),
                "with_buffer_sample": int(have.sum()),
                "mutually_visible": n_shared,
                "gated_out_fraction": gated_out,
                "reliable": len(reasons) == 0,
                "unreliable_because": reasons,
                "depth_difference": stats(dz[gated])}
        report["cameras"][c] = info

        s = info["depth_difference"]
        tag = "" if info["reliable"] else "   UNRELIABLE: " + "; ".join(reasons)
        if s:
            print(f"{c:<7} n={n_shared:6d} ({100 * info['in_frame_fraction']:.1f} % "
                  f"in frame)  bias {s['median_mm']:+7.2f} mm  |d| median "
                  f"{s['abs_median_mm']:6.2f} mm  p90 {s['abs_p90_mm']:7.2f} mm  "
                  f"RMS {s['rms_mm']:7.2f} mm{tag}")
        else:
            print(f"{c:<7} no mutually visible points within the gate")

    good = {c: v for c, v in report["cameras"].items()
            if v["reliable"] and v["depth_difference"]}
    bad = [c for c, v in report["cameras"].items() if not v["reliable"]]
    report["reliable_cameras"] = sorted(good)
    report["unreliable_cameras"] = sorted(bad)

    if good:
        worst = max(v["depth_difference"]["abs_p90_mm"] for v in good.values())
        report["worst_abs_p90_mm_reliable"] = worst
        print(f"\nworst |depth difference| p90 over reliable cameras "
              f"({', '.join(sorted(good))}): {worst:.2f} mm")
        if worst < 10.0:
            print("Under 10 mm. The sheets are merged. Remaining sparsity is\n"
                  "coverage, which needs views or baseline, not another correction.")
        elif worst < 30.0:
            print("10-30 mm. No longer dominated by scale, tilt or offset. A small\n"
                  "bias with a wide spread is noise, not a systematic error - check\n"
                  "a0 in the depth_affine report, since the correction amplifies\n"
                  "depth noise by that factor.")
        else:
            print("Above 30 mm on shared surface between reliable cameras. Check\n"
                  "the bias signs and the per-plane RMS from depth_affine before\n"
                  "trusting any box height from this cloud.")
    else:
        print("\nNo camera met the reliability floor. Nothing here is quotable -\n"
              "the shared field of view is too small in this capture.")
    if bad:
        print(f"excluded from the headline: {', '.join(sorted(bad))}")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()