#!/usr/bin/env python3
"""
coverage.py — what each camera subset can see. No DA3, no frames, no scan
beyond gt_scene.json. Run this before spending GPU time.

    python3 coverage.py --calib /home/jetson/Projects/Calibration4/results \
                        --gt gt_scene.json

Prints three things:

1. Lens check. Focal length per camera from fx x pixel pitch. If the three
   disagree by more than ~1%, that is a per-view depth scale difference a
   rig-wide alpha cannot absorb -- and a camera with both an outlying focal
   and a high reprojection RMS is a calibration problem, not a longer lens.

2. Clipped parcels. A parcel whose top face leaves the frame cannot be
   dimensioned from that view. If it is clipped in EVERY view it must be
   excluded from grading entirely; grade_subsets.py --exclude takes the list.

3. Per-subset coverage area. Single-camera subsets use a 1-view rule,
   multi-camera subsets a 2-view rule. This is the part of the study with
   real statistical power -- it does not depend on how many parcels you have.
"""
import argparse
import itertools
import json
import numpy as np

import rigkit as rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--gt", default="gt_scene.json")
    ap.add_argument("--pitch-um", type=float, default=3.45,
                    help="sensor pixel pitch (IMX264/265 2/3in = 3.45)")
    ap.add_argument("--parcel-h", type=float, default=0.30,
                    help="nominal parcel height for the coverage map (m)")
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--extent", type=float, nargs=4,
                    default=(-1.8, 1.8, -1.4, 1.4), metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--margin", type=int, default=0,
                    help="pixels of required clearance from the image edge")
    a = ap.parse_args()

    rig = rk.Rig(a.calib)
    gt = json.load(open(a.gt))
    pitch = a.pitch_um * 1e-3

    print("=== lenses ===")
    fs = []
    for c in rig.cams:
        fx, fy = rig.focal_mm(c, pitch)
        hf, vf = rig.fov_deg(c)
        w, h = rig.size[c]
        K = rig.K[c]
        fs.append(fx)
        print(f"  {c:7s} f={fx:6.3f}/{fy:6.3f} mm  HFOV={hf:5.2f} VFOV={vf:5.2f}  "
              f"pp offset=({K[0,2]-(w-1)/2:+6.1f},{K[1,2]-(h-1)/2:+6.1f}) px")
    spread = 100 * (max(fs) - min(fs)) / np.mean(fs)
    zdeck = -gt["deck_plane"]["d"]
    print(f"  focal spread {spread:.2f}%  ->  {spread/100*zdeck*1000:.0f} mm of "
          f"depth scale at {zdeck:.2f} m")
    if spread > 1.0:
        print("  NOTE: feed each camera ITS OWN intrinsics to DA3. A shared K would")
        print("        inject this as depth bias that the per-subset alpha then hides.")

    print("\n=== geometry ===")
    for c in rig.cams:
        C = rig.centre(c)
        print(f"  {c:7s} at ({C[0]:+.3f},{C[1]:+.3f},{C[2]:+.3f})")
    for x, y in itertools.combinations(rig.cams, 2):
        print(f"  baseline {x}-{y}: {rig.baseline(x,y):.3f} m")

    print("\n=== parcel clipping (top-face corners in frame) ===")
    excluded = []
    for p in gt["parcels"]:
        X = np.array(p["corners_xyz"])
        row, full = [], 0
        for c in rig.cams:
            k = int(rig.visible(X, c, a.margin).sum())
            full += (k == 4)
            row.append(f"{c[0].upper()}:{k}/4")
        tag = ""
        if full == 0:
            excluded.append(p["id"])
            tag = "  <-- clipped everywhere, EXCLUDE"
        print(f"  {p['id']:4s} {p['L_mm']:6.1f}x{p['W_mm']:6.1f}x{p['H_mm']:6.1f} mm  "
              + "  ".join(row) + f"   full views: {full}{tag}")
    if excluded:
        print(f"\n  --exclude {','.join(excluded)}")
    usable = [p for p in gt["parcels"] if p["id"] not in excluded]
    print(f"  {len(usable)} usable parcels")
    if len(usable) < 8:
        print(f"  n={len(usable)} cannot rank seven subsets. Read the deck-plane")
        print("  metrics as the result; read parcel deltas per parcel, never as a sigma.")

    print("\n=== coverage ===")
    x0, x1, y0, y1 = a.extent
    xs = np.arange(x0, x1 + 1e-9, a.grid)
    ys = np.arange(y0, y1 + 1e-9, a.grid)
    X, Y = np.meshgrid(xs, ys)
    pts = rk.from_deck(X.ravel(), Y.ravel(),
                       np.full(X.size, a.parcel_h), gt)
    vis = {c: rig.visible(pts, c, a.margin).reshape(X.shape) for c in rig.cams}
    cell = a.grid ** 2
    print(f"  {'subset':20s} {'area m2':>8s} {'y-extent':>16s} {'max base':>9s}  parcels")
    rows = []
    for s in rk.SUBSETS:
        cs = s.split("+")
        if any(c not in rig.cams for c in cs):
            continue
        m = vis[cs[0]] if len(cs) == 1 else \
            (sum(vis[c].astype(int) for c in cs) >= 2)
        yr = ys[m.any(1)] if m.any() else np.array([np.nan])
        bl = max([rig.baseline(p, q) for p, q in itertools.combinations(cs, 2)]) \
            if len(cs) > 1 else 0.0
        ok = 0
        for p in usable:
            Xc = np.array([p["top_centre_xyz"]])
            k = int(rig.n_views(Xc, cs, a.margin)[0])
            ok += (k >= (1 if len(cs) == 1 else 2))
        print(f"  {s:20s} {m.sum()*cell:8.2f} "
              f"[{yr.min():+.2f},{yr.max():+.2f}]".rjust(17)
              + f" {bl:9.3f}  {ok}/{len(usable)}")
        rows.append(dict(subset=s, area_m2=round(float(m.sum() * cell), 3),
                         max_baseline_m=round(bl, 4),
                         parcels_covered=ok, parcels_total=len(usable)))
    json.dump(rows, open("coverage.json", "w"), indent=1)
    print("\n  wrote coverage.json")
    print("  (1-view rule for single cameras, 2-view rule for subsets; "
          f"parcel top at {a.parcel_h*1000:.0f} mm; distortion applied)")


if __name__ == "__main__":
    main()