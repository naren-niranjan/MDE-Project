#!/usr/bin/env python3
"""
grade_boxes.py

Grade boxes.json against the reference scan: the parcel records the robot is
actually sent, not the point cloud they were derived from.

Why this is separate from gt_compare.py
---------------------------------------
gt_compare.py measures the CLOUD. It answers whether the depth is right. It
says nothing about whether the segmentation then read the right rectangle off
that depth, and those fail differently: a parcel can sit in a perfectly
reconstructed cloud and still be reported at half its width because a
threshold cut the top face, or be reported correctly in a cloud that is 20 mm
out. Two stages, two measurements, and only the second one is the deliverable.

It also separates the two errors that matter differently to a pick. A DIMENSION
error means the gripper is told the wrong size. A CENTRE error means it is sent
to the wrong place, and at a 40 mm suction radius a 30 mm centre error is the
difference between landing on the parcel and landing over an edge. They are
reported apart.

Parcels are matched to the scan by plan-view position, not by identity, because
box ids are assigned per capture and are not stable across them.

    python grade_boxes.py --scan scans/cell.ply --transform scan_to_rig.json \\
        --deck-rig 3.19 \\
        --boxes runs/seg_.../frame_00000/segmentation/boxes.json \\
        --boxes runs/seg_.../frame_00001/segmentation/boxes.json \\
        --boxes runs/seg_.../frame_00002/segmentation/boxes.json

Keep this file beside scan_frame.py and gt_compare.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import scan_frame as sf
import gt_compare as gc


def scan_parcels(scan_rig, deck_rig, cell=0.005):
    """The reference parcel table, in the scan's own conveyor frame."""
    P = sf.voxel(scan_rig, 0.006)
    n, d = sf.deck_plane(P, deck_rig, label="scan")
    bf = sf.belt_frame(P, n, d)
    L = sf.to_frame(P, bf)
    ext = (float(L[:, 0].min()), float(L[:, 0].max()),
           float(L[:, 1].min()), float(L[:, 1].max()))
    g = sf.Grid(ext, cell)
    mF, _ = g.surface(L, 2)             # the scan graded against itself
    rows = gc.parcel_table(mF, mF, g)
    return rows, bf


def load_boxes(path, bf):
    """Parcel records, with their centres taken into the scan's frame."""
    rec = json.loads(Path(path).read_text())
    out = []
    for b in rec.get("boxes", []):
        c = sf.to_frame(np.asarray(b["position_m"])[None, :], bf)[0]
        dm = b["dimensions_m"]
        vc = b.get("view_consensus") or {}
        out.append({
            "u": float(c[0]), "v": float(c[1]), "h_centre": float(c[2]),
            "l": dm["length"] * 1e3, "w": dm["width"] * 1e3,
            "h": dm["height"] * 1e3,
            "pickable": bool(b["pickable"]),
            "n_views": len(vc.get("views_used", [])),
            "spread_mm": vc.get("inter_view_offset_mm"),
            "capture": rec.get("capture"),
        })
    return out


def match(ref, got, max_mm=150.0):
    """Nearest reference parcel in plan view, if it is near enough.

    Position rather than id: ids are assigned per capture, in an order that
    depends on what was detected, so they do not survive a comparison across
    captures. Position does.
    """
    best, bd = None, max_mm
    for r in ref:
        d = float(np.hypot(got["u"] * 1e3 - r["u"] * 1e3,
                           got["v"] * 1e3 - r["v"] * 1e3))
        if d < bd:
            best, bd = r, d
    return best, bd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Grade boxes.json against the reference scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--transform", type=Path, default=Path("scan_to_rig.json"))
    ap.add_argument("--deck-rig", type=float, default=None)
    ap.add_argument("--boxes", action="append", type=Path, required=True,
                    metavar="BOXES_JSON",
                    help="a segmentation/boxes.json. Repeat it: several "
                         "captures of one scene turn a single grading into a "
                         "measurement of repeatability as well as accuracy")
    ap.add_argument("--match-mm", type=float, default=150.0,
                    help="how far a reported centre may sit from a reference "
                         "parcel and still be called the same parcel")
    ap.add_argument("--pickable-only", action="store_true",
                    help="grade only the parcels the pipeline would act on. "
                         "This is the number that matters operationally: a "
                         "detection correctly held back is not an error")
    args = ap.parse_args()

    T, _ = sf.load_transform(args.transform)
    scan_rig = sf.apply_T(T, sf.read_ply(args.scan))
    ref, bf = scan_parcels(scan_rig, args.deck_rig)
    print(f"[scan  ] {len(ref)} reference parcels")

    per_ref = defaultdict(list)
    unmatched, held = 0, 0
    for path in args.boxes:
        got = load_boxes(path, bf)
        for b in got:
            if args.pickable_only and not b["pickable"]:
                held += 1
                continue
            r, d = match(ref, b, args.match_mm)
            if r is None:
                unmatched += 1
                print(f"[extra ] {Path(path).parent.parent.name}: a parcel at "
                      f"u={b['u']:+.2f} v={b['v']:+.2f} matches nothing in the "
                      f"scan, {b['l']:.0f} x {b['w']:.0f} x {b['h']:.0f} mm, "
                      f"{'PICKABLE' if b['pickable'] else 'held'}")
                continue
            per_ref[(round(r["u"], 3), round(r["v"], 3))].append((r, b, d))

    print(f"\n[grade ] {'u':>6} {'v':>6} {'n':>2}  {'scan LxWxh':>20}  "
          f"{'reported':>20}  {'dL':>5} {'dW':>5} {'dh':>5}  {'dcentre':>7} "
          f"{'views':>5} {'pick':>4}")
    dl, dw, dh, dc, sds = [], [], [], [], []
    for key in sorted(per_ref, key=lambda k: k[1]):
        group = per_ref[key]
        r = group[0][0]
        b = {k: float(np.mean([g[1][k] for g in group]))
             for k in ("l", "w", "h", "u", "v")}
        d = float(np.mean([g[2] for g in group]))
        nv = int(np.median([g[1]["n_views"] for g in group]))
        pick = all(g[1]["pickable"] for g in group)
        e = (b["l"] - r["gt_l"], b["w"] - r["gt_w"], b["h"] - r["gt_h"])
        dl.append(e[0])
        dw.append(e[1])
        dh.append(e[2])
        dc.append(d)
        if len(group) > 1:
            sds.append([float(np.std([g[1][k] for g in group], ddof=1))
                        for k in ("l", "w", "h")])
        print(f"[grade ] {r['u']:>6.2f} {r['v']:>6.2f} {len(group):>2}  "
              f"{r['gt_l']:>6.0f} x{r['gt_w']:>6.0f} x{r['gt_h']:>5.0f}  "
              f"{b['l']:>6.0f} x{b['w']:>6.0f} x{b['h']:>5.0f}  "
              f"{e[0]:>+5.0f} {e[1]:>+5.0f} {e[2]:>+5.0f}  {d:>6.0f}  "
              f"{nv:>5} {'yes' if pick else 'HOLD':>4}")

    if not dl:
        print("\nnothing matched; check --transform and --deck-rig")
        return 1

    print(f"\n[acc   ] against the scan, over {len(dl)} parcels"
          + (" the pipeline would pick" if args.pickable_only else ""))
    for lbl, v in (("length", dl), ("width", dw), ("height", dh)):
        a = np.array(v)
        print(f"[acc   ]   {lbl:<7} mean {a.mean():+6.1f} mm   sd "
              f"{a.std(ddof=1) if len(a) > 1 else 0.0:5.1f} mm   worst "
              f"{a[np.argmax(np.abs(a))]:+6.1f} mm")
    a = np.array(dc)
    print(f"[acc   ]   centre  mean {a.mean():6.1f} mm   worst {a.max():6.1f} mm"
          f"   (a pick lands off the parcel when this approaches the cup "
          f"radius)")

    # Is what is left a function of how far the cameras disagreed? If it is,
    # the lever is the inter-view error and nothing else; if it is not, the
    # error is in the segmentation rather than in the depth.
    pairs = [(abs(g[1]["h"] - g[0]["gt_h"]), g[1]["spread_mm"])
             for grp in per_ref.values() for g in grp
             if g[1]["spread_mm"] is not None]
    if len(pairs) > 4:
        a = np.array(pairs, dtype=float)
        if a[:, 1].std() > 1e-6 and a[:, 0].std() > 1e-6:
            r_ = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
            print(f"\n[cause ] height error against inter-view spread: r = "
                  f"{r_:+.2f} over {len(a)} records, spread "
                  f"{a[:, 1].min():.0f} to {a[:, 1].max():.0f} mm")
            print("[cause ]   strongly positive means the residual IS the "
                  "inter-view disagreement, and the only lever left is a "
                  "correction that varies with position rather than height. "
                  "Near zero means the depth is fine and the loss is in the "
                  "segmentation, which is much cheaper to fix.")

    if sds:
        s = np.array(sds)
        print(f"\n[rep   ] repeatability over the captures, median sd: "
              f"length {np.median(s[:, 0]):.1f} mm   width "
              f"{np.median(s[:, 1]):.1f} mm   height {np.median(s[:, 2]):.1f} mm")
        print("[rep   ] repeatability is not accuracy. A face reconstructed "
              "from one camera is reported the same way every capture and can "
              "still be wrong by a hundred millimetres; the accuracy block "
              "above is the one to quote.")
    if held:
        print(f"\n[hold  ] {held} detection(s) held back and not graded. A "
              f"parcel correctly refused is not an error, but check they are "
              f"the ones you expect")
    if unmatched:
        print(f"[extra ] {unmatched} reported parcel(s) matched nothing in the "
              f"scan. Those are false detections and they are worth more "
              f"attention than any dimension error here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())