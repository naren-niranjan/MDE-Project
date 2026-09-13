#!/usr/bin/env python3
"""
belt_from_scan.py

Write belt.json from the reference scan, in the rig's world frame, for
box_segment.py to load with --seg-belt-file.

Why the scan and not a capture
------------------------------
The belt footprint bounds the parcel search, so any variation in it moves every
reported dimension with it. box_segment.py already solves that by freezing the
footprint to a file, but the file it freezes is measured on a DA3 capture, and
on this rig that capture reads the belt 880 mm wide where the scan reads 951.
Every parcel is then measured inside a bound that is 35 mm too tight on each
side, and the ones near the edge trip --seg-belt-edge-warn for a reason that is
not their own.

The conveyor is a fixed piece of the cell and the scanner measures it to a few
millimetres. Measure it once, here, and the bound stops being a function of the
depth model.

    python belt_from_scan.py --scan scans/cell.ply \
        --transform scan_to_rig.json --deck-rig 3.08 --out belt.json

The output is the schema box_segment.load_belt expects. Its corners are in
world metres, so it is independent of whatever plane the segmentation fits on
the day.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import scan_frame as sf


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the conveyor footprint on the reference scan and "
                    "write belt.json in the rig world frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--transform", type=Path, default=Path("scan_to_rig.json"))
    ap.add_argument("--deck-rig", type=float, default=None,
                    help="rough reference-camera-to-deck distance, m, to seed "
                         "the plane fit. Without it the largest plane is "
                         "taken, which in a wide scan is the FLOOR")
    ap.add_argument("--margin", type=float, default=0.050,
                    help="written into the file as margin_m; the search bound "
                         "is the footprint grown by this much, so a parcel "
                         "overhanging the edge is still found")
    ap.add_argument("--max-height", type=float, default=0.60,
                    help="objects standing this far above the deck are counted "
                         "as being on the belt when the footprint is solved. "
                         "Anything on the belt is on the belt by definition, "
                         "and including it is what lets the footprint survive "
                         "a loaded conveyor")
    ap.add_argument("--close", type=float, default=0.30,
                    help="gap bridged before the components are taken; must "
                         "exceed the widest parcel and stay below the distance "
                         "to any other surface at deck height")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="shrink the measured footprint by this much on every "
                         "side before writing. Use it only if the scan picks "
                         "up the side frames as belt")
    ap.add_argument("--out", type=Path, default=Path("belt.json"))
    args = ap.parse_args()

    T, tmeta = sf.load_transform(args.transform)
    scan = sf.apply_T(T, sf.read_ply(args.scan))
    P = sf.voxel(scan, 0.006)
    print(f"[load ] scan {len(P)} points in the rig world frame")

    n, d = sf.deck_plane(P, args.deck_rig)
    print(f"[deck ] n={np.round(n, 5)} d={d:.4f} m from the reference camera")

    bf = sf.belt_frame(P, n, d, close_m=args.close, max_height=args.max_height)
    L = bf["length_m"] - 2 * args.trim
    W = bf["width_m"] - 2 * args.trim
    if L <= 0 or W <= 0:
        raise SystemExit("--trim removed the whole footprint")
    bx, by, c = bf["x_axis"], bf["y_axis"], bf["centre"]
    corners = np.array([c - bx * L / 2 - by * W / 2,
                        c + bx * L / 2 - by * W / 2,
                        c + bx * L / 2 + by * W / 2,
                        c - bx * L / 2 + by * W / 2])

    belt = {
        "centre_m": c.tolist(),
        "corners_m": corners.tolist(),
        "x_axis": bx.tolist(),
        "y_axis": by.tolist(),
        "length_m": float(L),
        "width_m": float(W),
        "n_points": int(len(P)),
        "n_components": 1,
        "raster_cell_m": 0.01,
        "closed_m": args.close,
        "included_objects": True,
        "margin_m": args.margin,
        "source": f"reference scan {args.scan.name} via {args.transform.name}",
        "frozen_to": str(args.out),
        "scan_to_rig": tmeta,
    }
    args.out.write_text(json.dumps(belt, indent=2))
    print(f"[belt ] {L * 1e3:.0f} x {W * 1e3:.0f} mm, search bound is that "
          f"plus {args.margin * 1e3:.0f} mm")
    print(f"\nwritten: {args.out}")
    print(f"use it with:  --seg-belt-file {args.out}   (and NOT "
          f"--seg-belt-refit, which would overwrite it from a DA3 capture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())