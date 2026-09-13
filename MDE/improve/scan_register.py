#!/usr/bin/env python3
"""
scan_register.py

Solve the rigid transform that puts the reference scanner's cloud into the
rig's world frame, and freeze it.

Why freeze it
-------------
The scanner does not move between scans and neither do the cameras, so the
transform is a fixed property of the cell. Re-solving it inside every tool
would let it vary by a millimetre or two per run, and that variation would be
indistinguishable from the depth error the scan is there to measure. Solved
once, written to scan_to_rig.json, and read thereafter, the reference frame is
the same in the correction fit and in the grading.

Solve it against a capture whose RAW depth is stored, on a belt loaded enough
that the parcels can pin the offset along it. An empty belt is a translation-
invariant strip and the registration will slide along it.

    python scan_register.py \
        --scan scans/pointcloud_20260818_140030.ply \
        --capture runs/live_.../frame_00000 \
        --deck-scan 3.31 --deck-rig 3.08 \
        --out scan_to_rig.json

Re-run it if the scanner is moved, if the cameras are recalibrated, or if the
conveyor is repositioned. Nothing else needs re-running: scan_align.py and
gt_compare.py both read the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import scan_frame as sf


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Solve and freeze the reference scanner to rig world "
                    "transform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scan", type=Path, required=True,
                    help="reference scan of the cell, in the scanner's frame")
    ap.add_argument("--capture", type=Path, default=None,
                    help="a frame_NNNNN directory from da3_stream.py holding "
                         "depth_raw_<cam>.npy, K_<cam>.npy and E_<cam>.npy")
    ap.add_argument("--cloud", type=Path, default=None,
                    help="a fused cloud already in the rig world frame, as an "
                         "alternative to --capture. Prefer --capture: it lets "
                         "the registration use the RAW depth, so a wrong "
                         "correction cannot pull the frame with it")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--deck-scan", type=float, default=None,
                    help="rough scanner-to-deck distance, m. Without it the "
                         "largest plane is taken, which in a wide scan is the "
                         "FLOOR")
    ap.add_argument("--deck-rig", type=float, default=None,
                    help="rough reference-camera-to-deck distance, m")
    ap.add_argument("--voxel", type=float, default=0.006)
    ap.add_argument("--conf-percentile", type=float, default=20.0,
                    help="drop points below this per-camera confidence "
                         "percentile before registering, as the streamer does. "
                         "Only applied when conf_<cam>.npy was written")
    ap.add_argument("--edge-thresh", type=float, default=0.02,
                    help="relative depth gradient above which a pixel is "
                         "treated as a discontinuity. This is the one that "
                         "matters: without it the hanging veil around every "
                         "parcel edge is registered as though it were surface")
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--deck-window", type=float, default=0.15,
                    help="half-width of the search window about --deck-rig and "
                         "--deck-scan. Wide enough that a seed off by a "
                         "hundred millimetres still finds the deck, since DA3's "
                         "absolute scale moves with --process-res")
    ap.add_argument("--deck-band", type=float, default=0.050,
                    help="how far below the fitted deck a point may sit and "
                         "still count as deck when the belt footprint is "
                         "measured. It must exceed the per-view layering, "
                         "which in an uncorrected cloud is a couple of "
                         "centimetres; too tight and whole views drop out at "
                         "the belt edges and the footprint comes back narrow")
    ap.add_argument("--overlay", type=Path,
                    default=Path("registration_overlay.png"),
                    help="silhouettes of both clouds at the chosen alignment. "
                         "Look at this before anything else when the "
                         "correlation is low: a misalignment shifts every "
                         "parcel the same way, a changed scene does not")
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--free-icp", action="store_true",
                    help="let the refinement rotate out of plane. The deck "
                         "frames are built first, so a legitimate refinement "
                         "cannot need it; this exists only to reproduce a "
                         "diverged fit deliberately")
    ap.add_argument("--max-tilt-deg", type=float, default=2.0,
                    help="refuse to write when the refinement tipped the "
                         "vertical axis by more than this. The two decks are "
                         "parallel by construction before it runs, so any "
                         "real out-of-plane component is ICP hinging one "
                         "cloud off the other to buy a lower residual")
    ap.add_argument("--probe", action="store_true",
                    help="list the largest planes in the rig cloud with their "
                         "distances, then exit. Read --deck-rig off this "
                         "rather than guessing it")
    ap.add_argument("--probe-planes", type=int, default=6)
    ap.add_argument("--no-filter", action="store_true",
                    help="register the raw union with nothing removed. It will "
                         "be dominated by the veil; this exists to show that")
    ap.add_argument("--out", type=Path, default=Path("scan_to_rig.json"))
    ap.add_argument("--write-scan-in-rig", type=Path, default=None,
                    help="also write the transformed scan, to be opened beside "
                         "fused.ply as a visual check")
    ap.add_argument("--write-rig-cloud", type=Path, default=None,
                    help="also write the filtered rig cloud the registration "
                         "actually used, which is what to open the scan "
                         "against when the residual looks wrong")
    args = ap.parse_args()

    if bool(args.capture) == bool(args.cloud):
        raise SystemExit("give exactly one of --capture or --cloud")

    filters = ({"conf_percentile": 0.0, "edge_thresh": 0.0,
                "max_incidence": 90.0} if args.no_filter else
               {"conf_percentile": args.conf_percentile,
                "edge_thresh": args.edge_thresh,
                "edge_dilate": args.edge_dilate,
                "max_incidence": args.max_incidence})

    scan = sf.read_ply(args.scan)
    print(f"[load ] scan {len(scan)} points")

    if args.capture:
        arrays = sf.load_capture_depths(args.capture, args.cameras, raw=True)
        rig = sf.rig_cloud(arrays, **filters)
        print(f"[load ] rig  {len(rig)} points from raw depth, cameras "
              f"{sorted(arrays)}")
        if not any("conf" in a for a in arrays.values()):
            print("[note ] no conf_<cam>.npy in this capture, so the "
                  "confidence percentile was not applied")
    else:
        rig = sf.read_ply(args.cloud)
        print(f"[load ] rig  {len(rig)} points from {args.cloud}")
        print("[note ] registering against a fused cloud, so whatever "
              "correction produced it is baked into this frame. Prefer "
              "--capture.")

    if args.probe:
        print("\n--- rig cloud ---")
        sf.probe_planes(rig, args.probe_planes)
        print("\n--- scan ---")
        sf.probe_planes(scan, args.probe_planes)
        return 0

    T, meta = sf.solve_scan_to_rig(scan, rig, args.deck_scan, args.deck_rig,
                                   args.voxel, planar=not args.free_icp,
                                   overlay=args.overlay,
                                   below=args.deck_band,
                                   window=args.deck_window)
    R = T[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"\n[solve] rotation {ang:.3f} deg   translation "
          f"{np.round(T[:3, 3], 4)} m")
    print(f"[solve] residual median {meta['icp_median_mm']:.1f} mm   "
          f"out of plane {meta['icp_out_of_plane_deg']:.2f} deg")

    if meta["icp_out_of_plane_deg"] > args.max_tilt_deg:
        raise SystemExit(
            f"refusing to write: the refinement tipped the vertical axis by "
            f"{meta['icp_out_of_plane_deg']:.2f} deg, above --max-tilt-deg "
            f"{args.max_tilt_deg:.1f}. Both decks are parallel before it runs, "
            f"so this is a diverged fit: the scan has been hinged off the rig "
            f"cloud along the belt, which touches at one end and separates at "
            f"the other. Fix the start rather than the end, in this order: "
            f"--probe to check --deck-rig, then the silhouette correlation "
            f"above, then whether the scan and the capture hold the same "
            f"parcel arrangement.")

    if meta["score"] < 0.75 or meta["icp_median_mm"] > 25.0:
        print("[WARN ] this registration is not good enough to fit a "
              "correction against. In order of likelihood:")
        print("[WARN ]   0. LOOK AT " + str(args.overlay) + " FIRST. If the "
              "parcels sit on each other bar a shift, it is alignment; if "
              "some sit and others do not, the scene changed between the "
              "scan and the capture and no setting repairs that")
        print("[WARN ]   1. the veil. Check the [filt ] lines above: the edge "
              "filter should be removing five per cent or more. --edge-thresh "
              "is a gradient PER PIXEL, so doubling --process-res halves it: "
              "at 1008 use --edge-thresh 0.01 --edge-dilate 2")
        print("[WARN ]   2. the scan and the capture are of different parcel "
              "arrangements. The silhouette correlation is the tell; it "
              f"came out {meta['score']:.3f} and wants to be above 0.8")
        print("[WARN ]   3. the deck seeds. Compare the two [frame] deck "
              "distances against a tape measure")
        print("[WARN ]   4. the belt widths on the two [frame] lines. If the "
              "rig one is far narrower than the scan's, its deck plane "
              "locked onto one view's sheet rather than the deck")
        print("[WARN ] Open the two written clouds together before going on.")

    meta.update({"scan": str(args.scan),
                 "rig_source": str(args.capture or args.cloud),
                 "rotation_deg": ang})
    sf.save_transform(args.out, T, meta)
    print(f"\nwritten: {args.out}")

    if args.write_scan_in_rig:
        sf.write_ply(args.write_scan_in_rig, sf.apply_T(T, scan))
        print(f"written: {args.write_scan_in_rig}")
    if args.write_rig_cloud:
        sf.write_ply(args.write_rig_cloud, rig)
        print(f"written: {args.write_rig_cloud}")
    if args.write_scan_in_rig or args.write_rig_cloud:
        print("open those two together; they should sit on each other")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())