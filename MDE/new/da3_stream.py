#!/usr/bin/env python3
"""
da3_stream.py

Live four-camera acquisition feeding Depth Anything 3 multi-view fusion,
conditioned on the calibrated intrinsics and extrinsics of the current rig,
producing one fused point cloud per capture, and, on the same keypress, the
segmented parcels and their pick data.

Self-contained by design: nothing is imported from da3_fuse.py or from
depth_align.py, because the rig has been rebuilt around the Edmund Optics
TECHSPEC C Series 12 mm f/1.8 lenses (EO 58-001) and the earlier ground-truth
constants no longer apply. The correction is re-implemented here rather than
imported, deliberately, so that the streamer has no import-time dependency on
the calibration tooling. box_segment.py and box_geometry.py are the exception,
and they are optional: if either is absent the streamer runs exactly as before
and says so once at start-up.

Changes in this revision
------------------------
1. THE CORRECTION IS NOW IN HEIGHT ABOVE THE DECK, NOT IN INVERSE DEPTH. Each
   view carries a deck plane in its own camera frame and a linear map on the
   height measured perpendicular to it:

       h  = z (n_cam . r) - d_cam
       h' = alpha h + beta
       z' = (h' + d_cam) / (n_cam . r)

   That is the variable the error actually lives in. A depth model of this
   family reconstructs a rise of h as h(1 - k), with k near 0.17 on this rig,
   and correcting for that in disparity over a depth span of 0.22 m out of
   3.07 m was a badly conditioned reparameterisation of the same thing. It
   produced a condition number of 1.6e4 and a leave-one-out error of 43 to
   86 mm.

2. THE RESIDUAL DISPARITY GRID IS GONE. An 8 x 8 grid, a quarter covered, its
   empty cells at zero and the whole thing bilinearly stretched onto a 504
   pixel map, put 80 to 110 mm of depth correction in some cells and none in
   the neighbouring ones. On a conveyor that reads as a fold. --no-affine-grid
   is accepted and ignored so that existing shell scripts still run.

3. THE CORRECTION FILE IS depth_correction.json, MODEL deck_height_linear.
   An old depth_affine.json is refused by name rather than silently
   misapplied, since its parameters mean something different.

4. WHAT THE CORRECTION DOES AND DOES NOT DO. It removes the per-view part: the
   four views place the same surface on four sheets, and that is what makes a
   cloud look layered. It does not remove the part shared by all four views
   inside one capture, which is about 93 per cent of the capture-to-capture
   scatter, because DA3 solves the four frames jointly and one error appears
   in all four at once. That part moves all four sheets together, so it does
   not cause layering, and it is invisible to any inter-view consistency
   check. It remains as an absolute height uncertainty, carried in the
   correction file under height_uncertainty, and it is the number to quote
   downstream rather than anything derived from view agreement.

5. --fuse-mode tsdf integrates the four depth maps into a truncated signed
   distance volume instead of concatenating four point clouds. A union of four
   sheets is still four sheets no matter how fine the voxel. Use it once the
   correction is in place, not before.

6. A layer report prints after every capture: the dominant plane is fitted,
   then each camera's median signed offset from it is reported, at belt level
   and again for whatever stands above it. That number is the layering, in
   millimetres, and it is the one to watch.

7. SEGMENTATION AND BOX GEOMETRY RUN ON EVERY CAPTURE. Pressing p now writes a
   capture that can be re-opened rather than only a cloud: the corrected depth,
   the intrinsics and extrinsics DA3 returned, the processed images, the fused
   cloud, and then the parcels found in it, as records, as overlays and as 3D
   geometry. On by default when box_segment.py and box_geometry.py are beside
   this file; --no-segment and --no-geom switch the two halves off.

   The segmentation is fed the same filtered per-view world points that go into
   the fusion, so what is segmented is what is displayed, not a second
   back-projection with its own filtering.

What p writes
-------------
    frame_00000/
      fused.ply                        the fused cloud
      fused_by_camera.ply              one colour per view, which is how
                                       layering is seen rather than guessed at
      depth_<cam>.npy  depth_raw_<cam>.npy  conf_<cam>.npy
      K_<cam>.npy  E_<cam>.npy         as DA3 returned them, for the processed
                                       grid, not the rectified full frame
      proc_<cam>.png                   the images DA3 saw
      layers.json                      the per-view layer report
      segmentation/
        boxes.json  boxes.csv          the parcel records and pick poses
        segmented.ply                  coloured by parcel, conveyor grey
        segmented.png  seg_<cam>.png    the top-down render and the overlays
        boxes_mesh.ply  boxes_lines.ply the cuboids as geometry
        scene_boxed.ply                 cloud and boxes in one file, optional

    boxes_log.csv                      every parcel of every capture, one file,
                                       which is what makes a repeatability
                                       sweep a comparison rather than a folder
                                       to concatenate

The arrays are written before the segmentation runs, deliberately. If the
segmentation fails, the capture is still complete and box_scene.py can be
pointed at frame_NNNNN afterwards with different thresholds.

Settled findings carried forward
--------------------------------
No pad_square, no rot_k, no manual rescaling of the returned K. Depth is Z
along the optical axis, not ray distance. Extrinsics are world-to-camera,
x_cam = R x_world + t. align_to_input_ext_scale makes the returned extrinsics
consistent with the supplied baseline, it is not an independent metric anchor.

Rectification
-------------
getOptimalNewCameraMatrix at alpha = 0 returns a different valid ROI per
camera, and mixed sizes in one DA3 batch trigger a centre crop to the smallest
common dimension, shifting the principal point without changing the focal
length. --rect-mode common crops every view to one shared size, centred within
its own valid ROI, with the principal point adjusted. Keep it as the default.

Examples
--------
Check calibration and geometry without touching hardware:
    python da3_stream.py --calib-dir /home/jetson/Projects/Calibration_4_5/results \\
        --self-test

Live heat map, capture, segment and export on p:
    python da3_stream.py --calib-dir /home/jetson/Projects/Calibration_4_5/results \\
        --correction depth_correction.json --save-mode all --save-npy \\
        --seg-plane-distance 3.10 3.25 --seg-max-height 0.40 \\
        --seg-belt-file belt.json --seg-views all \\
        --out-dir runs/live_$(date +%Y%m%d_%H%M%S)

Throughput measurement, with the segmentation out of the way:
    python da3_stream.py --calib-dir /home/jetson/Projects/Calibration_4_5/results \\
        --no-preview --capture-trigger continuous --save-mode none \\
        --no-segment --duration 120 --warmup 3 --json live_timing.json
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import select
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import torch
except ImportError:
    torch = None

# Optional, and kept optional on purpose: a rig that has not had the
# segmentation copied onto it should still stream and still fuse.
try:
    import box_segment as bs
except ImportError as exc:  # noqa: BLE001
    bs = None
    _SEG_IMPORT_ERROR = exc
try:
    import box_geometry as bg
except ImportError as exc:  # noqa: BLE001
    bg = None
    _GEOM_IMPORT_ERROR = exc


CAMERAS = {
    "left": "260505158",
    "center": "261100628",
    "right": "261100627",
    "top": "261100631",
}

PIXEL_FORMATS = {"RGB8": (3, "rgb"), "BGR8": (3, "bgr"), "BayerRG8": (1, "bayer")}

PALETTE = [(0.90, 0.30, 0.25), (0.25, 0.65, 0.90),
           (0.35, 0.75, 0.40), (0.95, 0.75, 0.20),
           (0.70, 0.45, 0.85), (0.95, 0.55, 0.75)]

DEFAULT_CALIB = Path("/home/jetson/Projects/Calibration_4_5/results")
DEFAULT_LENS = "EO-58-001-12mm"
CORRECTION_MODEL = "deck_height_linear"
GIGE_MB_S = 125.0          # practical ceiling of one 1000BASE-T camera link


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live four-camera DA3 fusion on the 12 mm rig, with "
                    "parcel segmentation and box geometry on each capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/live"))
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--expect-lens", default=DEFAULT_LENS,
                    help="lens_id every calibration record must carry; empty "
                         "string accepts whatever is on disk provided all "
                         "four agree")
    ap.add_argument("--self-test", action="store_true",
                    help="run the geometry round-trip, and the calibration "
                         "and rectification checks if --calib-dir exists, "
                         "without opening a camera or loading the model")

    # correction
    ap.add_argument("--correction", "--affine", dest="correction", type=Path,
                    default=None,
                    help="depth_correction.json from depth_align.py, applied "
                         "per view before back-projection. This is what "
                         "removes the layered surfaces; without it each "
                         "camera places the same surface on its own sheet")
    ap.add_argument("--no-affine-grid", dest="dead_grid_flag",
                    action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-view-correction", dest="view_correction",
                    action="store_false", default=True,
                    help="apply only the rig-wide part of the correction and "
                         "leave the per-view deviations out. Useful for "
                         "showing what the per-view term is worth, and as a "
                         "fallback if the per-view term was written past a "
                         "failed gate")

    # acquisition
    ap.add_argument("--width", type=int, default=2448)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--pixel-format", default="RGB8",
                    choices=sorted(PIXEL_FORMATS),
                    help="RGB8 is 15.0 MB/frame on this sensor; BayerRG8 is "
                         "5.0 MB and is debayered on the host")
    ap.add_argument("--bayer-code", default="COLOR_BayerRG2RGB",
                    help="OpenCV conversion used only for BayerRG8")
    ap.add_argument("--exposure-us", type=float, default=20000.0)
    ap.add_argument("--gain-db", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="per-camera acquisition rate; 0 derives a rate that "
                         "fits --link-budget-mb-s")
    ap.add_argument("--link-budget-mb-s", type=float, default=250.0,
                    help="aggregate bandwidth the uplink to the Jetson is "
                         "assumed to carry, in MB/s; a 2.5 GbE port is about "
                         "312 MB/s at line rate, so leave headroom")
    ap.add_argument("--no-link-limit", dest="link_limit", action="store_false",
                    default=True,
                    help="do not set DeviceLinkThroughputLimit from the budget")
    ap.add_argument("--num-buffers", type=int, default=6)
    ap.add_argument("--buffer-timeout-ms", type=int, default=2000)
    ap.add_argument("--throughput-limit", type=int, default=0,
                    help="explicit per-camera DeviceLinkThroughputLimit in "
                         "bytes/s; overrides the budget calculation")
    ap.add_argument("--ptp", action="store_true",
                    help="enable IEEE 1588 on all cameras and report status")

    # synchronisation
    ap.add_argument("--max-sync-ms", type=float, default=-1.0,
                    help="reject a frame set whose timestamps span more than "
                         "this; negative derives it from the frame period, "
                         "0 disables the check")
    ap.add_argument("--use-device-timestamp", action="store_true",
                    help="judge spread by camera timestamps rather than host "
                         "arrival; only meaningful with --ptp locked")

    # capture triggering
    ap.add_argument("--capture-trigger", "--trigger", dest="capture_trigger",
                    choices=["key", "settle", "continuous"], default="key",
                    help="what causes a FUSION. Inference and the heat map run "
                         "continuously regardless. key waits for p")
    ap.add_argument("--capture-key", default="p",
                    help="key that forces a capture in any trigger mode. It is "
                         "read from the preview window when there is one and "
                         "from the terminal when there is not, so a headless "
                         "run over SSH still captures on the key")
    ap.add_argument("--settle-thresh", type=float, default=1.5,
                    help="mean absolute grey-level difference below which the "
                         "reference view counts as static")
    ap.add_argument("--settle-frames", type=int, default=4,
                    help="consecutive static frames required to fire")
    ap.add_argument("--settle-rearm", type=float, default=4.0,
                    help="difference that must be exceeded before the trigger "
                         "will fire again")
    ap.add_argument("--settle-timeout", type=float, default=0.0,
                    help="fire anyway after this many seconds of waiting; "
                         "0 waits indefinitely for genuine motion")

    # model
    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--process-res", type=int, default=504,
                    help="multiple of the ViT patch size 14; 504 = 14x36, "
                         "1008 = 14x72 at roughly four times the cost")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2,
                    help="inference passes run and discarded at start-up, "
                         "before the loop, to exclude autotuning and allocator "
                         "warm-up from the timings")
    ap.add_argument("--start-timeout", type=float, default=30.0,
                    help="seconds to wait at start-up for every camera to "
                         "deliver its first frame")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="isotropic downscale applied to the rectified image "
                         "before inference, with intrinsics scaled to match")
    ap.add_argument("--align-ext-scale", dest="align_ext_scale",
                    action="store_true", default=True,
                    help="ask DA3 to report extrinsics on the supplied "
                         "baseline scale; consistent, not a metric anchor")
    ap.add_argument("--no-align-ext-scale", dest="align_ext_scale",
                    action="store_false")

    # filtering
    ap.add_argument("--conf-percentile", type=float, default=40.0,
                    help="drop points below this per-camera confidence "
                         "percentile; 0 disables")
    ap.add_argument("--edge-thresh", type=float, default=0.02,
                    help="relative depth gradient above which a pixel is "
                         "treated as a discontinuity; 0 disables")
    ap.add_argument("--edge-dilate", type=int, default=2,
                    help="how far the discontinuity rejection is grown, in "
                         "depth-grid pixels. The model smooths across a box "
                         "edge over several pixels, and those samples land "
                         "between the box top and the belt as a hanging veil")
    ap.add_argument("--max-incidence", type=float, default=70.0,
                    help="drop samples whose surface normal exceeds this "
                         "angle from the view ray; 90 disables. A grazing "
                         "sample carries the largest depth error for the "
                         "smallest reprojection error, which is exactly the "
                         "sample that lands off-surface")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--fuse-mode", choices=["union", "tsdf"], default="union",
                    help="union concatenates the four clouds, so four sheets "
                         "stay four sheets. tsdf integrates them into one "
                         "surface, which is only correct once --correction "
                         "has removed the systematic per-view differences")
    ap.add_argument("--tsdf-trunc", type=float, default=0.02,
                    help="TSDF truncation distance in m; it must exceed the "
                         "residual disagreement between views or the surfaces "
                         "will not merge")
    ap.add_argument("--tsdf-depth-trunc", type=float, default=6.0)
    ap.add_argument("--rect-mode", choices=["common", "roi", "full"],
                    default="common",
                    help="common crops every view to one shared size, the "
                         "only mode that guarantees no batch centre-crop")
    ap.add_argument("--undistort", dest="undistort", action="store_true",
                    default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")

    # diagnostics
    ap.add_argument("--layer-report", dest="layer_report",
                    action="store_true", default=True,
                    help="after each capture, fit the dominant plane and "
                         "report each camera's offset from it")
    ap.add_argument("--no-layer-report", dest="layer_report",
                    action="store_false")
    ap.add_argument("--layer-band-mm", type=float, default=40.0,
                    help="half-thickness of the band about the dominant plane "
                         "used to measure each view's offset")
    ap.add_argument("--layer-above-mm", type=float, default=60.0,
                    help="points more than this above the dominant plane are "
                         "reported separately, as the parcel-top layering")

    # output
    ap.add_argument("--save-mode", choices=["none", "latest", "all"],
                    default="all")
    ap.add_argument("--save-per-camera", action="store_true")
    ap.add_argument("--save-npy", action="store_true",
                    help="also write depth, confidence, K and E arrays on "
                         "every capture. A capture forced with the capture key "
                         "writes them regardless, so that a capture taken by "
                         "hand can always be re-opened by box_scene.py")
    ap.add_argument("--no-save-proc", dest="save_proc", action="store_false",
                    default=True,
                    help="do not write the processed images beside the arrays. "
                         "They are what the camera overlays are drawn on, so "
                         "without them a capture can be re-segmented offline "
                         "but not re-drawn")
    ap.add_argument("--colour-by-camera", dest="colour_by_camera",
                    action="store_true", default=True,
                    help="also write fused_by_camera.ply, one colour per "
                         "view, which is how layering is seen rather than "
                         "guessed at")
    ap.add_argument("--no-colour-by-camera", dest="colour_by_camera",
                    action="store_false")
    ap.add_argument("--view-cloud", action="store_true",
                    help="open the fused cloud in a viewer after each "
                         "capture; blocks until the window is closed")
    ap.add_argument("--no-preview", dest="preview", action="store_false",
                    default=True)
    ap.add_argument("--preview-width", type=int, default=560)
    ap.add_argument("--status-interval", type=float, default=2.0,
                    help="seconds between console status lines while idle")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds; 0 runs until quit")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--json", default="live_diagnostics.json")

    # segmentation and geometry, both on by default when the modules are here
    ap.add_argument("--no-segment", action="store_true",
                    help="do not segment captures. Segmentation of a "
                         "four-camera scene costs a few hundred milliseconds, "
                         "so switch it off for a throughput measurement")
    ap.add_argument("--no-geom", action="store_true",
                    help="segment, but do not export the parcels as geometry")
    ap.add_argument("--no-box-log", dest="box_log", action="store_false",
                    default=True,
                    help="do not append every parcel to one boxes_log.csv at "
                         "the run root")
    if bs is not None:
        bs.add_arguments(ap)
    if bg is not None:
        bg.add_arguments(ap)

    args = ap.parse_args()

    # --segment exists only when box_segment.py imported. Default it on here
    # rather than in box_segment, where the offline driver wants the opposite.
    if bs is not None:
        args.segment = bool(getattr(args, "segment", False)) or not args.no_segment
        if args.no_segment:
            args.segment = False
    else:
        args.segment = False
    if bg is not None:
        args.geom = bool(getattr(args, "geom", False)) or not args.no_geom
        if args.no_geom or not args.segment:
            args.geom = False
    else:
        args.geom = False
    return args


# --------------------------------------------------------------------------
# terminal key reading
# --------------------------------------------------------------------------

class KeyPoller:
    """Non-blocking single-character reads from a terminal.

    For headless runs, where there is no preview window to take a key from.
    The terminal is put into cbreak so a keystroke arrives without a newline,
    and the settings are restored on stop, including after an exception, since
    otherwise the shell is left with no echo.

    Does nothing at all when stdin is not a terminal, which is what happens
    under nohup or a service manager, so the same code runs there unguarded.
    """

    def __init__(self):
        self._fd = None
        self._saved = None

    def start(self):
        try:
            import termios
            import tty
        except ImportError:
            return self
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def stop(self):
        if self._fd is not None and self._saved is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        self._fd = None
        self._saved = None

    def poll(self) -> int:
        if self._fd is None:
            return -1
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return -1
        ch = os.read(self._fd, 1)
        return ch[0] if ch else -1


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def load_intrinsics(calib_dir: Path, name: str) -> dict:
    path = calib_dir / f"intrinsics_{name}.json"
    if not path.exists():
        raise SystemExit(f"missing intrinsics file: {path}")
    data = json.loads(path.read_text())
    return {
        "K": np.asarray(data["camera_matrix"], np.float64).reshape(3, 3),
        "dist": np.asarray(data.get("dist_coeffs", [0.0] * 5), np.float64).ravel(),
        "size": tuple(int(v) for v in data["image_size"]),
        "rms_px": float(data.get("rms_reprojection_error_px", float("nan"))),
        "lens_id": data.get("lens_id"),
        "lens_mm": data.get("lens_mm"),
        "serial": data.get("serial"),
        "fx_over_expected": data.get("fx_over_expected"),
        "distortion_max_field_pct": data.get("distortion_max_field_pct"),
    }


def load_extrinsics(calib_dir: Path, names, reference: str) -> dict:
    path = calib_dir / "extrinsics.json"
    if not path.exists():
        raise SystemExit(f"missing extrinsics file: {path}")
    data = json.loads(path.read_text())
    out = {}
    for n in names:
        if n not in data:
            raise SystemExit(f"camera {n!r} absent from {path}")
        rec = data[n]
        E = np.eye(4)
        E[:3, :3] = np.asarray(rec["R"], np.float64).reshape(3, 3)
        E[:3, 3] = np.asarray(rec["t"], np.float64).ravel()
        det = float(np.linalg.det(E[:3, :3]))
        if abs(det - 1.0) > 1e-6:
            raise SystemExit(f"{n}: rotation determinant {det:.9f} is not 1; "
                             f"the extrinsics file is not a rigid transform")
        if rec.get("reference") not in (None, reference):
            raise SystemExit(f"{n} is referenced to {rec.get('reference')!r}, "
                             f"not {reference!r}")
        out[n] = {"E": E,
                  "lens_id": rec.get("lens_id"),
                  "serial": rec.get("serial"),
                  "rms_px": rec.get("rms_error_px"),
                  "baseline_m": rec.get("baseline_m"),
                  "n_shared_views": rec.get("n_shared_views")}
    return out


def check_lens(intr, ext, names, expect):
    tags = {}
    for n in names:
        tags[f"intrinsics_{n}"] = intr[n]["lens_id"]
        tags[f"extrinsics_{n}"] = ext[n]["lens_id"]
    present = {v for v in tags.values() if v is not None}
    missing = [k for k, v in tags.items() if v is None]
    if missing:
        raise SystemExit("calibration records carry no lens_id: "
                         + ", ".join(missing)
                         + "\nre-run the calibration with the lens tag written "
                           "out, or the fusion cannot tell a 12 mm set from an "
                           "8 mm one")
    if len(present) > 1:
        raise SystemExit(f"mixed lens tags in the calibration set: {sorted(present)}")
    found = present.pop()
    if expect and found != expect:
        raise SystemExit(f"calibration lens tag is {found!r} but --expect-lens "
                         f"is {expect!r}; this calibration belongs to a "
                         f"different optical configuration")
    return found


def check_serials(intr, ext, names):
    for n in names:
        want = CAMERAS[n]
        for src, rec in (("intrinsics", intr[n]), ("extrinsics", ext[n])):
            got = rec.get("serial")
            if got is not None and str(got) != want:
                raise SystemExit(f"{src}_{n} carries serial {got}, but {n} is "
                                 f"{want} in this script; the calibration and "
                                 f"the rig disagree about which camera is which")


def camera_centre(E):
    return -E[:3, :3].T @ E[:3, 3]


def rt(E):
    E = np.asarray(E, np.float64)
    return E[:3, :3], E[:3, 3]


# --------------------------------------------------------------------------
# depth correction
# --------------------------------------------------------------------------

def load_correction(path: Path, names, lens, view_terms=True):
    """
    Load a per-view height correction written by depth_align.py.

    Two hard gates. The MODEL NAME, because an inverse-depth depth_affine.json
    carries parameters that mean something entirely different and applying
    them here would produce a clean-looking cloud in the wrong place. And the
    LENS TAG, because a correction solved on the 8 mm rig has the same shape
    as one solved on the 12 mm rig and nothing downstream could tell.

    view_terms=False keeps the rig-wide part and discards the per-view
    deviations, which is the right fallback if the per-view part was written
    past a failed gate.
    """
    data = json.loads(Path(path).read_text())
    model = data.get("model")
    if model != CORRECTION_MODEL:
        raise SystemExit(
            f"{path} declares model {model!r}, not {CORRECTION_MODEL!r}. An "
            f"inverse-depth depth_affine.json from the previous depth_align.py "
            f"cannot be applied here: its a and b are a map on disparity, not "
            f"on height above the deck. Re-run depth_align.py.")

    tag = data.get("lens_id")
    if lens and tag and tag != lens:
        raise SystemExit(f"{path} was solved for lens {tag!r} but the rig is "
                         f"running {lens!r}; refusing to apply it")

    shared = data.get("shared") or {}
    views = data.get("views", {})
    out = {}
    for n in names:
        rec = views.get(n)
        if rec is None:
            print(f"[corr ] {n}: absent from {path}, left uncorrected")
            continue
        n_cam = np.asarray(rec["n_cam"], np.float64).ravel()
        nrm = float(np.linalg.norm(n_cam))
        if nrm < 1e-9:
            raise SystemExit(f"{path}: {n} carries a degenerate deck normal")
        n_cam = n_cam / nrm
        if view_terms:
            alpha, beta = float(rec["alpha"]), float(rec["beta"])
        else:
            p, q = float(shared["p"]), float(shared["q"])
            alpha, beta = 1.0 / p, -q / p
        if alpha <= 0:
            raise SystemExit(f"{path}: {n} carries a non-positive alpha")
        out[n] = {"alpha": alpha, "beta": beta,
                  "n_cam": n_cam, "d_cam": float(rec["d_cam"]),
                  "p_v": rec.get("p_v"), "q_v": rec.get("q_v"),
                  "slope_fitted": bool(rec.get("slope_fitted", False)),
                  "sigma_mm": rec.get("delayering_sigma_mm")}
    if not out:
        raise SystemExit(f"{path} carries no usable view corrections")
    return out, data


def pixel_rays(shape, K):
    h, w = shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def apply_height_correction(depth, K, n_cam, d_cam, alpha, beta):
    """
    h = z (n_cam . r) - d_cam ; h' = alpha h + beta ; z' = (h' + d_cam)/(n . r)

    NaN where the ray is parallel to the deck plane or the correction drives
    the depth non-positive. Those are genuine failures at that pixel, not
    something to clamp: a clamped pixel would be back-projected to a plausible
    wrong position and there would be no way to notice.

    The plane is a 3D quantity in the camera's own frame, so it does not care
    what resolution the depth map arrived at. Pass the intrinsics that belong
    to the map being corrected, which for DA3 means pred.intrinsics rather
    than the rectified K.
    """
    z = np.asarray(depth, np.float64)
    nr = pixel_rays(z.shape, K) @ np.asarray(n_cam, np.float64)
    h = z * nr - float(d_cam)
    h2 = float(alpha) * h + float(beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(np.abs(nr) > 1e-9, (h2 + float(d_cam)) / nr, np.nan)
    return np.where(np.isfinite(z2) & (z2 > 0), z2, np.nan)


def correct_depths(depth, names, corr, K_out):
    if not corr:
        return depth, None
    out = np.array(depth, np.float64, copy=True)
    shifts = {}
    for i, n in enumerate(names):
        rec = corr.get(n)
        if rec is None:
            continue
        before = out[i]
        after = apply_height_correction(before, K_out[i], rec["n_cam"],
                                        rec["d_cam"], rec["alpha"],
                                        rec["beta"])
        ok = np.isfinite(before) & np.isfinite(after) & (before > 0)
        shifts[n] = (float(np.median((after[ok] - before[ok]) * 1e3))
                     if ok.any() else float("nan"))
        out[i] = after
    return out, shifts


# --------------------------------------------------------------------------
# rectification
# --------------------------------------------------------------------------

def build_rect_plan(intr, names, size, mode, enabled):
    w, h = size
    plan = {}

    if not enabled:
        for n in names:
            plan[n] = {"map1": None, "map2": None, "crop": (0, 0, w, h),
                       "K": intr[n]["K"].copy()}
        return plan

    raw = {}
    for n in names:
        K, dist = intr[n]["K"], intr[n]["dist"]
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
        x, y, rw, rh = roi
        if rw <= 0 or rh <= 0:
            x, y, rw, rh = 0, 0, w, h
        raw[n] = {"newK": newK, "roi": (x, y, rw, rh)}

    if mode == "full":
        for n in names:
            raw[n]["crop"] = (0, 0, w, h)
    elif mode == "roi":
        for n in names:
            raw[n]["crop"] = raw[n]["roi"]
    else:
        cw = min(raw[n]["roi"][2] for n in names)
        ch = min(raw[n]["roi"][3] for n in names)
        for n in names:
            x, y, rw, rh = raw[n]["roi"]
            raw[n]["crop"] = (x + (rw - cw) // 2, y + (rh - ch) // 2, cw, ch)

    for n in names:
        K, dist = intr[n]["K"], intr[n]["dist"]
        newK = raw[n]["newK"].copy()
        x, y, cw, ch = raw[n]["crop"]
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, raw[n]["newK"], (w, h), cv2.CV_16SC2)
        newK[0, 2] -= x
        newK[1, 2] -= y
        plan[n] = {"map1": map1, "map2": map2, "crop": (x, y, cw, ch), "K": newK}
    return plan


def scale_K(K, s):
    if s == 1.0:
        return K.copy()
    Ks = K.copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


# --------------------------------------------------------------------------
# depth-map geometry
# --------------------------------------------------------------------------

def points_camera(depth, K):
    """Depth is Z along the optical axis, not distance along the ray."""
    return pixel_rays(depth.shape, K) * depth[..., None]


def edge_mask(depth, rel_thresh, dilate=1):
    d = np.asarray(depth, np.float64)
    finite = np.isfinite(d) & (d > 0)
    filled = np.where(finite, d, np.nanmedian(d[finite]) if finite.any() else 1.0)
    gy, gx = np.gradient(filled)
    grad = np.hypot(gx, gy) / np.maximum(filled, 1e-6)
    smooth = (grad <= rel_thresh) & finite
    if dilate > 0:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        smooth = cv2.erode(smooth.astype(np.uint8), k).astype(bool)
    return smooth


def normals_from_points(pts_cam):
    du = np.zeros_like(pts_cam)
    dv = np.zeros_like(pts_cam)
    du[:, 1:-1] = pts_cam[:, 2:] - pts_cam[:, :-2]
    dv[1:-1, :] = pts_cam[2:, :] - pts_cam[:-2, :]
    n = np.cross(du, dv)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def incidence_mask(pts_cam, max_deg):
    n = normals_from_points(pts_cam)
    rays = pts_cam / np.maximum(
        np.linalg.norm(pts_cam, axis=-1, keepdims=True), 1e-12)
    cos = np.abs(np.sum(n * rays, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    ang[~np.isfinite(ang)] = 90.0
    return ang <= max_deg, ang


def to_world(pts_cam, E):
    R, t = rt(E)
    return (pts_cam - t) @ R          # R^T (x_cam - t)


def to_o3d(points, colors=None, rgb01=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb01 is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(rgb01, (len(points), 1)))
    elif colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, np.float64) / 255.0)
    return pcd


# --------------------------------------------------------------------------
# layering diagnostics
# --------------------------------------------------------------------------

def fit_plane_ransac(points, thresh=0.010, iters=400, seed=0, subsample=60000):
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, np.float64)
    if len(pts) < 3:
        return None, None
    sample = pts
    if len(sample) > subsample:
        sample = sample[rng.choice(len(sample), subsample, replace=False)]
    best_n, best_d, best_count = None, None, -1
    for _ in range(iters):
        idx = rng.choice(len(sample), 3, replace=False)
        a, b, c = sample[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        d = float(n @ a)
        count = int((np.abs(sample @ n - d) <= thresh).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        return None, None
    for _ in range(3):
        inl = np.abs(pts @ best_n - best_d) <= thresh
        if inl.sum() < 3:
            break
        sel = pts[inl]
        c = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
        best_n = vt[-1] / np.linalg.norm(vt[-1])
        best_d = float(best_n @ c)
    return best_n, best_d


def layer_report(per_view_points, centres, band_m, above_m):
    """
    How far apart the four views place the same surface.

    The dominant plane of the combined cloud is the belt. Each view's median
    signed offset from it is that view's sheet, and the spread across views is
    the layering, in millimetres, at belt level. The same is then done for
    everything standing above the belt, which is where it matters, because the
    per-view loss of relief only separates the sheets further as height grows.

    This is a per-view measurement and says nothing about absolute accuracy:
    all four views can agree perfectly and all four be wrong together.
    """
    allpts = np.vstack([p for p in per_view_points.values() if len(p)])
    if len(allpts) < 100:
        return None
    n, d = fit_plane_ransac(allpts, thresh=0.010)
    if n is None:
        return None
    if float(np.mean(centres @ n - d)) < 0:
        n, d = -n, -d                     # height positive towards the cameras

    out = {"plane_normal": n.tolist(), "plane_offset_m": float(d),
           "belt": {}, "above": {}}
    for name, pts in per_view_points.items():
        if not len(pts):
            continue
        h = pts @ n - d
        near = np.abs(h) <= band_m
        if near.sum() >= 50:
            out["belt"][name] = round(float(np.median(h[near])) * 1e3, 2)
        hi = h > above_m
        if hi.sum() >= 50:
            out["above"][name] = round(float(np.median(h[hi])) * 1e3, 2)
    for key in ("belt", "above"):
        v = list(out[key].values())
        out[key + "_spread_mm"] = round(max(v) - min(v), 2) if len(v) > 1 else None
    return out


def print_layer_report(rep):
    if rep is None:
        print("[layer] no dominant plane found")
        return
    belt = "  ".join(f"{k}{v:+7.1f}" for k, v in rep["belt"].items())
    print(f"[layer] belt      {belt}   spread "
          f"{rep['belt_spread_mm'] if rep['belt_spread_mm'] is not None else 0:.1f} mm")
    if rep["above"]:
        above = "  ".join(f"{k}{v:+7.1f}" for k, v in rep["above"].items())
        print(f"[layer] above     {above}   spread "
              f"{rep['above_spread_mm'] if rep['above_spread_mm'] is not None else 0:.1f} mm")
    print("[layer] spread is how far apart the views place the same surface. "
          "It is what makes a cloud look layered, and it is the only part a "
          "per-view correction can remove.")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(model_id, device):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained(model_id)
    return m.to(device) if hasattr(m, "to") else m


def as_numpy(v):
    if v is None:
        return None
    if torch is not None and isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def run_inference(model, frames, Ks_in, Es, args):
    """One DA3 pass, with the None-returning fields filled from the inputs."""
    t0 = time.perf_counter()
    if args.mode == "prior":
        pred = model.inference(frames, intrinsics=Ks_in, extrinsics=Es,
                               align_to_input_ext_scale=args.align_ext_scale,
                               process_res=args.process_res)
    else:
        pred = model.inference(frames, process_res=args.process_res)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3

    depth = as_numpy(pred.depth)
    conf = as_numpy(getattr(pred, "conf", None))
    K_out = as_numpy(getattr(pred, "intrinsics", None))
    E_out = as_numpy(getattr(pred, "extrinsics", None))
    rgb_proc = as_numpy(getattr(pred, "processed_images", None))
    is_metric = int(getattr(pred, "is_metric", 0))

    dh, dw = depth.shape[-2:]
    if K_out is None:
        K_out = np.stack([scale_K(Ks_in[i], dw / frames[i].shape[1])
                          for i in range(len(frames))])
    if E_out is None:
        E_out = Es
    if rgb_proc is None:
        rgb_proc = np.stack([cv2.resize(f, (dw, dh), interpolation=cv2.INTER_AREA)
                             for f in frames])

    return {"depth": depth, "conf": conf, "K": K_out, "E": E_out,
            "rgb": rgb_proc, "is_metric": is_metric, "ms": ms,
            "shape": (int(dh), int(dw))}


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------

def set_node(nodemap, name, value, log):
    try:
        nodemap[name].value = value
    except Exception as exc:  # noqa: BLE001
        log.append(f"{name}: not set ({exc})")
        return False
    log.append(f"{name}: {value}")
    return True


def buffer_to_array(item, bpp):
    total = item.width * item.height * bpp
    raw = (ctypes.c_ubyte * total).from_address(ctypes.addressof(item.pbytes))
    shape = (item.height, item.width, bpp) if bpp > 1 else (item.height, item.width)
    return np.ndarray(buffer=raw, dtype=np.uint8, shape=shape).copy()


def to_rgb(frame, family, bayer_code):
    if family == "rgb":
        return frame
    if family == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(frame, bayer_code)


class CameraWorker(threading.Thread):
    """Acquire one camera, rectify via precomputed maps, publish the latest."""

    def __init__(self, name, device, args, plan, stop):
        super().__init__(name=f"acq-{name}", daemon=True)
        self.cam_name = name
        self.device = device
        self.args = args
        self.stop = stop
        self.map1 = plan["map1"]
        self.map2 = plan["map2"]
        self.crop = plan["crop"]

        self.bpp, self.family = PIXEL_FORMATS[args.pixel_format]
        self.bayer_code = getattr(cv2, args.bayer_code, cv2.COLOR_BayerRG2RGB)

        self._lock = threading.Lock()
        self._frame = None
        self._host_t = 0.0
        self._dev_ts = 0
        self._seq = 0

        self.frames = 0
        self.incomplete = 0
        self.failures = 0
        self.wait_ms = []
        self.rectify_ms = []
        self.stamps = deque(maxlen=120)
        self.errors = []

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._host_t, self._dev_ts, self._seq

    def run(self):
        while not self.stop.is_set():
            item = None
            try:
                t0 = time.perf_counter()
                item = self.device.get_buffer(timeout=self.args.buffer_timeout_ms)
                t1 = time.perf_counter()

                if item.is_incomplete:
                    self.incomplete += 1
                    continue

                dev_ts = int(getattr(item, "timestamp_ns", 0) or 0)
                rgb = to_rgb(buffer_to_array(item, self.bpp),
                             self.family, self.bayer_code)

                if self.map1 is not None:
                    rgb = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR)
                x, y, w, h = self.crop
                rgb = rgb[y:y + h, x:x + w]
                t2 = time.perf_counter()

                self.wait_ms.append((t1 - t0) * 1e3)
                self.rectify_ms.append((t2 - t1) * 1e3)
                self.stamps.append(t2)
                self.frames += 1

                with self._lock:
                    self._frame = rgb
                    self._host_t = t2
                    self._dev_ts = dev_ts
                    self._seq += 1
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                if len(self.errors) < 10:
                    self.errors.append(repr(exc))
                if self.failures > 50:
                    self.errors.append("acquisition thread giving up")
                    break
            finally:
                if item is not None:
                    try:
                        self.device.requeue_buffer(item)
                    except Exception:  # noqa: BLE001
                        pass

    def fps(self):
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0

    def summary(self, elapsed):
        return {"frames_acquired": self.frames,
                "incomplete_buffers": self.incomplete,
                "acquisition_failures": self.failures,
                "mean_fps": round(self.frames / elapsed, 3) if elapsed > 0 else 0.0,
                "buffer_wait": stage_stats(self.wait_ms),
                "rectify": stage_stats(self.rectify_ms),
                "errors": self.errors}


def stage_stats(vals):
    if not vals:
        return None
    ordered = sorted(vals)
    i95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {"n": len(vals),
            "mean_ms": round(statistics.fmean(vals), 2),
            "median_ms": round(statistics.median(vals), 2),
            "p95_ms": round(ordered[i95], 2),
            "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2)}


def plan_link(args, n_cameras):
    """Size the frame rate and the per-camera throughput limit to the uplink."""
    bpp = PIXEL_FORMATS[args.pixel_format][0]
    frame_mb = args.width * args.height * bpp / 1e6
    per_cam_cap = GIGE_MB_S / frame_mb
    budget_cap = args.link_budget_mb_s / (frame_mb * n_cameras)

    if args.fps > 0:
        fps = args.fps
        source = "requested"
    else:
        fps = max(1.0, round(min(per_cam_cap, budget_cap) * 0.9, 1))
        source = "derived from the link budget"

    aggregate = frame_mb * fps * n_cameras
    limit = args.throughput_limit
    if limit <= 0 and args.link_limit:
        limit = int(args.link_budget_mb_s * 1e6 / n_cameras)

    return {"frame_mb": frame_mb, "fps": fps, "fps_source": source,
            "aggregate_mb_s": aggregate, "per_cam_cap_fps": per_cam_cap,
            "budget_cap_fps": budget_cap, "throughput_limit": limit}


def configure(device, args, link, log):
    nm = device.nodemap
    set_node(nm, "Width", args.width, log)
    set_node(nm, "Height", args.height, log)
    set_node(nm, "PixelFormat", args.pixel_format, log)
    set_node(nm, "AcquisitionMode", "Continuous", log)
    set_node(nm, "TriggerMode", "Off", log)
    set_node(nm, "ExposureAuto", "Off", log)
    set_node(nm, "ExposureTime", args.exposure_us, log)
    set_node(nm, "GainAuto", "Off", log)
    set_node(nm, "Gain", args.gain_db, log)
    if link["fps"] > 0:
        set_node(nm, "AcquisitionFrameRateEnable", True, log)
        set_node(nm, "AcquisitionFrameRate", float(link["fps"]), log)
    if link["throughput_limit"] > 0:
        set_node(nm, "DeviceLinkThroughputLimitMode", "On", log)
        set_node(nm, "DeviceLinkThroughputLimit", int(link["throughput_limit"]), log)
    if args.ptp:
        set_node(nm, "PtpEnable", True, log)

    st = device.tl_stream_nodemap
    set_node(st, "StreamAutoNegotiatePacketSize", True, log)
    set_node(st, "StreamPacketResendEnable", True, log)
    set_node(st, "StreamBufferHandlingMode", "NewestOnly", log)


def wait_for_ptp(devices, names, timeout_s=15.0):
    states = ["Unavailable"] * len(devices)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        states = []
        for d in devices:
            try:
                states.append(str(d.nodemap["PtpStatus"].value))
            except Exception:  # noqa: BLE001
                states.append("Unavailable")
        if all(s in ("Master", "Slave") for s in states):
            return dict(zip(names, states)), True
        time.sleep(0.5)
    return dict(zip(names, states)), False


# --------------------------------------------------------------------------
# triggering
# --------------------------------------------------------------------------

class SettleTrigger:
    """
    Fire once the reference view stops changing, re-arm once it moves again.

    A single noisy frame decays the static counter rather than resetting it,
    so one flicker does not restart the whole settle period.
    """

    def __init__(self, thresh, need, rearm):
        self.thresh = thresh
        self.need = need
        self.rearm = rearm
        self.prev = None
        self.static_count = 0
        self.armed = True
        self.last_diff = float("inf")

    def update(self, rgb):
        small = cv2.resize(rgb, (160, 128), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self.prev is None:
            self.prev = grey
            return False
        diff = float(np.mean(np.abs(grey - self.prev)))
        self.prev = grey
        self.last_diff = diff

        if diff > self.rearm:
            self.armed = True
            self.static_count = 0
            return False
        if diff <= self.thresh:
            self.static_count += 1
        else:
            self.static_count = max(0, self.static_count - 1)
        if self.armed and self.static_count >= self.need:
            self.armed = False
            self.static_count = 0
            return True
        return False

    def state(self):
        return (f"diff={self.last_diff:.2f} static={self.static_count}/"
                f"{self.need} armed={'yes' if self.armed else 'no'}")


# --------------------------------------------------------------------------
# per-capture processing
# --------------------------------------------------------------------------

def backproject(names, depth, conf, K_out, E_out, rgb_proc, args):
    clouds, tinted, per_cam, world_pts = {}, [], [], {}
    for i, n in enumerate(names):
        d = depth[i].astype(np.float64)
        c = conf[i] if conf is not None else None
        K_i = K_out[i]

        valid = np.isfinite(d) & (d > 0)
        drops = {}

        thr = None
        if args.conf_percentile > 0 and c is not None and valid.any():
            thr = float(np.percentile(c[valid], args.conf_percentile))
            before = int(valid.sum())
            valid &= c >= thr
            drops["conf"] = before - int(valid.sum())

        if args.edge_thresh > 0:
            before = int(valid.sum())
            valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
            drops["edge"] = before - int(valid.sum())

        safe = np.where(np.isfinite(d), d, 0.0)
        pts_cam = points_camera(safe, K_i)

        inc_med = None
        if args.max_incidence < 90:
            ok, ang = incidence_mask(pts_cam, args.max_incidence)
            inc_med = float(np.median(ang[valid])) if valid.any() else None
            before = int(valid.sum())
            valid &= ok
            drops["grazing"] = before - int(valid.sum())

        pts_world = to_world(pts_cam[valid], E_out[i])
        world_pts[n] = pts_world
        cols = rgb_proc[i][valid] if rgb_proc is not None else None
        clouds[n] = to_o3d(pts_world, colors=cols)
        if args.colour_by_camera:
            tinted.append(to_o3d(pts_world, rgb01=PALETTE[i % len(PALETTE)]))

        dv = d[valid]
        per_cam.append({
            "camera": n,
            "median_depth_m": float(np.median(dv)) if dv.size else None,
            "depth_p05_m": float(np.percentile(dv, 5)) if dv.size else None,
            "depth_p95_m": float(np.percentile(dv, 95)) if dv.size else None,
            "incidence_median_deg": inc_med,
            "n_points": int(valid.sum()),
            "conf_threshold": thr,
            "dropped": drops,
        })
    return clouds, tinted, per_cam, world_pts


def fuse_tsdf(names, depth, K_out, E_out, rgb_proc, args):
    """
    Integrate the four views into one surface instead of stacking four.

    A union of point clouds preserves every disagreement between the views as
    a separate sheet, however fine the voxel. A TSDF averages the signed
    distances, so views that disagree by less than the truncation distance
    resolve to a single surface. Views that disagree by MORE than it produce
    two surfaces anyway, which is why the truncation has to exceed the
    residual and why this is worth doing only after --correction has removed
    the systematic part of that residual.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=max(args.voxel, 1e-4),
        sdf_trunc=args.tsdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i, n in enumerate(names):
        d = np.ascontiguousarray(depth[i], np.float32)
        d[~np.isfinite(d)] = 0.0
        colour = np.ascontiguousarray(rgb_proc[i], np.uint8)
        h, w = d.shape
        K = K_out[i]
        intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(colour), o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=args.tsdf_depth_trunc,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, np.asarray(E_out[i], np.float64))
    return volume.extract_point_cloud()


def segmentation_input(names, world_pts):
    """The points the segmentation is given, with the camera each came from.

    These are the SAME filtered per-view points that go into the fusion, so
    what is segmented is what is displayed. Back-projecting a second time with
    different filtering would let the overlays and the records disagree about
    which pixels a pose came from, and there would be nothing in the output to
    show that they had.

    The source array is what makes the per-view reconciliation in
    face_consensus.py possible. Without it every point claims the same camera
    and the inter-view figures come back as perfect agreement, which is not the
    same thing as the cameras agreeing.
    """
    pts, src = [], []
    for i, n in enumerate(names):
        p = world_pts.get(n)
        if p is None or not len(p):
            continue
        pts.append(p)
        src.append(np.full(len(p), i, dtype=np.int32))
    if not pts:
        return None, None
    return np.concatenate(pts), np.concatenate(src)


def append_box_log(path: Path, rows):
    """Every parcel of every capture, in one file.

    A repeatability sweep is a comparison across captures, so the rows have to
    sit together. Per-capture boxes.csv files would have to be concatenated
    afterwards, and the capture column is the only thing tying a row back to
    the arrays that produced it.
    """
    if not rows:
        return
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=bs.BOX_CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def mosaic(tiles, width):
    resized = []
    for t in tiles:
        s = width / t.shape[1]
        resized.append(cv2.resize(t, (width, max(1, int(round(t.shape[0] * s))))))
    h = max(t.shape[0] for t in resized)
    resized = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
               if t.shape[0] < h else t for t in resized]
    rows = [np.hstack(resized[i:i + 2]) for i in range(0, len(resized), 2)]
    full = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], full - r.shape[1], 3), np.uint8)])
            if r.shape[1] < full else r for r in rows]
    return np.vstack(rows)


def with_banner(image, lines, colour):
    bar = np.zeros((26 * len(lines) + 8, image.shape[1], 3), np.uint8)
    for j, line in enumerate(lines):
        cv2.putText(bar, line, (8, 20 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def depth_preview(names, depth, workers, args, info_lines):
    tiles = []
    for i, n in enumerate(names):
        d = depth[i].astype(np.float32)
        finite = np.isfinite(d) & (d > 0)
        if finite.any():
            lo, hi = np.percentile(d[finite], [2, 98])
            norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            norm = np.zeros_like(d)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~finite] = 0
        label = f"{n}  {workers[i].fps():.1f} fps in"
        for org, col, th in (((8, 22), (0, 0, 0), 3), ((8, 22), (255, 255, 255), 1)):
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        col, th, cv2.LINE_AA)
        tiles.append(img)
    return with_banner(mosaic(tiles, args.preview_width), info_lines, (0, 255, 0))


# --------------------------------------------------------------------------
# start-up reporting and self-test
# --------------------------------------------------------------------------

def report_calibration(intr, ext, names, lens, plan, args, link):
    print(f"[lens ] {lens}  ({intr[names[0]]['lens_mm']} mm)")
    for n in names:
        K = intr[n]["K"]
        print(f"[intr ] {n:<7} fx={K[0,0]:8.2f}  cx={K[0,2]:7.2f}  "
              f"cy={K[1,2]:7.2f}  rms={intr[n]['rms_px']:.3f} px  "
              f"fx/expected={intr[n]['fx_over_expected']:.4f}")
    print()
    for n in names:
        E = ext[n]["E"]
        c = camera_centre(E)
        claimed = ext[n]["baseline_m"]
        measured = float(np.linalg.norm(c))
        flag = ""
        if claimed is not None and abs(measured - claimed) > 1e-4:
            flag = f"  MISMATCH vs baseline_m {claimed:.6f}"
        print(f"[pose ] {n:<7} C={np.round(c, 4)}  |C|={measured:.4f} m  "
              f"rms={ext[n]['rms_px']:.3f} px{flag}")
    print()
    sizes = {n: (plan[n]['crop'][2], plan[n]['crop'][3]) for n in names}
    for n in names:
        K = plan[n]["K"]
        w, h = sizes[n]
        fov_x = 2 * np.degrees(np.arctan(w / (2 * K[0, 0])))
        fov_y = 2 * np.degrees(np.arctan(h / (2 * K[1, 1])))
        print(f"[rect ] {n:<7} {w}x{h}  fx={K[0,0]:8.2f}  "
              f"cx={K[0,2]:7.2f} cy={K[1,2]:7.2f}  fov={fov_x:.1f}x{fov_y:.1f} deg")
    if len(set(sizes.values())) > 1:
        print("[WARN] the rectified views do not share a size. DA3 will "
              "centre-crop the batch to the smallest common dimension, "
              "shifting the principal point. Use --rect-mode common.")
    else:
        print(f"[rect ] all views share {next(iter(sizes.values()))}, no batch crop")

    print(f"[link ] {args.pixel_format} {link['frame_mb']:.2f} MB/frame  "
          f"per-camera GigE ceiling {link['per_cam_cap_fps']:.1f} fps  "
          f"uplink ceiling {link['budget_cap_fps']:.1f} fps")
    print(f"[link ] running at {link['fps']:.1f} fps ({link['fps_source']}), "
          f"{link['aggregate_mb_s']:.0f} MB/s aggregate, per-camera limit "
          + (f"{link['throughput_limit'] / 1e6:.0f} MB/s"
             if link["throughput_limit"] > 0 else "not set"))
    if link["aggregate_mb_s"] > args.link_budget_mb_s:
        print("[WARN] that exceeds --link-budget-mb-s. Expect incomplete "
              "buffers and a large arrival spread. Lower --fps or switch to "
              "--pixel-format BayerRG8.")


def report_correction(corr, meta, names, args):
    print()
    shared = (meta or {}).get("shared") or {}
    if shared:
        print(f"[corr ] rig-wide: keeps {shared.get('p', float('nan')) * 100:.1f} "
              f"per cent of every rise, so alpha={shared.get('alpha', 0):.6f} "
              f"beta={shared.get('beta', 0) * 1e3:+.2f} mm")
    for n in names:
        rec = corr.get(n)
        if rec is None:
            continue
        sig = ("" if rec["sigma_mm"] is None
               else f"  delayering sigma {rec['sigma_mm']:.1f} mm")
        print(f"[corr ] {n:<7} alpha={rec['alpha']:.6f}  "
              f"beta={rec['beta'] * 1e3:+7.2f} mm  deck standoff "
              f"{-rec['d_cam']:.4f} m  slope "
              f"{'fitted' if rec['slope_fitted'] else 'held at the rig mean'}"
              f"{sig}")
    unc = (meta or {}).get("height_uncertainty")
    if unc:
        print(f"[corr ] absolute height uncertainty after this correction: "
              f"{unc['constant_mm']:.1f} mm + "
              f"{unc['per_mm_of_height'] * 1000:.1f} mm per metre of height, "
              f"which is {unc.get('at_predict_height_mm', float('nan')):.1f} mm "
              f"at parcel-top height.")
        print("[corr ] that term is shared by all four views, so it is NOT "
              "reduced by view consensus. Quote it downstream rather than "
              "anything derived from inter-view agreement.")
    lay = (meta or {}).get("layering") or {}
    if lay.get("before") and lay.get("after"):
        for label in lay["before"]:
            b = lay["before"][label].get("before_median_mm")
            a = (lay["after"].get(label) or {}).get("after_median_mm")
            if b is not None and a is not None:
                print(f"[corr ] measured layering at '{label}': {b:.1f} mm "
                      f"before, {a:.1f} mm after")
    if not args.view_correction:
        print("[corr ] --no-view-correction: the per-view deviations are "
              "being discarded and only the rig-wide term applied, so the "
              "layering will NOT improve.")


def report_segmentation_setup(args, seg_p, geom_p):
    """What will happen on a capture, said once, before anything happens."""
    print()
    if bs is None:
        print(f"[seg  ] box_segment.py not importable ({_SEG_IMPORT_ERROR}); "
              f"captures will be clouds only")
        return
    if not args.segment:
        print("[seg  ] segmentation off (--no-segment); captures will be "
              "clouds only")
        return
    print(f"[seg  ] on. Every capture is segmented and written to "
          f"frame_NNNNN/segmentation")
    if seg_p.plane_distance is None:
        print("[WARN] no --seg-plane-distance given. The conveyor is fitted as "
              "the largest coplanar surface, which in a four-camera rig is "
              "usually the FLOOR: the deck then reads as a 0.8 m tall object "
              "and fragments into dozens of false parcels. Read the deck "
              "distance off the median depths printed at warm-up and pass it.")
    else:
        print(f"[seg  ] conveyor accepted between "
              f"{seg_p.plane_distance[0]:.3f} and "
              f"{seg_p.plane_distance[1]:.3f} m from the reference camera")
    if seg_p.belt_file:
        print(f"[seg  ] belt footprint frozen to {seg_p.belt_file}")
    else:
        print("[seg  ] no --seg-belt-file, so the belt footprint is re-solved "
              "every capture and the reported dimensions move with it. Freeze "
              "it once on a lightly loaded belt.")
    if bg is None:
        print(f"[geom ] box_geometry.py not importable ({_GEOM_IMPORT_ERROR}); "
              f"no box geometry will be written")
    elif not args.geom:
        print("[geom ] off (--no-geom)")
    else:
        parts = [k for k, on in (("boxes_mesh.ply", geom_p.mesh),
                                 ("boxes_lines.ply", geom_p.lines),
                                 ("scene_boxed.ply", geom_p.combined)) if on]
        print(f"[geom ] on, writing {', '.join(parts)}"
              + (f", every {geom_p.every} captures" if geom_p.every > 1 else ""))
        if geom_p.combined:
            print("[geom ] scene_boxed.ply samples the box mesh to point "
                  "density, which costs hundreds of milliseconds and scales "
                  "with the size of the scene. Drop --geom-combined for a "
                  "continuous run.")


def geometry_self_test():
    rng = np.random.default_rng(0)
    K = np.array([[3479.6, 0.0, 1200.0],
                  [0.0, 3479.6, 1000.0],
                  [0.0, 0.0, 1.0]])
    R = cv2.Rodrigues(np.array([0.03, -0.02, 0.7]))[0]
    t = np.array([-0.27, 0.12, 0.001])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t

    world = rng.uniform([-0.6, -0.5, 2.6], [0.6, 0.5, 3.4], size=(400, 3))
    cam = world @ R.T + t
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]

    rays = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0],
                     (uv[:, 1] - K[1, 2]) / K[1, 1],
                     np.ones(len(uv))], axis=-1)
    recovered = to_world(rays * cam[:, 2:3], E)
    err = float(np.abs(recovered - world).max())
    C_err = float(np.abs(camera_centre(E) - (-R.T @ t)).max())

    depth = np.full((32, 40), 3.0)
    depth[:, 20:] = 3.5
    pts = points_camera(depth, K)
    z_err = float(np.abs(pts[..., 2] - depth).max())
    smooth = edge_mask(depth, 0.02, 1)
    ok_ang, _ = incidence_mask(pts, 85.0)

    print(f"[test ] world round-trip max error   {err:.3e} m")
    print(f"[test ] camera centre max error      {C_err:.3e} m")
    print(f"[test ] depth is Z, max error        {z_err:.3e} m")
    print(f"[test ] edge filter rejects          {int((~smooth).sum())} of "
          f"{smooth.size} pixels at the step")
    print(f"[test ] incidence filter keeps       {int(ok_ang.sum())} of "
          f"{ok_ang.size} pixels on a fronto-parallel plane")

    # ---- depth correction ------------------------------------------
    # A known height map is imposed on a plane and a box top, and the
    # correction has to invert it exactly. The identity must also be the
    # identity, which is what stops a silently mis-parameterised file from
    # moving a cloud that needed no moving.
    Kd = np.array([[430.0, 0.0, 252.0], [0.0, 430.0, 210.0], [0.0, 0.0, 1.0]])
    n_cam = np.array([-0.0339, -0.0185, -0.9993])
    n_cam /= np.linalg.norm(n_cam)
    d_cam = -3.0671
    alpha, beta = 1.0 / 0.8280, 0.0011 / 0.8280
    shape = (60, 80)
    nr = pixel_rays(shape, Kd) @ n_cam
    h_true = np.zeros(shape)
    h_true[:, 40:] = 0.218
    z_true = (h_true + d_cam) / nr
    h_meas = (h_true - beta) / alpha
    z_meas = (h_meas + d_cam) / nr
    z_back = apply_height_correction(z_meas, Kd, n_cam, d_cam, alpha, beta)
    corr_err = float(np.nanmax(np.abs(z_back - z_true)))
    ident = apply_height_correction(z_meas, Kd, n_cam, d_cam, 1.0, 0.0)
    ident_err = float(np.nanmax(np.abs(ident - z_meas)))
    print(f"[test ] height correction round trip {corr_err:.3e} m")
    print(f"[test ] identity correction moves    {ident_err:.3e} m")

    # ---- the correction must collapse a synthetic layering ---------
    # The four per-view offsets and losses of relief measured on this rig,
    # imposed on one plane and one box top. The correction is built from the
    # same numbers and the spread has to fall to nothing.
    off = {"left": 6.8, "center": -0.6, "right": 0.8, "top": -10.9}
    k = {"left": 0.1618, "center": 0.1792, "right": 0.1554, "top": 0.1810}
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 60), np.linspace(-0.4, 0.4, 60))
    views, fixed = {}, {}
    for n in off:
        p_v = 1.0 - k[n]
        q_v = off[n] * 1e-3
        belt = np.stack([gx.ravel(), gy.ravel(),
                         np.full(gx.size, q_v)], axis=1)
        top = np.stack([gx.ravel() * 0.3, gy.ravel() * 0.3,
                        np.full(gx.size, 0.218 * p_v + q_v)], axis=1)
        views[n] = np.vstack([belt, top])
        h = views[n][:, 2]
        fixed[n] = np.column_stack([views[n][:, 0], views[n][:, 1],
                                    (h - q_v) / p_v])
    centres = np.array([[0.0, 0.0, 3.0]] * 4)
    rep = layer_report(views, centres, 0.040, 0.060)
    rep2 = layer_report(fixed, centres, 0.040, 0.060)
    belt_spread = rep["belt_spread_mm"]
    above_spread = rep["above_spread_mm"]
    fixed_above = rep2["above_spread_mm"]
    print(f"[test ] layer report: belt spread {belt_spread:.1f} mm "
          f"(expected 17.7), above {above_spread:.1f} mm (expected 21.9)")
    print(f"[test ] after the per-view correction, above spread "
          f"{fixed_above:.2f} mm")

    trig = SettleTrigger(1.5, 4, 4.0)
    base = np.zeros((128, 160, 3), np.uint8)
    fires = [trig.update(base) for _ in range(6)]
    settle_ok = fires == [False, False, False, False, True, False]
    print(f"[test ] settle trigger fires on frame "
          f"{fires.index(True) + 1 if True in fires else 'never'} of a static "
          f"sequence")

    # ---- the segmentation and geometry path ------------------------
    # A synthetic deck with two boxes on it, back-projected from four
    # positions, is enough to prove the whole capture path runs and returns
    # cuboids of the right size. It says nothing about the real rig.
    seg_ok = "skipped"
    if bs is not None:
        deck = np.random.default_rng(1).uniform([-0.6, -0.4], [0.6, 0.4],
                                                size=(40000, 2))
        pts = [np.column_stack([deck[:, 0], deck[:, 1],
                                np.full(len(deck), 3.070)])]
        for cx, cy, hgt in ((-0.25, 0.0, 0.200), (0.25, 0.0, 0.150)):
            g = np.random.default_rng(2).uniform([-0.15, -0.15],
                                                 [0.15, 0.15], size=(9000, 2))
            pts.append(np.column_stack([g[:, 0] + cx, g[:, 1] + cy,
                                        np.full(len(g), 3.070 - hgt)]))
        allp = np.vstack(pts)
        src = np.tile(np.arange(4, dtype=np.int32),
                      len(allp) // 4 + 1)[:len(allp)]
        sp = bs.SegParams(voxel=0.006, plane_distance=(3.0, 3.15),
                          max_height=0.40, belt_only=False,
                          face_per_view=False, min_views_pickable=1)
        res = bs.segment_boxes(allp, src, ["left", "center", "right", "top"],
                               sp)
        dims = sorted(round(b["dimensions_m"]["height"] * 1e3)
                      for b in res.get("boxes", []))
        seg_ok = f"{len(res.get('boxes', []))} boxes, heights {dims} mm"
        print(f"[test ] segmentation path: {seg_ok} (expected 2 boxes at "
              f"[150, 200] mm)")
        if bg is not None:
            gp = bg.GeomParams(enabled=True)
            mesh = bg.build_box_mesh(res, gp)
            print(f"[test ] box geometry: {len(mesh.triangles)} triangles, "
                  f"{len(bg.build_line_set(res, gp).lines)} edges")

    bad = (err > 1e-9 or C_err > 1e-12 or z_err > 1e-12 or not settle_ok
           or corr_err > 1e-9 or ident_err > 1e-12
           or abs(belt_spread - 17.7) > 0.5
           or abs(above_spread - 21.9) > 0.5 or fixed_above > 0.5)
    print("[test ] geometry FAIL" if bad else "[test ] geometry ok")
    return 1 if bad else 0


def wait_for_first_frames(workers, timeout_s):
    """Block until every camera has published, so warm-up has real data."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        snap = [w.latest() for w in workers]
        if all(s is not None for s in snap):
            return [s[0] for s in snap]
        if all(not w.is_alive() for w in workers):
            return None
        time.sleep(0.05)
    return None


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    names = list(args.cameras)
    unknown = [n for n in names if n not in CAMERAS]
    if unknown:
        raise SystemExit(f"unknown camera name(s): {', '.join(unknown)}")
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} is not among {names}")
    if args.process_res % 14:
        print(f"[warn ] process_res {args.process_res} is not a multiple of 14; "
              f"DA3 will round it")
    if (args.capture_trigger == "key" and not args.preview
            and not args.self_test and not sys.stdin.isatty()):
        raise SystemExit("--capture-trigger key needs either the preview "
                         "window or a terminal on stdin to receive the "
                         "keypress. Use --capture-trigger settle or continuous "
                         "when running detached.")
    if args.dead_grid_flag:
        print("[note ] --no-affine-grid is accepted and ignored. The residual "
              "disparity grid no longer exists; the correction is a linear "
              "map on height above the deck and carries no grid.")

    seg_p = bs.params_from_args(args) if (bs is not None and args.segment) else None
    geom_p = None
    if bg is not None and args.geom and seg_p is not None:
        geom_p = bg.replace_enabled(bg.params_from_args(args, seg_params=seg_p),
                                    True)

    # ---- calibration --------------------------------------------------
    have_calib = args.calib_dir.exists()
    if not have_calib and not args.self_test:
        raise SystemExit(f"--calib-dir {args.calib_dir} does not exist")

    intr = ext = plan = link = None
    lens = ""
    corr, corr_meta = None, None
    if have_calib:
        intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
        ext = load_extrinsics(args.calib_dir, names, args.reference)
        lens = check_lens(intr, ext, names, args.expect_lens)
        check_serials(intr, ext, names)

        for n in names:
            want = (args.width, args.height)
            if intr[n]["size"] != want:
                raise SystemExit(f"{n}: calibrated at {intr[n]['size']} but the "
                                 f"cameras are being configured for {want}")

        plan = build_rect_plan(intr, names, (args.width, args.height),
                               args.rect_mode, args.undistort)
        Ks_rect = {n: plan[n]["K"] for n in names}
        Es = np.stack([ext[n]["E"] for n in names])
        link = plan_link(args, len(names))
        report_calibration(intr, ext, names, lens, plan, args, link)

        # ---- depth correction ----------------------------------------
        if args.correction is not None:
            corr, corr_meta = load_correction(args.correction, names, lens,
                                              args.view_correction)
            report_correction(corr, corr_meta, names, args)
            want = corr_meta.get("rectified_sizes") or {}
            for n in names:
                if n in want and list(want[n]) != [plan[n]["crop"][2],
                                                   plan[n]["crop"][3]]:
                    print(f"[WARN] {n}: the correction was solved with a "
                          f"{want[n]} rectified image but this run produces "
                          f"[{plan[n]['crop'][2]}, {plan[n]['crop'][3]}]. The "
                          f"deck plane is a 3D quantity and still applies, but "
                          f"the calibration has changed since the correction "
                          f"was fitted and it should be re-solved.")
            if args.mode == "noprior":
                print("[WARN] --mode noprior returns DA3's own extrinsics, so "
                      "the camera frames are not the calibrated ones and the "
                      "deck plane carried in the correction does not belong to "
                      "them. The correction will be wrong. Re-fit it in the "
                      "same mode it will be applied in.")
        else:
            print("\n[corr ] none supplied. Each view keeps its own depth "
                  "offset and its own loss of relief, so the same surface will "
                  "appear as several sheets. Pass --correction "
                  "depth_correction.json.")

        if args.fuse_mode == "tsdf" and corr is None:
            print("[WARN] --fuse-mode tsdf without --correction will average "
                  "four sheets that disagree systematically, producing one "
                  "surface in the wrong place. Correct first.")

        if seg_p is not None and corr is None:
            print("[WARN] segmentation without --correction measures parcel "
                  "tops that the four views place on four different sheets. "
                  "The heights and the inter-view figures in boxes.json will "
                  "be dominated by that, not by the parcels.")

        # A fixed sync window tighter than the frame period rejects
        # everything, which is what produced a climbing rejection counter and
        # no output. Derive it from the rate actually being run.
        if args.max_sync_ms < 0:
            args.max_sync_ms = round(0.75 * 1000.0 / max(link["fps"], 1e-6), 1)
            print(f"[sync ] --max-sync-ms auto set to {args.max_sync_ms:.1f} ms "
                  f"(0.75 of a {1000.0 / link['fps']:.0f} ms frame period)")

    report_segmentation_setup(args, seg_p, geom_p)

    if args.self_test:
        if not have_calib:
            print(f"[test ] --calib-dir {args.calib_dir} does not exist, so "
                  f"the calibration and rectification checks were skipped")
        print()
        return geometry_self_test()

    if o3d is None:
        raise SystemExit("open3d is required for fusion")

    from arena_api.system import system  # late import so --self-test needs no SDK

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (args.out_dir / "frames.jsonl").open("w")

    # ---- cameras ------------------------------------------------------
    infos = system.device_infos
    if not infos:
        raise SystemExit("no cameras found; check the Arena library path, the "
                         "NIC and that no stale session holds the devices")
    found = {i["serial"]: i for i in infos}
    missing = [n for n in names if CAMERAS[n] not in found]
    if missing:
        raise SystemExit(f"missing camera(s): {', '.join(missing)}; "
                         f"detected {sorted(found)}")

    devices = system.create_device([found[CAMERAS[n]] for n in names])

    config_log = {}
    stop = threading.Event()
    workers = []
    poller = KeyPoller()
    ptp_status = None
    exit_code = 0
    processed = 0
    inferences = 0
    rejected_sync = 0
    consecutive_rejects = 0
    accepted_sets = 0
    warmup_report = None
    last_layer = None
    last_seg = None
    last_boxes = None
    t_measure = None
    t_loop = time.perf_counter()
    stage_ms = {k: [] for k in ("assemble", "inference", "correct",
                                "backproject", "fuse", "write", "segment",
                                "geometry", "cycle")}
    cycle_stamps = deque(maxlen=30)
    spread_history = deque(maxlen=60)
    capture_key = ord(args.capture_key[0].lower())
    centres = np.stack([camera_centre(ext[n]["E"]) for n in names])

    try:
        for n, dev in zip(names, devices):
            log = []
            configure(dev, args, link, log)
            config_log[n] = log

        if args.ptp:
            ptp_status, locked = wait_for_ptp(devices, names)
            print(f"[ptp  ] {ptp_status}" + ("" if locked else "   NOT LOCKED"))
            if not locked and args.use_device_timestamp:
                print("[WARN] --use-device-timestamp without a PTP lock "
                      "compares clocks that are not related")

        for n, dev in zip(names, devices):
            dev.start_stream(args.num_buffers)
            workers.append(CameraWorker(n, dev, args, plan[n], stop))
        for w in workers:
            w.start()

        # ---- model ----------------------------------------------------
        print(f"[model] loading {args.model}")
        t0 = time.perf_counter()
        model = load_model(args.model, args.device)
        print(f"[model] ready in {time.perf_counter() - t0:.1f} s")

        # ---- warm-up, outside the loop --------------------------------
        if args.warmup > 0:
            print(f"[warm ] waiting for the first frame from every camera "
                  f"(up to {args.start_timeout:.0f} s)")
            first = wait_for_first_frames(workers, args.start_timeout)
            if first is None:
                raise SystemExit("cameras did not deliver a first frame; check "
                                 "the link budget, cabling and packet size")
            if args.input_scale != 1.0:
                s = args.input_scale
                first = [cv2.resize(f, (int(round(f.shape[1] * s)),
                                        int(round(f.shape[0] * s))),
                                    interpolation=cv2.INTER_AREA) for f in first]
                Ks_warm = np.stack([scale_K(Ks_rect[n], s) for n in names])
            else:
                Ks_warm = np.stack([Ks_rect[n] for n in names])

            for i in range(args.warmup):
                out = run_inference(model, first, Ks_warm, Es, args)
                print(f"[warm ] pass {i + 1}/{args.warmup}  {out['ms']:.0f} ms")
            dh, dw = out["shape"]
            fx_ratio = [float(out["K"][i][0, 0]
                              / (Ks_warm[i][0, 0] * dw / first[i].shape[1]))
                        for i in range(len(names))]
            meds = [float(np.median(out["depth"][i][np.isfinite(out["depth"][i])
                                                    & (out["depth"][i] > 0)]))
                    for i in range(len(names))]
            warmup_report = {"depth_shape": [dh, dw],
                             "is_metric": out["is_metric"],
                             "fx_ratio": [round(v, 4) for v in fx_ratio],
                             "median_depth_m": [round(v, 4) for v in meds]}
            print(f"[warm ] depth grid {dh}x{dw}  metric={out['is_metric']}")
            for n, r, m in zip(names, fx_ratio, meds):
                flag = "" if 0.98 < r < 1.02 else "   FX RATIO OUT OF BAND"
                print(f"[warm ] {n:<7} median depth {m:6.3f} m  "
                      f"fx_ratio {r:.4f}{flag}")
            if corr:
                _, shifts = correct_depths(out["depth"], names, corr, out["K"])
                for n in names:
                    if n in shifts:
                        print(f"[warm ] {n:<7} correction moves the median "
                              f"depth by {shifts[n]:+.1f} mm")
                print("[warm ] a correction that moves the belt by tens of "
                      "millimetres on a rig whose belt was already right is "
                      "the sign of a stale or mismatched correction file.")
            print("[warm ] check those medians against the deck standoff "
                  "before trusting any cloud")
            if seg_p is not None and seg_p.plane_distance is None:
                print(f"[warm ] those medians are the number "
                      f"--seg-plane-distance wants: pass a window of a few "
                      f"centimetres either side of the deck, not of the floor")
            t_measure = time.perf_counter()

        trigger = (SettleTrigger(args.settle_thresh, args.settle_frames,
                                 args.settle_rearm)
                   if args.capture_trigger == "settle" else None)
        ref_idx = names.index(args.reference)

        window = "DA3 live depth"
        if args.preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        else:
            poller.start()
            if args.capture_trigger == "key":
                print("[run  ] no preview window; the capture key is read from "
                      "the terminal instead")
        print(f"[run  ] capture trigger = {args.capture_trigger}; "
              f"'{args.capture_key}' captures"
              + (", segments and exports" if seg_p is not None else "")
              + ", q or escape quits")

        def read_key() -> int:
            """The pending key, from the window when there is one."""
            if args.preview:
                return int(cv2.waitKey(1) & 0xFF)
            return poller.poll()

        last_seqs = [0] * len(workers)
        t_loop = time.perf_counter()
        t_status = t_loop
        t_wait_start = t_loop
        pending_capture = False

        def status_line(state):
            fps_str = " ".join(f"{w.cam_name[:1]}{w.fps():4.1f}" for w in workers)
            spread = (f"{statistics.median(spread_history):.0f}"
                      if spread_history else "n/a")
            trig_str = f"  {trigger.state()}" if trigger else ""
            print(f"[stat ] {state}  in fps [{fps_str}]  accepted "
                  f"{accepted_sets}  rejected {rejected_sync}  median spread "
                  f"{spread} ms  inferences {inferences}  clouds "
                  f"{processed}{trig_str}")

        while True:
            now = time.perf_counter()
            if args.duration > 0 and now - t_loop >= args.duration:
                break
            if args.max_frames > 0 and processed >= args.max_frames:
                break

            t_cycle = time.perf_counter()

            # ---- assemble a frame set --------------------------------
            t0 = time.perf_counter()
            snap = [w.latest() for w in workers]
            if any(s is None for s in snap) or all(
                    s[3] == last for s, last in zip(snap, last_seqs)):
                if time.perf_counter() - t_status >= args.status_interval:
                    status_line("no new set")
                    t_status = time.perf_counter()
                time.sleep(0.005)
                key = read_key()
                if key >= 0:
                    if key in (ord("q"), 27):
                        break
                    if key == capture_key:
                        pending_capture = True
                if all(not w.is_alive() for w in workers):
                    print("all acquisition threads have stopped", file=sys.stderr)
                    exit_code = 1
                    break
                continue

            if args.use_device_timestamp and all(s[2] for s in snap):
                spread_ms = (max(s[2] for s in snap) - min(s[2] for s in snap)) / 1e6
            else:
                spread_ms = (max(s[1] for s in snap) - min(s[1] for s in snap)) * 1e3
            spread_history.append(spread_ms)

            if args.max_sync_ms > 0 and spread_ms > args.max_sync_ms:
                rejected_sync += 1
                consecutive_rejects += 1
                last_seqs = [s[3] for s in snap]
                if consecutive_rejects == 100:
                    print(f"[WARN] 100 consecutive sets rejected on sync "
                          f"spread; median spread is "
                          f"{statistics.median(spread_history):.0f} ms against "
                          f"a {args.max_sync_ms:.0f} ms limit. Raise "
                          f"--max-sync-ms, enable --ptp, or lower --fps so the "
                          f"cameras are not fighting for the uplink.")
                if time.perf_counter() - t_status >= args.status_interval:
                    status_line("rejecting on sync")
                    t_status = time.perf_counter()
                time.sleep(0.002)
                continue

            consecutive_rejects = 0
            accepted_sets += 1
            frames = [s[0] for s in snap]
            last_seqs = [s[3] for s in snap]
            assemble_ms = (time.perf_counter() - t0) * 1e3

            # ---- inference, every set, so the heat map is live -------
            if args.input_scale != 1.0:
                s = args.input_scale
                frames = [cv2.resize(f, (int(round(f.shape[1] * s)),
                                         int(round(f.shape[0] * s))),
                                     interpolation=cv2.INTER_AREA)
                          for f in frames]
                Ks_in = np.stack([scale_K(Ks_rect[n], s) for n in names])
            else:
                Ks_in = np.stack([Ks_rect[n] for n in names])

            out = run_inference(model, frames, Ks_in, Es, args)
            raw_depth, conf = out["depth"], out["conf"]
            K_out, E_out, rgb_proc = out["K"], out["E"], out["rgb"]
            inference_ms = out["ms"]
            is_metric = out["is_metric"]
            dh, dw = out["shape"]
            inferences += 1

            t0 = time.perf_counter()
            depth, _ = correct_depths(raw_depth, names, corr, K_out)
            correct_ms = (time.perf_counter() - t0) * 1e3

            if t_measure is None:
                t_measure = time.perf_counter()

            # ---- decide whether this set becomes a cloud -------------
            fire = pending_capture
            forced = pending_capture
            pending_capture = False
            if args.capture_trigger == "continuous":
                fire = True
            elif trigger is not None and trigger.update(frames[ref_idx]):
                fire = True
            if (not fire and args.settle_timeout > 0
                    and time.perf_counter() - t_wait_start >= args.settle_timeout):
                fire = True
                forced = True
                if trigger is not None:
                    trigger.armed = False
                    trigger.static_count = 0
                print(f"[fire ] forced after {args.settle_timeout:.0f} s of "
                      f"waiting (--settle-timeout)")

            backproject_ms = fuse_ms = write_ms = segment_ms = 0.0
            geom_ms = {}
            n_fused = n_before = 0
            per_cam = []
            seg_record = None

            if fire:
                t_wait_start = time.perf_counter()

                t0 = time.perf_counter()
                clouds, tinted, per_cam, world_pts = backproject(
                    names, depth, conf, K_out, E_out, rgb_proc, args)
                backproject_ms = (time.perf_counter() - t0) * 1e3

                t0 = time.perf_counter()
                if args.fuse_mode == "tsdf":
                    fused = fuse_tsdf(names, depth, K_out, E_out, rgb_proc, args)
                    n_before = len(fused.points)
                else:
                    fused = o3d.geometry.PointCloud()
                    for n in names:
                        fused += clouds[n]
                    n_before = len(fused.points)
                    if args.voxel > 0:
                        fused = fused.voxel_down_sample(args.voxel)
                fuse_ms = (time.perf_counter() - t0) * 1e3
                n_fused = len(fused.points)

                if args.layer_report:
                    last_layer = layer_report(world_pts, centres,
                                              args.layer_band_mm / 1e3,
                                              args.layer_above_mm / 1e3)
                    print_layer_report(last_layer)

                # ---- where this capture goes -------------------------
                # A capture forced by the key is always written, whatever
                # --save-mode says, because the key means "keep this one".
                save_this = args.save_mode != "none" or forced
                full_capture = forced or args.save_npy
                if args.save_mode == "latest" and not forced:
                    stem = args.out_dir
                else:
                    stem = args.out_dir / f"frame_{processed:05d}"
                stem.mkdir(parents=True, exist_ok=True)

                # ---- write -------------------------------------------
                t0 = time.perf_counter()
                if save_this:
                    o3d.io.write_point_cloud(str(stem / "fused.ply"), fused)
                    if args.save_per_camera:
                        for n in names:
                            o3d.io.write_point_cloud(
                                str(stem / f"cloud_{n}.ply"), clouds[n])
                    if tinted:
                        merged = o3d.geometry.PointCloud()
                        for p in tinted:
                            merged += p
                        o3d.io.write_point_cloud(
                            str(stem / "fused_by_camera.ply"), merged)
                    if full_capture:
                        # Everything box_scene.py needs to re-open this capture
                        # offline: the corrected depth, the raw depth beside it
                        # so the correction can be audited, and the intrinsics
                        # and extrinsics that belong to the PROCESSED grid.
                        for i, n in enumerate(names):
                            np.save(stem / f"depth_{n}.npy", depth[i])
                            np.save(stem / f"depth_raw_{n}.npy", raw_depth[i])
                            if conf is not None:
                                np.save(stem / f"conf_{n}.npy", conf[i])
                            np.save(stem / f"K_{n}.npy", K_out[i])
                            np.save(stem / f"E_{n}.npy", E_out[i])
                            if args.save_proc and rgb_proc is not None:
                                img = np.asarray(rgb_proc[i])
                                if img.dtype != np.uint8:
                                    img = np.clip(
                                        img * (255.0 if img.max() <= 1.0 + 1e-6
                                               else 1.0), 0, 255).astype(np.uint8)
                                cv2.imwrite(str(stem / f"proc_{n}.png"),
                                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if last_layer is not None:
                        (stem / "layers.json").write_text(
                            json.dumps(last_layer, indent=2))
                write_ms = (time.perf_counter() - t0) * 1e3

                # ---- segment and export ------------------------------
                if seg_p is not None:
                    t0 = time.perf_counter()
                    seg_pts, seg_src = segmentation_input(names, world_pts)
                    if seg_pts is None:
                        print("[seg  ] no points survived filtering, nothing "
                              "to segment")
                    else:
                        cam_c = [camera_centre(np.asarray(E_out[i], np.float64))
                                 for i in range(len(names))]
                        result = bs.segment_boxes(seg_pts, seg_src, names,
                                                  seg_p, cam_centres=cam_c)
                        segment_ms = (time.perf_counter() - t0) * 1e3

                        seg_dir = stem / "segmentation"
                        written, rows = bs.write_results(
                            result, seg_dir, capture_index=processed,
                            timestamp=datetime.now().astimezone().isoformat(
                                timespec="seconds"),
                            images=rgb_proc, K_all=K_out, E_all=E_out, p=seg_p,
                            extra_meta={
                                "capture_dir": str(stem),
                                "correction_applied": bool(corr),
                                "view_terms_applied": (bool(corr)
                                                       and args.view_correction),
                                "fuse_mode": args.fuse_mode,
                                "layers": last_layer})

                        if geom_p is not None:
                            gwritten, geom_ms = bg.write_geometry(
                                result, seg_dir, geom_p,
                                cloud=(bs.coloured_cloud(result)
                                       if geom_p.combined else None),
                                cam_centres=cam_c, capture_index=processed)
                            written.update(gwritten)

                        if args.box_log:
                            append_box_log(args.out_dir / "boxes_log.csv", rows)

                        print(bs.summarise(result))
                        for w in result.get("warnings", []):
                            print(f"[seg  ] {w}")
                        last_seg = result
                        last_boxes = len(result.get("boxes", []))
                        seg_record = {
                            "n_boxes": last_boxes,
                            "n_pickable": sum(1 for b in result.get("boxes", [])
                                              if b["pickable"]),
                            "conveyor_distance_m": (
                                (result.get("conveyor") or {}).get(
                                    "distance_from_reference_camera_m")),
                            "deck_spread_mm": (
                                (result.get("conveyor") or {}).get(
                                    "per_view_spread_mm")),
                            "warnings": result.get("warnings", []),
                            "timings_ms": result.get("timings_ms", {}),
                            "files": {k: str(v) for k, v in written.items()},
                        }
                processed += 1

                if args.view_cloud:
                    o3d.visualization.draw_geometries(
                        [fused], window_name=f"capture {processed}")

            geometry_ms = float(sum(geom_ms.values())) if geom_ms else 0.0
            cycle_ms = (time.perf_counter() - t_cycle) * 1e3
            cycle_stamps.append(time.perf_counter())
            for k, v in (("assemble", assemble_ms), ("inference", inference_ms),
                         ("correct", correct_ms),
                         ("backproject", backproject_ms), ("fuse", fuse_ms),
                         ("write", write_ms), ("segment", segment_ms),
                         ("geometry", geometry_ms), ("cycle", cycle_ms)):
                stage_ms[k].append(v)

            fx_ratio = [float(K_out[i][0, 0]
                              / (Ks_in[i][0, 0] * dw / frames[i].shape[1]))
                        for i in range(len(names))]

            if fire:
                record = {
                    "frame": processed - 1,
                    "t_s": round(time.perf_counter() - t_loop, 4),
                    "forced_trigger": forced,
                    "sync_spread_ms": round(spread_ms, 3),
                    "is_metric": is_metric,
                    "correction_applied": bool(corr),
                    "view_terms_applied": bool(corr) and args.view_correction,
                    "fuse_mode": args.fuse_mode,
                    "fx_ratio": [round(v, 4) for v in fx_ratio],
                    "depth_shape": [dh, dw],
                    "n_points_fused": int(n_fused),
                    "n_points_before_voxel": int(n_before),
                    "layers": last_layer,
                    "segmentation": seg_record,
                    "per_camera": per_cam,
                    "timings_ms": {"assemble": round(assemble_ms, 2),
                                   "inference": round(inference_ms, 2),
                                   "correct": round(correct_ms, 2),
                                   "backproject": round(backproject_ms, 2),
                                   "fuse": round(fuse_ms, 2),
                                   "write": round(write_ms, 2),
                                   "segment": round(segment_ms, 2),
                                   "geometry": round(geometry_ms, 2),
                                   **{k: v for k, v in geom_ms.items()},
                                   "cycle": round(cycle_ms, 2)},
                }
                jsonl.write(json.dumps(record) + "\n")
                jsonl.flush()

                meds = [p["median_depth_m"] for p in per_cam
                        if p["median_depth_m"]]
                depth_str = (f"  median depth {min(meds):.3f}-{max(meds):.3f} m"
                             if meds else "")
                fx_flag = "" if all(0.98 < v < 1.02 for v in fx_ratio) else "  FX!"
                box_str = (f"  boxes {seg_record['n_boxes']}"
                           f"({seg_record['n_pickable']} pickable)"
                           if seg_record else "")
                print(f"[{processed:05d}] captured  infer {inference_ms:6.1f}  "
                      f"corr {correct_ms:5.1f}  bp {backproject_ms:6.1f}  "
                      f"fuse {fuse_ms:6.1f}  write {write_ms:6.1f}  "
                      f"seg {segment_ms:6.1f}  geom {geometry_ms:5.1f}  "
                      f"pts {n_fused:7d}  sync {spread_ms:5.1f} ms"
                      f"{depth_str}{box_str}{fx_flag}")
                t_status = time.perf_counter()

            rate = 0.0
            if len(cycle_stamps) > 1:
                span = cycle_stamps[-1] - cycle_stamps[0]
                rate = (len(cycle_stamps) - 1) / span if span > 0 else 0.0

            if args.preview:
                layer_str = ""
                if last_layer and last_layer.get("above_spread_mm") is not None:
                    layer_str = (f"   layer spread "
                                 f"{last_layer['above_spread_mm']:.0f} mm")
                box_str = ("" if last_boxes is None
                           else f"   boxes {last_boxes}")
                info = [f"live {rate:.2f} Hz   infer {inference_ms:.0f} ms   "
                        f"'{args.capture_key}' captures   q quits",
                        f"inferences {inferences}   clouds {processed}   "
                        f"sync {spread_ms:.1f} ms   "
                        f"correction {'on' if corr else 'OFF'}"
                        f"{layer_str}{box_str}"]
                cv2.imshow(window, depth_preview(names, depth, workers, args,
                                                 info))
            elif time.perf_counter() - t_status >= args.status_interval:
                status_line("streaming")
                t_status = time.perf_counter()

            key = read_key()
            if key >= 0:
                if key in (ord("q"), 27):
                    break
                if key == capture_key:
                    pending_capture = True

            if all(not w.is_alive() for w in workers):
                print("all acquisition threads have stopped", file=sys.stderr)
                exit_code = 1
                break

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stop.set()
        poller.stop()
        for w in workers:
            w.join(timeout=5.0)
        for d in devices:
            try:
                d.stop_stream()
            except Exception:  # noqa: BLE001
                pass
        try:
            system.destroy_device()
        except Exception:  # noqa: BLE001
            pass
        if args.preview:
            cv2.destroyAllWindows()
        jsonl.close()

    # ---- summary ------------------------------------------------------
    elapsed = max(time.perf_counter() - (t_measure or t_loop), 1e-9)
    summary = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        "lens_id": lens,
        "link_plan": link,
        "warmup": warmup_report,
        "correction": ({"path": str(args.correction),
                        "created": (corr_meta or {}).get("created"),
                        "lens_id": (corr_meta or {}).get("lens_id"),
                        "model": (corr_meta or {}).get("model"),
                        "view_terms_applied": args.view_correction,
                        "height_uncertainty": (
                            corr_meta or {}).get("height_uncertainty"),
                        "views": {n: {"alpha": corr[n]["alpha"],
                                      "beta": corr[n]["beta"],
                                      "d_cam": corr[n]["d_cam"]}
                                  for n in corr}} if corr else None),
        "segmentation": ({"enabled": True,
                          "parameters": {k: v for k, v in
                                         vars(seg_p).items()
                                         if not k.startswith("_")},
                          "geometry": (None if geom_p is None
                                       else vars(geom_p)),
                          "last_capture": ({
                              "n_boxes": len(last_seg.get("boxes", [])),
                              "warnings": last_seg.get("warnings", []),
                              "conveyor": last_seg.get("conveyor"),
                          } if last_seg else None)}
                         if seg_p is not None else {"enabled": False}),
        "calibration": {
            n: {"serial": CAMERAS[n],
                "intrinsic_rms_px": intr[n]["rms_px"],
                "extrinsic_rms_px": ext[n]["rms_px"],
                "baseline_m": ext[n]["baseline_m"],
                "distortion_max_field_pct": intr[n]["distortion_max_field_pct"],
                "rectified_size": [plan[n]["crop"][2], plan[n]["crop"][3]],
                "rectified_fx": float(plan[n]["K"][0, 0]),
                "camera_centre_m": camera_centre(ext[n]["E"]).tolist()}
            for n in names},
        "rect_mode": args.rect_mode,
        "fuse_mode": args.fuse_mode,
        "pad_square": False,
        "rot_k_applied": False,
        "ptp_status": ptp_status,
        "node_configuration": config_log,
        "inferences": inferences,
        "frames_processed": processed,
        "sets_accepted": accepted_sets,
        "measured_duration_s": round(elapsed, 3),
        "mean_inference_hz": round(inferences / elapsed, 4) if inferences else 0.0,
        "sync_rejections": rejected_sync,
        "sync_spread_median_ms": (round(statistics.median(spread_history), 2)
                                  if spread_history else None),
        "last_layer_report": last_layer,
        "stage_timings": {k: stage_stats(v) for k, v in stage_ms.items()},
        "per_camera_acquisition": {w.cam_name: w.summary(elapsed) for w in workers},
    }
    if torch is not None and torch.cuda.is_available():
        summary["memory"] = {
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}

    Path(args.json).write_text(json.dumps(summary, indent=2,
                                          default=lambda o: str(o)))
    print(f"\n{inferences} inferences in {elapsed:.1f} s "
          f"({summary['mean_inference_hz']:.3f} Hz), {processed} clouds captured")
    print(f"per-capture records: {args.out_dir / 'frames.jsonl'}")
    if seg_p is not None and args.box_log:
        print(f"parcel log         : {args.out_dir / 'boxes_log.csv'}")
    print(f"summary            : {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())