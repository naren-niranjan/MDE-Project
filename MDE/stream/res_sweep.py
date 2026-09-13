#!/usr/bin/env python3
"""
res_sweep.py

Accuracy against latency as a function of DA3 processing resolution, measured
on one static scene, under the correction stack the system actually runs.

Revision notes
--------------
The first run of this script produced a null footprint column and a destroyed
control. Five faults, all in the reporting rather than the measurement, and all
fixed here:

  1  parcel dimensions live at box["dimensions_m"] in METRES. The previous
     breadth-first search for keys named length_mm found nothing, so every
     parcel scored as unmatched and n_matched was 0 at all four resolutions.

  2  the top-face residual lives at box["top_face"]["plane_residual_rms_m"].
     The breadth-first search reached view_consensus.per_view[0]
     .residual_rms_mm first and reported ONE VIEW'S residual instead of the
     combined fit. Similar magnitude, different quantity.

  3  inter_view_face_spread_mm was medianed over every parcel including
     single-view ones, where the value is zero by construction. With 67 to 91
     per cent of faces single-view, the median was zero and the only control
     that auto-align does not make tautological reported nothing. It is now
     taken over multi-view faces alone, with the count quoted beside it so a
     median over two parcels is visible as such.

  4  deck_check fitted the largest plane in a slab around the expected
     standoff and accepted whatever came back. At three of the four
     resolutions that was a wall with an X normal, not the deck. The fit now
     rejects planes whose normal is more than --deck-max-tilt from the
     reference optical axis and retries on the remaining points.

  5  the yaw spread across resolutions matched parcels by LIST INDEX, which is
     meaningless when the parcel count changes from 9 to 12 between rows. It
     returned an empty dictionary. Matching is now by plan-view position
     against the lowest resolution, which also required fixing box_centre
     (the centre is at box["position_m"]) and box_yaw_deg (the yaw is at
     box["yaw_in_conveyor_frame_deg"]). Neither key was being found either.

Two diagnostics were added because the first run showed they were needed:
a scale probe, since the fitted conveyor distance drifted about 6 per cent
across the sweep on identical pixels and that drift is larger than the
sampling improvement the resolution buys; and a warning when
--seg-cluster-eps is below the measured inter-view disagreement, which is what
splits one parcel into one cluster per camera.

Operating mode
--------------
Auto-align is ON by default, because that is how the cell runs. Every
consequence of that choice is built into the measurement:

  the aligner converges
      The per-view offset is solved from the scene's dominant plane and
      smoothed by an EMA, so one call is not the value the running system
      would be using. The --repeats passes therefore double as convergence:
      the aligner is updated on every pass at the deployment EMA, the geometry
      comes from the last pass, and the offset spread across passes is
      reported as a stability figure.

  the aligner costs time
      Its solve runs on the depth map at --auto-align-stride, so at a fixed
      stride its cost grows with the pixel count. Measured at 14.8 ms at 504
      and 63.5 ms at 1008. That is part of the end-to-end budget and is timed
      separately. --auto-align-stride-scale holds the sample count constant
      instead.

  the deck control becomes tautological
      The aligner fits the deck and forces every camera's median onto it, so
      inter-camera spread AT THE DECK is near zero at every resolution by
      construction. It is reported only as a check that the aligner ran.

      The control that survives is the inter-view disagreement on the PARCEL
      TOP FACES, above the plane the offset was solved on. That is the
      spatially structured component the aligner cannot reach, and it is only
      defined on faces more than one camera actually contributed to.

What this answers
-----------------
--process-res 504 on a 2448 px long side scales the focal length by 0.206 and
puts the ground sampling distance at the deck near 7 mm. The question is which
reported quantity degrades with that, and what recovering it costs.

  footprint length and width
      expected to improve roughly in proportion to GSD, PROVIDED the metric
      scale is anchored. It is not anchored by auto-align, which solves
      relative offsets only, so without --load-affine the footprint column
      measures scale drift as much as sampling. The scale probe below is there
      to make that visible rather than to hide it.

  top-face plane residual
      expected to be flat. Measured flat: 2.8, 2.5, 2.6, 2.7 mm across a 2.6x
      change in GSD.

  inter-view face disagreement
      expected to be flat. It is a structured bias, not noise, and sampling
      does not average it away.

What it does not measure
------------------------
It does not separate DA3's absolute scale error from resolution. Auto-align
removes relative disagreement between views and cannot detect a bias common to
all four, because its reference plane comes from the reference camera. It also
does not measure throughput under load: inference runs on a held frame set
with nothing competing for the GPU, so the milliseconds are a floor.

Resolution values
-----------------
The backbone patches at 14 px, so values should be multiples of 14. Measured
inference scales as roughly the 1.1 power of the token count on this hardware,
so near-linear rather than quadratic; attention is not yet dominant at these
sizes.

Ground truth
------------
    --gt 400 300 250                 repeatable, millimetres, L W H
    --gt-json parcels.json           [{"length_mm":400, "width_mm":300,
                                       "height_mm":250, "x":0.1, "y":-0.2}, ...]

One entry is compared against every detected parcel. Several with x,y are
matched by plan-view position, which is the honest form. Several without x,y
are matched greedily on dimensions, which by construction cannot report a
parcel as the wrong size and is flagged in the record.

Outputs
-------
  sweep.csv          one row per resolution
  sweep.json         the full record, including per-parcel detail
  sweep.png          the accuracy against latency figure
  res_<n>/           the segmentation products at that resolution

Keep beside da3_stream.py, da3_fuse.py, box_segment.py, face_consensus.py and
online_align.py.

Example
-------
python res_sweep.py --calib-dir /home/jetson/Projects/Calibration_4_1/results \\
    --frames-dir runs/ressweep_.../frames --res 504 616 728 1008 --repeats 6 \\
    --pad-square --seg-plane-distance 3.16 --seg-belt-file belt.json \\
    --gt 400 300 250 --out-dir runs/ressweep_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
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

from da3_fuse import (
    load_intrinsics,
    load_extrinsics,
    camera_centre,
    points_camera,
    edge_mask,
    incidence_mask,
    to_world,
    to_o3d,
    load_model,
    as_numpy,
)

# One definition of acquisition, rectification and depth correction, imported
# rather than copied so that a change to the streaming pipeline cannot leave
# this sweep silently measuring something else.
from da3_stream import (
    CAMERAS,
    PIXEL_FORMATS,
    buffer_to_array,
    to_rgb,
    configure,
    build_maps,
    load_affine,
    apply_affine,
)

from box_segment import (
    add_arguments as add_seg_arguments,
    params_from_args as seg_params_from_args,
    segment_boxes,
    write_results as write_seg_results,
    plane_ransac,
)

from face_consensus import deck_offsets_per_view

try:
    from online_align import OnlineAligner
except ImportError as exc:
    OnlineAligner = None
    _ALIGN_IMPORT_ERROR = exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


PATCH = 14
BASE_RES = 504          # the resolution --auto-align-stride is quoted against
OPTICAL_AXIS = np.array([0.0, 0.0, 1.0])   # world frame is the reference camera


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep DA3 processing resolution on one static scene, "
                    "under the auto-align correction the cell runs, and report "
                    "accuracy against latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/ressweep"))
    ap.add_argument("--cameras", nargs="+", default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")

    # the sweep itself
    ap.add_argument("--res", nargs="+", type=int, default=[504, 616, 728, 1008],
                    help="processing resolutions to test. Multiples of 14, "
                         "since the backbone patches at 14 px")
    ap.add_argument("--repeats", type=int, default=6,
                    help="timed inference passes per resolution. With "
                         "auto-align these also serve as the aligner's "
                         "convergence, so a value below about 4 leaves the "
                         "offset short of where the running system would have "
                         "it after a second of streaming")
    ap.add_argument("--warmup-per-res", type=int, default=2,
                    help="passes discarded at each resolution before timing. "
                         "Each new resolution retriggers kernel autotuning and "
                         "a fresh allocation, so one warm-up at the start of "
                         "the sweep does not cover the later resolutions")

    # frames
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="load a previously saved rectified frame set instead "
                         "of grabbing one. Reusing the frames from an earlier "
                         "sweep makes the two directly comparable, since every "
                         "resolution then sees identical pixels in both runs")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--settle-frames", type=int, default=5,
                    help="frames discarded per camera before the set is taken")

    # acquisition
    ap.add_argument("--width", type=int, default=2448)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--pixel-format", default="RGB8", choices=sorted(PIXEL_FORMATS))
    ap.add_argument("--bayer-code", default="COLOR_BayerRG2RGB")
    ap.add_argument("--exposure-us", type=float, default=15000)
    ap.add_argument("--gain-db", type=float, default=0)
    ap.add_argument("--num-buffers", type=int, default=6)
    ap.add_argument("--buffer-timeout-ms", type=int, default=4000)
    ap.add_argument("--throughput-limit", type=int, default=0)
    ap.add_argument("--ptp", action="store_true")
    ap.add_argument("--undistort", dest="undistort", action="store_true", default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")

    # model
    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pad-square", action="store_true",
                    help="pad each frame to square on the right and bottom "
                         "before inference, so a batch of mixed aspect ratios "
                         "is not centre-cropped. Padding the far edges only "
                         "leaves the principal point untouched. The padded "
                         "region is excluded from back-projection explicitly")

    # depth correction, the operating configuration
    ap.add_argument("--auto-align", dest="auto_align", action="store_true",
                    default=True,
                    help="re-solve the per-view additive offset from the "
                         "scene's dominant plane. ON by default because the "
                         "cell runs this way")
    ap.add_argument("--no-auto-align", dest="auto_align", action="store_false",
                    help="freeze the offsets at whatever --load-affine "
                         "supplies. Not the operating configuration")
    ap.add_argument("--auto-align-ema", type=float, default=0.3,
                    help="smoothing on the offset, matching the cell. Applied "
                         "across the --repeats passes so it converges as it "
                         "would live")
    ap.add_argument("--auto-align-stride", type=int, default=3,
                    help="pixel stride used when fitting the plane, quoted at "
                         "504. At a fixed stride the aligner's cost grows with "
                         "the pixel count: 14.8 ms at 504 against 63.5 ms at "
                         "1008 on this rig")
    ap.add_argument("--auto-align-stride-scale", action="store_true",
                    help="scale the stride with the resolution so the aligner "
                         "sees a constant number of samples, holding its cost "
                         "flat across the sweep")
    ap.add_argument("--auto-align-tol", type=float, default=0.12)
    ap.add_argument("--auto-align-max-step", type=float, default=0.15)
    ap.add_argument("--load-affine", type=Path, default=None,
                    help="supplies the per-view SCALE terms, which auto-align "
                         "does not solve, and the absolute anchor. The offsets "
                         "in it are overridden by the aligner. Without this, "
                         "the footprint column carries DA3's resolution "
                         "dependent scale drift")
    ap.add_argument("--apply-absolute", action="store_true")

    # filtering, matching da3_stream.py defaults exactly
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--conf-min", type=float, default=0.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--voxel", type=float, default=0.004)

    # deck plane, reported as an aligner check rather than as a control
    ap.add_argument("--deck-distance", type=float, default=3.16,
                    help="expected deck standoff along the reference camera's "
                         "optical axis, metres. Selects the band the plane is "
                         "fitted to, so the floor and the parcel tops cannot "
                         "be picked instead")
    ap.add_argument("--deck-tol", type=float, default=0.06,
                    help="half-width of that band, metres. Must stay below the "
                         "shortest parcel or a parcel top enters the fit")
    ap.add_argument("--deck-max-tilt", type=float, default=15.0,
                    help="reject a candidate deck plane whose normal lies "
                         "further than this from the reference optical axis. "
                         "Without this the largest plane in the band can be a "
                         "wall, which is what happened at three of four "
                         "resolutions in the first run")
    ap.add_argument("--deck-max-attempts", type=int, default=4,
                    help="candidate planes tried in the band before giving up, "
                         "each fitted on the points the previous candidates "
                         "did not claim")

    # ground truth
    ap.add_argument("--gt", nargs=3, type=float, action="append",
                    metavar=("L_MM", "W_MM", "H_MM"),
                    help="known parcel dimensions in millimetres, repeatable")
    ap.add_argument("--gt-json", type=Path, default=None)
    ap.add_argument("--gt-match-radius", type=float, default=0.25,
                    help="plan-view distance within which a detected parcel "
                         "matches a ground-truth entry, and within which a "
                         "parcel at one resolution is taken to be the same "
                         "parcel at another, metres")

    # arrays for the anchoring solve
    ap.add_argument("--dump-arrays", action="store_true",
                    help="write depth, conf, K and E per camera into each "
                         "res_<n>/ directory, in the layout layer_align.py "
                         "reads. That closes the anchoring loop: solve "
                         "depth_affine.json once per resolution on these "
                         "arrays, then pass it back through --load-affine "
                         "--apply-absolute so the footprint column measures "
                         "sampling rather than DA3's resolution dependent "
                         "scale")
    ap.add_argument("--dump-depth", choices=["raw", "corrected"], default="raw",
                    help="which depth goes into depth_<cam>.npy. RAW by "
                         "default, because da3_stream.py applies the loaded "
                         "coefficients as g_a*(a_i*Z + b_i) + g_b to the raw "
                         "model output, so coefficients solved on corrected "
                         "depth cannot be fed back through that formula. Note "
                         "this is the OPPOSITE of da3_stream.py's capture "
                         "convention, where depth_<cam>.npy is the corrected "
                         "depth and depth_raw_<cam>.npy the model output. The "
                         "other one is written alongside either way, and "
                         "arrays.json records which is which")
    ap.add_argument("--dump-images", dest="dump_images", action="store_true",
                    default=True,
                    help="also write the processed RGB as <cam>.png, which is "
                         "the only image whose size matches the returned "
                         "intrinsics and therefore the only one layer_align.py "
                         "can colour a cloud with")
    ap.add_argument("--no-dump-images", dest="dump_images", action="store_false")

    # products
    ap.add_argument("--write-clouds", action="store_true")
    ap.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    ap.add_argument("--csv", default="sweep.csv")
    ap.add_argument("--json", default="sweep.json")

    if add_seg_arguments is not None:
        add_seg_arguments(ap)

    return ap.parse_args()


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def grab_static_set(names, args, maps):
    """One rectified frame per camera from a scene that is not moving.

    No synchronisation logic. The scene is static, so a frame period of spread
    between the cameras carries no penalty, and that is the reason the sweep is
    run static: it removes sync from the list of things that could explain a
    difference between two resolutions.
    """
    from arena_api.system import system

    infos = system.device_infos
    if not infos:
        raise SystemExit("no cameras found; check the Arena library path and NIC")
    found = {i["serial"]: i for i in infos}
    missing = [n for n in names if CAMERAS[n] not in found]
    if missing:
        raise SystemExit(f"missing camera(s): {', '.join(missing)}; "
                         f"detected {sorted(found)}")

    devices = system.create_device([found[CAMERAS[n]] for n in names])
    frames = {}
    try:
        for n, dev in zip(names, devices):
            configure(dev, args, [])
            dev.start_stream(args.num_buffers)

        for n, dev in zip(names, devices):
            bpp, family = PIXEL_FORMATS[args.pixel_format]
            bayer = getattr(cv2, args.bayer_code, cv2.COLOR_BayerRG2RGB)
            rgb = None
            for _ in range(args.settle_frames + 1):
                item = dev.get_buffer(timeout=args.buffer_timeout_ms)
                try:
                    if item.is_incomplete:
                        continue
                    rgb = to_rgb(buffer_to_array(item, bpp), family, bayer)
                finally:
                    dev.requeue_buffer(item)
            if rgb is None:
                raise SystemExit(f"{n}: no complete buffer in "
                                 f"{args.settle_frames + 1} attempts")
            map1, map2, roi = maps[n]
            if map1 is not None:
                rgb = cv2.remap(rgb, map1, map2, cv2.INTER_LINEAR)
                x, y, w, h = roi
                if w > 0 and h > 0:
                    rgb = rgb[y:y + h, x:x + w]
            frames[n] = rgb
            print(f"[grab ] {n:<8} {rgb.shape[1]}x{rgb.shape[0]}")
    finally:
        for d in devices:
            try:
                d.stop_stream()
            except Exception:  # noqa: BLE001
                pass
        system.destroy_device()
    return frames


def load_frames(frames_dir, names):
    frames = {}
    for n in names:
        p = frames_dir / f"frame_{n}.png"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {p}")
        frames[n] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"[load ] {n:<8} {frames[n].shape[1]}x{frames[n].shape[0]} "
              f"from {p.name}")
    return frames


def save_frames(frames, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(out_dir / f"frame_{n}.png"),
                    cv2.cvtColor(frames[n], cv2.COLOR_RGB2BGR))
    print(f"[save ] rectified frame set written to {out_dir}")


def pad_square(img):
    """Pad on the right and bottom to a square, replicating the edge.

    Padding the far edges only leaves cx and cy unchanged, so the input
    intrinsics need no adjustment. Edge replication rather than zeros, because
    a hard black border reads to the model as a depth discontinuity and bleeds
    into the samples beside it.
    """
    h, w = img.shape[:2]
    s = max(h, w)
    if h == s and w == s:
        return img, 1.0, 1.0
    out = cv2.copyMakeBorder(img, 0, s - h, 0, s - w, cv2.BORDER_REPLICATE)
    return out, w / s, h / s


# --------------------------------------------------------------------------
# parcel record accessors
# --------------------------------------------------------------------------
# FIX 1, 2 and 5. These read the record at its actual paths rather than
# searching for plausible key names. The breadth-first search the previous
# revision used found either nothing at all or, worse, a same-named field on a
# nested per-view record, which is a different quantity reported under the
# right heading. Explicit paths cannot do that. Each accessor falls back to
# None and the caller warns once with the keys that are present.

def box_dims_mm(b):
    """Length, width and height in millimetres.

    box_segment writes these under dimensions_m, in metres. Note that height
    there is the top face above the SUPPORT, which for a stacked parcel is the
    parcel beneath it rather than the conveyor; support.height_is_inferred says
    which.
    """
    d = b.get("dimensions_m") or {}
    out = {}
    for axis in ("length", "width", "height"):
        v = d.get(axis)
        out[axis] = float(v) * 1e3 if v is not None else None
    return out


def box_height_inferred(b):
    s = b.get("support") or {}
    v = s.get("height_is_inferred")
    return bool(v) if v is not None else None


def box_residual_mm(b):
    """RMS residual of the COMBINED top-face plane fit, in millimetres.

    Not view_consensus.per_view[n].residual_rms_mm, which is one camera's fit
    to its own sheet and will look clean precisely when the cameras disagree
    most with each other.
    """
    tf = b.get("top_face") or {}
    v = tf.get("plane_residual_rms_m")
    return float(v) * 1e3 if v is not None else None


def box_residual_p95_mm(b):
    tf = b.get("top_face") or {}
    v = tf.get("plane_residual_p95_m")
    return float(v) * 1e3 if v is not None else None


def box_coverage(b):
    tf = b.get("top_face") or {}
    v = tf.get("coverage")
    return float(v) if v is not None else None


def _consensus(b):
    return b.get("view_consensus") or {}


def box_single_view(b):
    v = b.get("single_view")
    if v is None:
        v = _consensus(b).get("single_view")
    return bool(v) if v is not None else None


def box_views_used(b):
    v = _consensus(b).get("views_used")
    return list(v) if isinstance(v, list) else []


def box_face_spread_mm(b):
    """Inter-view disagreement on this face before reconciliation.

    Zero by construction on a single-view face, which is why the caller must
    exclude those before taking a median.
    """
    v = _consensus(b).get("inter_view_offset_mm")
    return float(v) if v is not None else None


def box_height_uncertainty_mm(b):
    v = _consensus(b).get("height_uncertainty_mm")
    return float(v) if v is not None else None


def box_centre(b):
    """Parcel centre in world metres."""
    v = b.get("position_m")
    if v is not None:
        return np.asarray(v, dtype=float)[:3]
    tf = b.get("top_face") or {}
    v = tf.get("centre_m")
    if v is not None:
        return np.asarray(v, dtype=float)[:3]
    v = b.get("pose_world")
    if v is not None:
        arr = np.asarray(v, dtype=float)
        if arr.shape == (4, 4):
            return arr[:3, 3]
    return None


def box_yaw_deg(b):
    for k in ("yaw_in_conveyor_frame_deg", "yaw_deg"):
        v = b.get(k)
        if v is not None:
            return float(v)
    return None


def conveyor_distance_m(seg):
    """Distance from the reference camera to the fitted conveyor plane.

    Used as a scale probe. On identical pixels this should not move with
    resolution; in the first run it drifted about 6 per cent, which is larger
    than the sampling improvement the resolution buys and therefore has to be
    reported next to any footprint figure.
    """
    c = seg.get("conveyor")
    if not isinstance(c, dict):
        return None
    for k in ("distance_m", "plane_distance_m", "offset_m", "standoff_m", "d"):
        v = c.get(k)
        if isinstance(v, (int, float)):
            return abs(float(v))
    return None


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def load_ground_truth(args):
    gt = []
    if args.gt_json is not None:
        for e in json.loads(args.gt_json.read_text()):
            gt.append({"length_mm": float(e["length_mm"]),
                       "width_mm": float(e["width_mm"]),
                       "height_mm": float(e.get("height_mm", float("nan"))),
                       "x": (float(e["x"]) if "x" in e else None),
                       "y": (float(e["y"]) if "y" in e else None)})
    for triple in (args.gt or []):
        gt.append({"length_mm": float(triple[0]), "width_mm": float(triple[1]),
                   "height_mm": float(triple[2]), "x": None, "y": None})
    return gt


def match_to_gt(boxes, gt, radius):
    """Pair each detected parcel with a ground-truth entry.

    Positional matching where x and y are supplied, since that is the only form
    in which a dimension error is a measurement rather than an assignment.
    Otherwise greedy on dimensions, which by construction cannot report a
    parcel as the wrong size, and is flagged accordingly.
    """
    if not gt or not boxes:
        return [], "none"

    if len(gt) == 1:
        return [(bi, 0) for bi in range(len(boxes))], "single_reference"

    if all(g["x"] is not None and g["y"] is not None for g in gt):
        pairs, used = [], set()
        for bi, b in enumerate(boxes):
            c = box_centre(b)
            if c is None:
                continue
            best, best_d = None, radius
            for gi, g in enumerate(gt):
                if gi in used:
                    continue
                d = float(np.hypot(c[0] - g["x"], c[1] - g["y"]))
                if d < best_d:
                    best, best_d = gi, d
            if best is not None:
                used.add(best)
                pairs.append((bi, best))
        return pairs, "position"

    pairs, used = [], set()
    for bi, b in enumerate(boxes):
        d = box_dims_mm(b)
        if d["length"] is None or d["width"] is None:
            continue
        bl, bw = max(d["length"], d["width"]), min(d["length"], d["width"])
        best, best_c = None, float("inf")
        for gi, g in enumerate(gt):
            if gi in used:
                continue
            gl = max(g["length_mm"], g["width_mm"])
            gw = min(g["length_mm"], g["width_mm"])
            c = abs(bl - gl) + abs(bw - gw)
            if c < best_c:
                best, best_c = gi, c
        if best is not None:
            used.add(best)
            pairs.append((bi, best))
    return pairs, "dimension_greedy"


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def backproject(names, depth, conf, K_out, E_out, rgb_proc, args, valid_box):
    """The same filter chain and back-projection as a live capture.

    Per-point camera provenance is retained, because the deck and face
    disagreements are per-camera quantities and merging destroys them.
    """
    pts_list, src_list, col_list, per_cam = [], [], [], []
    for i, n in enumerate(names):
        d = depth[i].astype(np.float64)
        c = conf[i]
        valid = np.isfinite(d) & (d > 0)

        if valid_box is not None:
            fx_frac, fy_frac = valid_box[i]
            h, w = d.shape
            box = np.zeros(d.shape, dtype=bool)
            box[:max(1, int(round(h * fy_frac))),
                :max(1, int(round(w * fx_frac)))] = True
            valid &= box

        thr = None
        if args.conf_percentile > 0 and valid.any():
            thr = float(np.percentile(c[valid], args.conf_percentile))
            valid &= c >= thr
        if args.conf_min > 0:
            valid &= c >= args.conf_min
        if args.edge_thresh > 0:
            valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)

        pts_cam = points_camera(d, K_out[i])
        ang = None
        if args.max_incidence < 90:
            ok, ang = incidence_mask(pts_cam, args.max_incidence)
            valid &= ok

        pts_world = to_world(pts_cam[valid], E_out[i])
        pts_list.append(pts_world)
        src_list.append(np.full(len(pts_world), i, dtype=np.int32))
        col_list.append(np.asarray(rgb_proc[i][valid], dtype=np.float64) / 255.0)

        dv = d[valid]
        fx = float(K_out[i][0, 0])
        med = float(np.median(dv)) if dv.size else float("nan")
        per_cam.append({
            "camera": n,
            "n_points": int(valid.sum()),
            "processed_shape": [int(d.shape[0]), int(d.shape[1])],
            "fx_px": round(fx, 2),
            "median_depth_m": round(med, 4) if np.isfinite(med) else None,
            "gsd_mm": (round(med / fx * 1e3, 3)
                       if np.isfinite(med) and fx > 0 else None),
            "median_incidence_deg": (round(float(np.median(ang[valid])), 2)
                                     if ang is not None and dv.size else None),
            "conf_threshold": thr,
        })

    pts = np.concatenate(pts_list) if pts_list else np.empty((0, 3))
    src = np.concatenate(src_list) if src_list else np.empty(0, np.int32)
    col = np.concatenate(col_list) if col_list else np.empty((0, 3))
    return pts, src, col, per_cam


def deck_check(pts, src, names, args, p, auto_align):
    """Fit the deck plane in a band around its expected standoff.

    FIX 4. The previous revision took whatever plane_ransac returned from the
    band. In the band there is also the conveyor's side frame and, at this
    standoff, part of the machine cabinet, and at three of four resolutions
    those won: the accepted normal was near [1, 0, 0], perpendicular to the
    deck. Every per-camera median and the spread computed from them then
    referred to a wall.

    Candidates are now rejected unless their normal lies within
    --deck-max-tilt of the reference optical axis, and the fit is retried on
    the points the rejected candidates did not claim.

    Under auto-align this remains a CHECK rather than a control: the aligner
    solved the offsets against this plane, so the spread is near zero by
    construction and only a large value carries information.
    """
    if len(pts) == 0:
        return None
    band = np.abs(pts[:, 2] - args.deck_distance) <= args.deck_tol
    if int(band.sum()) < 500:
        return {"ok": False,
                "reason": f"only {int(band.sum())} points within "
                          f"{args.deck_tol * 1e3:.0f} mm of the expected deck "
                          f"standoff {args.deck_distance:.2f} m. Either the "
                          f"deck is not there or the depth is unanchored."}

    Q, S = pts[band], src[band]
    pool = np.arange(len(Q))
    rejected = []
    for _ in range(max(1, args.deck_max_attempts)):
        if len(pool) < 500:
            break
        n, d, inl = plane_ransac(Q[pool], p.plane_thresh, p.plane_iter)
        if n is None or len(inl) < 300:
            break
        n = np.asarray(n, dtype=float)
        if float(n @ OPTICAL_AXIS) < 0:
            n, d = -n, -d
        tilt = float(np.degrees(np.arccos(
            np.clip(abs(float(n @ OPTICAL_AXIS)), 0.0, 1.0))))
        claimed = pool[inl]
        if tilt <= args.deck_max_tilt:
            h = Q @ n + d
            per_cam = deck_offsets_per_view(h, S, names, claimed)
            meds = [v["median_mm"] for v in per_cam.values()]
            return {
                "ok": True,
                "is_control": not auto_align,
                "note": ("tautological under auto-align: the aligner fitted "
                         "this plane and forced the medians onto it, so a "
                         "small spread confirms only that it ran"
                         if auto_align else
                         "a genuine control: the offsets were frozen, so this "
                         "spread is the uncorrected disagreement"),
                "n_band_points": int(band.sum()),
                "n_inliers": int(len(claimed)),
                "normal": [round(float(v), 5) for v in n],
                "tilt_from_axis_deg": round(tilt, 3),
                "offset_m": round(float(d), 5),
                "rms_mm": round(float(np.sqrt(np.mean(h[claimed] ** 2))) * 1e3, 3),
                "per_camera": per_cam,
                "inter_camera_spread_mm": (round(max(meds) - min(meds), 3)
                                           if len(meds) > 1 else None),
                "candidates_rejected": rejected,
            }
        rejected.append({"tilt_from_axis_deg": round(tilt, 2),
                         "n_inliers": int(len(claimed)),
                         "normal": [round(float(v), 4) for v in n]})
        pool = np.setdiff1d(pool, claimed, assume_unique=False)

    return {"ok": False,
            "reason": f"no plane within {args.deck_max_tilt:.0f} deg of the "
                      f"optical axis was found in the band",
            "candidates_rejected": rejected}


# --------------------------------------------------------------------------
# arrays for the anchoring solve
# --------------------------------------------------------------------------

def dump_arrays(stem, names, depth_raw, depth, conf, K_out, E_out, rgb_proc,
                valid_box, args, res):
    """Write the per-camera arrays layer_align.py needs, per resolution.

    Without these there is no path from this sweep to an anchor: layer_align.py
    reads depth_<cam>.npy, conf_<cam>.npy, K_<cam>.npy and E_<cam>.npy from a
    capture directory, and res_sweep.py was not writing them. Every footprint
    figure therefore carried DA3's scale, which on this rig runs 8.4 per cent
    short of truth at 504 and 3.4 per cent short at 1008, so a third of any
    apparent improvement across the sweep was the scale converging rather than
    the sampling improving.

    Three things matter about what is written here.

    RAW depth by default. da3_stream.py applies loaded coefficients as
    g_a * (a_i * Z + b_i) + g_b to the raw model output, and layer_align.py
    likewise extracts its reference planes from uncorrected depth. Coefficients
    solved on already-corrected depth are a correction of a correction and
    cannot be fed back through that formula. Note that this inverts
    da3_stream.py's own capture convention, where depth_<cam>.npy is corrected
    and depth_raw_<cam>.npy is the model output. Mixing a capture from one with
    a solve intended for the other silently produces coefficients that look
    plausible and are wrong, so arrays.json records which convention this
    directory uses.

    The padded border is zeroed. With --pad-square the replicated edge carries
    invented depth that back-projection excludes through valid_box, but
    layer_align.py has no such mask and admits any sample with d > 0. Left in,
    that border would join the plane fits and drag the solved scale. Zero is
    the sentinel its valid_mask already rejects.

    The intrinsics are DA3's returned values, which belong to the processed
    image and not to the rectified frame, so the RGB written beside them is the
    processed image for the same reason.
    """
    which = args.dump_depth
    primary = depth_raw if which == "raw" else depth
    secondary = depth if which == "raw" else depth_raw
    secondary_stem = "depth_corrected" if which == "raw" else "depth_raw"

    per_cam = []
    for i, n in enumerate(names):
        d = np.array(primary[i], dtype=np.float64, copy=True)
        o = np.array(secondary[i], dtype=np.float64, copy=True)

        valid_fraction = 1.0
        if valid_box is not None:
            fx_frac, fy_frac = valid_box[i]
            h, w = d.shape
            keep = np.zeros(d.shape, dtype=bool)
            keep[:max(1, int(round(h * fy_frac))),
                 :max(1, int(round(w * fx_frac)))] = True
            d[~keep] = 0.0
            o[~keep] = 0.0
            valid_fraction = float(keep.mean())

        np.save(stem / f"depth_{n}.npy", d)
        np.save(stem / f"{secondary_stem}_{n}.npy", o)
        np.save(stem / f"conf_{n}.npy", conf[i])
        np.save(stem / f"K_{n}.npy", K_out[i])
        np.save(stem / f"E_{n}.npy", E_out[i])

        if args.dump_images:
            p = rgb_proc[i]
            p8 = p if p.dtype == np.uint8 else (np.clip(p, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(stem / f"{n}.png"), cv2.cvtColor(p8, cv2.COLOR_RGB2BGR))

        good = np.isfinite(d) & (d > 0)
        per_cam.append({
            "camera": n,
            "shape": [int(d.shape[0]), int(d.shape[1])],
            "unpadded_fraction": round(valid_fraction, 4),
            "n_valid": int(good.sum()),
            "median_depth_m": (round(float(np.median(d[good])), 4)
                               if good.any() else None),
            "fx_px": round(float(K_out[i][0, 0]), 2),
        })

    manifest = {
        "process_res": res,
        "depth_convention": {
            "depth_<cam>.npy": ("the RAW model output, before any correction"
                                if which == "raw" else
                                "the CORRECTED depth, after the per-view "
                                "offset and any loaded coefficients"),
            f"{secondary_stem}_<cam>.npy": ("the corrected depth"
                                            if which == "raw" else
                                            "the raw model output"),
            "note": ("this inverts da3_stream.py's capture convention, in "
                     "which depth_<cam>.npy is the corrected depth. Solve on "
                     "the raw arrays or the coefficients cannot be applied "
                     "through da3_stream.py's own formula."),
        },
        "intrinsics": ("DA3's returned intrinsics for the processed image, not "
                       "the rectified frame. <cam>.png is that processed image."),
        "padded_border": ("zeroed" if valid_box is not None else "not applicable"),
        "filters_NOT_applied": ("none of the confidence, edge or incidence "
                                "filters have been applied to these arrays. "
                                "layer_align.py applies its own, and its "
                                "defaults already match this script's"),
        "per_camera": per_cam,
        "next_step": (f"python layer_align.py --capture-dir {stem} "
                      f"--mode both --holdout --n-planes 3"),
        "anchor_caveat": ("layer_align.py's absolute block assumes the nearest "
                          "reference plane is the deck at da3_fuse.GT_DECK_PERP_M. "
                          "Confirm that constant is the deck standoff from the "
                          "REFERENCE camera's optical centre before trusting "
                          "any dimension solved from it."),
    }
    (stem / "arrays.json").write_text(json.dumps(manifest, indent=2))
    print(f"[dump ] arrays for layer_align written to {stem} "
          f"({which} depth in depth_<cam>.npy)")
    return manifest


# --------------------------------------------------------------------------
# one resolution
# --------------------------------------------------------------------------

def run_resolution(res, model, names, frames, Ks_in, Es, args, seg_params,
                   per_cam_affine, global_affine, valid_box, out_dir,
                   affine_meta):
    print(f"\n{'=' * 72}\n[res  ] process_res {res}"
          + ("" if res % PATCH == 0 else
             f"   NOT a multiple of {PATCH}; the backbone will round it"))
    stem = out_dir / f"res_{res:04d}"
    stem.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def infer():
        if args.mode == "prior":
            return model.inference(frames, intrinsics=Ks_in, extrinsics=Es,
                                   align_to_input_ext_scale=True,
                                   process_res=res)
        return model.inference(frames, process_res=res)

    # ---- aligner, constructed fresh at this resolution -------------------
    # Carrying one across resolutions would let an offset solved at 504 leak
    # into 1008 through the EMA, which is the confound this sweep exists to
    # avoid.
    aligner, stride = None, args.auto_align_stride
    if args.auto_align:
        if OnlineAligner is None:
            raise SystemExit(f"--auto-align needs online_align.py in the "
                             f"working directory: {_ALIGN_IMPORT_ERROR}")
        if args.auto_align_stride_scale:
            stride = max(1, int(round(args.auto_align_stride * res / BASE_RES)))
        base_a = ({n: per_cam_affine[n][0] for n in names}
                  if per_cam_affine else {n: 1.0 for n in names})
        aligner = OnlineAligner(names, args.reference, base_a=base_a,
                                stride=stride, ema=args.auto_align_ema,
                                assign_tol=args.auto_align_tol,
                                max_step=args.auto_align_max_step)
        print(f"[corr ] auto-align on, stride {stride}, ema "
              f"{args.auto_align_ema}, converging over {args.repeats} passes")

    for k in range(max(0, args.warmup_per_res)):
        t0 = time.perf_counter()
        infer()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[warm ] discarded pass {k + 1}, "
              f"{(time.perf_counter() - t0) * 1e3:.0f} ms")

    # ---- timed passes, with the aligner converging alongside -------------
    times, align_ms, offset_history = [], [], []
    kept, align_diag = None, None
    for r in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        pred = infer()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

        depth_raw = as_numpy(pred.depth)
        K_out = as_numpy(pred.intrinsics)
        E_out = as_numpy(pred.extrinsics)

        if aligner is not None:
            t1 = time.perf_counter()
            b_now, align_diag = aligner.update(depth_raw, K_out, E_out)
            align_ms.append((time.perf_counter() - t1) * 1e3)
            offset_history.append({n: round(b_now[n] * 1e3, 3) for n in names})
            if align_diag is not None and not align_diag.get("ok"):
                print(f"[WARN] pass {r}: auto-align failed, "
                      f"{align_diag.get('reason')}. The offsets in use are the "
                      f"previous ones, so this resolution is not being "
                      f"measured under the correction it appears to be.")
        kept = pred                       # the last pass, where the EMA settled

    print(f"[infer] median {statistics.median(times):.0f} ms over "
          f"{len(times)} passes, min {min(times):.0f}, max {max(times):.0f}")
    if align_ms:
        print(f"[align] median {statistics.median(align_ms):.1f} ms, "
              f"{statistics.median(align_ms) / statistics.median(times) * 100:.1f}% "
              f"of inference")

    depth_raw = as_numpy(kept.depth)
    conf = as_numpy(kept.conf)
    K_out = as_numpy(kept.intrinsics)
    E_out = as_numpy(kept.extrinsics)
    rgb_proc = as_numpy(kept.processed_images)
    is_metric = int(getattr(kept, "is_metric", 0))

    # ---- offset stability across passes ----------------------------------
    stability = None
    if len(offset_history) > 1:
        tail = offset_history[1:]         # the first pass is unsmoothed by design
        stability = {n: round(max(o[n] for o in tail) - min(o[n] for o in tail), 3)
                     for n in names}
        print(f"[align] offsets mm {offset_history[-1]}   "
              f"drift across passes {stability}")
        if max(stability.values()) > 5.0:
            print(f"[WARN] the offset moved {max(stability.values()):.1f} mm "
                  f"across passes on a STATIC scene. That is the aligner's own "
                  f"noise, and the live system carries it into every frame.")

    # ---- correction ------------------------------------------------------
    t0 = time.perf_counter()
    if aligner is not None:
        eff = {n: ((per_cam_affine[n][0] if per_cam_affine else 1.0),
                   aligner.b[n]) for n in names}
        depth = apply_affine(depth_raw, names, eff, global_affine)
        offsets_mm = {n: round(aligner.b[n] * 1e3, 2) for n in names}
    elif per_cam_affine is not None:
        depth = apply_affine(depth_raw, names, per_cam_affine, global_affine)
        offsets_mm = {n: round(per_cam_affine[n][1] * 1e3, 2) for n in names}
    else:
        depth = depth_raw
        offsets_mm = {n: 0.0 for n in names}
    correct_ms = (time.perf_counter() - t0) * 1e3

    # ---- back-projection -------------------------------------------------
    t0 = time.perf_counter()
    pts, src, col, per_cam = backproject(names, depth, conf, K_out, E_out,
                                         rgb_proc, args, valid_box)
    backproject_ms = (time.perf_counter() - t0) * 1e3
    for e in per_cam:
        gsd = e["gsd_mm"] if e["gsd_mm"] is not None else float("nan")
        print(f"[proj ] {e['camera']:<8} {e['n_points']:>8d} pts   "
              f"fx {e['fx_px']:7.1f} px   GSD {gsd:5.2f} mm   "
              f"median depth {e['median_depth_m']} m")

    # Written before segmentation, so a resolution whose conveyor fit fails
    # still yields the arrays the anchor is solved from. The failure that
    # produced this ordering was a run in which all four resolutions returned
    # no parcels and no arrays, leaving nothing to diagnose from.
    arrays_meta = None
    if args.dump_arrays:
        arrays_meta = dump_arrays(stem, names, depth_raw, depth, conf, K_out,
                                  E_out, rgb_proc, valid_box, args, res)

    deck = deck_check(pts, src, names, args, seg_params, args.auto_align)
    if deck and deck.get("ok"):
        print(f"[deck ] rms {deck['rms_mm']:.2f} mm over {deck['n_inliers']} "
              f"inliers, tilt {deck['tilt_from_axis_deg']:.1f} deg   "
              f"spread {deck['inter_camera_spread_mm']} mm"
              + ("  (forced by the aligner)" if args.auto_align else ""))
        for c in deck.get("candidates_rejected", []):
            print(f"        rejected a plane at {c['tilt_from_axis_deg']:.0f} "
                  f"deg from the optical axis with {c['n_inliers']} inliers; "
                  f"that is a wall, not the deck")
    elif deck:
        print(f"[deck ] not measured: {deck.get('reason')}")

    # ---- segmentation ----------------------------------------------------
    t0 = time.perf_counter()
    seg = segment_boxes(pts, src, names, seg_params,
                        cam_centres=[camera_centre(E_out[i])
                                     for i in range(len(names))])
    segment_ms = (time.perf_counter() - t0) * 1e3
    try:
        write_seg_results(seg, stem, capture_index=0, timestamp=stamp,
                          images=rgb_proc, K_all=K_out, E_all=E_out,
                          p=seg_params,
                          extra_meta={"process_res": res,
                                      "depth_correction": affine_meta,
                                      "auto_align_state": align_diag})
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] segmentation products not written: {exc!r}")

    boxes = seg.get("boxes", []) or []
    print(f"[seg  ] {len(boxes)} parcel(s) in {segment_ms:.0f} ms")
    for w in seg.get("warnings", []):
        print(f"[seg  ] {w}")

    gsd_all = [e["gsd_mm"] for e in per_cam if e["gsd_mm"] is not None]
    fx_all = [float(K_out[i][0, 0]) for i in range(len(names))]
    align_med = statistics.median(align_ms) if align_ms else 0.0

    if args.write_clouds and o3d is not None:
        cloud = to_o3d(pts, colors=col * 255.0)
        if args.voxel > 0:
            cloud = cloud.voxel_down_sample(args.voxel)
        o3d.io.write_point_cloud(str(stem / "fused.ply"), cloud)

    return {
        "process_res": res,
        "timestamp": stamp,
        "is_metric": is_metric,
        "inference_ms": {"median": round(statistics.median(times), 2),
                         "min": round(min(times), 2),
                         "max": round(max(times), 2),
                         "n": len(times),
                         "all": [round(v, 2) for v in times]},
        "align_ms": {"median": round(align_med, 2),
                     "stride": stride,
                     "all": [round(v, 2) for v in align_ms]} if align_ms else None,
        "stage_ms": {"correct": round(correct_ms, 2),
                     "backproject": round(backproject_ms, 2),
                     "segment": round(segment_ms, 2)},
        "pipeline_ms": round(statistics.median(times) + align_med + correct_ms
                             + backproject_ms + segment_ms, 2),
        "gpu_peak_alloc_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                              if torch is not None and torch.cuda.is_available()
                              else None),
        "fx_px_mean": round(float(np.mean(fx_all)), 2),
        "gsd_mm_mean": round(float(np.mean(gsd_all)), 3) if gsd_all else None,
        "n_points": int(len(pts)),
        "per_camera": per_cam,
        "offsets_mm": offsets_mm,
        "offset_history_mm": offset_history,
        "offset_drift_mm": stability,
        "auto_align": align_diag,
        "deck": deck,
        "arrays": arrays_meta,
        "conveyor": seg.get("conveyor"),
        "conveyor_distance_m": conveyor_distance_m(seg),
        "segmentation": {"ok": seg.get("ok"), "reason": seg.get("reason"),
                         "n_boxes": len(boxes),
                         "warnings": seg.get("warnings", []),
                         "boxes": boxes},
        "output_dir": str(stem),
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(entry, gt, args, warned):
    """Footprint error against ground truth, plus the controls.

    Length and width are compared as the larger and smaller of each pair, so a
    parcel reported with its axes swapped is not counted as two large errors.

    FIX 3. The inter-view spread and the height uncertainty are taken over
    MULTI-VIEW faces only. On a single-view face the spread is zero because
    there is nothing to disagree with, so including those in the median
    reports zero whenever most faces are single-view, which is exactly the
    regime this rig is in. The multi-view count is reported beside the figure
    so a median over two parcels is not mistaken for a population statistic.
    """
    boxes = entry["segmentation"]["boxes"]
    pairs, mode = match_to_gt(boxes, gt, args.gt_match_radius)

    detail, err_l, err_w, err_h = [], [], [], []
    for bi, gi in pairs:
        b, g = boxes[bi], gt[gi]
        d = box_dims_mm(b)
        if d["length"] is None or d["width"] is None:
            if not warned["dims"]:
                warned["dims"] = True
                print(f"[WARN] no dimensions_m block on the parcel record. "
                      f"Keys present: {sorted(b.keys())}.")
            continue
        bl, bw = max(d["length"], d["width"]), min(d["length"], d["width"])
        gl = max(g["length_mm"], g["width_mm"])
        gw = min(g["length_mm"], g["width_mm"])
        el, ew = bl - gl, bw - gw
        err_l.append(el)
        err_w.append(ew)
        eh = None
        # A parcel resting on another reports its height above that support,
        # not above the conveyor, so scoring it against a ground-truth height
        # would be comparing two different quantities.
        if (d["height"] is not None and np.isfinite(g["height_mm"])
                and not box_height_inferred(b)):
            eh = d["height"] - g["height_mm"]
            err_h.append(eh)
        detail.append({"box": bi, "gt": gi,
                       "measured_mm": [round(bl, 1), round(bw, 1),
                                       round(d["height"], 1)
                                       if d["height"] is not None else None],
                       "truth_mm": [gl, gw, g["height_mm"]],
                       "error_mm": [round(el, 1), round(ew, 1),
                                    round(eh, 1) if eh is not None else None],
                       "height_inferred": box_height_inferred(b),
                       "coverage": box_coverage(b),
                       "top_face_residual_rms_mm": box_residual_mm(b),
                       "views_used": box_views_used(b),
                       "single_view": box_single_view(b),
                       "inter_view_offset_mm": box_face_spread_mm(b)})

    # Controls are read from every parcel, matched or not: they do not depend
    # on ground truth and discarding the unmatched ones shrinks the sample for
    # no reason.
    resid = [v for v in (box_residual_mm(b) for b in boxes) if v is not None]
    resid95 = [v for v in (box_residual_p95_mm(b) for b in boxes) if v is not None]

    multi = [b for b in boxes if box_single_view(b) is False]
    single_n = sum(1 for b in boxes if box_single_view(b) is True)
    spread = [v for v in (box_face_spread_mm(b) for b in multi) if v is not None]
    unc = [v for v in (box_height_uncertainty_mm(b) for b in multi) if v is not None]

    if boxes and not resid and not warned["resid"]:
        warned["resid"] = True
        print(f"[WARN] no top_face.plane_residual_rms_m on the parcel record. "
              f"Keys present: {sorted(boxes[0].keys())}.")
    if boxes and not multi and not warned["spread"]:
        warned["spread"] = True
        print(f"[WARN] every top face at this resolution is single-view, so "
              f"the inter-view control has no sample at all. That is a finding "
              f"about the rig rather than a gap in the report: with the "
              f"cameras disagreeing by more than --seg-cluster-eps at the "
              f"parcel tops, each view becomes its own cluster.")

    def med(v):
        return round(float(np.median(v)), 2) if v else None

    def mae(v):
        return round(float(np.mean(np.abs(v))), 2) if v else None

    def bias(v):
        return round(float(np.mean(v)), 2) if v else None

    footprint = [abs(v) for v in err_l] + [abs(v) for v in err_w]
    entry["score"] = {
        "match_mode": mode,
        "n_matched": len(detail),
        "length_mae_mm": mae(err_l),
        "length_bias_mm": bias(err_l),
        "width_mae_mm": mae(err_w),
        "width_bias_mm": bias(err_w),
        "height_mae_mm": mae(err_h),
        "height_bias_mm": bias(err_h),
        "n_height_scored": len(err_h),
        "footprint_mae_mm": round(float(np.mean(footprint)), 2) if footprint else None,
        "footprint_p95_mm": (round(float(np.percentile(footprint, 95)), 2)
                             if len(footprint) >= 3 else None),
        "top_face_residual_rms_mm": med(resid),
        "top_face_residual_p95_mm": med(resid95),
        "n_faces": len(boxes),
        "single_view_faces": single_n,
        "multi_view_faces": len(multi),
        "multi_view_fraction": (round(len(multi) / len(boxes), 3)
                                if boxes else None),
        "inter_view_face_spread_mm": med(spread),
        "inter_view_face_spread_max_mm": (round(float(max(spread)), 2)
                                          if spread else None),
        "height_uncertainty_mm": med(unc),
        "per_parcel": detail,
    }
    if mode == "dimension_greedy" and len(gt) > 1:
        entry["score"]["caveat"] = (
            "parcels were matched to ground truth on dimensions, so a parcel "
            "measured badly enough to resemble a different entry is scored "
            "against that entry instead. Supply x and y in --gt-json for a "
            "matching that cannot do this.")
    return entry


def match_across_resolutions(entries, radius):
    """Track the same physical parcel across resolutions by plan-view position.

    FIX 5. The previous revision compared parcel LIST INDEX between rows, which
    is meaningless when segmentation returns 9 parcels at one resolution and 12
    at another; it silently produced an empty result. Anchors are taken from
    the lowest resolution and each later resolution contributes the nearest
    parcel within the radius.

    Yaw is reported as a spread rather than an error, because the parcels are
    placed by hand and their true yaw is not known to better than the quantity
    being measured. It is a repeatability figure.
    """
    if not entries:
        return []
    ref = entries[0]
    anchors = []
    for bi, b in enumerate(ref["segmentation"]["boxes"]):
        c = box_centre(b)
        if c is None:
            continue
        anchors.append({"anchor_xy": [round(float(c[0]), 4), round(float(c[1]), 4)],
                        "per_resolution": {}})

    for e in entries:
        used = set()
        for a in anchors:
            ax, ay = a["anchor_xy"]
            best, best_d = None, radius
            for bi, b in enumerate(e["segmentation"]["boxes"]):
                if bi in used:
                    continue
                c = box_centre(b)
                if c is None:
                    continue
                dist = float(np.hypot(c[0] - ax, c[1] - ay))
                if dist < best_d:
                    best, best_d = bi, dist
            if best is None:
                continue
            used.add(best)
            b = e["segmentation"]["boxes"][best]
            d = box_dims_mm(b)
            a["per_resolution"][str(e["process_res"])] = {
                "yaw_deg": box_yaw_deg(b),
                "length_mm": round(d["length"], 1) if d["length"] else None,
                "width_mm": round(d["width"], 1) if d["width"] else None,
                "views_used": box_views_used(b),
                "match_distance_mm": round(best_d * 1e3, 1),
            }

    out = []
    for a in anchors:
        pr = a["per_resolution"]
        if len(pr) < 2:
            continue
        yaws = [v["yaw_deg"] for v in pr.values() if v["yaw_deg"] is not None]
        lens = [v["length_mm"] for v in pr.values() if v["length_mm"] is not None]
        a["n_resolutions"] = len(pr)
        a["yaw_spread_deg"] = round(max(yaws) - min(yaws), 2) if len(yaws) > 1 else None
        a["length_spread_mm"] = round(max(lens) - min(lens), 1) if len(lens) > 1 else None
        out.append(a)
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

CSV_FIELDS = ["process_res", "fx_px_mean", "gsd_mm_mean", "inference_ms_median",
              "inference_ms_min", "inference_ms_max", "align_ms_median",
              "align_stride", "segment_ms", "pipeline_ms", "gpu_peak_alloc_gb",
              "n_points", "n_boxes", "n_matched", "footprint_mae_mm",
              "footprint_p95_mm", "length_mae_mm", "width_mae_mm",
              "height_mae_mm", "height_bias_mm", "n_height_scored",
              "top_face_residual_rms_mm", "top_face_residual_p95_mm",
              "multi_view_faces", "multi_view_fraction",
              "inter_view_face_spread_mm", "inter_view_face_spread_max_mm",
              "height_uncertainty_mm", "offset_drift_max_mm",
              "conveyor_distance_m", "scale_drift_pct", "deck_ok",
              "deck_tilt_deg", "deck_rms_mm", "deck_spread_mm"]


def csv_row(e):
    s = e.get("score", {}) or {}
    d = e.get("deck") or {}
    a = e.get("align_ms") or {}
    drift = e.get("offset_drift_mm") or {}
    return {
        "process_res": e["process_res"],
        "fx_px_mean": e["fx_px_mean"],
        "gsd_mm_mean": e["gsd_mm_mean"],
        "inference_ms_median": e["inference_ms"]["median"],
        "inference_ms_min": e["inference_ms"]["min"],
        "inference_ms_max": e["inference_ms"]["max"],
        "align_ms_median": a.get("median"),
        "align_stride": a.get("stride"),
        "segment_ms": e["stage_ms"]["segment"],
        "pipeline_ms": e["pipeline_ms"],
        "gpu_peak_alloc_gb": e["gpu_peak_alloc_gb"],
        "n_points": e["n_points"],
        "n_boxes": e["segmentation"]["n_boxes"],
        "n_matched": s.get("n_matched"),
        "footprint_mae_mm": s.get("footprint_mae_mm"),
        "footprint_p95_mm": s.get("footprint_p95_mm"),
        "length_mae_mm": s.get("length_mae_mm"),
        "width_mae_mm": s.get("width_mae_mm"),
        "height_mae_mm": s.get("height_mae_mm"),
        "height_bias_mm": s.get("height_bias_mm"),
        "n_height_scored": s.get("n_height_scored"),
        "top_face_residual_rms_mm": s.get("top_face_residual_rms_mm"),
        "top_face_residual_p95_mm": s.get("top_face_residual_p95_mm"),
        "multi_view_faces": s.get("multi_view_faces"),
        "multi_view_fraction": s.get("multi_view_fraction"),
        "inter_view_face_spread_mm": s.get("inter_view_face_spread_mm"),
        "inter_view_face_spread_max_mm": s.get("inter_view_face_spread_max_mm"),
        "height_uncertainty_mm": s.get("height_uncertainty_mm"),
        "offset_drift_max_mm": (round(max(drift.values()), 3) if drift else None),
        "conveyor_distance_m": e.get("conveyor_distance_m"),
        "scale_drift_pct": e.get("scale_drift_pct"),
        "deck_ok": bool(d.get("ok")),
        "deck_tilt_deg": d.get("tilt_from_axis_deg") if d.get("ok") else None,
        "deck_rms_mm": d.get("rms_mm") if d.get("ok") else None,
        "deck_spread_mm": d.get("inter_camera_spread_mm") if d.get("ok") else None,
    }


def print_table(entries, auto_align):
    print(f"\n{'=' * 122}")
    print(f"{'res':>5}  {'GSD mm':>7}  {'infer ms':>9}  {'align ms':>9}  "
          f"{'pipe ms':>8}  {'parcels':>7}  {'multi':>6}  {'foot MAE':>9}  "
          f"{'face RMS':>9}  {'view spread':>11}  {'scale %':>8}  "
          f"{'deck spread':>11}")
    for e in entries:
        s = e.get("score", {}) or {}
        d = e.get("deck") or {}
        a = e.get("align_ms") or {}

        def f(v, w, p=2):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

        multi = (f"{s.get('multi_view_faces')}/{s.get('n_faces')}"
                 if s.get("n_faces") is not None else "-")
        print(f"{e['process_res']:>5}  {f(e['gsd_mm_mean'], 7)}  "
              f"{f(e['inference_ms']['median'], 9, 0)}  {f(a.get('median'), 9, 1)}  "
              f"{f(e['pipeline_ms'], 8, 0)}  {e['segmentation']['n_boxes']:>7}  "
              f"{multi:>6}  {f(s.get('footprint_mae_mm'), 9)}  "
              f"{f(s.get('top_face_residual_rms_mm'), 9)}  "
              f"{f(s.get('inter_view_face_spread_mm'), 11)}  "
              f"{f(e.get('scale_drift_pct'), 8)}  "
              f"{f(d.get('inter_camera_spread_mm') if d.get('ok') else None, 11)}")
    print(f"{'=' * 122}")
    print("foot MAE is the term expected to scale with GSD, but it also "
          "carries the scale column:\nan unanchored scale drift of a few per "
          "cent is tens of millimetres on a 400 mm parcel\nand will swamp the "
          "sampling improvement. Face RMS and view spread are the controls.")
    if auto_align:
        print("deck spread is NOT a control here. The aligner fitted that "
              "plane and forced the\nmedians onto it, so a small value "
              "confirms only that it ran.")


def make_plot(entries, path, auto_align):
    if plt is None:
        print("[plot ] matplotlib not available; skipping the figure")
        return None
    res = [e["process_res"] for e in entries]
    lat = [e["pipeline_ms"] for e in entries]
    inf = [e["inference_ms"]["median"] for e in entries]
    aln = [(e.get("align_ms") or {}).get("median") for e in entries]
    gsd = [e["gsd_mm_mean"] for e in entries]
    foot = [(e.get("score") or {}).get("footprint_mae_mm") for e in entries]
    face = [(e.get("score") or {}).get("top_face_residual_rms_mm") for e in entries]
    view = [(e.get("score") or {}).get("inter_view_face_spread_mm") for e in entries]

    def clean(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    a = ax[0]
    for series, style, colour, label in (
            (foot, "o-", "#F47920", "footprint MAE"),
            (face, "s--", "#4D4D4D", "top-face plane RMS"),
            (view, "^:", "#8A8A8A", "inter-view face spread")):
        x, y = clean(res, series)
        if x:
            a.plot(x, y, style, color=colour, label=label)
    a.set_xlabel("process_res (px)")
    a.set_ylabel("error (mm)")
    a.set_title("Error against processing resolution")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    b = ax[1]
    for series, style, colour, label in (
            (inf, "o-", "#F47920", "inference"),
            (aln, "s--", "#4D4D4D", "auto-align solve"),
            (lat, "^:", "#8A8A8A", "pipeline total")):
        x, y = clean(res, series)
        if x:
            b.plot(x, y, style, color=colour, label=label)
    b.set_yscale("log")
    b.set_xlabel("process_res (px)")
    b.set_ylabel("time (ms, log)")
    b.set_title("Cost against processing resolution")
    b.grid(alpha=0.3, which="both")
    b.legend(fontsize=8)

    c = ax[2]
    x, y = clean(lat, foot)
    if x:
        c.plot(x, y, "o-", color="#F47920")
        for xi, yi, ri in zip(lat, foot, res):
            if yi is not None:
                c.annotate(str(ri), (xi, yi), textcoords="offset points",
                           xytext=(6, 5), fontsize=8)
    else:
        c.text(0.5, 0.5, "no footprint score\n(no ground truth matched)",
               ha="center", va="center", transform=c.transAxes,
               fontsize=9, color="#8A8A8A")
    c.axvline(100.0, color="#B00020", ls="--", lw=1.0)
    c.annotate("100 ms budget", (100.0, c.get_ylim()[1]), rotation=90,
               va="top", ha="right", fontsize=8, color="#B00020")
    c.set_xscale("log")
    c.set_xlabel("end-to-end pipeline time (ms, log)")
    c.set_ylabel("footprint MAE (mm)")
    c.set_title("Accuracy against latency"
                + (", auto-align on" if auto_align else ", offsets frozen"))
    c.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[plot ] {path}")
    return path


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if o3d is None:
        raise SystemExit("open3d is required")

    names = list(args.cameras)
    unknown = [n for n in names if n not in CAMERAS]
    if unknown:
        raise SystemExit(f"unknown camera name(s): {', '.join(unknown)}")
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} not among {names}")

    resolutions = sorted(set(int(r) for r in args.res))
    bad = [r for r in resolutions if r % PATCH]
    if bad:
        print(f"[WARN] {bad} are not multiples of {PATCH}. The backbone patches "
              f"at {PATCH} px and will round them, so the resolution used will "
              f"not be the one reported here.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_align:
        print("[mode ] auto-align ON, the operating configuration. The deck "
              "spread is therefore\n        tautological and the inter-view "
              "face disagreement is the control.")
        if args.repeats < 4:
            print(f"[WARN] --repeats {args.repeats} with ema "
                  f"{args.auto_align_ema} leaves the offset at roughly "
                  f"{1 - (1 - args.auto_align_ema) ** args.repeats:.0%} of its "
                  f"converged value.")
    else:
        print("[mode ] auto-align OFF. This is not how the cell runs.")

    if getattr(args, "seg_plane_distance", None) is None:
        print("[WARN] no --seg-plane-distance given, so the conveyor is "
              "whichever plane is largest.\n        On this rig that is the "
              "floor, not the deck, and every parcel height is then\n"
              "        measured above the floor. It takes a BAND, MIN_M "
              "MAX_M, and the band is in\n        the model's coordinates "
              "rather than physical metres: unanchored, the deck\n        "
              "reads several per cent short and moves with the resolution. "
              f"Try\n        --seg-plane-distance "
              f"{args.deck_distance * 0.88:.2f} "
              f"{args.deck_distance * 1.01:.2f}, and read the deck line below "
              f"to see\n        where the plane actually sits before "
              f"tightening it.")
    if getattr(args, "seg_belt_file", None) is None:
        print("[WARN] no --seg-belt-file given, so the belt footprint is "
              "re-solved from each\n        resolution's own cloud and moves "
              "with the depth scale, taking the parcel\n        search bounds "
              "with it. Measure it once and freeze it.")

    if not (args.gt or args.gt_json):
        print("[WARN] no ground truth supplied, so footprint accuracy cannot "
              "be scored. Latency, plane residuals and the inter-view spread "
              "are still reported. Pass --gt L W H in millimetres.")
    gt = load_ground_truth(args)

    seg_params = seg_params_from_args(args)

    # ---- depth correction ------------------------------------------------
    per_cam_affine, global_affine = None, (1.0, 0.0)
    if args.load_affine is not None:
        per_cam_affine, global_affine, _ = load_affine(
            args.load_affine, names, args.apply_absolute)
        print(f"[corr ] scale terms and anchor from {args.load_affine}")
        if args.auto_align:
            print("        the offsets in that file are overridden by the "
                  "aligner, as they are live")
    elif args.apply_absolute:
        print("[WARN] --apply-absolute does nothing without --load-affine; the "
              "anchor lives in that file.")
    elif args.auto_align:
        print("[corr ] no --load-affine: scale terms are unity and only the "
              "offsets are solved.\n        DA3's absolute scale is framing "
              "dependent, so the footprint figures below\n        carry a "
              "resolution dependent scale term. Watch the scale drift column.")

    affine_meta = {"source": str(args.load_affine) if args.load_affine else None,
                   "absolute_applied": bool(args.apply_absolute),
                   "auto_align": bool(args.auto_align),
                   "auto_align_ema": args.auto_align_ema}

    # ---- calibration and frames ------------------------------------------
    intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
    ext = load_extrinsics(args.calib_dir, names, args.reference)
    Es = np.stack([ext[n] for n in names])

    maps, Ks_rect = {}, {}
    for n in names:
        m, newK = build_maps(intr[n]["K"], intr[n]["dist"],
                             (args.width, args.height), args.undistort)
        maps[n] = m
        Ks_rect[n] = newK

    if args.frames_dir is not None:
        frames_d = load_frames(args.frames_dir, names)
    else:
        frames_d = grab_static_set(names, args, maps)
        if args.save_frames:
            save_frames(frames_d, names, args.out_dir / "frames")

    sizes = {n: (frames_d[n].shape[1], frames_d[n].shape[0]) for n in names}
    print(f"[rect ] rectified sizes: {sizes}")

    valid_box = None
    if args.pad_square:
        padded, valid_box = [], []
        for n in names:
            img, fx_frac, fy_frac = pad_square(frames_d[n])
            padded.append(img)
            valid_box.append((fx_frac, fy_frac))
        frames = padded
        print(f"[pad  ] padded to square; valid fractions "
              f"{[(round(a, 3), round(b, 3)) for a, b in valid_box]}")
    else:
        frames = [frames_d[n] for n in names]
        if len(set(sizes.values())) > 1:
            print("[WARN] the rectified images do not share a size, so the "
                  "batch will be centre-cropped. Pass --pad-square.")

    Ks_in = np.stack([Ks_rect[n] for n in names])

    print(f"[model] loading {args.model}")
    t0 = time.perf_counter()
    model = load_model(args.model, args.device)
    print(f"[model] ready in {time.perf_counter() - t0:.1f} s")

    # ---- sweep -----------------------------------------------------------
    entries = []
    warned = {"dims": False, "resid": False, "spread": False}
    for res in resolutions:
        try:
            e = run_resolution(res, model, names, frames, Ks_in, Es, args,
                               seg_params, per_cam_affine, global_affine,
                               valid_box, args.out_dir, affine_meta)
        except RuntimeError as exc:
            # Running out of memory at the top of the sweep is a result rather
            # than a crash: it is the point at which the resolution stops being
            # available on this hardware at all.
            print(f"[FAIL ] process_res {res}: {exc!r}")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            entries.append({"process_res": res, "failed": repr(exc)})
            continue
        entries.append(score(e, gt, args, warned))

    ok = [e for e in entries if "failed" not in e]
    if not ok:
        raise SystemExit("every resolution failed; nothing to report")

    # ---- scale probe -----------------------------------------------------
    # On identical pixels the fitted conveyor distance should not move with
    # resolution. If it does, that drift is a multiplicative error on every
    # dimension reported at that resolution and has to be read alongside the
    # footprint column rather than after it.
    # deck_check fits the same surface and succeeds even when segmentation
    # does not, which is exactly when the probe is most wanted: the run that
    # rejected every conveyor candidate reported a null drift at all four
    # resolutions while its own deck fit had already measured 5.5 per cent.
    for e in ok:
        if e.get("conveyor_distance_m") is None:
            d = e.get("deck") or {}
            if d.get("ok"):
                e["conveyor_distance_m"] = round(abs(float(d["offset_m"])), 4)
                e["conveyor_distance_source"] = "deck_check"
        elif e.get("conveyor_distance_m") is not None:
            e.setdefault("conveyor_distance_source", "segmentation")

    ref_dist = next((e.get("conveyor_distance_m") for e in ok
                     if e.get("conveyor_distance_m")), None)
    for e in ok:
        v = e.get("conveyor_distance_m")
        e["scale_drift_pct"] = (round((v / ref_dist - 1.0) * 100, 3)
                                if v and ref_dist else None)
    drifts = [e["scale_drift_pct"] for e in ok if e.get("scale_drift_pct") is not None]
    if drifts and max(abs(v) for v in drifts) > 1.0:
        worst = max(drifts, key=abs)
        print(f"\n[WARN] the fitted conveyor distance drifts {worst:+.1f} per "
              f"cent across the sweep on identical pixels. On a 400 mm parcel "
              f"that is {abs(worst) * 4:.0f} mm of dimensional error, which is "
              f"larger than the sampling improvement the resolution buys. The "
              f"footprint column is not comparable across rows until the scale "
              f"is anchored, and because the drift is itself resolution "
              f"dependent one frozen anchor cannot serve the whole sweep: "
              f"layer_align.py has to be run per resolution.")

    # ---- clustering against the measured disagreement --------------------
    eps_mm = float(getattr(seg_params, "cluster_eps", 0.02)) * 1e3
    worst_spread = max((e["score"].get("inter_view_face_spread_max_mm") or 0.0)
                       for e in ok)
    if worst_spread > eps_mm:
        print(f"[WARN] the cameras disagree by up to {worst_spread:.0f} mm on "
              f"parcel top faces while --seg-cluster-eps is {eps_mm:.0f} mm. "
              f"Two views of one face therefore fall into separate clusters "
              f"and one parcel is reported as several, each carried by a "
              f"single camera. That is why the multi-view fraction is low and "
              f"why most faces are unpickable under --seg-min-views 2.")

    print_table(ok, args.auto_align)

    tracked = match_across_resolutions(ok, args.gt_match_radius)
    if tracked:
        print(f"\nparcels tracked across resolutions by plan-view position "
              f"({len(tracked)} of {ok[0]['segmentation']['n_boxes']} anchors "
              f"matched in two or more rows)")
        for t in tracked:
            print(f"        at ({t['anchor_xy'][0]:+.2f}, "
                  f"{t['anchor_xy'][1]:+.2f}) m over {t['n_resolutions']} "
                  f"resolutions: yaw spread "
                  f"{t['yaw_spread_deg'] if t['yaw_spread_deg'] is not None else float('nan'):6.2f} deg   "
                  f"length spread "
                  f"{t['length_spread_mm'] if t['length_spread_mm'] is not None else float('nan'):6.1f} mm")
        print("        Yaw spread is repeatability, not accuracy: the true "
              "yaw of a hand-placed\n        parcel is not known to better "
              "than the quantity being measured.")

    counts = {e["process_res"]: e["segmentation"]["n_boxes"] for e in ok}
    if len(set(counts.values())) > 1:
        print(f"\n[WARN] the parcel count is not constant across the sweep: "
              f"{counts}. A resolution that finds a different number of "
              f"parcels is not being compared on the same population.")

    odrifts = [max((e.get("offset_drift_mm") or {"_": 0.0}).values()) for e in ok]
    if odrifts and max(odrifts) > 5.0:
        print(f"[WARN] the auto-align offset drifted up to {max(odrifts):.1f} "
              f"mm across passes on a static scene. That noise is present in "
              f"the live system on every frame.")

    if args.dump_arrays:
        print(f"\n[dump ] the anchoring loop, one solve per resolution, since "
              f"DA3's scale is\n        resolution dependent and a single "
              f"frozen anchor cannot serve the sweep:")
        for e in ok:
            print(f"        python layer_align.py --capture-dir "
                  f"{e['output_dir']} --mode both --holdout --n-planes 3")
        print(f"        then re-run this sweep once per resolution with "
              f"--load-affine <that\n        resolution's depth_affine.json> "
              f"--apply-absolute. Check da3_fuse.GT_DECK_PERP_M\n        is "
              f"the deck standoff from the reference camera first: the whole "
              f"anchor rests\n        on it.")

    csv_path = args.out_dir / args.csv
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for e in ok:
            w.writerow(csv_row(e))

    fig_path = make_plot(ok, args.out_dir / "sweep.png", args.auto_align) \
        if args.plot else None

    report = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": ("auto-align, the operating configuration" if args.auto_align
                 else "frozen offsets, not the operating configuration"),
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        "cameras": {n: CAMERAS[n] for n in names},
        "rectified_sizes": {n: list(sizes[n]) for n in names},
        "ground_truth": gt,
        "depth_correction": affine_meta,
        "controls": {
            "valid": ["top_face_residual_rms_mm",
                      "inter_view_face_spread_mm (multi-view faces only)"],
            "tautological_under_auto_align": ["deck_spread_mm"],
            "confounded_without_an_anchor": ["footprint_mae_mm",
                                             "height_mae_mm"],
        },
        "scale_probe": {
            "reference_conveyor_distance_m": ref_dist,
            "drift_pct_by_resolution": {str(e["process_res"]):
                                        e.get("scale_drift_pct") for e in ok},
        },
        "resolutions": entries,
        "parcels_tracked_across_resolutions": tracked,
        "parcel_counts": counts,
        "figure": str(fig_path) if fig_path else None,
    }
    (args.out_dir / args.json).write_text(json.dumps(report, indent=2, default=str))

    print(f"\ntable  : {csv_path}")
    print(f"report : {args.out_dir / args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())