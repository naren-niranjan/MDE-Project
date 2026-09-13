#!/usr/bin/env python3
"""
res_sweep.py

Accuracy against latency as a function of DA3 processing resolution, measured
on one static scene, under the correction stack the system actually runs.

Revision note, the aligner interface
------------------------------------
OnlineAligner.update() now returns per-camera (a, b) rather than only the
offsets, because the aligner can solve both terms when a second reference
surface is available. Three sites here consumed the old shape and raised

    TypeError: can't multiply sequence by non-int of type 'float'

on the first timed pass. They are fixed, and the sweep now also records the
per-view SCALE alongside the offset so that a resolution which changes the
solved scale is visible rather than implied.

The sweep defaults to the OFFSET-ONLY aligner, which is the configuration this
cell runs. The two-plane solve is available with --auto-align-two-plane, but on
this rig it is not identifiable: the workspace gives 400 mm of height against a
3.1 m standoff, so a per-plane offset noise of about 4 mm becomes 1.7 per cent
of slope uncertainty and 55 mm of intercept swing, which is larger than the
bias being corrected. The bounded quantity that survives is the focal spread,
0.77 per cent across this rig, which over the 400 mm working range is 3 mm.

Earlier revision notes, retained
--------------------------------
  1  parcel dimensions live at box["dimensions_m"] in METRES
  2  the top-face residual lives at box["top_face"]["plane_residual_rms_m"],
     not at view_consensus.per_view[0].residual_rms_mm, which is ONE VIEW'S fit
  3  inter_view_face_spread_mm is taken over MULTI-VIEW faces only, since on a
     single-view face it is zero by construction
  4  deck_check rejects planes whose normal is more than --deck-max-tilt from
     the reference optical axis, because at three of four resolutions the
     largest plane in the band was a wall
  5  parcels are matched across resolutions by plan-view position, not by list
     index, which is meaningless when the count changes between rows

Operating mode
--------------
Auto-align is ON by default, because that is how the cell runs. Every
consequence of that choice is built into the measurement:

  the aligner converges
      the offset is smoothed by an EMA, so one call is not the value the
      running system would use. The --repeats passes double as convergence and
      the spread across passes is reported as a stability figure

  the aligner costs time
      its solve runs on the depth map at --auto-align-stride, so at a fixed
      stride its cost grows with the pixel count. --auto-align-stride-scale
      holds the sample count constant instead

  the deck control becomes tautological
      the aligner fits the deck and forces every camera's median onto it, so
      inter-camera spread AT THE DECK is near zero at every resolution by
      construction. The control that survives is the inter-view disagreement on
      the PARCEL TOP FACES

What this answers
-----------------
Which reported quantity degrades with resolution, and what recovering it costs.

  footprint length and width
      expected to improve roughly in proportion to GSD, PROVIDED the metric
      scale is anchored. It is not anchored by auto-align, so without
      --load-affine the footprint column measures scale drift as much as
      sampling. The scale probe exists to make that visible

  top-face plane residual
      expected to be flat

  inter-view face disagreement
      expected to be flat; it is a structured bias, not noise

  REFERENCE FLATNESS
      not measured here directly. Run --dump-arrays and then, per resolution,
          python layer_align.py --capture-dir res_NNNN --mode diagnose \\
              --assign-tol 0.015 --plane-thresh 0.006 --n-planes 1
      and read the [flat ] line. On the 12 mm rig the reference camera
      reconstructs its own deck with 6.6 mm rms and a -12.5 to +5.8 mm bow at
      504 px. No inter-camera alignment can produce a residual below that, so
      whether it falls with resolution decides whether the sampling or the
      model is the limit.

Ground truth
------------
    --gt 400 300 250                 repeatable, millimetres, L W H
    --gt-json parcels.json           [{"length_mm":400, ..., "x":0.1, "y":-0.2}]

Outputs
-------
  sweep.csv, sweep.json, sweep.png, res_<n>/

Keep beside da3_stream.py, da3_fuse.py, box_segment.py, face_consensus.py and
online_align.py.
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
BASE_RES = 504
OPTICAL_AXIS = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sweep DA3 processing resolution on one static scene, "
                    "under the correction the cell runs, and report accuracy "
                    "against latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/ressweep"))
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")

    ap.add_argument("--res", nargs="+", type=int, default=[504, 616, 728, 1008],
                    help="processing resolutions to test, multiples of 14")
    ap.add_argument("--repeats", type=int, default=6,
                    help="timed inference passes per resolution; these also "
                         "serve as the aligner's convergence")
    ap.add_argument("--warmup-per-res", type=int, default=2,
                    help="passes discarded at each resolution before timing. "
                         "Each new resolution retriggers kernel autotuning")

    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="load a previously saved rectified frame set instead "
                         "of grabbing one, so two sweeps see identical pixels")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--settle-frames", type=int, default=5)

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
                    help="pad each frame to square on the RIGHT AND BOTTOM "
                         "before inference, so a batch of mixed aspect ratios "
                         "is not centre-cropped. Padding the far edges leaves "
                         "the principal point untouched")

    # depth correction
    ap.add_argument("--auto-align", dest="auto_align", action="store_true",
                    default=True,
                    help="re-solve the per-view offset from the scene's "
                         "dominant plane. ON by default because the cell runs "
                         "this way")
    ap.add_argument("--no-auto-align", dest="auto_align", action="store_false",
                    help="freeze the coefficients at whatever --load-affine "
                         "supplies. Not the operating configuration")
    ap.add_argument("--auto-align-two-plane", dest="auto_align_two_plane",
                    action="store_true", default=False,
                    help="also solve the per-view SCALE from a second surface "
                         "above the deck. OFF by default: with 400 mm of "
                         "workspace height against a 3.1 m standoff the slope "
                         "is not identifiable, and the solved intercept swings "
                         "by tens of millimetres between passes on a static "
                         "scene. The bounded scale error that remains is the "
                         "0.77 per cent focal spread, 3 mm over the working "
                         "range")
    ap.add_argument("--auto-align-upper-band", nargs=2, type=float,
                    default=[0.120, 0.600], metavar=("MIN_M", "MAX_M"),
                    help="height band searched for that second surface; used "
                         "only with --auto-align-two-plane")
    ap.add_argument("--auto-align-min-span", type=float, default=0.080)
    ap.add_argument("--auto-align-max-scale-step", type=float, default=0.010)
    ap.add_argument("--auto-align-ema", type=float, default=0.3,
                    help="smoothing on the coefficients, matching the cell")
    ap.add_argument("--auto-align-stride", type=int, default=3,
                    help="pixel stride when fitting the plane, quoted at 504")
    ap.add_argument("--auto-align-stride-scale", action="store_true",
                    help="scale the stride with the resolution so the aligner "
                         "sees a constant number of samples")
    ap.add_argument("--auto-align-tol", type=float, default=0.050)
    ap.add_argument("--auto-align-max-step", type=float, default=0.15)
    ap.add_argument("--load-affine", type=Path, default=None,
                    help="supplies the per-view SCALE terms and the absolute "
                         "anchor. The offsets in it are overridden by the "
                         "aligner. Without this the footprint column carries "
                         "DA3's resolution dependent scale drift")
    ap.add_argument("--apply-absolute", action="store_true")

    # filtering, matching da3_stream.py defaults exactly
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--conf-min", type=float, default=0.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--voxel", type=float, default=0.004)

    # deck plane, reported as an aligner check rather than as a control
    ap.add_argument("--deck-distance", type=float, default=3.145,
                    help="expected deck standoff along the reference camera's "
                         "optical axis in DA3's UNANCHORED coordinates, not "
                         "physical metres. Read it off box_segment.py --probe")
    ap.add_argument("--deck-tol", type=float, default=0.06,
                    help="half-width of that band; must stay below the "
                         "shortest parcel")
    ap.add_argument("--deck-max-tilt", type=float, default=15.0,
                    help="reject a candidate deck plane whose normal lies "
                         "further than this from the reference optical axis")
    ap.add_argument("--deck-max-attempts", type=int, default=4)

    # ground truth
    ap.add_argument("--gt", nargs=3, type=float, action="append",
                    metavar=("L_MM", "W_MM", "H_MM"))
    ap.add_argument("--gt-json", type=Path, default=None)
    ap.add_argument("--gt-match-radius", type=float, default=0.25)

    # arrays for the anchoring solve
    ap.add_argument("--dump-arrays", action="store_true",
                    help="write depth, conf, K and E per camera into each "
                         "res_<n>/ directory, in the layout layer_align.py "
                         "reads. This is also how the REFERENCE FLATNESS is "
                         "measured per resolution")
    ap.add_argument("--dump-depth", choices=["raw", "corrected"], default="raw",
                    help="which depth goes into depth_<cam>.npy. RAW by "
                         "default, because da3_stream.py applies loaded "
                         "coefficients to the raw model output. Note this is "
                         "the OPPOSITE of da3_stream.py's capture convention; "
                         "arrays.json records which is which")
    ap.add_argument("--dump-images", dest="dump_images", action="store_true",
                    default=True)
    ap.add_argument("--no-dump-images", dest="dump_images", action="store_false")

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
    carries no penalty, and that is the reason the sweep is run static: it
    removes sync from the list of things that could explain a difference
    between two resolutions.
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
    a hard black border reads to the model as a depth discontinuity.
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

def box_dims_mm(b):
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
    """RMS residual of the COMBINED top-face plane fit, in millimetres."""
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
    """Inter-view disagreement before reconciliation.

    Zero by construction on a single-view face, which is why the caller must
    exclude those before taking a median.
    """
    v = _consensus(b).get("inter_view_offset_mm")
    return float(v) if v is not None else None


def box_height_uncertainty_mm(b):
    v = _consensus(b).get("height_uncertainty_mm")
    return float(v) if v is not None else None


def box_centre(b):
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
    resolution; when it does, that drift is a multiplicative error on every
    dimension reported at that resolution.
    """
    c = seg.get("conveyor")
    if not isinstance(c, dict):
        return None
    for k in ("distance_from_reference_camera_m", "distance_m", "offset_m"):
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
    """The same filter chain and back-projection as a live capture."""
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

    Candidates are rejected unless their normal lies within --deck-max-tilt of
    the reference optical axis, and the fit is retried on the points the
    rejected candidates did not claim. Without that, the conveyor's side frame
    and the machine cabinet win, and every per-camera median then refers to a
    wall.

    Under auto-align this is a CHECK rather than a control: the aligner solved
    the offsets against this plane, so the spread is near zero by construction.
    """
    if len(pts) == 0:
        return None
    band = np.abs(pts[:, 2] - args.deck_distance) <= args.deck_tol
    if int(band.sum()) < 500:
        return {"ok": False,
                "reason": f"only {int(band.sum())} points within "
                          f"{args.deck_tol * 1e3:.0f} mm of the expected deck "
                          f"standoff {args.deck_distance:.2f} m. Either the "
                          f"deck is not there or --deck-distance is stale: it "
                          f"is in DA3's unanchored coordinates, not physical "
                          f"metres, and it moves with the lens and the "
                          f"resolution"}

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
                         "this plane and forced the medians onto it"
                         if auto_align else
                         "a genuine control: the offsets were frozen"),
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
# arrays for the anchoring and flatness solves
# --------------------------------------------------------------------------

def dump_arrays(stem, names, depth_raw, depth, conf, K_out, E_out, rgb_proc,
                valid_box, args, res):
    """Write the per-camera arrays layer_align.py needs, per resolution.

    Three things matter about what is written.

    RAW depth by default. da3_stream.py applies loaded coefficients to the raw
    model output, and layer_align.py likewise extracts its reference planes
    from uncorrected depth. Coefficients solved on already-corrected depth are
    a correction of a correction. Note this INVERTS da3_stream.py's own capture
    convention, so arrays.json records which convention this directory uses.

    The padded border is zeroed. With --pad-square the replicated edge carries
    invented depth that back-projection excludes through valid_box, but
    layer_align.py has no such mask and admits any sample with d > 0.

    The intrinsics are DA3's returned values, which belong to the processed
    image, so the RGB written beside them is the processed image.
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
                                "the CORRECTED depth"),
            f"{secondary_stem}_<cam>.npy": ("the corrected depth"
                                            if which == "raw" else
                                            "the raw model output"),
            "note": ("this inverts da3_stream.py's capture convention, in "
                     "which depth_<cam>.npy is the corrected depth"),
        },
        "intrinsics": ("DA3's returned intrinsics for the processed image. "
                       "<cam>.png is that processed image."),
        "padded_border": ("zeroed" if valid_box is not None else "not applicable"),
        "filters_NOT_applied": ("none of the confidence, edge or incidence "
                                "filters have been applied to these arrays"),
        "per_camera": per_cam,
        "reference_flatness_check": (
            f"python layer_align.py --capture-dir {stem} --mode diagnose "
            f"--assign-tol 0.015 --plane-thresh 0.006 --n-planes 1"),
        "anchor_solve": (f"python layer_align.py --capture-dir {stem} "
                         f"--mode both --holdout --n-planes 3"),
    }
    (stem / "arrays.json").write_text(json.dumps(manifest, indent=2))
    print(f"[dump ] arrays written to {stem} ({which} depth in depth_<cam>.npy)")
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
    # Carrying one across resolutions would let coefficients solved at 504 leak
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
                                max_step=args.auto_align_max_step,
                                two_plane=args.auto_align_two_plane,
                                upper_min_height=args.auto_align_upper_band[0],
                                upper_max_height=args.auto_align_upper_band[1],
                                min_span=args.auto_align_min_span,
                                max_scale_step=args.auto_align_max_scale_step)
        print(f"[corr ] auto-align on, "
              f"{'TWO planes' if args.auto_align_two_plane else 'offset only'}, "
              f"stride {stride}, ema {args.auto_align_ema}, converging over "
              f"{args.repeats} passes")

    for k in range(max(0, args.warmup_per_res)):
        t0 = time.perf_counter()
        infer()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[warm ] discarded pass {k + 1}, "
              f"{(time.perf_counter() - t0) * 1e3:.0f} ms")

    # ---- timed passes, with the aligner converging alongside -------------
    times, align_ms = [], []
    offset_history, scale_history = [], []
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
            # update() returns {name: (a, b)}, not the bare offsets.
            coeffs, align_diag = aligner.update(depth_raw, K_out, E_out)
            align_ms.append((time.perf_counter() - t1) * 1e3)
            offset_history.append({n: round(coeffs[n][1] * 1e3, 3) for n in names})
            scale_history.append({n: round(coeffs[n][0], 6) for n in names})
            if align_diag is not None and not align_diag.get("ok"):
                print(f"[WARN] pass {r}: auto-align failed, "
                      f"{align_diag.get('reason')}. The coefficients in use "
                      f"are the previous ones, so this resolution is not being "
                      f"measured under the correction it appears to be.")
            elif align_diag is not None and align_diag.get("degraded"):
                print(f"[note ] pass {r}: no second reference surface, offset "
                      f"solved alone")
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

    # ---- coefficient stability across passes ----------------------------
    stability, scale_stability = None, None
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
    if len(scale_history) > 1 and args.auto_align_two_plane:
        tail = scale_history[1:]
        scale_stability = {n: round(max(o[n] for o in tail)
                                    - min(o[n] for o in tail), 6)
                           for n in names}
        print(f"[align] scale {scale_history[-1]}   drift {scale_stability}")
        worst = max(scale_stability.values())
        if worst > 0.003:
            print(f"[WARN] the per-view scale moved {worst * 100:.2f} per cent "
                  f"across passes on a static scene, which is "
                  f"{worst * args.deck_distance * 1e3:.0f} mm at the deck. The "
                  f"slope is not identifiable over this workspace; run without "
                  f"--auto-align-two-plane.")

    # ---- correction ------------------------------------------------------
    t0 = time.perf_counter()
    if aligner is not None:
        # The aligner owns both terms now. Overriding its scale with the stored
        # one would discard whatever the two-plane solve found, and in
        # offset-only mode its scale IS the stored one, so this is correct in
        # both modes.
        eff = aligner.coefficients()
        depth = apply_affine(depth_raw, names, eff, global_affine)
        offsets_mm = {n: round(eff[n][1] * 1e3, 2) for n in names}
        scales = {n: round(eff[n][0], 6) for n in names}
    elif per_cam_affine is not None:
        depth = apply_affine(depth_raw, names, per_cam_affine, global_affine)
        offsets_mm = {n: round(per_cam_affine[n][1] * 1e3, 2) for n in names}
        scales = {n: round(per_cam_affine[n][0], 6) for n in names}
    else:
        depth = depth_raw
        offsets_mm = {n: 0.0 for n in names}
        scales = {n: 1.0 for n in names}
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
    # still yields the arrays the anchor and the flatness check are solved from.
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
        "align_ms": ({"median": round(align_med, 2), "stride": stride,
                      "all": [round(v, 2) for v in align_ms]}
                     if align_ms else None),
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
        "scales": scales,
        "offset_history_mm": offset_history,
        "scale_history": scale_history,
        "offset_drift_mm": stability,
        "scale_drift": scale_stability,
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

    The inter-view spread and the height uncertainty are taken over MULTI-VIEW
    faces only. On a single-view face the spread is zero because there is
    nothing to disagree with.
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
        # so scoring it against a ground-truth height compares two different
        # quantities.
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
              f"about the rig rather than a gap in the report.")

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
            "against that entry instead. Supply x and y in --gt-json.")
    return entry


def match_across_resolutions(entries, radius):
    """Track the same physical parcel across resolutions by plan-view position.

    Comparing parcel LIST INDEX between rows is meaningless when segmentation
    returns 9 parcels at one resolution and 12 at another. Anchors are taken
    from the lowest resolution and each later resolution contributes the
    nearest parcel within the radius.

    Yaw is reported as a spread rather than an error, because the parcels are
    placed by hand and their true yaw is not known to better than the quantity
    being measured.
    """
    if not entries:
        return []
    ref = entries[0]
    anchors = []
    for bi, b in enumerate(ref["segmentation"]["boxes"]):
        c = box_centre(b)
        if c is None:
            continue
        anchors.append({"anchor_xy": [round(float(c[0]), 4),
                                      round(float(c[1]), 4)],
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
              "scale_drift_max", "align_model",
              "conveyor_distance_m", "scale_drift_pct", "deck_ok",
              "deck_tilt_deg", "deck_rms_mm", "deck_spread_mm"]


def csv_row(e):
    s = e.get("score", {}) or {}
    d = e.get("deck") or {}
    a = e.get("align_ms") or {}
    drift = e.get("offset_drift_mm") or {}
    sdrift = e.get("scale_drift") or {}
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
        "scale_drift_max": (round(max(sdrift.values()), 6) if sdrift else None),
        "align_model": (e.get("auto_align") or {}).get("model"),
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
    print("foot MAE also carries the scale column: an unanchored drift of a "
          "few per cent is\ntens of millimetres on a 400 mm parcel and will "
          "swamp the sampling improvement.\nFace RMS and view spread are the "
          "controls.")
    if auto_align:
        print("deck spread is NOT a control here. The aligner fitted that "
              "plane and forced the\nmedians onto it.")
    print("\nThe REFERENCE FLATNESS is not in this table. With --dump-arrays, "
          "run per resolution:\n  python layer_align.py --capture-dir "
          "res_NNNN --mode diagnose --assign-tol 0.015 \\\n"
          "      --plane-thresh 0.006 --n-planes 1\nand read the [flat ] line. "
          "No inter-camera alignment can beat it.")


def make_plot(entries, path, auto_align):
    if plt is None:
        print("[plot ] matplotlib not available; skipping the figure")
        return None
    res = [e["process_res"] for e in entries]
    lat = [e["pipeline_ms"] for e in entries]
    inf = [e["inference_ms"]["median"] for e in entries]
    aln = [(e.get("align_ms") or {}).get("median") for e in entries]
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
                + (", auto-align on" if auto_align else ", coefficients frozen"))
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
        print(f"[WARN] {bad} are not multiples of {PATCH}; the backbone will "
              f"round them, so the resolution used is not the one reported.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_align:
        mode = ("two-plane" if args.auto_align_two_plane else "offset-only")
        print(f"[mode ] auto-align ON, {mode}. The deck spread is therefore "
              f"tautological and\n        the inter-view face disagreement is "
              f"the control.")
        if args.auto_align_two_plane:
            print("[WARN] the two-plane solve is not identifiable over a "
                  "400 mm workspace at a\n       3.1 m standoff: 4 mm of "
                  "per-plane noise is 1.7 per cent of slope and\n       55 mm "
                  "of intercept swing. Watch the scale drift lines.")
        if args.repeats < 4:
            print(f"[WARN] --repeats {args.repeats} with ema "
                  f"{args.auto_align_ema} leaves the coefficients at roughly "
                  f"{1 - (1 - args.auto_align_ema) ** args.repeats:.0%} of "
                  f"converged.")
    else:
        print("[mode ] auto-align OFF. This is not how the cell runs.")

    if getattr(args, "seg_plane_distance", None) is None:
        print("[WARN] no --seg-plane-distance given, so the conveyor is "
              "whichever plane is largest.\n        On this rig that may be "
              "the floor or a wall, and every parcel height is\n        then "
              "measured above the wrong surface. It takes a BAND, MIN_M "
              "MAX_M, in the\n        model's coordinates rather than physical "
              f"metres. Try\n        --seg-plane-distance "
              f"{args.deck_distance - 0.06:.2f} "
              f"{args.deck_distance + 0.06:.2f}, and read the deck line below.")
    if getattr(args, "seg_belt_file", None) is None:
        print("[WARN] no --seg-belt-file given, so the belt footprint is "
              "re-solved from each\n        resolution's own cloud and moves "
              "with the depth scale.")

    if not (args.gt or args.gt_json):
        print("[WARN] no ground truth supplied, so footprint accuracy cannot "
              "be scored. Latency,\n        plane residuals and the "
              "inter-view spread are still reported. Pass\n        "
              "--gt L W H in millimetres.")
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
        print("[WARN] --apply-absolute does nothing without --load-affine.")
    elif args.auto_align:
        print("[corr ] no --load-affine: scale terms are unity and only the "
              "offsets are solved.\n        DA3's absolute scale is framing "
              "dependent, so the footprint figures below\n        carry a "
              "resolution dependent scale term. Watch the scale drift column.")

    affine_meta = {"source": str(args.load_affine) if args.load_affine else None,
                   "absolute_applied": bool(args.apply_absolute),
                   "auto_align": bool(args.auto_align),
                   "auto_align_two_plane": bool(args.auto_align_two_plane),
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
    # dimension reported at that resolution. deck_check fits the same surface
    # and succeeds even when segmentation does not, which is exactly when the
    # probe is most wanted.
    for e in ok:
        if e.get("conveyor_distance_m") is None:
            d = e.get("deck") or {}
            if d.get("ok"):
                e["conveyor_distance_m"] = round(abs(float(d["offset_m"])), 4)
                e["conveyor_distance_source"] = "deck_check"
        else:
            e.setdefault("conveyor_distance_source", "segmentation")

    ref_dist = next((e.get("conveyor_distance_m") for e in ok
                     if e.get("conveyor_distance_m")), None)
    for e in ok:
        v = e.get("conveyor_distance_m")
        e["scale_drift_pct"] = (round((v / ref_dist - 1.0) * 100, 3)
                                if v and ref_dist else None)
    drifts = [e["scale_drift_pct"] for e in ok
              if e.get("scale_drift_pct") is not None]
    if drifts and max(abs(v) for v in drifts) > 1.0:
        worst = max(drifts, key=abs)
        print(f"\n[WARN] the fitted conveyor distance drifts {worst:+.1f} per "
              f"cent across the sweep on identical pixels. On a 400 mm parcel "
              f"that is {abs(worst) * 4:.0f} mm of dimensional error, larger "
              f"than the sampling improvement the resolution buys. Because the "
              f"drift is itself resolution dependent, one frozen anchor cannot "
              f"serve the whole sweep: layer_align.py has to be run per "
              f"resolution.")

    # ---- clustering against the measured disagreement --------------------
    eps_mm = float(getattr(seg_params, "cluster_eps", 0.02)) * 1e3
    worst_spread = max((e["score"].get("inter_view_face_spread_max_mm") or 0.0)
                       for e in ok)
    if worst_spread > eps_mm:
        print(f"[WARN] the cameras disagree by up to {worst_spread:.0f} mm on "
              f"parcel top faces while --seg-cluster-eps is {eps_mm:.0f} mm, "
              f"so two views of one face fall into separate clusters and one "
              f"parcel is reported as several, each carried by a single "
              f"camera.")

    print_table(ok, args.auto_align)

    tracked = match_across_resolutions(ok, args.gt_match_radius)
    if tracked:
        print(f"\nparcels tracked across resolutions by plan-view position "
              f"({len(tracked)} of {ok[0]['segmentation']['n_boxes']} anchors "
              f"matched in two or more rows)")
        for t in tracked:
            ys = t['yaw_spread_deg']
            ls = t['length_spread_mm']
            print(f"        at ({t['anchor_xy'][0]:+.2f}, "
                  f"{t['anchor_xy'][1]:+.2f}) m over {t['n_resolutions']} "
                  f"resolutions: yaw spread "
                  f"{ys if ys is not None else float('nan'):6.2f} deg   "
                  f"length spread "
                  f"{ls if ls is not None else float('nan'):6.1f} mm")
        print("        Yaw spread is repeatability, not accuracy.")

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
        print(f"\n[dump ] per-resolution reference flatness, which bounds every "
              f"alignment figure:")
        for e in ok:
            print(f"        python layer_align.py --capture-dir "
                  f"{e['output_dir']} --mode diagnose --assign-tol 0.015 "
                  f"--plane-thresh 0.006 --n-planes 1")
        print(f"\n[dump ] and the anchoring loop, one solve per resolution:")
        for e in ok:
            print(f"        python layer_align.py --capture-dir "
                  f"{e['output_dir']} --mode both --holdout --n-planes 3")

    csv_path = args.out_dir / args.csv
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for e in ok:
            w.writerow(csv_row(e))

    fig_path = (make_plot(ok, args.out_dir / "sweep.png", args.auto_align)
                if args.plot else None)

    report = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": ("auto-align "
                 + ("two-plane" if args.auto_align_two_plane else "offset-only")
                 if args.auto_align else "frozen coefficients"),
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
            "not_measured_here": ["reference_flatness, see arrays.json"],
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