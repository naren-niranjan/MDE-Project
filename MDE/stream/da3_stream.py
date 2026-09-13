#!/usr/bin/env python3
"""
da3_stream.py

Live four-camera depth streaming with DA3, with per-view depth correction,
point clouds captured on demand, parcel segmentation on capture, and
continuous performance logging.

Behaviour
---------
The four cameras run free at their native rate into a latest-frame slot each.
The main thread takes whatever set is current, runs DA3 once, applies the
per-view depth correction and displays the resulting depth maps.
Back-projection, fusion and segmentation do NOT run in the streaming loop:
they run only when a capture is requested, so the streaming rate is not paying
for a point cloud it will discard.

Depth correction
----------------
DA3 carries a per-view depth bias that puts the four back-projected clouds on
parallel offset sheets of the same physical surface. Measured on this rig the
offsets were left -6, centre +2, right -11 and top -58 mm, near-constant
across an 840 mm change in distance, so the bias is additive rather than
multiplicative and a per-view affine removes it. Applying the coefficients
solved by layer_align.py collapsed the inter-camera spread from 43.5 mm to
7.0 mm, which is within about 1.5 mm of the reference view's own flatness.

  --load-affine     applies Z -> a_i * Z + b_i per camera, aligning every view
                    to the reference. This is the correction that removes
                    layering.
  --apply-absolute  additionally applies the global scale and shift solved
                    against the ChArUco ground truth, which anchors the whole
                    set metrically. Independent of the above: relative
                    alignment fixes layering, absolute anchoring fixes the
                    common offset, and neither substitutes for the other.
  --auto-align      re-solves the additive term every frame from the scene's
                    dominant plane, so exposure and gain changes are tracked.
                    The scale terms still come from --load-affine, and the
                    metric anchor still comes from --apply-absolute, so
                    auto-align replaces neither.

The correction is applied to the depth BEFORE preview and before capture, so
what is on screen is what gets written. Uncorrected depth is preserved in the
capture as depth_raw_<cam>.npy unless --no-save-raw-depth.

Coefficients solved on one scene and applied to another are an assumption, not
a result. Validate with:

    python layer_align.py --capture-dir <new capture> --mode apply \\
        --load-affine <the same json> --assign-tol 0.12 --n-planes 2

If the spread stays near the value seen when fitting, the correction
generalises. If it does not, the coefficients are scene-specific and that is
itself the finding.

Fusion
------
The four clouds are concatenated. That is not a weakness to be engineered
away: a voxel filter at 4 mm cannot average two sheets 44 mm apart, because
they fall in different voxels, so a concatenated cloud showing layering is
reporting a real disagreement between the depth maps rather than a defect in
the merge. The only honest improvements are to stop admitting samples that
were never going to agree, and to measure how much of the cloud more than one
camera actually saw.

  --max-incidence   surfaces seen near-tangentially are sampled by rays
                    further apart than the depth quantum and reconstruct as
                    combed sheets. The old default of 85 degrees rejected
                    almost nothing; 70 removes the combing at source. Raise it
                    only if the point count collapses on a surface you need.
  --conf-min        an absolute confidence floor, applied on top of the
                    per-camera percentile. A percentile always discards the
                    same fraction of every view, so a view that is uniformly
                    poor still contributes its best 60 per cent; a floor is
                    what removes a view that has nothing worth keeping.
  --fuse-corroborate
                    voxel size at which a point is checked for support from a
                    DIFFERENT camera. Each point is labelled with the number of
                    distinct cameras present in its voxel or any of the 26
                    around it. Nothing is discarded unless --fuse-min-views
                    says so; the label is written to the capture and reported.
  --fuse-min-views  discard points no other camera corroborates. Off by
                    default, because the uncorroborated fraction is a
                    measurement worth keeping rather than a defect worth
                    hiding.
  --fuse-weighted   collapse each voxel by a weighted mean of the points in
                    it, weighting by confidence and by the cosine of the
                    incidence angle, instead of the uniform mean Open3D uses.
                    This only affects voxels several cameras already agree on,
                    which is precisely where a mean is meaningful.
  --fuse-subsets    for every subset of the cameras, report what fraction of
                    the occupied volume that subset covers and what fraction
                    it corroborates. This is the direct measurement of how
                    many cameras the workcell needs, taken from one capture.

Segmentation
------------
With --segment, every capture is also passed to box_segment.py, which fits the
conveyor plane, clusters what stands above it, reconciles each parcel top face
across the cameras that see it, and reports a pick pose, footprint, height and
quality record per parcel. It writes boxes.json, boxes.csv, segmented.ply and
an annotated view per camera into the capture directory, and appends every
parcel to boxes_log.csv in the run directory.

The camera centres are passed through, so the segmentation can weight a
grazing view below a face-on one and detect a combed face. Segmentation
receives the per-point camera provenance, not the merged cloud, because
merging destroys exactly the information the reconciliation needs.

Dimensions inherit whatever scale error the depth carries, so run with
--apply-absolute and validate against a parcel of known size before quoting
any of the figures. Poses are in the world frame of the fusion, which is the
reference camera's optical frame, and still need the hand-eye transform.

Keys
    p           capture: back-project, fuse, write and segment a point cloud
                from the depth maps currently on screen
    r           reset the rolling performance window
    q or Esc    quit

Performance logging
-------------------
  perf.csv      one row per processed frame set: per-camera acquisition rate
                and frame count, sync spread, per-stage milliseconds, output
                rate, and the fraction of acquired frames the pipeline had to
                discard because inference was still busy
  boxes_log.csv one row per parcel per capture, across the whole run
  console       a rolling summary every --perf-interval seconds
  summary JSON  aggregate statistics on exit, including median and p95 for
                every stage

The discard fraction is the number that matters for SRQ3. Acquisition running
at 20 fps while fusion runs at 2 Hz means nine of every ten frames are dropped
by design; the log makes that explicit rather than leaving it implied by two
unrelated rate figures.

All geometry is imported from da3_fuse.py so the verified back-projection has
exactly one definition. Keep da3_fuse.py, online_align.py, box_segment.py and
face_consensus.py in the same directory.

Example
-------
python da3_stream.py --calib-dir /home/jetson/Projects/Calibration_4_1/results \\
    --load-affine runs/live_20260810_132154/capture_00000/aligned/depth_affine.json \\
    --apply-absolute --auto-align --segment \\
    --out-dir runs/live_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import itertools
import json
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

from arena_api.system import system

try:
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
        PALETTE,
    )
except ImportError as exc:
    raise SystemExit(f"cannot import da3_fuse.py from the working directory: {exc}")

try:
    from online_align import OnlineAligner
except ImportError as exc:
    OnlineAligner = None
    _ONLINE_IMPORT_ERROR = exc

try:
    from box_segment import (
        add_arguments as add_seg_arguments,
        params_from_args as seg_params_from_args,
        segment_boxes,
        write_results as write_seg_results,
        summarise as summarise_boxes,
        BOX_CSV_FIELDS,
    )
except ImportError as exc:
    add_seg_arguments = None
    _SEG_IMPORT_ERROR = exc


CAMERAS = {
    "left": "260505158",
    "center": "261100628",
    "top": "261100631",
    "right": "261100627",
}

PIXEL_FORMATS = {"RGB8": (3, "rgb"), "BGR8": (3, "bgr"), "BayerRG8": (1, "bayer")}

# Colour ramp for the view-count cloud: 1 view red, 2 amber, 3 green, 4 blue.
VIEW_COUNT_RGB = [(0.35, 0.35, 0.35), (0.90, 0.25, 0.20), (0.95, 0.70, 0.20),
                  (0.35, 0.80, 0.35), (0.25, 0.60, 0.95)]


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live DA3 depth streaming with per-view correction, "
                    "on-demand point cloud capture and parcel segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/stream"))
    ap.add_argument("--cameras", nargs="+", default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")

    # acquisition
    ap.add_argument("--width", type=int, default=2448)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--pixel-format", default="RGB8", choices=sorted(PIXEL_FORMATS))
    ap.add_argument("--bayer-code", default="COLOR_BayerRG2RGB",
                    help="used only with --pixel-format BayerRG8; DA3 wants RGB")
    ap.add_argument("--exposure-us", type=float, default=15000)
    ap.add_argument("--gain-db", type=float, default=0)
    ap.add_argument("--num-buffers", type=int, default=6)
    ap.add_argument("--buffer-timeout-ms", type=int, default=2000)
    ap.add_argument("--throughput-limit", type=int, default=0,
                    help="per-camera DeviceLinkThroughputLimit in bytes/s; "
                         "0 leaves the camera setting alone")
    ap.add_argument("--ptp", action="store_true",
                    help="enable IEEE 1588 on all cameras and report status")

    # synchronisation
    ap.add_argument("--max-sync-ms", type=float, default=0.0,
                    help="reject a frame set whose frames arrived more than "
                         "this far apart. DISABLED by default: free-running "
                         "cameras spread by up to one frame period, so at "
                         "8 fps any threshold below ~125 ms rejects everything "
                         "and the pipeline produces nothing. The spread is "
                         "logged to perf.csv either way, so measure it before "
                         "choosing a value")
    ap.add_argument("--stall-warn", type=float, default=3.0,
                    help="seconds without a processed set before printing a "
                         "diagnostic; 0 disables")
    ap.add_argument("--use-device-timestamp", action="store_true",
                    help="judge spread by camera clocks; needs --ptp locked")

    # model
    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2,
                    help="passes discarded before timing starts, excluding "
                         "autotuning and allocator warm-up")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="isotropic downscale before inference, intrinsics "
                         "scaled to match")

    # depth correction
    ap.add_argument("--load-affine", type=Path, default=None,
                    help="depth_affine.json from layer_align.py. Applies "
                         "Z -> a_i * Z + b_i per camera, which is what removes "
                         "inter-camera layering")
    ap.add_argument("--apply-absolute", action="store_true",
                    help="also apply the global scale and shift from the same "
                         "file's 'absolute' block, anchoring the set to the "
                         "ChArUco ground truth")
    ap.add_argument("--auto-align", action="store_true",
                    help="estimate the per-camera depth offset every frame "
                         "from the scene's dominant plane instead of trusting "
                         "stored coefficients. Survives exposure and gain "
                         "changes, which stored coefficients do not")
    ap.add_argument("--auto-align-stride", type=int, default=3,
                    help="pixel stride used when fitting the plane; 3 keeps "
                         "roughly a ninth of the samples and is ample")
    ap.add_argument("--auto-align-ema", type=float, default=0.3,
                    help="smoothing on the estimated offset, 1.0 for none. "
                         "Lower values reject per-frame noise but lag a real "
                         "change in the model's bias")
    ap.add_argument("--auto-align-tol", type=float, default=0.12,
                    help="a sample counts toward the plane if it lies within "
                         "this distance of it; must exceed the offset being "
                         "corrected, and stay below the shortest parcel or a "
                         "parcel top is counted as conveyor")
    ap.add_argument("--auto-align-max-step", type=float, default=0.15,
                    help="largest single-frame change permitted in the offset, "
                         "guarding against a spurious plane fit")
    ap.add_argument("--save-raw-depth", dest="save_raw_depth",
                    action="store_true", default=True,
                    help="keep the uncorrected depth alongside the corrected "
                         "one in each capture")
    ap.add_argument("--no-save-raw-depth", dest="save_raw_depth",
                    action="store_false")

    # filtering, mirroring da3_fuse.py
    ap.add_argument("--conf-percentile", type=float, default=40.0,
                    help="drop the worst this-percentage of each camera's "
                         "samples. Being a percentile it always removes the "
                         "same fraction of every view, however good or bad "
                         "that view is; pair it with --conf-min")
    ap.add_argument("--conf-min", type=float, default=0.0,
                    help="absolute confidence floor applied on top of the "
                         "percentile. This is what removes a view that has "
                         "nothing worth keeping, which a percentile cannot. "
                         "0 disables; read a sensible value off the "
                         "conf_threshold figures in capture.json")
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0,
                    help="drop samples whose surface normal exceeds this angle "
                         "from the view ray. At 3 m and 504 px processing a "
                         "surface seen beyond about 70 degrees is sampled by "
                         "rays further apart than the depth quantum and "
                         "reconstructs as a combed sheet, which no downstream "
                         "filter can repair. The previous default of 85 "
                         "rejected almost nothing. 90 disables")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--undistort", dest="undistort", action="store_true", default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")

    # fusion
    ap.add_argument("--fuse-corroborate", type=float, default=0.010,
                    help="voxel size at which each point is checked for a "
                         "contribution from a DIFFERENT camera, counting its "
                         "own voxel and the 26 around it. Should exceed the "
                         "point spacing and stay below the inter-camera "
                         "disagreement you are trying to detect. 0 disables")
    ap.add_argument("--fuse-min-views", type=int, default=1,
                    help="discard points that fewer than this many cameras "
                         "support. 1 keeps everything, which is the default "
                         "because the uncorroborated fraction is a measurement "
                         "rather than a defect. 2 gives a cloud in which every "
                         "point has been seen twice, at the cost of the whole "
                         "periphery of the workspace")
    ap.add_argument("--fuse-weighted", action="store_true",
                    help="collapse each voxel by a mean weighted by confidence "
                         "and by the cosine of the incidence angle, rather "
                         "than the uniform mean Open3D uses. Only changes "
                         "voxels several cameras already agree on")
    ap.add_argument("--fuse-subsets", action="store_true", default=True,
                    help="report coverage and corroboration for every subset "
                         "of the cameras, which is the direct measurement of "
                         "how many cameras the cell needs")
    ap.add_argument("--no-fuse-subsets", dest="fuse_subsets",
                    action="store_false")

    # capture
    ap.add_argument("--capture-key", default="p",
                    help="single character that triggers a point cloud capture")
    ap.add_argument("--capture-per-camera", action="store_true",
                    help="also write the four per-camera clouds")
    ap.add_argument("--capture-colour-by-camera", action="store_true",
                    help="also write a cloud tinted one colour per camera, "
                         "which is how layering is checked by eye")
    ap.add_argument("--capture-view-count", action="store_true",
                    help="also write a cloud coloured by how many cameras "
                         "support each point, which is how a single-view "
                         "region is found by eye")
    ap.add_argument("--capture-images", action="store_true", default=True,
                    help="also write the rectified RGB frames and depth arrays")
    ap.add_argument("--no-capture-images", dest="capture_images",
                    action="store_false")

    # display and logging
    ap.add_argument("--preview-width", type=int, default=560,
                    help="width of each depth tile in the preview mosaic")
    ap.add_argument("--depth-range", nargs=2, type=float, default=None,
                    metavar=("MIN_M", "MAX_M"),
                    help="fixed colour map range in metres; omit to autoscale "
                         "per frame, which makes the display flicker but shows "
                         "detail at every distance")
    ap.add_argument("--no-preview", dest="preview", action="store_false", default=True,
                    help="headless; capture is then unavailable")
    ap.add_argument("--perf-interval", type=float, default=5.0,
                    help="seconds between rolling console summaries")
    ap.add_argument("--perf-window", type=int, default=30,
                    help="frame sets retained in the rolling rate estimate")
    ap.add_argument("--perf-csv", default="perf.csv",
                    help="written inside --out-dir")
    ap.add_argument("--json", default="stream_summary.json",
                    help="written inside --out-dir")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds; 0 runs until quit")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many processed sets; 0 is unlimited")

    # segmentation, defined in box_segment.py so the thresholds have exactly
    # one definition
    if add_seg_arguments is not None:
        add_seg_arguments(ap)

    return ap.parse_args()


# --------------------------------------------------------------------------
# depth correction
# --------------------------------------------------------------------------

def load_affine(path: Path, names, apply_absolute):
    """Read layer_align.py coefficients and build the correction to apply.

    Two independent corrections live in that file:

      coefficients   per-camera a and b, mapping each view onto the reference
                     view's depth. This is what removes layering.
      absolute       the scale and shift measured against ground truth,
                     reported as measured = scale * true + shift. Correcting
                     therefore requires the INVERSE, true = (measured - shift)
                     / scale, applied after the per-camera stage.
    """
    data = json.loads(path.read_text())
    coeffs = data.get("coefficients")
    if not isinstance(coeffs, dict):
        raise SystemExit(f"{path} has no 'coefficients' block")

    missing = [n for n in names if n not in coeffs]
    if missing:
        raise SystemExit(f"{path} lacks coefficients for: {', '.join(missing)}")

    per_cam = {n: (float(coeffs[n]["a"]), float(coeffs[n]["b"])) for n in names}

    ref = data.get("reference")
    if ref is not None and ref in per_cam:
        a_r, b_r = per_cam[ref]
        if abs(a_r - 1.0) > 1e-9 or abs(b_r) > 1e-9:
            print(f"[WARN] the reference camera {ref!r} carries a non-identity "
                  f"correction (a={a_r}, b={b_r}). Coefficients are normally "
                  f"solved relative to it, so this file may not be what you "
                  f"think it is.")

    g_a, g_b = 1.0, 0.0
    absolute = data.get("absolute")
    if apply_absolute:
        if not isinstance(absolute, dict):
            raise SystemExit(f"--apply-absolute given but {path} has no "
                             f"'absolute' block; re-run layer_align.py with at "
                             f"least two reference planes")
        scale = float(absolute["scale_estimate"])
        shift = float(absolute["shift_estimate_m"])
        if abs(scale) < 1e-6:
            raise SystemExit(f"degenerate scale estimate {scale} in {path}")
        g_a, g_b = 1.0 / scale, -shift / scale

    return per_cam, (g_a, g_b), data


def apply_affine(depth, names, per_cam, glob):
    """Z -> g_a * (a_i * Z + b_i) + g_b, leaving invalid samples untouched.

    Invalid samples must be excluded explicitly: a non-zero b would otherwise
    turn a zero or negative sentinel into a plausible-looking depth and inject
    a plane of phantom points at the offset distance.
    """
    g_a, g_b = glob
    out = np.array(depth, dtype=np.float64, copy=True)
    for i, n in enumerate(names):
        a, b = per_cam[n]
        d = out[i]
        good = np.isfinite(d) & (d > 0)
        d[good] = g_a * (a * d[good] + b) + g_b
        out[i] = d
    return out


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def voxel_keys(points, voxel, pad=1):
    """Integer voxel coordinates and a single linear key per point.

    A linear key is used rather than np.unique over rows, which would sort a
    structured array and dominate the runtime on a two-million-point cloud.
    The pad leaves room for the plus and minus one shift of a neighbourhood
    search without any index going negative.
    """
    q = np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)
    q -= q.min(axis=0)
    q += pad
    dims = q.max(axis=0) + pad + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    return q, dims, key


def _popcount(v):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(v.astype(np.uint32)).astype(np.int16)
    out = np.zeros(len(v), dtype=np.int16)
    x = v.astype(np.int64)
    while np.any(x):
        out += (x & 1).astype(np.int16)
        x >>= 1
    return out


def corroboration(points, sources, n_cams, voxel):
    """How many distinct cameras support each point.

    A point is supported by a camera if that camera put a sample in the same
    voxel or in any of the twenty-six around it, so a point sitting a
    millimetre inside a voxel boundary is not counted as unsupported merely
    because its neighbour fell in the next cell.

    The neighbourhood search runs over the OCCUPIED VOXELS rather than over the
    points, so its cost scales with the number of voxels and not with the two
    million samples the four cameras produce. The result is mapped back to the
    points through the inverse index.

    Returns the per-point count, the per-voxel camera bitmask, and the inverse
    index, the last two so that the subset report can reuse them.
    """
    n = len(points)
    if voxel <= 0 or n == 0:
        return np.ones(n, dtype=np.int16), None, None, None

    q, dims, key = voxel_keys(points, voxel)
    # np.unique returns the optional outputs in a fixed order, index before
    # inverse, regardless of the order the flags are written in.
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = inv.ravel()
    q_u = q[first]

    own = np.zeros(len(uniq), dtype=np.int32)
    np.bitwise_or.at(own, inv, (1 << np.asarray(sources, dtype=np.int32)))

    merged = own.copy()
    for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
        if dx == dy == dz == 0:
            continue
        nk = (((q_u[:, 0] + dx) * dims[1] + (q_u[:, 1] + dy)) * dims[2]
              + (q_u[:, 2] + dz))
        pos = np.searchsorted(uniq, nk)
        np.clip(pos, 0, len(uniq) - 1, out=pos)
        hit = uniq[pos] == nk
        merged[hit] |= own[pos[hit]]

    counts_voxel = _popcount(merged)
    return counts_voxel[inv], merged, inv, len(uniq)


def subset_coverage(masks, cam_names):
    """Coverage and corroboration for every subset of the cameras.

    For each subset, the fraction of occupied voxels that subset reaches at
    all, and the fraction it reaches with at least two of its members. The
    second number is the one that matters: a voxel one camera sees is a
    measurement no other camera can check, and this is the measurement of how
    much of the workspace would lose that check if a camera were removed.
    """
    if masks is None:
        return None
    total = len(masks)
    if total == 0:
        return None
    out = []
    n = len(cam_names)
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            bits = 0
            for i in combo:
                bits |= (1 << i)
            sub = masks & bits
            covered = int(np.count_nonzero(sub))
            corrob = int(np.count_nonzero(_popcount(sub) >= 2))
            out.append({
                "cameras": [cam_names[i] for i in combo],
                "n_cameras": r,
                "voxels_covered": covered,
                "coverage_fraction": round(covered / total, 4),
                "voxels_corroborated": corrob,
                "corroborated_fraction": round(corrob / total, 4),
                "corroborated_of_covered": round(corrob / max(covered, 1), 4),
            })
    out.sort(key=lambda e: (e["n_cameras"], -e["corroborated_fraction"]))
    return {"total_voxels": total, "subsets": out}


def weighted_voxel_merge(points, colours, weights, voxel):
    """Collapse each voxel to a weighted mean of the points inside it.

    Open3D's voxel filter takes the uniform mean, which gives a grazing sample
    at low confidence the same say as a face-on one at high confidence. Where
    several cameras occupy the same voxel that is exactly the wrong weighting.
    Where only one does, the weighting has no effect, so this changes only the
    regions the cameras already agree on.

    np.bincount rather than np.add.at: the latter is an unbuffered scatter and
    costs seconds on a cloud this size.
    """
    _, _, key = voxel_keys(points, voxel, pad=0)
    uniq, inv = np.unique(key, return_inverse=True)
    m = len(uniq)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    wsum = np.bincount(inv, weights=w, minlength=m)

    P = np.empty((m, 3), dtype=np.float64)
    for k in range(3):
        P[:, k] = np.bincount(inv, weights=points[:, k] * w, minlength=m) / wsum
    C = None
    if colours is not None:
        C = np.empty((m, 3), dtype=np.float64)
        cols = np.asarray(colours, dtype=np.float64)
        for k in range(3):
            C[:, k] = np.bincount(inv, weights=cols[:, k] * w, minlength=m) / wsum
    return P, C, inv, m


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

    def __init__(self, name, device, args, maps, stop):
        super().__init__(name=f"acq-{name}", daemon=True)
        self.cam_name = name
        self.device = device
        self.args = args
        self.stop = stop
        self.map1, self.map2, self.roi = maps

        self.bpp, self.family = PIXEL_FORMATS[args.pixel_format]
        self.bayer_code = getattr(cv2, args.bayer_code, cv2.COLOR_BayerRG2RGB)

        self._lock = threading.Lock()
        self._frame = None
        self._host_t = 0.0
        self._dev_ts = 0
        self._seq = 0

        self.frames = 0          # buffers successfully acquired
        self.consumed = 0        # frames that reached inference
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
                rgb = to_rgb(buffer_to_array(item, self.bpp), self.family, self.bayer_code)

                if self.map1 is not None:
                    rgb = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR)
                    x, y, w, h = self.roi
                    if w > 0 and h > 0:
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
        return {
            "frames_acquired": self.frames,
            "frames_consumed": self.consumed,
            "frames_discarded": self.frames - self.consumed,
            "discard_fraction": round(1.0 - self.consumed / self.frames, 4) if self.frames else None,
            "incomplete_buffers": self.incomplete,
            "acquisition_failures": self.failures,
            "mean_acquired_fps": round(self.frames / elapsed, 3) if elapsed > 0 else 0.0,
            "buffer_wait": stat_block(self.wait_ms),
            "rectify": stat_block(self.rectify_ms),
            "errors": self.errors,
        }


def stat_block(vals):
    if not vals:
        return None
    ordered = sorted(vals)
    i95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {"n": len(ordered),
            "mean_ms": round(statistics.fmean(ordered), 2),
            "median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[i95], 2),
            "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2)}


def configure(device, args, log):
    nm = device.nodemap
    set_node(nm, "Width", args.width, log)
    set_node(nm, "Height", args.height, log)
    set_node(nm, "PixelFormat", args.pixel_format, log)
    set_node(nm, "AcquisitionMode", "Continuous", log)
    set_node(nm, "ExposureAuto", "Off", log)
    set_node(nm, "ExposureTime", args.exposure_us, log)
    set_node(nm, "GainAuto", "Off", log)
    set_node(nm, "Gain", args.gain_db, log)
    if args.throughput_limit > 0:
        set_node(nm, "DeviceLinkThroughputLimitMode", "On", log)
        set_node(nm, "DeviceLinkThroughputLimit", args.throughput_limit, log)
    if args.ptp:
        set_node(nm, "PtpEnable", True, log)

    st = device.tl_stream_nodemap
    set_node(st, "StreamAutoNegotiatePacketSize", True, log)
    set_node(st, "StreamPacketResendEnable", True, log)
    set_node(st, "StreamBufferHandlingMode", "NewestOnly", log)


def wait_for_ptp(devices, names, timeout_s=15.0):
    t0 = time.perf_counter()
    states = ["Unavailable"] * len(devices)
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


def build_maps(K, dist, size, enabled):
    """Precompute rectification maps, replicating cv2.undistort exactly."""
    w, h = size
    if not enabled:
        return (None, None, (0, 0, w, h)), K
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        newK = newK.copy()
        newK[0, 2] -= x
        newK[1, 2] -= y
    else:
        roi = (0, 0, w, h)
    return (map1, map2, roi), newK


def scale_K(K, s):
    if s == 1.0:
        return K
    Ks = K.copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def capture_cloud(index, names, frames, depth, depth_raw, conf, K_out, E_out,
                  rgb_proc, args, out_dir, affine_meta, seg_params=None):
    """Back-project, fuse, write, and optionally segment. Runs only on request.

    depth is already corrected; depth_raw is the model output before
    correction, kept so that a capture can be re-derived under different
    coefficients without re-running inference.

    The per-camera world points are retained alongside the fused cloud because
    segmentation needs to know which camera each point came from. That
    provenance is what allows a parcel resting on a single view's bias to be
    flagged rather than silently reported as a measurement, and it is destroyed
    by the merge, so segmentation is given the unmerged arrays.
    """
    t_start = time.perf_counter()
    stem = out_dir / f"capture_{index:05d}"
    stem.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")

    t0 = time.perf_counter()
    per_cam = []
    pts_list, src_list, col_list, wgt_list = [], [], [], []
    for i, n in enumerate(names):
        d = depth[i].astype(np.float64)
        c = conf[i]
        K_i = K_out[i]

        valid = np.isfinite(d) & (d > 0)
        drops = {}

        thr = None
        if args.conf_percentile > 0 and valid.any():
            thr = float(np.percentile(c[valid], args.conf_percentile))
            before = int(valid.sum())
            valid &= c >= thr
            drops["conf"] = before - int(valid.sum())

        if args.conf_min > 0:
            before = int(valid.sum())
            valid &= c >= args.conf_min
            drops["conf_floor"] = before - int(valid.sum())

        if args.edge_thresh > 0:
            before = int(valid.sum())
            valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
            drops["edge"] = before - int(valid.sum())

        pts_cam = points_camera(d, K_i)

        # The incidence angle is kept, not just the pass or fail, because it is
        # the natural weight for the merge: a sample taken at 65 degrees is
        # worth less than one taken at 10, and discarding the angle throws that
        # away.
        ang = None
        if args.max_incidence < 90:
            ok, ang = incidence_mask(pts_cam, args.max_incidence)
            before = int(valid.sum())
            valid &= ok
            drops["grazing"] = before - int(valid.sum())

        pts_world = to_world(pts_cam[valid], E_out[i])
        cols = rgb_proc[i][valid]
        pts_list.append(pts_world)
        src_list.append(np.full(len(pts_world), i, dtype=np.int32))
        col_list.append(np.asarray(cols, dtype=np.float64) / 255.0)

        cv_ = c[valid].astype(np.float64)
        hi = float(np.percentile(cv_, 95)) if cv_.size else 1.0
        w_conf = np.clip(cv_ / max(hi, 1e-9), 0.0, 1.0)
        if ang is not None:
            w_inc = np.cos(np.radians(np.clip(ang[valid], 0.0, 89.0)))
        else:
            w_inc = np.ones_like(cv_)
        wgt_list.append(np.maximum(w_conf * w_inc, 1e-4))

        dv = d[valid]
        per_cam.append({"camera": n,
                        "median_depth_m": float(np.median(dv)) if dv.size else None,
                        "depth_p05_m": float(np.percentile(dv, 5)) if dv.size else None,
                        "depth_p95_m": float(np.percentile(dv, 95)) if dv.size else None,
                        "median_incidence_deg": (float(np.median(ang[valid]))
                                                 if ang is not None and dv.size
                                                 else None),
                        "n_points": int(valid.sum()),
                        "conf_threshold": thr,
                        "dropped": drops})
    backproject_ms = (time.perf_counter() - t0) * 1e3

    pts_all = np.concatenate(pts_list) if pts_list else np.empty((0, 3))
    src_all = np.concatenate(src_list) if src_list else np.empty(0, np.int32)
    col_all = np.concatenate(col_list) if col_list else np.empty((0, 3))
    wgt_all = np.concatenate(wgt_list) if wgt_list else np.empty(0)

    # ---- corroboration -------------------------------------------------
    # How much of this cloud did more than one camera actually see. Nothing is
    # discarded on the strength of this unless --fuse-min-views says so: a
    # single-view region is a fact about the rig, and deleting it would remove
    # the evidence rather than the problem.
    t0 = time.perf_counter()
    views_per_point, voxel_masks, _, n_voxels = corroboration(
        pts_all, src_all, len(names), args.fuse_corroborate)
    hist = {int(k): int(v) for k, v in
            zip(*np.unique(views_per_point, return_counts=True))}
    corr_record = {
        "voxel_m": args.fuse_corroborate,
        "occupied_voxels": n_voxels,
        "points_by_view_count": hist,
        "single_view_fraction": round(
            hist.get(1, 0) / max(len(pts_all), 1), 4),
        "per_camera_single_view_fraction": {
            n: round(float(np.count_nonzero((src_all == i) & (views_per_point == 1))
                           / max(int(np.count_nonzero(src_all == i)), 1)), 4)
            for i, n in enumerate(names)},
    }

    dropped_uncorroborated = 0
    if args.fuse_min_views > 1:
        keep = views_per_point >= args.fuse_min_views
        dropped_uncorroborated = int(np.count_nonzero(~keep))
        pts_all, src_all = pts_all[keep], src_all[keep]
        col_all, wgt_all = col_all[keep], wgt_all[keep]
        views_per_point = views_per_point[keep]
    corr_record["points_discarded"] = dropped_uncorroborated

    subsets = subset_coverage(voxel_masks, names) if args.fuse_subsets else None
    corroborate_ms = (time.perf_counter() - t0) * 1e3

    # ---- merge ---------------------------------------------------------
    t0 = time.perf_counter()
    n_before = int(len(pts_all))
    if args.voxel > 0 and args.fuse_weighted:
        P, C, _, _ = weighted_voxel_merge(pts_all, col_all, wgt_all, args.voxel)
        fused = o3d.geometry.PointCloud()
        fused.points = o3d.utility.Vector3dVector(P)
        fused.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    else:
        fused = o3d.geometry.PointCloud()
        fused.points = o3d.utility.Vector3dVector(pts_all)
        fused.colors = o3d.utility.Vector3dVector(np.clip(col_all, 0, 1))
        if args.voxel > 0:
            fused = fused.voxel_down_sample(args.voxel)
    fuse_ms = (time.perf_counter() - t0) * 1e3

    # ---- write ---------------------------------------------------------
    t0 = time.perf_counter()
    o3d.io.write_point_cloud(str(stem / "fused.ply"), fused)
    if args.capture_per_camera:
        for i, n in enumerate(names):
            sel = src_all == i
            o3d.io.write_point_cloud(
                str(stem / f"cloud_{n}.ply"),
                to_o3d(pts_all[sel], colors=col_all[sel] * 255.0))
    if args.capture_colour_by_camera:
        merged = o3d.geometry.PointCloud()
        for i in range(len(names)):
            sel = src_all == i
            merged += to_o3d(pts_all[sel], rgb01=PALETTE[i % len(PALETTE)])
        o3d.io.write_point_cloud(str(stem / "fused_by_camera.ply"), merged)
    if args.capture_view_count:
        ramp = np.asarray(VIEW_COUNT_RGB, dtype=np.float64)
        idx = np.clip(views_per_point, 0, len(ramp) - 1).astype(np.int32)
        o3d.io.write_point_cloud(
            str(stem / "fused_by_view_count.ply"),
            to_o3d(pts_all, colors=ramp[idx] * 255.0))
    if args.capture_images:
        for i, n in enumerate(names):
            cv2.imwrite(str(stem / f"{n}.png"), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            # the processed image as well, because DA3's returned intrinsics
            # belong to it and not to the full-resolution rectified frame, so
            # it is the only image the parcels can be drawn on afterwards
            proc = rgb_proc[i]
            proc8 = proc if proc.dtype == np.uint8 else \
                (np.clip(proc, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(stem / f"proc_{n}.png"), cv2.cvtColor(proc8, cv2.COLOR_RGB2BGR))
            np.save(stem / f"depth_{n}.npy", depth[i])
            if args.save_raw_depth and depth_raw is not None:
                np.save(stem / f"depth_raw_{n}.npy", depth_raw[i])
            np.save(stem / f"conf_{n}.npy", conf[i])
            np.save(stem / f"K_{n}.npy", K_out[i])
            np.save(stem / f"E_{n}.npy", E_out[i])
    write_ms = (time.perf_counter() - t0) * 1e3

    # ---- segmentation and pick data -----------------------------------
    seg_summary, seg_ms = None, 0.0
    if seg_params is not None:
        t0 = time.perf_counter()
        # Unmerged points with their camera provenance, and the optical centres,
        # so the per-view reconciliation can weight a grazing view below a
        # face-on one and detect a combed face.
        seg = segment_boxes(pts_all, src_all, names, seg_params,
                            cam_centres=[camera_centre(E_out[i])
                                         for i in range(len(names))])
        written, rows = write_seg_results(
            seg, stem, capture_index=index, timestamp=stamp,
            images=rgb_proc, K_all=K_out, E_all=E_out, p=seg_params,
            extra_meta={"depth_correction": affine_meta,
                        "corroboration": corr_record})
        seg_ms = (time.perf_counter() - t0) * 1e3

        # One running log across the whole session, so a repeatability sweep is
        # one file rather than one file per capture.
        log = out_dir / "boxes_log.csv"
        new = not log.exists()
        with log.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=BOX_CSV_FIELDS)
            if new:
                writer.writeheader()
            writer.writerows(rows)

        seg_summary = {
            "ok": seg.get("ok"),
            "reason": seg.get("reason"),
            "n_boxes": len(seg.get("boxes", [])),
            "warnings": seg.get("warnings", []),
            "conveyor": seg.get("conveyor"),
            "timings_ms": seg.get("timings_ms"),
            "files": {k: str(v) for k, v in written.items()},
        }
        print(f"[seg  ] {len(seg.get('boxes', []))} parcel(s) in {seg_ms:.0f} ms")
        print(summarise_boxes(seg))
        for w in seg.get("warnings", []):
            print(f"[seg  ] {w}")

    bbox = fused.get_axis_aligned_bounding_box()
    record = {
        "capture": index,
        "timestamp": stamp,
        "depth_correction": affine_meta,
        "per_camera": per_cam,
        "corroboration": corr_record,
        "subset_coverage": subsets,
        "fused": {"n_points_before_voxel": int(n_before),
                  "n_points_written": int(len(fused.points)),
                  "voxel_m": args.voxel,
                  "weighted": bool(args.fuse_weighted),
                  "min_views": args.fuse_min_views,
                  "bbox_min": np.asarray(bbox.min_bound).tolist(),
                  "bbox_max": np.asarray(bbox.max_bound).tolist()},
        "segmentation": seg_summary,
        "timings_ms": {"backproject": round(backproject_ms, 2),
                       "corroborate": round(corroborate_ms, 2),
                       "fuse": round(fuse_ms, 2),
                       "write": round(write_ms, 2),
                       "segment": round(seg_ms, 2),
                       "total": round((time.perf_counter() - t_start) * 1e3, 2)},
    }
    (stem / "capture.json").write_text(json.dumps(record, indent=2))
    return record, stem


def print_subsets(subsets, limit_per_size=3):
    """The best few subsets at each size, which is the SRQ2 answer in one look."""
    if not subsets:
        return
    print(f"[views] corroboration over {subsets['total_voxels']} occupied voxels")
    by_size = {}
    for e in subsets["subsets"]:
        by_size.setdefault(e["n_cameras"], []).append(e)
    for r in sorted(by_size):
        for e in by_size[r][:limit_per_size]:
            print(f"        {r} cam  {'+'.join(e['cameras']):<28} "
                  f"covers {e['coverage_fraction'] * 100:5.1f}%   "
                  f"corroborated {e['corroborated_fraction'] * 100:5.1f}%   "
                  f"({e['corroborated_of_covered'] * 100:5.1f}% of what it sees)")


# --------------------------------------------------------------------------
# display
# --------------------------------------------------------------------------

def colourise(d, fixed_range):
    finite = np.isfinite(d) & (d > 0)
    if fixed_range is not None:
        lo, hi = fixed_range
    elif finite.any():
        lo, hi = np.percentile(d[finite], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~finite] = 0
    return img, lo, hi


def build_preview(names, depth, workers, args, info_lines):
    tiles = []
    for i, n in enumerate(names):
        img, lo, hi = colourise(depth[i], args.depth_range)
        s = args.preview_width / img.shape[1]
        img = cv2.resize(img, (args.preview_width, max(1, int(img.shape[0] * s))))
        label = f"{n}  in {workers[i].fps():5.1f} fps  {lo:.2f}-{hi:.2f} m"
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
        tiles.append(img)

    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
             if t.shape[0] < h else t for t in tiles]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
            if r.shape[1] < width else r for r in rows]
    mosaic = np.vstack(rows)

    bar = np.zeros((24 * len(info_lines) + 10, mosaic.shape[1], 3), np.uint8)
    for j, (line, colour) in enumerate(info_lines):
        cv2.putText(bar, line, (8, 20 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, mosaic])


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if o3d is None:
        raise SystemExit("open3d is required")

    names = list(args.cameras)
    unknown = [n for n in names if n not in CAMERAS]
    if unknown:
        raise SystemExit(f"unknown camera name(s): {', '.join(unknown)}")
    if not args.preview:
        print("[note ] headless mode: point cloud capture is unavailable")
    if args.fuse_min_views > len(names):
        raise SystemExit(f"--fuse-min-views {args.fuse_min_views} exceeds the "
                         f"{len(names)} cameras in use, so every point would "
                         f"be discarded")

    capture_code = ord(args.capture_key[0].lower())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- depth correction ---------------------------------------------
    per_cam_affine, global_affine, affine_raw = None, (1.0, 0.0), None
    if args.load_affine is not None:
        per_cam_affine, global_affine, affine_raw = load_affine(
            args.load_affine, names, args.apply_absolute)
        print(f"[corr ] depth correction from {args.load_affine}")
        for n in names:
            a, b = per_cam_affine[n]
            print(f"        {n:<8} a={a:.6f}  b={b * 1e3:+8.2f} mm")
        if args.apply_absolute:
            g_a, g_b = global_affine
            print(f"        global   a={g_a:.6f}  b={g_b * 1e3:+8.2f} mm "
                  f"(absolute anchor)")
        else:
            print("        no absolute anchor applied; the cloud is internally "
                  "consistent but keeps the common metric offset")
    else:
        print("[corr ] no depth correction; expect inter-camera layering in "
              "any captured cloud")
        if args.apply_absolute:
            print("[WARN] --apply-absolute does nothing without --load-affine: "
                  "the anchor lives in that file. Any segmentation dimensions "
                  "are unanchored.")

    affine_meta = {
        "source": str(args.load_affine) if args.load_affine else None,
        "per_camera": {n: {"a": per_cam_affine[n][0], "b": per_cam_affine[n][1]}
                       for n in names} if per_cam_affine else None,
        "global": {"a": global_affine[0], "b": global_affine[1]},
        "absolute_applied": bool(args.apply_absolute),
        "auto_align": bool(args.auto_align),
    }

    aligner = None
    if args.auto_align:
        if OnlineAligner is None:
            raise SystemExit(f"--auto-align needs online_align.py in the "
                             f"working directory: {_ONLINE_IMPORT_ERROR}")
        base_a = ({n: per_cam_affine[n][0] for n in names}
                  if per_cam_affine else {n: 1.0 for n in names})
        aligner = OnlineAligner(names, args.reference, base_a=base_a,
                                stride=args.auto_align_stride,
                                ema=args.auto_align_ema,
                                assign_tol=args.auto_align_tol,
                                max_step=args.auto_align_max_step)
        print(f"[corr ] auto-align on: the offset is re-solved every frame "
              f"from the dominant plane, so exposure and gain changes are "
              f"tracked. Scale terms come from "
              f"{'the loaded file' if per_cam_affine else 'unity'}.")

    # ---- fusion --------------------------------------------------------
    print(f"[fuse ] incidence limit {args.max_incidence:.0f} deg, "
          f"corroboration voxel {args.fuse_corroborate * 1e3:.0f} mm, "
          f"min views {args.fuse_min_views}, "
          f"{'weighted' if args.fuse_weighted else 'uniform'} voxel merge")
    if args.max_incidence >= 85:
        print("[WARN] an incidence limit of 85 degrees or more rejects almost "
              "nothing. Surfaces seen that obliquely reconstruct as combed "
              "sheets that no later stage can repair; 70 is the working value "
              "on this rig.")
    if args.fuse_corroborate > 0 and args.fuse_corroborate <= args.voxel:
        print(f"[WARN] --fuse-corroborate {args.fuse_corroborate * 1e3:.0f} mm "
              f"is at or below the point spacing implied by --voxel "
              f"{args.voxel * 1e3:.0f} mm, so most points will look "
              f"uncorroborated simply because the neighbourhood is too small "
              f"to contain a second camera's sample.")

    # ---- segmentation --------------------------------------------------
    seg_params = None
    if getattr(args, "segment", False):
        if add_seg_arguments is None:
            raise SystemExit(f"--segment needs box_segment.py in the working "
                             f"directory: {_SEG_IMPORT_ERROR}")
        seg_params = seg_params_from_args(args)
        print(f"[seg  ] segmentation on capture: conveyor plane at "
              f"{seg_params.plane_thresh * 1e3:.0f} mm, parcels above "
              f"{seg_params.min_height * 1e3:.0f} mm, up to "
              f"{seg_params.max_levels} level(s) per cluster")
        if aligner is not None and seg_params.min_height >= args.auto_align_tol:
            print(f"[WARN] --auto-align-tol {args.auto_align_tol * 1e3:.0f} mm "
                  f"is at or below the segmentation's minimum parcel height "
                  f"{seg_params.min_height * 1e3:.0f} mm, so a short parcel can "
                  f"be counted as conveyor by the aligner and then reported as "
                  f"a parcel by the segmentation. Raise the tolerance or accept "
                  f"that short parcels drag the offset.")
        if not args.apply_absolute:
            print("[seg  ] no absolute anchor: footprint and height figures are "
                  "internally consistent but not metrically anchored")
        if args.fuse_min_views > 1:
            print(f"[WARN] --fuse-min-views {args.fuse_min_views} removes "
                  f"uncorroborated points before segmentation, so a parcel only "
                  f"one camera sees will not be reported at all rather than "
                  f"being reported and flagged. That is a choice about what a "
                  f"missed parcel costs against a wrong pose.")
    affine_meta["segmentation"] = bool(seg_params)

    # ---- calibration --------------------------------------------------
    intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
    ext = load_extrinsics(args.calib_dir, names, args.reference)
    Es = np.stack([ext[n] for n in names])
    for n, E in zip(names, Es):
        c = camera_centre(E)
        print(f"[pose ] {n:<8} C={c.round(4)}  |C|={np.linalg.norm(c):.4f} m")

    maps, Ks_rect = {}, {}
    for n in names:
        m, newK = build_maps(intr[n]["K"], intr[n]["dist"],
                             (args.width, args.height), args.undistort)
        maps[n] = m
        Ks_rect[n] = newK

    rect_sizes = {n: (maps[n][2][2], maps[n][2][3]) for n in names}
    print(f"[rect ] undistorted sizes: {rect_sizes}")
    if len(set(rect_sizes.values())) > 1:
        print("[WARN] the undistorted images do not share a size. Mixed aspect "
              "ratios in one DA3 batch trigger a centre crop to the smallest "
              "common dimension, shifting the principal point without changing "
              "the focal length. Check fx_ratio in perf.csv before trusting "
              "any captured cloud.")

    # ---- cameras ------------------------------------------------------
    infos = system.device_infos
    if not infos:
        raise SystemExit("no cameras found; check the Arena library path and NIC")
    found = {i["serial"]: i for i in infos}
    missing = [n for n in names if CAMERAS[n] not in found]
    if missing:
        raise SystemExit(f"missing camera(s): {', '.join(missing)}; detected {sorted(found)}")

    devices = system.create_device([found[CAMERAS[n]] for n in names])

    config_log = {}
    stop = threading.Event()
    workers = []
    ptp_status = None
    exit_code = 0
    processed = 0
    captures = 0
    rejected_sync = 0
    capture_records = []
    t_measure = None
    t_loop = time.perf_counter()
    stage_ms = {k: [] for k in ("assemble", "inference", "correct", "preview", "cycle")}
    out_stamps = deque(maxlen=args.perf_window)

    csv_path = args.out_dir / args.perf_csv
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["frame", "t_s", "out_hz_rolling", "sync_spread_ms",
         "assemble_ms", "inference_ms", "correct_ms", "preview_ms", "cycle_ms",
         "is_metric", "affine_applied"]
        + [f"fx_ratio_{n}" for n in names]
        + [f"offset_mm_{n}" for n in names]
        + [f"median_depth_{n}" for n in names]
        + [f"in_fps_{n}" for n in names]
        + [f"acquired_{n}" for n in names]
        + [f"consumed_{n}" for n in names]
        + [f"incomplete_{n}" for n in names]
        + ["discard_fraction", "sync_rejections", "captures"])

    try:
        for n, dev in zip(names, devices):
            log = []
            configure(dev, args, log)
            config_log[n] = log

        if args.ptp:
            ptp_status, locked = wait_for_ptp(devices, names)
            print(f"[ptp  ] {ptp_status}" + ("" if locked else "   NOT LOCKED"))

        for n, dev in zip(names, devices):
            dev.start_stream(args.num_buffers)
            workers.append(CameraWorker(n, dev, args, maps[n], stop))
        for w in workers:
            w.start()

        print(f"[model] loading {args.model}")
        t0 = time.perf_counter()
        model = load_model(args.model, args.device)
        print(f"[model] ready in {time.perf_counter() - t0:.1f} s")

        window = "DA3 live depth"
        if args.preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print(f"[keys ] {args.capture_key} capture"
              + (" and segment" if seg_params else "")
              + "   r reset window   q quit")

        last_seqs = [0] * len(workers)
        warmups_left = max(0, args.warmup)
        last_report = time.perf_counter()
        last_progress = time.perf_counter()
        sync_spreads = deque(maxlen=200)
        t_loop = time.perf_counter()

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
                if args.stall_warn > 0 and time.perf_counter() - last_progress >= args.stall_warn:
                    last_progress = time.perf_counter()
                    waiting = [w.cam_name for w, s in zip(workers, snap) if s is None]
                    if waiting:
                        print(f"[stall] no frame yet from: {', '.join(waiting)}. "
                              f"Acquisition counts "
                              f"{ {w.cam_name: w.frames for w in workers} }, "
                              f"incomplete "
                              f"{ {w.cam_name: w.incomplete for w in workers} }")
                    else:
                        print("[stall] every camera is still on the frame "
                              "already processed; waiting for new acquisitions")
                if args.preview and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                time.sleep(0.002)
                continue

            if args.use_device_timestamp and all(s[2] for s in snap):
                spread_ms = (max(s[2] for s in snap) - min(s[2] for s in snap)) / 1e6
            else:
                spread_ms = (max(s[1] for s in snap) - min(s[1] for s in snap)) * 1e3

            if args.max_sync_ms > 0 and spread_ms > args.max_sync_ms:
                rejected_sync += 1
                sync_spreads.append(spread_ms)
                if args.stall_warn > 0 and time.perf_counter() - last_progress >= args.stall_warn:
                    last_progress = time.perf_counter()
                    med = statistics.median(sync_spreads)
                    fps = sum(w.fps() for w in workers) / max(len(workers), 1)
                    period = 1e3 / fps if fps > 0 else float("nan")
                    print(f"[stall] {rejected_sync} sets rejected on sync. "
                          f"Median spread {med:.1f} ms against --max-sync-ms "
                          f"{args.max_sync_ms:.1f} ms. Cameras are free-running "
                          f"at {fps:.1f} fps, a {period:.0f} ms frame period, so "
                          f"a threshold below that rejects everything. Raise it "
                          f"or pass --max-sync-ms 0.")
                if args.preview and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                time.sleep(0.002)
                continue

            frames = [s[0] for s in snap]
            last_seqs = [s[3] for s in snap]
            for w in workers:
                w.consumed += 1
            assemble_ms = (time.perf_counter() - t0) * 1e3

            # ---- inference -------------------------------------------
            if args.input_scale != 1.0:
                s = args.input_scale
                frames = [cv2.resize(f, (int(round(f.shape[1] * s)), int(round(f.shape[0] * s))),
                                     interpolation=cv2.INTER_AREA) for f in frames]
                Ks_in = np.stack([scale_K(Ks_rect[n], s) for n in names])
            else:
                Ks_in = np.stack([Ks_rect[n] for n in names])

            t0 = time.perf_counter()
            if args.mode == "prior":
                pred = model.inference(frames, intrinsics=Ks_in, extrinsics=Es,
                                       align_to_input_ext_scale=True,
                                       process_res=args.process_res)
            else:
                pred = model.inference(frames, process_res=args.process_res)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1e3

            depth_raw = as_numpy(pred.depth)
            conf = as_numpy(pred.conf)
            K_out = as_numpy(pred.intrinsics)
            E_out = as_numpy(pred.extrinsics)
            rgb_proc = as_numpy(pred.processed_images)
            is_metric = int(getattr(pred, "is_metric", 0))

            # ---- depth correction ------------------------------------
            # Applied here, before preview and before capture, so the display
            # shows exactly the depth any written cloud is built from.
            t0 = time.perf_counter()
            align_diag = None
            offsets_mm = [0.0] * len(names)
            if aligner is not None:
                b_now, align_diag = aligner.update(depth_raw, K_out, E_out)
                eff = {n: ((per_cam_affine[n][0] if per_cam_affine else 1.0),
                           b_now[n]) for n in names}
                depth = apply_affine(depth_raw, names, eff, global_affine)
                offsets_mm = [b_now[n] * 1e3 for n in names]
            elif per_cam_affine is not None:
                depth = apply_affine(depth_raw, names, per_cam_affine, global_affine)
                offsets_mm = [per_cam_affine[n][1] * 1e3 for n in names]
            else:
                depth = depth_raw
            correct_ms = (time.perf_counter() - t0) * 1e3

            if warmups_left > 0:
                warmups_left -= 1
                print(f"[warm ] discarded warm-up pass, {inference_ms:.0f} ms")
                if warmups_left == 0:
                    t_measure = time.perf_counter()
                    for w in workers:
                        w.frames = 0
                        w.consumed = 0
                        w.stamps.clear()
                continue
            if t_measure is None:
                t_measure = time.perf_counter()

            out_stamps.append(time.perf_counter())
            out_hz = 0.0
            if len(out_stamps) > 1:
                span = out_stamps[-1] - out_stamps[0]
                out_hz = (len(out_stamps) - 1) / span if span > 0 else 0.0

            fx_ratio = [float(K_out[i][0, 0] /
                              (Ks_in[i][0, 0] * depth[i].shape[1] / frames[i].shape[1]))
                        for i in range(len(names))]

            med_depth = []
            for i in range(len(names)):
                d = depth[i]
                good = np.isfinite(d) & (d > 0)
                med_depth.append(float(np.median(d[good])) if good.any() else float("nan"))

            acquired_total = sum(w.frames for w in workers)
            consumed_total = sum(w.consumed for w in workers)
            discard = 1.0 - consumed_total / acquired_total if acquired_total else 0.0

            # ---- display ---------------------------------------------
            t0 = time.perf_counter()
            key = -1
            if args.preview:
                fx_bad = not all(0.98 < r < 1.02 for r in fx_ratio)
                if aligner is not None:
                    ok = align_diag is not None and align_diag.get("ok")
                    corr = "auto-aligned" if ok else "AUTO-ALIGN FAILED"
                    corr_colour = (0, 255, 0) if ok else (0, 0, 255)
                    off = "  ".join(f"{n} {v:+6.1f}" for n, v in zip(names, offsets_mm))
                elif per_cam_affine:
                    corr = "corrected" + (" + anchored" if args.apply_absolute else "")
                    corr_colour = (0, 255, 0)
                    off = "  ".join(f"{n} {v:+6.1f}" for n, v in zip(names, offsets_mm))
                else:
                    corr = "UNCORRECTED"
                    corr_colour = (0, 165, 255)
                    off = "no offsets applied"
                info = [
                    (f"out {out_hz:5.2f} Hz   infer {inference_ms:6.1f} ms   "
                     f"in {sum(w.fps() for w in workers) / len(workers):5.1f} fps/cam   "
                     f"discarded {discard * 100:4.1f}%", (0, 255, 0)),
                    (f"frame {processed}   sync {spread_ms:5.1f} ms   "
                     f"rejects {rejected_sync}   captures {captures}   "
                     f"metric={is_metric}   fx_ratio "
                     f"{min(fx_ratio):.3f}-{max(fx_ratio):.3f}",
                     (0, 0, 255) if fx_bad else (0, 220, 220)),
                    (f"depth {corr}   median "
                     f"{min(med_depth):.3f}-{max(med_depth):.3f} m", corr_colour),
                    (f"offset mm  {off}", corr_colour),
                    (f"[{args.capture_key}] capture"
                     + (" and segment" if seg_params else "")
                     + "    [r] reset window    [q] quit",
                     (200, 200, 200)),
                ]
                cv2.imshow(window, build_preview(names, depth, workers, args, info))
                key = cv2.waitKey(1) & 0xFF
            preview_ms = (time.perf_counter() - t0) * 1e3

            cycle_ms = (time.perf_counter() - t_cycle) * 1e3
            for k, v in (("assemble", assemble_ms), ("inference", inference_ms),
                         ("correct", correct_ms), ("preview", preview_ms),
                         ("cycle", cycle_ms)):
                stage_ms[k].append(v)

            csv_writer.writerow(
                [processed, round(time.perf_counter() - t_measure, 4),
                 round(out_hz, 4), round(spread_ms, 3),
                 round(assemble_ms, 3), round(inference_ms, 3),
                 round(correct_ms, 3), round(preview_ms, 3), round(cycle_ms, 3),
                 is_metric, int(per_cam_affine is not None)]
                + [round(r, 4) for r in fx_ratio]
                + [round(v, 3) for v in offsets_mm]
                + [round(v, 4) for v in med_depth]
                + [round(w.fps(), 3) for w in workers]
                + [w.frames for w in workers]
                + [w.consumed for w in workers]
                + [w.incomplete for w in workers]
                + [round(discard, 4), rejected_sync, captures])
            csv_file.flush()
            processed += 1
            last_progress = time.perf_counter()

            # ---- keys -------------------------------------------------
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                out_stamps.clear()
                for w in workers:
                    w.stamps.clear()
                print("[perf ] rolling window reset")
            if key == capture_code:
                meta = dict(affine_meta)
                if align_diag is not None:
                    meta = {**meta, "auto_align_state": align_diag}
                rec, stem = capture_cloud(
                    captures, names, frames, depth,
                    depth_raw if (per_cam_affine is not None or aligner is not None) else None,
                    conf, K_out, E_out, rgb_proc, args, args.out_dir, meta,
                    seg_params=seg_params)
                capture_records.append(rec)
                captures += 1
                meds = [p["median_depth_m"] for p in rec["per_camera"]
                        if p["median_depth_m"]]
                c = rec["corroboration"]
                print(f"[cap  ] {stem.name}  {rec['fused']['n_points_written']} pts  "
                      f"bp {rec['timings_ms']['backproject']:.0f} ms  "
                      f"corr {rec['timings_ms']['corroborate']:.0f} ms  "
                      f"fuse {rec['timings_ms']['fuse']:.0f} ms  "
                      f"write {rec['timings_ms']['write']:.0f} ms  "
                      f"seg {rec['timings_ms']['segment']:.0f} ms"
                      + (f"  depth {min(meds):.3f}-{max(meds):.3f} m" if meds else ""))
                print(f"[views] {c['single_view_fraction'] * 100:.1f}% of the "
                      f"cloud is seen by one camera only   per camera "
                      + "  ".join(
                          f"{k} {v * 100:.0f}%"
                          for k, v in c["per_camera_single_view_fraction"].items()))
                print_subsets(rec.get("subset_coverage"))

            # ---- periodic console summary -----------------------------
            if time.perf_counter() - last_report >= args.perf_interval:
                last_report = time.perf_counter()
                inf = stage_ms["inference"]
                rates = "  ".join(f"{w.cam_name} {w.fps():5.1f}" for w in workers)
                print(f"[perf ] out {out_hz:5.2f} Hz over {processed} sets   "
                      f"infer med {statistics.median(inf):6.1f} ms   "
                      f"in fps [{rates}]   discarded {discard * 100:4.1f}%   "
                      f"sync rejects {rejected_sync}")

            if all(not w.is_alive() for w in workers):
                print("all acquisition threads have stopped", file=sys.stderr)
                exit_code = 1
                break

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=5.0)
        for d in devices:
            try:
                d.stop_stream()
            except Exception:  # noqa: BLE001
                pass
        system.destroy_device()
        if args.preview:
            cv2.destroyAllWindows()
        csv_file.close()

    # ---- summary ------------------------------------------------------
    elapsed = max(time.perf_counter() - (t_measure or t_loop), 1e-9)
    acquired_total = sum(w.frames for w in workers)
    consumed_total = sum(w.consumed for w in workers)

    n_boxes = [r["segmentation"]["n_boxes"] for r in capture_records
               if r.get("segmentation")]
    single_fracs = [r["corroboration"]["single_view_fraction"]
                    for r in capture_records if r.get("corroboration")]

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cameras": {n: CAMERAS[n] for n in names},
        "rectified_sizes": {n: list(rect_sizes[n]) for n in names},
        "depth_correction": affine_meta,
        "auto_align": aligner.summary() if aligner is not None else None,
        "fusion": {
            "max_incidence_deg": args.max_incidence,
            "conf_percentile": args.conf_percentile,
            "conf_min": args.conf_min,
            "corroborate_voxel_m": args.fuse_corroborate,
            "min_views": args.fuse_min_views,
            "weighted": bool(args.fuse_weighted),
            "single_view_fraction_median": (statistics.median(single_fracs)
                                            if single_fracs else None),
        },
        "segmentation": {
            "enabled": bool(seg_params),
            "captures_segmented": len(n_boxes),
            "parcels_total": int(sum(n_boxes)) if n_boxes else 0,
            "parcels_per_capture_median": (statistics.median(n_boxes)
                                           if n_boxes else None),
        },
        "ptp_status": ptp_status,
        "node_configuration": config_log,
        "measured_duration_s": round(elapsed, 3),
        "frames_processed": processed,
        "output_hz_mean": round(processed / elapsed, 4) if processed else 0.0,
        "acquisition": {
            "frames_acquired_total": acquired_total,
            "frames_consumed_total": consumed_total,
            "discard_fraction": round(1.0 - consumed_total / acquired_total, 4) if acquired_total else None,
            "sync_rejections": rejected_sync,
        },
        "stage_timings": {k: stat_block(v) for k, v in stage_ms.items()},
        "per_camera_acquisition": {w.cam_name: w.summary(elapsed) for w in workers},
        "captures": capture_records,
    }
    if torch is not None and torch.cuda.is_available():
        summary["memory"] = {
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}

    (args.out_dir / args.json).write_text(json.dumps(summary, indent=2))

    print(f"\nprocessed {processed} sets in {elapsed:.1f} s "
          f"({summary['output_hz_mean']:.3f} Hz output)")
    if acquired_total:
        print(f"acquired {acquired_total} frames, consumed {consumed_total}, "
              f"discarded {summary['acquisition']['discard_fraction'] * 100:.1f}%")
    inf = stage_ms["inference"]
    if inf:
        b = stat_block(inf)
        print(f"inference: median {b['median_ms']:.1f} ms, p95 {b['p95_ms']:.1f} ms, "
              f"n={b['n']}")
    cor = stage_ms["correct"]
    if cor and (per_cam_affine is not None or aligner is not None):
        b = stat_block(cor)
        print(f"correction: median {b['median_ms']:.2f} ms")
    print(f"captures written : {captures}")
    if single_fracs:
        print(f"single-view      : median "
              f"{statistics.median(single_fracs) * 100:.1f}% of the cloud seen "
              f"by one camera only, over {len(single_fracs)} captures")
    if n_boxes:
        seg_times = [r["timings_ms"]["segment"] for r in capture_records
                     if r.get("segmentation")]
        print(f"parcels found    : {sum(n_boxes)} over {len(n_boxes)} captures, "
              f"median {statistics.median(n_boxes):.0f} per capture, "
              f"segmentation median {statistics.median(seg_times):.0f} ms")
        print(f"parcel log       : {args.out_dir / 'boxes_log.csv'}")
    if rejected_sync:
        print(f"sync rejections  : {rejected_sync} "
              f"(--max-sync-ms {args.max_sync_ms:.1f}); if this dwarfs the "
              f"processed count the threshold is below one frame period")
    print(f"per-frame log    : {csv_path}")
    print(f"summary          : {args.out_dir / args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())