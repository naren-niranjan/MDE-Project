#!/usr/bin/env python3
"""
da3_live.py

Live four-camera acquisition feeding DA3 multi-view fusion, producing a fused
point cloud per processed frame set.

Design
------
Acquisition and inference are decoupled. Four acquisition threads run the
cameras at their native rate into a latest-frame slot each; the main thread
takes whatever set is current, runs DA3 once, and back-projects. Frames that
arrive while inference is running are discarded rather than queued, so the
pipeline always operates on the newest scene rather than falling behind.

All geometry is imported from da3_fuse.py so that the verified back-projection
path has exactly one definition. Keep both files in the same directory.

Synchronisation
---------------
The cameras free-run, so a set assembled from four latest-frame slots spans up
to one frame period. At 20 fps that is 50 ms, during which a conveyor at
0.2 m/s moves 10 mm, which is larger than the depth floor the rig is capable
of. Three mitigations, in increasing order of rigour:

  --max-sync-ms   rejects sets whose host arrival times are too far apart.
                  Cheap, but host arrival ordering is only a proxy for
                  exposure ordering.
  --ptp           enables IEEE 1588 on all cameras so that device timestamps
                  share a clock, making --use-device-timestamp meaningful.
  --trigger settle  waits for the scene to stop moving before processing,
                  which removes the problem rather than bounding it. This is
                  the correct mode for pick-pose extraction.

None of these make the exposures simultaneous. That requires a hardware
trigger or PTP scheduled action commands.

Examples
--------
Continuous, preview only, nothing written:
    python da3_live.py --calib-dir /home/jetson/Projects/Calibration_4_1/results

Event driven, writing one cloud per settled scene:
    python da3_live.py --calib-dir /home/jetson/Projects/Calibration_4_1/results \\
        --trigger settle --save-mode all --out-dir runs/live_$(date +%Y%m%d_%H%M%S)

Throughput measurement for SRQ3, no disk writes, no preview:
    python da3_live.py --calib-dir ... --no-preview --save-mode none \\
        --duration 120 --warmup 3 --json live_timing.json
"""

from __future__ import annotations

import argparse
import ctypes
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

# Single source of truth for the verified geometry. Any change to the
# back-projection must happen in da3_fuse.py, not here.
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


CAMERAS = {
    "left": "260505158",
    "center": "261100628",
    "top": "261100631",
    "right": "261100627",
}

PIXEL_FORMATS = {"RGB8": (3, "rgb"), "BGR8": (3, "bgr"), "BayerRG8": (1, "bayer")}


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live four-camera DA3 fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/live"))
    ap.add_argument("--cameras", nargs="+", default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")

    # acquisition
    ap.add_argument("--width", type=int, default=2448)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--pixel-format", default="RGB8", choices=sorted(PIXEL_FORMATS))
    ap.add_argument("--bayer-code", default="COLOR_BayerRG2RGB",
                    help="OpenCV code used only when --pixel-format is BayerRG8. "
                         "The model wants RGB, so convert to RGB here.")
    ap.add_argument("--exposure-us", type=float, default=20000.0)
    ap.add_argument("--gain-db", type=float, default=18.7)
    ap.add_argument("--num-buffers", type=int, default=6)
    ap.add_argument("--buffer-timeout-ms", type=int, default=2000)
    ap.add_argument("--throughput-limit", type=int, default=0,
                    help="per-camera DeviceLinkThroughputLimit in bytes/s; "
                         "0 leaves the camera setting alone")
    ap.add_argument("--ptp", action="store_true",
                    help="enable IEEE 1588 on all cameras and report status")

    # synchronisation
    ap.add_argument("--max-sync-ms", type=float, default=60.0,
                    help="reject a frame set whose arrival times span more "
                         "than this; 0 disables the check")
    ap.add_argument("--use-device-timestamp", action="store_true",
                    help="judge sync spread by camera timestamps instead of "
                         "host arrival; only meaningful with --ptp locked")

    # triggering
    ap.add_argument("--trigger", choices=["continuous", "settle", "key"],
                    default="continuous")
    ap.add_argument("--settle-thresh", type=float, default=1.5,
                    help="mean absolute grey-level difference below which the "
                         "reference view counts as static")
    ap.add_argument("--settle-frames", type=int, default=4,
                    help="consecutive static frames required to fire")
    ap.add_argument("--settle-rearm", type=float, default=4.0,
                    help="difference that must be exceeded before the trigger "
                         "will fire again")

    # model
    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2,
                    help="inference passes run and discarded before timing "
                         "starts, to exclude autotuning and allocator warm-up")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="isotropic downscale applied to the undistorted image "
                         "before inference, with the intrinsics scaled to "
                         "match; reduces host-to-device transfer cost")

    # filtering, mirroring da3_fuse.py defaults
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=85.0)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--undistort", dest="undistort", action="store_true", default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")

    # output
    ap.add_argument("--save-mode", choices=["none", "latest", "all"], default="latest",
                    help="none writes no clouds; latest overwrites a single "
                         "fused.ply; all writes one numbered cloud per frame")
    ap.add_argument("--save-per-camera", action="store_true",
                    help="also write the four per-camera clouds; only "
                         "meaningful with --save-mode latest or all")
    ap.add_argument("--colour-by-camera", action="store_true")
    ap.add_argument("--no-preview", dest="preview", action="store_false", default=True)
    ap.add_argument("--preview-width", type=int, default=560)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds; 0 runs until quit")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many processed frame sets; 0 is unlimited")
    ap.add_argument("--json", default="live_diagnostics.json")
    return ap.parse_args()


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
    """DA3 consumes RGB, so every path converges there."""
    if family == "rgb":
        return frame
    if family == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(frame, bayer_code)


class CameraWorker(threading.Thread):
    """Acquire one camera, undistort via precomputed maps, publish the latest."""

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
                raw = buffer_to_array(item, self.bpp)
                rgb = to_rgb(raw, self.family, self.bayer_code)

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
        def stats(vals):
            if not vals:
                return None
            ordered = sorted(vals)
            i95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
            return {"mean_ms": round(statistics.fmean(vals), 3),
                    "median_ms": round(statistics.median(vals), 3),
                    "p95_ms": round(ordered[i95], 3)}
        return {"frames_acquired": self.frames,
                "incomplete_buffers": self.incomplete,
                "acquisition_failures": self.failures,
                "mean_fps": round(self.frames / elapsed, 3) if elapsed > 0 else 0.0,
                "buffer_wait": stats(self.wait_ms),
                "rectify": stats(self.rectify_ms),
                "errors": self.errors}


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
    """Poll PtpStatus until every camera reports Master or Slave."""
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


def build_maps(K, dist, size, enabled):
    """Precompute the rectification maps once, replicating cv2.undistort."""
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
    """Scale intrinsics for an isotropic image resize, pixel-centre convention."""
    if s == 1.0:
        return K
    Ks = K.copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


# --------------------------------------------------------------------------
# triggering
# --------------------------------------------------------------------------

class SettleTrigger:
    """Fire once the reference view stops changing, re-arm once it moves again."""

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
            self.static_count = 0
        if self.armed and self.static_count >= self.need:
            self.armed = False
            self.static_count = 0
            return True
        return False


# --------------------------------------------------------------------------
# per-frame processing
# --------------------------------------------------------------------------

def backproject(names, depth, conf, K_out, E_out, rgb_proc, args):
    """Identical filtering and back-projection to da3_fuse.py."""
    clouds, tinted, per_cam = {}, [], []
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

        if args.edge_thresh > 0:
            before = int(valid.sum())
            valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
            drops["edge"] = before - int(valid.sum())

        pts_cam = points_camera(d, K_i)

        if args.max_incidence < 90:
            ok, _ = incidence_mask(pts_cam, args.max_incidence)
            before = int(valid.sum())
            valid &= ok
            drops["grazing"] = before - int(valid.sum())

        pts_world = to_world(pts_cam[valid], E_out[i])
        clouds[n] = to_o3d(pts_world, colors=rgb_proc[i][valid])
        if args.colour_by_camera:
            tinted.append(to_o3d(pts_world, rgb01=PALETTE[i % len(PALETTE)]))

        dv = d[valid]
        per_cam.append({
            "camera": n,
            "median_depth_m": float(np.median(dv)) if dv.size else None,
            "n_points": int(valid.sum()),
            "conf_threshold": thr,
            "dropped": drops,
        })
    return clouds, tinted, per_cam


def depth_preview(names, depth, workers, args, info_lines):
    """Colourised depth mosaic. Cheap, and it does not need the Open3D GUI."""
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
        s = args.preview_width / img.shape[1]
        img = cv2.resize(img, (args.preview_width, max(1, int(img.shape[0] * s))))
        label = f"{n}  {workers[i].fps():.1f} fps in"
        for org, col, th in (((8, 22), (0, 0, 0), 3), ((8, 22), (255, 255, 255), 1)):
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
        tiles.append(img)

    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
             if t.shape[0] < h else t for t in tiles]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    if len(rows) > 1 and rows[-1].shape[1] < rows[0].shape[1]:
        pad = np.zeros((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
        rows[-1] = np.hstack([rows[-1], pad])
    mosaic = np.vstack(rows)

    bar = np.zeros((26 * len(info_lines) + 8, mosaic.shape[1], 3), np.uint8)
    for j, line in enumerate(info_lines):
        cv2.putText(bar, line, (8, 20 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (args.out_dir / "frames.jsonl").open("w")

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
    if len({(w, h) for w, h in rect_sizes.values()}) > 1:
        print("[WARN] the undistorted images do not share a size. Mixed "
              "aspect ratios in one DA3 batch trigger a centre crop to the "
              "smallest common dimension, which shifts the principal point "
              "without changing the focal length. Verify fx_ratio in the "
              "per-frame records before trusting these clouds.")

    # ---- cameras ------------------------------------------------------
    infos = system.device_infos
    if not infos:
        raise SystemExit("no cameras found; check the Arena library path and NIC")
    found = {i["serial"]: i for i in infos}
    missing = [n for n in names if CAMERAS[n] not in found]
    if missing:
        raise SystemExit(f"missing camera(s): {', '.join(missing)}; "
                         f"detected {sorted(found)}")

    devices = system.create_device([found[CAMERAS[n]] for n in names])
    config_log = {}
    stop = threading.Event()
    workers = []
    ptp_status = None
    exit_code = 0
    processed = 0
    rejected_sync = 0
    t_measure = None
    t_loop = time.perf_counter()
    stage_ms = {k: [] for k in ("assemble", "inference", "backproject",
                                "fuse", "write", "cycle")}
    cycle_stamps = deque(maxlen=30)

    try:
        for n, dev in zip(names, devices):
            log = []
            configure(dev, args, log)
            config_log[n] = log

        if args.ptp:
            ptp_status, locked = wait_for_ptp(devices, names)
            print(f"[ptp  ] {ptp_status}" + ("" if locked else "  NOT LOCKED"))

        for n, dev in zip(names, devices):
            dev.start_stream(args.num_buffers)
            workers.append(CameraWorker(n, dev, args, maps[n], stop))
        for w in workers:
            w.start()

        # ---- model ----------------------------------------------------
        print(f"[model] loading {args.model}")
        t0 = time.perf_counter()
        model = load_model(args.model, args.device)
        print(f"[model] ready in {time.perf_counter() - t0:.1f} s")

        trigger = SettleTrigger(args.settle_thresh, args.settle_frames,
                                args.settle_rearm) if args.trigger == "settle" else None
        ref_idx = names.index(args.reference) if args.reference in names else 0

        window = "DA3 live depth"
        if args.preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        last_seqs = [0] * len(workers)
        warmups_left = max(0, args.warmup)
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
            if any(s is None for s in snap):
                time.sleep(0.01)
                if args.preview and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            if all(s[3] == last for s, last in zip(snap, last_seqs)):
                time.sleep(0.002)
                if args.preview and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            if args.use_device_timestamp and all(s[2] for s in snap):
                spread_ms = (max(s[2] for s in snap) - min(s[2] for s in snap)) / 1e6
            else:
                spread_ms = (max(s[1] for s in snap) - min(s[1] for s in snap)) * 1e3

            if args.max_sync_ms > 0 and spread_ms > args.max_sync_ms:
                rejected_sync += 1
                time.sleep(0.002)
                continue

            frames = [s[0] for s in snap]
            last_seqs = [s[3] for s in snap]
            assemble_ms = (time.perf_counter() - t0) * 1e3

            # ---- gate on the trigger ---------------------------------
            fire = True
            if trigger is not None:
                fire = trigger.update(frames[ref_idx])
            elif args.trigger == "key":
                fire = False

            key = -1
            if args.preview and not fire:
                info = [f"waiting  trigger={args.trigger}"
                        + (f"  diff={trigger.last_diff:.2f}" if trigger else ""),
                        f"sync spread {spread_ms:6.1f} ms   rejected {rejected_sync}"]
                small = [cv2.resize(f, (args.preview_width, int(f.shape[0] * args.preview_width / f.shape[1])))
                         for f in frames]
                h = max(s.shape[0] for s in small)
                small = [np.vstack([s, np.zeros((h - s.shape[0], s.shape[1], 3), np.uint8)])
                         if s.shape[0] < h else s for s in small]
                rows = [np.hstack(small[i:i + 2]) for i in range(0, len(small), 2)]
                mosaic = np.vstack(rows) if len(set(r.shape[1] for r in rows)) == 1 else rows[0]
                bar = np.zeros((26 * len(info) + 8, mosaic.shape[1], 3), np.uint8)
                for j, line in enumerate(info):
                    cv2.putText(bar, line, (8, 20 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, cv2.cvtColor(np.vstack([bar, mosaic]), cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if args.trigger == "key" and key == ord(" "):
                    fire = True
            elif not fire:
                time.sleep(0.005)

            if not fire:
                continue

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

            depth = as_numpy(pred.depth)
            conf = as_numpy(pred.conf)
            K_out = as_numpy(pred.intrinsics)
            E_out = as_numpy(pred.extrinsics)
            rgb_proc = as_numpy(pred.processed_images)
            is_metric = int(getattr(pred, "is_metric", 0))

            if warmups_left > 0:
                warmups_left -= 1
                print(f"[warm ] discarded warm-up pass, {inference_ms:.0f} ms")
                if warmups_left == 0:
                    t_measure = time.perf_counter()
                continue
            if t_measure is None:
                t_measure = time.perf_counter()

            # ---- back-project and fuse -------------------------------
            t0 = time.perf_counter()
            clouds, tinted, per_cam = backproject(names, depth, conf, K_out,
                                                  E_out, rgb_proc, args)
            backproject_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            fused = o3d.geometry.PointCloud()
            for n in names:
                fused += clouds[n]
            n_before = len(fused.points)
            if args.voxel > 0:
                fused = fused.voxel_down_sample(args.voxel)
            fuse_ms = (time.perf_counter() - t0) * 1e3

            # ---- write -----------------------------------------------
            t0 = time.perf_counter()
            if args.save_mode != "none":
                if args.save_mode == "all":
                    stem = args.out_dir / f"frame_{processed:05d}"
                    stem.mkdir(exist_ok=True)
                else:
                    stem = args.out_dir
                o3d.io.write_point_cloud(str(stem / "fused.ply"), fused)
                if args.save_per_camera:
                    for n in names:
                        o3d.io.write_point_cloud(str(stem / f"cloud_{n}.ply"), clouds[n])
                if tinted:
                    merged = o3d.geometry.PointCloud()
                    for p in tinted:
                        merged += p
                    o3d.io.write_point_cloud(str(stem / "fused_by_camera.ply"), merged)
            write_ms = (time.perf_counter() - t0) * 1e3

            cycle_ms = (time.perf_counter() - t_cycle) * 1e3
            cycle_stamps.append(time.perf_counter())
            for k, v in (("assemble", assemble_ms), ("inference", inference_ms),
                         ("backproject", backproject_ms), ("fuse", fuse_ms),
                         ("write", write_ms), ("cycle", cycle_ms)):
                stage_ms[k].append(v)

            fx_ratio = [float(K_out[i][0, 0] / (Ks_in[i][0, 0] * depth[i].shape[1] / frames[i].shape[1]))
                        for i in range(len(names))]

            record = {
                "frame": processed,
                "t_s": round(time.perf_counter() - t_loop, 4),
                "sync_spread_ms": round(spread_ms, 3),
                "is_metric": is_metric,
                "fx_ratio": [round(v, 4) for v in fx_ratio],
                "n_points_fused": int(len(fused.points)),
                "n_points_before_voxel": int(n_before),
                "per_camera": per_cam,
                "timings_ms": {"assemble": round(assemble_ms, 2),
                               "inference": round(inference_ms, 2),
                               "backproject": round(backproject_ms, 2),
                               "fuse": round(fuse_ms, 2),
                               "write": round(write_ms, 2),
                               "cycle": round(cycle_ms, 2)},
            }
            jsonl.write(json.dumps(record) + "\n")
            jsonl.flush()
            processed += 1

            rate = 0.0
            if len(cycle_stamps) > 1:
                span = cycle_stamps[-1] - cycle_stamps[0]
                rate = (len(cycle_stamps) - 1) / span if span > 0 else 0.0

            meds = [p["median_depth_m"] for p in per_cam if p["median_depth_m"]]
            depth_str = (f"  median depth {min(meds):.3f}-{max(meds):.3f} m"
                         if meds else "")
            print(f"[{processed:05d}] cycle {cycle_ms:7.1f} ms ({rate:4.2f} Hz)  "
                  f"infer {inference_ms:6.1f}  bp {backproject_ms:6.1f}  "
                  f"write {write_ms:6.1f}  pts {len(fused.points):7d}  "
                  f"sync {spread_ms:5.1f} ms{depth_str}")

            if args.preview:
                info = [f"frame {processed}   {rate:.2f} Hz   infer {inference_ms:.0f} ms   "
                        f"cycle {cycle_ms:.0f} ms",
                        f"points {len(fused.points)}   sync spread {spread_ms:.1f} ms   "
                        f"metric={is_metric}   rejected {rejected_sync}"]
                cv2.imshow(window, depth_preview(names, depth, workers, args, info))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

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
        jsonl.close()

    # ---- summary ------------------------------------------------------
    elapsed = max(time.perf_counter() - (t_measure or t_loop), 1e-9)

    def stats(vals):
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

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        "cameras": {n: CAMERAS[n] for n in names},
        "rectified_sizes": {n: list(rect_sizes[n]) for n in names},
        "ptp_status": ptp_status,
        "node_configuration": config_log,
        "frames_processed": processed,
        "measured_duration_s": round(elapsed, 3),
        "mean_cycle_hz": round(processed / elapsed, 4) if processed else 0.0,
        "sync_rejections": rejected_sync,
        "stage_timings": {k: stats(v) for k, v in stage_ms.items()},
        "per_camera_acquisition": {w.cam_name: w.summary(elapsed) for w in workers},
    }
    if torch is not None and torch.cuda.is_available():
        summary["memory"] = {
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}

    Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"\nprocessed {processed} frame sets in {elapsed:.1f} s "
          f"({summary['mean_cycle_hz']:.3f} Hz)")
    print(f"per-frame records: {args.out_dir / 'frames.jsonl'}")
    print(f"summary          : {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())