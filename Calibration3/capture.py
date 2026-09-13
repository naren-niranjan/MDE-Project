"""Auto-capture with multi-strategy ChArUco detection.

For each frame from each camera, the detector is tried with several
preprocessing strategies in sequence -- whichever finds the most corners wins:

  1. Raw grayscale
  2. CLAHE-enhanced (helps with specular highlights / uneven warehouse lighting)
  3. Half-resolution (helps when markers are big and oversampling hurts)
  4. Histogram-equalized (last-ditch global contrast)

A set is auto-saved only when ALL of these hold:
  * Every camera detects >= MIN_CORNERS_SAVE ChArUco corners.
  * The board is steady (centroid jitter <= STABILITY_MAX_PX) for STABILITY_FRAMES.
  * The capture is in sync: per-camera hardware-timestamp spread <= MAX_SYNC_SPREAD_MS.
  * Every camera's board ROI is sharp (Laplacian var >= BLUR_MIN) and not clipped
    (fraction of saturated pixels <= SAT_MAX).
  * The pose is NEW -- it adds coverage we don't already have (grid cell x area
    bucket x aspect bucket on the reference camera), so we don't pile up redundant
    frames in one spot.

Why these gates exist
---------------------
The old movement gate ("centroid moved >= N px") happily collected 50 frames
clustered in the image center at one tilt -- weak distortion/focal observability,
and it rejected the most valuable poses (an in-place tilt barely moves the
centroid). It also said nothing about whether the three cameras saw the board at
the *same instant*: with free-running acquisition and NewestOnly buffers, the
three "newest" frames can be a full frame period apart, and that residual skew is
exactly what inflates extrinsic reprojection error.

Hardware synchronization (the real fix for sync)
------------------------------------------------
Free-running cameras have independent power-on timestamp counters, so the raw
per-camera timestamp spread is meaningless (days apart) and the sync gate can
never pass. Worse, even with PTP-disciplined clocks, free-running acquisition
leaves the three cameras up to a full frame period out of phase.

So when USE_HARDWARE_SYNC is on we:
  * enable PTP (IEEE-1588) on all cameras and wait for them to elect a
    master/slave and converge on a common timebase, and
  * arm each camera to expose one frame per scheduled Action0 command.
Each loop iteration latches the current PTP time, schedules an action a few ms
in the future, and broadcasts it. All cameras start exposure at the *same* PTP
instant, so every set is simultaneous by construction and the timestamp spread
drops to microseconds. The sync gate then becomes a cheap verification rather
than the thing that's silently blocking every capture.

Set USE_HARDWARE_SYNC = False to fall back to free-running acquisition (e.g. if
PTP/action-command nodes aren't available on your firmware) -- but note the
extrinsic-accuracy problem the sync exists to solve will come back.

Keys
----
    SPACE   force a manual capture (always works, even with 0 detections)
    a       toggle auto-capture
    d       drop last set
    c       clear the coverage map (start pose-diversity tracking over)
    +/-     raise/lower MIN_CORNERS_SAVE
    s       save the current raw frames to debug_*.png so we can diagnose offline
    q / ESC quit
"""

import os
import re
import json
import time
import threading

import cv2
import numpy as np

from arena_api.system import system
from arena_api.buffer import BufferFactory

import config


# === Tuning =================================================================
PREVIEW_WIDTH       = 640
MIN_CORNERS_DETECT  = 4      # "is the board roughly here" -> used for gating/preview
MIN_CORNERS_SAVE    = 20     # a saved frame must be this rich to be worth keeping
STABILITY_FRAMES    = 8
STABILITY_MAX_PX    = 3
MIN_MOVEMENT_PX     = 15     # debounce so we don't double-capture across a cell edge
CAPTURE_COOLDOWN_S  = 0.4
DEBUG_EVERY_FRAMES  = 20

# Synchronization: max allowed spread between the three cameras' hardware
# timestamps for a set to be considered simultaneous.
#   With PTP + scheduled action commands (USE_HARDWARE_SYNC=True) all cameras
#   expose at the same PTP instant, so the real spread is well under 0.1 ms --
#   you can safely tighten this toward 0.2 once you've watched the printed
#   `sync=` values settle. 2.0 ms is a loose ceiling that also tolerates the
#   first few frames while PTP servo offset finishes converging.
MAX_SYNC_SPREAD_MS  = 2.0

# Frame-quality gates (computed on the board ROI, not the whole frame).
BLUR_MIN            = 60.0   # Laplacian variance; tune from the printed values
SAT_MAX             = 0.20   # max fraction of clipped (>=250) pixels in the ROI
QUALITY_GATE_ON     = True

# Coverage / pose-diversity buckets (evaluated on the reference camera).
GRID_COLS           = 4
GRID_ROWS           = 3
AREA_BUCKETS        = 4      # board bbox area as a fraction of the image
ASPECT_BUCKETS      = 3      # bbox aspect ratio -- crude tilt proxy w/o intrinsics

LUCID_VENDOR    = "Lucid Vision Labs"
LUCID_MODEL_SUB = "TRI050S"

# === Hardware synchronization (PTP + scheduled Action0 trigger) =============
USE_HARDWARE_SYNC   = True

# Action-command identity. The system-level fire keys MUST match the per-camera
# Action0 keys, and all three cameras share the same keys so one broadcast fires
# the whole group.
ACTION_DEVICE_KEY   = 1
ACTION_GROUP_KEY    = 1
ACTION_GROUP_MASK   = 1

# How far in the future to schedule each action, in ns. Must comfortably exceed
# the time for the broadcast to reach every camera; 20 ms is safe over GigE.
ACTION_DELTA_NS     = 20_000_000

# Block at most this long waiting for a triggered frame before treating the
# iteration as a dropped trigger and skipping it.
GET_BUFFER_TIMEOUT_MS = 2000

# How long to wait for PTP master/slave election to settle.
PTP_LOCK_TIMEOUT_S  = 20.0
# Extra settle time after election so the servo offset converges.
PTP_SETTLE_S        = 2.0


# === Detector with permissive parameters ====================================
def build_detector(board):
    """Trigger-path detector. Used ONLY to count corners and locate the board
    for gating/preview -- never for the actual calibration corners. So we
    deliberately skip sub-pixel refinement here: it's expensive at full res over
    four strategies x three cameras every frame (which starves the preview FPS,
    and thus the jitter sampling), and the accurate corners come from
    config.get_charuco_detector() offline anyway."""
    ap = cv2.aruco.DetectorParameters()
    ap.adaptiveThreshWinSizeMin   = 3
    ap.adaptiveThreshWinSizeMax   = 53
    ap.adaptiveThreshWinSizeStep  = 10
    ap.adaptiveThreshConstant     = 7
    ap.minMarkerPerimeterRate     = 0.01
    ap.maxMarkerPerimeterRate     = 4.0
    ap.polygonalApproxAccuracyRate = 0.08
    ap.minCornerDistanceRate      = 0.03
    ap.minDistanceToBorder        = 1
    ap.minMarkerDistanceRate      = 0.03
    ap.cornerRefinementMethod     = cv2.aruco.CORNER_REFINE_NONE

    cp = cv2.aruco.CharucoParameters()
    try: cp.tryRefineMarkers = True
    except AttributeError: pass
    try: cp.minMarkers = 1
    except AttributeError: pass

    return cv2.aruco.CharucoDetector(board, cp, ap)


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def detect_multistrat(img_bgr, detector):
    """Try several preprocessing strategies; return best (corners, ids, n, strat)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    strategies = [
        ("raw",   gray,                                 1.0),
        ("clahe", _CLAHE.apply(gray),                   1.0),
        ("half",  cv2.resize(gray, (W // 2, H // 2)),   2.0),
        ("eq",    cv2.equalizeHist(gray),               1.0),
    ]

    best = (None, None, 0, "none")
    for label, im, upscale in strategies:
        c, ids, _, _ = detector.detectBoard(im)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            if c is not None and upscale != 1.0:
                c = c * upscale
            best = (c, ids, n, label)
        if best[2] >= 12:  # got most of them, stop trying
            break
    return best


# === Quality / coverage helpers =============================================
def corners_bbox(corners):
    """Axis-aligned bounding box (x0, y0, x1, y1) of the detected corners."""
    pts = corners.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def roi_quality(img_bgr, corners):
    """Return (blur, sat) on the board ROI.
    blur = Laplacian variance (higher = sharper).
    sat  = fraction of pixels clipped at >=250 (specular highlights)."""
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = corners_bbox(corners)
    # clamp + tiny pad
    x0 = max(0, int(x0) - 4); y0 = max(0, int(y0) - 4)
    x1 = min(W, int(x1) + 4); y1 = min(H, int(y1) + 4)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 1.0
    roi = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    sat  = float(np.mean(roi >= 250))
    return blur, sat


def pose_signature(corners, img_w, img_h):
    """A coarse (cell_col, cell_row, area_bucket, aspect_bucket) key describing
    where/how the board sits in the reference view. Used to reject redundant
    poses. Needs no intrinsics -- area and aspect are adequate proxies for
    distance and tilt at the capture stage."""
    pts = corners.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    col = min(GRID_COLS - 1, int(cx / img_w * GRID_COLS))
    row = min(GRID_ROWS - 1, int(cy / img_h * GRID_ROWS))

    x0, y0, x1, y1 = corners_bbox(corners)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    area_frac = (bw * bh) / float(img_w * img_h)
    area_b = min(AREA_BUCKETS - 1, int(area_frac * AREA_BUCKETS / 0.6))  # ~0.6 = full

    aspect = bw / bh
    # aspect ~1 face-on; departs from 1 as the board tilts
    aspect_b = min(ASPECT_BUCKETS - 1, int(abs(np.log(aspect)) * ASPECT_BUCKETS))
    return (col, row, area_b, aspect_b)


# === Pixel format / SDK =====================================================
def _pfmt_bgr8():
    try:
        from arena_api.enums import PixelFormat
        return PixelFormat.BGR8
    except Exception:
        return 0x02180015

PFMT_BGR8 = _pfmt_bgr8()


def _buffer_timestamp_ns(buf):
    """Hardware timestamp in ns, or None if this arena_api build doesn't expose
    it. Tried in order of likelihood across SDK versions."""
    for attr in ("timestamp_ns", "timestamp"):
        try:
            v = getattr(buf, attr)
            if callable(v):
                v = v()
            if v is not None:
                return int(v)
        except Exception:
            pass
    return None


# === PTP + action-command synchronization ===================================
def _set_ptp_enable(device, on=True):
    """Enable PTP across SDK/firmware node-name variants."""
    nm = device.nodemap
    for name in ("PtpEnable", "GevIEEE1588"):
        try:
            nm[name].value = on
            return name
        except Exception:
            continue
    raise RuntimeError("No PTP enable node (PtpEnable/GevIEEE1588) on device.")


def _ptp_status(device):
    """Read PTP status string across node-name variants."""
    nm = device.nodemap
    for name in ("PtpStatus", "GevIEEE1588Status"):
        try:
            return str(nm[name].value)
        except Exception:
            continue
    return "Unknown"


def _configure_sync_trigger(device):
    """Enable PTP and arm the camera to expose on a scheduled Action0 command.
    All cameras sharing the same device/group key fire from one broadcast
    action, so every saved set is exposed at the same PTP instant."""
    nm = device.nodemap
    _set_ptp_enable(device, True)

    # Action-command target config (must match the system-level fire keys).
    try: nm["ActionUnconditionalMode"].value = "On"
    except Exception: pass
    nm["ActionSelector"].value  = 0
    nm["ActionDeviceKey"].value = ACTION_DEVICE_KEY
    nm["ActionGroupKey"].value  = ACTION_GROUP_KEY
    nm["ActionGroupMask"].value = ACTION_GROUP_MASK

    # Expose exactly one frame per Action0 pulse.
    nm["TriggerSelector"].value = "FrameStart"
    nm["TriggerMode"].value     = "On"
    nm["TriggerSource"].value   = "Action0"


def setup_action_command_system():
    """System-level scheduled-action-command config. Keys MUST match the
    per-camera Action0 keys; target is the network broadcast address so the
    single fire reaches all cameras at once."""
    snm = system.tl_system_nodemap
    snm["ActionCommandDeviceKey"].value = ACTION_DEVICE_KEY
    snm["ActionCommandGroupKey"].value  = ACTION_GROUP_KEY
    snm["ActionCommandGroupMask"].value = ACTION_GROUP_MASK
    snm["ActionCommandTargetIP"].value  = 0xFFFFFFFF  # broadcast


def wait_for_ptp_lock(devices, timeout_s=PTP_LOCK_TIMEOUT_S):
    """Block until exactly one camera is Master and the rest are Slave."""
    t0 = time.time()
    statuses = {}
    while time.time() - t0 < timeout_s:
        statuses = {n: _ptp_status(d) for n, d in devices.items()}
        masters = [n for n, s in statuses.items() if s == "Master"]
        slaves  = [n for n, s in statuses.items() if s == "Slave"]
        if len(masters) == 1 and len(slaves) == len(devices) - 1:
            print(f"  PTP locked: master={masters[0]}, slaves={slaves}")
            return statuses
        print("  PTP settling: "
              + ", ".join(f"{n}={s}" for n, s in statuses.items()))
        time.sleep(1.0)
    raise RuntimeError(
        f"PTP failed to elect master/slave within {timeout_s:.0f}s. "
        f"Last status: {statuses}. Check that all cameras are on the same "
        f"network segment/switch and that PTP is supported.")


def latch_ptp_ns(device):
    """Latch and read the device's current PTP time (ns)."""
    nm = device.nodemap
    for cmd, val in (("PtpDataSetLatch",  "PtpDataSetLatchValue"),
                     ("TimestampLatch",   "TimestampLatchValue")):
        try:
            nm[cmd].execute()
            return int(nm[val].value)
        except Exception:
            continue
    raise RuntimeError("Cannot latch PTP/timestamp on reference device.")


def fire_synced_capture(ref_device):
    """Schedule and broadcast one Action0 a few ms in the future. Every armed
    camera exposes a single frame at that shared PTP instant. Returns the
    scheduled execute time (ns)."""
    t_exec = latch_ptp_ns(ref_device) + ACTION_DELTA_NS
    snm = system.tl_system_nodemap
    snm["ActionCommandExecuteTime"].value = t_exec
    snm["ActionCommandFireCommand"].execute()
    return t_exec


def configure_device(device, use_sync=USE_HARDWARE_SYNC):
    nm = device.nodemap
    sm = device.tl_stream_nodemap
    sm["StreamAutoNegotiatePacketSize"].value = True
    sm["StreamPacketResendEnable"].value      = True
    sm["StreamBufferHandlingMode"].value      = "NewestOnly"
    nm["Width"].value       = config.WIDTH
    nm["Height"].value      = config.HEIGHT
    nm["PixelFormat"].value = "BayerRG8"
    # NOTE: for multi-camera consistency, pinning ExposureTime/Gain to fixed
    # values is preferable to Auto -- auto-exposure hunts independently per
    # camera, drifting board brightness between views and frame-to-frame, and
    # the hunting itself can introduce blur. Driven from config for now.
    for k, v in (("ExposureAuto", config.EXPOSURE_AUTO),
                 ("GainAuto",     config.GAIN_AUTO)):
        try:    nm[k].value = v
        except Exception: pass
    nm["AcquisitionMode"].value = "Continuous"

    if use_sync:
        # PTP + Action0 trigger must be set before start_stream(). In triggered
        # mode the camera produces no frames until the first action fires.
        _configure_sync_trigger(device)


def grab_bgr(device, timeout_ms=GET_BUFFER_TIMEOUT_MS):
    """Return (bgr_image, timestamp_ns_or_None)."""
    buf = device.get_buffer(timeout=timeout_ms)
    try:
        ts = _buffer_timestamp_ns(buf)
        converted = BufferFactory.convert(buf, PFMT_BGR8)
        try:
            h, w = converted.height, converted.width
            flat = np.ctypeslib.as_array(converted.pdata, shape=(h * w * 3,))
            bgr  = flat.reshape(h, w, 3).copy()
        finally:
            BufferFactory.destroy(converted)
    finally:
        device.requeue_buffer(buf)
    return bgr, ts


def open_cameras(use_sync=USE_HARDWARE_SYNC):
    all_infos = system.device_infos
    lucid_infos = [d for d in all_infos
                   if d.get("vendor") == LUCID_VENDOR
                   and LUCID_MODEL_SUB in d.get("model", "")]
    if not lucid_infos:
        raise RuntimeError(f"No Lucid {LUCID_MODEL_SUB} cameras found.")
    devices_all = system.create_device(lucid_infos)
    by_serial = {}
    for d in devices_all:
        try:    by_serial[d.nodemap["DeviceSerialNumber"].value] = d
        except Exception: pass
    devices = {}
    for name, serial in config.CAMERAS.items():
        if serial == "FILL_IN_SERIAL":
            raise RuntimeError("config.py still has FILL_IN_SERIAL.")
        if serial not in by_serial:
            raise RuntimeError(f"Camera '{name}' serial {serial} not found.")
        devices[name] = by_serial[serial]
        configure_device(devices[name], use_sync=use_sync)

    # Start every stream. In triggered mode no frames flow until the first
    # action fires, which is fine -- PTP election runs regardless of streaming.
    for d in devices.values():
        d.start_stream()

    if use_sync:
        setup_action_command_system()
        print("Waiting for PTP lock (this takes a few seconds)...")
        wait_for_ptp_lock(devices)
        time.sleep(PTP_SETTLE_S)  # let the servo offset converge
        print(f"  PTP settled. Action-command sync armed "
              f"(lead={ACTION_DELTA_NS/1e6:.0f}ms).")

    return devices


def parallel_grab(devices):
    """Return (frames: name->bgr, timestamps: name->ts_ns_or_None)."""
    results = {}
    def worker(name, dev):
        try:    results[name] = grab_bgr(dev)
        except Exception as e: results[name] = e
    threads = [threading.Thread(target=worker, args=(n, d), daemon=True)
               for n, d in devices.items()]
    for t in threads: t.start()
    for t in threads: t.join()
    frames, timestamps = {}, {}
    for n, v in results.items():
        if isinstance(v, Exception): raise v
        frames[n], timestamps[n] = v
    return frames, timestamps


def synced_grab(devices, ref_cam, use_sync=USE_HARDWARE_SYNC):
    """Fire one synchronized action (if enabled) then collect the resulting
    frames. Falls back to a plain free-running grab when sync is off."""
    if use_sync:
        fire_synced_capture(devices[ref_cam])
    return parallel_grab(devices)


def parallel_detect(frames, detector):
    """name -> (corners_full_res, ids, n_corners, strategy_label)."""
    results = {}
    def worker(name, img):
        results[name] = detect_multistrat(img, detector)
    threads = [threading.Thread(target=worker, args=(n, img), daemon=True)
               for n, img in frames.items()]
    for t in threads: t.start()
    for t in threads: t.join()
    return results


def centroid_of(corners):
    return corners.reshape(-1, 2).mean(axis=0)


def sync_spread_ms(timestamps):
    """Spread (ms) between camera hardware timestamps, or None if unavailable."""
    vals = [t for t in timestamps.values() if t is not None]
    if len(vals) < 2:
        return None
    return (max(vals) - min(vals)) / 1e6


def draw_tile(preview_bgr, corners_full, ids, n_corners, strat, ok,
              label, scale_full_to_preview):
    out = preview_bgr.copy()
    if corners_full is not None and ids is not None and len(ids) > 0:
        cp = (corners_full * scale_full_to_preview).astype(np.float32)
        cv2.aruco.drawDetectedCornersCharuco(out, cp, ids,
                                             (0,255,0) if ok else (0,165,255))
    color = (0,255,0) if ok else (0,165,255)
    cv2.putText(out, f"{label}: {n_corners}c ({strat})",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def save_set(frames, counter, meta, base_dir):
    """Write each camera's frame plus one sidecar JSON per set."""
    paths = {}
    for name, img in frames.items():
        p = os.path.join(base_dir, name, f"frame_{counter:04d}.png")
        cv2.imwrite(p, img)
        paths[name] = p
    meta_dir = os.path.join(base_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"set_{counter:04d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return paths


def save_debug(frames):
    """Save raw frames to debug_*.png in cwd, for offline analysis."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    for name, img in frames.items():
        cv2.imwrite(f"debug_{name}_{ts}.png", img)
    print(f"  wrote debug_*_{ts}.png in {os.getcwd()}")


def next_index(base_dir, cameras):
    """max(existing index) + 1 -- robust to gaps left by dropped sets, unlike a
    plain count which would collide and overwrite."""
    first_dir = os.path.join(base_dir, next(iter(cameras)))
    idxs = []
    for f in os.listdir(first_dir):
        m = re.match(r"frame_(\d+)\.png$", f)
        if m:
            idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0


def reference_camera(cameras):
    ref = getattr(config, "REFERENCE_CAMERA", None)
    if ref and ref in cameras:
        return ref
    return next(iter(cameras))


def main():
    global MIN_CORNERS_SAVE

    base_dir = config.CAPTURE_DIR
    os.makedirs(base_dir, exist_ok=True)
    for name in config.CAMERAS:
        os.makedirs(os.path.join(base_dir, name), exist_ok=True)

    devices  = open_cameras()
    board    = config.get_charuco_board()
    detector = build_detector(board)

    ref_cam = reference_camera(config.CAMERAS)
    counter = next_index(base_dir, config.CAMERAS)

    auto_mode    = True
    stable_count = 0
    prev_cents   = None
    last_cent    = None        # reference-camera centroid of last captured set
    last_t       = 0.0
    last_set     = None
    frame_count  = 0
    covered      = set()       # pose signatures we already have

    print(f"Streaming {len(devices)} camera(s). Resuming at index {counter}.")
    print(f"Reference camera: {ref_cam}. "
          f"Save threshold: {MIN_CORNERS_SAVE} corners.")
    sync_mode = "hardware (PTP+Action0)" if USE_HARDWARE_SYNC else "free-run"
    print(f"Sync mode: {sync_mode}.")
    print("Gates: sync <= %.1fms, blur >= %.0f, sat <= %.2f, new pose only."
          % (MAX_SYNC_SPREAD_MS, BLUR_MIN, SAT_MAX))
    print("SPACE = manual save. 's' dumps debug frames. 'c' clears coverage.\n")

    try:
        while True:
            frame_count += 1

            # Fire a synchronized action (if enabled) and collect the set. A
            # dropped action packet or transient GigE hiccup just times out --
            # skip the iteration rather than crashing the capture session.
            try:
                frames, timestamps = synced_grab(devices, ref_cam,
                                                  USE_HARDWARE_SYNC)
            except Exception as e:
                print(f"  grab skipped ({e}); retrying")
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
                continue

            detections = parallel_detect(frames, detector)

            usable = {n: detections[n][2] >= MIN_CORNERS_SAVE for n in detections}
            all_ok = all(usable.values())
            spread = sync_spread_ms(timestamps)

            # Per-camera quality on the board ROI (only meaningful when detected).
            quality = {}
            for n in detections:
                c = detections[n][0]
                if c is not None and detections[n][2] >= MIN_CORNERS_DETECT:
                    quality[n] = roi_quality(frames[n], c)
                else:
                    quality[n] = (0.0, 1.0)

            if frame_count % DEBUG_EVERY_FRAMES == 0:
                summary = ", ".join(
                    f"{n}:{detections[n][2]}({detections[n][3]})"
                    f"[b{quality[n][0]:.0f}/s{quality[n][1]:.02f}]"
                    for n in detections)
                sp = "n/a" if spread is None else f"{spread:.3f}ms"
                print(f"  [{frame_count}] {summary}   sync={sp}  "
                      f"stable={stable_count}/{STABILITY_FRAMES} "
                      f"poses={len(covered)} min={MIN_CORNERS_SAVE}")

            status_msg = ""
            if auto_mode:
                if all_ok:
                    cents = {n: centroid_of(detections[n][0]) for n in detections}
                    if prev_cents is not None:
                        jitter = max(np.linalg.norm(cents[n] - prev_cents[n])
                                     for n in cents)
                    else:
                        jitter = 0.0
                    if jitter <= STABILITY_MAX_PX:
                        stable_count += 1
                    else:
                        stable_count = 0
                    prev_cents = cents

                    ready = (stable_count >= STABILITY_FRAMES
                             and (time.time() - last_t) > CAPTURE_COOLDOWN_S)

                    if ready:
                        # --- sync gate ---
                        if spread is not None and spread > MAX_SYNC_SPREAD_MS:
                            status_msg = f"UNSYNCED {spread:.1f}ms (skip)"
                            stable_count = 0
                            prev_cents = None
                        else:
                            # --- quality gate ---
                            bad_q = [
                                n for n in quality
                                if QUALITY_GATE_ON and (
                                    quality[n][0] < BLUR_MIN
                                    or quality[n][1] > SAT_MAX)
                            ]
                            if bad_q:
                                worst = bad_q[0]
                                b, s = quality[worst]
                                status_msg = (f"low quality {worst} "
                                              f"(b{b:.0f}/s{s:.02f}) - reframe")
                            else:
                                # --- coverage gate ---
                                sig = pose_signature(
                                    detections[ref_cam][0],
                                    config.WIDTH, config.HEIGHT)
                                ref_c = cents[ref_cam]
                                moved = (last_cent is None
                                         or np.linalg.norm(ref_c - last_cent)
                                            >= MIN_MOVEMENT_PX)
                                if sig in covered:
                                    status_msg = "pose covered - new angle/dist"
                                elif not moved:
                                    status_msg = "too close to last - move board"
                                else:
                                    meta = {
                                        "index": counter,
                                        "timestamps_ns": timestamps,
                                        "sync_spread_ms": spread,
                                        "corners": {n: int(detections[n][2])
                                                    for n in detections},
                                        "strategy": {n: detections[n][3]
                                                     for n in detections},
                                        "quality": {n: {"blur": quality[n][0],
                                                        "sat": quality[n][1]}
                                                    for n in detections},
                                        "pose_signature": list(sig),
                                        "manual": False,
                                        "time": time.strftime("%Y%m%d_%H%M%S"),
                                    }
                                    paths = save_set(frames, counter,
                                                     meta, base_dir)
                                    last_set = (counter, paths, False)
                                    covered.add(sig)
                                    counter += 1
                                    last_cent    = ref_c
                                    last_t       = time.time()
                                    stable_count = 0
                                    prev_cents   = None
                                    status_msg   = "CAPTURED"
                                    print(f"  auto-captured set #{counter - 1} "
                                          f"({len(covered)} poses) sig={sig}")
                else:
                    stable_count = 0
                    prev_cents = None
                    weak = [f"{n}({detections[n][2]})"
                            for n, ok in usable.items() if not ok]
                    status_msg = f"need >={MIN_CORNERS_SAVE}: {', '.join(weak)}"

            tiles = []
            for name in config.CAMERAS:
                img = frames[name]
                h, w = img.shape[:2]
                ph = int(h * PREVIEW_WIDTH / w)
                preview = cv2.resize(img, (PREVIEW_WIDTH, ph))
                scale = PREVIEW_WIDTH / w
                c, ids, nc, strat = detections[name]
                tiles.append(draw_tile(preview, c, ids, nc, strat,
                                       usable[name], name, scale))
            H = max(t.shape[0] for t in tiles)
            padded = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
                      for t in tiles]
            view = np.hstack(padded)

            mode = "AUTO" if auto_mode else "MANUAL"
            bar_color = (200, 255, 200) if all_ok else (160, 160, 255)
            bar = np.zeros((50, view.shape[1], 3), dtype=np.uint8)
            sp = "n/a" if spread is None else f"{spread:.3f}ms"
            txt = (f"{mode}  captures={counter}  poses={len(covered)}  "
                   f"stable={stable_count:2d}/{STABILITY_FRAMES}  "
                   f"sync={sp}  min={MIN_CORNERS_SAVE}  {status_msg}")
            cv2.putText(bar, txt, (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, bar_color, 2, cv2.LINE_AA)
            view = np.vstack([bar, view])

            cv2.imshow("calibration capture", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == 32:
                # Manual escape hatch. Sub-threshold manual saves go to a
                # separate manual/ tree so they don't pollute the frame count
                # used to cross-check intrinsic vs extrinsic shared views.
                sub_threshold = not all_ok
                target_dir = (os.path.join(base_dir, "manual")
                              if sub_threshold else base_dir)
                if sub_threshold:
                    for name in config.CAMERAS:
                        os.makedirs(os.path.join(target_dir, name),
                                    exist_ok=True)
                m_idx = next_index(target_dir, config.CAMERAS)
                meta = {
                    "index": m_idx,
                    "timestamps_ns": timestamps,
                    "sync_spread_ms": spread,
                    "corners": {n: int(detections[n][2]) for n in detections},
                    "manual": True,
                    "sub_threshold": sub_threshold,
                    "time": time.strftime("%Y%m%d_%H%M%S"),
                }
                paths = save_set(frames, m_idx, meta, target_dir)
                last_set = (m_idx, paths, sub_threshold)
                last_t = time.time()
                stable_count = 0
                prev_cents = None
                if not sub_threshold:
                    counter = next_index(base_dir, config.CAMERAS)
                    last_cent = centroid_of(detections[ref_cam][0])
                    covered.add(pose_signature(detections[ref_cam][0],
                                               config.WIDTH, config.HEIGHT))
                where = "manual/" if sub_threshold else ""
                print(f"  manual set #{m_idx} {where}".rstrip())
            elif k == ord('a'):
                auto_mode = not auto_mode
                stable_count = 0
                prev_cents = None
                print(f"  auto-capture {'ON' if auto_mode else 'OFF'}")
            elif k == ord('d') and last_set is not None:
                idx, paths, was_sub = last_set
                for p in paths.values():
                    try:    os.remove(p)
                    except OSError: pass
                droot = os.path.join(base_dir, "manual") if was_sub else base_dir
                try:    os.remove(os.path.join(droot, "meta",
                                               f"set_{idx:04d}.json"))
                except OSError: pass
                if not was_sub:
                    counter   = next_index(base_dir, config.CAMERAS)
                    last_cent = None
                last_set = None
                print(f"  dropped set #{idx}")
            elif k == ord('c'):
                covered.clear()
                last_cent = None
                print("  coverage map cleared")
            elif k in (ord('+'), ord('=')):
                MIN_CORNERS_SAVE = min(40, MIN_CORNERS_SAVE + 1)
                print(f"  MIN_CORNERS_SAVE -> {MIN_CORNERS_SAVE}")
            elif k == ord('-'):
                MIN_CORNERS_SAVE = max(1, MIN_CORNERS_SAVE - 1)
                print(f"  MIN_CORNERS_SAVE -> {MIN_CORNERS_SAVE}")
            elif k == ord('s'):
                save_debug(frames)
    finally:
        for d in devices.values():
            try:    d.stop_stream()
            except Exception: pass
        system.destroy_device()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()