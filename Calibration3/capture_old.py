"""Auto-capture with multi-strategy ChArUco detection.

For each frame from each camera, the detector is tried with several
preprocessing strategies in sequence -- whichever finds the most corners wins:

  1. Raw grayscale
  2. CLAHE-enhanced (helps with specular highlights / uneven warehouse lighting)
  3. Half-resolution (helps when markers are big and oversampling hurts)
  4. Histogram-equalized (last-ditch global contrast)

A set is auto-saved when:
  * Every camera detects >= MIN_CORNERS ChArUco corners.
  * The board is steady (centroid jitter <= STABILITY_MAX_PX) for STABILITY_FRAMES.
  * The board has moved >= MIN_MOVEMENT_PX from the previously captured pose.
  * The capture is in sync: inter-camera hardware-timestamp spread
    <= MAX_SYNC_SPREAD_MS (only enforced when USE_HARDWARE_SYNC is on).

Hardware synchronization (PTP + scheduled Action0)
--------------------------------------------------
Free-running acquisition with NewestOnly buffers leaves the three cameras up to a
full frame period (~33ms at 30fps) out of phase, and that residual skew is what
inflates extrinsic reprojection error. Worse, each camera's free-running
power-on timestamp counter sits on its own epoch, so the raw spread is
meaningless (days apart).

So when USE_HARDWARE_SYNC is on we:
  * enable PTP (IEEE-1588) on all cameras and wait for them to elect a
    master/slave and converge on a common timebase, and
  * arm each camera to expose one frame per scheduled Action0 command.
Each loop iteration latches the current PTP time, schedules an action a few ms in
the future, and broadcasts it to all cameras at once. Every camera starts
exposure at the *same* PTP instant, so a set is simultaneous by construction and
the printed `sync=` spread drops to microseconds -- the sync gate becomes a cheap
sanity check rather than something that silently blocks captures.

Set USE_HARDWARE_SYNC = False to fall back to free-running parallel grabs (e.g.
if PTP / action-command nodes aren't exposed on your firmware) -- but the
extrinsic-accuracy problem the sync exists to solve will come back.

Keys
----
    SPACE   force a manual capture (always works, even with 0 detections)
    a       toggle auto-capture
    d       drop last set
    +/-     raise/lower MIN_CORNERS
    s       save the current raw frames to debug_*.png so we can diagnose offline
    q / ESC quit
"""

import os
import time
import threading

import cv2
import numpy as np

from arena_api.system import system
from arena_api.buffer import BufferFactory

import config


# === Tuning =================================================================
PREVIEW_WIDTH       = 640
MIN_CORNERS         = 4
STABILITY_FRAMES    = 8
STABILITY_MAX_PX    = 15
MIN_MOVEMENT_PX     = 120
CAPTURE_COOLDOWN_S  = 0.4
DEBUG_EVERY_FRAMES  = 20

LUCID_VENDOR    = "Lucid Vision Labs"
LUCID_MODEL_SUB = "TRI050S"


# === Hardware synchronization (PTP + scheduled Action0 trigger) =============
USE_HARDWARE_SYNC   = True

# Max allowed spread between the cameras' hardware timestamps for a set to count
# as simultaneous. With PTP + scheduled actions the real spread is well under
# 0.1ms; 2.0ms is a loose ceiling that also tolerates the first few frames while
# the PTP servo offset finishes converging. Tighten toward 0.2 once you've
# watched the printed `sync=` values settle.
MAX_SYNC_SPREAD_MS  = 2.0

# Action-command identity (must match across every camera + the broadcast fire).
ACTION_DEVICE_KEY   = 1
ACTION_GROUP_KEY    = 1
ACTION_GROUP_MASK   = 1
ACTION_TARGET_IP    = 0xFFFFFFFF      # broadcast to every camera on the segment
ACTION_DELTA_NS     = 20_000_000      # schedule 20ms into the future each fire

PTP_LOCK_TIMEOUT_S  = 20.0           # how long to wait for master/slave election
PTP_SETTLE_S        = 2.0            # extra servo-settle once locked
GRAB_TIMEOUT_MS     = 2000          # per-camera buffer wait after a fire


# --- small nodemap helpers --------------------------------------------------
def _node_set_first(nodemap, names, value):
    """Set the first node name that exists. Returns the name used, or None."""
    for n in names:
        try:
            nodemap[n].value = value
            return n
        except Exception:
            continue
    return None


def _node_get_first(nodemap, names):
    """Return (name, value) for the first node that exists, else (None, None)."""
    for n in names:
        try:
            return n, nodemap[n].value
        except Exception:
            continue
    return None, None


def _node_exec_first(nodemap, names):
    """Execute the first command node that exists. Returns the name, or None."""
    for n in names:
        try:
            nodemap[n].execute()
            return n
        except Exception:
            continue
    return None


# === Detector with permissive parameters ====================================
def build_detector(board):
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
    ap.cornerRefinementMethod     = cv2.aruco.CORNER_REFINE_SUBPIX
    ap.cornerRefinementWinSize    = 5
    ap.cornerRefinementMaxIterations = 30
    ap.cornerRefinementMinAccuracy   = 0.1

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


# === Pixel format / SDK =====================================================
def _pfmt_bgr8():
    try:
        from arena_api.enums import PixelFormat
        return PixelFormat.BGR8
    except Exception:
        return 0x02180015

PFMT_BGR8 = _pfmt_bgr8()


def _buffer_timestamp_ns(buf):
    """Read a buffer's hardware timestamp in ns, tolerating SDK attr naming."""
    for attr in ("timestamp_ns", "timestamp"):
        try:
            v = getattr(buf, attr)
            return int(v() if callable(v) else v)
        except Exception:
            continue
    return None


def configure_device(device):
    nm = device.nodemap
    sm = device.tl_stream_nodemap
    sm["StreamAutoNegotiatePacketSize"].value = True
    sm["StreamPacketResendEnable"].value      = True
    sm["StreamBufferHandlingMode"].value      = "NewestOnly"
    nm["Width"].value       = config.WIDTH
    nm["Height"].value      = config.HEIGHT
    nm["PixelFormat"].value = "BayerRG8"
    for k, v in (("ExposureAuto", config.EXPOSURE_AUTO),
                 ("GainAuto",     config.GAIN_AUTO)):
        try:    nm[k].value = v
        except Exception: pass
    nm["AcquisitionMode"].value = "Continuous"

    if USE_HARDWARE_SYNC:
        _configure_sync_trigger(device)


def _configure_sync_trigger(device):
    """Enable PTP and arm the camera to expose one frame per Action0 command.

    Done BEFORE start_stream(). Node names vary across firmware, so each set
    tolerates the common variants.
    """
    nm = device.nodemap

    # Put the camera on the PTP timebase.
    _node_set_first(nm, ("PtpEnable", "GevIEEE1588"), True)

    # Trigger every FrameStart off Action0 instead of free-running.
    _node_set_first(nm, ("TriggerSelector",), "FrameStart")
    _node_set_first(nm, ("TriggerMode",), "On")
    _node_set_first(nm, ("TriggerSource",), "Action0")

    # Action-command identity -- must match the broadcast fire below.
    _node_set_first(nm, ("ActionUnconditionalMode",), "On")
    _node_set_first(nm, ("ActionSelector",), 0)
    _node_set_first(nm, ("ActionDeviceKey",), ACTION_DEVICE_KEY)
    _node_set_first(nm, ("ActionGroupKey",),  ACTION_GROUP_KEY)
    _node_set_first(nm, ("ActionGroupMask",), ACTION_GROUP_MASK)


def _configure_action_broadcast():
    """Configure the system-level scheduled-action broadcast keys (once)."""
    snm = system.tl_system_nodemap
    _node_set_first(snm, ("ActionCommandDeviceKey",), ACTION_DEVICE_KEY)
    _node_set_first(snm, ("ActionCommandGroupKey",),  ACTION_GROUP_KEY)
    _node_set_first(snm, ("ActionCommandGroupMask",), ACTION_GROUP_MASK)
    _node_set_first(snm, ("ActionCommandTargetIP",),  ACTION_TARGET_IP)


def _ptp_status(device):
    _, val = _node_get_first(device.nodemap, ("PtpStatus", "GevIEEE1588Status"))
    return val


def wait_for_ptp_lock(devices, timeout_s=PTP_LOCK_TIMEOUT_S):
    """Block until every camera reports Master or Slave (PTP converged)."""
    print("  waiting for PTP master/slave election ...")
    t0 = time.time()
    locked_states = ("Master", "Slave")
    while time.time() - t0 < timeout_s:
        states = {n: _ptp_status(d) for n, d in devices.items()}
        if all(s in locked_states for s in states.values()):
            print(f"  PTP locked: {states}")
            time.sleep(PTP_SETTLE_S)  # let the servo offset settle
            return True
        time.sleep(0.5)
    print(f"  WARNING: PTP did not lock within {timeout_s:.0f}s "
          f"(states={states}). Are all cameras on the same switch/segment?")
    return False


def _latch_ptp_ns(ref_device):
    """Latch and read the reference camera's current PTP time in ns."""
    nm = ref_device.nodemap
    _node_exec_first(nm, ("PtpDataSetLatch", "TimestampLatch"))
    _, val = _node_get_first(nm, ("PtpDataSetLatchValue", "TimestampLatchValue"))
    return None if val is None else int(val)


def fire_scheduled_action(ref_device, delta_ns=ACTION_DELTA_NS):
    """Latch PTP now, schedule an action delta_ns out, broadcast it.

    Returns the scheduled execute-time in ns (or None if PTP couldn't be read).
    """
    now_ns = _latch_ptp_ns(ref_device)
    if now_ns is None:
        return None
    exec_ns = now_ns + delta_ns
    snm = system.tl_system_nodemap
    _node_set_first(snm, ("ActionCommandExecuteTime",), exec_ns)
    _node_exec_first(snm, ("ActionCommandFireCommand",))
    return exec_ns


def grab_bgr(device, timeout_ms=None):
    """Grab one buffer -> (bgr, timestamp_ns)."""
    buf = device.get_buffer(timeout=timeout_ms) if timeout_ms else device.get_buffer()
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


def open_cameras():
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
        configure_device(devices[name])       # configures PTP/trigger pre-stream
        devices[name].start_stream()

    if USE_HARDWARE_SYNC:
        _configure_action_broadcast()
        wait_for_ptp_lock(devices)
    return devices


def _reference_name(devices):
    """Camera to latch PTP from -- prefer config.REFERENCE_CAMERA, else center."""
    ref = getattr(config, "REFERENCE_CAMERA", None)
    if ref in devices:
        return ref
    if "center" in devices:
        return "center"
    return next(iter(devices))


def parallel_grab(devices):
    """Free-running fallback (USE_HARDWARE_SYNC=False). name -> bgr."""
    results = {}
    def worker(name, dev):
        try:    results[name] = grab_bgr(dev)[0]
        except Exception as e: results[name] = e
    threads = [threading.Thread(target=worker, args=(n, d), daemon=True)
               for n, d in devices.items()]
    for t in threads: t.start()
    for t in threads: t.join()
    for v in results.values():
        if isinstance(v, Exception): raise v
    return results


def synced_grab(devices, ref_name):
    """Fire one scheduled action, then collect the triggered frame from each cam.

    Returns (frames, timestamps) where frames is name->bgr and timestamps is
    name->timestamp_ns. Raises if any camera errors; the caller skips the frame.
    """
    fire_scheduled_action(devices[ref_name])

    frames, stamps = {}, {}
    def worker(name, dev):
        try:
            bgr, ts = grab_bgr(dev, timeout_ms=GRAB_TIMEOUT_MS)
            frames[name] = bgr
            stamps[name] = ts
        except Exception as e:
            frames[name] = e
    threads = [threading.Thread(target=worker, args=(n, d), daemon=True)
               for n, d in devices.items()]
    for t in threads: t.start()
    for t in threads: t.join()
    for v in frames.values():
        if isinstance(v, Exception): raise v
    return frames, stamps


def sync_spread_ms(stamps):
    """Max inter-camera timestamp spread in ms, or None if unavailable."""
    vals = [t for t in stamps.values() if t is not None]
    if len(vals) < 2:
        return None
    return (max(vals) - min(vals)) / 1e6


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


def save_set(frames, counter):
    paths = {}
    for name, img in frames.items():
        p = os.path.join(config.CAPTURE_DIR, name, f"frame_{counter:04d}.png")
        cv2.imwrite(p, img)
        paths[name] = p
    return paths


def save_debug(frames):
    """Save raw frames to debug_*.png in cwd, for offline analysis."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    for name, img in frames.items():
        cv2.imwrite(f"debug_{name}_{ts}.png", img)
    print(f"  wrote debug_*_{ts}.png in {os.getcwd()}")


def main():
    global MIN_CORNERS

    os.makedirs(config.CAPTURE_DIR, exist_ok=True)
    for name in config.CAMERAS:
        os.makedirs(os.path.join(config.CAPTURE_DIR, name), exist_ok=True)

    devices  = open_cameras()
    ref_name = _reference_name(devices)
    board    = config.get_charuco_board()
    detector = build_detector(board)

    first_dir = os.path.join(config.CAPTURE_DIR, next(iter(config.CAMERAS)))
    counter   = len([f for f in os.listdir(first_dir) if f.startswith("frame_")])

    auto_mode    = True
    stable_count = 0
    prev_cents   = None
    last_cents   = None
    last_t       = 0.0
    last_set     = None
    frame_count  = 0
    spread_ms    = None

    sync_label = ("hardware-synced (PTP + Action0)" if USE_HARDWARE_SYNC
                  else "FREE-RUNNING (USE_HARDWARE_SYNC=False)")
    print(f"Streaming {len(devices)} camera(s). Resuming at index {counter}.")
    print(f"Sync: {sync_label}. Reference camera: {ref_name}.")
    print("Multi-strategy detection. SPACE saves manually. "
          "'s' dumps debug frames if nothing is detecting.\n")

    try:
        while True:
            frame_count += 1

            if USE_HARDWARE_SYNC:
                try:
                    frames, stamps = synced_grab(devices, ref_name)
                except Exception as e:
                    # a dropped trigger packet shouldn't kill the session
                    if frame_count % DEBUG_EVERY_FRAMES == 0:
                        print(f"  [sync] grab skipped: {e}")
                    continue
                spread_ms = sync_spread_ms(stamps)
            else:
                frames    = parallel_grab(devices)
                spread_ms = None

            detections = parallel_detect(frames, detector)

            usable = {n: detections[n][2] >= MIN_CORNERS for n in detections}
            all_ok = all(usable.values())
            in_sync = (spread_ms is None) or (spread_ms <= MAX_SYNC_SPREAD_MS)

            if frame_count % DEBUG_EVERY_FRAMES == 0:
                summary = ", ".join(
                    f"{n}:{detections[n][2]}({detections[n][3]})"
                    for n in detections)
                sync_dbg = "n/a" if spread_ms is None else f"{spread_ms:.3f}ms"
                print(f"  [{frame_count}] {summary}   sync={sync_dbg} "
                      f"stable={stable_count}/{STABILITY_FRAMES} "
                      f"min={MIN_CORNERS}")

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

                    moved_enough = True
                    if last_cents is not None:
                        mean_move = np.mean([
                            np.linalg.norm(cents[n] - last_cents[n])
                            for n in cents])
                        if mean_move < MIN_MOVEMENT_PX:
                            moved_enough = False
                            stable_count = 0
                            status_msg = (f"move board "
                                          f"({mean_move:.0f}/{MIN_MOVEMENT_PX}px)")

                    if not in_sync:
                        stable_count = 0
                        status_msg = (f"out of sync "
                                      f"({spread_ms:.2f}/{MAX_SYNC_SPREAD_MS}ms)")

                    if (moved_enough
                        and in_sync
                        and stable_count >= STABILITY_FRAMES
                        and (time.time() - last_t) > CAPTURE_COOLDOWN_S):
                        paths = save_set(frames, counter)
                        last_set = (counter, paths)
                        counter += 1
                        last_cents   = cents
                        last_t       = time.time()
                        stable_count = 0
                        prev_cents   = None
                        status_msg   = "CAPTURED"
                        print(f"  auto-captured set #{counter - 1} "
                              f"(sync={'n/a' if spread_ms is None else f'{spread_ms:.3f}ms'})")
                else:
                    stable_count = 0
                    prev_cents = None
                    weak = [f"{n}({detections[n][2]})"
                            for n, ok in usable.items() if not ok]
                    status_msg = f"need >={MIN_CORNERS}: {', '.join(weak)}"

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
            bar_color = (200, 255, 200) if (all_ok and in_sync) else (160, 160, 255)
            bar = np.zeros((50, view.shape[1], 3), dtype=np.uint8)
            sync_str = "sync=--" if spread_ms is None else f"sync={spread_ms:.3f}ms"
            txt = (f"{mode}  captures={counter}  {sync_str}  "
                   f"stable={stable_count:2d}/{STABILITY_FRAMES}  "
                   f"min={MIN_CORNERS}  {status_msg}")
            cv2.putText(bar, txt, (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, bar_color, 2, cv2.LINE_AA)
            view = np.vstack([bar, view])

            cv2.imshow("calibration capture", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == 32:
                paths = save_set(frames, counter)
                last_set = (counter, paths)
                counter += 1
                last_t = time.time()
                stable_count = 0
                prev_cents = None
                if all_ok:
                    last_cents = {n: centroid_of(detections[n][0])
                                  for n in detections}
                print(f"  manual set #{counter - 1}")
            elif k == ord('a'):
                auto_mode = not auto_mode
                stable_count = 0
                prev_cents = None
                print(f"  auto-capture {'ON' if auto_mode else 'OFF'}")
            elif k == ord('d') and last_set is not None:
                idx, paths = last_set
                for p in paths.values():
                    try:    os.remove(p)
                    except OSError: pass
                counter    = idx
                last_set   = None
                last_cents = None
                print(f"  dropped set #{idx}")
            elif k in (ord('+'), ord('=')):
                MIN_CORNERS = min(16, MIN_CORNERS + 1)
                print(f"  MIN_CORNERS -> {MIN_CORNERS}")
            elif k == ord('-'):
                MIN_CORNERS = max(1, MIN_CORNERS - 1)
                print(f"  MIN_CORNERS -> {MIN_CORNERS}")
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