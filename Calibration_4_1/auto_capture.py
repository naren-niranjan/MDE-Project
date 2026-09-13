#!/usr/bin/env python3
"""Automatic ChArUco capture for the four-Triton rig.

Pixel path: RGB8 ONLY. The camera does the demosaic on-sensor-board and hands
over finished RGB; the host does no debayering, no white balance, no gamma, no
CLAHE, no resize, no denoise on the pixels that get written. What lands in the
PNG is byte-for-byte what came out of the camera, with exactly one exception:
the channel order is reversed at write time (RGB -> BGR) because cv2.imwrite
serialises arrays as BGR. That is an index reorder, not a transform -- skip it
and every saved frame has red and blue swapped. Detection works on a grayscale
COPY; the copy is never saved.

Two modes, matching the two downstream calibration scripts:

  intrinsics   Cameras free-run independently. A frame is saved for a camera
               whenever THAT camera individually passes the quality gate
               (corners, sharpness, stability, pose novelty). Output goes to
               config.CAPTURE_DIR/<name>/frame_XXXX.png. You can run all four
               at once and walk the board from camera to camera, or restrict
               with --cameras left.

  extrinsics   Cameras are PTP-synchronized and hardware-triggered with a
               single Scheduled Action Command (Action0), so the SAME frame
               index is the SAME instant on every camera -- the property
               calibrate_extrinsics.py depends on. A frame set is saved when
               the gate passes; by default that means the REFERENCE camera
               sees the board AND at least one other camera does too
               (--gate ref+1). A plain "any 2 cameras" gate is available
               (--gate any2), but note that a left+right frame without the
               reference is never used by calibrate_extrinsics.py, which
               calibrates each target against the reference only -- so ref+1
               is the default. Output goes to config.EXTRINSIC_CAPTURE_DIR/.
               Only the cameras that passed detection get the frame written,
               which keeps the shared-filename semantics honest.

               With four cameras -- three of them in a row plus one above --
               almost no board pose is visible to all four at once, and you
               should not try to find one. Shoot the pairs separately, each
               run still writing into the same directory:
                   --cameras center left
                   --cameras center right
                   --cameras center top
               The reference is forced into every extrinsics run.

Quality gate (both modes), tuned for the < 1.0 px (really < 0.5 px intrinsic)
RMS target:
  * >= --min-corners ChArUco corners (default 20; the offline scripts accept
    12, the stricter capture gate leaves margin)
  * Laplacian-variance sharpness over the board region >= --min-sharpness
    (rejects motion blur / defocus, the frames that later show up as the
    outliers dragging RMS from ~1.1 to ~1.6)
  * intrinsics only: corner positions stable vs the previous grab (< 2 px
    mean shift -- the board is being held still, not swept)
  * pose novelty vs already-accepted views (>= --novel-rot degrees rotation
    OR >= --novel-trans metres translation from every accepted pose), so 50
    saved frames are 50 DIFFERENT views with real tilt diversity instead of
    50 copies of the same fronto-parallel shot that feeds the focal/distance
    degeneracy

Detection uses config.get_charuco_detector() and, by default, the raw+CLAHE
two-strategy pass identical to the offline calibrators, so a frame the gate
accepts will not come back corner-starved offline. --no-clahe restricts
detection to the raw grayscale if you want the gate to see exactly what the
sensor sent; it costs recall on unevenly lit boards. Either way, CLAHE touches
a temporary grayscale copy only -- never the saved file.

Sync (extrinsics): PtpEnable on all cameras, wait for exactly one Master and
the rest Slave, put every camera in TriggerMode=On / TriggerSource=Action0,
then fire Scheduled Action Commands from the host at PTP time now+delta.
Per-frame sensor timestamps are checked and the set is rejected if the spread
exceeds --max-skew-ms (with Action0 it should be microseconds).

Bandwidth: RGB8 at 2448x2048 is ~15 MB per frame per camera. Four cameras
firing on the same Action0 put ~60 MB onto the link in one burst. If the
switch uplink to the Jetson is the 2.5G port rather than a 10G path, cap each
camera with --throughput-limit (e.g. 60000000 for ~60 MB/s each) so the burst
is paced instead of colliding -- dropped packets become resends, resends
become latency, and latency is what shows up later as timestamp skew.

Usage:
  python auto_capture.py --mode intrinsics
  python auto_capture.py --mode intrinsics --cameras left --target 50
  python auto_capture.py --mode extrinsics --cameras center top --target 40 \
      --action-ip 192.168.1.255
  python auto_capture.py --mode extrinsics --no-preview          # headless

Quit with 'q' in the preview window or Ctrl-C.
"""
import os

# GenICam's node cache must live somewhere user-writable on the Jetson or the
# Arena SDK import path fails with opaque cache errors. Set BEFORE arena_api.
os.environ.setdefault("GENICAM_CACHE_V3_1",
                      os.path.expanduser("~/.genicam_cache"))
os.makedirs(os.environ["GENICAM_CACHE_V3_1"], exist_ok=True)

import sys
import time
import json
import glob
import ctypes
import argparse

import cv2
import numpy as np

import config

try:
    from arena_api.system import system
except ImportError:
    sys.exit("arena_api not importable -- activate the environment with the "
             "Arena SDK Python wheel installed.")

# === Defaults (all overridable on the CLI) ==================================
MIN_CORNERS_CAPTURE = 20      # stricter than the offline scripts' 12
MIN_SHARPNESS       = 120.0   # Laplacian variance over the board bbox
NOVEL_ROT_DEG       = 6.0     # min rotation vs every accepted view
NOVEL_TRANS_M       = 0.08    # min translation vs every accepted view
STABLE_SHIFT_PX     = 2.0     # intrinsics: max mean corner shift vs prev grab
MAX_SKEW_MS         = 1.0     # extrinsics: max sensor-timestamp spread
ACTION_DELAY_S      = 0.25    # scheduled action fires this far in the future
ACTION_KEY          = 1       # device/group key/mask for Action0
PTP_TIMEOUT_S       = 45.0
GRAB_TIMEOUT_MS     = 4000

# Nominal K for the capture-time novelty pose only (solvePnP needs *some*
# intrinsics before calibration exists; 8 mm / 3.45 um => fx ~2319 px).
_EXPECTED_FX = 8.0 / 0.00345

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# === Device discovery / configuration =======================================

def open_devices(names):
    """Open the requested cameras BY SERIAL. Discovery is filtered by serial
    so a stray non-Lucid GigE device on the subnet can never be opened by
    accident."""
    wanted = {n: config.CAMERAS[n] for n in names}
    infos = system.device_infos                     # property, NOT a method
    by_serial = {i["serial"]: i for i in infos}
    missing = [f"{n} (SN {s})" for n, s in wanted.items()
               if s not in by_serial]
    if missing:
        found = ", ".join(sorted(by_serial)) or "none"
        raise RuntimeError(
            f"Cameras not found on the network: {', '.join(missing)}. "
            f"Discovered serials: {found}. Check PoE, the camera subnet on "
            f"mgbe3_0, and that no other process holds the cameras.")
    devs = {}
    for name, serial in wanted.items():
        devs[name] = system.create_device([by_serial[serial]])[0]  # list arg
        print(f"  opened {name:6s} SN {serial}")
    return devs


def configure_device(name, dev, throughput_limit=None):
    """Common node setup. Runs BEFORE start_stream: Width/Height on a locked
    stream throws SC_ERR_ACCESS_DENIED (-1005), and a camera left in
    TriggerMode=On by a crashed PTP session makes every later free-run grab
    time out with SC_ERR_TIMEOUT (-1011) -- so triggering is disabled first,
    unconditionally."""
    nm = dev.nodemap

    # 1. Kill leftover trigger state from any previous (possibly crashed) run.
    try:
        nm["TriggerSelector"].value = "FrameStart"
        nm["TriggerMode"].value = "Off"
    except Exception as e:
        print(f"  [{name}] warn: could not reset trigger state: {e}")

    nm["AcquisitionMode"].value = "Continuous"

    # 2. ROI before streaming.
    for node, val in (("Width", config.WIDTH), ("Height", config.HEIGHT)):
        try:
            if nm[node].value != val:
                nm[node].value = val
        except Exception as e:
            print(f"  [{name}] warn: {node} not set ({e}) -- if this is "
                  f"-1005, another session holds the stream; power-cycle or "
                  f"close it.")

    # 3. Pixel format: RGB8, no alternative. The camera demosaics on board and
    # sends finished colour, so nothing on the host reconstructs pixels. Costs
    # 3 bytes/px (~7 FPS at 5 MP on 1 GigE) versus BayerRG8's 1 byte/px, and
    # that trade is the whole point -- the frame written to disk is the
    # camera's own output, not a host reconstruction of it.
    nm["PixelFormat"].value = "RGB8"
    actual = nm["PixelFormat"].value
    if actual != "RGB8":
        raise RuntimeError(
            f"[{name}] PixelFormat is '{actual}', not RGB8. The camera "
            f"rejected the format -- check firmware and that no other client "
            f"is holding it.")

    # 4. Exposure/gain are handled by lock_exposure() after configuration:
    # either a fixed value from the CLI, or auto-converge-then-LOCK, so the
    # shoot still runs with pinned settings (non-negotiable for calibration)
    # but adapted to the actual hall lighting instead of a hardcoded 3 ms
    # that lands three stops under.

    # 5. Optional per-camera bandwidth cap. Four RGB8 5 MP streams is ~60 MB
    # per synchronized burst; if the uplink can't absorb that, pacing each
    # camera here is far better than letting the switch drop and resend.
    if throughput_limit:
        try:
            nm["DeviceLinkThroughputLimitMode"].value = "On"
            nm["DeviceLinkThroughputLimit"].value = int(throughput_limit)
            print(f"  [{name}] throughput capped at "
                  f"{int(throughput_limit)/1e6:.0f} MB/s")
        except Exception as e:
            print(f"  [{name}] warn: throughput limit not applied ({e})")
    else:
        try:
            nm["DeviceLinkThroughputLimitMode"].value = "Off"
        except Exception:
            pass

    # 6. Stream layer: negotiate jumbo packets, enable resend (four cameras
    # bursting simultaneously through the uplink WILL drop the odd packet;
    # resend makes that invisible instead of a torn frame).
    s = dev.tl_stream_nodemap
    s["StreamAutoNegotiatePacketSize"].value = True
    s["StreamPacketResendEnable"].value = True
    s["StreamBufferHandlingMode"].value = "NewestOnly"


def grab_rgb(dev, timeout_ms=GRAB_TIMEOUT_MS):
    """One RGB8 frame exactly as the camera sent it, plus its sensor timestamp
    (ns). The only operation performed is a copy out of the SDK buffer before
    it is requeued -- no colour conversion, no demosaic, no scaling. Anything
    other than 24 bpp means the format negotiation silently failed upstream,
    which is worth an exception rather than a quiet host-side reconstruction.
    """
    buf = dev.get_buffer(timeout=timeout_ms)
    try:
        h, w = buf.height, buf.width
        bpp = int(buf.bits_per_pixel)
        if bpp != 24:
            raise RuntimeError(
                f"expected RGB8 (24 bpp), camera delivered {bpp} bpp -- "
                f"PixelFormat was changed behind this script's back")
        ptr = ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_ubyte))
        rgb = np.ctypeslib.as_array(ptr, (h, w, 3)).copy()
        ts = int(buf.timestamp_ns)
    finally:
        dev.requeue_buffer(buf)
    return rgb, ts


def save_frame(path, rgb):
    """Write the camera's RGB frame to PNG unmodified.

    cv2.imwrite serialises an array as BGR, so the channel order is reversed
    on the way out. That reversal is a lossless index permutation -- no
    resampling, no colour-space maths, no bit-depth change -- and it is what
    makes the file on disk carry the camera's actual colours. PNG is lossless,
    so the decoded pixels come back identical to the sensor's output.
    """
    if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def lock_exposure(devs, args):
    """Pin exposure and gain for the whole shoot.

    --exposure <us>  : set that value on every camera (with --gain, default 0)
    --exposure auto  : run ExposureAuto/GainAuto=Continuous for --ae-settle
                       seconds while streaming, read back the converged
                       values, then switch auto OFF and write them back as
                       fixed settings. If auto wanted more exposure than
                       --max-exposure-us (motion-blur bound for a hand-held
                       board), the exposure is capped there and the shortfall
                       is moved into gain (dB = 20*log10(wanted/cap)).
                       The result is then clamped to --max-gain-db (18.7 dB
                       by default) and to the sensor's own gain range.

    The gain ceiling matters more than it looks. Auto exposure optimises for
    a well-exposed picture; calibration wants a LOW-NOISE picture, and those
    two diverge fast above ~20 dB. If auto asks for 34 dB, the honest reading
    is that the scene is several stops too dark for the shutter it is allowed
    -- capping the gain makes that visible instead of hiding it under noise.

    Either way, the shoot itself runs with auto OFF: exposure that drifts
    between views (or differs between cameras at the same instant on the
    extrinsics shoot) is exactly what pinned settings exist to prevent. This
    is an acquisition setting, not image processing -- the sensor integrates
    differently, nothing is done to the pixels afterwards.
    """
    if args.exposure != "auto":
        exp = float(args.exposure)
        for name, dev in devs.items():
            nm = dev.nodemap
            nm["ExposureAuto"].value = "Off"
            nm["ExposureTime"].value = exp
            nm["GainAuto"].value = "Off"
            nm["Gain"].value = float(args.gain)
            print(f"  [{name}] exposure locked: {exp:.0f} us, "
                  f"gain {float(args.gain):.1f} dB (fixed)")
        return

    # -- auto-converge phase --
    for name, dev in devs.items():
        nm = dev.nodemap
        # Bound what auto exposure may pick, so it can't converge to a value
        # that blurs a hand-held board. Node names per Lucid Triton firmware;
        # missing nodes are skipped harmlessly.
        try:
            nm["ExposureAutoLimitAuto"].value = "Off"
        except Exception:
            pass
        for node, val in (("ExposureAutoUpperLimit", float(args.max_exposure_us)),
                          ("ExposureAutoLowerLimit", 100.0)):
            try:
                nm[node].value = val
            except Exception:
                pass
        nm["ExposureAuto"].value = "Continuous"
        nm["GainAuto"].value = "Continuous"

    print(f"  auto-exposure converging for {args.ae_settle:.1f} s ...")
    for dev in devs.values():
        dev.start_stream()
    try:
        t0 = time.time()
        while time.time() - t0 < args.ae_settle:
            for dev in devs.values():
                try:
                    buf = dev.get_buffer(timeout=2000)
                    dev.requeue_buffer(buf)
                except Exception:
                    pass
    finally:
        for name, dev in devs.items():
            nm = dev.nodemap
            exp = float(nm["ExposureTime"].value)
            gain = float(nm["Gain"].value)
            nm["ExposureAuto"].value = "Off"
            nm["GainAuto"].value = "Off"

            note = ""
            if exp > args.max_exposure_us + 1.0:
                extra = 20.0 * np.log10(exp / args.max_exposure_us)
                exp = float(args.max_exposure_us)
                gain += extra
                note = f" (exposure capped, +{extra:.1f} dB moved to gain)"
            elif exp >= args.max_exposure_us - 1.0:
                # Auto sat exactly on the ceiling, so nothing was moved into
                # gain -- but auto had already pushed gain up on its own to
                # compensate, which is the number to look at.
                note = " (exposure at the --max-exposure-us ceiling)"

            # Hard gain ceiling. Read noise on the IMX264 grows with gain, and
            # noise is what destroys sub-pixel corner localisation -- a bright
            # grainy board yields worse corners than a slightly dark clean
            # one. Auto exposure has no idea it is feeding a calibration, so
            # it will happily converge at 34 dB; this is where that stops.
            if gain > args.max_gain_db:
                shortfall = gain - args.max_gain_db
                gain = float(args.max_gain_db)
                note += (f" (gain capped at {gain:.1f} dB, {shortfall:.1f} dB "
                         f"under what auto asked for -- frames WILL be dark; "
                         f"open the aperture, add light, or raise "
                         f"--max-gain-db)")
            try:
                gmin, gmax = float(nm["Gain"].min), float(nm["Gain"].max)
                if gain > gmax:
                    note += f" (sensor limit {gmax:.1f} dB)"
                    gain = gmax
                elif gain < gmin:
                    gain = gmin
            except Exception:
                pass

            nm["ExposureTime"].value = exp
            nm["Gain"].value = gain
            print(f"  [{name}] exposure locked: {exp:.0f} us, "
                  f"gain {gain:.1f} dB{note}")
        for dev in devs.values():
            dev.stop_stream()


def verify_levels(devs, flush=6):
    """Stream a few frames with the locked settings and report what they look
    like, so a too-dark or blown-out shoot is caught in the first two seconds
    instead of after 200 rejected frames. Purely diagnostic -- reads pixels,
    changes nothing."""
    print("  checking levels with the locked settings ...")
    for dev in devs.values():
        dev.start_stream()
    try:
        for name, dev in devs.items():
            mean = clip = None
            for _ in range(flush):          # discard until settings apply
                try:
                    rgb, _ = grab_rgb(dev, 3000)
                except Exception:
                    continue
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                mean = float(g.mean())
                clip = float((g >= 250).mean() * 100.0)
            if mean is None:
                print(f"  [{name}] level check: no frame returned")
                continue
            if mean < 25:
                verdict = "TOO DARK -- the board will not detect"
            elif mean < 50:
                verdict = "dark; expect corner-starved frames"
            elif clip > 2.0:
                verdict = "highlights clipping; drop exposure or gain"
            else:
                verdict = "ok"
            print(f"  [{name}] level check: mean {mean:.0f}/255, "
                  f"clipped {clip:.1f}%  -> {verdict}")
    finally:
        for dev in devs.values():
            try:
                dev.stop_stream()
            except Exception:
                pass


# === PTP + Scheduled Action Command sync (extrinsics) =======================

def enable_ptp(devs):
    """Enable PTP on every camera and block until the clocks have negotiated
    exactly one Master with the rest Slave."""
    for name, dev in devs.items():
        dev.nodemap["PtpEnable"].value = True
    print("  waiting for PTP negotiation (one Master, rest Slave) ...")
    states = {}
    t0 = time.time()
    while time.time() - t0 < PTP_TIMEOUT_S:
        states = {n: d.nodemap["PtpStatus"].value for n, d in devs.items()}
        vals = list(states.values())
        if vals.count("Master") == 1 and all(v in ("Master", "Slave")
                                             for v in vals):
            print("  PTP locked: " +
                  ", ".join(f"{n}={s}" for n, s in states.items()))
            return states
        time.sleep(1.0)
    raise RuntimeError(f"PTP did not converge within {PTP_TIMEOUT_S:.0f} s "
                       f"(last states: {states}). Every camera in the run "
                       f"must sit on the same L2 segment through the PoE "
                       f"switch.")


def arm_action_trigger(devs):
    """Put every camera into hardware-triggered mode listening for Action0."""
    for name, dev in devs.items():
        nm = dev.nodemap
        try:
            nm["ActionUnconditionalMode"].value = "On"
        except Exception:
            pass                                   # not present on all fw
        nm["ActionSelector"].value = 0
        nm["ActionDeviceKey"].value = ACTION_KEY
        nm["ActionGroupKey"].value = ACTION_KEY
        nm["ActionGroupMask"].value = ACTION_KEY
        nm["TriggerSelector"].value = "FrameStart"
        nm["TriggerSource"].value = "Action0"
        nm["TriggerMode"].value = "On"
        print(f"  [{name}] armed on Action0")


def _ip_to_int(ip):
    a, b, c, d = (int(x) for x in ip.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def fire_scheduled_action(any_dev, action_ip, delay_s=ACTION_DELAY_S):
    """Fire one Action0 at PTP time now+delay. The device Timestamp IS PTP
    time once PtpEnable is on, so latching it on any camera gives the shared
    clock."""
    nm = any_dev.nodemap
    nm["TimestampLatch"].execute()
    exec_ns = int(nm["Timestamp"].value) + int(delay_s * 1e9)

    tl = system.tl_system_nodemap
    tl["ActionCommandDeviceKey"].value = ACTION_KEY
    tl["ActionCommandGroupKey"].value = ACTION_KEY
    tl["ActionCommandGroupMask"].value = ACTION_KEY
    try:
        tl["ActionCommandTargetIP"].value = _ip_to_int(action_ip)
    except Exception:
        pass                    # older TL builds: default broadcast is used
    tl["ActionCommandExecuteTime"].value = exec_ns
    tl["ActionCommandFireCommand"].execute()


def disarm(devs):
    for name, dev in devs.items():
        nm = dev.nodemap
        try:
            nm["TriggerSelector"].value = "FrameStart"
            nm["TriggerMode"].value = "Off"
        except Exception:
            pass
        try:
            nm["PtpEnable"].value = False
        except Exception:
            pass


# === Detection / quality gate ===============================================

def evaluate(rgb, detector, min_corners, min_sharp, use_clahe=True):
    """Detect the board and score the frame.

    Works on a grayscale COPY of the camera frame. Nothing here writes back
    into `rgb`, so the frame that gets saved is untouched regardless of what
    detection had to do to find the corners.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    strategies = [gray] + ([_CLAHE.apply(gray)] if use_clahe else [])
    best = (None, None, 0)
    for im in strategies:
        cc, ids, _, _ = detector.detectBoard(im)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            best = (cc, ids, n)
    cc, ids, n = best

    out = {"ok": False, "n_corners": n, "sharpness": 0.0,
           "corners": None, "ids": None, "why": ""}
    if n < min_corners:
        mean = float(gray.mean())
        out["why"] = f"{n} corners < {min_corners}"
        if mean < 20.0:
            out["why"] += (f", image nearly black (mean {mean:.0f}/255) -- "
                           f"exposure/gain problem, not a board problem")
        return out

    pts = cc.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(0)).astype(int) - 10
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 10
    roi = gray[max(0, y0):min(gray.shape[0], y1),
               max(0, x0):min(gray.shape[1], x1)]
    sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    out["sharpness"] = sharp
    if sharp < min_sharp:
        out["why"] = f"sharpness {sharp:.0f} < {min_sharp:.0f} (blur)"
        return out

    out.update(ok=True, corners=cc, ids=ids, why="ok")
    return out


def board_pose(ev, board, image_size):
    """Rough board pose with nominal intrinsics, for the novelty gate only.
    IPPE (planar solver) with ITERATIVE fallback; accuracy is irrelevant here,
    only pose-to-pose DIFFERENCES matter."""
    w, h = image_size
    K = np.array([[_EXPECTED_FX, 0, w / 2],
                  [0, _EXPECTED_FX, h / 2],
                  [0, 0, 1]], dtype=np.float64)
    chess = board.getChessboardCorners()
    obj = np.array([chess[int(i)] for i in ev["ids"].ravel()], np.float32)
    pts = ev["corners"].reshape(-1, 2).astype(np.float32)
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, pts, K, None, flags=flag)
        except cv2.error:
            continue
        if ok:
            return rvec.ravel(), tvec.ravel()
    return None, None


def rot_angle_deg(r1, r2):
    R1, _ = cv2.Rodrigues(np.asarray(r1, np.float64))
    R2, _ = cv2.Rodrigues(np.asarray(r2, np.float64))
    return float(np.degrees(np.arccos(
        np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def is_novel(rvec, tvec, accepted, min_rot, min_trans):
    """True if this pose differs from EVERY accepted pose by at least min_rot
    degrees OR min_trans metres. Forces view diversity instead of 50 copies of
    the same shot."""
    if rvec is None:
        return True                       # can't judge -> don't block capture
    for r0, t0 in accepted:
        if (rot_angle_deg(rvec, r0) < min_rot and
                np.linalg.norm(np.asarray(tvec) - np.asarray(t0)) < min_trans):
            return False
    return True


# === Bookkeeping ============================================================

def next_index(root, names):
    """Resume numbering after the highest existing frame index across all
    per-camera folders, so re-runs append instead of overwrite."""
    hi = -1
    for n in names:
        for f in glob.glob(os.path.join(root, n, "frame_*.png")):
            try:
                hi = max(hi, int(os.path.basename(f)[6:10]))
            except ValueError:
                pass
    return hi + 1


def next_index_all(root):
    """Highest index across EVERY camera folder under root, not just the ones
    in this run. With four cameras shot as separate pairs, the extrinsics
    directory accumulates across runs and indices must never be reused -- a
    center/left frame_0007 and a center/top frame_0007 would silently claim to
    be the same instant."""
    hi = -1
    for f in glob.glob(os.path.join(root, "*", "frame_*.png")):
        try:
            hi = max(hi, int(os.path.basename(f)[6:10]))
        except ValueError:
            pass
    return hi + 1


def append_log(root, entry):
    path = os.path.join(root, "capture_log.json")
    log = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def make_preview(frames, evals, status, tile_w=460, cols=2):
    """Preview mosaic, laid out as a grid so four 5 MP views stay on screen.

    Everything here operates on resized COPIES for display only.
    """
    tiles = []
    for name in frames:
        rgb = frames[name]
        ev = evals.get(name)
        h, w = rgb.shape[:2]
        s = tile_w / w
        tile = cv2.cvtColor(
            cv2.resize(rgb, (tile_w, int(h * s))), cv2.COLOR_RGB2BGR)
        if ev and ev["corners"] is not None:
            pts = (ev["corners"].reshape(-1, 2) * s).astype(int)
            for p in pts:
                cv2.circle(tile, tuple(p), 3,
                           (0, 255, 0) if ev["ok"] else (0, 165, 255), -1)
        colour = (0, 255, 0) if (ev and ev["ok"]) else (0, 0, 255)
        txt = (f"{name}: {ev['n_corners']} c, "
               f"sh {ev['sharpness']:.0f}" if ev else f"{name}: -")
        cv2.putText(tile, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    colour, 2)
        tiles.append(tile)

    if not tiles:
        return np.zeros((36, 320, 3), np.uint8)

    th, tw = tiles[0].shape[:2]
    ncols = 1 if len(tiles) == 1 else min(cols, len(tiles))
    while len(tiles) % ncols:                       # pad the last row
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [cv2.hconcat(tiles[i:i + ncols])
            for i in range(0, len(tiles), ncols)]
    mosaic = cv2.vconcat(rows) if len(rows) > 1 else rows[0]

    bar = np.zeros((36, mosaic.shape[1], 3), np.uint8)
    cv2.putText(bar, status, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return cv2.vconcat([bar, mosaic])


# === Capture loops ==========================================================

def run_intrinsics(devs, args, detector, board):
    """Free-running per-camera gated capture into config.CAPTURE_DIR."""
    root = config.CAPTURE_DIR
    for n in devs:
        os.makedirs(os.path.join(root, n), exist_ok=True)

    counts = {n: len(glob.glob(os.path.join(root, n, "frame_*.png")))
              for n in devs}
    idx = {n: next_index(root, [n]) for n in devs}
    accepted = {n: [] for n in devs}          # (rvec, tvec) of saved views
    prev_pts = {n: None for n in devs}
    last_save = {n: 0.0 for n in devs}

    for dev in devs.values():
        dev.start_stream()
    print(f"\nIntrinsics capture -> {root}/  "
          f"target {args.target} per camera "
          f"(existing: {', '.join(f'{n}={c}' for n, c in counts.items())})")
    print("Work each camera with real +/-30-45 deg tilt about BOTH axes, at "
          "three distances, out to the frame edges.\n")

    try:
        while any(c < args.target for c in counts.values()):
            frames, evals = {}, {}
            now = time.time()
            for name, dev in devs.items():
                try:
                    frames[name], _ = grab_rgb(dev)
                except Exception as e:
                    print(f"  [{name}] grab failed: {e}")
                    continue
                if counts[name] >= args.target:
                    evals[name] = {"ok": False, "n_corners": 0,
                                   "sharpness": 0, "corners": None,
                                   "ids": None, "why": "target reached"}
                    continue
                ev = evaluate(frames[name], detector,
                              args.min_corners, args.min_sharpness,
                              args.clahe)
                evals[name] = ev
                if not ev["ok"]:
                    prev_pts[name] = None
                    continue

                pts = ev["corners"].reshape(-1, 2)

                # stability: board held still vs the previous grab
                if prev_pts[name] is not None and \
                        len(prev_pts[name]) == len(pts):
                    shift = float(np.mean(
                        np.linalg.norm(pts - prev_pts[name], axis=1)))
                    if shift > STABLE_SHIFT_PX:
                        prev_pts[name] = pts
                        ev["why"] = f"moving ({shift:.1f} px)"
                        ev["ok"] = False
                        continue
                prev_pts[name] = pts

                if now - last_save[name] < args.cooldown:
                    ev["why"] = "cooldown"
                    ev["ok"] = False
                    continue

                rvec, tvec = board_pose(ev, board,
                                        (config.WIDTH, config.HEIGHT))
                if not is_novel(rvec, tvec, accepted[name],
                                args.novel_rot, args.novel_trans):
                    ev["why"] = "not novel -- move/tilt the board"
                    ev["ok"] = False
                    continue

                fname = f"frame_{idx[name]:04d}.png"
                save_frame(os.path.join(root, name, fname), frames[name])
                accepted[name].append((rvec, tvec))
                counts[name] += 1
                idx[name] += 1
                last_save[name] = now
                append_log(root, {
                    "mode": "intrinsics", "camera": name, "file": fname,
                    "n_corners": ev["n_corners"],
                    "sharpness": round(ev["sharpness"], 1),
                    "wall_time": now})
                print(f"  [{name}] saved {fname}  "
                      f"({counts[name]}/{args.target}, "
                      f"{ev['n_corners']} corners, "
                      f"sharp {ev['sharpness']:.0f})")

            if args.preview and frames:
                status = "  ".join(f"{n} {counts[n]}/{args.target}"
                                   for n in devs)
                cv2.imshow("auto_capture (q quits)",
                           make_preview(frames, evals, status))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        for dev in devs.values():
            try:
                dev.stop_stream()
            except Exception:
                pass
    print("\nIntrinsics capture done: " +
          ", ".join(f"{n}={c}" for n, c in counts.items()))


def run_extrinsics(devs, args, detector, board):
    """PTP + Action0 synchronized capture into config.EXTRINSIC_CAPTURE_DIR."""
    root = config.EXTRINSIC_CAPTURE_DIR
    ref = config.REFERENCE_CAMERA
    if ref not in devs:
        raise RuntimeError(f"Reference camera '{ref}' must be included in "
                           f"an extrinsics run.")
    for n in devs:
        os.makedirs(os.path.join(root, n), exist_ok=True)

    enable_ptp(devs)
    arm_action_trigger(devs)
    for dev in devs.values():
        dev.start_stream()

    # Index continues past every camera folder in root, including ones not in
    # this run, so pair-by-pair shoots never collide on a filename.
    idx = next_index_all(root)
    saved = 0
    accepted = []                                # reference-camera poses
    last_save = 0.0
    grab_timeout = int(ACTION_DELAY_S * 1000) + GRAB_TIMEOUT_MS

    targets = [n for n in devs if n != ref]
    print(f"\nExtrinsics capture -> {root}/  target {args.target} frame sets "
          f"(resuming at index {idx})")
    print(f"Cameras: {ref} (reference) + {', '.join(targets)}")
    print(f"Gate: "
          f"{'reference + >=1 target' if args.gate == 'ref+1' else 'any 2 cameras'}"
          f", timestamp skew < {args.max_skew_ms} ms")
    print("Hold the board in the overlap regions with real tilt; cover the "
          f"{ref}<->{' and '.join(targets)} overlap"
          f"{'s' if len(targets) > 1 else ''} across the shoot.\n")

    try:
        while saved < args.target:
            fire_scheduled_action(devs[ref], args.action_ip)
            frames, stamps, evals = {}, {}, {}
            for name, dev in devs.items():
                try:
                    frames[name], stamps[name] = grab_rgb(dev, grab_timeout)
                except Exception as e:
                    print(f"  [{name}] no frame after action fire: {e} "
                          f"(if -1011 persists: camera missed the action -- "
                          f"check ActionCommandTargetIP reaches the camera "
                          f"subnet, e.g. --action-ip 192.168.1.255)")
            if len(frames) < len(devs):
                time.sleep(args.period)
                continue

            skew_ms = (max(stamps.values()) - min(stamps.values())) / 1e6
            if skew_ms > args.max_skew_ms:
                print(f"  frame set rejected: timestamp skew {skew_ms:.3f} ms "
                      f"> {args.max_skew_ms} ms (PTP drift?)")
                time.sleep(args.period)
                continue

            for name in frames:
                evals[name] = evaluate(frames[name], detector,
                                       args.min_corners, args.min_sharpness,
                                       args.clahe)
            ok = [n for n in evals if evals[n]["ok"]]

            if args.gate == "ref+1":
                gate_pass = (ref in ok) and len(ok) >= 2
            elif args.gate == "ref+all":
                gate_pass = len(ok) == len(devs)
            else:                                              # any2
                gate_pass = len(ok) >= 2

            why = ""
            if gate_pass and time.time() - last_save < args.cooldown:
                gate_pass, why = False, "cooldown"

            if gate_pass and ref in ok:
                rvec, tvec = board_pose(evals[ref], board,
                                        (config.WIDTH, config.HEIGHT))
                if not is_novel(rvec, tvec, accepted,
                                args.novel_rot, args.novel_trans):
                    gate_pass, why = False, "not novel -- move/tilt the board"
                elif rvec is not None:
                    pending_pose = (rvec, tvec)
                else:
                    pending_pose = None
            else:
                pending_pose = None

            if gate_pass:
                fname = f"frame_{idx:04d}.png"
                for name in ok:                 # only cameras that saw it
                    save_frame(os.path.join(root, name, fname), frames[name])
                if pending_pose is not None:
                    accepted.append(pending_pose)
                append_log(root, {
                    "mode": "extrinsics", "file": fname, "cameras": ok,
                    "reference": ref,
                    "skew_ms": round(skew_ms, 4),
                    "corners": {n: evals[n]["n_corners"] for n in ok},
                    "sharpness": {n: round(evals[n]["sharpness"], 1)
                                  for n in ok},
                    "timestamps_ns": {n: stamps[n] for n in ok},
                    "wall_time": time.time()})
                idx += 1
                saved += 1
                last_save = time.time()
                print(f"  saved {fname} on [{', '.join(ok)}]  "
                      f"({saved}/{args.target}, skew {skew_ms*1000:.0f} us)")
            else:
                detail = why or " | ".join(
                    f"{n}:{evals[n]['why']}" for n in evals
                    if not evals[n]["ok"])
                print(f"  gate miss ({len(ok)} ok: {', '.join(ok) or '-'}) "
                      f"{detail}")

            if args.preview:
                status = (f"saved {saved}/{args.target}  "
                          f"skew {skew_ms*1000:.0f} us  gate={args.gate}")
                cv2.imshow("auto_capture (q quits)",
                           make_preview(frames, evals, status))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(args.period)
    finally:
        for dev in devs.values():
            try:
                dev.stop_stream()
            except Exception:
                pass
        disarm(devs)
    print(f"\nExtrinsics capture done: {saved} frame sets in {root}/")


# === Entry point ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Gated auto-capture for ChArUco calibration shoots "
                    "(RGB8 only, frames saved exactly as the camera sends "
                    "them).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("intrinsics", "extrinsics"),
                   required=True)
    p.add_argument("--cameras", nargs="+", default=list(config.CAMERAS),
                   choices=list(config.CAMERAS),
                   help="subset of cameras; in extrinsics mode the reference "
                        "is added automatically if you leave it out, so "
                        "'--cameras top' shoots the reference<->top pair")
    p.add_argument("--target", type=int, default=None,
                   help="frames per camera (intrinsics, default 50) or frame "
                        "sets (extrinsics, default 40)")
    p.add_argument("--gate", choices=("ref+1", "ref+all", "any2"),
                   default="ref+1",
                   help="extrinsics save condition; ref+1 = reference sees "
                        "the board plus at least one other camera, ref+all = "
                        "every camera in the run sees it (rarely satisfiable "
                        "with the full four-camera set)")
    p.add_argument("--exposure", default="auto",
                   help="'auto' = converge auto-exposure once at startup then "
                        "lock, or a fixed value in microseconds (e.g. 12000)")
    p.add_argument("--gain", type=float, default=18.7,
                   help="dB; used only with a fixed --exposure value")
    p.add_argument("--max-gain-db", type=float, default=18.7,
                   help="ceiling on the gain auto-exposure is allowed to "
                        "converge to; above this, sensor noise costs more "
                        "corner accuracy than the extra brightness buys")
    p.add_argument("--max-exposure-us", type=float, default=20000.0,
                   help="cap for auto exposure (motion-blur bound); any "
                        "shortfall is compensated with gain")
    p.add_argument("--ae-settle", type=float, default=3.0,
                   help="s of streaming for auto-exposure to converge")
    p.add_argument("--throughput-limit", type=int, default=0,
                   help="per-camera DeviceLinkThroughputLimit in bytes/s "
                        "(0 = uncapped); use e.g. 60000000 when four RGB8 "
                        "streams share a 2.5G uplink")
    p.add_argument("--no-clahe", dest="clahe", action="store_false",
                   help="detect on the raw grayscale only, skipping the CLAHE "
                        "fallback (CLAHE never touches saved pixels either "
                        "way; this only changes what the gate can find)")
    p.add_argument("--min-corners", type=int, default=MIN_CORNERS_CAPTURE)
    p.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS)
    p.add_argument("--novel-rot", type=float, default=NOVEL_ROT_DEG,
                   help="deg; min rotation vs every accepted view")
    p.add_argument("--novel-trans", type=float, default=NOVEL_TRANS_M,
                   help="m; min translation vs every accepted view")
    p.add_argument("--cooldown", type=float, default=1.5,
                   help="s between saves, so the board can be repositioned")
    p.add_argument("--period", type=float, default=0.8,
                   help="s between action fires (extrinsics)")
    p.add_argument("--max-skew-ms", type=float, default=MAX_SKEW_MS)
    p.add_argument("--action-ip", default="255.255.255.255",
                   help="ActionCommand target IP; use the camera subnet "
                        "broadcast (e.g. 192.168.1.255) on the multi-homed "
                        "Jetson so the broadcast leaves via mgbe3_0")
    p.add_argument("--no-preview", dest="preview", action="store_false")
    args = p.parse_args()
    if args.target is None:
        args.target = 50 if args.mode == "intrinsics" else 40

    names = list(dict.fromkeys(args.cameras))
    if args.mode == "extrinsics":
        ref = config.REFERENCE_CAMERA
        if ref not in names:
            names.insert(0, ref)
        if len(names) < 2:
            sys.exit("Extrinsics needs the reference plus at least one "
                     "target camera.")
        # Keep config order so the preview grid matches the physical layout.
        names = [n for n in config.CAMERAS if n in names]

    print("Opening cameras ...")
    devs = open_devices(names)
    try:
        for name, dev in devs.items():
            configure_device(name, dev, args.throughput_limit)
        lock_exposure(devs, args)
        verify_levels(devs)

        detector = config.get_charuco_detector()
        board = config.get_charuco_board()

        if args.mode == "intrinsics":
            run_intrinsics(devs, args, detector, board)
        else:
            run_extrinsics(devs, args, detector, board)
    except KeyboardInterrupt:
        print("\nInterrupted -- cleaning up.")
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        try:
            system.destroy_device()
        except Exception:
            pass


if __name__ == "__main__":
    main()