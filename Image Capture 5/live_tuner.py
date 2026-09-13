"""
Live exposure / gain tuner for one Lucid camera, with real-time sliders.

Streams a single camera and gives you two trackbars in the preview window:

    exposure   (log-scaled slider, microseconds)
    gain       (linear slider, 0.1 dB steps)

Drag either and the camera reconfigures live. A brightness + clipping readout is
overlaid so you can see the effect immediately while you also turn the physical
iris ring. Target: mean around 100-150, low %clipped, low %black, at the
SHORTEST exposure and LOWEST gain that still looks good.

Keys (focus the preview window):
    q    quit and print the final values to pin

Usage:
    python live_tuner.py --name center
    python live_tuner.py --cam center=261100631 --name center
    python live_tuner.py --name left --flip --exposure-us 8000 --gain-db 6
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from arena_api.system import system

try:
    import config
except ImportError:
    config = None


EXP_SLIDER_MAX = 1000          # resolution of the log exposure slider
GAIN_STEP_DB   = 0.1           # gain slider granularity


def _set(nm, key, value):
    try:
        nm[key].value = value
        return True
    except Exception:
        return False


def _node_range(nm, key, fallback_lo, fallback_hi):
    try:
        node = nm[key]
        lo = getattr(node, 'min', None)
        hi = getattr(node, 'max', None)
        lo = float(lo) if lo is not None else fallback_lo
        hi = float(hi) if hi is not None else fallback_hi
        return lo, hi
    except Exception:
        return fallback_lo, fallback_hi


def discover_one(serial, tries=6, delay=1.0):
    infos = None
    for _ in range(tries):
        infos = system.device_infos
        if infos:
            break
        time.sleep(delay)
    if not infos:
        raise RuntimeError("No Lucid devices on the network.")
    by_serial = {d['serial']: d for d in infos}
    if serial not in by_serial:
        raise RuntimeError(f"serial {serial} not found. Seen: {', '.join(by_serial)}")
    return system.create_device([by_serial[serial]])[0]


def _disable_triggering(nm):
    for sel in ('FrameStart', 'AcquisitionStart', 'FrameBurstStart',
                'FrameActive', 'ExposureStart'):
        try:
            nm['TriggerSelector'].value = sel
            nm['TriggerMode'].value = 'Off'
        except Exception:
            pass
    _set(nm, 'TriggerMode', 'Off')
    _set(nm, 'TriggerSource', 'Software')


def apply_exposure(nm, exposure_us, gain_db, exp_lo, exp_hi, gain_lo, gain_hi):
    exposure_us = min(max(exposure_us, exp_lo), exp_hi)
    gain_db = min(max(gain_db, gain_lo), gain_hi)
    _set(nm, 'ExposureAuto', 'Off')
    _set(nm, 'GainAuto', 'Off')
    _set(nm, 'ExposureTime', exposure_us)
    _set(nm, 'Gain', gain_db)
    return exposure_us, gain_db


# --- log mapping between slider position and exposure ----------------------

def pos_to_exp(pos, exp_lo, exp_hi):
    frac = pos / EXP_SLIDER_MAX
    return exp_lo * (exp_hi / exp_lo) ** frac


def exp_to_pos(exp, exp_lo, exp_hi):
    exp = min(max(exp, exp_lo), exp_hi)
    frac = math.log(exp / exp_lo) / math.log(exp_hi / exp_lo)
    return int(round(frac * EXP_SLIDER_MAX))


def resolve_serial(args):
    if args.cam:
        for item in args.cam:
            name, serial = item.split('=', 1)
            if name.strip() == args.name:
                return serial.strip()
    if config is not None and getattr(config, 'CAMERAS', None):
        cams = dict(config.CAMERAS)
        if args.name in cams:
            return cams[args.name]
    raise SystemExit(f"Could not resolve serial for camera {args.name!r}. "
                     f"Pass --cam {args.name}=SERIAL or add it to config.CAMERAS.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='center',
                    help="Which camera to tune (key in config.CAMERAS or --cam).")
    ap.add_argument('--cam', action='append', default=None, metavar='NAME=SERIAL')
    ap.add_argument('--exposure-us', type=float, default=8000.0,
                    help="Starting exposure (slider initialises here).")
    ap.add_argument('--gain-db', type=float, default=0.0,
                    help="Starting gain (slider initialises here).")
    ap.add_argument('--exp-max', type=float, default=50000.0,
                    help="Upper end of the exposure slider in us (default 50000).")
    ap.add_argument('--flip', action='store_true',
                    help="Rotate 180 (use for the left camera's mounted roll).")
    ap.add_argument('--display-width', type=int, default=960,
                    help="Downscale the preview to this width.")
    return ap.parse_args()


def main():
    args = parse_args()
    serial = resolve_serial(args)
    print(f"Opening {args.name} ({serial}) ...")
    dev = discover_one(serial)
    nm = dev.nodemap
    snm = dev.tl_stream_nodemap

    try:
        snm['StreamBufferHandlingMode'].value = 'NewestOnly'
        snm['StreamAutoNegotiatePacketSize'].value = True
        snm['StreamPacketResendEnable'].value = True
        _disable_triggering(nm)
        _set(nm, 'PixelFormat', 'RGB8')
        _set(nm, 'AcquisitionMode', 'Continuous')
        _set(nm, 'BalanceWhiteAuto', 'Continuous')
        _set(nm, 'ExposureAuto', 'Off')
        _set(nm, 'GainAuto', 'Off')

        # Slider bounds come from the actual node limits (clamped to sane defaults).
        exp_lo, exp_hi_node = _node_range(nm, 'ExposureTime', 20.0, 1_000_000.0)
        exp_lo = max(exp_lo, 20.0)
        exp_hi = min(exp_hi_node, args.exp_max)
        if exp_hi <= exp_lo:
            exp_hi = exp_lo * 100.0
        gain_lo, gain_hi = _node_range(nm, 'Gain', 0.0, 48.0)

        exposure_us, gain_db = apply_exposure(
            nm, args.exposure_us, args.gain_db, exp_lo, exp_hi, gain_lo, gain_hi)

        dev.start_stream()

        win = f"tuner: {args.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        gain_slider_max = int(round((gain_hi - gain_lo) / GAIN_STEP_DB))

        # Trackbars poll-driven: callback is a no-op, we read positions each frame.
        cv2.createTrackbar('exposure', win,
                           exp_to_pos(exposure_us, exp_lo, exp_hi),
                           EXP_SLIDER_MAX, lambda v: None)
        cv2.createTrackbar('gain x0.1dB', win,
                           int(round((gain_db - gain_lo) / GAIN_STEP_DB)),
                           gain_slider_max, lambda v: None)

        print(f"exposure slider range: {exp_lo:.0f}..{exp_hi:.0f} us (log)")
        print(f"gain slider range:     {gain_lo:.1f}..{gain_hi:.1f} dB")
        print("Drag sliders to adjust live. Press q in the window to quit.")

        while True:
            try:
                buf = dev.get_buffer(timeout=3000)
            except Exception:
                continue
            h, w = buf.height, buf.width
            arr = np.ctypeslib.as_array(buf.pdata, shape=(h, w, 3)).astype(np.uint8, copy=True)
            arr = arr[:, :, ::-1].copy()          # RGB -> BGR
            dev.requeue_buffer(buf)

            if args.flip:
                arr = cv2.rotate(arr, cv2.ROTATE_180)

            # Read sliders and apply only when they actually moved.
            epos = cv2.getTrackbarPos('exposure', win)
            gpos = cv2.getTrackbarPos('gain x0.1dB', win)
            target_exp = pos_to_exp(epos, exp_lo, exp_hi)
            target_gain = gain_lo + gpos * GAIN_STEP_DB
            if abs(target_exp - exposure_us) > 0.5 or abs(target_gain - gain_db) > 0.05:
                exposure_us, gain_db = apply_exposure(
                    nm, target_exp, target_gain, exp_lo, exp_hi, gain_lo, gain_hi)

            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            clip_hi = float((gray >= 250).mean() * 100.0)
            clip_lo = float((gray <= 5).mean() * 100.0)

            scale = args.display_width / w
            disp = cv2.resize(arr, (args.display_width, int(h * scale)))

            lines = [
                f"exp {exposure_us:7.0f} us   gain {gain_db:5.1f} dB",
                f"mean {mean:6.1f}   clip: {clip_hi:4.1f}%   black: {clip_lo:4.1f}%",
            ]
            colour = (0, 0, 255) if clip_hi > 5 or mean > 200 or mean < 40 else (0, 220, 0)
            for i, ln in enumerate(lines):
                y = 26 + i * 26
                cv2.putText(disp, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 1, cv2.LINE_AA)

            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        print(f"\nFinal: exposure {exposure_us:.0f} us, gain {gain_db:.1f} dB")
        print("Pin it with:")
        print(f"  python capture_sync.py --exposure-us {exposure_us:.0f} "
              f"--gain-db {gain_db:.1f}")

    finally:
        try:
            dev.stop_stream()
        except Exception:
            pass
        cv2.destroyAllWindows()
        system.destroy_device()


if __name__ == "__main__":
    main()