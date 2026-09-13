"""
Live aperture / exposure tuner for one Lucid camera.

Streams a single camera at a FIXED exposure and gain and shows a live preview
with a brightness + clipping readout, so you can turn the physical iris ring
until the image is correctly exposed.

Why this workflow: if auto exposure maxes out (very long exposure + very high
gain), the sensor is light-starved -- the aperture is stopped down too far.
Open the iris while watching the readout until:
  * mean sits around 100-150 (mid-grey),
  * %clipped (near-white) stays low (a few % at most),
  * %black is small.
Then you can drop exposure and gain and still have a clean image.

Keys (focus the preview window):
    q        quit
    w / s    exposure  +/- 20%
    e / d    gain      +/- 2 dB
    r        reset to the values passed on the CLI

Usage:
    python aperture_tuner.py --name center
    python aperture_tuner.py --cam center=261100631 --name center
    python aperture_tuner.py --name left --flip --exposure-us 8000 --gain-db 6
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from arena_api.system import system

try:
    import config
except ImportError:
    config = None


def _set(nm, key, value):
    try:
        nm[key].value = value
        return True
    except Exception:
        return False


def _get(nm, key):
    try:
        return nm[key].value
    except Exception:
        return None


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


def clamp(nm, key, val):
    try:
        node = nm[key]
        lo, hi = getattr(node, 'min', None), getattr(node, 'max', None)
        if lo is not None:
            val = max(val, float(lo))
        if hi is not None:
            val = min(val, float(hi))
    except Exception:
        pass
    return val


def apply_exposure(nm, exposure_us, gain_db):
    _set(nm, 'ExposureAuto', 'Off')
    _set(nm, 'GainAuto', 'Off')
    exposure_us = clamp(nm, 'ExposureTime', exposure_us)
    gain_db = clamp(nm, 'Gain', gain_db)
    _set(nm, 'ExposureTime', exposure_us)
    _set(nm, 'Gain', gain_db)
    return exposure_us, gain_db


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
    ap.add_argument('--exposure-us', type=float, default=8000.0)
    ap.add_argument('--gain-db', type=float, default=0.0)
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

    exposure_us, gain_db = args.exposure_us, args.gain_db

    try:
        snm['StreamBufferHandlingMode'].value = 'NewestOnly'
        snm['StreamAutoNegotiatePacketSize'].value = True
        snm['StreamPacketResendEnable'].value = True
        _disable_triggering(nm)
        _set(nm, 'PixelFormat', 'RGB8')
        _set(nm, 'AcquisitionMode', 'Continuous')
        _set(nm, 'BalanceWhiteAuto', 'Continuous')
        exposure_us, gain_db = apply_exposure(nm, exposure_us, gain_db)

        dev.start_stream()
        win = f"tuner: {args.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("Preview running. Focus the window; keys: q w s e d r")

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

            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            clip_hi = float((gray >= 250).mean() * 100.0)
            clip_lo = float((gray <= 5).mean() * 100.0)

            scale = args.display_width / w
            disp = cv2.resize(arr, (args.display_width, int(h * scale)))

            lines = [
                f"exp {exposure_us:7.0f} us   gain {gain_db:5.1f} dB",
                f"mean {mean:6.1f}   clip>={250}: {clip_hi:4.1f}%   black: {clip_lo:4.1f}%",
            ]
            # warn colour: red text if clipping is heavy
            colour = (0, 0, 255) if clip_hi > 5 or mean > 200 else (0, 255, 0)
            for i, ln in enumerate(lines):
                y = 26 + i * 26
                cv2.putText(disp, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 1, cv2.LINE_AA)

            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('w'):
                exposure_us, gain_db = apply_exposure(nm, exposure_us * 1.2, gain_db)
            elif k == ord('s'):
                exposure_us, gain_db = apply_exposure(nm, exposure_us / 1.2, gain_db)
            elif k == ord('e'):
                exposure_us, gain_db = apply_exposure(nm, exposure_us, gain_db + 2)
            elif k == ord('d'):
                exposure_us, gain_db = apply_exposure(nm, exposure_us, gain_db - 2)
            elif k == ord('r'):
                exposure_us, gain_db = apply_exposure(nm, args.exposure_us, args.gain_db)

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