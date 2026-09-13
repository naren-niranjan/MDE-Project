"""
Simultaneous capture from three Lucid TRI050S-CC GigE cameras.

Free-running cameras can't be exposed at exactly the same instant in software
(that needs PTP + Action Commands, or a hardware trigger line). This script
does the closest software equivalent:

  * All three streams are started ahead of time with the buffer mode set to
    'NewestOnly', so get_buffer() always returns the most recently arrived
    frame.
  * One thread per camera is parked at a Barrier and then released together.
    The inter-camera grab calls fire within microseconds of each other.
  * Each device's hardware timestamp (timestamp_ns) is recorded so the actual
    inter-camera skew is visible in the meta.json and on stdout.

Usage
-----
    python capture_sync.py                     # capture one set and exit
    python capture_sync.py --loop              # Enter = grab, q+Enter = quit
    python capture_sync.py --outdir my_caps    # custom output directory
    python capture_sync.py --fps 60            # request a frame rate

Each capture creates one timestamped subdirectory containing center.png,
left.png, right.png, and a meta.json with per-camera device timestamps.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from arena_api.system import system

try:
    import config                              # CAMERAS = {'center': serial, ...}
except ImportError:
    config = None


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def discover_devices(name_to_serial: dict, tries: int = 6, delay_s: float = 1.0):
    infos = None
    for _ in range(tries):
        infos = system.device_infos               # property, not a method
        if infos:
            break
        time.sleep(delay_s)
    if not infos:
        raise RuntimeError("No Lucid devices on the network.")

    by_serial = {d['serial']: d for d in infos}
    devs = {}
    for name, serial in name_to_serial.items():
        if serial not in by_serial:
            available = ", ".join(by_serial) or "<none>"
            raise RuntimeError(
                f"{name!r} (serial {serial}) not found. Seen: {available}")
        # create_device wants a *list* of device_infos -- pass [info] for safety
        # across SDK versions, then take the single result back out.
        devs[name] = system.create_device([by_serial[serial]])[0]
        print(f"  bound {name:6s} <- {serial}")
    return devs


# ---------------------------------------------------------------------------
# Per-camera configuration
# ---------------------------------------------------------------------------

def _set_if_possible(nm, key, value):
    try:
        nm[key].value = value
    except Exception:
        pass


def configure(dev, fps: float | None = None):
    """Configure stream + pixel format. Returns the pixel format actually set."""
    nm  = dev.nodemap
    snm = dev.tl_stream_nodemap

    # Stream: always serve the freshest frame, be resilient to packet loss.
    snm['StreamBufferHandlingMode'].value      = 'NewestOnly'
    snm['StreamAutoNegotiatePacketSize'].value = True
    snm['StreamPacketResendEnable'].value      = True

    # Try pixel formats in preference order. BGR8/RGB8 are debayered on-camera
    # (no host work); BayerRG8 is raw and gets debayered in _grab_one via cv2.
    chosen = None
    last_err = None
    for cand in ('BGR8', 'RGB8', 'BayerRG8'):
        try:
            nm['PixelFormat'].value = cand
            chosen = cand
            break
        except Exception as e:
            last_err = e
    if chosen is None:
        raise RuntimeError(
            f"Camera doesn't support BGR8, RGB8, or BayerRG8. Last error: {last_err}")

    _set_if_possible(nm, 'AcquisitionMode',  'Continuous')
    _set_if_possible(nm, 'ExposureAuto',     'Continuous')
    _set_if_possible(nm, 'GainAuto',         'Continuous')
    _set_if_possible(nm, 'BalanceWhiteAuto', 'Continuous')

    if fps is not None:
        try:
            nm['AcquisitionFrameRateEnable'].value = True
            nm['AcquisitionFrameRate'].value = float(fps)
        except Exception as e:
            print(f"  (warning) couldn't set frame rate to {fps}: {e}")

    return chosen


# ---------------------------------------------------------------------------
# Synchronised grab
# ---------------------------------------------------------------------------

def _grab_one(dev, name, pixel_format, barrier, out, errors, timeout_ms=3000):
    try:
        barrier.wait(timeout=5.0)
        t_call = time.perf_counter_ns()
        buf    = dev.get_buffer(timeout=timeout_ms)
        ts_dev = int(getattr(buf, 'timestamp_ns', 0) or 0)
        h, w   = buf.height, buf.width

        # Interpret the buffer based on the format we asked the camera for.
        if pixel_format in ('BGR8', 'RGB8'):
            arr = np.ctypeslib.as_array(buf.pdata,
                                        shape=(h, w, 3)).astype(np.uint8, copy=True)
            if pixel_format == 'RGB8':
                arr = arr[:, :, ::-1].copy()           # flip to BGR for OpenCV
        elif pixel_format.startswith('Bayer'):
            raw = np.ctypeslib.as_array(buf.pdata,
                                        shape=(h, w)).astype(np.uint8, copy=True)
            bayer_codes = {
                'BayerRG8': cv2.COLOR_BayerRG2BGR,
                'BayerGB8': cv2.COLOR_BayerGB2BGR,
                'BayerGR8': cv2.COLOR_BayerGR2BGR,
                'BayerBG8': cv2.COLOR_BayerBG2BGR,
            }
            arr = cv2.cvtColor(raw, bayer_codes.get(pixel_format, cv2.COLOR_BayerRG2BGR))
        else:
            raise RuntimeError(f"Unsupported pixel format in grab: {pixel_format}")

        dev.requeue_buffer(buf)
        out[name] = {'image': arr, 't_call_ns': t_call, 't_device_ns': ts_dev}
    except Exception as e:
        errors[name] = repr(e)


def snapshot(devices: dict, pf_per_cam: dict):
    barrier = threading.Barrier(len(devices))
    results, errors = {}, {}
    threads = [
        threading.Thread(target=_grab_one,
                         args=(dev, name, pf_per_cam[name],
                               barrier, results, errors),
                         name=f"grab-{name}",
                         daemon=True)
        for name, dev in devices.items()
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    if errors:
        raise RuntimeError("grab errors: " + json.dumps(errors))
    return results


# ---------------------------------------------------------------------------
# Save & report
# ---------------------------------------------------------------------------

def save_set(results: dict, outdir: str):
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    set_dir = os.path.join(outdir, stamp)
    os.makedirs(set_dir, exist_ok=True)

    meta = {'capture_local_time': stamp, 'cameras': {}}
    for name, r in results.items():
        path = os.path.join(set_dir, f"{name}.png")
        cv2.imwrite(path, r['image'])
        meta['cameras'][name] = {
            't_call_ns':   r['t_call_ns'],
            't_device_ns': r['t_device_ns'],
            'file':        f"{name}.png",
            'shape':       list(r['image'].shape),
        }
    with open(os.path.join(set_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return set_dir


def report_skew(results: dict):
    ts_dev  = {n: r['t_device_ns'] for n, r in results.items() if r['t_device_ns']}
    ts_call = {n: r['t_call_ns']   for n, r in results.items()}

    if len(ts_dev) == len(results):
        ref = min(ts_dev.values())
        span_ms = (max(ts_dev.values()) - ref) / 1e6
        print("  device timestamps (relative):")
        for n in results:
            print(f"    {n:6s}  {(ts_dev[n] - ref)/1e6:+8.3f} ms")
        print(f"  inter-camera skew: {span_ms:.3f} ms")
    else:
        ref = min(ts_call.values())
        span_ms = (max(ts_call.values()) - ref) / 1e6
        print("  (no device timestamps -- host call-time deltas)")
        for n in results:
            print(f"    {n:6s}  {(ts_call[n] - ref)/1e6:+8.3f} ms")
        print(f"  host-side span: {span_ms:.3f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default='snapshots',
                    help="Where to write the per-capture subdirectories.")
    ap.add_argument('--loop', action='store_true',
                    help="Interactive mode: Enter to grab, q+Enter to quit.")
    ap.add_argument('--fps', type=float, default=None,
                    help="Request a frame rate on each camera.")
    ap.add_argument('--warmup', type=int, default=15,
                    help="Frames to drain per camera before the first capture.")
    ap.add_argument('--cam', action='append', default=None,
                    metavar='NAME=SERIAL',
                    help="Override which cameras to use, e.g. "
                         "--cam center=212100123 --cam left=... --cam right=... "
                         "(otherwise read from config.CAMERAS).")
    return ap.parse_args()


def resolve_cameras(args):
    if args.cam:
        result = {}
        for item in args.cam:
            if '=' not in item:
                raise SystemExit(f"--cam expects NAME=SERIAL, got {item!r}")
            name, serial = item.split('=', 1)
            result[name.strip()] = serial.strip()
        return result
    if config is None or not getattr(config, 'CAMERAS', None):
        raise SystemExit("No config.CAMERAS found and --cam not given.")
    return dict(config.CAMERAS)


def main():
    args    = parse_args()
    cameras = resolve_cameras(args)
    if len(cameras) != 3:
        print(f"(note) running with {len(cameras)} camera(s) -- expected 3.")

    print("Discovering cameras ...")
    devices = discover_devices(cameras)

    try:
        print("Configuring streams ...")
        pf_per_cam = {}
        for name, dev in devices.items():
            pf = configure(dev, fps=args.fps)
            pf_per_cam[name] = pf
            print(f"  {name:6s} pixel format: {pf}")
            dev.start_stream()

        if args.warmup:
            print(f"Warming up ({args.warmup} frames per camera) ...")
            for _ in range(args.warmup):
                for dev in devices.values():
                    try:
                        b = dev.get_buffer(timeout=3000)
                        dev.requeue_buffer(b)
                    except Exception:
                        pass

        def do_capture():
            print("\nCapturing ...")
            t0 = time.perf_counter()
            results = snapshot(devices, pf_per_cam)
            t1 = time.perf_counter()
            set_dir = save_set(results, args.outdir)
            print(f"  saved -> {set_dir}   (wall {(t1-t0)*1e3:.1f} ms)")
            report_skew(results)

        if args.loop:
            print("\nReady. Press Enter to capture, or q+Enter to quit.")
            while True:
                try:
                    k = input("> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if k == 'q':
                    break
                do_capture()
        else:
            do_capture()

    finally:
        for dev in devices.values():
            try: dev.stop_stream()
            except Exception: pass
        system.destroy_device()
        print("\nClosed all cameras.")


if __name__ == "__main__":
    main()