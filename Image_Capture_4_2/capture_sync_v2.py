"""
Simultaneous capture from four Lucid TRI050S-CC GigE cameras.

Free-running cameras can't be exposed at exactly the same instant in software
(that needs PTP + Action Commands, or a hardware trigger line). This script
does the closest software equivalent:

  * All four streams are started ahead of time with the buffer mode set to
    'NewestOnly', so get_buffer() always returns the most recently arrived
    frame.
  * One thread per camera is parked at a Barrier and then released together.
    The inter-camera grab calls fire within microseconds of each other.
  * Each device's hardware timestamp (timestamp_ns) is recorded so the actual
    inter-camera skew is visible in the meta.json and on stdout.

READ THIS BEFORE USING THE OUTPUT GEOMETRICALLY. The barrier synchronises the
grab CALLS, not the exposures. Each camera hands back whatever frame it most
recently finished, so the worst-case skew is one frame period -- at 20 ms
exposure with four RGB8 streams sharing the link, that is tens of milliseconds,
not microseconds. Fine for a static scene. Not fine for extrinsics, and not
fine for anything moving. For those, use auto_capture.py --mode extrinsics,
which drives PTP + a scheduled Action0 and gets the skew into microseconds.
The skew report below tells you which regime you are actually in.

Exposure / gain
---------------
Auto exposure and auto gain are DISABLED by default, and every camera gets the
IDENTICAL fixed exposure and gain. Auto modes re-meter per camera and per
frame, so brightness drifts between captures and, worse, differs between
cameras in the same set -- poison for calibration and for cross-camera colour
and scale consistency.

Defaults are 20000 us and 18.7 dB, matching what the calibration shoot locks
to, so snapshots taken here are directly comparable with the frames the
calibration was computed from. Pass --auto-exposure only if you explicitly
want free metering.

Because the two settings are only useful if they actually LAND, every camera's
values are read back after configuration and compared. Node min/max clamping is
per-camera, so a request outside one camera's range would silently leave that
camera brighter or darker than the rest; a mismatch is now reported loudly.

White balance is converged once during warmup and then frozen (--wb lock, the
default). Leaving BalanceWhiteAuto on Continuous lets each camera drift to its
own ratios, which defeats the point of pinning exposure and gain.

NOTE on trigger state
---------------------
If these cameras were previously armed for the PTP + scheduled-Action0
pipeline, they still have TriggerMode=On / TriggerSource=Action0 baked into
their persistent nodemap. Reconnecting and setting AcquisitionMode=Continuous
does NOT clear that. A camera left in triggered mode will stream happily but
never deliver a frame, so every get_buffer() returns SC_ERR_TIMEOUT (-1011).
configure() now explicitly disables triggering up front to guarantee a clean
free-running state.

NOTE on bandwidth
-----------------
Four RGB8 frames at 2448x2048 is ~15 MB each, ~60 MB per set. Free-running,
that is a sustained load rather than a burst. If the switch uplink to the
Jetson is the 2.5G port, use --throughput-limit to pace each camera (e.g.
60000000 bytes/s) -- otherwise the cameras compete, packets drop, resends pile
up, and the inter-camera skew you measure is mostly queueing, not acquisition.

Usage
-----
    python capture_sync.py                       # capture one set and exit
    python capture_sync.py --loop                # Enter = grab, q+Enter = quit
    python capture_sync.py --outdir my_caps      # custom output directory
    python capture_sync.py --fps 60              # request a frame rate
    python capture_sync.py --exposure-us 5000    # fixed 5 ms exposure
    python capture_sync.py --gain-db 3           # fixed 3 dB gain
    python capture_sync.py --auto-exposure       # re-enable auto metering
    python capture_sync.py --throughput-limit 60000000

Each capture creates one timestamped subdirectory containing left.png,
center.png, top.png, right.png, and a meta.json with per-camera serials,
device timestamps, and the exposure settings in force.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from arena_api.system import system

try:
    import config                              # CAMERAS = {'left': serial, ...}
except ImportError:
    config = None

# Defaults chosen to match the calibration shoot's locked settings.
DEFAULT_EXPOSURE_US = 20000.0
DEFAULT_GAIN_DB     = 25.7

# Above this the "synchronised" set is really just four nearby frames.
SKEW_WARN_MS = 5.0


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


def _disable_triggering(nm):
    """Force the camera into a clean free-running state.

    The PTP + Action-command pipeline leaves TriggerMode=On with
    TriggerSource=Action0 persisted on the device. If we don't undo that here,
    the camera will stream but wait forever for a trigger that never fires, and
    every get_buffer() call times out with SC_ERR_TIMEOUT (-1011).

    We walk the common TriggerSelector values because TriggerMode is a
    per-selector setting -- clearing only the active selector can leave another
    one armed.
    """
    for sel in ('FrameStart', 'AcquisitionStart', 'FrameBurstStart',
                'FrameActive', 'ExposureStart'):
        try:
            nm['TriggerSelector'].value = sel
            nm['TriggerMode'].value     = 'Off'
        except Exception:
            pass
    # Belt-and-braces: clear TriggerMode without touching the selector, in case
    # this model exposes it as a flat node.
    _set_if_possible(nm, 'TriggerMode', 'Off')
    # Point the trigger source somewhere harmless so a stray Action0 broadcast
    # can't sneak a trigger in mid-capture.
    _set_if_possible(nm, 'TriggerSource', 'Software')


def _set_fixed_exposure(nm, exposure_us: float):
    """Turn auto exposure OFF and pin ExposureTime to exposure_us.

    ExposureTime is only writable once ExposureAuto is 'Off', so the order
    matters. We clamp to the node's advertised min/max where available so an
    out-of-range request degrades to the nearest legal value instead of raising.
    Clamping is per-camera, which is exactly why verify_uniform_exposure()
    exists -- a silent clamp on one camera breaks the whole point of fixing
    these values.
    """
    try:
        nm['ExposureAuto'].value = 'Off'
    except Exception as e:
        print(f"  (warning) couldn't set ExposureAuto=Off: {e}")

    try:
        node = nm['ExposureTime']
        lo = getattr(node, 'min', None)
        hi = getattr(node, 'max', None)
        val = float(exposure_us)
        if lo is not None:
            val = max(val, float(lo))
        if hi is not None:
            val = min(val, float(hi))
        node.value = val
        if val != float(exposure_us):
            print(f"  (note) exposure clamped to {val:.0f} us "
                  f"(requested {exposure_us:.0f})")
    except Exception as e:
        print(f"  (warning) couldn't set ExposureTime={exposure_us}: {e}")


def _set_fixed_gain(nm, gain_db: float):
    """Turn auto gain OFF and pin Gain to gain_db."""
    try:
        nm['GainAuto'].value = 'Off'
    except Exception as e:
        print(f"  (warning) couldn't set GainAuto=Off: {e}")

    try:
        node = nm['Gain']
        lo = getattr(node, 'min', None)
        hi = getattr(node, 'max', None)
        val = float(gain_db)
        if lo is not None:
            val = max(val, float(lo))
        if hi is not None:
            val = min(val, float(hi))
        node.value = val
        if val != float(gain_db):
            print(f"  (note) gain clamped to {val:.2f} dB "
                  f"(requested {gain_db:.2f})")
    except Exception as e:
        print(f"  (warning) couldn't set Gain={gain_db}: {e}")


def configure(dev, fps: float | None = None, disable_ptp: bool = False,
              auto_exposure: bool = False,
              exposure_us: float = DEFAULT_EXPOSURE_US,
              gain_db: float = DEFAULT_GAIN_DB,
              wb: str = 'lock',
              throughput_limit: int = 0):
    """Configure stream + pixel format. Always uses RGB8.

    By default exposure and gain are fixed (auto OFF) and identical on every
    camera. Set auto_exposure=True to restore free metering.
    """
    nm  = dev.nodemap
    snm = dev.tl_stream_nodemap

    # Stream: always serve the freshest frame, be resilient to packet loss.
    snm['StreamBufferHandlingMode'].value      = 'NewestOnly'
    snm['StreamAutoNegotiatePacketSize'].value = True
    snm['StreamPacketResendEnable'].value      = True

    # CRITICAL: undo any trigger/Action config left on the camera by the PTP +
    # Action-command pipeline before we try to free-run. Must happen before
    # start_stream(); otherwise get_buffer() times out (-1011).
    _disable_triggering(nm)

    # Optionally drop PTP so the camera's acquisition clock is purely local.
    # Harmless to leave PTP enabled for free-running capture, but disabling it
    # removes one more source of surprising behaviour. Note that with PTP off
    # the device timestamps are no longer on a common clock, so the skew report
    # becomes per-camera-relative and much less meaningful.
    if disable_ptp:
        _set_if_possible(nm, 'PtpEnable', False)

    # Four sustained RGB8 streams can exceed a 2.5G uplink. Pacing each camera
    # is better than letting the switch arbitrate by dropping packets.
    if throughput_limit:
        try:
            nm['DeviceLinkThroughputLimitMode'].value = 'On'
            nm['DeviceLinkThroughputLimit'].value = int(throughput_limit)
            print(f"  throughput capped at {int(throughput_limit)/1e6:.0f} MB/s")
        except Exception as e:
            print(f"  (warning) throughput limit not applied: {e}")
    else:
        _set_if_possible(nm, 'DeviceLinkThroughputLimitMode', 'Off')

    # Force RGB8 on every camera. Debayered on-camera (no host work). Fail
    # loudly if a camera can't deliver it -- a mixed-format rig would silently
    # corrupt the calibration data.
    try:
        nm['PixelFormat'].value = 'RGB8'
    except Exception as e:
        raise RuntimeError(
            f"{dev} doesn't support RGB8 (or it's not selectable here): {e}")
    chosen = 'RGB8'

    _set_if_possible(nm, 'AcquisitionMode', 'Continuous')

    # Exposure / gain: fixed by default so brightness is stable across captures
    # and consistent between all four cameras.
    if auto_exposure:
        _set_if_possible(nm, 'ExposureAuto', 'Continuous')
        _set_if_possible(nm, 'GainAuto',     'Continuous')
        print("  exposure: AUTO (Continuous)")
    else:
        _set_fixed_exposure(nm, exposure_us)
        _set_fixed_gain(nm, gain_db)
        print(f"  exposure: FIXED {exposure_us:.0f} us, gain {gain_db:.2f} dB")

    # White balance. 'lock' converges during warmup and is frozen afterwards by
    # freeze_white_balance(); 'auto' leaves it drifting per camera; 'off' uses
    # whatever ratios are already persisted on the device. WB does not affect
    # geometric calibration, but it does affect any photometric or learned
    # depth model that sees colour.
    if wb == 'auto' or wb == 'lock':
        _set_if_possible(nm, 'BalanceWhiteAuto', 'Continuous')
    else:
        _set_if_possible(nm, 'BalanceWhiteAuto', 'Off')

    if fps is not None:
        try:
            nm['AcquisitionFrameRateEnable'].value = True
            nm['AcquisitionFrameRate'].value = float(fps)
        except Exception as e:
            print(f"  (warning) couldn't set frame rate to {fps}: {e}")

    return chosen


def freeze_white_balance(devices: dict):
    """Switch BalanceWhiteAuto off after warmup so the converged ratios stick."""
    for name, dev in devices.items():
        _set_if_possible(dev.nodemap, 'BalanceWhiteAuto', 'Off')
    print("  white balance frozen at the warmup-converged ratios")


def read_exposure(devices: dict):
    """Read back ExposureTime / Gain from every camera."""
    vals = {}
    for name, dev in devices.items():
        nm = dev.nodemap
        try:
            vals[name] = (float(nm['ExposureTime'].value),
                          float(nm['Gain'].value))
        except Exception:
            vals[name] = (None, None)
    return vals


def verify_uniform_exposure(devices: dict, requested_us, requested_db):
    """Confirm all cameras ended up on the SAME exposure and gain.

    Node ranges differ slightly between units and clamping happens silently
    per camera. If one camera sits at a different exposure, every cross-camera
    comparison downstream inherits a brightness offset that looks like a
    lighting or albedo difference rather than a settings bug.
    """
    vals = read_exposure(devices)
    print("  readback:")
    for name, (e, g) in vals.items():
        if e is None:
            print(f"    {name:6s}  <unreadable>")
        else:
            print(f"    {name:6s}  {e:8.0f} us   {g:6.2f} dB")

    good = {n: v for n, v in vals.items() if v[0] is not None}
    if len(good) < 2:
        return vals
    exps = [v[0] for v in good.values()]
    gains = [v[1] for v in good.values()]
    if max(exps) - min(exps) > 1.0 or max(gains) - min(gains) > 0.05:
        print("  ** CAMERAS ARE NOT MATCHED. At least one clamped to a "
              "different value than the rest, so the set will have a "
              "brightness offset between views. Pick settings inside every "
              "camera's range. **")
    elif requested_us is not None and abs(exps[0] - requested_us) > 1.0:
        print(f"  ** All cameras agree at {exps[0]:.0f} us but that is not the "
              f"requested {requested_us:.0f} us -- the request was outside the "
              f"sensor range. **")
    return vals


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

        # Cast explicitly: buf.pdata is a void pointer on some SDK builds and
        # np.ctypeslib.as_array needs a typed pointer to shape it.
        ptr = ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_ubyte))

        # Interpret the buffer based on the format we asked the camera for.
        if pixel_format in ('BGR8', 'RGB8'):
            arr = np.ctypeslib.as_array(ptr, shape=(h, w, 3)).copy()
            if pixel_format == 'RGB8':
                # Channel reorder only -- no interpolation, no colour maths.
                # cv2.imwrite serialises as BGR, so this is what makes the PNG
                # carry the camera's actual colours.
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pixel_format.startswith('Bayer'):
            raw = np.ctypeslib.as_array(ptr, shape=(h, w)).copy()
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

def save_set(results: dict, outdir: str, cameras: dict,
             exposure_meta: dict | None = None):
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    set_dir = os.path.join(outdir, stamp)
    os.makedirs(set_dir, exist_ok=True)

    meta = {'capture_local_time': stamp, 'cameras': {}}
    if exposure_meta is not None:
        meta['exposure'] = exposure_meta
    for name, r in results.items():
        path = os.path.join(set_dir, f"{name}.png")
        cv2.imwrite(path, r['image'])
        meta['cameras'][name] = {
            'serial':      cameras.get(name),
            't_call_ns':   r['t_call_ns'],
            't_device_ns': r['t_device_ns'],
            'file':        f"{name}.png",
            'shape':       list(r['image'].shape),
        }

    ts = [r['t_device_ns'] for r in results.values() if r['t_device_ns']]
    if len(ts) == len(results) and ts:
        meta['skew_ms'] = (max(ts) - min(ts)) / 1e6
    with open(os.path.join(set_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return set_dir


def report_skew(results: dict, order=None):
    names = order or list(results)
    names = [n for n in names if n in results]
    ts_dev  = {n: r['t_device_ns'] for n, r in results.items() if r['t_device_ns']}
    ts_call = {n: r['t_call_ns']   for n, r in results.items()}

    if len(ts_dev) == len(results):
        ref = min(ts_dev.values())
        span_ms = (max(ts_dev.values()) - ref) / 1e6
        print("  device timestamps (relative):")
        for n in names:
            print(f"    {n:6s}  {(ts_dev[n] - ref)/1e6:+8.3f} ms")
        print(f"  inter-camera skew: {span_ms:.3f} ms")
        if span_ms > SKEW_WARN_MS:
            worst = max(ts_dev, key=lambda n: ts_dev[n])
            print(f"  ** {span_ms:.1f} ms skew is a frame-period effect, not "
                  f"synchronisation ({worst} is the laggard). Fine for a "
                  f"static scene; do NOT use this set for extrinsics or for "
                  f"anything that moved. Use auto_capture.py --mode "
                  f"extrinsics (PTP + Action0) for that. **")
    else:
        ref = min(ts_call.values())
        span_ms = (max(ts_call.values()) - ref) / 1e6
        print("  (no device timestamps -- host call-time deltas)")
        for n in names:
            print(f"    {n:6s}  {(ts_call[n] - ref)/1e6:+8.3f} ms")
        print(f"  host-side span: {span_ms:.3f} ms")
        print("  (host call times say nothing about when the sensors actually "
              "integrated -- enable PTP for a real number)")


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
    ap.add_argument('--disable-ptp', action='store_true',
                    help="Disable PTP on each camera for purely local free-run.")
    ap.add_argument('--auto-exposure', action='store_true',
                    help="Re-enable auto exposure + auto gain (default is OFF).")
    ap.add_argument('--exposure-us', type=float, default=DEFAULT_EXPOSURE_US,
                    help="Fixed exposure time in microseconds, applied "
                         "identically to every camera. "
                         "Ignored when --auto-exposure is set.")
    ap.add_argument('--gain-db', type=float, default=DEFAULT_GAIN_DB,
                    help="Fixed gain in dB, applied identically to every "
                         "camera. Ignored when --auto-exposure is set.")
    ap.add_argument('--wb', choices=('lock', 'auto', 'off'), default='lock',
                    help="lock = converge during warmup then freeze; auto = "
                         "leave each camera metering independently; off = use "
                         "the ratios already on the device.")
    ap.add_argument('--throughput-limit', type=int, default=0,
                    help="Per-camera DeviceLinkThroughputLimit in bytes/s "
                         "(0 = uncapped). Try 60000000 if four RGB8 streams "
                         "share a 2.5G uplink.")
    ap.add_argument('--cameras', nargs='+', default=None,
                    help="Subset of config.CAMERAS names to capture, e.g. "
                         "--cameras left center. Default: all of them.")
    ap.add_argument('--cam', action='append', default=None,
                    metavar='NAME=SERIAL',
                    help="Override which cameras to use, e.g. "
                         "--cam left=260505158 --cam center=261100628 "
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
    cams = dict(config.CAMERAS)
    if args.cameras:
        unknown = [n for n in args.cameras if n not in cams]
        if unknown:
            raise SystemExit(f"Unknown camera name(s): {', '.join(unknown)}. "
                             f"Known: {', '.join(cams)}")
        cams = {n: s for n, s in cams.items() if n in set(args.cameras)}
    return cams


def main():
    args    = parse_args()
    cameras = resolve_cameras(args)
    expected = len(getattr(config, 'CAMERAS', {}) or {}) or len(cameras)
    if len(cameras) != expected:
        print(f"(note) running with {len(cameras)} of {expected} camera(s).")

    exposure_meta = {
        'auto_exposure': bool(args.auto_exposure),
        'exposure_us':   None if args.auto_exposure else args.exposure_us,
        'gain_db':       None if args.auto_exposure else args.gain_db,
        'white_balance': args.wb,
        'pixel_format':  'RGB8',
    }

    print("Discovering cameras ...")
    devices = discover_devices(cameras)

    try:
        print("Configuring streams ...")
        pf_per_cam = {}
        for name, dev in devices.items():
            print(f"  {name}:")
            pf = configure(dev, fps=args.fps, disable_ptp=args.disable_ptp,
                           auto_exposure=args.auto_exposure,
                           exposure_us=args.exposure_us, gain_db=args.gain_db,
                           wb=args.wb,
                           throughput_limit=args.throughput_limit)
            pf_per_cam[name] = pf
            print(f"  {name:6s} pixel format: {pf}")
            dev.start_stream()

        if args.warmup:
            print(f"Warming up ({args.warmup} frames per camera) ...")
            ok = 0
            total = args.warmup * len(devices)
            for _ in range(args.warmup):
                for dev in devices.values():
                    try:
                        b = dev.get_buffer(timeout=3000)
                        dev.requeue_buffer(b)
                        ok += 1
                    except Exception:
                        pass
            print(f"  warmup: {ok}/{total} frames drained")
            if ok == 0:
                raise RuntimeError(
                    "Warmup drained 0 frames -- cameras are streaming but not "
                    "delivering. Most likely still in triggered mode "
                    "(TriggerMode=On / TriggerSource=Action0), or GigE bandwidth "
                    "starvation on RGB8. Check trigger state and link config, "
                    "and try --throughput-limit 60000000.")
            if ok < total * 0.5:
                print("  ** Fewer than half the warmup frames arrived. With "
                      "four RGB8 streams this usually means the link is "
                      "saturated -- try --throughput-limit or lower --fps. **")

        if args.wb == 'lock' and not args.auto_exposure:
            freeze_white_balance(devices)

        if not args.auto_exposure:
            print("Verifying exposure is identical across cameras ...")
            vals = verify_uniform_exposure(devices, args.exposure_us,
                                           args.gain_db)
            exposure_meta['readback'] = {
                n: {'exposure_us': v[0], 'gain_db': v[1]}
                for n, v in vals.items()}

        order = [n for n in (getattr(config, 'CAMERAS', {}) or cameras)
                 if n in devices]

        def do_capture():
            print("\nCapturing ...")
            t0 = time.perf_counter()
            results = snapshot(devices, pf_per_cam)
            t1 = time.perf_counter()
            set_dir = save_set(results, args.outdir, cameras, exposure_meta)
            print(f"  saved -> {set_dir}   (wall {(t1-t0)*1e3:.1f} ms)")
            report_skew(results, order)

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