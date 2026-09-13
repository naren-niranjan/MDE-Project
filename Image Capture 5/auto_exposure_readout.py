"""
Find the exposure / gain that auto exposure settles on, per camera.

Auto exposure only meters while the camera is streaming, so a bare readback can
return a stale value. This script enables ExposureAuto=Continuous, streams for a
short window, watches ExposureTime / Gain converge, and prints the final settled
numbers -- ready to paste into capture_sync.py's --exposure-us / --gain-db.

    python auto_exposure_readout.py                 # ~40 frames per camera
    python auto_exposure_readout.py --frames 80     # let it settle longer
    python auto_exposure_readout.py --cam center=261100631 --cam left=260505158 ...

Once you have the number, pin it:

    python capture_sync.py --exposure-us <value> --gain-db <value>
"""

from __future__ import annotations

import argparse
import time

from arena_api.system import system

try:
    import config
except ImportError:
    config = None


def _set_if_possible(nm, key, value):
    try:
        nm[key].value = value
    except Exception:
        pass


def discover(name_to_serial, tries=6, delay=1.0):
    infos = None
    for _ in range(tries):
        infos = system.device_infos
        if infos:
            break
        time.sleep(delay)
    if not infos:
        raise RuntimeError("No Lucid devices on the network.")
    by_serial = {d['serial']: d for d in infos}
    devs = {}
    for name, serial in name_to_serial.items():
        if serial not in by_serial:
            raise RuntimeError(
                f"{name!r} ({serial}) not found. Seen: {', '.join(by_serial)}")
        devs[name] = system.create_device([by_serial[serial]])[0]
    return devs


def _disable_triggering(nm):
    for sel in ('FrameStart', 'AcquisitionStart', 'FrameBurstStart',
                'FrameActive', 'ExposureStart'):
        try:
            nm['TriggerSelector'].value = sel
            nm['TriggerMode'].value = 'Off'
        except Exception:
            pass
    _set_if_possible(nm, 'TriggerMode', 'Off')
    _set_if_possible(nm, 'TriggerSource', 'Software')


def _read(nm, key):
    try:
        return nm[key].value
    except Exception:
        return None


def settle(dev, name, frames, quiet):
    nm = dev.nodemap
    snm = dev.tl_stream_nodemap

    snm['StreamBufferHandlingMode'].value = 'NewestOnly'
    snm['StreamAutoNegotiatePacketSize'].value = True
    snm['StreamPacketResendEnable'].value = True
    _disable_triggering(nm)
    _set_if_possible(nm, 'PixelFormat', 'RGB8')
    _set_if_possible(nm, 'AcquisitionMode', 'Continuous')

    # Turn auto ON so the camera meters and chooses its own exposure/gain.
    _set_if_possible(nm, 'ExposureAuto', 'Continuous')
    _set_if_possible(nm, 'GainAuto', 'Continuous')

    dev.start_stream()
    last_exp = last_gain = None
    try:
        for i in range(frames):
            try:
                b = dev.get_buffer(timeout=3000)
                dev.requeue_buffer(b)
            except Exception:
                continue
            exp = _read(nm, 'ExposureTime')
            gain = _read(nm, 'Gain')
            last_exp, last_gain = exp, gain
            if not quiet and (i % 5 == 0 or i == frames - 1):
                es = f"{exp:8.1f} us" if exp is not None else "   ? us"
                gs = f"{gain:5.2f} dB" if gain is not None else "  ? dB"
                print(f"    [{name:6s}] frame {i:3d}:  exp {es}   gain {gs}")
    finally:
        dev.stop_stream()

    return last_exp, last_gain


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', type=int, default=40,
                    help="Frames to stream per camera so auto can converge.")
    ap.add_argument('--quiet', action='store_true',
                    help="Only print the final settled value.")
    ap.add_argument('--cam', action='append', default=None, metavar='NAME=SERIAL')
    return ap.parse_args()


def resolve(args):
    if args.cam:
        out = {}
        for item in args.cam:
            name, serial = item.split('=', 1)
            out[name.strip()] = serial.strip()
        return out
    if config is None or not getattr(config, 'CAMERAS', None):
        raise SystemExit("No config.CAMERAS and no --cam given.")
    return dict(config.CAMERAS)


def main():
    args = parse_args()
    cams = resolve(args)
    print("Discovering ...")
    devs = discover(cams)

    settled = {}
    try:
        for name, dev in devs.items():
            print(f"\nMetering {name} ({args.frames} frames) ...")
            exp, gain = settle(dev, name, args.frames, args.quiet)
            settled[name] = (exp, gain)
    finally:
        system.destroy_device()

    print("\n" + "=" * 46)
    print("Settled auto-exposure values:")
    for name, (exp, gain) in settled.items():
        es = f"{exp:.0f}" if exp is not None else "?"
        gs = f"{gain:.2f}" if gain is not None else "?"
        print(f"  {name:6s}  exposure {es:>7s} us   gain {gs:>6s} dB")
    print("=" * 46)
    print("\nTo pin these, e.g. for the center camera:")
    c = next(iter(settled.values()))
    if c[0] is not None:
        print(f"  python capture_sync.py --exposure-us {c[0]:.0f} "
              f"--gain-db {0 if c[1] is None else c[1]:.1f}")


if __name__ == "__main__":
    main()