#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
measure_acquisition.py -- sustained frame-set rate for the four-camera rig,
with the payload accounting the thesis quotes beside it.

Closes: \gap{measured acquisition rate and overhead accounting}

Section 5.5 states a payload bound of about 4.16 frame sets per second
against a 250 MB/s effective uplink, an observed figure of about 3.7, and
infers roughly 11 per cent protocol and scheduling overhead. That
inference needs the observed figure to be a measurement rather than a
recollection. This script provides it: it acquires for a fixed wall-clock
window, counts complete synchronised sets, and reports the rate with its
spread, the per-camera rates, incomplete sets, and the payload actually
moved.

    python3 measure_acquisition.py --seconds 60
    python3 measure_acquisition.py --seconds 60 --pixel-format BayerRG8

Writes acquisition_report.md (numbers plus a LaTeX sentence to paste)
and acquisition_raw.json (every set timestamp, for the spread).

Requires arena_api, as the streaming pipeline does. Run it with the
streamer stopped: two processes cannot own the cameras at once.
"""

import argparse
import io
import json
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=60.0,
                    help='acquisition window, wall clock')
    ap.add_argument('--pixel-format', default=None,
                    help='override; default is whatever the camera holds')
    ap.add_argument('--timeout-ms', type=int, default=2000)
    ap.add_argument('--model-filter', default='TRI050S',
                    help='substring of DeviceModelName identifying the rig '
                         'cameras; other GigE devices on the network are '
                         'skipped')
    ap.add_argument('--out', default='acquisition_report.md')
    a = ap.parse_args()

    try:
        from arena_api.system import system
    except ImportError:
        raise SystemExit("arena_api not importable. Run inside the same "
                         "environment the streamer uses.")

    # discover first, so a discovery failure is distinguishable from an
    # access failure
    try:
        infos = system.device_infos
    except Exception as e:
        raise SystemExit("device discovery failed: %r" % e)
    if not infos:
        raise SystemExit(
            "no cameras on the subnet. Check link and interface:\n"
            "  ip -br addr | grep -i eth\n"
            "  ping -c1 <camera ip>")
    print("discovered %d device(s): %s" % (
        len(infos), ', '.join("%s@%s" % (i.get('serial', '?'),
                                         i.get('ip', '?')) for i in infos)),
        file=sys.stderr)

    # Open only the rig cameras. Other GigE Vision devices may share the
    # network -- the Roboception reference sensor does -- and they are
    # held exclusively by their own controllers. Opening the whole
    # discovery list makes one refusal abort the run.
    wanted = [i for i in infos
              if a.model_filter.lower() in str(i.get('model', '')).lower()]
    if not wanted:
        raise SystemExit(
            "no device matched --model-filter %r. Discovered models: %s"
            % (a.model_filter,
               ', '.join(sorted({str(i.get('model', '?')) for i in infos}))))
    skipped = len(infos) - len(wanted)
    if skipped:
        print("  skipping %d non-rig device(s): %s" % (
            skipped, ', '.join("%s (%s)" % (i.get('model', '?'),
                                            i.get('ip', '?'))
                               for i in infos if i not in wanted)),
            file=sys.stderr)

    try:
        devices = system.create_device(wanted)
    except Exception as e:
        if 'ACCESS_DENIED' in str(e) or '-1005' in str(e):
            raise SystemExit(
                "SC_ERR_ACCESS_DENIED: another process holds the cameras.\n"
                "Arena grants exclusive access, so the streamer, a viewer, "
                "or a dead run still owning the handles will block this.\n"
                "  ps aux | grep -iE 'stream|live|multicam|arena' | grep -v grep\n"
                "  # kill it, then retry. If nothing is running, the handles\n"
                "  # are stale: power-cycle the cameras or wait for the\n"
                "  # heartbeat timeout (~3 s by default, longer if raised).\n"
                "If the refusing device is not a rig camera, narrow "
                "--model-filter.")
        raise
    if not devices:
        raise SystemExit("discovery saw cameras but none could be opened")
    print("opened: %d" % len(devices), file=sys.stderr)

    info = []
    for d in devices:
        n = d.nodemap
        if a.pixel_format:
            try:
                n['PixelFormat'].value = a.pixel_format
            except Exception as e:
                print("  could not set PixelFormat: %r" % e, file=sys.stderr)
        rec = {
            'serial': str(n['DeviceSerialNumber'].value),
            'model':  str(n['DeviceModelName'].value),
            'pixel_format': str(n['PixelFormat'].value),
            'width':  int(n['Width'].value),
            'height': int(n['Height'].value),
        }
        for key, node in (('acq_frame_rate', 'AcquisitionFrameRate'),
                          ('exposure_us', 'ExposureTime'),
                          ('link_speed_Bps', 'DeviceLinkSpeed'),
                          ('throughput_limit_Bps', 'DeviceLinkThroughputLimit')):
            try:
                rec[key] = float(node and n[node].value)
            except Exception:
                rec[key] = None
        info.append(rec)
        d.start_stream(10)

    # ---- acquire ---------------------------------------------------
    set_times, incomplete, per_cam = [], 0, [0] * len(devices)
    t0 = time.perf_counter()
    deadline = t0 + a.seconds
    try:
        while time.perf_counter() < deadline:
            got = 0
            for i, d in enumerate(devices):
                try:
                    buf = d.get_buffer(timeout=a.timeout_ms)
                    if not buf.is_incomplete:
                        got += 1
                        per_cam[i] += 1
                    d.requeue_buffer(buf)
                except Exception:
                    pass
            if got == len(devices):
                set_times.append(time.perf_counter())
            elif got:
                incomplete += 1
    finally:
        for d in devices:
            try: d.stop_stream()
            except Exception: pass
        system.destroy_device()

    elapsed = time.perf_counter() - t0
    n_sets = len(set_times)
    if n_sets < 2:
        raise SystemExit("fewer than two complete sets acquired; nothing to report")

    gaps = [b - a_ for a_, b in zip(set_times, set_times[1:])]
    rate_mean = n_sets / elapsed
    inst = [1.0 / g for g in gaps if g > 0]
    inst.sort()

    # ---- payload accounting ---------------------------------------
    bpp = 3 if '8' in info[0]['pixel_format'] and 'Bayer' not in info[0]['pixel_format'] else 1
    if 'RGB' in info[0]['pixel_format']:
        bpp = 3
    frame_MB = info[0]['width'] * info[0]['height'] * bpp / 1e6
    set_MB = frame_MB * len(devices)
    observed_MBps = set_MB * rate_mean
    BUDGET_MBPS = 250.0
    bound = BUDGET_MBPS / set_MB
    overhead_pct = 100.0 * (1.0 - rate_mean / bound)

    json.dump({'cameras': info, 'elapsed_s': elapsed, 'sets': n_sets,
               'set_times': set_times, 'per_camera_frames': per_cam,
               'incomplete_sets': incomplete},
              io.open('acquisition_raw.json', 'w'), indent=1)

    med = statistics.median(inst)
    p5, p95 = inst[int(0.05 * len(inst))], inst[int(0.95 * len(inst))]

    rpt = []
    w = rpt.append
    w("# Acquisition rate measurement\n")
    w("Window %.1f s, %d cameras, %s at %d x %d.\n"
      % (elapsed, len(devices), info[0]['pixel_format'],
         info[0]['width'], info[0]['height']))
    w("| quantity | value |")
    w("|---|---|")
    w("| complete frame sets | %d |" % n_sets)
    w("| mean rate | %.2f sets/s |" % rate_mean)
    w("| median instantaneous | %.2f sets/s |" % med)
    w("| 5th--95th percentile | %.2f--%.2f sets/s |" % (p5, p95))
    w("| incomplete sets | %d |" % incomplete)
    w("| per-camera frames | %s |" % ', '.join(map(str, per_cam)))
    w("| bytes per frame | %.2f MB |" % frame_MB)
    w("| bytes per set | %.2f MB |" % set_MB)
    w("| payload moved | %.1f MB/s |" % observed_MBps)
    w("| payload bound at %d MB/s | %.2f sets/s |" % (BUDGET_MBPS, bound))
    w("| shortfall against bound | %.1f %% |" % overhead_pct)
    w("\n## Sentence for Section 5.5\n")
    w("Sustained acquisition was measured over %.0f\\,s of continuous"
      % elapsed)
    w("streaming: %d complete four-camera sets, a mean of" % n_sets)
    w("%.2f\\,sets per second with a median instantaneous rate of"
      % rate_mean)
    w("%.2f and a 5th-to-95th percentile band of %.2f to %.2f."
      % (med, p5, p95))
    w("At %.2f\\,MB per synchronised set this moves %.0f\\,MB/s,"
      % (set_MB, observed_MBps))
    w("against a payload bound of %.2f\\,sets per second on the"
      % bound)
    w("%d\\,MB/s budget --- a shortfall of %.0f\\,\\%%, attributable to"
      % (BUDGET_MBPS, overhead_pct))
    w("protocol and scheduling overhead not separated here.")
    if incomplete:
        w("\n%d incomplete sets were discarded." % incomplete)

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(rpt) + "\n")
    print("\n".join(rpt[:20]), file=sys.stderr)
    print("\nwritten: %s and acquisition_raw.json" % a.out, file=sys.stderr)


if __name__ == '__main__':
    main()