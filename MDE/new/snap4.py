#!/usr/bin/env python3
"""
snap4.py — one synchronised, full-resolution, UNRECTIFIED frame per camera,
for the subset study. Sits beside da3_stream.py and imports its camera
configuration so the frames are acquired exactly as the streamer acquires them.

    python3 snap4.py --out captures/scene_a

Why raw and not the streamer's proc_<cam>.png: those are the images DA3 saw,
already rectified and downsampled to process_res. da3_offline.py rectifies from
the calibration itself, so it needs the same full-resolution input the
calibration was shot on. Rectifying twice shifts the principal point and the
intrinsics no longer describe the image.

It grabs several synchronised sets, keeps the tightest, and reports clipping
per camera BEFORE you scan -- so you can fix exposure while the parcels are
still in place rather than discovering it after the scan is taken.
"""
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import da3_stream as ds


def clipping(rgb):
    """Fraction of pixels saturated in any channel, and the mean level."""
    a = rgb if rgb.ndim == 3 else rgb[..., None]
    hi = float((a.max(axis=2) >= 254).mean())
    lo = float((a.max(axis=2) <= 2).mean())
    lap = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                              cv2.CV_64F).var())
    return hi, lo, float(a.mean()), lap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", type=Path, default=ds.DEFAULT_CALIB)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--expect-lens", default=ds.DEFAULT_LENS)
    ap.add_argument("--width", type=int, default=2448)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--pixel-format", default="RGB8",
                    choices=sorted(ds.PIXEL_FORMATS))
    ap.add_argument("--bayer-code", default="COLOR_BayerRG2RGB")
    ap.add_argument("--exposure-us", type=float, default=19000.0)
    ap.add_argument("--gain-db", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--link-budget-mb-s", type=float, default=250.0)
    ap.add_argument("--no-link-limit", dest="link_limit",
                    action="store_false", default=True)
    ap.add_argument("--throughput-limit", type=int, default=0)
    ap.add_argument("--num-buffers", type=int, default=6)
    ap.add_argument("--buffer-timeout-ms", type=int, default=2000)
    ap.add_argument("--ptp", action="store_true")
    ap.add_argument("--tries", type=int, default=12,
                    help="synchronised sets to consider; the tightest is kept")
    ap.add_argument("--start-timeout", type=float, default=30.0)
    ap.add_argument("--pattern", default="{cam}.png",
                    help="output filename per camera. Check what "
                         "da3_offline.py --images expects and match it.")
    ap.add_argument("--also-rectified", action="store_true",
                    help="additionally write rect_<cam>.png, for eyeballing "
                         "only -- not the study input")
    a = ap.parse_args()

    names = list(a.cameras)
    bad = [n for n in names if n not in ds.CAMERAS]
    if bad:
        raise SystemExit(f"unknown camera(s): {bad}")

    intr = {n: ds.load_intrinsics(a.calib_dir, n) for n in names}
    ext = ds.load_extrinsics(a.calib_dir, names, a.reference)
    lens = ds.check_lens(intr, ext, names, a.expect_lens)
    ds.check_serials(intr, ext, names)
    for n in names:
        if intr[n]["size"] != (a.width, a.height):
            raise SystemExit(f"{n}: calibrated at {intr[n]['size']}, "
                             f"capturing {(a.width, a.height)}")
    print(f"[lens ] {lens}")

    # enabled=False -> workers publish the raw frame, no remap, no crop
    raw_plan = ds.build_rect_plan(intr, names, (a.width, a.height),
                                  "full", False)
    link = ds.plan_link(a, len(names))
    print(f"[link ] {link['frame_mb']:.2f} MB/frame, running "
          f"{link['fps']:.1f} fps ({link['fps_source']})")

    from arena_api.system import system
    infos = system.device_infos
    found = {i["serial"]: i for i in infos}
    missing = [n for n in names if ds.CAMERAS[n] not in found]
    if missing:
        raise SystemExit(f"missing camera(s): {missing}; detected {sorted(found)}")
    devices = system.create_device([found[ds.CAMERAS[n]] for n in names])

    stop = threading.Event()
    workers = []
    try:
        for n, dev in zip(names, devices):
            ds.configure(dev, a, link, [])
        if a.ptp:
            st, locked = ds.wait_for_ptp(devices, names)
            print(f"[ptp  ] {st}" + ("" if locked else "   NOT LOCKED"))
        for n, dev in zip(names, devices):
            dev.start_stream(a.num_buffers)
            workers.append(ds.CameraWorker(n, dev, a, raw_plan[n], stop))
        for w in workers:
            w.start()

        if ds.wait_for_first_frames(workers, a.start_timeout) is None:
            raise SystemExit("cameras did not deliver a first frame")

        best, best_spread, seen = None, float("inf"), set()
        t0 = time.perf_counter()
        while len(seen) < a.tries and time.perf_counter() - t0 < 30.0:
            snap = [w.latest() for w in workers]
            if any(s is None for s in snap):
                time.sleep(0.01)
                continue
            key = tuple(s[3] for s in snap)
            if key in seen:
                time.sleep(0.005)
                continue
            seen.add(key)
            spread = (max(s[1] for s in snap) - min(s[1] for s in snap)) * 1e3
            if spread < best_spread:
                best_spread = spread
                best = [np.array(s[0], copy=True) for s in snap]
        if best is None:
            raise SystemExit("no synchronised set assembled")
        print(f"[sync ] kept the tightest of {len(seen)} sets: "
              f"{best_spread:.1f} ms spread")
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=5.0)
        for d in devices:
            try:
                d.stop_stream()
            except Exception:  # noqa: BLE001
                pass
        try:
            system.destroy_device()
        except Exception:  # noqa: BLE001
            pass

    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "lens_id": lens, "cameras": names, "reference": a.reference,
            "rectified": False, "image_size": [a.width, a.height],
            "exposure_us": a.exposure_us, "gain_db": a.gain_db,
            "pixel_format": a.pixel_format,
            "sync_spread_ms": round(best_spread, 3),
            "calib_dir": str(a.calib_dir), "quality": {}}

    print(f"\n{'cam':8s} {'clipped':>8s} {'black':>7s} {'mean':>6s} {'lapvar':>8s}")
    worst = 0.0
    for n, img in zip(names, best):
        hi, lo, mean, lap = clipping(img)
        worst = max(worst, hi)
        flag = "   <- RECAPTURE, drop ~1.5 stops" if hi > 0.15 else ""
        print(f"{n:8s} {100*hi:7.1f}% {100*lo:6.1f}% {mean:6.1f} {lap:8.1f}{flag}")
        meta["quality"][n] = {"clipped_frac": round(hi, 5),
                              "black_frac": round(lo, 5),
                              "mean_level": round(mean, 2),
                              "laplacian_var": round(lap, 1)}
        p = a.out / a.pattern.format(cam=n)
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if a.also_rectified:
            rp = ds.build_rect_plan(intr, [n], (a.width, a.height),
                                    "common", True)[n]
            r = cv2.remap(img, rp["map1"], rp["map2"], cv2.INTER_LINEAR)
            x, y, w, h = rp["crop"]
            cv2.imwrite(str(a.out / f"rect_{n}.png"),
                        cv2.cvtColor(r[y:y+h, x:x+w], cv2.COLOR_RGB2BGR))

    (a.out / "snapshot.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(names)} raw frames to {a.out}")
    if worst > 0.15:
        print("\n*** clipping above 15%. Lower --exposure-us by about a third "
              "and re-run\n    BEFORE taking the scan. A blown parcel lid "
              "costs a whole resolved layer.")
    else:
        print("\nexposure ok. Take the scan NOW, without moving anything.")


if __name__ == "__main__":
    main()