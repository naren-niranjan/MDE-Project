#!/usr/bin/env python3
"""
capture_gt.py

Capture reference sets for depth_align.py: four cameras, one ChArUco board,
one DA3 inference per capture, with everything the alignment needs written to
disk.

Why a separate tool
-------------------
The correction must be fitted on depths produced by the production inference
path, because DA3's scale depends on the framing and on the batch a view was
run in. A correction fitted on single-image inference, or at a different
process_res, does not transfer. This file therefore imports da3_stream rather
than reimplementing any of it: the rectification plan, the camera
configuration, the acquisition threads and the inference call are literally
the same code objects. Keep both files in the same directory.

What it adds over da3_stream.py
-------------------------------
1. It saves the rectified image per view. The alignment front end solves the
   board pose per capture per camera, so it needs the pixels, not only the
   depth. da3_stream writes depth, K and E but no images.

2. It checks the board before spending an inference. Detection runs on all
   four views on the capture keypress, and by default a capture is refused
   unless every camera sees enough corners. Discovering afterwards that the
   top camera missed the board in nine captures out of fifteen is an
   expensive way to find out.

3. It writes a manifest recording model, process_res, mode and rectification
   settings, so depth_align.py can refuse a set assembled from mixed
   inference configurations.

On board placement
------------------
Do not level the board or worry about its height being exactly nominal. The
pose is solved per capture, so tilt and placement scatter are measured rather
than assumed, and the riser heights serve only as a cross-check. What does
matter is that all four cameras see the board, which is what the gating is
for, and that you move it between captures to cover each camera's field.

Example
-------
Run once per height, pressing space for each board placement:

    python capture_gt.py --label deck   --nominal-mm 0   --out-dir runs/gt_deck
    python capture_gt.py --label riser1 --nominal-mm 130 --out-dir runs/gt_riser1
    python capture_gt.py --label riser  --nominal-mm 218 --out-dir runs/gt_riser
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import da3_stream as ds
except ImportError as exc:
    raise SystemExit(f"cannot import da3_stream.py from this directory: {exc}")

try:
    import board_gt as bg
except ImportError as exc:
    raise SystemExit(f"cannot import board_gt.py from this directory: {exc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Capture four-camera ChArUco reference sets for "
                    "depth_align.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--calib-dir", type=Path, default=ds.DEFAULT_CALIB)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", required=True,
                    help="name for this height, e.g. deck, riser1, riser")
    ap.add_argument("--nominal-mm", type=float, required=True,
                    help="nominal height of the supporting surface above the "
                         "deck; recorded for cross-checking, never used as a "
                         "fitting input")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--expect-lens", default=ds.DEFAULT_LENS)

    # acquisition, mirroring da3_stream so the paths cannot diverge
    ap.add_argument("--width", type=int, default=2448)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--pixel-format", default="RGB8",
                    choices=sorted(ds.PIXEL_FORMATS))
    ap.add_argument("--bayer-code", default="COLOR_BayerRG2RGB")
    ap.add_argument("--exposure-us", type=float, default=20000.0)
    ap.add_argument("--gain-db", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--link-budget-mb-s", type=float, default=250.0)
    ap.add_argument("--no-link-limit", dest="link_limit",
                    action="store_false", default=True)
    ap.add_argument("--num-buffers", type=int, default=6)
    ap.add_argument("--buffer-timeout-ms", type=int, default=2000)
    ap.add_argument("--throughput-limit", type=int, default=0)
    ap.add_argument("--ptp", action="store_true")
    ap.add_argument("--max-sync-ms", type=float, default=-1.0)
    ap.add_argument("--use-device-timestamp", action="store_true")
    ap.add_argument("--start-timeout", type=float, default=30.0)

    # inference, which must match production exactly
    ap.add_argument("--model", default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--align-ext-scale", dest="align_ext_scale",
                    action="store_true", default=True)
    ap.add_argument("--no-align-ext-scale", dest="align_ext_scale",
                    action="store_false")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rect-mode", choices=["common", "roi", "full"],
                    default="common")
    ap.add_argument("--undistort", dest="undistort", action="store_true",
                    default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")

    # board
    ap.add_argument("--squares-x", type=int, default=12)
    ap.add_argument("--squares-y", type=int, default=9)
    ap.add_argument("--square-mm", type=float, default=60.0)
    ap.add_argument("--marker-mm", type=float, default=47.0)
    ap.add_argument("--dictionary", default="DICT_5X5_250")
    ap.add_argument("--legacy", dest="legacy", action="store_true", default=True)
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")
    ap.add_argument("--min-corners", type=int, default=30,
                    help="corners a view must detect before it counts as "
                         "seeing the board")
    ap.add_argument("--no-require-all", dest="require_all",
                    action="store_false", default=True,
                    help="allow a capture even when a camera misses the board")
    ap.add_argument("--detect-scale", type=float, default=0.5,
                    help="downscale used for the live visibility check; the "
                         "capture check always runs at full resolution")
    ap.add_argument("--detect-every", type=float, default=1.5,
                    help="seconds between live visibility checks")

    ap.add_argument("--preview-width", type=int, default=560)
    ap.add_argument("--status-interval", type=float, default=2.0)
    ap.add_argument("--save-clouds", action="store_true",
                    help="also write per-camera point clouds, for eyeballing")
    return ap.parse_args()


def count_corners(image, board, adict, scale=1.0):
    """Corner count only. Pose is solved later, per capture, by depth_align."""
    img = image
    if scale != 1.0:
        img = cv2.resize(image, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    obj, pts = bg.detect_board(img, board, adict)
    return 0 if obj is None else len(obj)


def main() -> int:
    args = parse_args()
    names = list(args.cameras)
    for n in names:
        if n not in ds.CAMERAS:
            raise SystemExit(f"unknown camera {n!r}")

    # ---- calibration, reusing da3_stream's checks ----------------------
    intr = {n: ds.load_intrinsics(args.calib_dir, n) for n in names}
    ext = ds.load_extrinsics(args.calib_dir, names, args.reference)
    lens = ds.check_lens(intr, ext, names, args.expect_lens)
    ds.check_serials(intr, ext, names)
    for n in names:
        if intr[n]["size"] != (args.width, args.height):
            raise SystemExit(f"{n}: calibrated at {intr[n]['size']}, "
                             f"configured for {(args.width, args.height)}")

    plan = ds.build_rect_plan(intr, names, (args.width, args.height),
                              args.rect_mode, args.undistort)
    Ks_rect = {n: plan[n]["K"] for n in names}
    Es = np.stack([ext[n]["E"] for n in names])
    link = ds.plan_link(args, len(names))
    ds.report_calibration(intr, ext, names, lens, plan, args, link)

    if args.max_sync_ms < 0:
        args.max_sync_ms = round(0.75 * 1000.0 / max(link["fps"], 1e-6), 1)
        print(f"[sync ] --max-sync-ms auto set to {args.max_sync_ms:.1f} ms")

    board, adict = bg.make_board(args)
    print(f"[board] {args.squares_x}x{args.squares_y}  square {args.square_mm} "
          f"mm  marker {args.marker_mm} mm  {args.dictionary}  "
          f"legacy={args.legacy}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from arena_api.system import system

    infos = system.device_infos
    if not infos:
        raise SystemExit("no cameras found")
    found = {i["serial"]: i for i in infos}
    missing = [n for n in names if ds.CAMERAS[n] not in found]
    if missing:
        raise SystemExit(f"missing camera(s): {', '.join(missing)}")
    devices = system.create_device([found[ds.CAMERAS[n]] for n in names])

    stop = threading.Event()
    workers = []
    captured = 0
    rejected_board = 0
    config_log = {}

    try:
        for n, dev in zip(names, devices):
            log = []
            ds.configure(dev, args, link, log)
            config_log[n] = log
        if args.ptp:
            status, locked = ds.wait_for_ptp(devices, names)
            print(f"[ptp  ] {status}" + ("" if locked else "   NOT LOCKED"))
        for n, dev in zip(names, devices):
            dev.start_stream(args.num_buffers)
            workers.append(ds.CameraWorker(n, dev, args, plan[n], stop))
        for w in workers:
            w.start()

        print(f"[model] loading {args.model}")
        model = ds.load_model(args.model, args.device)

        first = ds.wait_for_first_frames(workers, args.start_timeout)
        if first is None:
            raise SystemExit("cameras did not deliver a first frame")
        Ks_in = np.stack([ds.scale_K(Ks_rect[n], args.input_scale)
                          for n in names])
        for i in range(max(0, args.warmup)):
            out = ds.run_inference(model, first, Ks_in, Es, args)
            print(f"[warm ] pass {i + 1}  {out['ms']:.0f} ms")

        window = "capture_gt"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("\n[run  ] space captures, q or escape quits")
        print("[run  ] move the board between captures so each camera sees it "
              "in different parts of its field")

        last_seqs = [0] * len(workers)
        counts = {n: -1 for n in names}
        t_detect = 0.0

        while True:
            snap = [w.latest() for w in workers]
            if any(s is None for s in snap):
                if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                    break
                continue

            spread_ms = (max(s[1] for s in snap) - min(s[1] for s in snap)) * 1e3
            frames = [s[0] for s in snap]
            last_seqs = [s[3] for s in snap]

            now = time.perf_counter()
            if now - t_detect >= args.detect_every:
                counts = {n: count_corners(f, board, adict, args.detect_scale)
                          for n, f in zip(names, frames)}
                t_detect = now

            seen = [n for n in names if counts[n] >= args.min_corners]
            status = "  ".join(
                f"{n}:{counts[n] if counts[n] >= 0 else '?'}"
                + ("+" if counts[n] >= args.min_corners else "-")
                for n in names)
            info = [f"{args.label}  nominal {args.nominal_mm:.0f} mm   "
                    f"captured {captured}   refused {rejected_board}",
                    f"corners  {status}   sync {spread_ms:.0f} ms"]
            colour = (0, 255, 0) if len(seen) == len(names) else (0, 165, 255)
            view = ds.with_banner(ds.mosaic(frames, args.preview_width),
                                  info, colour)
            cv2.imshow(window, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue

            # ---- full-resolution visibility check --------------------
            full = {n: count_corners(f, board, adict, 1.0)
                    for n, f in zip(names, frames)}
            short = [n for n in names if full[n] < args.min_corners]
            print(f"[check] corners " + "  ".join(f"{n}:{full[n]}" for n in names))
            if short and args.require_all:
                rejected_board += 1
                print(f"[skip ] {', '.join(short)} did not see the board; "
                      f"reposition and try again, or pass --no-require-all")
                continue
            if args.max_sync_ms > 0 and spread_ms > args.max_sync_ms:
                print(f"[skip ] sync spread {spread_ms:.0f} ms exceeds "
                      f"{args.max_sync_ms:.0f} ms; hold still and retry")
                continue

            # ---- inference and save ----------------------------------
            if args.input_scale != 1.0:
                s = args.input_scale
                frames_in = [cv2.resize(f, (int(round(f.shape[1] * s)),
                                            int(round(f.shape[0] * s))),
                                        interpolation=cv2.INTER_AREA)
                             for f in frames]
            else:
                frames_in = frames

            out = ds.run_inference(model, frames_in, Ks_in, Es, args)
            stem = args.out_dir / f"frame_{captured:05d}"
            stem.mkdir(exist_ok=True)

            for i, n in enumerate(names):
                # The image the model actually consumed, so the board pose is
                # solved in the same frame the depth lives in.
                cv2.imwrite(str(stem / f"image_{n}.png"),
                            cv2.cvtColor(frames_in[i], cv2.COLOR_RGB2BGR))
                np.save(stem / f"Kin_{n}.npy", Ks_in[i])
                np.save(stem / f"depth_{n}.npy", out["depth"][i])
                if out["conf"] is not None:
                    np.save(stem / f"conf_{n}.npy", out["conf"][i])
                np.save(stem / f"K_{n}.npy", out["K"][i])
                np.save(stem / f"E_{n}.npy", out["E"][i])

            meta = {
                "label": args.label,
                "nominal_mm": args.nominal_mm,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "corners": full,
                "sync_spread_ms": round(spread_ms, 2),
                "is_metric": out["is_metric"],
                "depth_shape": list(out["shape"]),
                "image_shape": [int(frames_in[0].shape[0]),
                                int(frames_in[0].shape[1])],
                "inference_ms": round(out["ms"], 2),
            }
            (stem / "meta.json").write_text(json.dumps(meta, indent=2))

            if args.save_clouds:
                try:
                    import open3d as o3d
                    for i, n in enumerate(names):
                        d = out["depth"][i].astype(np.float64)
                        ok = np.isfinite(d) & (d > 0)
                        pts = ds.points_camera(d, out["K"][i])[ok]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(
                            ds.to_world(pts, out["E"][i]))
                        o3d.io.write_point_cloud(str(stem / f"cloud_{n}.ply"), pcd)
                except ImportError:
                    print("[warn ] open3d unavailable, clouds not written")

            captured += 1
            print(f"[save ] {stem}  infer {out['ms']:.0f} ms  "
                  f"corners {min(full.values())}-{max(full.values())}")

    except KeyboardInterrupt:
        print("\ninterrupted")
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
        cv2.destroyAllWindows()

    manifest = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": args.label,
        "nominal_mm": args.nominal_mm,
        "lens_id": lens,
        "cameras": names,
        "reference": args.reference,
        "captures": captured,
        "refused_board_not_visible": rejected_board,
        # depth_align must refuse a set mixing these, because DA3's scale
        # depends on the framing and the batch a view was run in.
        "inference": {"model": args.model, "mode": args.mode,
                      "process_res": args.process_res,
                      "align_ext_scale": args.align_ext_scale,
                      "input_scale": args.input_scale,
                      "rect_mode": args.rect_mode,
                      "undistort": args.undistort},
        "rectified_sizes": {n: [plan[n]["crop"][2], plan[n]["crop"][3]]
                            for n in names},
        "board": {"squares": [args.squares_x, args.squares_y],
                  "square_mm": args.square_mm, "marker_mm": args.marker_mm,
                  "dictionary": args.dictionary, "legacy": args.legacy},
        "node_configuration": config_log,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ncaptured {captured} set(s) to {args.out_dir}")
    print(f"manifest: {args.out_dir / 'manifest.json'}")
    if captured < 5:
        print("Five placements per height is a reasonable minimum; fewer "
              "leaves the fit leaning on a small part of each field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())