#!/usr/bin/env python3
"""Live multi-camera viewer for the four Lucid Triton TRI050S-C GigE cameras.

Each selected camera is acquired in its own thread and the most recent frame
from every camera is tiled into a single mosaic window. Per-camera acquisition
timing is accumulated throughout the run and written to a diagnostic JSON file
on exit.

Display rotation
----------------
The cameras are not mounted at the same roll, so a raw mosaic shows the same
scene four different ways up. The rotation applied for display used to be a
hardcoded table. That table was written for a particular set of extrinsics,
and swapping the lenses meant touching every camera, so it is no longer
trustworthy: a mount nudged by a few degrees while the C-mount was turned
changes the roll without changing anything else visibly.

The rotation is therefore derived from extrinsics.json when --calib-dir is
given, printed so it can be checked, and only falls back to the static table
when no calibration is available. The quarter-turn count is
round(roll / 90) % 4, and if a camera's roll is far from a multiple of 90 that
is reported, because rotating it in quarter turns is then only approximate and
the mosaic will look slightly tilted for a real reason.

The rotation is DISPLAY ONLY. Frames saved with the s key are written exactly
as the camera sent them, unrotated, because everything downstream consumes the
calibrated extrinsics and would disagree with a pre-rotated image.

Keys
    q or Esc    quit
    s           save the current frame of every camera into --save-dir

Example
    python stream_cameras.py --calib-dir ../Calibration_4_5/results \
        --pixel-format RGB8 --exposure-us 15000 --gain-db 0
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from arena_api.system import system


# Serial numbers as fixed in the calibration configuration. The physical
# arrangement is a horizontal row of left, centre and right with the fourth
# camera raised above the row.
CAMERAS = {
    "left": "260505158",
    "center": "261100628",
    "top": "261100631",
    "right": "261100627",
}

REFERENCE_CAMERA = "center"

# Fallback in-plane rotation, used ONLY when no extrinsics are available.
# These came from the pre-lens-change rig and are kept solely so the viewer is
# still usable before a calibration exists; pass --calib-dir to derive them
# from the current solve instead.
FALLBACK_ROTATION_K = {"left": 3, "center": 0, "top": 1, "right": 2}

# Pixel format name -> (bytes per pixel, conversion family).
PIXEL_FORMATS = {
    "RGB8": (3, "rgb"),
    "BGR8": (3, "bgr"),
    "Mono8": (1, "mono"),
    "BayerRG8": (1, "bayer"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the four Lucid Triton cameras into a tiled viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cameras",
        default=",".join(CAMERAS),
        help="Comma separated camera names to stream, in mosaic order.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Directory holding extrinsics.json. When given, the display "
             "rotation is derived from the solved poses rather than from the "
             "static fallback table, which is stale after any remount.",
    )
    parser.add_argument("--width", type=int, default=2448, help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=2048, help="Sensor height in pixels.")
    parser.add_argument(
        "--pixel-format",
        default="RGB8",
        choices=sorted(PIXEL_FORMATS),
        help="Camera pixel format. RGB8 debayers on the camera and is the "
             "faster option; BayerRG8 moves that cost onto the host.",
    )
    parser.add_argument(
        "--bayer-code",
        default="COLOR_BayerRG2BGR",
        help="OpenCV conversion code used when --pixel-format is BayerRG8.",
    )
    parser.add_argument(
        "--exposure-us", type=float, default=15000.0,
        help="Exposure time in microseconds. Matches da3_stream.py, so what "
             "you see here is what the depth pipeline will be fed.")
    parser.add_argument(
        "--gain-db", type=float, default=0.0,
        help="Analogue gain in decibels. The old 18.7 dB default was sized "
             "for the 8 mm lens; the 12 mm f/1.8 needs far less even stopped "
             "down to f/5.6, and gain is what costs depth quality.")
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Leave exposure and gain under camera control instead of pinning "
             "them. Useful for judging how much light the scene actually has "
             "after a lens change; never use it for anything measured.",
    )
    parser.add_argument(
        "--throughput-limit",
        type=int,
        default=0,
        help="Per-camera DeviceLinkThroughputLimit in bytes per second. "
             "0 leaves the current camera setting untouched. Reduce this if "
             "incomplete buffers appear with four cameras on one link.",
    )
    parser.add_argument("--num-buffers", type=int, default=10, help="Stream buffers allocated per camera.")
    parser.add_argument("--buffer-timeout-ms", type=int, default=2000, help="get_buffer timeout per frame.")
    parser.add_argument("--tile-width", type=int, default=612, help="Width of each mosaic tile in pixels.")
    parser.add_argument("--display-hz", type=float, default=30.0, help="Mosaic refresh rate.")
    parser.add_argument(
        "--rot-sign",
        type=int,
        default=1,
        choices=(1, -1),
        help="Sign applied to the per-camera rotation count when displaying. "
             "The mapping from a solved roll to a screen rotation depends on "
             "the frame convention, so this stays empirical: flip it if the "
             "tiles come out upside down.",
    )
    parser.add_argument("--no-rotate", action="store_true", help="Disable display rotation entirely.")
    parser.add_argument("--no-display", action="store_true", help="Run headless and report timing only.")
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds. 0 runs until quit.")
    parser.add_argument("--fps-window", type=int, default=120, help="Frames retained for the rolling rate estimate.")
    parser.add_argument("--save-dir", default="stream_snapshots", help="Destination for frames saved with the s key.")
    parser.add_argument("--json", default="stream_diagnostics.json", help="Diagnostic summary written on exit.")
    return parser.parse_args()


# === display rotation =======================================================

def rotation_from_extrinsics(calib_dir, names, reference=REFERENCE_CAMERA):
    """Quarter-turn counts derived from the solved poses.

    extrinsics.json stores X_target = R @ X_reference + t, so R maps the
    reference camera's frame into each target's. The reference camera's image
    right direction, seen in a target's image plane, is (R[0,0], R[1,0]); the
    angle of that vector is the in-plane roll between the two.

    Returns (rotation_k, roll_deg, note). rotation_k is empty if the file is
    unavailable, and the caller falls back to the static table.
    """
    if calib_dir is None:
        return {}, {}, "no --calib-dir given"
    path = Path(calib_dir) / "extrinsics.json"
    if not path.exists():
        return {}, {}, f"{path} not found"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {}, {}, f"{path} could not be read ({exc})"

    rot, rolls = {}, {}
    for n in names:
        entry = data.get(n)
        if not isinstance(entry, dict) or "R" not in entry:
            continue
        R = np.asarray(entry["R"], dtype=float).reshape(3, 3)
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        rolls[n] = roll
        rot[n] = int(round(roll / 90.0)) % 4
    if not rot:
        return {}, {}, f"{path} holds no usable rotations"
    lens = {v.get("lens_id") for v in data.values() if isinstance(v, dict)}
    lens = {l for l in lens if l}
    note = f"derived from {path}"
    if lens:
        note += f" (lens {', '.join(sorted(lens))})"
    return rot, rolls, note


def report_rotation(rot, rolls, note, names):
    print(f"Display rotation: {note}")
    for n in names:
        if n not in rot:
            print(f"  {n:>7}: not solved, falling back to "
                  f"{FALLBACK_ROTATION_K.get(n, 0)}")
            continue
        residual = rolls[n] - 90.0 * rot[n]
        residual = (residual + 180.0) % 360.0 - 180.0
        flag = ""
        if abs(residual) > 15.0:
            flag = ("   ** far from a right angle; quarter turns cannot "
                    "square this view and the tile will look tilted **")
        elif abs(residual) > 5.0:
            flag = "   (mount is a few degrees off square)"
        print(f"  {n:>7}: roll {rolls[n]:+7.1f} deg -> rotate {rot[n]} x 90 "
              f"(residual {residual:+5.1f} deg){flag}")


def set_node(nodemap, name: str, value, log: list) -> bool:
    """Assign a node value, recording the outcome rather than raising."""
    try:
        nodemap[name].value = value
    except Exception as exc:  # noqa: BLE001 - node availability varies by firmware
        log.append(f"{name}: not set ({exc})")
        return False
    log.append(f"{name}: {value}")
    return True


def buffer_to_array(item, bytes_per_pixel: int) -> np.ndarray:
    """Copy an Arena buffer into an owned numpy array.

    The copy is mandatory. The buffer is requeued immediately afterwards and
    the underlying memory is then reused by the acquisition engine.
    """
    total = item.width * item.height * bytes_per_pixel
    raw = (ctypes.c_ubyte * total).from_address(ctypes.addressof(item.pbytes))
    if bytes_per_pixel > 1:
        shape = (item.height, item.width, bytes_per_pixel)
    else:
        shape = (item.height, item.width)
    return np.ndarray(buffer=raw, dtype=np.uint8, shape=shape).copy()


def to_bgr(frame: np.ndarray, family: str, bayer_code: int) -> np.ndarray:
    if family == "bgr":
        return frame
    if family == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if family == "mono":
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame, bayer_code)


class CameraWorker(threading.Thread):
    """Acquire from one camera and publish the most recent frame."""

    def __init__(self, name: str, device, args: argparse.Namespace, stop: threading.Event):
        super().__init__(name=f"acq-{name}", daemon=True)
        self.cam_name = name
        self.device = device
        self.args = args
        self.stop = stop

        self.bytes_per_pixel, self.family = PIXEL_FORMATS[args.pixel_format]
        self.bayer_code = getattr(cv2, args.bayer_code, cv2.COLOR_BayerRG2BGR)

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_id = 0

        self.wait_ms: list[float] = []
        self.convert_ms: list[float] = []
        self.stamps = deque(maxlen=args.fps_window)
        self.frames = 0
        self.incomplete = 0
        self.timeouts = 0
        self.errors: list[str] = []
        self.levels = deque(maxlen=30)

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest is None:
                return None, 0
            return self._latest, self._latest_id

    def mean_level(self):
        return statistics.fmean(self.levels) if self.levels else float("nan")

    def run(self) -> None:
        while not self.stop.is_set():
            item = None
            try:
                t0 = time.perf_counter()
                item = self.device.get_buffer(timeout=self.args.buffer_timeout_ms)
                t1 = time.perf_counter()

                if item.is_incomplete:
                    self.incomplete += 1
                    continue

                raw = buffer_to_array(item, self.bytes_per_pixel)
                bgr = to_bgr(raw, self.family, self.bayer_code)
                t2 = time.perf_counter()

                self.wait_ms.append((t1 - t0) * 1e3)
                self.convert_ms.append((t2 - t1) * 1e3)
                self.stamps.append(t2)
                self.frames += 1
                # Cheap exposure check on a decimated copy, so a shoot that is
                # several stops under after a lens or aperture change is
                # obvious in the header rather than after the fact.
                if self.frames % 10 == 0:
                    self.levels.append(float(bgr[::16, ::16].mean()))

                with self._lock:
                    self._latest = bgr
                    self._latest_id += 1
            except Exception as exc:  # noqa: BLE001 - timeouts surface as generic errors
                self.timeouts += 1
                if len(self.errors) < 10:
                    self.errors.append(repr(exc))
                if self.timeouts > 50:
                    self.errors.append("too many consecutive acquisition failures, thread exiting")
                    break
            finally:
                if item is not None:
                    try:
                        self.device.requeue_buffer(item)
                    except Exception:  # noqa: BLE001
                        pass

    def rolling_fps(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0

    def summary(self, elapsed: float) -> dict:
        def stats(values: list[float]) -> dict:
            if not values:
                return {"mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
            ordered = sorted(values)
            idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
            return {
                "mean_ms": round(statistics.fmean(values), 3),
                "median_ms": round(statistics.median(values), 3),
                "p95_ms": round(ordered[idx], 3),
                "max_ms": round(ordered[-1], 3),
            }

        return {
            "frames": self.frames,
            "incomplete_buffers": self.incomplete,
            "acquisition_failures": self.timeouts,
            "mean_fps": round(self.frames / elapsed, 3) if elapsed > 0 else 0.0,
            "rolling_fps_at_exit": round(self.rolling_fps(), 3),
            "mean_level": (round(self.mean_level(), 1)
                           if self.levels else None),
            "buffer_wait": stats(self.wait_ms),
            "host_convert": stats(self.convert_ms),
            "errors": self.errors,
        }


def configure(device, name: str, args: argparse.Namespace) -> list:
    """Apply acquisition settings and return a log of what was accepted."""
    log: list[str] = []
    nodemap = device.nodemap

    # Any trigger state left behind by a crashed calibration run makes every
    # free-run grab here time out, so it is cleared first.
    try:
        nodemap["TriggerSelector"].value = "FrameStart"
        nodemap["TriggerMode"].value = "Off"
        log.append("TriggerMode: Off")
    except Exception as exc:  # noqa: BLE001
        log.append(f"TriggerMode: not reset ({exc})")

    set_node(nodemap, "Width", args.width, log)
    set_node(nodemap, "Height", args.height, log)
    set_node(nodemap, "PixelFormat", args.pixel_format, log)
    set_node(nodemap, "AcquisitionMode", "Continuous", log)

    if args.auto_exposure:
        set_node(nodemap, "ExposureAuto", "Continuous", log)
        set_node(nodemap, "GainAuto", "Continuous", log)
    else:
        set_node(nodemap, "ExposureAuto", "Off", log)
        set_node(nodemap, "ExposureTime", args.exposure_us, log)
        set_node(nodemap, "GainAuto", "Off", log)
        set_node(nodemap, "Gain", args.gain_db, log)

    if args.throughput_limit > 0:
        set_node(nodemap, "DeviceLinkThroughputLimitMode", "On", log)
        set_node(nodemap, "DeviceLinkThroughputLimit", args.throughput_limit, log)

    stream = device.tl_stream_nodemap
    set_node(stream, "StreamAutoNegotiatePacketSize", True, log)
    set_node(stream, "StreamPacketResendEnable", True, log)
    # NewestOnly keeps the viewer on live frames rather than draining a backlog.
    set_node(stream, "StreamBufferHandlingMode", "NewestOnly", log)

    return log


def build_mosaic(workers: list[CameraWorker], args: argparse.Namespace,
                 rotation_k: dict) -> np.ndarray | None:
    tiles = []
    for worker in workers:
        frame, _ = worker.latest()
        if frame is None:
            tile = np.zeros((args.tile_width * 3 // 4, args.tile_width, 3), dtype=np.uint8)
            label = f"{worker.cam_name}: waiting"
        else:
            if not args.no_rotate:
                k = (rotation_k.get(worker.cam_name, 0) * args.rot_sign) % 4
                if k:
                    frame = np.ascontiguousarray(np.rot90(frame, k))
            scale = args.tile_width / frame.shape[1]
            tile = cv2.resize(frame, (args.tile_width, max(1, int(round(frame.shape[0] * scale)))),
                              interpolation=cv2.INTER_AREA)
            level = worker.mean_level()
            label = (f"{worker.cam_name}  {worker.rolling_fps():.1f} fps  "
                     f"n={worker.frames}  lvl {level:.0f}"
                     if np.isfinite(level) else
                     f"{worker.cam_name}  {worker.rolling_fps():.1f} fps  "
                     f"n={worker.frames}")
            if worker.incomplete:
                label += f"  incomplete={worker.incomplete}"
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        return None

    tile_h = max(t.shape[0] for t in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < tile_h:
            pad = np.zeros((tile_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)

    cols = 2 if len(padded) > 1 else 1
    rows = []
    for start in range(0, len(padded), cols):
        chunk = padded[start:start + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def save_snapshot(workers: list[CameraWorker], args: argparse.Namespace) -> Path:
    """Write the current frame of every camera, UNROTATED.

    The display rotation is not applied here on purpose. These frames feed
    measure_deck.py and the depth pipeline, both of which use the calibrated
    extrinsics; a pre-rotated image would disagree with the pose that
    describes it.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = Path(args.save_dir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    for worker in workers:
        frame, _ = worker.latest()
        if frame is not None:
            cv2.imwrite(str(out / f"{worker.cam_name}.png"), frame)
    return out


def main() -> int:
    args = parse_args()

    names = [n.strip() for n in args.cameras.split(",") if n.strip()]
    unknown = [n for n in names if n not in CAMERAS]
    if unknown:
        print(f"Unknown camera name(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Known names: {', '.join(CAMERAS)}", file=sys.stderr)
        return 2

    wanted = {CAMERAS[n]: n for n in names}

    solved_rot, rolls, rot_note = rotation_from_extrinsics(args.calib_dir, names)
    rotation_k = {n: solved_rot.get(n, FALLBACK_ROTATION_K.get(n, 0))
                  for n in names}
    if solved_rot:
        report_rotation(solved_rot, rolls, rot_note, names)
    else:
        print(f"Display rotation: {rot_note}; using the STATIC fallback table "
              f"{ {n: rotation_k[n] for n in names} }.")
        print("  That table predates the lens change. Pass --calib-dir so the "
              "rotation follows the current solve.")
    print()

    t_start = time.perf_counter()
    infos = system.device_infos
    if not infos:
        print("No cameras found. Check the Arena shared library path and the "
              "camera network interface.", file=sys.stderr)
        return 1

    found = {info["serial"]: info for info in infos}
    missing = [f"{name} ({serial})" for serial, name in wanted.items() if serial not in found]
    if missing:
        print(f"Missing camera(s): {', '.join(missing)}", file=sys.stderr)
        print(f"Detected serials: {', '.join(sorted(found))}", file=sys.stderr)
        return 1

    selected_infos = [found[CAMERAS[n]] for n in names]
    devices = system.create_device(selected_infos)
    t_open = time.perf_counter()

    stop = threading.Event()
    workers: list[CameraWorker] = []
    config_log: dict[str, list] = {}
    exit_code = 0
    t_configured = t_open
    t_loop = t_open
    elapsed = 0.0

    try:
        for name, device in zip(names, devices):
            config_log[name] = configure(device, name, args)
            device.start_stream(args.num_buffers)
            workers.append(CameraWorker(name, device, args, stop))
        t_configured = time.perf_counter()

        for worker in workers:
            worker.start()

        print(f"Streaming {len(workers)} camera(s): {', '.join(names)}")
        print(f"Format {args.pixel_format} at {args.width}x{args.height}")
        if args.auto_exposure:
            print("Auto exposure is ON, so nothing measured from these frames "
                  "is comparable between them.")
        else:
            print(f"Exposure {args.exposure_us:.0f} us, gain "
                  f"{args.gain_db:.1f} dB (pinned)")
        if args.no_display:
            print("Headless mode. Press Ctrl+C to stop.")
        else:
            print("Press q to quit, s to save an unrotated snapshot.")

        window = "Triton four-camera stream"
        if not args.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        period = 1.0 / args.display_hz if args.display_hz > 0 else 0.0
        t_loop = time.perf_counter()
        last_report = t_loop
        levels_warned = False

        while True:
            now = time.perf_counter()
            if args.duration > 0 and (now - t_loop) >= args.duration:
                break

            # A shoot several stops under after a lens or aperture change looks
            # like a dark preview and nothing else, so say it once explicitly.
            if not levels_warned and all(w.levels for w in workers):
                levels_warned = True
                dark = [w.cam_name for w in workers if w.mean_level() < 30]
                if dark:
                    print(f"[WARN] very dark frames from {', '.join(dark)} "
                          f"(mean below 30/255). Open the iris, add light, or "
                          f"raise --exposure-us; do not fix this with gain if "
                          f"the frames are going into the depth pipeline.")

            if args.no_display:
                time.sleep(0.5)
                if now - last_report >= 2.0:
                    rates = "  ".join(f"{w.cam_name} {w.rolling_fps():5.1f}" for w in workers)
                    print(f"[{now - t_loop:7.1f} s] {rates}")
                    last_report = now
            else:
                mosaic = build_mosaic(workers, args, rotation_k)
                if mosaic is not None:
                    cv2.imshow(window, mosaic)
                key = cv2.waitKey(max(1, int(period * 1e3))) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    print(f"Saved snapshot to {save_snapshot(workers, args)}")

            if all(not w.is_alive() for w in workers):
                print("All acquisition threads have stopped.", file=sys.stderr)
                exit_code = 1
                break

        elapsed = time.perf_counter() - t_loop

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_loop
        print("\nInterrupted.")
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=5.0)
        for device in devices:
            try:
                device.stop_stream()
            except Exception:  # noqa: BLE001
                pass
        system.destroy_device()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = max(elapsed, 1e-9)
    diagnostics = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {
            "cameras": {n: CAMERAS[n] for n in names},
            "calib_dir": str(args.calib_dir) if args.calib_dir else None,
            "width": args.width,
            "height": args.height,
            "pixel_format": args.pixel_format,
            "exposure_us": None if args.auto_exposure else args.exposure_us,
            "gain_db": None if args.auto_exposure else args.gain_db,
            "auto_exposure": args.auto_exposure,
            "num_buffers": args.num_buffers,
            "throughput_limit_bps": args.throughput_limit or None,
            "display": not args.no_display,
        },
        "display_rotation": {
            "source": rot_note if solved_rot else "static fallback table",
            "rotation_k": rotation_k,
            "roll_deg": {n: round(v, 2) for n, v in rolls.items()},
            "rot_sign": args.rot_sign,
            "applied_to_saved_frames": False,
        },
        "startup_timing": {
            "enumerate_and_open_s": round(t_open - t_start, 3),
            "configure_and_start_s": round(t_configured - t_open, 3),
        },
        "stream_duration_s": round(elapsed, 3),
        "node_configuration": config_log,
        "per_camera": {w.cam_name: w.summary(elapsed) for w in workers},
    }

    Path(args.json).write_text(json.dumps(diagnostics, indent=2))
    print(f"\nDiagnostics written to {args.json}")
    for name, entry in diagnostics["per_camera"].items():
        print(f"  {name:>7}: {entry['frames']:6d} frames  {entry['mean_fps']:6.2f} fps  "
              f"incomplete {entry['incomplete_buffers']:4d}  failures {entry['acquisition_failures']:4d}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())