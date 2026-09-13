#!/usr/bin/env python3
"""
box_scene.py

Offline driver: segment a stored 3D scene and export the parcels as geometry.

The construction of the cuboids lives in box_geometry.py, which da3_stream.py
also imports, so the boxes written here and the boxes written by a live run are
built by the same code and cannot drift apart.

The FILTERING is imported from da3_stream.py for the same reason, and that is a
change from earlier revisions of this file. It used to back-project every pixel
with finite positive depth, while the streamer dropped low-confidence samples,
pixels at a depth discontinuity, and grazing samples before any of them reached
box_segment. The two therefore segmented different clouds from the same
capture: measured on this rig the offline footprints came out about 9 mm larger
per edge, because the smoothing veil around every parcel edge was still in the
cloud. Anything tuned offline was then tuned against a scene the rig does not
produce, which is the opposite of what an offline driver is for.

Use this to tune the thresholds against a stored capture, or to re-export a
capture that was streamed without --geom:

    python box_scene.py --capture-dir runs/live_.../frame_00000 \
        --seg-plane-distance 3.10 3.23 --seg-max-height 0.60 \
        --seg-views all --geom --geom-combined --geom-pick-ring

Pass the same --conf-percentile, --edge-thresh, --edge-dilate and
--max-incidence the capture was streamed with. They are recorded in
frames.jsonl and in live_diagnostics.json under settings, so there is no need
to remember them.

A capture directory is preferred over a pre-fused cloud, because the per-point
camera source survives and the per-view reconciliation can run. The same scene
loaded from one PLY reports no inter-view disagreement, which is not the same
thing as the cameras agreeing.

Read the deck distance off --probe before choosing --seg-plane-distance. The
largest coplanar surface in a four-camera cloud is usually the floor.

Keep this file beside box_segment.py, box_geometry.py, face_consensus.py and
da3_stream.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d is required")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import box_segment as bs
    import box_geometry as bg
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"cannot import box_segment.py or box_geometry.py: {exc}")

# One definition of the depth-map geometry and of the filtering, shared with
# the streamer. If this import fails the fallbacks below reproduce the
# back-projection but NOT the filtering, and the driver says so rather than
# quietly segmenting a different cloud.
_HAVE_STREAM = True
try:
    from da3_stream import (points_camera, to_world, edge_mask, incidence_mask)
except Exception as exc:  # noqa: BLE001
    _HAVE_STREAM = False
    _STREAM_ERROR = exc

    def points_camera(depth, K):
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        z = np.asarray(depth, dtype=float)
        return np.stack([(u - K[0, 2]) * z / K[0, 0],
                         (v - K[1, 2]) * z / K[1, 1], z], axis=-1)

    def to_world(pts, E):
        """w2c extrinsics, so world = R^T (x_cam - t)."""
        E = np.asarray(E, dtype=float)
        return (np.asarray(pts, dtype=float) - E[:3, 3]) @ E[:3, :3]

    edge_mask = incidence_mask = None


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_capture(cap: Path, cameras, args):
    """The capture, back-projected and filtered as the streamer filters it.

    depth_<cam>.npy is the CORRECTED depth, which is what the live pipeline
    segmented. depth_raw_<cam>.npy sits beside it for auditing the correction
    and for re-fitting it; --raw-depth selects it, and a result from raw depth
    is not comparable with a live one.
    """
    stem = "depth_raw" if args.raw_depth else "depth"
    names = [n for n in cameras if (cap / f"{stem}_{n}.npy").exists()]
    if not names:
        raise SystemExit(f"no {stem}_<cam>.npy arrays found in {cap}")

    pts, src, images, Ks, Es, report = [], [], [], [], [], []
    for i, n in enumerate(names):
        depth = np.load(cap / f"{stem}_{n}.npy").astype(np.float64)
        K = np.asarray(np.load(cap / f"K_{n}.npy"), dtype=float)
        E = np.asarray(np.load(cap / f"E_{n}.npy"), dtype=float)
        conf_path = cap / f"conf_{n}.npy"
        conf = (np.load(conf_path).astype(np.float64)
                if conf_path.exists() else None)

        valid = np.isfinite(depth) & (depth > 0)
        n_all = int(valid.sum())
        drops = {}

        if args.conf_percentile > 0 and conf is not None and valid.any():
            thr = float(np.percentile(conf[valid], args.conf_percentile))
            before = int(valid.sum())
            valid &= conf >= thr
            drops["conf"] = before - int(valid.sum())

        if args.edge_thresh > 0 and edge_mask is not None:
            before = int(valid.sum())
            valid &= edge_mask(depth, args.edge_thresh, args.edge_dilate)
            drops["edge"] = before - int(valid.sum())

        pts_cam = points_camera(np.where(np.isfinite(depth), depth, 0.0), K)

        if args.max_incidence < 90 and incidence_mask is not None:
            before = int(valid.sum())
            valid &= incidence_mask(pts_cam, args.max_incidence)[0]
            drops["grazing"] = before - int(valid.sum())

        world = to_world(pts_cam[valid], E)
        pts.append(world)
        src.append(np.full(len(world), i, dtype=np.int32))
        Ks.append(K)
        Es.append(E)
        report.append((n, n_all, len(world), drops))

        proc = cap / f"proc_{n}.png"
        if proc.exists() and cv2 is not None:
            images.append(cv2.cvtColor(cv2.imread(str(proc)), cv2.COLOR_BGR2RGB))
        else:
            images.append(np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8))

    for n, n_all, kept, drops in report:
        detail = "  ".join(f"{k} -{v}" for k, v in drops.items()) or "none"
        print(f"[filt ] {n:<7} kept {kept:>8} of {n_all:>8}   {detail}")
    if not _HAVE_STREAM:
        print(f"[WARN] da3_stream.py could not be imported ({_STREAM_ERROR}), "
              f"so the edge and incidence filters were NOT applied. This cloud "
              f"still holds the smoothing veil around every parcel edge and "
              f"the footprints will come out larger than the live pipeline "
              f"reports. Thresholds tuned here will not transfer.")

    return np.concatenate(pts), np.concatenate(src), names, images, Ks, Es


def load_clouds(specs):
    """Read one or more clouds given as NAME=PATH, or as bare paths."""
    pts, src, names = [], [], []
    for spec in specs:
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"cloud not found: {p}")
        xyz = np.asarray(o3d.io.read_point_cloud(str(p)).points, dtype=float)
        if not len(xyz):
            raise SystemExit(f"cloud holds no points: {p}")
        pts.append(xyz)
        src.append(np.full(len(xyz), len(names), dtype=np.int32))
        names.append(name)
    return np.concatenate(pts), np.concatenate(src), names


def merge_timings(a, b):
    out = dict(a or {})
    out.update(b or {})
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Segment a stored 3D scene and export the parcels as "
                    "geometry beside the coloured cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = ap.add_argument_group("input")
    src.add_argument("--capture-dir", type=Path, default=None,
                     help="a frame_NNNNN directory holding depth_<cam>.npy, "
                          "K_<cam>.npy, E_<cam>.npy and proc_<cam>.png")
    src.add_argument("--cloud", nargs="+", default=None, metavar="NAME=PATH",
                     help="clouds already on disk. Name them per camera to "
                          "keep the sources apart; a single unnamed cloud "
                          "works but leaves nothing to reconcile")
    src.add_argument("--cameras", nargs="+",
                     default=["left", "center", "right", "top"])
    src.add_argument("--raw-depth", action="store_true",
                     help="segment depth_raw_<cam>.npy instead of the "
                          "corrected depth. For auditing what the correction "
                          "did; the result is not comparable with a live one")
    src.add_argument("--out-dir", type=Path, default=None,
                     help="defaults to a 'scene' folder beside the input")
    src.add_argument("--view", action="store_true",
                     help="open the result in a window as well as writing it")
    src.add_argument("--probe", action="store_true",
                     help="list the largest planes with their distance from "
                          "the reference camera, then exit. Read the deck "
                          "distance for --seg-plane-distance off this")
    src.add_argument("--probe-planes", type=int, default=6)

    # The same filtering the streamer applies before the points reach
    # box_segment. Defaults match da3_stream.py so that a capture segmented
    # here without further argument reproduces the live cloud.
    filt = ap.add_argument_group(
        "per-view filtering, applied before segmentation exactly as "
        "da3_stream.py applies it")
    filt.add_argument("--conf-percentile", type=float, default=40.0,
                      help="drop points below this per-camera confidence "
                           "percentile; 0 disables. Only applied when "
                           "conf_<cam>.npy was written with the capture")
    filt.add_argument("--edge-thresh", type=float, default=0.02,
                      help="relative depth gradient above which a pixel is "
                           "treated as a discontinuity; 0 disables. It is a "
                           "gradient PER PIXEL, so it must be halved when "
                           "--process-res is doubled")
    filt.add_argument("--edge-dilate", type=int, default=2)
    filt.add_argument("--max-incidence", type=float, default=70.0)

    bs.add_arguments(ap)
    bg.add_arguments(ap)
    args = ap.parse_args()

    if bool(args.capture_dir) == bool(args.cloud):
        raise SystemExit("give exactly one of --capture-dir or --cloud")

    images = K_all = E_all = centres = None
    if args.capture_dir:
        points, source, names, images, K_all, E_all = load_capture(
            args.capture_dir, args.cameras, args)
        centres = [bs.camera_centre(E) for E in E_all]
        default_out = args.capture_dir / "scene"
    else:
        points, source, names = load_clouds(args.cloud)
        default_out = Path(args.cloud[0].split("=", 1)[-1]).parent / "scene"

    p = bs.params_from_args(args)
    if len(names) < 2 and p.face_per_view:
        # One source means every point claims the same camera, so the
        # reconciliation would compare a view against itself and report perfect
        # agreement. Switching it off keeps the record honest.
        p = bs.replace(p, face_per_view=False)
        print("[note ] one source only, so per-view reconciliation is off and "
              "the inter-view columns will be empty")

    if args.probe:
        return bs.probe(points, p, args.probe_planes)

    t0 = time.perf_counter()
    result = bs.segment_boxes(points, source, names, p, cam_centres=centres)

    out_dir = Path(args.out_dir or default_out)
    written, _ = bs.write_results(
        result, out_dir,
        capture_index=(args.capture_dir.name if args.capture_dir else "cloud"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        images=images, K_all=K_all, E_all=E_all, p=p,
        extra_meta={"source": str(args.capture_dir or args.cloud),
                    "raw_depth": bool(args.raw_depth),
                    "filtering": {"conf_percentile": args.conf_percentile,
                                  "edge_thresh": args.edge_thresh,
                                  "edge_dilate": args.edge_dilate,
                                  "max_incidence": args.max_incidence,
                                  "applied": _HAVE_STREAM}})

    # The offline driver defaults the export on: nothing is being timed here,
    # and a stored capture is opened precisely to be looked at.
    gp = bg.params_from_args(args, seg_params=p)
    if not gp.enabled:
        gp = bg.replace_enabled(gp, True)
    cloud = bs.coloured_cloud(result)
    geom, geom_ms = bg.write_geometry(result, out_dir, gp, cloud=cloud,
                                      cam_centres=centres)
    written.update(geom)

    print(bs.summarise(result))
    for w in result.get("warnings", []):
        print(f"[warn ] {w}")
    print(f"timings ms: {merge_timings(result.get('timings_ms'), geom_ms)}  "
          f"wall {(time.perf_counter() - t0) * 1e3:.0f} ms")
    for k, v in written.items():
        print(f"  {k:18s} {v}")

    if args.view:
        try:
            geoms = [cloud]
            mesh = bg.build_box_mesh(result, gp, centres)
            if len(mesh.triangles):
                geoms.append(mesh)
            o3d.visualization.draw_geometries(
                geoms, window_name=f"{len(result.get('boxes', []))} parcels")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn ] no display available: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())