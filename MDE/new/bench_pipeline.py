#!/usr/bin/env python3
"""
bench_pipeline.py

Time the perception pipeline stage by stage, across camera count and process
resolution, calling the same da3_stream entry points da3_offline.py calls.

WHY IT MIRRORS da3_offline RATHER THAN REIMPLEMENTING
------------------------------------------------------
A benchmark that calls model.inference directly measures a pipeline nobody
runs. da3_offline goes through run_inference, which chooses between the prior
and no-prior call signatures, applies align_to_input_ext_scale, and fills in
the fields DA3 returns as None. None of that is free and all of it is part of
the frame. So this calls run_inference, build_rect_plan and backproject exactly
as da3_offline does, and also records the ms that run_inference reports for
itself, since that figure synchronises CUDA and is the one the live pipeline
logs.

WHAT THE STAGES ANSWER TO
-------------------------
One end-to-end number gives the cycle time and no guidance. The stages have
different optimisation routes: rectification is a remap and answers to a cached
GPU map, inference answers to TensorRT, precision and batching, back-projection
answers to vectorisation and voxel size. Knowing which dominates decides which
is worth the effort.

THE QUESTION BEHIND IT
----------------------
Whether DA3's cost grows linearly or quadratically with view count. It attends
across views, so if four cameras cost four times one, the accuracy answer and
the timing answer agree. If they cost sixteen times, the fourth camera has to
earn its place against a cycle-time budget rather than only against coverage.
Both models are fitted to the measurements and the report says which fits.

METHOD
------
The model is loaded once and reused, because a load per run is not what a
production pipeline pays. Raw frames are read once and held, so disk is not
timed. Rectification maps are built once per camera set. Every configuration is
warmed up before timing, since a first pass through a new shape pays for
allocation and kernel selection. Peak GPU memory is recorded per configuration
and reset between, because on a Thor the memory ceiling may bind before the
time budget does, and an out-of-memory failure is recorded as a row rather than
ending the run.

EXAMPLE
-------
    python3 bench_pipeline.py --images captures/scene_a --model <checkpoint> \\
        --res 504 700 1008 1512 2002 2450 --repeats 10

Keep this file beside da3_stream.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import torch
except ImportError:
    torch = None

try:
    import da3_stream as ds
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"da3_stream.py must import cleanly: {exc!r}")


CAMS = ["center", "left", "top", "right"]


def sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mem_mb():
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)


def reset_mem():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def read_raw(images, cams):
    """Every camera's raw frame, read once and held.

    Disk is not part of the compute budget, and re-reading each repeat would
    put page cache behaviour into the timings.
    """
    raw = {}
    for c in cams:
        hit = None
        for pat in (f"{c}.png", f"{c}.jpg", f"frame_{c}.png", f"{c}_0000.png"):
            p = Path(images) / pat
            if p.exists():
                hit = p
                break
        if hit is None:
            found = sorted(Path(images).glob(f"*{c}*"))
            hit = found[0] if found else None
        if hit is None:
            raise SystemExit(f"no image for {c} under {images}")
        raw[c] = cv2.cvtColor(cv2.imread(str(hit)), cv2.COLOR_BGR2RGB)
    return raw


def fit_power(n, t):
    """Exponent k in t ~ n**k, with how well linear and quadratic each fit."""
    n, t = np.asarray(n, float), np.asarray(t, float)
    ok = (n > 0) & (t > 0)
    if ok.sum() < 3:
        return None
    k, c = np.polyfit(np.log(n[ok]), np.log(t[ok]), 1)

    def err(p):
        return float(np.sqrt(np.mean((np.exp(c) * n[ok] ** p - t[ok]) ** 2)))

    return {"exponent": round(float(k), 3),
            "rms_linear_ms": round(err(1.0) * 1e3, 2),
            "rms_quadratic_ms": round(err(2.0) * 1e3, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage-by-stage timing across camera count and resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--reference", default="center")
    ap.add_argument("--res", nargs="+", type=int,
                    default=[504, 700, 1008, 1512, 2002, 2450])
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--all-subsets", action="store_true")
    ap.add_argument("--mode", choices=("prior", "noprior"), default="prior",
                    help="run_inference's own name for the pose prior. On this "
                         "rig 'prior' is also what makes DA3 use the supplied "
                         "intrinsics instead of assuming a 60 degree lens")
    ap.add_argument("--align-ext-scale", action="store_true", default=True)
    ap.add_argument("--rect-mode", default="common",
                    choices=("common", "roi", "full"))
    ap.add_argument("--undistort", action="store_true", default=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--model", default=None,
                    help="the checkpoint da3_offline.py passes as args.model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=2)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--out", type=Path, default=Path("bench_pipeline.csv"))
    args = ap.parse_args()

    if args.model is None:
        raise SystemExit(
            "pass --model with the same checkpoint da3_offline.py uses, or "
            "this times a different model from the one the accuracy study "
            "measured. Find it with: grep -n add_argument da3_stream.py | "
            "grep -i model")

    if args.subsets:
        subsets = [s.split("+") for s in args.subsets]
    elif args.all_subsets:
        subsets = [list(s) for k in range(1, 5)
                   for s in itertools.combinations(CAMS, k)]
    else:
        subsets = [["center"], ["left", "top"], ["left", "top", "right"],
                   list(CAMS)]

    print(f"[model] da3_stream.load_model({args.model!r}, {args.device!r})")
    t0 = time.perf_counter()
    model = ds.load_model(args.model, args.device)
    print(f"[model] loaded in {time.perf_counter() - t0:.1f} s, reused for "
          f"every configuration")

    raw = read_raw(args.images, CAMS)
    h0, w0 = list(raw.values())[0].shape[:2]
    print(f"[input] {len(raw)} raw frames held in memory, {w0}x{h0}")

    rows = []
    for cams in subsets:
        name = "+".join(cams)
        intr = {c: ds.load_intrinsics(args.calib_dir, c) for c in cams}
        ext = ds.load_extrinsics(args.calib_dir, cams, args.reference)
        size = intr[cams[0]]["size"]
        plan = ds.build_rect_plan(intr, cams, size, args.rect_mode,
                                  args.undistort)
        Es = np.stack([np.asarray(ext[c]["E"], float) for c in cams])

        for res in args.res:
            reset_mem()
            acc, infer_ms, pred = {}, [], None
            try:
                for i in range(args.warmup + args.repeats):
                    t = {}
                    sync()
                    t0 = time.perf_counter()
                    frames, Ks_in = [], []
                    for c in cams:
                        m1 = plan[c].get("map1")
                        m2 = plan[c].get("map2")
                        img = (raw[c] if m1 is None else
                               cv2.remap(raw[c], m1, m2, cv2.INTER_LINEAR))
                        x, y, w, h = plan[c]["crop"]
                        frames.append(np.ascontiguousarray(
                            img[y:y + h, x:x + w]))
                        Ks_in.append(np.asarray(plan[c]["K"], float))
                    Ks_in = np.stack(Ks_in)
                    sync()
                    t["rectify"] = time.perf_counter() - t0

                    ia = SimpleNamespace(mode=args.mode, process_res=res,
                                         align_ext_scale=args.align_ext_scale)
                    sync()
                    t0 = time.perf_counter()
                    pred = ds.run_inference(model, frames, Ks_in, Es, ia)
                    sync()
                    t["inference"] = time.perf_counter() - t0

                    ba = SimpleNamespace(conf_percentile=args.conf_percentile,
                                         edge_thresh=args.edge_thresh,
                                         edge_dilate=args.edge_dilate,
                                         max_incidence=args.max_incidence,
                                         colour_by_camera=False)
                    sync()
                    t0 = time.perf_counter()
                    ds.backproject(cams, pred["depth"], pred["conf"],
                                   pred["K"], pred["E"], pred["rgb"], ba)
                    sync()
                    t["backproject"] = time.perf_counter() - t0

                    if i < args.warmup:
                        continue
                    for k, v in t.items():
                        acc[k] = acc.get(k, 0.0) + v
                    infer_ms.append(float(pred.get("ms", 0.0)))
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001
                # A configuration can fail for two quite different reasons and
                # both are results rather than crashes to abandon the sweep
                # over: out of memory at high resolution, and a degenerate
                # Umeyama when --mode prior is asked of fewer than three
                # non-collinear cameras. DA3 raises the second as
                # GeometryException from evo, not RuntimeError, so this catches
                # broadly and records what happened.
                why = ("out of memory" if "memory" in str(exc).lower()
                       else "degenerate pose alignment"
                       if "umeyama" in str(exc).lower()
                       or "degenerate" in str(exc).lower()
                       else type(exc).__name__)
                print(f"[bench] {name:24s} res {res:>4d}  FAILED  {why}: "
                      f"{str(exc)[:70]}")
                rows.append({"subset": name, "n_cams": len(cams), "res": res,
                             "failed": True, "fail_reason": why,
                             "peak_mem_mb": peak_mem_mb()})
                if why == "degenerate pose alignment":
                    print(f"        {name} has fewer than three non-collinear "
                          f"centres, so --mode prior cannot align to it. Use "
                          f"--mode noprior for this subset.")
                continue

            per = {k: v / args.repeats for k, v in acc.items()}
            total = sum(per.values())
            row = {"subset": name, "n_cams": len(cams), "res": res,
                   "grid": "x".join(str(v) for v in pred["shape"]),
                   "total_ms": round(total * 1e3, 2),
                   "fps": round(1.0 / total, 2) if total > 0 else None,
                   "infer_self_ms": round(float(np.median(infer_ms)), 2),
                   "peak_mem_mb": peak_mem_mb(), "failed": False}
            for k in ("rectify", "inference", "backproject"):
                row[f"{k}_ms"] = round(per.get(k, 0.0) * 1e3, 2)
                row[f"{k}_pct"] = (round(100 * per.get(k, 0.0) / total, 1)
                                   if total > 0 else None)
            rows.append(row)
            print(f"[bench] {name:24s} res {res:>4d} grid {row['grid']:>10s}  "
                  f"total {row['total_ms']:>8.1f} ms  {row['fps']:>6.2f} fps  "
                  f"infer {row['inference_ms']:>8.1f} "
                  f"({row['inference_pct']:>4.1f}%)  mem {row['peak_mem_mb']}")

    keys = sorted({k for r in rows for k in r})
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten {args.out}")

    good = [r for r in rows if not r.get("failed")]
    print("\ninference cost against camera count")
    for res in args.res:
        v = sorted((r["n_cams"], r["inference_ms"] / 1e3)
                   for r in good if r["res"] == res)
        if len(v) < 3:
            continue
        print(f"  res {res:>4d}: "
              + "  ".join(f"n={a} {b * 1e3:.0f}ms" for a, b in v))
        fit = fit_power([a for a, _ in v], [b for _, b in v])
        if fit:
            verdict = ("closer to LINEAR"
                       if fit["rms_linear_ms"] <= fit["rms_quadratic_ms"]
                       else "closer to QUADRATIC")
            print(f"            exponent {fit['exponent']:+.2f}, {verdict}")
    print("\nExponent near 1: the fourth camera costs a quarter more and the "
          "coverage answer and the timing answer agree. Near 2: it costs "
          "several times more and must be argued against the cycle time.")

    if good:
        print("\ntotal frame time, ms: camera count against process-res")
        counts = sorted({r["n_cams"] for r in good})
        print(f"  {'res':>6s}" + "".join(f"{f'n={n}':>12s}" for n in counts))
        for res in sorted({r["res"] for r in good}):
            cells = []
            for n in counts:
                m = [r for r in good if r["res"] == res and r["n_cams"] == n]
                cells.append(f"{m[0]['total_ms']:>11.0f}" if m else f"{'-':>11s}")
            print(f"  {res:>6d}" + "".join(cells))
        print("\nframes per second, same layout")
        print(f"  {'res':>6s}" + "".join(f"{f'n={n}':>12s}" for n in counts))
        for res in sorted({r["res"] for r in good}):
            cells = []
            for n in counts:
                m = [r for r in good if r["res"] == res and r["n_cams"] == n]
                cells.append(f"{m[0]['fps']:>11.2f}" if m else f"{'-':>11s}")
            print(f"  {res:>6d}" + "".join(cells))
        print("\npeak GPU memory, MB, same layout")
        print(f"  {'res':>6s}" + "".join(f"{f'n={n}':>12s}" for n in counts))
        for res in sorted({r["res"] for r in good}):
            cells = []
            for n in counts:
                m = [r for r in good if r["res"] == res and r["n_cams"] == n]
                cells.append(f"{m[0]['peak_mem_mb']:>11.0f}"
                             if m and m[0]["peak_mem_mb"] else f"{'-':>11s}")
            print(f"  {res:>6d}" + "".join(cells))

        fine = max(r["res"] for r in good)
        print(f"\nwhich stage dominates at res {fine}")
        for r in [x for x in good if x["res"] == fine]:
            parts = sorted(((r.get(f"{k}_pct") or 0, k) for k in
                            ("rectify", "inference", "backproject")),
                           reverse=True)
            print(f"  {r['subset']:24s} " + "  ".join(
                f"{k} {p:.0f}%" for p, k in parts if p >= 1))
        print("\nInference answers to TensorRT, precision and batching; "
              "rectification to a cached GPU remap; back-projection to "
              "vectorisation and voxel size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())