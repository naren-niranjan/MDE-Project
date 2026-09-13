#!/usr/bin/env python3
"""
sweep_subsets.py

One scene, every camera subset, every process-res. Calls da3_offline.py once per
combination so that DA3 solves each subset as its own joint problem.

WHY A DRIVER AND NOT A LOOP IN THE SHELL
----------------------------------------
Three things have to be identical across the sweep or the comparison is not a
comparison, and all three are easy to lose in a shell loop:

1.  ONE SET OF IMAGES. snap4.py writes raw, unrectified, synchronised frames.
    Every subset is run from those same files. Re-capturing between subsets
    would put the per-capture drift, which is 62 to 93 per cent of the error on
    this rig, straight into the ranking.

2.  ONE POSE-PRIOR SETTING. --pose-prior never by default. DA3 aligns predicted
    poses to supplied extrinsics with Umeyama Sim(3), which needs three
    non-collinear centres: singles and pairs cannot use it at all, and on this
    rig center+left+right has s2/s1 = 0.014 and cannot either. Letting the prior
    switch on where it happens to be possible makes the sweep a ranking of
    conditioning rather than of camera count. Use --pose-prior always as a
    SECOND sweep over the subsets that can carry it, and compare the two.

3.  NO CORRECTION. depth_correction.json carries a deck plane and alpha/beta
    fitted at depth_shape 420x504. The deck reads about 3.08 m at process-res
    504 and 3.19 m at 1008, so the same file applied at 1008 corrects towards a
    plane that is 110 mm from where the model put it. The sweep therefore runs
    uncorrected and grade_subsets.py fits p and q per run against the ground
    truth. Pass --correction only if you have one fitted at that grid.

Subset names are canonicalised through rigkit, so top+center and center+top
resolve to one directory rather than two.

EXAMPLE
-------
    python3 snap4.py --out captures/scene_a

    python3 sweep_subsets.py --images captures/scene_a \\
        --out-root runs_da3/scene_a --res 504 700 1008

    python3 grade_subsets.py --gt gt/pointcloud_20260826_095616.ply \\
        --runs-root runs_da3/scene_a --out grades/scene_a
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run da3_offline.py once per camera subset per "
                    "process-res, from one captured scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", type=Path, required=True,
                    help="a snap4.py output directory of raw frames")
    ap.add_argument("--out-root", type=Path, default=Path("runs_da3"))
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--cameras", nargs="+", default=rigkit.CAMS)
    ap.add_argument("--res", nargs="+", type=int, default=[504, 700, 1008],
                    help="process-res values; each must be a multiple of the "
                         "ViT patch size 14 or DA3 rounds it and the label "
                         "stops describing the run")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these subsets, e.g. left+top+right")
    ap.add_argument("--pose-prior", choices=("auto", "always", "never"),
                    default="never")
    ap.add_argument("--upright", action="store_true",
                    help="pass --upright to da3_offline. Worth checking with "
                         "rig_geometry.py first: a subset sweep run without it "
                         "on a rig with a rolled mount is partly a ranking of "
                         "mount angle")
    ap.add_argument("--reference", default="center",
                    help="the frame the extrinsics records were written "
                         "against. It does NOT have to be in the subset")
    ap.add_argument("--offline", default="da3_offline.py")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to da3_offline.py")
    ap.add_argument("--force", action="store_true",
                    help="re-run subsets that already have a fused.ply")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bad = [r for r in args.res if r % 14]
    if bad:
        print(f"[warn ] {bad} are not multiples of 14; DA3 will round them and "
              f"the directory label will not match the grid actually used")

    cams = list(args.cameras)
    k_max = args.k_max or len(cams)
    subs = [rigkit.canon(s, cams) for s in
            (args.only or ["+".join(c) for k in range(args.k_min, k_max + 1)
                           for c in itertools.combinations(cams, k)])]
    subs = sorted(set(subs), key=lambda s: (len(s.split("+")), s))

    jobs = [(s, r) for s in subs for r in args.res]
    print(f"[plan ] {len(subs)} subsets x {len(args.res)} resolutions "
          f"= {len(jobs)} inference runs")
    print(f"[plan ] images {args.images}   pose prior {args.pose_prior}   "
          f"reference {args.reference}")

    done, skipped, failed = 0, 0, []
    t_all = time.perf_counter()
    for i, (sub, res) in enumerate(jobs, 1):
        out = args.out_root / f"{sub}_res{res}"
        if (out / "fused.ply").exists() and not args.force:
            print(f"[skip ] {i:3d}/{len(jobs)}  {out} already has a fused.ply")
            skipped += 1
            continue
        cmd = [sys.executable, args.offline,
               "--images", str(args.images),
               "--calib-dir", str(args.calib_dir),
               "--cameras", *sub.split("+"),
               "--reference", args.reference,
               "--out-dir", str(out),
               "--pose-prior", args.pose_prior,
               "--process-res", str(res)]
        if args.upright:
            cmd.append("--upright")
        cmd += [a for a in args.extra if a != "--"]

        print(f"\n[run  ] {i:3d}/{len(jobs)}  {sub}  res {res}")
        print("        " + " ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        rc = subprocess.call(cmd)
        dt = time.perf_counter() - t0
        if rc:
            print(f"[FAIL ] {sub} res {res} exited {rc} after {dt:.0f} s")
            failed.append((sub, res, rc))
        else:
            print(f"[ok   ] {sub} res {res} in {dt:.0f} s")
            done += 1

    print(f"\n{done} run(s) written, {skipped} skipped, {len(failed)} failed, "
          f"{(time.perf_counter() - t_all) / 60:.1f} min total")
    for sub, res, rc in failed:
        print(f"  FAILED {sub} res {res} rc {rc}")
    if failed:
        print("A subset of one or two cameras failing with a Umeyama or rank "
              "error means the pose prior reached it. Re-run that one with "
              "--pose-prior never, and do not mix the two in one grading.")
    print(f"\nNext:\n  python3 grade_subsets.py --gt <ground truth ply> "
          f"--runs-root {args.out_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())