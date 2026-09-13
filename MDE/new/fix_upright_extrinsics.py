#!/usr/bin/env python3
"""
fix_upright_extrinsics.py

Repair the extrinsics of runs written by da3_offline.py --upright, and patch
da3_offline.py so the fault does not recur.

THE FAULT
---------
--upright rotates a camera mounted inverted to gravity-upright before inference,
because DA3's monocular prior is trained on gravity-aligned imagery. Afterwards
da3_offline.py rotates the depth back and moves the principal point back, but it
leaves the returned extrinsics alone. Those belong to the ROTATED camera, which
differs from the real one by 180 degrees about the optical axis, so every
flipped view is reported pointing away from where it is.

Among flipped cameras the error cancels in any relative rotation, so a subset
made only of flipped views looks healthy. Mix in one unflipped view and the
whole subset fuses into two scenes back to back. On this rig only `center` sits
below the 90 degree roll threshold, which is why every subset containing it
failed at 179 to 180 degrees and no other subset did.

THE REPAIR
----------
Rotating the image by 180 degrees is a virtual camera with x' = Rz x, where
Rz = diag(-1, -1, 1). DA3 returned x' = R' x_world + t', so the real pose is

    R = Rz R'        t = Rz t'

Rz is symmetric and its own inverse, so this is also how it would be undone.
Depth along the optical axis is invariant to a roll and the principal point was
already handled, so nothing else on disk is wrong and NO SECOND INFERENCE IS
NEEDED. The original arrays are kept as E_<cam>_asreturned.npy and a marker file
prevents the rotation being applied twice.

WHICH CAMERAS WERE FLIPPED
--------------------------
Read from the calibration, exactly as da3_offline.py reads it: roll about the
optical axis is atan2(R[0,1], R[0,0]) and anything beyond --roll-thresh was
flipped. The list is printed before anything is written, so it can be checked
against the [roll ] lines the sweep produced.

EXAMPLE
-------
    python3 fix_upright_extrinsics.py --runs-root runs_da3/scene_b --dry-run
    python3 fix_upright_extrinsics.py --runs-root runs_da3/scene_b
    python3 fix_upright_extrinsics.py --patch-source da3_offline.py

    python3 check_fusion.py --runs-root runs_da3/scene_b --report

Keep this file beside da3_offline.py and da3_stream.py.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

try:
    import da3_stream as ds
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"da3_stream.py must import cleanly: {exc!r}")


CAMS = ["center", "left", "top", "right"]
RZ = np.diag([-1.0, -1.0, 1.0])
MARKER = ".upright_extrinsics_fixed"

PATCH_ANCHOR = """            K_out = np.array(K_out, copy=True, dtype=float)
            K_out[i] = K"""

PATCH_BODY = """            K_out = np.array(K_out, copy=True, dtype=float)
            K_out[i] = K
            # The extrinsics DA3 returned belong to the ROTATED camera, which
            # differs from the real one by 180 degrees about the optical axis.
            # Depth is invariant to that roll and the principal point is handled
            # above, but the pose is not: x_cam = Rz (R' x_world + t') with
            # Rz = diag(-1, -1, 1). Without this every flipped view is reported
            # pointing 180 degrees away from where it is, and any subset mixing
            # flipped and unflipped cameras fuses into two scenes back to back.
            # Among flipped views alone the error cancels, so it does not show
            # up in a subset that happens to be all-flipped.
            # ONLY when DA3 predicted its own poses. With the pose prior on
            # it echoes the SUPPLIED extrinsics, which are already in the real
            # camera frame, and rotating those breaks what was correct. The
            # camera centre is invariant under this rotation, so the damage
            # shows up as a 180 degree orientation error beside a perfect
            # centre match, which is exactly what --pose-prior always produced.
            if not np.allclose(np.asarray(E_out[i], float)[:3, :3],
                               np.asarray(Es[i], float)[:3, :3], atol=1e-6):
                Rz = np.diag([-1.0, -1.0, 1.0])
                E_out = np.array(E_out, copy=True, dtype=float)
                E_out[i][:3, :3] = Rz @ E_out[i][:3, :3]
                E_out[i][:3, 3] = Rz @ E_out[i][:3, 3]
            else:
                print(f"[roll ] {nm}: DA3 returned the supplied extrinsics, "
                      f"so they are already in the real camera frame and are "
                      f"left alone")"""


def rolls(calib_dir: Path, cams, reference: str):
    ext = ds.load_extrinsics(Path(calib_dir), cams, reference)
    out = {}
    for c in cams:
        R = np.asarray(ext[c]["E"], float)[:3, :3]
        out[c] = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
    return out


def parse_run(path: Path):
    subset = re.sub(r"_res\d+$", "", path.name)
    cams = [c for c in subset.replace(",", "+").split("+") if c]
    if not cams or any(c not in CAMS for c in cams):
        return None
    return cams


def fix_run(run: Path, cams, flipped, args):
    if (run / MARKER).exists() and not args.force:
        return "already fixed"
    touched = []
    for c in cams:
        if c not in flipped:
            continue
        p = run / f"E_{c}.npy"
        if not p.exists():
            return f"missing {p.name}"
        E = np.load(p).astype(np.float64)
        backup = run / f"E_{c}_asreturned.npy"
        if not backup.exists():
            shutil.copy2(p, backup)
        E = np.array(E, copy=True)
        E[:3, :3] = RZ @ E[:3, :3]
        E[:3, 3] = RZ @ E[:3, 3]
        if not args.dry_run:
            np.save(p, E)
        touched.append(c)
    if not touched:
        return "no flipped camera in this subset"
    if not args.dry_run:
        (run / MARKER).write_text(
            "E_<cam>.npy rotated by diag(-1,-1,1) to undo the --upright roll; "
            "originals kept as E_<cam>_asreturned.npy\n")
    return "rotated " + "+".join(touched)


def patch_source(path: Path, dry_run: bool):
    src = path.read_text()
    if "Rz = np.diag([-1.0, -1.0, 1.0])" in src:
        print(f"[patch] {path} already carries the fix")
        return
    if PATCH_ANCHOR not in src:
        raise SystemExit(
            f"could not find the un-rotation block in {path}. Apply the change "
            f"by hand: inside the `if any(flip.values())` loop, after K_out is "
            f"rewritten, add\n\n{PATCH_BODY.split(chr(10), 2)[2]}\n")
    out = src.replace(PATCH_ANCHOR, PATCH_BODY, 1)
    if dry_run:
        print(f"[patch] {path} would be patched ({len(out) - len(src)} chars)")
        return
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(out)
    print(f"[patch] {path} patched, original kept as {path.name}.bak")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Undo the 180 degree roll left in the extrinsics by "
                    "da3_offline.py --upright.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--run", type=Path, action="append", default=None)
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--reference", default="center")
    ap.add_argument("--roll-thresh", type=float, default=90.0,
                    help="must match the --roll-thresh the sweep was run with")
    ap.add_argument("--patch-source", type=Path, default=None,
                    help="also patch this copy of da3_offline.py")
    ap.add_argument("--force", action="store_true",
                    help="re-apply even where the marker file exists. Applying "
                         "the rotation twice restores the fault, so this exists "
                         "only for a run whose marker was lost")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    roll = rolls(args.calib_dir, CAMS, args.reference)
    flipped = {c for c in CAMS if abs(roll[c]) > args.roll_thresh}
    print(f"roll about the optical axis, threshold {args.roll_thresh:.0f} deg")
    for c in CAMS:
        print(f"  {c:7s} {roll[c]:+8.1f} deg   "
              + ("flipped for inference" if c in flipped else "left alone"))
    if not flipped:
        print("\nno camera exceeds the threshold, so --upright did nothing and "
              "there is nothing to undo")
        return 0
    print(f"\nany subset mixing {sorted(flipped)} with "
          f"{sorted(set(CAMS) - flipped)} is the one that fused into two "
          f"scenes back to back")

    if args.patch_source:
        print()
        patch_source(args.patch_source, args.dry_run)

    runs = [Path(d) for d in (args.run or [])]
    if args.runs_root:
        runs += [d for d in sorted(Path(args.runs_root).iterdir()) if d.is_dir()]
    if not runs:
        return 0

    print(f"\n{'run':34s} result")
    print("-" * 66)
    n = 0
    for run in runs:
        cams = parse_run(run)
        if cams is None:
            continue
        res = fix_run(run, cams, flipped, args)
        print(f"{run.name:34s} {res}")
        if res.startswith("rotated"):
            n += 1
    print(f"\n{n} run(s) {'would be' if args.dry_run else ''} repaired. No "
          f"inference was re-run: depth along the optical axis is invariant to "
          f"a roll, so only the pose was ever wrong.")
    print("\nNext:\n  python3 check_fusion.py --runs-root <root> --report")
    print("Every subset should now sit in the same few degrees as left+right, "
          "left+top and top+right already did. If the center-containing ones "
          "are still near 180, the rotation was applied in the wrong "
          "direction and E_<cam>_asreturned.npy is the way back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())