#!/usr/bin/env python3
"""
tag_lens.py — write lens_id into calibration records that lack it.

da3_stream.py refuses to run on records with no lens_id, and rightly so: it is
the only thing stopping a 12 mm extrinsic set being fused with 8 mm images. The
Calibration4 records carry serials on some files and nothing on others, so the
tag has to be added.

The focal length in the intrinsics already says which lens it is, so this reads
it rather than trusting a hand-typed value:

    python3 tag_lens.py --calib /home/jetson/Projects/Calibration4/results --dry-run
    python3 tag_lens.py --calib /home/jetson/Projects/Calibration4/results --write

Originals are copied to *.bak before anything is modified. Writes lens_id into
each intrinsics_<cam>.json and into each camera record in extrinsics.json,
since da3_stream checks both.
"""
import argparse
import json
import os
import shutil

# nominal focal -> lens id, with the tolerance a C-mount EFL spec allows
KNOWN = [(8.0, "EO-58-001-8mm"), (12.0, "EO-58-001-12mm"),
         (16.0, "EO-58-001-16mm"), (25.0, "EO-58-001-25mm")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cameras", nargs="+", default=["center", "left", "right"])
    ap.add_argument("--pitch-um", type=float, default=3.45,
                    help="sensor pixel pitch; IMX264/265 2/3in = 3.45")
    ap.add_argument("--lens-id", help="force this id instead of inferring it")
    ap.add_argument("--tol", type=float, default=6.0,
                    help="percent from nominal still accepted as that lens")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.dry_run):
        a.dry_run = True

    pitch = a.pitch_um * 1e-3
    focals, paths = {}, {}
    for c in a.cameras:
        for nm in (f"intrinsics_{c}.json", f"{c}_intrinsics.json"):
            p = os.path.join(a.calib, nm)
            if os.path.exists(p):
                paths[c] = p
                break
        else:
            raise SystemExit(f"no intrinsics for {c} in {a.calib}")
        K = json.load(open(paths[c]))["camera_matrix"]
        focals[c] = K[0][0] * pitch

    print(f"{'camera':8s} {'fx (mm)':>9s}  file")
    for c in a.cameras:
        print(f"  {c:6s} {focals[c]:9.3f}  {os.path.basename(paths[c])}")
    mean = sum(focals.values()) / len(focals)
    spread = 100 * (max(focals.values()) - min(focals.values())) / mean
    print(f"  mean {mean:.3f} mm, spread {spread:.2f}%")

    if a.lens_id:
        lens = a.lens_id
        print(f"\nforced lens_id = {lens}")
    else:
        hits = [(nom, lid) for nom, lid in KNOWN
                if abs(mean - nom) / nom * 100 <= a.tol]
        if len(hits) != 1:
            raise SystemExit(
                f"\n{mean:.3f} mm matches {len(hits)} known lenses within "
                f"{a.tol}% -- pass --lens-id explicitly.")
        nom, lens = hits[0]
        print(f"\ninferred lens_id = {lens}  "
              f"({mean:.3f} mm vs {nom:.1f} nominal, "
              f"{100*(mean-nom)/nom:+.2f}%)")
    if spread > 2.0:
        print(f"  [warn] {spread:.2f}% spread across cameras is high for one "
              f"lens type. Check the camera with the outlying focal for a weak "
              f"calibration before trusting the tag.")

    targets = [(paths[c], None) for c in a.cameras]
    ext = os.path.join(a.calib, "extrinsics.json")
    if os.path.exists(ext):
        targets.append((ext, a.cameras))
    else:
        print(f"  [warn] no extrinsics.json in {a.calib}; "
              f"da3_stream checks that too")

    print()
    for path, cams in targets:
        j = json.load(open(path))
        changed = []
        if cams is None:
            if j.get("lens_id") != lens:
                j["lens_id"] = lens
                changed.append("lens_id")
        else:
            for c in cams:
                if c in j and j[c].get("lens_id") != lens:
                    j[c]["lens_id"] = lens
                    changed.append(c)
        if not changed:
            print(f"  [ok  ] {os.path.basename(path)} already tagged")
            continue
        if a.dry_run:
            print(f"  [dry ] {os.path.basename(path)} would set "
                  f"{', '.join(changed)}")
            continue
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        with open(path, "w") as f:
            json.dump(j, f, indent=1)
        print(f"  [write] {os.path.basename(path)} set {', '.join(changed)} "
              f"(backup {os.path.basename(bak)})")

    if a.dry_run:
        print("\ndry run -- nothing written. Re-run with --write.")
    else:
        print(f"\nNow pass --expect-lens {lens} to da3_offline.py "
              f"and da3_stream.py.")


if __name__ == "__main__":
    main()