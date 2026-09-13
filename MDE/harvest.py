#!/usr/bin/env python3
"""
harvest.py - collect the records needed to close the thesis documentation gaps.

Run this on the machine that holds the experiment outputs. It does not assume
a schema: it walks the tree, reads every JSON/CSV it finds, flattens them, and
writes an inventory you can grep. It then checks that inventory against the
specific questions the thesis needs answered and tells you which are covered
and which are still missing.

Usage
-----
    python3 harvest.py --root ~/Projects/MDE --out ~/gapwork
    python3 harvest.py --root ~/Projects/MDE --root ~/Projects/Calibration_4_5 \
                       --out ~/gapwork

Outputs (in --out)
------------------
    inventory.csv      every scalar found, as file / keypath / value
    targets_report.txt which gap questions are answerable, which are not
    files_seen.txt     every file walked, with size and mtime
    unreadable.txt     files that failed to parse, with the reason

Stdlib only. Safe to run repeatedly; it writes nothing outside --out.
"""

import argparse
import csv
import io
import json
import os
import sys
import time

MAX_BYTES = 40 * 1024 * 1024        # skip very large files
JSON_EXT = {".json"}
CSV_EXT = {".csv", ".tsv"}
TEXT_EXT = {".txt", ".log", ".yaml", ".yml", ".ini", ".cfg", ".sh"}


# --------------------------------------------------------------------------
# The questions the thesis needs answered. Each target lists substrings that,
# if they appear in a flattened key path, suggest the record is present.
# Edit freely: these are hints, not requirements.
# --------------------------------------------------------------------------
TARGETS = [
    dict(id="G01", gap="camera count for the stage-share measurement",
         hints=["stage", "rectif", "backproj", "back_proj", "infer"],
         needs=["n_cameras", "num_cameras", "cameras", "subset", "views"],
         note="Per-stage timing record must also say how many cameras were "
              "streaming. Look in the timing/diagnostic JSON written next to "
              "the resolution sweep."),

    dict(id="G02", gap="per-repeat raw timings, and mean vs median",
         hints=["repeat", "trial", "samples", "timings", "elapsed", "perf"],
         needs=["repeat", "samples", "raw"],
         note="Ten repeats per operating point. Need all ten values, not the "
              "central figure, plus which statistic the table reports."),

    dict(id="G03", gap="run register: 22 attempts vs 15 subsets",
         hints=["subset", "run", "camera", "arm", "prior", "gauge",
                "deck_offset", "exclud", "reject"],
         needs=["subset", "run"],
         note="One row per attempted run: cameras, resolution, arm "
              "(prior/prior-free), correction state, gauge outcome, exclusion "
              "reason. 14 of 22 were excluded; those rows matter most."),

    dict(id="G04", gap="fitted deck_height_linear coefficients per camera",
         hints=["deck_height", "coeff", "scale", "offset", "gain",
                "field", "s_", "a_", "b_", "c_"],
         needs=["camera", "cam", "serial"],
         note="Coefficients (s, a, b, c) per camera WITH UNITS, plus which "
              "were fitted and which fell back to offset-only. Usually in "
              "depth_correction.json."),

    dict(id="G05", gap="fold membership for leave-one-height-out",
         hints=["fold", "holdout", "held_out", "loo", "cv", "height"],
         needs=["fold", "holdout", "held"],
         note="Which height was held out in each fold, which captures were in "
              "each fold, and the per-fold RMS behind the 7-11 mm range."),

    dict(id="G06", gap="checkpoint identifier and its licence",
         hints=["checkpoint", "model", "weights", "revision", "hash",
                "commit", "hf", "repo"],
         needs=["checkpoint", "model", "revision"],
         note="Exact checkpoint string and revision/hash. Licence must be "
              "recorded separately for weights and for repository code."),

    dict(id="G07", gap="AbsRel measurement record",
         hints=["absrel", "abs_rel", "rmse", "delta1", "d1", "metric"],
         needs=["mask", "support", "count", "n_points", "reference"],
         note="Support mask, sample count, reference used, registration state "
              "and correction state for the ~1.4% AbsRel figure."),

    dict(id="G08", gap="internal tensor shapes per operating point",
         hints=["shape", "resolution", "process_res", "input", "tensor",
                "infer_size"],
         needs=["shape", "res"],
         note="Source frame size, processing resolution, and the tensor the "
              "network actually consumed, logged separately."),

    dict(id="G09", gap="per-campaign calibration record",
         hints=["rms", "reproj", "campaign", "intrinsic", "extrinsic",
                "stereo", "pair"],
         needs=["camera", "pair", "views", "n_images"],
         note="Per-pair identities for the residuals, capture counts, and "
              "what changed between campaigns."),

    dict(id="G10", gap="per-camera geometry (centres, deck-normal distance)",
         hints=["extrinsic", "translation", "rotation", "center", "centre",
                "R", "t", "baseline"],
         needs=["camera", "cam", "serial"],
         note="Camera centres in the world frame and each camera's "
              "perpendicular distance to the deck. Derivable from the "
              "calibration export."),

    dict(id="G11", gap="segmentation parameter table",
         hints=["seg", "eps", "min_samples", "thresh", "ransac", "plane",
                "iou", "cluster"],
         needs=["thresh", "eps", "ransac"],
         note="Every threshold in the shared SEG_ARGS array, the anisotropic "
              "clustering metric, and the union-find overlap rule."),

    dict(id="G12", gap="per-camera absolute deck error vs metrology chain",
         hints=["deck", "distance", "standoff", "measured", "predicted"],
         needs=["camera", "cam", "serial"],
         note="The 95 mm figure exists for the top camera. The same "
              "comparison for the other three turns one observation into a "
              "four-camera result."),

    dict(id="G13", gap="per-parcel inter-view disagreement distribution",
         hints=["inter_view", "view_consensus", "disagree", "spread",
                "height_uncertainty", "offset_mm"],
         needs=["parcel", "box", "cluster"],
         note="boxes.csv / boxes_log.csv carry view_consensus columns. The "
              "distribution replaces 'up to 44 mm'."),

    dict(id="G14", gap="measured parcel pitch distribution",
         hints=["pitch", "spacing", "gap", "centroid", "footprint"],
         needs=["parcel", "box"],
         note="Supports the 4 cm ICP correspondence choice."),

    dict(id="G15", gap="acquisition rate and overhead accounting",
         hints=["fps", "frame_rate", "acquisition", "dropped", "bandwidth",
                "mbps", "throughput"],
         needs=["fps", "rate"],
         note="Measured 3.7 vs the 4.16 payload bound: duration, delivered "
              "and dropped frames."),
]


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------
def flatten(obj, prefix="", out=None, depth=0, max_depth=12):
    """Flatten nested JSON into {keypath: scalar}. Lists become [i] paths.
    Long homogeneous numeric lists are summarised rather than expanded."""
    if out is None:
        out = {}
    if depth > max_depth:
        out[prefix] = "<max depth>"
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            flatten(v, p, out, depth + 1, max_depth)
    elif isinstance(obj, list):
        nums = [x for x in obj if isinstance(x, (int, float))]
        if len(obj) > 24 and len(nums) == len(obj):
            out[prefix + "[len]"] = len(obj)
            out[prefix + "[min]"] = min(obj)
            out[prefix + "[max]"] = max(obj)
            out[prefix + "[first8]"] = ", ".join(str(x) for x in obj[:8])
        else:
            for i, v in enumerate(obj[:64]):
                flatten(v, f"{prefix}[{i}]", out, depth + 1, max_depth)
            if len(obj) > 64:
                out[prefix + "[truncated]"] = len(obj)
    else:
        out[prefix] = obj
    return out


def read_json(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def read_csv_header(path, max_rows=3):
    """Return {header: first-value} so column names land in the inventory."""
    out = {}
    with io.open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        delim = "\t" if path.endswith(".tsv") else ","
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except Exception:
            pass
        reader = csv.reader(f, delimiter=delim)
        try:
            header = next(reader)
        except StopIteration:
            return out
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
        out["[columns]"] = ", ".join(header)
        out["[n_columns]"] = len(header)
        for j, h in enumerate(header):
            vals = [r[j] for r in rows if j < len(r)]
            if vals:
                out[f"col.{h}"] = " | ".join(vals)
    return out


# --------------------------------------------------------------------------
# Walk
# --------------------------------------------------------------------------
def walk(roots, out_dir, skip_dirs):
    inventory = []          # (file, keypath, value)
    files_seen = []
    unreadable = []

    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            unreadable.append((root, "not a directory"))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in skip_dirs and not d.startswith(".")]
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError as e:
                    unreadable.append((path, str(e)))
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in JSON_EXT | CSV_EXT | TEXT_EXT:
                    continue
                files_seen.append((path, st.st_size,
                                   time.strftime("%Y-%m-%d %H:%M",
                                                 time.localtime(st.st_mtime))))
                if st.st_size > MAX_BYTES:
                    unreadable.append((path, "too large, skipped"))
                    continue
                try:
                    if ext in JSON_EXT:
                        flat = flatten(read_json(path))
                    elif ext in CSV_EXT:
                        flat = read_csv_header(path)
                    else:
                        continue          # text files listed, not parsed
                except Exception as e:
                    unreadable.append((path, f"{type(e).__name__}: {e}"))
                    continue
                for k, v in flat.items():
                    s = str(v)
                    if len(s) > 300:
                        s = s[:300] + "..."
                    inventory.append((path, k, s))

    return inventory, files_seen, unreadable


def check_targets(inventory):
    """For each target, find inventory rows whose keypath or filename matches."""
    lowered = [(p, k, v, (p + " " + k).lower()) for p, k, v in inventory]
    report = []
    for t in TARGETS:
        hits = []
        for p, k, v, blob in lowered:
            if any(h.lower() in blob for h in t["hints"]):
                hits.append((p, k, v))
        # does any hit also carry the disambiguating field?
        strong = [h for h in hits
                  if any(n.lower() in (h[0] + " " + h[1]).lower()
                         for n in t["needs"])]
        report.append((t, hits, strong))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="directory to scan; repeat for several")
    ap.add_argument("--out", default="./gapwork", help="output directory")
    ap.add_argument("--skip", action="append", default=[],
                    help="directory name to skip; repeat as needed")
    args = ap.parse_args()

    skip = set(["node_modules", "__pycache__", "venv", ".git", "site-packages"]
               + args.skip)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("scanning...", file=sys.stderr)
    inventory, files_seen, unreadable = walk(args.root, out_dir, skip)

    inv_path = os.path.join(out_dir, "inventory.csv")
    with io.open(inv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "keypath", "value"])
        w.writerows(inventory)

    with io.open(os.path.join(out_dir, "files_seen.txt"), "w",
                 encoding="utf-8") as f:
        for p, sz, mt in sorted(files_seen):
            f.write(f"{mt}  {sz:>12,}  {p}\n")

    with io.open(os.path.join(out_dir, "unreadable.txt"), "w",
                 encoding="utf-8") as f:
        for p, why in unreadable:
            f.write(f"{why}\t{p}\n")

    report = check_targets(inventory)
    rp = os.path.join(out_dir, "targets_report.txt")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("GAP COVERAGE REPORT\n")
        f.write(f"scanned {len(files_seen)} files, "
                f"{len(inventory)} scalar records\n")
        f.write("=" * 72 + "\n\n")
        covered = partial = missing = 0
        for t, hits, strong in report:
            if strong:
                status, covered = "LIKELY PRESENT", covered + 1
            elif hits:
                status, partial = "PARTIAL", partial + 1
            else:
                status, missing = "NOT FOUND", missing + 1
            f.write(f"[{t['id']}] {status}: {t['gap']}\n")
            f.write(f"    {t['note']}\n")
            for p, k, v in (strong or hits)[:8]:
                f.write(f"      {p}\n        {k} = {v}\n")
            if len(strong or hits) > 8:
                f.write(f"      ... {len(strong or hits) - 8} more "
                        f"(grep inventory.csv)\n")
            f.write("\n")
        f.write("=" * 72 + "\n")
        f.write(f"likely present {covered} | partial {partial} | "
                f"not found {missing}\n")
        f.write("\nNOT FOUND does not mean the record does not exist. It means\n"
                "no key path matched. Widen the hints in TARGETS, or the value\n"
                "was never written and must be recovered by re-running the\n"
                "analysis with logging added.\n")

    print(f"\nwrote:\n  {inv_path}\n  {rp}\n"
          f"  {os.path.join(out_dir, 'files_seen.txt')}\n"
          f"  {os.path.join(out_dir, 'unreadable.txt')}\n", file=sys.stderr)
    with io.open(rp, encoding="utf-8") as f:
        tail = f.read().strip().splitlines()[-6:]
    print("\n".join(tail), file=sys.stderr)


if __name__ == "__main__":
    main()