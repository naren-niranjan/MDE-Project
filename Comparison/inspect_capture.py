#!/usr/bin/env python3
"""
inspect_capture.py — look at a capture directory and report what the subset
study can be built from. Run this once; it settles the naming questions.

    python3 inspect_capture.py --capture-dir /path/to/capture
    python3 inspect_capture.py --capture-dir /path/to/capture --boxes runs/center/boxes.json

Reports:
  * which per-view clouds exist and which camera each maps to
  * point counts, extents, and whether each looks like it is in the rig frame
  * the schema of a boxes.json, if you point at one
"""
import argparse
import glob
import json
import os
import numpy as np

import rigkit as rk

CLOUD_EXT = (".ply", ".pcd")
DEPTH_EXT = (".npy", ".npz")


def guess_cam(name):
    n = os.path.basename(name).lower()
    for c in rk.CAMS:
        if c in n:
            return c
    for k, c in (("cam0", "center"), ("cam1", "left"), ("cam2", "right"),
                 ("c.", "center"), ("l.", "left"), ("r.", "right")):
        if k in n:
            return c
    return None


def load_K(path):
    """Pull a 3x3 intrinsic matrix out of an npy/npz."""
    if path.endswith(".npz"):
        z = np.load(path)
        for k in z.files:
            a = z[k]
            if a.shape[-2:] == (3, 3):
                return np.asarray(a).reshape(-1, 3, 3)[0]
        return None
    a = np.load(path)
    return a.reshape(-1, 3, 3)[0] if a.shape[-2:] == (3, 3) else None


def compare_k(a):
    """Per-subset intrinsics, to expose rect-mode crop differences."""
    root = a.capture_root
    print(f"=== saved K per subset under {root} ===")
    table = {}
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for e in (".npy", ".npz"):
            for p in sorted(glob.glob(os.path.join(d, "**", "*" + e), recursive=True)):
                b = os.path.basename(p).lower()
                if "k" not in b and "intrin" not in b:
                    continue
                cam = guess_cam(p)
                if not cam:
                    continue
                try:
                    K = load_K(p)
                except Exception:
                    continue
                if K is not None:
                    table.setdefault(cam, {})[sub] = K
    if not table:
        print("  no K arrays found. Was --save-npy passed, or the capture "
              "forced with the capture key?")
        return
    for cam, per in sorted(table.items()):
        print(f"\n  {cam}")
        ref = None
        for sub, K in sorted(per.items()):
            tag = ""
            if ref is None:
                ref = K
            elif not np.allclose(K, ref, atol=1e-3):
                tag = "   <-- DIFFERS from " + sorted(per)[0]
            print(f"    {sub:22s} fx={K[0,0]:9.3f} fy={K[1,1]:9.3f} "
                  f"cx={K[0,2]:8.3f} cy={K[1,2]:8.3f}{tag}")
    print("\n  Any DIFFERS line means --rect-mode common gave that camera a")
    print("  different crop depending on the subset, so it did not see the same")
    print("  pixels in every run. Part of any measured difference is the crop.")
    print("  Options: report it as a caveat, or re-run with a rect mode that")
    print("  keeps the crop fixed and accept whatever else that costs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir")
    ap.add_argument("--capture-root", help="parent dir with one subdir per subset")
    ap.add_argument("--compare-k", action="store_true",
                    help="with --capture-root: print each camera's saved K per "
                         "subset, to expose --rect-mode common crop differences")
    ap.add_argument("--gt", default="gt_scene.json")
    ap.add_argument("--boxes")
    a = ap.parse_args()

    if a.compare_k or (a.capture_root and not a.capture_dir):
        return compare_k(a)
    if not a.capture_dir:
        raise SystemExit("give --capture-dir, or --capture-root [--compare-k]")
    print(f"=== {a.capture_dir} ===")
    for root, dirs, files in os.walk(a.capture_dir):
        depth = root[len(a.capture_dir):].count(os.sep)
        if depth > 2:
            continue
        rel = os.path.relpath(root, a.capture_dir)
        print(f"  {rel}/" if rel != "." else "  ./")
        for f in sorted(files)[:24]:
            sz = os.path.getsize(os.path.join(root, f))
            print(f"      {f:44s} {sz/1e6:8.2f} MB")
        if len(files) > 24:
            print(f"      ... {len(files)-24} more")

    clouds = []
    for e in CLOUD_EXT:
        clouds += glob.glob(os.path.join(a.capture_dir, "**", "*" + e),
                            recursive=True)
    clouds = sorted(set(clouds))
    print(f"\n=== {len(clouds)} cloud files ===")
    gt = json.load(open(a.gt)) if os.path.exists(a.gt) else None
    for p in clouds:
        cam = guess_cam(p)
        try:
            P = rk.load_cloud(p)
        except Exception as ex:
            print(f"  {os.path.relpath(p, a.capture_dir):48s} unreadable: {ex}")
            continue
        line = (f"  {os.path.relpath(p, a.capture_dir):48s} {len(P):9d} pts  "
                f"cam={cam or '?':6s} "
                f"z[{P[:,2].min():+.2f},{P[:,2].max():+.2f}]")
        if gt:
            n = np.array(gt["deck_plane"]["n"])
            d = gt["deck_plane"]["d"]
            near = (np.abs(P @ n + d) < 0.03).mean()
            line += f"  on-deck frac={near:5.1%}"
            line += "  RIG-FRAME OK" if near > 0.05 else "  <-- not rig frame?"
        print(line)

    # --- depth arrays: what box_scene.py actually consumes -----------------
    deps = []
    for e in DEPTH_EXT:
        deps += glob.glob(os.path.join(a.capture_dir, "**", "*" + e), recursive=True)
    deps = sorted(set(deps))
    print(f"\n=== {len(deps)} depth arrays ===")
    if not deps:
        print("  none. box_scene.py needs depth arrays -- run the DA3 capture step")
        print("  for this snapshot first, then point --capture-dir at its output.")
    for p_ in deps:
        cam = guess_cam(p_)
        try:
            arr, key = rk.load_depth(p_)
        except Exception as ex:
            print(f"  {os.path.relpath(p_, a.capture_dir):44s} unreadable: {ex}")
            continue
        f = arr[np.isfinite(arr) & (arr > 0)]
        rng = f"{f.min():.3f}..{f.max():.3f}" if f.size else "empty"
        print(f"  {os.path.relpath(p_, a.capture_dir):44s} shape={str(arr.shape):14s} "
              f"{str(arr.dtype):8s} cam={cam or '?':6s} valid={f.size/arr.size:5.1%} "
              f"range={rng}" + (f"  key='{key}'" if key else ""))
        if f.size:
            zdeck = -gt["deck_plane"]["d"] if gt else 3.31
            if abs(np.median(f) - zdeck) > 0.5 * zdeck:
                print(f"      NOTE: median {np.median(f):.3f} is far from the deck "
                      f"depth {zdeck:.2f} -- disparity or normalised depth, not metres?")
            if arr.shape[1] not in (2448,) and gt:
                print(f"      width {arr.shape[1]} != native 2448 -- pass "
                      f"--depth-res {arr.shape[1]}x{arr.shape[0]} to fuse_subset.py")

    if a.boxes and os.path.exists(a.boxes):
        j = json.load(open(a.boxes))
        print(f"\n=== {a.boxes} ===")
        print(f"  top-level type: {type(j).__name__}")
        if isinstance(j, dict):
            print(f"  keys: {list(j.keys())[:20]}")
            for k, v in j.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"  '{k}' is a list of {len(v)} records; "
                          f"first record keys: {list(v[0].keys())}")
                    print(f"    {json.dumps(v[0], indent=2)[:900]}")
                    break
        elif isinstance(j, list) and j:
            print(f"  list of {len(j)}; first record keys: {list(j[0].keys())}")
            print(f"    {json.dumps(j[0], indent=2)[:900]}")
        print("\n  -> paste this block back and I'll wire grade_subsets.py "
              "--boxes to it directly.")
    elif a.boxes:
        print(f"\n[missing] {a.boxes}")


if __name__ == "__main__":
    main()