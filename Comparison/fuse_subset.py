#!/usr/bin/env python3
"""
fuse_subset.py — build one point cloud per camera subset from the per-view
depth arrays in a capture directory.

box_scene.py consumes depth arrays and does not emit a fused cloud, so the
cloud-level comparison is built here. Unprojecting and concatenating is the
right operation: it is the point set fusion sees, so grading it measures
depth quality per subset with segmentation policy out of the way.

    python3 inspect_capture.py --capture-dir CAP          # learn the layout first
    python3 fuse_subset.py --capture-dir CAP \
        --calib /home/jetson/Projects/Calibration4/results --out runs
    python3 grade_subsets.py --dir runs --exclude P1 --csv results.csv

Two things that silently corrupt the result if wrong:

  --depth-res   The depth arrays are usually at process-res, not native
                2448x2048. Intrinsics are rescaled with the pixel-centre
                convention cx' = (cx+0.5)*s-0.5; the naive cx*s is off by
                ~0.3 px at 1008, which is a real lateral bias at 3.3 m.
                Auto-detected from the array shape unless you override.

  --depth-mode  'z' if values are depth along the optical axis, 'range' if
                Euclidean distance from the camera centre. Getting this wrong
                bends the deck into a bowl -- watch the deck tilt and residual
                columns in grade_subsets.py, they will jump.
"""
import argparse
import glob
import os
import numpy as np

import rigkit as rk

DEPTH_EXT = (".npy", ".npz")


def find_clouds(capture_dir):
    """cloud_<cam>.ply written by da3_stream/da3_offline backproject().

    Strongly preferred over the raw depth arrays: these have already been
    through edge_mask, the confidence percentile and incidence_mask using the
    pipeline's own code. Unprojecting depth_<cam>.npy directly skips all three
    and leaves the smoothed veil at every box edge, which is what inflates the
    measured footprint."""
    out = {}
    for c in rk.CAMS:
        for nm in (f"cloud_{c}.ply", f"{c}.ply", f"cloud_{c}.npy"):
            for p in glob.glob(os.path.join(capture_dir, "**", nm), recursive=True):
                out.setdefault(c, p)
    return out


def find_depth(capture_dir, pattern=None):
    cands = []
    if pattern:
        cands = glob.glob(os.path.join(capture_dir, pattern))
    else:
        for e in DEPTH_EXT:
            cands += glob.glob(os.path.join(capture_dir, "**", "*" + e),
                               recursive=True)
    out = {}
    # depth_<cam>.npy must win over depth_raw_/conf_/K_/E_<cam>.npy, which all
    # contain the camera name too. Exact match first, then prefixed, then any
    # non-sidecar file.
    SIDECAR = ("conf_", "k_", "e_", "depth_raw_", "mask_", "rgb_", "proc_")
    for c in rk.CAMS:
        exact = [p for p in cands if os.path.basename(p).lower()
                 in (f"depth_{c}.npy", f"depth_{c}.npz")]
        if exact:
            out[c] = sorted(exact)[0]
            continue
        pref = [p for p in cands
                if os.path.basename(p).lower().startswith(f"depth_{c}")
                and "raw" not in os.path.basename(p).lower()]
        if pref:
            out[c] = sorted(pref)[0]
            continue
        rest = [p for p in cands if c in os.path.basename(p).lower()
                and not os.path.basename(p).lower().startswith(SIDECAR)]
        if rest:
            out[c] = sorted(rest)[0]
    return out


def maybe_voxel(X, voxel):
    if voxel <= 0:
        return X
    key = np.floor(X / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return X[np.sort(idx)]


def sidecar_K(path, cam):
    """K_<cam>.npy written beside the depth: the intrinsics DA3 returned for the
    PROCESSED grid. Always better than rescaling the calibration K, because it
    already carries the rectification and any centre crop."""
    d = os.path.dirname(path)
    for nm in (f"K_{cam}.npy", f"K_{cam}.npz", f"{cam}_K.npy"):
        p = os.path.join(d, nm)
        if os.path.exists(p):
            try:
                a = np.load(p) if nm.endswith(".npy") else None
                if a is None:
                    z = np.load(p)
                    a = z[z.files[0]]
                a = np.asarray(a).reshape(-1, 3, 3)[0]
                return a
            except Exception:
                return None
    return None


def view_cloud(cam, path, rig, a):
    """One depth array -> rig-frame points."""
    arr, key = rk.load_depth(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if a.depth_res:
        dw, dh = (int(x) for x in a.depth_res.lower().split("x"))
    else:
        dw, dh = arr.shape[1], arr.shape[0]
    K = None if a.no_sidecar_k else sidecar_K(path, cam)
    src = "K_<cam>.npy" if K is not None else "rescaled calib"
    if K is None:
        K = rk.scale_K(rig.K[cam], rig.size[cam], (dw, dh))
    if a.stride > 1:
        arr = arr[::a.stride, ::a.stride]
        K = rk.scale_K(K, (dw, dh), (arr.shape[1], arr.shape[0]))
    X = rk.unproject(arr, K, mode=a.depth_mode)
    X = X[X[:, 2] < a.max_depth]
    X = rk.cam_to_rig(X, rig.R[cam], rig.t[cam])
    print(f"  {cam:7s} {os.path.basename(path):26s} {dw}x{dh} [{src}] -> "
          f"{len(X):8d} pts  z[{X[:,2].min():+.2f},{X[:,2].max():+.2f}]"
          + (f"  key='{key}'" if key else ""))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir",
                    help="one capture dir holding per-view depth arrays "
                         "(DA3 run per view, monocular)")
    ap.add_argument("--capture-root",
                    help="parent dir with one subdirectory PER SUBSET, e.g. "
                         "runs_da3/center+left/. Use this when DA3 infers "
                         "jointly over the views it is given, so each subset "
                         "has its own depth. Subset dirs may be partial; "
                         "missing ones are skipped.")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--pattern", help="glob for the depth arrays, e.g. 'depth_*.npy'")
    ap.add_argument("--depth-mode", choices=("z", "range"), default="z")
    ap.add_argument("--depth-res", help="WxH of the depth arrays; default = "
                                        "taken from the array shape")
    ap.add_argument("--max-depth", type=float, default=6.0,
                    help="drop points beyond this (m)")
    ap.add_argument("--stride", type=int, default=1,
                    help="pixel stride; use 2 if memory is tight. Must be the "
                         "same for every subset.")
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--from-depth", action="store_true",
                    help="unproject depth_<cam>.npy instead of using "
                         "cloud_<cam>.ply. UNFILTERED: skips edge, confidence "
                         "and incidence rejection. Diagnostics only.")
    ap.add_argument("--no-sidecar-k", action="store_true",
                    help="ignore K_<cam>.npy and rescale the calibration K "
                         "instead. Only for captures without sidecars.")
    a = ap.parse_args()

    if not (a.capture_dir or a.capture_root):
        raise SystemExit("give --capture-dir (monocular DA3) or "
                         "--capture-root (joint DA3, one subdir per subset)")
    rig = rk.Rig(a.calib)

    def gather(d):
        """Prefer filtered clouds; fall back to raw depth with a warning."""
        if not a.from_depth:
            cl = find_clouds(d)
            if cl:
                return cl, "cloud"
        dp = find_depth(d, a.pattern)
        if dp and not a.from_depth:
            print("  [WARN] no cloud_<cam>.ply here, falling back to raw depth.")
            print("         Those arrays have NOT been through edge, confidence")
            print("         or incidence rejection, so box edges keep their")
            print("         smoothed veil and footprints read large. Re-run")
            print("         da3_offline.py with --write-clouds.")
        return dp, "depth"

    def build(found, kind, label):
        P = {}
        print(f"{label}:")
        for c, path in sorted(found.items()):
            if kind == "cloud":
                X = rk.load_cloud(path)
                print(f"  {c:7s} {os.path.basename(path):26s} [filtered] -> "
                      f"{len(X):8d} pts  z[{X[:,2].min():+.2f},{X[:,2].max():+.2f}]")
                P[c] = X
            else:
                P[c] = view_cloud(c, path, rig, a)
        return P

    if a.capture_root:
        os.makedirs(a.out, exist_ok=True)
        done = 0
        for s in rk.SUBSETS:
            sub = os.path.join(a.capture_root, s)
            if not os.path.isdir(sub):
                print(f"[skip] {s}  (no {sub})")
                continue
            found, kind = gather(sub)
            cs = s.split("+")
            miss = [c for c in cs if c not in found]
            if miss:
                print(f"[skip] {s}  (no depth for {', '.join(miss)})")
                continue
            extra = [c for c in found if c not in cs]
            if extra:
                print(f"  [warn] {s} also has depth for {', '.join(extra)} "
                      f"-- ignoring, subset defines the views")
            P = build({c: found[c] for c in cs}, kind, f"[{s}]")
            X = np.vstack([P[c] for c in cs])
            X = maybe_voxel(X, a.voxel)
            out = os.path.join(a.out, s + ".ply")
            rk.save_ply(out, X)
            print(f"[ok  ] {s:20s} {len(X):9d} pts -> {out}\n")
            done += 1
        if not done:
            raise SystemExit(f"no subset subdirectories found under {a.capture_root}")
        print("Next:")
        print(f"  python3 grade_subsets.py --dir {a.out} --exclude P1 --csv results.csv")
        return

    found, kind = gather(a.capture_dir)
    if not found:
        raise SystemExit(f"nothing usable under {a.capture_dir}. "
                         f"Run inspect_capture.py, then pass --pattern.")
    P = build(found, kind, "per-view -> rig-frame cloud")

    missing = [c for c in rk.CAMS if c not in P]
    if missing:
        print(f"  [warn] no depth for: {', '.join(missing)} -- "
              f"subsets needing them will be skipped")

    os.makedirs(a.out, exist_ok=True)
    for s in rk.SUBSETS:
        cs = s.split("+")
        if any(c not in P for c in cs):
            print(f"[skip] {s}")
            continue
        X = maybe_voxel(np.vstack([P[c] for c in cs]), a.voxel)
        out = os.path.join(a.out, s + ".ply")
        rk.save_ply(out, X)
        print(f"[ok  ] {s:20s} {len(X):9d} pts -> {out}")

    print("\nNext:")
    print(f"  python3 grade_subsets.py --dir {a.out} --exclude P1 --csv results.csv")
    print("\nSanity check on the first run: deck tilt should land near the scan's")
    print("1.2 deg and deck residual in single-digit mm. A bowed deck, or a tilt")
    print("in the tens of degrees, means --depth-mode or --depth-res is wrong.")


if __name__ == "__main__":
    main()