#!/usr/bin/env python3
"""
da3_offline.py — run the da3_stream.py pipeline on images from disk.

da3_stream.py opens the cameras and has no replay mode, but everything after
acquisition is already decoupled: build_rect_plan builds the remap,
run_inference does one DA3 pass over a list of frames, correct_depths applies
the height correction, backproject makes the clouds. This script imports those
functions and feeds them images instead of buffers, so the offline path cannot
drift from the live one.

    python3 da3_offline.py \
        --images /home/jetson/Projects/Calibration4/snapshots/20260707_105753_062 \
        --calib-dir /home/jetson/Projects/Calibration4/results \
        --cameras center left right --reference center \
        --expect-lens "" --process-res 1008 --edge-thresh 0.01 --edge-dilate 1 \
        --out-dir runs_da3/center+left+right

Writes the same layout the live capture writes, so box_scene.py and
fuse_subset.py read it unchanged:

    depth_<cam>.npy  depth_raw_<cam>.npy  conf_<cam>.npy
    K_<cam>.npy      E_<cam>.npy          proc_<cam>.png
    fused.ply  cloud_<cam>.ply            (with --write-clouds)

Start with --probe: it resolves every da3_stream symbol and prints the
signatures without loading the model or touching the GPU.

NOTE ON THE SUBSET STUDY
    DA3 solves the given frames JOINTLY, so center's depth genuinely differs
    depending on which other views are in the call. Run this once per subset
    with a different --cameras and --out-dir; do not run it once with all
    three and slice afterwards.
"""
import argparse
import importlib
import inspect
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

NEEDED = ["load_intrinsics", "load_extrinsics", "build_rect_plan",
          "load_model", "run_inference", "backproject"]
OPTIONAL = ["check_lens", "check_serials", "load_correction", "correct_depths",
            "scale_K", "to_o3d", "layer_report", "print_layer_report"]


# ------------------------------------------------------------------ helpers
def import_stream(path):
    p = Path(path).resolve()
    sys.path.insert(0, str(p.parent))
    ds = importlib.import_module(p.stem)
    missing = [f for f in NEEDED if not hasattr(ds, f)]
    if missing:
        raise SystemExit(f"{p} has no {', '.join(missing)} -- "
                         f"run probe_da3_stream.py and check the names.")
    return ds


def show_api(ds):
    print(f"resolved {ds.__file__}\n")
    for f in NEEDED + OPTIONAL:
        fn = getattr(ds, f, None)
        mark = " " if f in NEEDED else "?"
        if fn is None:
            print(f"  {mark} {f:22s} MISSING")
        else:
            try:
                print(f"  {mark} {f:22s} {f}{inspect.signature(fn)}")
            except (TypeError, ValueError):
                print(f"  {mark} {f:22s} (no signature)")
    for c in ("DEFAULT_CALIB", "DEFAULT_LENS", "CAMERAS", "CORRECTION_MODEL"):
        if hasattr(ds, c):
            print(f"    const {c} = {str(getattr(ds, c))[:80]}")


def stream_args(ds, argv):
    """Build da3_stream's own args namespace, so every default matches."""
    keep = sys.argv
    sys.argv = ["da3_stream.py"] + argv
    try:
        return ds.parse_args()
    finally:
        sys.argv = keep


def find_images(d, names):
    """<cam>.png, proc_<cam>.png, frame_<cam>.png -- first match wins."""
    d = Path(d)
    out = {}
    for n in names:
        for pat in (f"{n}.png", f"proc_{n}.png", f"{n}.*", f"*{n}*.png",
                    f"*{n}*.jpg"):
            hit = sorted(d.glob(pat))
            hit = [h for h in hit if h.suffix.lower() in
                   (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")]
            if hit:
                out[n] = hit[0]
                break
    return out


DEPTH_KEYS = ("depth", "depths", "depth_map", "depth_maps", "metric_depth", "z")
CONF_KEYS = ("conf", "confidence", "conf_map", "confidences", "mask")
K_KEYS = ("intrinsics", "K", "Ks", "camera_intrinsics", "intrinsic")
E_KEYS = ("extrinsics", "E", "Es", "camera_extrinsics", "extrinsic", "poses")


def unwrap(pred, names, Ks_in, Es):
    """run_inference returns a tuple, a dict, or a prediction object."""
    if isinstance(pred, (tuple, list)):
        vals = list(pred) + [None] * (4 - len(pred))
        depth, conf, K_out, E_out = vals[:4]
    elif isinstance(pred, dict):
        g = lambda ks: next((pred[k] for k in ks
                             if pred.get(k) is not None), None)
        depth, conf = g(DEPTH_KEYS), g(CONF_KEYS)
        K_out, E_out = g(K_KEYS), g(E_KEYS)
    else:
        g = lambda ks: next((getattr(pred, k) for k in ks
                             if getattr(pred, k, None) is not None), None)
        depth, conf = g(DEPTH_KEYS), g(CONF_KEYS)
        K_out, E_out = g(K_KEYS), g(E_KEYS)
    if depth is None:
        keys = (list(pred.keys()) if isinstance(pred, dict)
                else [k for k in dir(pred) if not k.startswith("_")])
        raise SystemExit(
            f"no depth field on the {type(pred).__name__} from run_inference.\n"
            f"  fields present: {keys}\n"
            f"  add the right name to DEPTH_KEYS in da3_offline.py")
    K_out = Ks_in if K_out is None else K_out
    E_out = Es if E_out is None else E_out
    to_np = lambda v: None if v is None else np.asarray(
        v.detach().cpu().numpy() if hasattr(v, "detach") else v)
    return to_np(depth), to_np(conf), to_np(K_out), to_np(E_out)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--stream", default="da3_stream.py")
    ap.add_argument("--images", help="directory holding one image per camera")
    ap.add_argument("--cameras", nargs="+", default=["center", "left", "right"])
    ap.add_argument("--out-dir", default="runs_da3/offline")
    ap.add_argument("--probe", action="store_true",
                    help="resolve symbols and print signatures, then exit")
    ap.add_argument("--no-write-clouds", dest="write_clouds",
                    action="store_false", default=True,
                    help="skip fused.ply and cloud_<cam>.ply. NOT ADVISED: the "
                         "clouds are where edge_mask, the confidence percentile "
                         "and incidence_mask are applied, so without them only "
                         "the unfiltered depth arrays survive and anything "
                         "unprojecting them keeps the smoothed veil at box "
                         "edges.")
    ap.add_argument("--upright", action="store_true",
                    help="rotate any camera mounted inverted to gravity-upright "
                         "before inference, adjusting its intrinsics, then "
                         "rotate the depth back. DA3's monocular prior is "
                         "trained on gravity-aligned imagery, so an inverted "
                         "view gets a different prior than its neighbours. Roll "
                         "is read from the extrinsics.")
    ap.add_argument("--roll-thresh", type=float, default=90.0,
                    help="roll beyond this (deg) counts as inverted")
    ap.add_argument("--pose-prior", choices=("auto", "always", "never"),
                    default="never",
                    help="supply the calibrated extrinsics to DA3. DA3 aligns "
                         "its predicted poses to them with Umeyama Sim(3), "
                         "which needs 3+ non-collinear cameras: 1 and 2 view "
                         "subsets fail outright, and this rig's three cameras "
                         "are nearly collinear so even the triple is poorly "
                         "conditioned. 'never' (default) keeps conditioning "
                         "identical across all seven subsets, which is what "
                         "the comparison requires. 'auto' uses the prior only "
                         "where it is possible -- convenient, but confounds "
                         "the study.")
    ap.add_argument("--skip-lens-check", action="store_true",
                    help="bypass check_lens. Prefer tag_lens.py: an untagged "
                         "record is what lets a 12 mm extrinsic set be fused "
                         "with 8 mm images.")
    ap.add_argument("--already-rectified", action="store_true",
                    help="images are proc_*.png from a previous capture: skip "
                         "the remap. Rectifying twice is silent and ruinous.")
    a, passthru = ap.parse_known_args()

    ds = import_stream(a.stream)
    if a.probe:
        return show_api(ds)
    if not a.images:
        raise SystemExit("--images is required (or use --probe)")

    names = list(a.cameras)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    argv = passthru + ["--cameras"] + names + ["--out-dir", str(out)]
    args = stream_args(ds, argv)
    # The reference is whatever the extrinsics records were written against,
    # NOT a member of the subset. load_extrinsics checks each record's
    # "reference" field, so asking for 'left' on center-referenced records
    # fails even when left is the only camera. Extrinsics only need a shared
    # frame; the reference camera does not have to be in the batch.
    ref = getattr(args, "reference", "center")

    # ---- calibration ------------------------------------------------------
    calib = Path(getattr(args, "calib_dir", ".") or ".")
    intr = {n: ds.load_intrinsics(calib, n) for n in names}
    ext = ds.load_extrinsics(calib, names, ref)
    if hasattr(ds, "check_lens") and not a.skip_lens_check:
        try:
            ds.check_lens(intr, ext, names, getattr(args, "expect_lens", "") or "")
        except SystemExit:
            print("\n  check_lens rejected these records. If they carry no "
                  "lens_id at all:\n    python3 tag_lens.py --calib "
                  f"{calib} --write\n  then pass --expect-lens with the id it "
                  "writes. --skip-lens-check bypasses, but the tag is what "
                  "stops a 12 mm set being fused with 8 mm images.\n")
            raise
        except Exception as e:
            print(f"[warn] check_lens: {e}")

    # ---- images -----------------------------------------------------------
    found = find_images(a.images, names)
    miss = [n for n in names if n not in found]
    if miss:
        raise SystemExit(f"no image for {', '.join(miss)} in {a.images}")
    raw = {}
    for n in names:
        img = cv2.imread(str(found[n]), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read {found[n]}")
        raw[n] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"  {n:7s} {found[n].name:24s} {img.shape[1]}x{img.shape[0]}")

    size = (raw[names[0]].shape[1], raw[names[0]].shape[0])
    for n in names:
        if (raw[n].shape[1], raw[n].shape[0]) != size:
            print(f"[warn] {n} is a different size; mixed sizes in one DA3 "
                  f"batch trigger a centre crop to the smallest")

    # ---- rectify ----------------------------------------------------------
    if a.already_rectified:
        print("[rect ] skipped (--already-rectified)")
        frames = [raw[n] for n in names]
        Ks_in = [np.asarray(intr[n]["K"] if isinstance(intr[n], dict)
                            else intr[n].K, float) for n in names]
    else:
        plan = ds.build_rect_plan(intr, names, size, args.rect_mode,
                                  getattr(args, "undistort", True))
        frames, Ks_in = [], []
        for n in names:
            p = plan[n] if isinstance(plan, dict) else getattr(plan, n)

            def get(o, *keys):
                for k in keys:
                    if isinstance(o, dict):
                        if k in o:
                            return o[k]
                    elif hasattr(o, k):
                        return getattr(o, k)
                return None

            m1 = get(p, "map1", "mapx", "map_x")
            m2 = get(p, "map2", "mapy", "map_y")
            K = get(p, "K", "newK", "K_rect", "new_K")
            if K is None:
                raise SystemExit(f"no intrinsics on the rect plan for {n}; "
                                 f"fields are {p.keys() if isinstance(p, dict) else dir(p)}")
            img = raw[n] if m1 is None else cv2.remap(raw[n], m1, m2,
                                                      cv2.INTER_LINEAR)
            frames.append(img)
            Ks_in.append(np.asarray(K, float))
        print(f"[rect ] {args.rect_mode}, rectified to "
              f"{frames[0].shape[1]}x{frames[0].shape[0]}")

    # ---- gravity canonicalisation ----------------------------------------
    flip = {}
    if a.upright:
        for i, nm in enumerate(names):
            R = np.asarray(ext[nm]["E"], float)[:3, :3] if isinstance(ext[nm], dict) \
                and "E" in ext[nm] else np.asarray(ext[nm], float)[:3, :3]
            roll = np.degrees(np.arctan2(R[0, 1], R[0, 0]))
            flip[nm] = abs(roll) > a.roll_thresh
            print(f"[roll ] {nm:7s} {roll:+7.1f} deg"
                  + ("   -> rotating 180 for inference" if flip[nm] else ""))
        for i, nm in enumerate(names):
            if not flip[nm]:
                continue
            frames[i] = np.ascontiguousarray(frames[i][::-1, ::-1])
            h_, w_ = frames[i].shape[:2]
            K = Ks_in[i].copy()
            K[0, 2] = (w_ - 1) - K[0, 2]
            K[1, 2] = (h_ - 1) - K[1, 2]
            Ks_in[i] = K

    scale = float(getattr(args, "input_scale", 1.0) or 1.0)
    if scale != 1.0 and hasattr(ds, "scale_K"):
        frames = [cv2.resize(f, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA) for f in frames]
        Ks_in = [ds.scale_K(K, scale) for K in Ks_in]
        print(f"[scale] input_scale {scale}")

    Es = [np.asarray(ext[n]["E"] if isinstance(ext[n], dict) and "E" in ext[n]
                     else ext[n], float) for n in names]

    # ---- inference --------------------------------------------------------
    print(f"[model] {getattr(args, 'model', '?')} on "
          f"{getattr(args, 'device', '?')}, process_res {args.process_res}")
    model = ds.load_model(args.model, args.device)

    use_prior = (a.pose_prior == "always"
                 or (a.pose_prior == "auto" and len(names) >= 3))
    if use_prior:
        print(f"[prior] extrinsics supplied ({len(names)} views)")
        if len(names) >= 3:
            C = np.stack([-np.asarray(E)[:3, :3].T @ np.asarray(E)[:3, 3]
                          for E in Es])
            sv = np.linalg.svd(C - C.mean(0), compute_uv=False)
            if sv[1] / sv[0] < 0.05:
                print(f"  [warn] camera centres are nearly collinear "
                      f"(singular values {sv[0]:.3f}/{sv[1]:.3f}/{sv[2]:.3f}). "
                      f"Umeyama scale rests on the smallest axis and is fragile.")
        pred = ds.run_inference(model, frames, Ks_in, Es, args)
    else:
        # Intrinsics only. Skips DA3's _align_to_input_extrinsics_intrinsics,
        # so no Umeyama and no minimum camera count. Scale comes back from the
        # deck+floor fit in grade_subsets.py instead.
        print(f"[prior] none -- intrinsics only, uniform across all subsets")
        pred = model.inference(frames, intrinsics=Ks_in,
                               process_res=args.process_res)
    depth, conf, K_out, E_out = unwrap(pred, names, Ks_in, Es)
    # undo the rotation so depth, K and E are all back in the camera's own frame
    if any(flip.values()):
        depth = np.array(depth, copy=True)
        for i, nm in enumerate(names):
            if not flip[nm]:
                continue
            depth[i] = depth[i][::-1, ::-1]
            if conf is not None:
                conf = np.array(conf, copy=True)
                conf[i] = conf[i][::-1, ::-1]
            frames[i] = np.ascontiguousarray(frames[i][::-1, ::-1])
            h_, w_ = depth[i].shape[:2]
            K = np.array(K_out[i], copy=True, dtype=float)
            K[0, 2] = (w_ - 1) - K[0, 2]
            K[1, 2] = (h_ - 1) - K[1, 2]
            K_out = np.array(K_out, copy=True, dtype=float)
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
            Rz = np.diag([-1.0, -1.0, 1.0])
            E_out = np.array(E_out, copy=True, dtype=float)
            E_out[i][:3, :3] = Rz @ E_out[i][:3, :3]
            E_out[i][:3, 3] = Rz @ E_out[i][:3, 3]
        print(f"[roll ] depth rotated back for "
              f"{', '.join(n for n in names if flip[n])}")

    raw_depth = np.array(depth, copy=True)

    # DA3 resizes internally to process_res, so the depth grid is smaller than
    # the rectified frames. backproject() indexes rgb_proc with a mask shaped
    # like the depth, and proc_<cam>.png is meant to be "the image DA3 saw", so
    # both need the images at the depth resolution, not the rectified one.
    dh, dw = depth.shape[-2:]
    rgb_proc = []
    for f in frames:
        if (f.shape[0], f.shape[1]) != (dh, dw):
            f = cv2.resize(f, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb_proc.append(f)
    if (frames[0].shape[0], frames[0].shape[1]) != (dh, dw):
        print(f"[grid ] rectified {frames[0].shape[1]}x{frames[0].shape[0]} "
              f"-> depth grid {dw}x{dh}")
    print(f"[infer] depth {depth.shape}  median "
          f"{np.median(depth[np.isfinite(depth) & (depth > 0)]):.3f} m")

    # ---- optional correction ---------------------------------------------
    if getattr(args, "correction", None) and hasattr(ds, "load_correction"):
        corr, meta = ds.load_correction(
            args.correction, names, getattr(args, "expect_lens", "") or "",
            not getattr(args, "no_view_correction", False))
        depth = ds.correct_depths(depth, names, corr, K_out)
        print(f"[corr ] applied {args.correction}")
    else:
        print("[corr ] none -- grade_subsets.py fits its own alpha/beta")

    # ---- write, in the live layout ---------------------------------------
    for i, n in enumerate(names):
        np.save(out / f"depth_{n}.npy", depth[i])
        np.save(out / f"depth_raw_{n}.npy", raw_depth[i])
        if conf is not None:
            np.save(out / f"conf_{n}.npy", conf[i])
        np.save(out / f"K_{n}.npy", K_out[i])
        np.save(out / f"E_{n}.npy", E_out[i])
        img = np.asarray(rgb_proc[i])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"proc_{n}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[write] {out}")

    if not a.write_clouds:
        print("[cloud] skipped -- only unfiltered depth arrays were written")
    else:
        import open3d as o3d
        clouds, tinted, per_cam, world = ds.backproject(
            names, depth, conf, K_out, E_out, rgb_proc, args)
        fused = o3d.geometry.PointCloud()
        for n in names:
            fused += clouds[n]
            o3d.io.write_point_cloud(str(out / f"cloud_{n}.ply"), clouds[n])
        o3d.io.write_point_cloud(str(out / "fused.ply"), fused)
        print(f"[cloud] fused.ply {len(fused.points)} pts  "
              f"(edge {getattr(args,'edge_thresh','?')}/"
              f"{getattr(args,'edge_dilate','?')}, "
              f"conf p{getattr(args,'conf_percentile','?')}, "
              f"incidence {getattr(args,'max_incidence','?')} deg)")

    print(f"\nNext:\n  python3 inspect_capture.py --capture-dir {out}")


if __name__ == "__main__":
    main()