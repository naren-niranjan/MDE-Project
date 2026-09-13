#!/usr/bin/env python3
"""
check_fusion.py

Diagnose, and repair, the fusion of a subset run written by da3_offline.py.

THE FAULT THIS EXISTS FOR
-------------------------
da3_offline.py with --pose-prior never calls

    model.inference(frames, intrinsics=Ks_in, process_res=...)

and DA3 returns ITS OWN predicted extrinsics, in its own gauge: an arbitrary
global scale and an arbitrary frame. unwrap() substitutes the calibrated Es only
when the model returns None, which it does not, so backproject() places every
view using the prediction rather than the calibration. Four views then land in
four different places and the conveyor appears once per camera, crossing itself.
The depth arrays are unaffected. Only the placement is wrong, so the repair is
a re-fusion from disk and needs no second inference.

WHAT THE REPAIR IS
------------------
DA3 solves the frames of one subset jointly, so there is ONE gauge for the whole
run, not one per view. That gauge has seven degrees of freedom, and six of them
are irrelevant: the rig frame is defined by the calibration, so the rotation and
translation are simply discarded and the calibrated extrinsics used instead. The
seventh, scale, is recovered by comparing the predicted camera centres with the
calibrated ones through a Umeyama fit, and applied to the depth before
back-projection.

    s = argmin || s * C_pred - (R C_cal + t) ||

That anchor is free and needs no ground truth: the baselines are calibration
output, not something under test. It is the same quantity DA3's
align_to_input_ext_scale would have supplied had the prior been on.

A SINGLE-CAMERA RUN HAS NO SUCH ANCHOR
--------------------------------------
One camera has no baseline, so s is unrecoverable from the rig and the run is
metric only as far as DA3's own prior is metric. This is not a limitation of the
repair; it is the reason a one-camera configuration cannot be checked against
anything in its own frame. Such runs are passed through at s = 1 and flagged, so
that the grading shows what the monocular prior is worth on its own rather than
hiding it behind a borrowed scale.

WHAT TO WATCH IN THE OUTPUT
---------------------------
The rotation column. If DA3's predicted relative rotations differ from the
calibrated ones by more than a degree or two, the run is not merely mis-scaled,
it disagreed about where the cameras were pointing, and re-fusing it through the
calibration will produce a clean-looking cloud built on depth maps that were
solved under a different geometry. Those runs are reported and, unless --force,
not rewritten.

EXAMPLE
-------
    python3 check_fusion.py --runs-root runs_da3/scene_a --report
    python3 check_fusion.py --runs-root runs_da3/scene_a --rewrite

Then grade against fused_calib.ply rather than fused.ply:

    python3 grade_subsets.py --gt gt/pointcloud_20260826_095616.ply \\
        --runs-root runs_da3/scene_a --fused-name fused_calib.ply

Keep this file beside da3_stream.py and rigkit.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d is required")

try:
    import da3_stream as ds
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"da3_stream.py must import cleanly: {exc!r}")


CAMS = ["center", "left", "top", "right"]


def parse_run(path: Path):
    name = path.name
    m = re.search(r"_res(\d+)", name)
    subset = re.sub(r"_res\d+$", "", name)
    cams = [c for c in subset.replace(",", "+").split("+") if c]
    if not cams or any(c not in CAMS for c in cams):
        return None
    return {"dir": path, "cams": cams, "res_label": int(m.group(1)) if m else None}


def umeyama_scale(src, dst):
    """Similarity taking src onto dst, returning scale, rotation, translation.

    Two points determine a scale but not a full rotation; three or more
    determine both. With one point neither is determined and the caller must
    not ask.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    var_s = float((S ** 2).sum() / n)
    C = D.T @ S / n
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    s = float((sig * np.diag(W)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    resid = dst - (s * (src @ R.T) + t)
    return s, R, t, float(np.sqrt((resid ** 2).sum(1).mean()))


def relative_rotation_error(E_pred, E_cal, cams):
    """Largest disagreement about where a camera points, relative to the first.

    Absolute rotations live in different frames and cannot be compared. The
    relative ones can, and they are what the fusion actually depends on.
    """
    if len(cams) < 2:
        return 0.0
    worst = 0.0
    R0p = np.asarray(E_pred[cams[0]], float)[:3, :3]
    R0c = np.asarray(E_cal[cams[0]], float)[:3, :3]
    for c in cams[1:]:
        Rp = np.asarray(E_pred[c], float)[:3, :3] @ R0p.T
        Rc = np.asarray(E_cal[c], float)[:3, :3] @ R0c.T
        dR = Rp @ Rc.T
        ang = np.degrees(np.arccos(
            np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        worst = max(worst, float(ang))
    return worst


def load_run(run, calib_dir, reference):
    cams = run["cams"]
    d = run["dir"]
    depth, K, E_pred, conf, rgb = {}, {}, {}, {}, {}
    for c in cams:
        for stem, store in (("depth", depth), ("K", K), ("E", E_pred)):
            p = d / f"{stem}_{c}.npy"
            if not p.exists():
                return None, f"missing {p.name}"
            store[c] = np.load(p).astype(np.float64)
        p = d / f"conf_{c}.npy"
        conf[c] = np.load(p) if p.exists() else None
        p = d / f"proc_{c}.png"
        if p.exists():
            rgb[c] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        else:
            rgb[c] = np.zeros(depth[c].shape + (3,), np.uint8)
    ext = ds.load_extrinsics(Path(calib_dir), cams, reference)
    E_cal = {c: np.asarray(ext[c]["E"], float) for c in cams}
    return {"cams": cams, "depth": depth, "K": K, "E_pred": E_pred,
            "E_cal": E_cal, "conf": conf, "rgb": rgb}, None


def diagnose(data):
    cams = data["cams"]
    C_pred = np.stack([ds.camera_centre(data["E_pred"][c]) for c in cams])
    C_cal = np.stack([ds.camera_centre(data["E_cal"][c]) for c in cams])
    out = {"n_cams": len(cams),
           "rot_err_deg": round(relative_rotation_error(
               data["E_pred"], data["E_cal"], cams), 3)}
    if len(cams) == 1:
        out.update({"scale": 1.0, "scale_source": "none, single camera",
                    "centre_resid_mm": None})
        return out
    s, R, t, resid = umeyama_scale(C_pred, C_cal)
    if not np.isfinite(s) or s <= 1e-6:
        out.update({"scale": 1.0, "scale_source": "degenerate, held at 1",
                    "centre_resid_mm": None})
        return out
    out.update({"scale": round(float(s), 6),
                "scale_source": f"{len(cams)} calibrated baselines",
                "centre_resid_mm": round(resid * 1e3, 3)})
    # A pair has one baseline, so the residual is zero by construction and says
    # nothing. Only three or more cameras make it a test.
    if len(cams) == 2:
        out["centre_resid_mm"] = None
    return out


def intrinsic_scales(data, run, args):
    """Per-camera depth scale from the focal ratio, derived not fitted.

    DA3 returns its own intrinsics, and with no pose prior it evidently
    discards the ones it was given: on this rig it came back at 0.59 to 0.64 of
    the calibrated focal lengths, an implied 59 degree horizontal field where
    the 12 mm lenses give 39. That is the generic monocular prior meeting a
    narrow lens.

    A wrong focal makes the predicted depth wrong by the same factor, because
    the reconstruction is only self-consistent when z / fx matches the scene.
    So the fix is exact:

        s_cam = fx_calibrated / fx_DA3

    and it is PER CAMERA. The ratios differ across this rig by 7.4 per cent,
    which is the layering: four cameras each scaling the same deck differently
    is four decks. A single fitted gauge cannot remove that, however well it is
    measured, because there is no single number to remove.
    """
    try:
        import pin_markers as pm
    except Exception:  # noqa: BLE001
        return None
    cams = data["cams"]
    K_cal = pm.calibrated_K(run, cams, args)
    out = {}
    for cam in cams:
        kp = run / f"K_{cam}.npy"
        if cam not in K_cal or not kp.exists():
            return None
        out[cam] = float(K_cal[cam][0, 0]
                         / np.load(kp).astype(float)[0, 0])
    return out


def deck_per_camera_scales(data, run, args):
    """The scale that lands EACH camera's own deck on the belt plane.

    A single gauge per run cannot remove layering, because layering is not one
    number: measured across six camera pairs the inter-view spread scales with
    baseline at r = +0.96, slope 84 mm per metre, which under
    layering = (1 - k) * b means a residual per-camera depth-scale error of
    about 8.4 per cent. One median gauge leaves that untouched by construction.

    Anchoring each camera separately needs no fitting. The belt plane's position
    is known from the ground-truth pin, so for camera c

        s_c solves   median over c's points of  h(s_c)  ==  0

    where h is height above the belt plane and the points are scaled about that
    camera's own centre. The median is monotone in s, so a bisection settles it.

    This is a TEST as much as a correction. If layering really is per-camera
    scale error, anchoring each camera collapses it towards zero. If it does
    not, the 8.4 per cent has another cause and that is worth knowing.
    """
    try:
        import da3_stream as ds
        import rigkit  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    belt = json.loads(Path(args.belt).read_text())
    ex = np.asarray(belt["x_axis"], float)
    ey = np.asarray(belt["y_axis"], float)
    ex /= np.linalg.norm(ex)
    ey = ey - ex * float(ey @ ex)
    ey /= np.linalg.norm(ey)
    n = np.cross(ex, ey)
    n /= np.linalg.norm(n)
    centre = np.asarray(belt["centre_m"], float)
    if float(n @ (np.zeros(3) - centre)) < 0:
        n, ey = -n, -ey
    target = float(centre @ n)
    L, W = belt["length_m"] / 2, belt["width_m"] / 2

    ext = ds.load_extrinsics(Path(args.calib_dir), data["cams"],
                             args.reference)
    out = {}
    for cam in data["cams"]:
        f = run / f"cloud_calib_{cam}.ply"
        if not f.exists():
            return None
        P = rigkit.load_cloud(str(f))
        C = ds.camera_centre(np.asarray(ext[cam]["E"], float))

        def med_h(s, band=None):
            Q = C + (P - C) * s
            rel = Q - centre
            on = ((np.abs(rel @ ex) <= L) & (np.abs(rel @ ey) <= W))
            h = (Q @ n - target)[on]
            if band is not None:
                near = np.abs(h) <= band
                if int(near.sum()) > 500:
                    h = h[near]
            return float(np.median(h)) if len(h) else float("nan")

        lo, hi = 0.5, 2.5
        fa, fb = med_h(lo), med_h(hi)
        if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fa * med_h(mid) <= 0:
                hi = mid
            else:
                lo, fa = mid, med_h(mid)
        s = 0.5 * (lo + hi)
        for _ in range(3):                # settle onto the deck itself
            a2, b2 = s * 0.9, s * 1.1
            f2 = med_h(a2, args.deck_band_m)
            if not np.isfinite(f2) or f2 * med_h(b2, args.deck_band_m) > 0:
                break
            for _ in range(40):
                m2 = 0.5 * (a2 + b2)
                if f2 * med_h(m2, args.deck_band_m) <= 0:
                    b2 = m2
                else:
                    a2, f2 = m2, med_h(m2, args.deck_band_m)
            s = 0.5 * (a2 + b2)
        out[cam] = float(s)
    return out


def refuse(data, diag, args, per_cam_scale=None):
    """Re-fuse through the calibrated extrinsics, at the recovered scale.

    The filtering is da3_stream.backproject, unchanged, so the repaired cloud is
    filtered exactly as the live pipeline filters and the comparison against a
    live capture stays meaningful.
    """
    cams = data["cams"]
    s = float(diag["scale"])
    if per_cam_scale:
        # A camera can be missing from the gauge: it is built only from the
        # cameras that located a tag, and a tag can be out of frame, occluded
        # by a parcel, or too few pixels to decode. Falling back to the median
        # of the cameras that did find one is better than dropping the run, but
        # it is a fallback and says so, because that camera's own scale error
        # is then unmeasured and will show up as layering.
        s_med = float(np.median(list(per_cam_scale.values())))
        missing = [c for c in cams if c not in per_cam_scale]
        if missing:
            print(f"        no tag in {', '.join(missing)}; using the median "
                  f"of the others, {s_med:.4f}")
        depth = np.stack([data["depth"][c] * per_cam_scale.get(c, s_med)
                          for c in cams])
    else:
        depth = np.stack([data["depth"][c] * s for c in cams])
    conf = (np.stack([data["conf"][c] for c in cams])
            if all(data["conf"][c] is not None for c in cams) else None)
    K = np.stack([data["K"][c] for c in cams])
    E = np.stack([data["E_cal"][c] for c in cams])
    rgb = np.stack([data["rgb"][c] for c in cams])
    fake = SimpleNamespace(conf_percentile=args.conf_percentile,
                           edge_thresh=args.edge_thresh,
                           edge_dilate=args.edge_dilate,
                           max_incidence=args.max_incidence,
                           colour_by_camera=True)
    clouds, tinted, per_cam, world = ds.backproject(cams, depth, conf, K, E,
                                                    rgb, fake)
    return clouds, tinted, per_cam, world


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check DA3's predicted extrinsics against the calibration "
                    "and re-fuse each run in the calibrated frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", default=None)
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--reference", default="center")
    ap.add_argument("--report", action="store_true",
                    help="diagnose only, write nothing")
    ap.add_argument("--rewrite", action="store_true",
                    help="write fused_calib.ply and cloud_calib_<cam>.ply")
    ap.add_argument("--max-rot-err-deg", type=float, default=2.0,
                    help="relative rotation disagreement above which a run is "
                         "not rewritten. Beyond this the depth maps were "
                         "solved under a geometry the calibration does not "
                         "describe, and re-fusing them only hides that")
    ap.add_argument("--force", action="store_true",
                    help="rewrite even past --max-rot-err-deg")
    ap.add_argument("--scale-mode",
                    choices=("deck-per-camera", "intrinsics", "baseline",
                             "deck", "file", "none"),
                    default="intrinsics",
                    help="where the run's global scale comes from. 'deck' "
                         "reads gauge_probe.json, which identifies the "
                         "conveyor by its own width and works for a single "
                         "camera; it is also the anchor the production "
                         "pipeline already uses, via --seg-plane-distance. "
                         "'baseline' uses the Umeyama fit on the camera "
                         "centres, which on this rig leans on the axial "
                         "baseline DA3 gets wrong. 'none' holds the scale at 1. "
                         "'file' reads metric_fit.json from pin_alignment.py, "
                         "which measures the gauge against the rc_viscore "
                         "ground truth instead of against a plane fitted in "
                         "the reconstruction; four plane-identification "
                         "schemes were tried on this cell and three picked the "
                         "floor. 'intrinsics' is better than all of them: it "
                         "takes the scale from the ratio of calibrated to "
                         "returned focal length, PER CAMERA, which is derived "
                         "rather than fitted and is the only mode that can "
                         "remove the layering")
    ap.add_argument("--gauge", type=Path, default=Path("gauge_probe.json"),
                    help="probe_gauge.py output, required by --scale-mode deck")
    ap.add_argument("--gauge-from", default="center,left,right",
                    help="cameras whose deck reading sets the run's gauge. "
                         "One joint solve has ONE gauge, so a camera that "
                         "disagrees is carrying a per-view depth error, not a "
                         "different scale. On this rig top reads 7.2 per cent "
                         "high in 11 of 11 runs; folding it into the median "
                         "would spread its own error across the views that "
                         "were right. Left in the fusion, left in the record, "
                         "excluded from the anchor. Pass 'all' to include it")
    ap.add_argument("--max-gauge-spread", type=float, default=0.10,
                    help="refuse a run whose cameras disagree about their own "
                         "shared gauge by more than this; they came out of one "
                         "joint solve and must agree")
    ap.add_argument("--belt", type=Path, default=Path("belt.json"),
                    help="the frozen footprint, for --scale-mode "
                         "deck-per-camera")
    ap.add_argument("--deck-band-m", type=float, default=0.05,
                    help="band about the current estimate used to settle each "
                         "camera onto the deck rather than onto the whole "
                         "cloud's median")
    ap.add_argument("--rect-mode", default="common",
                    choices=("common", "roi", "full"),
                    help="must match the rectification the capture used, or "
                         "the rebuilt intrinsics belong to a different crop")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=2)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--out", type=Path, default=Path("fusion_check.json"))
    args = ap.parse_args()

    gauge = {}
    if args.scale_mode in ("deck", "file"):
        if not args.gauge.exists():
            raise SystemExit(
                f"--scale-mode {args.scale_mode} needs {args.gauge}; run "
                f"pin_alignment.py --refine --refine-all first")
        raw = json.loads(args.gauge.read_text())
        # pin_alignment writes {run: {...}}, probe_gauge wrote [{"run": ...}].
        gauge = (raw if isinstance(raw, dict)
                 else {r["run"]: r for r in raw})
        # Gauge files are keyed by directory NAME, and the same name exists
        # under several roots: left+top+right_res1008 lives in cfg1008, in
        # resweep and in resweep_prior. Applying one root's gauge to another's
        # run is silent and wrong, so the roots are compared before anything is
        # scaled.
        if args.runs_root:
            here = {d.name for d in Path(args.runs_root).iterdir()
                    if d.is_dir()}
            shared = here & set(gauge)
            if shared and len(shared) < len(here) * 0.5:
                print(f"[WARN] {args.gauge} names {len(shared)} of the "
                      f"{len(here)} runs under {args.runs_root}. Gauge files "
                      f"are keyed by directory name and the same name exists "
                      f"under several roots, so check this is the right file "
                      f"for this root.")
            src_root = {v.get("root") for v in gauge.values()
                        if isinstance(v, dict) and v.get("root")}
            if src_root and str(args.runs_root) not in src_root:
                raise SystemExit(
                    f"{args.gauge} was written for {sorted(src_root)}, not "
                    f"{args.runs_root}. Re-run the gauge step against this "
                    f"root, or pass the matching file.")

    runs = []
    for d in (args.run or []):
        r = parse_run(Path(d))
        if r:
            runs.append(r)
    if args.runs_root:
        for d in sorted(Path(args.runs_root).iterdir()):
            if d.is_dir():
                r = parse_run(d)
                if r:
                    runs.append(r)
    if not runs:
        raise SystemExit("no runs found")

    print(f"{'run':34s} {'n':>2s} {'scale':>9s} {'centres':>9s} "
          f"{'rot err':>8s}  verdict")
    print("-" * 84)
    records, rewritten, refused = [], 0, 0
    for run in sorted(runs, key=lambda r: (len(r["cams"]), r["dir"].name)):
        data, why = load_run(run, args.calib_dir, args.reference)
        if data is None:
            print(f"{run['dir'].name:34s} -- {why}")
            continue
        diag = diagnose(data)
        diag["run"] = run["dir"].name
        diag["scale_baseline"] = diag["scale"]

        if args.scale_mode == "none":
            diag["scale"] = 1.0
            diag["scale_source"] = "held at 1"
        elif args.scale_mode == "deck-per-camera":
            sc = deck_per_camera_scales(data, run["dir"], args)
            if sc is None:
                diag["scale_source"] = ("deck-per-camera needs "
                                        "cloud_calib_<cam>.ply and a belt "
                                        "file; run --scale-mode none --rewrite "
                                        "first")
                diag["scale"] = float("nan")
            else:
                diag["per_camera_scale"] = {k: round(v, 5)
                                            for k, v in sc.items()}
                diag["scale"] = float(np.median(list(sc.values())))
                v = np.array(list(sc.values()))
                diag["scale_spread"] = round(
                    float((v.max() - v.min()) / v.mean()), 4)
                diag["scale_source"] = ("each camera's own deck on the belt "
                                        "plane")
        elif args.scale_mode == "intrinsics":
            sc = intrinsic_scales(data, run["dir"], args)
            if sc is None:
                diag["scale_source"] = "intrinsics, could not rebuild them"
                diag["scale"] = float("nan")
            else:
                diag["per_camera_scale"] = {k: round(v, 5)
                                            for k, v in sc.items()}
                diag["scale"] = float(np.median(list(sc.values())))
                v = np.array(list(sc.values()))
                diag["scale_spread"] = round(
                    float((v.max() - v.min()) / v.mean()), 4)
                diag["scale_source"] = ("focal ratio per camera, "
                                        + "+".join(f"{k}:{v:.3f}"
                                                   for k, v in sc.items()))
        elif args.scale_mode == "file":
            g = gauge.get(run["dir"].name)
            if g is None or not np.isfinite(float(g.get("scale", float("nan")))):
                diag["scale_source"] = f"file, absent from {args.gauge}"
                diag["scale"] = float("nan")
            else:
                diag["scale"] = float(g["scale"])
                diag["scale_source"] = "file, via " + args.gauge.name
                if g.get("per_camera_scale"):
                    diag["per_camera_scale"] = g["per_camera_scale"]
                    diag["scale_source"] += ", per camera"
                for k in ("neighbour_fraction", "deck_offset_mm"):
                    if k in g:
                        diag[k] = g[k]
        elif args.scale_mode == "deck":
            g = gauge.get(run["dir"].name)
            want = ([c.strip() for c in args.gauge_from.split(",")]
                    if args.gauge_from != "all" else list(run["cams"]))
            got = {c: v["g_standoff"]
                   for c, v in ((g or {}).get("per_camera") or {}).items()
                   if v.get("identified") and c in want}
            if not got:
                diag["scale_source"] = ("deck, no anchor camera identified the "
                                        "conveyor")
                diag["scale"] = float("nan")
            else:
                vals = list(got.values())
                spread = ((max(vals) - min(vals)) / float(np.mean(vals))
                          if len(vals) > 1 else 0.0)
                if spread > args.max_gauge_spread:
                    diag["scale_source"] = (f"deck, anchor cameras disagree by "
                                            f"{spread * 100:.1f} per cent")
                    diag["scale"] = float("nan")
                else:
                    diag["scale"] = float(np.median(vals))
                    diag["scale_source"] = (f"conveyor width, "
                                            f"{'+'.join(sorted(got))}")
                    diag["gauge_spread"] = round(float(spread), 4)
                    diag["gauge_per_camera"] = {k: round(v, 5)
                                                for k, v in got.items()}

        ok = (diag["rot_err_deg"] <= args.max_rot_err_deg
              and np.isfinite(diag["scale"]))
        if not np.isfinite(diag["scale"]):
            verdict = diag["scale_source"]
        elif len(run["cams"]) == 1 and args.scale_mode != "deck":
            verdict = "single camera, scale is DA3's own prior"
        elif not ok:
            verdict = "ROTATIONS DISAGREE, not repairable by re-fusion"
        elif abs(diag["scale"] - 1.0) < 0.005:
            verdict = "already at the calibrated scale"
        else:
            verdict = f"rescale by {diag['scale']:.4f} and re-fuse"
        res = diag["centre_resid_mm"]
        print(f"{run['dir'].name:34s} {diag['n_cams']:>2d} "
              f"{diag['scale']:>9.5f} "
              + (f"{res:>7.2f}mm" if res is not None else f"{'-':>9s}")
              + f" {diag['rot_err_deg']:>7.2f}d  {verdict}")

        if args.rewrite and (ok or args.force):
            clouds, tinted, per_cam, _ = refuse(
                data, diag, args, diag.get("per_camera_scale"))
            fused = o3d.geometry.PointCloud()
            for c in run["cams"]:
                fused += clouds[c]
                o3d.io.write_point_cloud(
                    str(run["dir"] / f"cloud_calib_{c}.ply"), clouds[c])
            if args.voxel > 0:
                fused = fused.voxel_down_sample(args.voxel)
            o3d.io.write_point_cloud(str(run["dir"] / "fused_calib.ply"), fused)
            if tinted:
                merged = o3d.geometry.PointCloud()
                for p in tinted:
                    merged += p
                o3d.io.write_point_cloud(
                    str(run["dir"] / "fused_calib_by_camera.ply"), merged)
            diag["n_points"] = len(fused.points)
            diag["per_camera"] = per_cam
            rewritten += 1
        elif args.rewrite:
            refused += 1
        records.append(diag)

    args.out.write_text(json.dumps(records, indent=2, default=float))
    print(f"\n{len(records)} run(s) checked, {rewritten} rewritten, "
          f"{refused} refused on rotation disagreement")
    print(f"written: {args.out}")
    if rewritten:
        print("\nOpen fused_calib_by_camera.ply first. One colour per view: if "
              "the conveyor still shows once per camera, the fault is not the "
              "gauge and no re-fusion will fix it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())