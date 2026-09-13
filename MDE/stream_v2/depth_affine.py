#!/usr/bin/env python3
"""
depth_affine.py  (v4)

Per-camera inverse-depth affine correction against two or more ChArUco planes,
now with flying-pixel and grazing-incidence rejection done in depth space before
backprojection.

    1/z' = A(x~,y~) / z  +  B(x~,y~)

Why the edge filter belongs here
--------------------------------
After v3 the offset degree of freedom is closed: biases came out -7.0, +4.3 and
+0.6 mm across the three non-reference cameras. What is left is spread, and it is
wildly uneven -- left's |d| median is 37.4 mm against right's 7.7 mm -- even
though both cameras fit the GT planes equally well (deck RMS 6.9 vs 6.8 mm, floor
17.8 vs 17.2 mm). A transform error cannot do that: it would show up on the
planes too. Something specific to parcel surfaces is responsible, and at box
edges the obvious candidate is flying pixels, which earlier work put at roughly
a quarter of all above-deck points.

The residual metric is also structurally biased toward them. overlap_residual
builds its z-buffer with np.minimum.at, so it takes the NEAREST sample on each
ray -- exactly the interpolated fiction hanging in front of a box edge. Filtering
one side would not be enough; the filter has to run before the clouds are
written, so both the query set and the buffer are clean. It also improves the
fused cloud itself, which is the actual deliverable.

Two filters, both in depth space where the neighbourhood structure still exists:

  discontinuity   a pixel is dropped when the spread of depth in its 3x3
                  neighbourhood exceeds --edge-rel-thresh of its own depth
                  (default 1 %, so ~31 mm at 3.08 m). The surviving mask is then
                  eroded by --erode pixels, because the interpolated pixels sit
                  beside the discontinuity, not on it.
  incidence       a pixel is dropped when the angle between its local surface
                  normal and the viewing ray exceeds --max-incidence. Grazing
                  samples carry the least information and the most error. Set to
                  90 to disable, and check what fraction that removes before
                  trusting it -- a box side seen from above is near-tangential by
                  geometry, so an aggressive limit deletes real side faces.

Both are off by default only in the sense that they are reported: every run
prints how many points each filter removed, so you can see the cost.

Example
-------
  python depth_affine.py \
      --capture runs/live_20260817_102732/capture_00002 \
      --gt gt_planes.json --planes deck,floor --reference center \
      --dump-per-camera runs/live_20260817_102732/capture_00002/percam_clean \
      --out    runs/live_20260817_102732/capture_00002/fused_clean.ply \
      --report runs/live_20260817_102732/capture_00002/depth_affine_clean.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not found. Activate the da3 environment.")


# ----------------------------------------------------------------------------
# plumbing
# ----------------------------------------------------------------------------

def to_4x4(M):
    M = np.asarray(M, dtype=np.float64)
    if M.size == 12:
        T = np.eye(4)
        T[:3, :4] = M.reshape(3, 4)
        return T
    if M.size == 16:
        return M.reshape(4, 4)
    raise ValueError(f"extrinsic has {M.size} elements, expected 12 or 16")


def load_camera(cap: Path, cam: str, depth_key: str):
    depth = np.squeeze(np.load(cap / f"{depth_key}_{cam}.npy").astype(np.float64))
    K = np.load(cap / f"K_{cam}.npy").astype(np.float64).reshape(3, 3)
    E = to_4x4(np.load(cap / f"E_{cam}.npy"))
    cp = cap / f"conf_{cam}.npy"
    conf = np.load(cp).astype(np.float64) if cp.exists() else None
    if conf is not None and conf.shape != depth.shape:
        conf = None
    return depth, K, E, conf


# ----------------------------------------------------------------------------
# depth-space filtering
# ----------------------------------------------------------------------------

def _shifted_stack(a):
    """The nine 3x3 neighbours of every pixel, edges replicated."""
    pads = np.pad(a, 1, mode="edge")
    h, w = a.shape
    return np.stack([pads[dy:dy + h, dx:dx + w]
                     for dy in range(3) for dx in range(3)], axis=0)


def point_map(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x * depth, y * depth, depth], axis=-1)


def discontinuity_mask(depth, valid, rel_thresh):
    """True where the 3x3 depth spread exceeds rel_thresh of the centre depth."""
    d = np.where(valid, depth, np.nan)
    st = _shifted_stack(d)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(st, axis=0) - np.nanmin(st, axis=0)
    return np.nan_to_num(spread, nan=np.inf) > rel_thresh * np.maximum(depth, 1e-6)


def erode(mask, k):
    """Shrink a boolean mask by k pixels (4-connected, iterated)."""
    out = mask.copy()
    for _ in range(max(k, 0)):
        st = _shifted_stack(out.astype(np.float64))
        out = st.min(axis=0) > 0.5
    return out


def incidence_deg(depth, valid, K):
    """Angle between the local surface normal and the viewing ray, per pixel."""
    P = point_map(depth, K)
    Pu = np.gradient(P, axis=1)
    Pv = np.gradient(P, axis=0)
    n = np.cross(Pu, Pv)
    nn = np.linalg.norm(n, axis=-1)
    view = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.abs(np.sum(n / np.maximum(nn, 1e-12)[..., None] * view, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    return np.where(valid & np.isfinite(ang), ang, 90.0)


def backproject(depth, K, conf, conf_pct, dmin, dmax,
                edge_rel_thresh, erode_px, max_incidence):
    valid = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    stages = {"valid_depth": int(valid.sum())}

    if conf is not None and conf_pct is not None and valid.any():
        valid &= conf >= np.percentile(conf[valid], conf_pct)
        stages["after_confidence"] = int(valid.sum())

    if edge_rel_thresh is not None and edge_rel_thresh > 0:
        valid &= ~discontinuity_mask(depth, valid, edge_rel_thresh)
        stages["after_discontinuity"] = int(valid.sum())

    if erode_px:
        valid = erode(valid, erode_px)
        stages["after_erode"] = int(valid.sum())

    if max_incidence is not None and max_incidence < 90.0:
        valid &= incidence_deg(depth, valid, K) <= max_incidence
        stages["after_incidence"] = int(valid.sum())

    P = point_map(depth, K)[valid]
    stages["kept"] = int(len(P))
    return P, stages


def as_pcd(P):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(P)
    return p


# ----------------------------------------------------------------------------
# GT
# ----------------------------------------------------------------------------

def _plane(n, perp):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return n, float(-perp if n[2] > 0 else perp)


def normalise_gt(gt, thickness_sign, thickness_override=None):
    planes, notes = {}, []
    if "reference_planes" in gt:
        for p in gt["reference_planes"]:
            n, d = _plane(p["normal"], float(p["perp_m"]))
            planes[p["label"]] = {"normal": n.tolist(), "d": d,
                                  "perp_m": float(p["perp_m"]), "source": "derived"}
        return gt.get("reference_camera"), planes, notes
    th = thickness_override
    if th is None:
        th = float(gt.get("board", {}).get("thickness_m", 0.0))
    sign = +1.0 if thickness_sign == "away" else -1.0
    for label, e in gt.items():
        if not isinstance(e, dict) or "normal_mean" not in e or "d_mean" not in e:
            continue
        n = np.asarray(e["normal_mean"], float)
        n = n / np.linalg.norm(n)
        perp = abs(float(e["d_mean"])) + sign * th
        n_s, d_s = _plane(n, perp)
        planes[label] = {"normal": n_s.tolist(), "d": d_s, "perp_m": perp,
                         "source": "raw (thickness applied here)"}
        own = e.get("surface_perp_dist_m")
        if own is not None and abs(abs((perp - float(own)) * 1000.0) - 2000.0 * th) < 5.0:
            notes.append(f"{label}: GT file's surface_perp_dist_m disagrees by two "
                         f"board thicknesses - sign convention differs upstream")
    return gt.get("reference_camera"), planes, notes


def plane_to_world(n_c, d_c, T_world_cam):
    T_cw = np.linalg.inv(T_world_cam)
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    n_w = R.T @ n_c
    d_w = float(n_c @ t + d_c)
    k = np.linalg.norm(n_w)
    n_w, d_w = n_w / k, d_w / k
    if d_w > 0:
        n_w, d_w = -n_w, -d_w
    return n_w, d_w


def plane_to_frame(n_w, d_w, T_world_cam):
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    n_c = R.T @ n_w
    d_c = float(n_w @ t + d_w)
    k = np.linalg.norm(n_c)
    return n_c / k, d_c / k


def fit_plane(P, thresh=0.006, iters=3000):
    model, inl = as_pcd(P).segment_plane(thresh, 3, iters)
    n = np.array(model[:3], float)
    k = np.linalg.norm(n)
    return n / k, model[3] / k, np.asarray(inl)


def select_plane_points(P, n_c, d_c, band, thresh=0.006, min_pts=500):
    sd = P @ n_c + d_c
    cand = np.abs(sd) < band
    if cand.sum() < min_pts:
        return None
    _, _, inl = fit_plane(P[cand], thresh)
    return np.flatnonzero(cand)[inl]


# ----------------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------------

MODELS = [("linear_A_linear_B", [0, 1, 2, 3, 4, 5]),
          ("linear_A_const_B",  [0, 1, 2, 3]),
          ("const_A_const_B",   [0, 3])]
SPATIAL_COLS = (1, 2, 4, 5)


def design(P, scale):
    x = P[:, 0] / P[:, 2] / scale[0]
    y = P[:, 1] / P[:, 2] / scale[1]
    z = P[:, 2]
    return np.stack([1.0 / z, x / z, y / z, np.ones_like(z), x, y], axis=1)


def rhs_for_plane(P, n_c, d_c):
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    m = n_c[0] * x + n_c[1] * y + n_c[2]
    return -m / d_c


def ridge_solve(F, y, cols, ridge):
    Fs = F[:, cols]
    n = len(cols)
    G = Fs.T @ Fs
    pen = np.zeros((n, n))
    ts = max(np.trace(G) / max(n, 1), 1e-12)
    for i, c in enumerate(cols):
        if c in SPATIAL_COLS:
            pen[i, i] = ridge * ts
    full = np.zeros(6)
    full[cols] = np.linalg.solve(G + pen, Fs.T @ y)
    return full


def apply_affine(P, theta, scale):
    x = P[:, 0] / P[:, 2] / scale[0]
    y = P[:, 1] / P[:, 2] / scale[1]
    z = P[:, 2]
    A = theta[0] + theta[1] * x + theta[2] * y
    B = theta[3] + theta[4] * x + theta[5] * y
    inv = A / z + B
    bad = ~np.isfinite(inv) | (inv <= 1e-9)
    zc = np.where(bad, z, 1.0 / np.where(bad, 1.0, inv))
    out = np.stack([P[:, 0] / P[:, 2] * zc, P[:, 1] / P[:, 2] * zc, zc], axis=1)
    return out, bad, zc / z


def plane_residual(P, n_c, d_c):
    r = P @ n_c + d_c
    return {"median_mm": float(np.median(r) * 1000.0),
            "rms_mm": float(np.sqrt((r ** 2).mean()) * 1000.0), "n": int(len(r))}


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--planes", default="deck,floor")
    ap.add_argument("--cameras", default="left,center,right,top")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--depth-key", default="depth", choices=["depth", "depth_raw"])
    ap.add_argument("--thickness-sign", default="away", choices=["away", "toward"])
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--band", type=float, default=0.30)
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=8.0)
    ap.add_argument("--edge-rel-thresh", type=float, default=0.01,
                    help="drop a pixel when its 3x3 depth spread exceeds this "
                         "fraction of its depth; 0 disables")
    ap.add_argument("--erode", type=int, default=2,
                    help="erode the surviving mask by this many pixels")
    ap.add_argument("--max-incidence", type=float, default=90.0,
                    help="drop pixels above this surface incidence in degrees; "
                         "90 disables. Check the cost before lowering it")
    ap.add_argument("--gain-limit", type=float, default=0.10)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--amplification-warn", type=float, default=1.25)
    ap.add_argument("--extrinsic-convention", default="cam_from_world",
                    choices=["world_from_cam", "cam_from_world"])
    ap.add_argument("--dump-per-camera", default=None)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cap = Path(args.capture)
    gt_ref, gt_planes, notes = normalise_gt(
        json.loads(Path(args.gt).read_text()), args.thickness_sign, args.thickness)
    for n in notes:
        print(f"NOTE: {n}", file=sys.stderr)

    labels = [s.strip() for s in args.planes.split(",") if s.strip()]
    if len(labels) < 2:
        sys.exit("--planes needs at least two labels")
    missing = [l for l in labels if l not in gt_planes]
    if missing:
        sys.exit(f"no GT plane(s) {missing}; have {sorted(gt_planes)}")

    perps = np.array([gt_planes[l]["perp_m"] for l in labels])
    lever = float(np.max(1.0 / perps) - np.min(1.0 / perps))
    print("anchors: " + ", ".join(f"{l} {gt_planes[l]['perp_m']:.4f} m" for l in labels))
    print(f"inverse-depth lever {lever:.4f} 1/m over "
          f"{(perps.max() - perps.min()) * 1000:.0f} mm"
          + ("   (short lever - A and B weakly separated)" if lever < 0.05 else ""))
    if len(labels) == 2:
        print("Two planes leave no residual to test the model with.")

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    ref = args.reference or gt_ref
    if ref not in cams:
        sys.exit(f"reference {ref!r} not in {cams}")

    report = {"capture": str(cap), "gt_file": args.gt, "planes": labels,
              "plane_perp_m": {l: gt_planes[l]["perp_m"] for l in labels},
              "inverse_depth_lever_1_per_m": lever, "reference": ref,
              "filters": {"edge_rel_thresh": args.edge_rel_thresh,
                          "erode_px": args.erode,
                          "max_incidence_deg": args.max_incidence},
              "ridge": args.ridge, "cameras": {}}

    Es, pts = {}, {}
    print()
    for c in cams:
        depth, K, E, conf = load_camera(cap, c, args.depth_key)
        Es[c] = E
        P, stages = backproject(depth, K, conf, args.conf_percentile,
                                args.depth_min, args.depth_max,
                                args.edge_rel_thresh, args.erode,
                                args.max_incidence)
        pts[c] = P
        report["cameras"][c] = {"filter_stages": stages}
        base = stages.get("after_confidence", stages["valid_depth"])
        print(f"{c:<7} filter {base} -> {stages['kept']} points "
              f"({100.0 * stages['kept'] / max(base, 1):.1f} % kept)")

    def T_of(c):
        return np.linalg.inv(Es[c]) if args.extrinsic_convention == "cam_from_world" \
            else Es[c]

    planes_w = {l: plane_to_world(np.asarray(gt_planes[l]["normal"], float),
                                  float(gt_planes[l]["d"]), T_of(ref))
                for l in labels}

    print()
    corrected, all_ok = {}, True
    for c in cams:
        info = report["cameras"][c]
        P = pts[c]

        sel = {}
        for l in labels:
            n_c, d_c = plane_to_frame(*planes_w[l], T_of(c))
            idx = select_plane_points(P, n_c, d_c, args.band)
            info[f"{l}_points"] = 0 if idx is None else int(len(idx))
            if idx is not None:
                sel[l] = (idx, n_c, d_c)
                info[f"{l}_before"] = plane_residual(P[idx], n_c, d_c)

        if len(sel) < 2:
            print(f"  {c}: found {len(sel)} of {len(labels)} planes - left uncorrected")
            info["status"] = f"only {len(sel)} plane(s) found"
            corrected[c] = P
            all_ok = False
            continue
        info["planes_used"] = sorted(sel)

        fit_idx = np.concatenate([v[0] for v in sel.values()])
        fx_ = P[fit_idx, 0] / P[fit_idx, 2]
        fy_ = P[fit_idx, 1] / P[fit_idx, 2]
        scale = (max(np.abs(fx_).max(), 1e-3), max(np.abs(fy_).max(), 1e-3))
        ax = np.abs(P[:, 0] / P[:, 2]).max()
        ay = np.abs(P[:, 1] / P[:, 2]).max()
        info["field_coverage"] = [float(scale[0] / max(ax, 1e-9)),
                                 float(scale[1] / max(ay, 1e-9))]

        blocks = []
        for l, (idx, n_c, d_c) in sel.items():
            w = 1.0 / np.sqrt(len(idx))
            blocks.append((design(P[idx], scale) * w,
                           rhs_for_plane(P[idx], n_c, d_c) * w))
        F = np.concatenate([b[0] for b in blocks], axis=0)
        y = np.concatenate([b[1] for b in blocks], axis=0)

        chosen = None
        for name, cols in MODELS:
            theta = ridge_solve(F, y, cols, args.ridge)
            Pc, bad, gain = apply_affine(P, theta, scale)
            cand = {"model": name, "theta": theta.tolist(),
                    "gain_fit_min": float(gain[fit_idx].min()),
                    "gain_fit_max": float(gain[fit_idx].max()),
                    "gain_all_min": float(gain.min()),
                    "gain_all_max": float(gain.max()),
                    "nonfinite_points": int(bad.sum())}
            if (1 - args.gain_limit < gain.min()) and (gain.max() < 1 + args.gain_limit):
                chosen = (cand, Pc)
                break
            info.setdefault("rejected_models", []).append(cand)

        if chosen is None:
            print(f"  {c}: every model left the gain outside "
                  f"+/-{args.gain_limit:.0%} - left uncorrected")
            info["status"] = "all models rejected"
            corrected[c] = P
            all_ok = False
            continue

        cand, Pc = chosen
        info.update(cand)
        info["status"] = "corrected"
        info["noise_amplification"] = float(cand["theta"][0])
        for l, (idx, n_c, d_c) in sel.items():
            info[f"{l}_after"] = plane_residual(Pc[idx], n_c, d_c)
        corrected[c] = Pc

        amp = cand["theta"][0]
        warn = "  <-- amplifies depth noise" if amp > args.amplification_warn else ""
        print(f"{c:<7} {cand['model']:<18} a0={amp:.5f} b0={cand['theta'][3]:+.5f}  "
              + "  ".join(f"{l} {info[f'{l}_before']['rms_mm']:.1f}->"
                          f"{info[f'{l}_after']['rms_mm']:.1f}" for l in sorted(sel))
              + f" mm RMS{warn}")

    report["all_cameras_corrected"] = all_ok
    if not all_ok and not args.allow_partial:
        print("\nAt least one camera is uncorrected. Not writing a fused cloud: an\n"
              "uncorrected deck falls inside the above-deck band, so residuals from\n"
              "it compare decks rather than parcels. Use --allow-partial to override.")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
            print(f"report -> {args.report}")
        return

    clouds = {}
    for c in cams:
        T = T_of(c)
        clouds[c] = as_pcd((T[:3, :3] @ corrected[c].T).T + T[:3, 3])

    n_w, d_w = planes_w[labels[0]]
    if args.dump_per_camera:
        dd = Path(args.dump_per_camera)
        dd.mkdir(parents=True, exist_ok=True)
        for c in cams:
            o3d.io.write_point_cloud(str(dd / f"cloud_{c}.ply"), clouds[c])
        (dd / "deck_plane.json").write_text(json.dumps(
            {"normal": n_w.tolist(), "d": float(d_w), "reference": ref,
             "anchor_plane": labels[0], "above_deck_band_m": [0.03, 0.40],
             "all_cameras_corrected": all_ok, "filters": report["filters"],
             "note": "affine corrected, edge filtered, pre-ICP"}, indent=2))
        report["dump_per_camera"] = str(dd)
        print(f"per-camera clouds -> {dd}")

    merged = o3d.geometry.PointCloud()
    for c in cams:
        merged += clouds[c]
    report["n_points_before_voxel"] = len(merged.points)
    merged = merged.voxel_down_sample(args.voxel)
    report["n_points_written"] = len(merged.points)
    report["fusion_voxel_m"] = args.voxel
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(args.out, merged)
    print(f"\nwrote {report['n_points_written']} points -> {args.out}")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()