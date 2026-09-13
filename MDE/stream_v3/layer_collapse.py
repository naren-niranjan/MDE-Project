#!/usr/bin/env python3
"""
layer_collapse.py  (v3.1)

Removes the per-camera scale and tilt error in the four-camera DA3 fusion
output, and reports what remains.

Scope, stated plainly
---------------------
This script anchors each camera to a ChArUco reference plane. A plane
constrains exactly three degrees of freedom: the standoff and the two tilt
axes. It is structurally blind to in-plane translation and to yaw, because a
plane is invariant under both. So a clean deck RMS from this script does NOT
mean the cameras are registered -- it means three of six DOF are. Use
inplane_offset.py (fed by --dump-per-camera) to measure the other three.

The correction
--------------
A 3-DOF multiplicative depth field per camera:

    d'(u,v) = d(u,v) * ( s + a*x~ + b*y~ ),   x~=(u-cx)/fx,  y~=(v-cy)/fy

Still radial from the optical centre at every pixel, so it cannot invent
lateral structure, but the linear variation across the field absorbs scale and
both tilt axes together. Closed-form least squares against the GT plane.

A scalar scale (--no-tilt) cannot do this: uniform scaling about the camera
centre is a similarity transform about a point, so it preserves plane
orientation exactly. It drives the MEDIAN deck offset to near zero while
leaving tilt untouched, and a median cancels tilt -- which is why the deck can
look fixed while the parcels stay tens of mm apart. Keep --no-tilt only for
the before/after comparison.

Other deliberate choices
------------------------
  * The deck fit is primed by the GT plane rather than taking the largest plane
    in the cloud. With the floor at 3.91 m and the deck at 3.08 m, "largest" is
    not a safe selector. Each camera's own fitted standoff is reported so the
    confusion is visible if it happens anyway.
  * The residual metric reports its saturation fraction. A nearest-neighbour
    p90 sitting near the cap means "no correspondence found", not "sheets that
    far apart", and a fusion voxel must not be derived from it.
  * The deck residual is reported as offset AND RMS, since only the RMS sees
    tilt.

GT input
--------
Either ChArUco schema:
  derived -- a "reference_planes" list, thickness already applied upstream.
  raw     -- top-level surface keys with normal_mean / d_mean for the board
             PATTERN plane, plus a "board" block giving thickness_m.

For the raw schema the thickness is applied here. The pattern is printed on the
top face of a board resting on the surface and the camera is above, so the
physical surface is one thickness FURTHER from the camera; that is the default.
--thickness-sign toward flips it.

Example
-------
  python layer_collapse.py \
      --capture runs/live_20260817_102732/capture_00002 \
      --gt gt_planes.json --reference center --anchor-plane deck \
      --dump-per-camera runs/live_20260817_102732/capture_00002/percam \
      --out    runs/live_20260817_102732/capture_00002/fused_collapsed.ply \
      --report runs/live_20260817_102732/capture_00002/layer_collapse.json
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
# loading
# ----------------------------------------------------------------------------

def to_4x4(M):
    """E may be stored as 3x4 [R|t] or 4x4. Normalise to 4x4."""
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


def check_K(K, depth):
    """K must be on the same grid as the depth map. Catches the padding class of
    bug, where a mismatched principal point silently rescales depth."""
    h, w = depth.shape
    cx, cy = K[0, 2], K[1, 2]
    return {"depth_shape": [int(h), int(w)],
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(cx), "cy": float(cy),
            "cx_over_w": float(cx / w), "cy_over_h": float(cy / h),
            "principal_point_suspect":
                not (0.3 * w < cx < 0.7 * w and 0.3 * h < cy < 0.7 * h)}


def backproject(depth, K, conf, conf_pct, dmin, dmax):
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    m = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    if conf is not None and conf_pct is not None and m.any():
        m &= conf >= np.percentile(conf[m], conf_pct)
    z = depth[m]
    return np.stack([(u[m] - cx) * z / fx, (v[m] - cy) * z / fy, z], axis=1)


def as_pcd(P):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(P)
    return p


def resolve_convention(cams, Es, cam_pts, forced=None):
    """E is either world_from_cam or cam_from_world. Pick whichever makes the
    clouds land on each other. A 3x4 [R|t] is almost always the OpenCV
    extrinsic, i.e. cam_from_world."""
    def spread(inv):
        C = []
        for c in cams:
            T = np.linalg.inv(Es[c]) if inv else Es[c]
            C.append(((T[:3, :3] @ cam_pts[c].T).T + T[:3, 3]).mean(axis=0))
        C = np.asarray(C)
        return float(np.linalg.norm(C - C.mean(axis=0), axis=1).mean())
    if forced == "world_from_cam":
        return False, None
    if forced == "cam_from_world":
        return True, None
    s = {False: spread(False), True: spread(True)}
    return s[True] < s[False], {"spread_world_from_cam_m": s[False],
                                "spread_cam_from_world_m": s[True]}


# ----------------------------------------------------------------------------
# GT
# ----------------------------------------------------------------------------

def _plane(n, perp):
    """Unit normal plus positive perpendicular distance -> (n, d) with
    n.X + d = 0, signed so the plane sits in front of the camera."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return n, float(-perp if n[2] > 0 else perp)


def normalise_gt(gt, thickness_sign, thickness_override=None):
    planes, notes = {}, []

    if "reference_planes" in gt:
        for p in gt["reference_planes"]:
            n, d = _plane(p["normal"], float(p["perp_m"]))
            planes[p["label"]] = {"normal": n.tolist(), "d": d,
                                  "perp_m": float(p["perp_m"]),
                                  "source": "derived (thickness applied upstream)",
                                  "spread_mm": p.get("spread_mm")}
        return gt.get("reference_camera"), planes, notes

    th = thickness_override
    if th is None:
        th = float(gt.get("board", {}).get("thickness_m", 0.0))
    if th == 0.0:
        notes.append("board thickness is 0 - planes are the PATTERN plane, not "
                     "the physical surface")
    sign = +1.0 if thickness_sign == "away" else -1.0

    for label, e in gt.items():
        if not isinstance(e, dict) or "normal_mean" not in e or "d_mean" not in e:
            continue
        n = np.asarray(e["normal_mean"], float)
        n = n / np.linalg.norm(n)
        perp_pat = abs(float(e["d_mean"]))
        perp = perp_pat + sign * th
        n_s, d_s = _plane(n, perp)
        rec = {"normal": n_s.tolist(), "d": d_s, "perp_m": perp,
               "perp_pattern_m": perp_pat, "thickness_m": th,
               "thickness_sign": thickness_sign,
               "source": "raw (thickness applied here)",
               "spread_mm": e.get("perp_dist_spread_mm"),
               "normal_spread_deg": e.get("normal_spread_deg")}
        own = e.get("surface_perp_dist_m")
        if own is not None:
            dmm = (perp - float(own)) * 1000.0
            rec["file_surface_perp_m"] = float(own)
            rec["disagreement_with_file_mm"] = dmm
            if abs(abs(dmm) - 2000.0 * th) < 5.0:
                notes.append(f"{label}: GT file's surface_perp_dist_m disagrees by "
                             f"{abs(dmm):.0f} mm (two board thicknesses) - sign "
                             f"convention differs upstream")
        planes[label] = rec

    return gt.get("reference_camera"), planes, notes


def plane_to_frame(n_w, d_w, T_world_cam):
    """World plane -> the camera frame of T_world_cam."""
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    n_c = R.T @ n_w
    d_c = float(n_w @ t + d_w)
    k = np.linalg.norm(n_c)
    return n_c / k, d_c / k


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


# ----------------------------------------------------------------------------
# planes
# ----------------------------------------------------------------------------

def fit_plane(P, thresh=0.006, iters=3000):
    model, inl = as_pcd(P).segment_plane(thresh, 3, iters)
    n = np.array(model[:3], float)
    k = np.linalg.norm(n)
    mask = np.zeros(len(P), bool)
    mask[np.asarray(inl)] = True
    return n / k, model[3] / k, mask


def fit_deck_primed(P, n_prior, d_prior, band, thresh=0.006):
    """Fit the deck using the GT plane as a prior, not by taking the largest
    plane. Returns (n, d, index array) or None."""
    sd = P @ n_prior + d_prior
    cand = np.abs(sd) < band
    if cand.sum() < 500:
        return None
    n, d, inl = fit_plane(P[cand], thresh)
    idx = np.flatnonzero(cand)[inl]
    if n @ n_prior < 0:
        n, d = -n, -d
    return n, d, idx


def sdist(P, n, d):
    return P @ n + d


def inplane_extent(P, n):
    """Largest in-plane spread of a point set, for judging how well a tilt fit
    is conditioned. A narrow deck patch cannot constrain tilt."""
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(n @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    u, v = P @ e1, P @ e2
    return float(u.max() - u.min()), float(v.max() - v.min())


# ----------------------------------------------------------------------------
# the correction
# ----------------------------------------------------------------------------

def solve_depth_field(P, n_c, d_c, allow_tilt=True):
    """
    Find g(x~,y~) = s + a*x~ + b*y~ minimising || g*(n.X) + d || over deck
    points X in the camera frame. Scaling depth by g at each pixel maps
    X -> g*X, so the residual is linear in (s,a,b): closed-form least squares.
    """
    q = P @ n_c
    xt = P[:, 0] / P[:, 2]
    yt = P[:, 1] / P[:, 2]
    F = np.stack([q, q * xt, q * yt], axis=1) if allow_tilt else q[:, None]
    theta, *_ = np.linalg.lstsq(F, -d_c * np.ones(len(q)), rcond=None)
    return theta


def field_gain(P, theta):
    if len(theta) == 1:
        return np.full(len(P), theta[0])
    xt = P[:, 0] / P[:, 2]
    yt = P[:, 1] / P[:, 2]
    return theta[0] + theta[1] * xt + theta[2] * yt


def field_summary(P, theta):
    g = field_gain(P, theta)
    return {"s": float(theta[0]),
            "a": float(theta[1]) if len(theta) == 3 else 0.0,
            "b": float(theta[2]) if len(theta) == 3 else 0.0,
            "gain_min": float(g.min()), "gain_max": float(g.max()),
            "gain_span_pct": float((g.max() - g.min()) * 100.0),
            "implied_scale_error_pct": float((theta[0] - 1.0) * 100.0)}


# ----------------------------------------------------------------------------
# residual
# ----------------------------------------------------------------------------

def nn_stats(src, dst, max_dist):
    """Nearest-neighbour distances with an explicit saturation count, so a p90
    pinned near the cap is recognisable as missing correspondence."""
    tree = o3d.geometry.KDTreeFlann(dst)
    pts = np.asarray(src.points)
    step = max(1, len(pts) // 20000)
    d, n_sat = [], 0
    for p in pts[::step]:
        k, _, sq = tree.search_knn_vector_3d(p, 1)
        if k == 1:
            dd = float(np.sqrt(sq[0]))
            if dd < max_dist:
                d.append(dd)
            else:
                n_sat += 1
    total = len(d) + n_sat
    if not d:
        return {"median_mm": None, "p90_mm": None, "samples": 0,
                "saturated_fraction": 1.0, "cap_mm": max_dist * 1000.0}
    a = np.asarray(d) * 1000.0
    return {"median_mm": float(np.median(a)), "p90_mm": float(np.percentile(a, 90)),
            "samples": len(d),
            "saturated_fraction": float(n_sat / total) if total else 1.0,
            "cap_mm": max_dist * 1000.0}


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="stream_v2 capture_XXXXX dir")
    ap.add_argument("--gt", required=True, help="ChArUco JSON, either schema")
    ap.add_argument("--cameras", default="left,center,right,top")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--depth-key", default="depth", choices=["depth", "depth_raw"])
    ap.add_argument("--anchor-plane", default="deck")
    ap.add_argument("--list-planes", action="store_true")
    ap.add_argument("--thickness-sign", default="away", choices=["away", "toward"])
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--no-tilt", action="store_true",
                    help="scalar scale only, for before/after comparison")
    ap.add_argument("--deck-band", type=float, default=0.30,
                    help="search half-width around the GT plane for deck points (m)")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=8.0)
    ap.add_argument("--above-deck-min", type=float, default=0.03)
    ap.add_argument("--above-deck-max", type=float, default=0.40)
    ap.add_argument("--nn-cap", type=float, default=0.25, help="m")
    ap.add_argument("--icp-max-corr", type=float, default=0.02)
    ap.add_argument("--no-icp", action="store_true")
    ap.add_argument("--extrinsic-convention", default="auto",
                    choices=["auto", "world_from_cam", "cam_from_world"])
    ap.add_argument("--dump-per-camera", default=None,
                    help="directory for cloud_<cam>.ply + deck_plane.json, "
                         "written after correction and before ICP; input to "
                         "inplane_offset.py")
    ap.add_argument("--voxel", type=float, default=None)
    ap.add_argument("--voxel-safety", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cap = Path(args.capture)
    gt_raw = json.loads(Path(args.gt).read_text())
    gt_ref, gt_planes, notes = normalise_gt(gt_raw, args.thickness_sign, args.thickness)
    for n in notes:
        print(f"NOTE: {n}", file=sys.stderr)

    if args.list_planes:
        for label, p in sorted(gt_planes.items()):
            print(f"{label:<8} perp={p['perp_m']:.5f} n={np.round(p['normal'], 4)} "
                  f"d={p['d']:+.5f} [{p['source']}]")
        return
    if not gt_planes:
        sys.exit("no usable planes in the GT file")
    if args.anchor_plane not in gt_planes:
        sys.exit(f"no GT plane {args.anchor_plane!r}; have {sorted(gt_planes)}")

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    ref = args.reference or gt_ref
    if ref not in cams:
        sys.exit(f"reference {ref!r} not in {cams}")

    report = {"capture": str(cap), "gt_file": args.gt, "reference": ref,
              "depth_key": args.depth_key, "anchor_plane": args.anchor_plane,
              "tilt_dof": not args.no_tilt, "gt_notes": notes, "cameras": {}}

    # --- load ---------------------------------------------------------------
    Es, cam_pts = {}, {}
    for c in cams:
        depth, K, E, conf = load_camera(cap, c, args.depth_key)
        chk = check_K(K, depth)
        if chk["principal_point_suspect"]:
            print(f"warning: {c} principal point at ({chk['cx_over_w']:.3f}, "
                  f"{chk['cy_over_h']:.3f}) of the depth grid", file=sys.stderr)
        Es[c] = E
        cam_pts[c] = backproject(depth, K, conf, args.conf_percentile,
                                 args.depth_min, args.depth_max)
        report["cameras"][c] = {"intrinsics": chk, "n_points": int(len(cam_pts[c]))}

    invert, conv = resolve_convention(
        cams, Es, cam_pts,
        None if args.extrinsic_convention == "auto" else args.extrinsic_convention)
    report["extrinsic_convention"] = "cam_from_world" if invert else "world_from_cam"
    if conv:
        report["extrinsic_convention_evidence"] = conv
        print(f"extrinsics read as {report['extrinsic_convention']} "
              f"(spreads {conv['spread_world_from_cam_m']:.3f} / "
              f"{conv['spread_cam_from_world_m']:.3f} m)")

    def T_of(c):
        return np.linalg.inv(Es[c]) if invert else Es[c]

    gp = gt_planes[args.anchor_plane]
    plane_w = plane_to_world(np.asarray(gp["normal"], float), float(gp["d"]), T_of(ref))
    print(f"GT {args.anchor_plane} perp {gp['perp_m']:.5f} m "
          f"(placement spread {gp.get('spread_mm')} mm)")

    # --- correct each camera in its own frame ------------------------------
    corrected = {}
    for c in cams:
        info = report["cameras"][c]
        P = cam_pts[c]
        n_c, d_c = plane_to_frame(*plane_w, T_of(c))
        info["gt_plane_perp_m"] = float(abs(d_c))

        got = fit_deck_primed(P, n_c, d_c, args.deck_band)
        if got is None:
            info["deck"] = "no deck points found in the GT band"
            corrected[c] = P
            print(f"  {c}: no deck points within {args.deck_band} m of the GT plane")
            continue
        own_n, own_d, idx = got
        eu, ev = inplane_extent(P[idx], own_n)
        info["deck_points"] = int(len(idx))
        info["deck_extent_m"] = [eu, ev]
        info["own_deck_perp_m"] = float(abs(own_d))
        info["own_vs_gt_perp_pct"] = float((abs(own_d) / abs(d_c) - 1.0) * 100.0)
        info["tilt_deg_before"] = float(np.degrees(np.arccos(
            np.clip(abs(own_n @ n_c), -1, 1))))
        sd0 = sdist(P[idx], n_c, d_c)
        info["deck_offset_mm_before"] = float(np.median(sd0) * 1000.0)
        info["deck_rms_mm_before"] = float(np.sqrt((sd0 ** 2).mean()) * 1000.0)
        if min(eu, ev) < 0.20:
            print(f"  {c}: deck patch is only {eu:.2f} x {ev:.2f} m - tilt fit is "
                  f"poorly conditioned, treat its tilt numbers with suspicion")

        theta = solve_depth_field(P[idx], n_c, d_c, allow_tilt=not args.no_tilt)
        fs = field_summary(P, theta)
        info["depth_field"] = fs
        ok = (0.90 < fs["s"] < 1.10) and fs["gain_span_pct"] < 15.0 and fs["gain_min"] > 0
        if not ok:
            info["depth_field_rejected"] = True
            print(f"  {c}: depth field implausible (s={fs['s']:.4f}, span "
                  f"{fs['gain_span_pct']:.2f} %) - left uncorrected")
            corrected[c] = P
            continue
        Pc = P * field_gain(P, theta)[:, None]
        corrected[c] = Pc

        got2 = fit_deck_primed(Pc, n_c, d_c, args.deck_band)
        if got2 is not None:
            n2, d2, idx2 = got2
            info["tilt_deg_after"] = float(np.degrees(np.arccos(
                np.clip(abs(n2 @ n_c), -1, 1))))
            sd1 = sdist(Pc[idx2], n_c, d_c)
            info["deck_offset_mm_after"] = float(np.median(sd1) * 1000.0)
            info["deck_rms_mm_after"] = float(np.sqrt((sd1 ** 2).mean()) * 1000.0)

        print(f"{c:<7} own_perp {info['own_deck_perp_m']:.4f} m "
              f"({info['own_vs_gt_perp_pct']:+.2f} %)  s={fs['s']:.5f} "
              f"span={fs['gain_span_pct']:.2f} %  "
              f"tilt {info['tilt_deg_before']:.3f}->"
              f"{info.get('tilt_deg_after', float('nan')):.3f} deg  "
              f"deck RMS {info['deck_rms_mm_before']:.1f}->"
              f"{info.get('deck_rms_mm_after', float('nan')):.1f} mm  "
              f"extent {eu:.2f}x{ev:.2f} m")

    # --- to world -----------------------------------------------------------
    clouds = {}
    for c in cams:
        T = T_of(c)
        clouds[c] = as_pcd((T[:3, :3] @ corrected[c].T).T + T[:3, 3])

    n_w, d_w = plane_w

    # --- dump per-camera clouds, before ICP touches anything ---------------
    if args.dump_per_camera:
        dd = Path(args.dump_per_camera)
        dd.mkdir(parents=True, exist_ok=True)
        for c in cams:
            o3d.io.write_point_cloud(str(dd / f"cloud_{c}.ply"), clouds[c])
        (dd / "deck_plane.json").write_text(json.dumps(
            {"normal": n_w.tolist(), "d": float(d_w), "reference": ref,
             "anchor_plane": args.anchor_plane,
             "above_deck_band_m": [args.above_deck_min, args.above_deck_max],
             "note": "corrected per-camera clouds, pre-ICP"}, indent=2))
        report["dump_per_camera"] = str(dd)
        print(f"per-camera clouds -> {dd}")

    def above(P):
        s = sdist(P, n_w, d_w)
        return P[(s > args.above_deck_min) & (s < args.above_deck_max)]

    ref_above = as_pcd(above(np.asarray(clouds[ref].points)))
    report["reference_above_deck_points"] = len(ref_above.points)

    # --- residual rigid refinement -----------------------------------------
    for c in cams:
        if c == ref or args.no_icp:
            continue
        src = as_pcd(above(np.asarray(clouds[c].points)))
        if len(src.points) < 500 or len(ref_above.points) < 500:
            report["cameras"][c]["icp"] = "skipped, too few above-deck points"
            continue
        res = o3d.pipelines.registration.registration_icp(
            src, ref_above, args.icp_max_corr, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        shift = float(np.linalg.norm(res.transformation[:3, 3]) * 1000.0)
        if shift < args.icp_max_corr * 1000.0:
            clouds[c].transform(res.transformation)
            report["cameras"][c]["icp_fitness"] = float(res.fitness)
            report["cameras"][c]["icp_shift_mm"] = shift
        else:
            report["cameras"][c]["icp_rejected_shift_mm"] = shift

    # --- residual -----------------------------------------------------------
    worst, saturated = 0.0, False
    for c in cams:
        if c == ref:
            continue
        src = as_pcd(above(np.asarray(clouds[c].points)))
        if len(src.points) < 100:
            continue
        st = nn_stats(src, ref_above, args.nn_cap)
        report["cameras"][c]["object_nn"] = st
        print(f"{c:<7} object nn median "
              f"{st['median_mm'] if st['median_mm'] is None else round(st['median_mm'], 1)} mm  "
              f"p90 {st['p90_mm'] if st['p90_mm'] is None else round(st['p90_mm'], 1)} mm  "
              f"saturated {st['saturated_fraction'] * 100:.1f} %")
        if st["p90_mm"]:
            worst = max(worst, st["p90_mm"])
        if st["saturated_fraction"] > 0.10:
            saturated = True

    report["residual_disagreement_p90_mm"] = worst
    report["residual_saturated"] = saturated

    if args.voxel is not None:
        voxel, src_ = args.voxel, "override"
    elif saturated or worst > 30.0:
        voxel, src_ = 0.004, "fallback (residual not trustworthy)"
    else:
        voxel, src_ = max(0.004, args.voxel_safety * worst / 1000.0), "measured"
    report["fusion_voxel_m"] = voxel
    report["voxel_source"] = src_

    merged = o3d.geometry.PointCloud()
    for c in cams:
        merged += clouds[c]
    report["n_points_before_voxel"] = len(merged.points)
    merged = merged.voxel_down_sample(voxel)
    report["n_points_written"] = len(merged.points)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(args.out, merged)

    print(f"\nresidual p90 {worst:.2f} mm   voxel {voxel * 1000:.2f} mm ({src_})")
    print(f"wrote {report['n_points_written']} points -> {args.out}")
    if saturated:
        print("\nA large share of above-deck points had no neighbour within the cap.\n"
              "That is coverage or registration, not a sheet separation you can\n"
              "voxel away. The deck anchor cannot see in-plane translation or yaw -\n"
              "run inplane_offset.py on the --dump-per-camera output next.")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()