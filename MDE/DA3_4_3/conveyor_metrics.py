#!/usr/bin/env python3
"""
conveyor_metrics.py

Measures the conveyor region only, which is the region of interest for
depalletising. Reports three things:

  1. Box top height above the deck    -- the manipulation-relevant number
  2. Box footprint and edge sharpness -- where flying-pixel curtains hurt
  3. Per-camera agreement, restricted to the conveyor

How the conveyor is located
---------------------------
Fitting the largest plane finds the FLOOR, because the floor dominates the
frame while the conveyor is a narrow band. So the deck is located by height
instead: fit the floor, then take the band at --deck-height above it. Nothing
else in the scene sits at conveyor height, so this isolates the deck without
depending on RANSAC picking it.

Height convention: the floor normal is oriented so the camera side is
positive, so "height" means height above the floor and the deck sits at
roughly +0.8375 m.

Curtain rejection (new)
-----------------------
Flying pixels form thin vertical sheets hanging off box silhouettes. DBSCAN
clusters them as separate objects, and they then pollute the cross-camera
matching -- a curtain in one view gets paired against a real box in another,
producing spurious spreads of hundreds of millimetres.

Two filters remove them:

    --min-box-short   a curtain is a sheet, so its short footprint side is
                      tiny (5-70 mm) while real cargo is >150 mm
    --max-skirt       a curtain has more mid-height points than top-face
                      points; real cargo seen from above has almost none

Rejected clusters are still reported under "curtains" so the artefact rate
stays visible rather than being silently hidden.

Roller caveat
-------------
The conveyor is roller-topped, so the deck is corrugated rather than flat.
The fitted deck plane lands somewhere between crown and valley, and the deck
flatness RMS includes the roller pitch. Box heights are measured against the
FITTED deck plane, not the ChArUco crown-to-floor figure, so that offset
cancels.

Example
-------
python conveyor_metrics.py \
    --cloud left=cloud_left.ply   --cloud center=cloud_center.ply \
    --cloud right=cloud_right.ply --cloud top=cloud_top.ply \
    --gt-box-height 0=250.0 --gt-box-height 1=310.0 \
    --out conveyor_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is required")

try:
    import cv2
except ImportError:
    cv2 = None


def parse_named(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def parse_gt(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected INDEX=MM, got {value!r}")
    idx, mm = value.split("=", 1)
    return int(idx), float(mm)


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------

def fit_plane(points, thresh, iters=4000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, inl = pcd.segment_plane(thresh, 3, iters)
    a, b, c, d = model
    n = np.array([a, b, c], float)
    s = np.linalg.norm(n)
    return n / s, d / s, np.asarray(inl)


def orient_toward_camera(n, d):
    """Camera sits at the origin; make its side of the plane positive."""
    return (n, d) if d > 0 else (-n, -d)


def plane_basis(normal):
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def min_area_rect(uv):
    """Oriented footprint of 2D points. Returns (long_mm, short_mm, area_mm2)."""
    if cv2 is not None and len(uv) >= 5:
        (_, _), (w, h), _ = cv2.minAreaRect(uv.astype(np.float32))
        a, b = sorted([w, h], reverse=True)
        return a * 1e3, b * 1e3, a * b * 1e6
    c = uv - uv.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    ext = proj.max(0) - proj.min(0)
    a, b = sorted(ext, reverse=True)
    return a * 1e3, b * 1e3, a * b * 1e6


# --------------------------------------------------------------------------
# analysis for one camera
# --------------------------------------------------------------------------

def analyse(points, args, label):
    out = {"n_points": int(len(points))}

    # ---- floor ------------------------------------------------------
    fn, fd, finl = fit_plane(points, args.plane_thresh)
    fn, fd = orient_toward_camera(fn, fd)
    height = points @ fn + fd
    out["floor"] = {"normal": fn.tolist(), "d": float(fd),
                    "inlier_frac": round(float(len(finl) / len(points)), 4)}

    # ---- deck band --------------------------------------------------
    band = np.abs(height - args.deck_height) <= args.deck_tol
    if band.sum() < args.min_deck_points:
        out["status"] = (f"only {int(band.sum())} points at deck height; "
                         f"widen --deck-tol or check --deck-height")
        print(f"\n[{label}] {out['status']}")
        return out
    deck_pts = points[band]

    dn, dd, dinl = fit_plane(deck_pts, args.plane_thresh)
    dn, dd = orient_toward_camera(dn, dd)
    if np.dot(dn, fn) < 0:
        dn, dd = -dn, -dd
    resid = deck_pts @ dn + dd
    tilt = float(np.degrees(np.arccos(np.clip(abs(np.dot(dn, fn)), -1, 1))))

    out["deck"] = {
        "n_points": int(band.sum()),
        "height_above_floor_mm": round(float(np.median(height[band])) * 1e3, 2),
        "flatness_rms_mm": round(float(np.sqrt(np.mean(resid ** 2))) * 1e3, 2),
        "flatness_p95_mm": round(float(np.percentile(np.abs(resid), 95)) * 1e3, 2),
        "tilt_vs_floor_deg": round(tilt, 4),
        "inlier_frac": round(float(len(dinl) / len(deck_pts)), 4),
        "normal": dn.tolist(), "d": float(dd),
    }

    # ---- cargo candidates -------------------------------------------
    h_deck = points @ dn + dd
    u, v = plane_basis(dn)
    cu, cv_ = points @ u, points @ v

    du, dv = deck_pts @ u, deck_pts @ v
    pad = args.lateral_pad
    on_belt = ((cu >= du.min() - pad) & (cu <= du.max() + pad) &
               (cv_ >= dv.min() - pad) & (cv_ <= dv.max() + pad))
    cargo = on_belt & (h_deck > args.box_min_height) & (h_deck < args.box_max_height)

    out["cargo_points"] = int(cargo.sum())
    if cargo.sum() < args.min_cluster_points:
        out["status"] = "no cargo found above the deck"
        print(f"\n[{label}] {out['status']}")
        return out

    cpcd = o3d.geometry.PointCloud()
    cpcd.points = o3d.utility.Vector3dVector(points[cargo])
    lbl = np.asarray(cpcd.cluster_dbscan(eps=args.cluster_eps,
                                         min_points=args.min_cluster_points))
    cpts = points[cargo]
    ch = h_deck[cargo]

    boxes, curtains = [], []
    for k in range(lbl.max() + 1):
        sel = lbl == k
        if sel.sum() < args.min_cluster_points:
            continue
        bp, bh = cpts[sel], ch[sel]

        top = float(np.percentile(bh, args.top_percentile))
        top_face = bh >= top - args.top_slab
        if top_face.sum() < 20:
            continue

        uv = np.stack([bp[top_face] @ u, bp[top_face] @ v], axis=1)
        lo, sh, area = min_area_rect(uv)

        mid = (bh >= 0.25 * top) & (bh <= 0.75 * top)
        skirt = float(mid.sum() / max(top_face.sum(), 1))
        tf_resid = bh[top_face] - np.median(bh[top_face])

        rec = {
            "n_points": int(sel.sum()),
            "top_height_mm": round(top * 1e3, 2),
            "top_height_p50_mm": round(float(np.median(bh[top_face])) * 1e3, 2),
            "top_face_rms_mm": round(float(np.sqrt(np.mean(tf_resid ** 2))) * 1e3, 2),
            "footprint_long_mm": round(lo, 1),
            "footprint_short_mm": round(sh, 1),
            "footprint_area_mm2": round(area, 0),
            "skirt_ratio": round(skirt, 3),
            "centroid_uv": [round(float(np.mean(bp[top_face] @ u)), 4),
                            round(float(np.mean(bp[top_face] @ v)), 4)],
        }

        # A curtain is a thin vertical sheet: tiny short side, or more
        # mid-height points than top-face points.
        why = []
        if sh < args.min_box_short:
            why.append(f"short side {sh:.0f} mm")
        if skirt > args.max_skirt:
            why.append(f"skirt {skirt:.2f}")
        if why:
            rec["rejected_because"] = ", ".join(why)
            curtains.append(rec)
        else:
            boxes.append(rec)

    boxes.sort(key=lambda b: b["centroid_uv"][0])
    for i, b in enumerate(boxes):
        b["index"] = i

    out.update({"boxes": boxes, "n_boxes": len(boxes),
                "curtains": curtains, "n_curtains": len(curtains),
                "status": "ok"})

    print(f"\n[{label}] deck at {out['deck']['height_above_floor_mm']:.1f} mm "
          f"above floor, flatness RMS {out['deck']['flatness_rms_mm']:.1f} mm, "
          f"tilt {out['deck']['tilt_vs_floor_deg']:.3f} deg")
    if boxes:
        hdr = (f"  {'box':<5}{'top_mm':>9}{'long_mm':>10}{'short_mm':>10}"
               f"{'topRMS':>9}{'skirt':>8}{'pts':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for b in boxes:
            print(f"  {b['index']:<5}{b['top_height_mm']:>9.1f}"
                  f"{b['footprint_long_mm']:>10.1f}{b['footprint_short_mm']:>10.1f}"
                  f"{b['top_face_rms_mm']:>9.1f}{b['skirt_ratio']:>8.3f}"
                  f"{b['n_points']:>8d}")
    if curtains:
        pts = sum(c["n_points"] for c in curtains)
        print(f"  rejected {len(curtains)} curtain cluster(s), {pts} pts "
              f"({pts / max(out['cargo_points'], 1):.1%} of cargo candidates)")
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Conveyor-region metrics for depalletising validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cloud", action="append", required=True,
                    type=parse_named, metavar="NAME=PATH")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="0 keeps native density, which matters for the "
                         "skirt ratio")
    ap.add_argument("--plane-thresh", type=float, default=0.010)

    g = ap.add_argument_group("conveyor location")
    g.add_argument("--deck-height", type=float, default=0.8375,
                   help="deck height above the floor, metres (ChArUco)")
    g.add_argument("--deck-tol", type=float, default=0.06)
    g.add_argument("--min-deck-points", type=int, default=500)
    g.add_argument("--lateral-pad", type=float, default=0.05)

    g = ap.add_argument_group("cargo segmentation")
    g.add_argument("--box-min-height", type=float, default=0.03)
    g.add_argument("--box-max-height", type=float, default=0.80)
    g.add_argument("--cluster-eps", type=float, default=0.05,
                   help="lower this if adjacent boxes merge into one cluster")
    g.add_argument("--min-cluster-points", type=int, default=200)
    g.add_argument("--top-percentile", type=float, default=95.0)
    g.add_argument("--top-slab", type=float, default=0.015)

    g = ap.add_argument_group("curtain rejection")
    g.add_argument("--min-box-short", type=float, default=150.0,
                   help="minimum short-side footprint in mm; thinner clusters "
                        "are flying-pixel sheets, not cargo")
    g.add_argument("--max-skirt", type=float, default=0.6,
                   help="clusters with more mid-height than top-face points "
                        "are curtains")

    ap.add_argument("--match-radius", type=float, default=0.15)
    ap.add_argument("--gt-box-height", action="append", default=[],
                    type=parse_gt, metavar="INDEX=MM",
                    help="caliper-measured height for a matched box index; "
                         "repeat per box")
    ap.add_argument("--out", type=Path, default=Path("conveyor_metrics.json"))
    args = ap.parse_args()

    report = {"params": {"deck_height_m": args.deck_height,
                         "deck_tol_m": args.deck_tol,
                         "min_box_short_mm": args.min_box_short,
                         "max_skirt": args.max_skirt},
              "cameras": {}}

    for name, path in args.cloud:
        if not path.exists():
            sys.exit(f"missing cloud: {path}")
        pcd = o3d.io.read_point_cloud(str(path))
        if args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)
        report["cameras"][name] = analyse(np.asarray(pcd.points), args, name)

    good = {n: r for n, r in report["cameras"].items()
            if r.get("status") == "ok" and r.get("boxes")}
    if len(good) < 2:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nfewer than two cameras produced boxes; skipping comparison")
        print(f"written: {args.out}")
        return

    ref_name = "center" if "center" in good else next(iter(good))
    cams = list(good.keys())
    gt = dict(args.gt_box_height)

    matches = []
    for rb in good[ref_name]["boxes"]:
        row = {"ref_index": rb["index"], "heights": {ref_name: rb["top_height_mm"]},
               "footprints": {ref_name: [rb["footprint_long_mm"],
                                         rb["footprint_short_mm"]]}}
        c0 = np.array(rb["centroid_uv"])
        for n, r in good.items():
            if n == ref_name:
                continue
            best, bd = None, args.match_radius
            for b in r["boxes"]:
                dist = np.linalg.norm(np.array(b["centroid_uv"]) - c0)
                if dist < bd:
                    best, bd = b, dist
            if best:
                row["heights"][n] = best["top_height_mm"]
                row["footprints"][n] = [best["footprint_long_mm"],
                                        best["footprint_short_mm"]]
        matches.append(row)

    print("\n" + "=" * 74)
    print("PER-CAMERA AGREEMENT ON THE CONVEYOR (box top height, mm)")
    head = f"  {'box':<6}" + "".join(f"{c:>11}" for c in cams) + f"{'spread':>10}"
    if gt:
        head += f"{'GT':>9}{'err':>9}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    spreads, errors = [], []
    for row in matches:
        vals = [row["heights"].get(c) for c in cams]
        present = [x for x in vals if x is not None]
        if len(present) < 2:
            continue
        sp = max(present) - min(present)
        spreads.append(sp)
        line = (f"  {row['ref_index']:<6}"
                + "".join(f"{x:>11.1f}" if x is not None else f"{'-':>11}"
                          for x in vals)
                + f"{sp:>10.1f}")
        if row["ref_index"] in gt:
            g_ = gt[row["ref_index"]]
            err = float(np.mean(present)) - g_
            errors.append(err)
            row["gt_mm"], row["error_mm"] = g_, round(err, 2)
            line += f"{g_:>9.1f}{err:>+9.1f}"
        elif gt:
            line += f"{'-':>9}{'-':>9}"
        print(line)

    report["cross_camera"] = {"reference": ref_name, "matched_boxes": matches}
    if spreads:
        report["cross_camera"]["median_spread_mm"] = round(float(np.median(spreads)), 2)
        report["cross_camera"]["max_spread_mm"] = round(float(max(spreads)), 2)
        print(f"\n  median spread across cameras : {np.median(spreads):.1f} mm")
        print(f"  worst spread                 : {max(spreads):.1f} mm")

    decks = [r["deck"]["height_above_floor_mm"] for r in good.values()]
    tilts = [r["deck"]["tilt_vs_floor_deg"] for r in good.values()]
    report["cross_camera"]["deck_height_spread_mm"] = round(
        float(max(decks) - min(decks)), 2)
    report["cross_camera"]["deck_height_mean_mm"] = round(float(np.mean(decks)), 2)
    print(f"  deck height, mean            : {np.mean(decks):.1f} mm "
          f"(GT {args.deck_height * 1e3:.1f}, "
          f"{np.mean(decks) - args.deck_height * 1e3:+.1f} mm)")
    print(f"  deck height spread           : {max(decks) - min(decks):.1f} mm")
    print(f"  deck tilt range              : {min(tilts):.3f}-{max(tilts):.3f} deg")

    if errors:
        report["cross_camera"]["mean_abs_error_mm"] = round(
            float(np.mean(np.abs(errors))), 2)
        print(f"  mean abs error vs calipers   : "
              f"{np.mean(np.abs(errors)):.1f} mm")
    print("=" * 74)

    worst = max((b["skirt_ratio"] for r in good.values() for b in r["boxes"]),
                default=0.0)
    n_curt = sum(r.get("n_curtains", 0) for r in good.values())
    if worst > 0.3 or n_curt:
        print(f"\n  {n_curt} curtain cluster(s) rejected; worst surviving skirt "
              f"ratio {worst:.2f}.")
        print(f"  To cut curtains at the source, LOWER --edge-thresh in "
              f"da3_fuse.py (the mask keeps pixels BELOW the threshold, so a "
              f"lower value filters more). Try 0.008.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()