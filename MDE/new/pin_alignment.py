#!/usr/bin/env python3
"""
pin_alignment.py

Pin the ground truth to the rig frame using the conveyor, taking from it only
what it actually constrains.

WHAT A CONVEYOR CONSTRAINS, AND WHAT IT DOES NOT
------------------------------------------------
The frame edges of the belt correspond between the ground truth and belt.json:
both are the outside of the conveyor frame, both measure about 900 mm. The ENDS
do not correspond. belt.json's ends are wherever the four-camera coverage ran
out; the ground truth sees 3119 to 3454 mm of real conveyor. Treating the four
corners as four point correspondences forces those non-corresponding ends onto
each other and smears the error around the whole rectangle, which is what put 65
to 100 mm on every corner of a pick that was perfectly good.

Used properly the belt fixes five degrees of freedom:

    deck plane                    normal and offset          3
    the two long frame edges      yaw about the normal,
                                  position across the belt   2

Position ALONG the belt is left free, because a conveyor looks the same wherever
you slide it. That is a property of the object. No picking technique recovers it
and none should try.

THE SIXTH, AND THE SCALE
------------------------
Both come from the reconstruction, by a two-parameter search: slide along the
belt, and scale. The parcels carry that search, since they are the only thing
breaking the along-belt symmetry, and they carry it well: a 1-D slide against a
loaded conveyor has one sharp minimum.

Scale has to be in the search because the reconstruction's gauge is unknown and
a wrong gauge changes the SPACING of the parcels along the belt, which the slide
would otherwise absorb into a wrong offset.

TWO PASSES
----------
The recovered scale belongs to the DEPTH, not to the cloud: scaling a fused
cloud about the world origin also scales the camera centres, which are
calibrated and must not move. So --refine reports the scale, you re-fuse with
it, and you run --refine once more on the re-fused cloud to settle the last
millimetres. The second pass should return a scale within a per cent of 1.

    python3 pin_alignment.py --gt gt/...ply --pick
    python3 pin_alignment.py --gt gt/...ply --points belt_corners.json \\
        --refine runs_da3/scene_b/center+left+top+right_res1008
    python3 check_fusion.py --runs-root runs_da3/scene_b \\
        --scale-mode file --gauge metric_fit.json --rewrite
    python3 pin_alignment.py --gt gt/...ply --points belt_corners.json \\
        --refine runs_da3/scene_b/center+left+top+right_res1008

Keep this file beside rigkit.py and belt.json.
"""

from __future__ import annotations

import argparse
import re
import itertools
import json
from pathlib import Path

import numpy as np

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")

try:
    import open3d as o3d
except ImportError:
    o3d = None


def apply(P, T):
    T = np.asarray(T, float)
    return np.asarray(P) @ T[:3, :3].T + T[:3, 3]


def pick(path):
    if o3d is None:
        raise SystemExit("open3d is required for --pick")
    pcd = o3d.io.read_point_cloud(str(path))
    print("shift-click the four conveyor FRAME corners in order round the "
          "belt, then close the window")
    print("the ends need not be precise: only the plane and the two long "
          "edges are used, and the length is never read")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="pick the four conveyor corners")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    if len(idx) != 4:
        raise SystemExit(f"picked {len(idx)} points, need exactly 4")
    return np.asarray(pcd.points)[idx]


def cyclic(C):
    """Put four coplanar corners into order round their own perimeter.

    Clicking the two left corners and then the two right ones is a natural way
    to pick a conveyor and gives a CROSSED quadrilateral: its two long "edges"
    are diagonals, and its two short edges run parallel instead of anti-
    parallel. Everything downstream then fails quietly, because the cross axis
    is taken from the difference of the short edges and that difference
    collapses to noise.

    Sorting by angle about the centroid in the corners' own best-fit plane
    fixes it for any convex quadrilateral, so the picking order stops mattering.
    """
    C = np.asarray(C, float)
    c = C.mean(0)
    _, _, Vt = np.linalg.svd(C - c, full_matrices=False)
    a, b = Vt[0], Vt[1]
    ang = np.arctan2((C - c) @ b, (C - c) @ a)
    return C[np.argsort(ang)]


def rect_frame(C):
    """Plane, long axis, cross axis and centre of a picked quadrilateral.

    The long axis is the mean of the two long edges rather than either one, so a
    corner clicked slightly off the end tilts the axis by half as much. The
    cross axis is taken from the two SHORT edges, which do correspond between
    the clouds, and the centre across the belt from the midpoints of the long
    edges. Nothing here reads the length as a measurement.
    """
    C = cyclic(C)
    e = [C[(i + 1) % 4] - C[i] for i in range(4)]
    lens = [float(np.linalg.norm(v)) for v in e]
    # the two longest opposite edges are the belt sides
    i0 = int(np.argmax(lens))
    long_pair = (i0, (i0 + 2) % 4)
    short_pair = ((i0 + 1) % 4, (i0 + 3) % 4)

    ex = e[long_pair[0]] / lens[long_pair[0]] - e[long_pair[1]] / lens[long_pair[1]]
    ex /= np.linalg.norm(ex)
    ey = e[short_pair[0]] / lens[short_pair[0]] - e[short_pair[1]] / lens[short_pair[1]]
    ey /= np.linalg.norm(ey)
    n = np.cross(ex, ey)
    n /= np.linalg.norm(n)
    ey = np.cross(n, ex)

    width = float(np.mean([lens[short_pair[0]], lens[short_pair[1]]]))
    taper = float(abs(lens[short_pair[0]] - lens[short_pair[1]]))
    length = float(np.mean([lens[long_pair[0]], lens[long_pair[1]]]))
    centre = C.mean(0)
    resid = (C - centre) @ n
    return {"centre": centre, "ex": ex, "ey": ey, "n": n,
            "width": width, "taper": taper, "length": length,
            "flatness_mm": float(np.abs(resid).max()) * 1e3}


def frame_transform(src_f, dst_f, yaw180=False, flip_n=False):
    """Take one belt frame onto the other.

    A rectangle is symmetric under a 180 degree rotation about its normal, so
    which way the axes come out depends on which corner was clicked first and
    which way round the picking went. The conveyor cannot resolve that: both
    orientations put the deck on the deck and the frame edges on the frame
    edges. Only the parcels break it, so all the candidates are built here and
    the slide search picks between them.

    The normal sign is a third ambiguity, resolved the same way rather than by
    assuming where the ground-truth sensor sat.
    """
    ex, ey, n = src_f["ex"], src_f["ey"], src_f["n"]
    if yaw180:
        ex, ey = -ex, -ey
    if flip_n:
        n, ey = -n, -ey
    A = np.column_stack([ex, ey, n])
    B = np.column_stack([dst_f["ex"], dst_f["ey"], dst_f["n"]])
    R = B @ A.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dst_f["centre"] - R @ src_f["centre"]
    return T


def candidates(src_f, dst_f):
    return [(f"yaw{180 if y else 0:>3}, normal{'-' if f else '+'}",
             frame_transform(src_f, dst_f, y, f))
            for y in (False, True) for f in (False, True)]


def load_per_camera(run, args):
    """Each camera's cloud with its own optical centre.

    Scaling depth moves a point along the ray from ITS OWN camera, so a fused
    cloud cannot be rescaled about a single origin unless every camera sits
    there. On this rig only centre does: left is 0.294 m away, top 0.589 and
    right 0.859, and top's offset is almost entirely axial. Scaling top's cloud
    about the origin demanded a gauge of 3.6 where the truth is near 1.7, and
    that wrong gauge then fed the slide search.
    """
    try:
        import da3_stream as ds
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"da3_stream.py is needed for the camera centres: "
                         f"{exc!r}")
    cams = [c for c in re.sub(r"_res\d+$", "", run.name).split("+") if c]
    ext = ds.load_extrinsics(Path(args.calib_dir), cams, args.reference)
    out = []
    for c in cams:
        f = run / f"cloud_calib_{c}.ply"
        if not f.exists():
            return None, cams
        out.append((rigkit.load_cloud(str(f)),
                    ds.camera_centre(np.asarray(ext[c]["E"], float))))
    return out, cams


def warp_per_camera(clouds, s, slide, ex):
    """Every camera scaled about its own centre, then the whole slid."""
    return np.vstack([C + (P - C) * s + ex * slide for P, C in clouds])


def scale_from_deck_multi(clouds, frame, lo=0.5, hi=4.0, band=0.15):
    """The gauge that puts the deck on the deck, each camera scaled about its
    own centre.

    Solved as a root of the median height rather than in closed form, because
    with several cameras the median of a sum is not the sum of medians. The
    median height DECREASES as the gauge grows -- a bigger scale pushes points
    further from their camera, hence closer to the deck -- so the bracket is
    handled by sign rather than assumed, which is what caught this out.

    Two passes: the first over the whole cloud, the second restricted to a band
    about the first answer, so floor seen past the belt edge stops pulling.
    """
    c, n = frame["centre"], frame["n"]
    target = float(c @ n)

    def med_h(s, sel_band=None):
        h = np.concatenate([(C + (P - C) * s) @ n - target for P, C in clouds])
        if sel_band is not None:
            near = np.abs(h) <= sel_band
            if int(near.sum()) > 500:
                h = h[near]
        return float(np.median(h))

    def solve(a, b, sel_band=None, iters=60):
        fa, fb = med_h(a, sel_band), med_h(b, sel_band)
        if fa * fb > 0:
            return None
        for _ in range(iters):
            m = 0.5 * (a + b)
            fm = med_h(m, sel_band)
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    s = solve(lo, hi)
    if s is None:
        return float("nan")
    for _ in range(3):
        r = solve(s * 0.85, s * 1.15, band)
        if r is None:
            break
        s = r
    return float(s)


def scale_from_deck(recon, frame, iters=4, band=0.15):
    """The gauge in closed form, from the requirement that the deck land on the
    deck.

    With the ground truth pinned, the conveyor plane's position in the rig
    frame is known, so

        median over the cloud of  s * (P . n)  ==  centre . n

    solves for s directly. No search, and nothing to identify: it is the same
    argument that showed the raw median depth of a loaded conveyor sits on the
    deck, because the parcels stand only 150 to 400 mm proud of it.

    Searching scale and slide together does not work here. Scaling about the
    origin also shifts the cloud ALONG the belt, by (s-1) times the belt
    centre's component on that axis, which on this rig is about -148 mm at
    s = 1.65. The two parameters therefore trade against each other and the
    optimum lies along a valley rather than in a basin, which is how the joint
    search slid to -541 mm against a +/-600 mm limit.

    A few passes restricted to a band about the current estimate pull the
    answer off the whole-cloud median and onto the deck itself, so floor seen
    past the belt edge stops contributing.
    """
    n = frame["n"]
    target = float(frame["centre"] @ n)
    d = recon @ n
    s = target / float(np.median(d))
    for _ in range(iters):
        h = s * d - target
        near = np.abs(h) <= band
        if int(near.sum()) < 500:
            break
        s = target / float(np.median(d[near]))
    return float(s)


def slide_by_footprint(gt_rig, recon_scaled, frame, args):
    """The along-belt shift, by correlating parcel FOOTPRINTS in plan.

    Scoring in 3D was the mistake. The slide is an in-plane quantity, but a
    nearest-neighbour test in three dimensions is dominated by the height
    error, and on this rig that error is far larger than any sensible
    tolerance: at a relief factor near 0.85 a 300 mm parcel comes back 45 mm
    short, so its points cannot find a neighbour within 20 mm however well the
    slide is placed. Every candidate then scored two or three per cent and none
    could be told from another.

    Projected onto the deck, a parcel's outline is right even when its height is
    not, and the shift becomes a one-dimensional correlation of two occupancy
    masks. A reversed belt gives a reversed pattern, so this separates the two
    yaw candidates as well, which the 3D score could not.
    """
    c, ex, ey, n = frame["centre"], frame["ex"], frame["ey"], frame["n"]

    def mask(Q):
        rel = Q - c
        h = rel @ n
        sel = (h > args.top_min) & (h < args.top_max)
        if int(sel.sum()) < 200:
            return None
        u, v = rel[sel] @ ex, rel[sel] @ ey
        # Bounded to the DECK across the belt. A wider window admits the side
        # rails and whatever stands beside them, and those run parallel to the
        # conveyor: they contribute the same bulk at every shift, so they add
        # nothing but dilution and the correlation flattens. It is what made
        # the two yaw candidates tie at exactly 13.7 per cent.
        keep = np.abs(v) <= args.cross
        u, v = u[keep], v[keep]
        iu = np.floor((u + args.span) / args.cell).astype(np.int64)
        iv = np.floor((v + args.cross) / args.cell).astype(np.int64)
        nu = int(2 * args.span / args.cell) + 1
        nv = int(2 * args.cross / args.cell) + 1
        ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        m = np.zeros((nu, nv), bool)
        m[iu[ok], iv[ok]] = True
        return m

    A, B = mask(gt_rig), mask(recon_scaled)
    if A is None or B is None:
        return 0.0, 0.0
    steps = int(args.slide_m / args.cell)
    # The window the reconstruction actually occupies along the belt. Scoring
    # outside it is what made the two statistics fail in opposite directions:
    # Jaccard over everything punished the ground truth for seeing 3360 mm of
    # conveyor against the reconstruction's 2300, while intersection over the
    # reconstruction alone was satisfied by any ground truth dense enough to
    # cover it, and scored a reversed belt at 94 per cent. Inside the window
    # both the agreement and the disagreement count.
    cols = np.flatnonzero(B.any(axis=1))
    if not len(cols):
        return 0.0, 0.0
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    Bw = B[c0:c1]
    best, best_v = 0, -1.0
    for k in range(-steps, steps + 1):
        lo, hi = c0 + k, c1 + k
        if lo < 0 or hi > A.shape[0]:
            continue
        Aw = A[lo:hi]
        ov = int(np.count_nonzero(Aw & Bw))
        # Intersection over the RECONSTRUCTION's area, not Jaccard. The ground
        # truth sees 3360 mm of conveyor and the reconstruction about 2300, so
        # the union is always far larger than any intersection can be and even
        # a perfect alignment scores well under half. This reaches 1 when the
        # reconstruction sits properly inside the ground truth, which is what
        # "aligned" means when one cloud is a subset of the other.
        v = ov / max(int(np.count_nonzero(Aw | Bw)), 1)
        if v > best_v:
            best, best_v = k, v
    return float(best * args.cell), best_v


def slide_and_scale(gt_rig, recon, frame, args, fixed_slide=None):
    """Search the two parameters the belt cannot give: along-belt shift, scale.

    Scored by how much of the reconstruction finds a ground-truth neighbour
    within a tolerance, which on a loaded conveyor is dominated by the parcels
    and therefore has one sharp optimum. A plain nearest-neighbour RMS would be
    flatter, because sliding a belt along itself always leaves the deck matching
    the deck.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise SystemExit("scipy is required for --refine")
    ex = frame["ex"]
    rng = np.random.default_rng(0)
    sub = recon[rng.choice(len(recon), min(len(recon), args.refine_sample),
                           replace=False)]
    Q_ref = sub
    tree = cKDTree(gt_rig[rng.choice(len(gt_rig),
                                     min(len(gt_rig), args.refine_sample * 4),
                                     replace=False)])
    centre = frame["centre"]
    n = frame["n"]

    def warp(P, s, d):
        # About the WORLD ORIGIN, which is the reference camera centre, because
        # the gauge error lives in the depth and depth scales each point about
        # its own camera. Scaling about the belt centre instead leaves a point
        # near the deck near the deck whatever s is, so the search can never
        # close a normal-direction gap and every candidate scores alike.
        return P * s + ex * d

    def score(s, d):
        """Fraction of the reconstruction's PARCEL points that find a
        ground-truth neighbour, and where the deck ended up.

        Scoring the whole cloud lets the deck decide, and a belt slid along
        itself still puts deck on deck: the informative points are outvoted by
        the uninformative ones. Worse, a scale that lands the reconstruction's
        deck on the ground truth's FLOOR scores almost as well as the right
        one, which is how center_res1008 came back at 1.935 with its deck 852
        mm low. Restricting the score to what stands above the belt removes
        both, because the parcels are the only thing that is not
        translationally symmetric along the conveyor.
        """
        Q = warp(sub, s, d)
        h = (Q - centre) @ n
        off = float(np.median(h))
        if abs(off) > args.max_deck_off:
            return -1.0, off          # the deck is not on the deck
        sel = h > args.top_min
        if int(sel.sum()) < 500:
            return -1.0, off
        dist, _ = tree.query(Q[sel], k=1,
                             distance_upper_bound=args.refine_tol)
        return float(np.count_nonzero(np.isfinite(dist))) / int(sel.sum()), off

    slides = ([float(fixed_slide)] if fixed_slide is not None
              else list(np.arange(-args.slide_m, args.slide_m + 1e-9,
                                  args.slide_step)))
    best, best_v = (1.0, slides[0]), -1.0
    for s in np.arange(*args.scale_range, args.scale_step):
        for d in slides:
            v, _ = score(s, d)
            if v > best_v:
                best, best_v = (float(s), float(d)), v
    if best_v < 0:
        return 1.0, 0.0, 0.0, float("nan")
    # one refinement pass on a finer grid about the winner
    s0, d0 = best
    fine_d = ([d0] if fixed_slide is not None
              else list(np.arange(d0 - args.slide_step, d0 + args.slide_step,
                                  args.slide_step / 8)))
    for s in np.arange(s0 - args.scale_step, s0 + args.scale_step,
                       args.scale_step / 8):
        for d in fine_d:
            v, _ = score(s, d)
            if v > best_v:
                best, best_v = (float(s), float(d)), v
    s0, d0 = best
    _, off = score(s0, d0)
    return best[0], best[1], best_v, off * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pin the ground truth to the rig frame from the conveyor, "
                    "using only what the conveyor constrains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--belt", type=Path, default=Path("belt.json"))
    ap.add_argument("--points", type=Path, default=None,
                    help='json holding {"gt": [[x,y,z] x4]}')
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--max-width-mismatch-mm", type=float, default=250.0,
                    help="gross sanity only. The picked width need NOT match "
                         "belt.json: you may pick the deck surface (about 790 "
                         "mm on this rig) while belt.json spans the conveyor "
                         "frame (about 900 mm). Both are correct measurements "
                         "of different things, and it does not matter, because "
                         "deck and frame share a CENTRELINE and the centreline "
                         "is what positions the fit across the belt. The width "
                         "itself places nothing")
    ap.add_argument("--max-taper-mm", type=float, default=50.0,
                    help="difference between the two short edges of the picked "
                         "quadrilateral. A conveyor has parallel sides, so a "
                         "taper is a mis-picked corner")
    ap.add_argument("--refine", type=Path, default=None,
                    help="a run directory; solves the along-belt shift and the "
                         "reconstruction's scale against its cloud")
    ap.add_argument("--fused-name", default="fused_calib.ply")
    ap.add_argument("--reference", default="center")
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--slide-m", type=float, default=1.20,
                    help="half-range of the along-belt search. The ground "
                         "truth sees 3360 mm of conveyor against belt.json's "
                         "2990, so the two centres can differ by a good "
                         "fraction of a metre before anything is wrong")
    ap.add_argument("--slide-step", type=float, default=0.010)
    ap.add_argument("--scale-range", nargs=2, type=float, default=[0.9, 2.1])
    ap.add_argument("--scale-step", type=float, default=0.02)
    ap.add_argument("--refine-tol", type=float, default=0.020)
    ap.add_argument("--top-max", type=float, default=0.600,
                    help="upper height bound on what counts as a parcel")
    ap.add_argument("--cell", type=float, default=0.010,
                    help="raster cell for the footprint correlation")
    ap.add_argument("--slide-mm", type=float, default=None,
                    help="set the along-belt shift by hand, in millimetres, "
                         "and skip the search. This is the honest option: the "
                         "slide is one scalar, it is the single quantity a "
                         "conveyor cannot constrain, and every automatic proxy "
                         "tried here has been contaminated by something else. "
                         "An eye reads it off two clouds in a viewer in two "
                         "minutes and is not fooled by near-periodic parcels")
    ap.add_argument("--yaw180", action="store_true",
                    help="force the reversed-belt orientation, when the "
                         "footprints are too symmetric for the search to say")
    ap.add_argument("--cross", type=float, default=0.42,
                    help="half-width across the belt admitted to the "
                         "correlation. Set it to the DECK, not the frame: "
                         "rails run parallel to the conveyor and are "
                         "translation invariant along it, so they dilute the "
                         "score without informing it")
    ap.add_argument("--span", type=float, default=2.5,
                    help="half-length of the raster along the belt")
    ap.add_argument("--top-min", type=float, default=0.060,
                    help="height above the belt plane a reconstruction point "
                         "must reach to count towards the score. The deck is "
                         "translationally symmetric along the conveyor and "
                         "therefore carries no information about the slide; "
                         "only the parcels do")
    ap.add_argument("--max-deck-off", type=float, default=0.080,
                    help="a candidate whose deck lands further than this from "
                         "the belt plane is rejected outright. Without it the "
                         "search will happily put the reconstruction's deck on "
                         "the ground truth's floor, which is a 1.26x scale "
                         "error that scores well")
    ap.add_argument("--refine-sample", type=int, default=40000)
    ap.add_argument("--refine-all", type=Path, default=None,
                    help="after --refine has settled the orientation and the "
                         "slide on the reference run, solve the SCALE alone "
                         "for every run under this root. Orientation and slide "
                         "belong to where the ground truth sits, which is one "
                         "answer for the whole sweep; only the gauge is per "
                         "run, because each run is its own joint solve")
    ap.add_argument("--out", type=Path, default=Path("gt_align.json"))
    ap.add_argument("--out-scale", type=Path, default=Path("metric_fit.json"))
    args = ap.parse_args()

    belt = json.loads(args.belt.read_text())
    dst = rect_frame(np.asarray(belt["corners_m"], float))
    print(f"belt.json      width {dst['width'] * 1e3:7.1f} mm   length "
          f"{dst['length'] * 1e3:7.1f} mm   taper {dst['taper'] * 1e3:.1f} mm")

    if args.pick:
        src_pts = pick(args.gt)
        Path("belt_corners.json").write_text(
            json.dumps({"gt": src_pts.tolist()}, indent=2))
        print("picked corners saved to belt_corners.json")
    elif args.points:
        src_pts = np.asarray(json.loads(args.points.read_text())["gt"], float)
    else:
        raise SystemExit("give --points or --pick")
    if len(src_pts) != 4:
        raise SystemExit("exactly four corners are needed")

    order_before = [float(np.linalg.norm(src_pts[(i + 1) % 4] - src_pts[i]))
                    for i in range(4)]
    if abs(order_before[0] - order_before[1]) < 0.25 * max(order_before):
        print("[note ] the picked corners were not in order round the "
              "perimeter, so they have been sorted. Two adjacent edges of "
              "similar length means the quadrilateral was crossed and its "
              "long 'edges' were diagonals.")
    src_f = rect_frame(src_pts)
    print(f"picked in gt   width {src_f['width'] * 1e3:7.1f} mm   length "
          f"{src_f['length'] * 1e3:7.1f} mm   taper "
          f"{src_f['taper'] * 1e3:.1f} mm   out of plane "
          f"{src_f['flatness_mm']:.1f} mm")
    dw = abs(src_f["width"] - dst["width"]) * 1e3
    print(f"\nwidths differ by {dw:.1f} mm. Neither the width nor the length "
          f"is used to place anything:")
    print(f"  the LENGTH cannot be: belt.json ends where the cameras stopped "
          f"seeing, the ground truth sees the real conveyor")
    print(f"  the WIDTH need not be: deck surface and conveyor frame are "
          f"different widths but share a centreline, and it is the centreline "
          f"that positions the fit across the belt")
    if dw > args.max_width_mismatch_mm:
        raise SystemExit(
            f"the widths differ by {dw:.0f} mm, which is more than the deck "
            f"surface and the frame can plausibly differ by. Check that "
            f"belt.json is this conveyor at all.")
    taper_frac = src_f["taper"] / max(src_f["width"], 1e-6)
    print(f"picked taper {src_f['taper'] * 1e3:.1f} mm is "
          f"{taper_frac * 100:.1f} per cent of the picked width")
    if src_f["taper"] * 1e3 > args.max_taper_mm:
        raise SystemExit(
            f"the picked quadrilateral tapers by {src_f['taper'] * 1e3:.0f} mm "
            f"between its two short edges. A conveyor has parallel sides, so "
            f"one corner is off, most likely at the far end where the cloud "
            f"thins. Re-pick that one.")

    cands = candidates(src_f, dst)
    print(f"\n{len(cands)} orientation candidates. A rectangle is symmetric "
          f"under a 180 degree yaw and the conveyor cannot say which is right; "
          f"the parcels can.")
    T = cands[0][1]
    scale, slide, hit = 1.0, 0.0, None
    if args.refine:
        rp = args.refine / args.fused_name
        if not rp.exists():
            raise SystemExit(f"{rp} not found")
        recon = rigkit.load_cloud(str(rp))
        print(f"\nsearching the three the belt cannot give: orientation, "
              f"along-belt shift +/-{args.slide_m * 1e3:.0f} mm, scale "
              f"{args.scale_range[0]} to {args.scale_range[1]}")
        gt_all = rigkit.load_cloud(str(args.gt))
        clouds, cams_ref = load_per_camera(args.refine, args)
        if clouds is None:
            raise SystemExit(
                f"{args.refine} has no cloud_calib_<cam>.ply. Run "
                f"check_fusion.py --scale-mode none --rewrite first: the gauge "
                f"has to be applied per camera about each camera's own centre, "
                f"and the fused cloud cannot be taken apart again.")
        s_fixed = scale_from_deck_multi(clouds, dst)
        if args.slide_mm is not None:
            print(f"  gauge from the deck, each camera about its own centre: "
                  f"{s_fixed:.4f}")
            print(f"  along-belt shift set by hand at "
                  f"{args.slide_mm:+.0f} mm, no search")
            lab = "yaw180, normal+" if args.yaw180 else "yaw  0, normal+"
            T = dict(cands)[lab]
            T = np.array(T, float)
            T[:3, 3] += dst["ex"] * (args.slide_mm / 1e3)
            args.out.write_text(json.dumps(
                {"T": T.tolist(), "gt": str(args.gt), "rigid_only": True,
                 "solved_from": "conveyor for five degrees of freedom, "
                                "along-belt shift set by hand",
                 "candidate": lab,
                 "slide_mm": float(args.slide_mm),
                 "recon_scale": round(float(s_fixed), 6)}, indent=2))
            print(f"written {args.out}")
            table = {}
            for d_ in sorted(Path(args.refine_all).iterdir()) \
                    if args.refine_all else []:
                cl_, _ = load_per_camera(d_, args)
                if cl_ is None:
                    continue
                s_ = scale_from_deck_multi(cl_, dst)
                if not np.isfinite(s_):
                    print(f"  {d_.name:34s} NO SOLUTION: the median height "
                          f"never crosses the deck plane over the search "
                          f"range, so this cloud has no scale that puts its "
                          f"deck on the deck")
                    continue
                # What the gauge actually achieved. Without this a wrong scale
                # looks like a number rather than a failure, and three runs of
                # a four-camera subset came back needing 0.74, 1.06 and 1.40.
                W = warp_per_camera(cl_, s_, 0.0, dst["ex"])
                off = float(np.median((W - dst["centre"]) @ dst["n"])) * 1e3
                per = {c: round(float(np.median(
                    ((C + (P - C) * s_) - dst["centre"]) @ dst["n"])) * 1e3, 1)
                    for P, C in cl_
                    for c in [None]} if False else None
                table[d_.name] = {
                    "scale": round(float(s_), 6),
                    "deck_offset_mm": round(off, 1),
                    "slide_mm": float(args.slide_mm),
                    "solved_against": str(args.gt)}
                flag = "" if abs(off) <= 30 else "   DECK DID NOT LAND"
                print(f"  {d_.name:34s} scale {s_:.4f}  deck {off:+6.0f} mm"
                      f"{flag}")
            if table:
                args.out_scale.write_text(json.dumps(table, indent=2))
                print(f"written {args.out_scale} with {len(table)} runs")
            return 0
        print(f"  gauge from the deck, each camera about its own centre: "
              f"{s_fixed:.4f}")
        recon_scaled = warp_per_camera(clouds, s_fixed, 0.0, dst["ex"])
        results = []
        for label, Tc in cands:
            sl_, ov_ = slide_by_footprint(apply(gt_all, Tc), recon_scaled,
                                          dst, args)
            results.append((ov_, s_fixed, sl_, label, Tc, 0.0))
            print(f"  {label:20s} slide {sl_ * 1e3:+7.0f} mm  footprint "
                  f"overlap {ov_ * 100:5.1f} per cent")
        results.sort(reverse=True, key=lambda r: r[0])
        hit, scale, slide, label, T, deck_off = results[0]
        runner = results[1][0]
        print(f"\n  taking {label}: {hit * 100:.1f} per cent against "
              f"{runner * 100:.1f} for the next best")
        if hit < 1.3 * max(runner, 1e-9):
            print("[WARN] the winner is not clearly ahead. On synthetic "
                  "parcels the correct yaw scores about 1.7 times the reversed "
                  "one, so a ratio near 1 means the footprints are genuinely "
                  "symmetric under reversal and a marker is needed to settle "
                  "it.")
        if hit < 0.60:
            print(f"[WARN] the best overlap is only {hit * 100:.0f} per cent. "
                  f"Scored inside the reconstruction's own window a correct "
                  f"alignment reaches 100 in testing, so below about 60 "
                  f"something is wrong that the yaw will not explain.")
        if abs(deck_off) > 40:
            print(f"[WARN] at the chosen scale the reconstruction's deck still "
                  f"sits {deck_off:+.0f} mm from the belt plane. Widen "
                  f"--scale-range; the right scale is roughly the current one "
                  f"times {(1 + deck_off / 3070.0):.3f}.")
        if hit < 0.15:
            print("[WARN] that is a weak optimum. Widen --slide-m, or check "
                  "that the picked corners really are the conveyor frame.")
        # The slide belongs to the ground truth's placement, so it goes into T.
        T[:3, 3] -= dst["ex"] * slide
        args.out_scale.write_text(json.dumps(
            {args.refine.name: {"scale": round(float(scale), 6),
                                "slide_mm": round(float(slide) * 1e3, 1),
                                "neighbour_fraction": round(float(hit), 4),
                                "solved_against": str(args.gt)}}, indent=2))
        print(f"written {args.out_scale}")

    if args.refine_all and args.refine:
        gt_rig = apply(rigkit.load_cloud(str(args.gt)), T)
        print(f"\nsolving the gauge alone for every run under "
              f"{args.refine_all}, with the orientation and the "
              f"{slide * 1e3:+.0f} mm slide held at the reference answer")
        table = {}
        for d_ in sorted(Path(args.refine_all).iterdir()):
            rp_ = d_ / args.fused_name
            if not d_.is_dir() or not rp_.exists():
                continue
            cl_, _ = load_per_camera(d_, args)
            if cl_ is None:
                print(f"  {d_.name:34s} no per-camera clouds, skipped")
                continue
            s_ = scale_from_deck_multi(cl_, dst)
            R_ = warp_per_camera(cl_, s_, 0.0, dst["ex"])
            # The slide is solved per run, not frozen: scaling about the origin
            # displaces each run along the belt by (s-1) times the belt centre's
            # component on that axis, so a run at a different gauge needs a
            # different slide to sit in the same place.
            sl_, h_ = slide_by_footprint(gt_rig, R_, dst, args)
            off_ = 0.0
            table[d_.name] = {"scale": round(float(s_), 6),
                              "footprint_overlap": round(float(h_), 4),
                              "slide_mm": round(float(sl_) * 1e3, 1),
                              "solved_against": str(args.gt)}
            flag = "" if h_ >= 0.60 else "   POOR FOOTPRINT OVERLAP"
            print(f"  {d_.name:34s} scale {s_:.4f}  slide {sl_ * 1e3:+7.0f} mm"
                  f"  overlap {h_ * 100:5.1f}%{flag}")
        args.out_scale.write_text(json.dumps(table, indent=2))
        print(f"written {args.out_scale} with {len(table)} runs")

    args.out.write_text(json.dumps(
        {"T": T.tolist(), "gt": str(args.gt), "rigid_only": True,
         "solved_from": "conveyor plane and frame edges, five degrees of "
                        "freedom; along-belt shift "
                        + ("solved against " + args.refine.name
                           if args.refine else "NOT SOLVED"),
         "along_belt_solved": bool(args.refine),
         "recon_scale": round(float(scale), 6),
         "picked_gt_corners": np.asarray(src_pts).tolist(),
         "width_mismatch_mm": round(float(dw), 2),
         "picked_taper_mm": round(float(src_f["taper"]) * 1e3, 2)}, indent=2))
    print(f"written {args.out}")
    if not args.refine:
        print("\nThe along-belt position is still a guess. Re-run with "
              "--refine <a run dir> before grading, or every parcel will be "
              "compared against whichever parcel happens to be beside it.")
    else:
        print(f"\nNext:\n  python3 check_fusion.py --runs-root <root> "
              f"--scale-mode file --gauge {args.out_scale} --rewrite\n"
              f"  then re-run this with --refine once more; the second pass "
              f"should return a scale within a per cent of 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())