#!/usr/bin/env python3
"""
soft_segment.py

Segmentation of a capture that contains shrink-wrapped parcels, palletised
units such as a layer of 2.5 l bottles, or anything else whose top is not a
clean plane, together with the rejection of the detections that such scenes
produce and that are not objects at all.

Run this instead of box_segment.py. It does not replace it: it runs it, then
works on what it left behind.

    python soft_segment.py --capture-dir runs/live_.../capture_00000 \
        --seg-plane-distance 3.10 3.22 --seg-belt-file belt.json

Why a second pass rather than different thresholds
--------------------------------------------------
The planar path in box_segment.py is the better estimator on a bare carton and
should keep those parcels. It fails on film and on bottle layers for a reason
that no threshold reaches: it models the top as one plane and defines the
footprint as that plane's inliers. On a wrapped parcel the returns are sparse
and holed, so the inlier set is a fraction of the top; on a bottle layer there
is no plane to be an inlier of. Loosening --seg-face-thresh or
--seg-max-residual far enough to admit either would also admit the conveyor
frame, the pallet edge and every ramp of interpolated depth in the scene.

So the planar pass runs unchanged, and this script then takes the points it did
not claim, models them as a 2.5D upper envelope instead of a plane, and reports
what that envelope is: a wrapped box, a bottle layer, a deformed sack, or
nothing worth picking. See film_surfaces.py for the surface model.

The false positives
-------------------
Three artefact classes appear in these scenes and none of them is filtered by
size, aspect or area, because each is the right size and shape to be a parcel:

    curtains        depth interpolated across the silhouette of a taller box,
                    which lands as a thin sheet beside it at an intermediate
                    height
    film ghosts     a specular return from a film surface placed at the wrong
                    depth entirely, floating above the belt
    doubled tops    the same wrapped top reported twice because the cameras
                    disagreed about its depth by more than a cluster radius

The tests here are geometric and use only the camera centres. A solid resting
on the belt hides the deck beneath its own footprint from every camera, so if
deck-height points come back from inside the footprint, nothing solid is there.
Interpolated depth lies on the plane through a camera centre and a neighbour's
top edge, so a detection made of such points is that neighbour's curtain. And a
camera cannot see through a parcel, so if every camera that contributed points
to a detection has its line of sight to it blocked by another detection, those
points were never measurements of it.

Each rejection is written to the record with the test that made it and the
margin by which it failed, so a scene can be re-tuned without guessing.

Outputs
-------
boxes.json, boxes.csv, segmented.ply and the renders, written by
box_segment.write_results in the usual layout, with boxes.csv extended by the
surface and screening columns. Keep this file beside box_segment.py,
film_surfaces.py, face_consensus.py and da3_fuse.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

import box_segment as bs
import film_surfaces as fsx


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class SoftParams:
    """Everything the envelope model and the screening need."""

    enabled: bool = True
    recover: bool = True
    screen: bool = True

    # envelope
    film_cell: float = 0.006
    film_env_percentile: float = 90.0
    film_close: float = 0.030
    film_fill_iters: int = 4
    film_min_cells: int = 250
    film_max_relief: float = 0.080
    film_planar_rms: float = 0.006
    film_dense_fill: float = 0.60

    # recovery clustering
    recover_close: float = 0.040
    recover_min_points: int = 250
    recover_min_coverage: float = 0.30
    recover_min_area: float = 0.010
    recover_max_levels: int = 3
    reopen: bool = True
    reopen_coverage: float = 0.55
    reopen_inlier_fraction: float = 0.50

    # granular tops
    granular_relief: float = 0.025
    granular_pitch_min: float = 0.045
    granular_pitch_max: float = 0.250
    granular_peak_ratio: float = 0.40
    granular_skew: float = 0.50
    allow_granular_pick: bool = False
    view_shift: bool = True

    # suction
    suction_tol: float = 0.010
    min_flat_fraction: float = 0.05

    # screening
    max_see_through: float = 0.35
    see_through_shrink: float = 0.80
    max_bleed_fraction: float = 0.35
    bleed_tol: float = 0.015
    bleed_min_drop: float = 0.050
    min_height_sigma: float = 3.0
    occlusion_check: bool = True

    # overlap resolution between the two passes
    prefer_soft_area_ratio: float = 1.30


class Params:
    """SegParams and SoftParams behind one attribute lookup.

    film_surfaces.py reads a handful of names that belong to the segmentation
    proper, such as top_band and pick_radius, and a handful that belong here.
    Delegating rather than duplicating keeps one definition of each.
    """

    def __init__(self, seg: bs.SegParams, soft: SoftParams):
        self.seg = seg
        self.soft = soft

    def __getattr__(self, name):
        if hasattr(self.soft, name):
            return getattr(self.soft, name)
        return getattr(self.seg, name)


def add_soft_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("wrapped and non-planar surfaces")
    g.add_argument("--soft-off", dest="soft_enabled", action="store_false",
                   default=True,
                   help="run the planar pass alone, exactly as box_segment.py "
                        "would, and write its result unaltered")
    g.add_argument("--soft-no-recover", dest="soft_recover",
                   action="store_false", default=True,
                   help="do not look for wrapped or granular tops in the "
                        "points the planar pass left unassigned; screen only")
    g.add_argument("--soft-no-screen", dest="soft_screen",
                   action="store_false", default=True,
                   help="do not apply the artefact tests. Every detection is "
                        "kept, including the curtains hanging off the taller "
                        "parcels, which is what the planar pass alone reports")

    g.add_argument("--soft-cell", type=float, default=0.006,
                   help="raster cell of the height envelope. Below the point "
                        "spacing it produces a hole in every other cell; well "
                        "above it the footprint is quantised to the cell")
    g.add_argument("--soft-env-percentile", type=float, default=90.0,
                   help="percentile of the points in a cell taken as its "
                        "height. 100 is the maximum and lets one flyer per "
                        "cell lift the whole surface")
    g.add_argument("--soft-close", type=float, default=0.030,
                   help="largest hole in a top surface that is bridged when "
                        "deciding whether it is one surface. Set it above the "
                        "hole a specular highlight leaves and below the gap "
                        "between two parcels")
    g.add_argument("--soft-fill-iters", type=int, default=4,
                   help="diffusion passes used to give an enclosed hole a "
                        "height. Only holes fully enclosed by the surface are "
                        "filled, so this cannot grow the footprint outwards")
    g.add_argument("--soft-min-cells", type=int, default=250,
                   help="cells a top surface must occupy to be fitted at all")
    g.add_argument("--soft-max-relief", type=float, default=0.080,
                   help="largest peak-to-trough deviation from the fitted "
                        "plane still called a surface. Above this the points "
                        "are not one top: they are two parcels, or a parcel "
                        "and the frame behind it")
    g.add_argument("--soft-planar-rms", type=float, default=0.006,
                   help="plane residual below which the top is called planar "
                        "and the planar pass's own estimate is kept")
    g.add_argument("--soft-dense-fill", type=float, default=0.60,
                   help="fill ratio above which a top counts as densely "
                        "measured; below it the top is treated as film even if "
                        "it is flat, because a flat sheet of half-measured "
                        "returns is what a transparent wrap gives")

    g.add_argument("--soft-recover-close", type=float, default=0.040,
                   help="gap bridged when grouping the unassigned points into "
                        "candidate surfaces. This is what puts a wrapped "
                        "parcel back together after the depth returns from it "
                        "were split into fragments")
    g.add_argument("--soft-recover-min-points", type=int, default=250)
    g.add_argument("--soft-recover-min-coverage", type=float, default=0.30,
                   help="fraction of the fitted rectangle that must carry "
                        "measured cells for a recovered surface to be "
                        "accepted. Low by design: a transparent wrap returns "
                        "far less than a carton, and demanding carton-like "
                        "coverage is exactly what loses it")
    g.add_argument("--soft-recover-min-area", type=float, default=0.010)
    g.add_argument("--soft-no-reopen", dest="soft_reopen",
                   action="store_false", default=True,
                   help="leave the planar pass's detections alone. By default "
                        "a planar detection whose own rectangle is mostly "
                        "empty has its points returned to the pool before the "
                        "recovery runs, because that is the signature of a "
                        "plane fitted to the coherent half of a wrapped top: "
                        "the parcel is reported at half its size and the other "
                        "half is consumed along with it, so the recovery would "
                        "otherwise never see the points it needs")
    g.add_argument("--soft-reopen-coverage", type=float, default=0.55,
                   help="rectangle occupancy below which a planar detection is "
                        "re-opened. Matches --seg-min-coverage, which is the "
                        "value at which box_segment.py already declines to "
                        "call the detection pickable")
    g.add_argument("--soft-reopen-inlier-fraction", type=float, default=0.50,
                   help="fraction of the top band that fell within the plane, "
                        "below which the detection is re-opened. A holed film "
                        "fails this before it fails coverage")
    g.add_argument("--soft-recover-levels", type=int, default=3,
                   help="surfaces peeled from one group of unassigned points. "
                        "Two tops that overlap in plan but sit at different "
                        "heights, a wrapped parcel beside a pallet for "
                        "instance, are one group on the raster, and fitting it "
                        "once returns the higher and loses the other")

    g.add_argument("--soft-granular-relief", type=float, default=0.025,
                   help="relief above which a periodic top is called granular, "
                        "which is the case of a bottle layer: caps standing "
                        "above shoulders, no plane anywhere on it")
    g.add_argument("--soft-granular-pitch", nargs=2, type=float,
                   default=[0.045, 0.250], metavar=("MIN_M", "MAX_M"),
                   help="range of repeat spacings searched for. A 2.5 l PET "
                        "bottle stands on roughly a 0.10 to 0.12 m pitch")
    g.add_argument("--soft-granular-peak", type=float, default=0.40,
                   help="autocorrelation peak, relative to zero lag, above "
                        "which the relief is called periodic rather than "
                        "wrinkled")
    g.add_argument("--soft-granular-skew", type=float, default=0.50,
                   help="skewness of the height distribution above which a "
                        "periodic top is called granular. Caps are a small "
                        "part of the area standing well above a broad base and "
                        "are strongly skewed; folds in a film are close to "
                        "symmetric and have a correlation length that would "
                        "otherwise read as a pitch")
    g.add_argument("--soft-no-view-shift", dest="soft_view_shift",
                   action="store_false", default=True,
                   help="fit the envelope on the raw points. By default each "
                        "camera is shifted onto the weighted consensus of the "
                        "views on that surface first, bounded by "
                        "--seg-max-view-shift, so the disagreement between the "
                        "cameras is not added to the relief of the object")
    g.add_argument("--soft-allow-granular-pick", action="store_true",
                   help="call a granular top pickable. By default it is held: "
                        "a flat cup has nothing to seal against on a field of "
                        "caps, and the flat-area fraction reported for it is "
                        "the evidence")

    g.add_argument("--soft-suction-tol", type=float, default=0.010,
                   help="height variation a cup of --seg-pick-radius is "
                        "assumed to tolerate under its footprint. Set it to "
                        "the compliance of the cup actually fitted: a bellows "
                        "cup swallows a centimetre of wrinkle, a flat one does "
                        "not, and the difference decides whether a wrapped "
                        "parcel is reported as pickable")
    g.add_argument("--soft-min-flat-fraction", type=float, default=0.05,
                   help="fraction of the top that must offer a seat that flat "
                        "before a non-planar surface is called pickable")

    g.add_argument("--soft-max-see-through", type=float, default=0.35,
                   help="fraction of a detection's own footprint where the "
                        "deck may still be visible beneath it. A solid on the "
                        "belt hides the deck from every camera, so above this "
                        "the detection is floating and is rejected")
    g.add_argument("--soft-see-through-shrink", type=float, default=0.80,
                   help="the footprint is shrunk by this factor before the "
                        "test, so deck just outside a slightly over-sized "
                        "rectangle does not count against it")
    g.add_argument("--soft-max-bleed", type=float, default=0.35,
                   help="fraction of a detection's points that may lie on the "
                        "sight line from their own camera past the top edge of "
                        "a taller neighbour. Above this the detection is that "
                        "neighbour's interpolated curtain")
    g.add_argument("--soft-bleed-tol", type=float, default=0.015,
                   help="distance from that sight plane still counted as on it")
    g.add_argument("--soft-bleed-min-drop", type=float, default=0.050,
                   help="how much taller a neighbour must be before it is "
                        "considered capable of casting a curtain")
    g.add_argument("--soft-min-height-sigma", type=float, default=3.0,
                   help="how many times the measured depth noise a detection "
                        "must stand above the deck. The noise used is the "
                        "deck plane residual combined with the disagreement "
                        "between the cameras on that face, so a 25 mm object "
                        "reported from views that disagree by 26 mm is not a "
                        "measurement of anything")
    g.add_argument("--soft-no-occlusion-check", dest="soft_occlusion",
                   action="store_false", default=True,
                   help="skip the test that rejects a detection every "
                        "contributing camera was blocked from seeing")
    g.add_argument("--soft-prefer-area-ratio", type=float, default=1.30,
                   help="a recovered surface overlapping a planar detection "
                        "replaces it when its footprint is at least this much "
                        "larger, which is the case of a wrapped parcel fitted "
                        "to half its top by the planar pass")


def soft_params_from_args(a) -> SoftParams:
    return SoftParams(
        enabled=a.soft_enabled, recover=a.soft_recover, screen=a.soft_screen,
        film_cell=a.soft_cell, film_env_percentile=a.soft_env_percentile,
        film_close=a.soft_close, film_fill_iters=a.soft_fill_iters,
        film_min_cells=a.soft_min_cells, film_max_relief=a.soft_max_relief,
        film_planar_rms=a.soft_planar_rms, film_dense_fill=a.soft_dense_fill,
        recover_close=a.soft_recover_close,
        recover_min_points=a.soft_recover_min_points,
        recover_min_coverage=a.soft_recover_min_coverage,
        recover_min_area=a.soft_recover_min_area,
        recover_max_levels=a.soft_recover_levels,
        reopen=a.soft_reopen, reopen_coverage=a.soft_reopen_coverage,
        reopen_inlier_fraction=a.soft_reopen_inlier_fraction,
        granular_relief=a.soft_granular_relief,
        granular_pitch_min=a.soft_granular_pitch[0],
        granular_pitch_max=a.soft_granular_pitch[1],
        granular_peak_ratio=a.soft_granular_peak,
        granular_skew=a.soft_granular_skew,
        view_shift=a.soft_view_shift,
        allow_granular_pick=a.soft_allow_granular_pick,
        suction_tol=a.soft_suction_tol,
        min_flat_fraction=a.soft_min_flat_fraction,
        max_see_through=a.soft_max_see_through,
        see_through_shrink=a.soft_see_through_shrink,
        max_bleed_fraction=a.soft_max_bleed, bleed_tol=a.soft_bleed_tol,
        bleed_min_drop=a.soft_bleed_min_drop,
        min_height_sigma=a.soft_min_height_sigma,
        occlusion_check=a.soft_occlusion,
        prefer_soft_area_ratio=a.soft_prefer_area_ratio,
    )


# --------------------------------------------------------------------------
# helpers shared with the planar record
# --------------------------------------------------------------------------

def frame_axes(conveyor):
    n = np.asarray(conveyor["normal"], float)
    d = float(conveyor["offset_m"])
    ex = np.asarray(conveyor["frame_x_axis"], float)
    ey = np.asarray(conveyor["frame_y_axis"], float)
    return n, d, ex, ey


def conv_uv(points, ex, ey):
    q = np.atleast_2d(np.asarray(points, float))
    return np.stack([q @ ex, q @ ey], axis=1)


def belt_mask(P, conveyor, margin):
    belt = (conveyor or {}).get("belt")
    if not belt:
        return np.ones(len(P), bool)
    c = np.asarray(belt["centre_m"], float)
    bx = np.asarray(belt["x_axis"], float)
    by = np.asarray(belt["y_axis"], float)
    rel = P - c
    return ((np.abs(rel @ bx) <= belt["length_m"] / 2 + margin)
            & (np.abs(rel @ by) <= belt["width_m"] / 2 + margin))


def pose_from(rect_centre, x_axis, normal):
    z = -np.asarray(normal, float)
    x = np.asarray(x_axis, float)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(rect_centre, float)
    return T, R


def make_box(face, rect, pts, src, cam_names, n_conv, d_conv, ex, ey, p,
             cluster, level, origin):
    """A record in the schema box_segment.py writes, from an envelope fit."""
    surf = face["surface"]
    h_face = float(np.asarray(rect["centre"]) @ n_conv + d_conv)
    T, R = pose_from(rect["centre"], rect["x_axis"], face["normal"])
    T_pre = T.copy()
    T_pre[:3, 3] = np.asarray(rect["centre"]) + np.asarray(face["normal"]) * p.approach_offset

    yaw = float(np.degrees(np.arctan2(float(R[:, 0] @ ey), float(R[:, 0] @ ex))))
    yaw = (yaw + 90.0) % 180.0 - 90.0

    counts = {cam_names[i]: int(np.count_nonzero(np.asarray(src) == i))
              for i in range(len(cam_names))}
    total = max(sum(counts.values()), 1)
    dominant = max(counts.values()) / total
    used = [k for k, v in counts.items() if v >= p.min_view_face_points]
    spread = surf.get("inter_view_offset_mm")

    pick = surf.get("pick_point_m")
    T_pick = None
    if pick is not None:
        T_pick = T.copy()
        T_pick[:3, 3] = np.asarray(pick, float)

    return {
        "id": -1,
        "cluster": int(cluster),
        "level": int(level),
        "n_points": int(len(pts)),
        "origin": origin,
        "top_face": {
            "centre_m": np.asarray(rect["centre"]).tolist(),
            "normal": np.asarray(face["normal"]).tolist(),
            "corners_m": np.asarray(rect["corners"]).tolist(),
            "height_above_conveyor_m": h_face,
            "tilt_deg": face["tilt_deg"],
            "plane_residual_rms_m": face["residual_rms_m"],
            "plane_residual_p95_m": face["residual_p95_m"],
            "inliers": face["inliers"],
            "inlier_fraction": face["inlier_fraction"],
            "coverage": rect["coverage"],
            "area_m2": rect["area_m2"],
            "growth": None,
        },
        "surface": surf,
        "dimensions_m": {
            "length": rect["length_m"],
            "width": rect["width_m"],
            "top_height_above_conveyor": h_face,
        },
        "pose_world": T.tolist(),
        "pre_pick_pose_world": T_pre.tolist(),
        "pick_pose_world": None if T_pick is None else T_pick.tolist(),
        "position_m": np.asarray(rect["centre"]).tolist(),
        "quaternion_wxyz": bs.rot_to_quat(R).tolist(),
        "rpy_deg_zyx": bs.rot_to_rpy_deg(R),
        "approach_vector": (-np.asarray(face["normal"])).tolist(),
        "yaw_in_conveyor_frame_deg": yaw,
        "points_per_camera": counts,
        "dominant_camera_fraction": float(dominant),
        "single_view": bool(dominant >= p.single_view_frac),
        "view_consensus": {
            "ok": True,
            "views_used": used,
            "views_dropped": [],
            "skipped": [],
            "per_view": surf.get("per_view", {}),
            "inter_view_offset_mm": spread,
            "height_uncertainty_mm": (None if spread is None
                                      else round(spread / 2.0, 2)),
            "single_view": len(used) <= 1,
            "source": "envelope fit, not the planar consensus",
        },
    }


# --------------------------------------------------------------------------
# recovery of wrapped and granular tops
# --------------------------------------------------------------------------

def recover_surfaces(result, p, cam_names, free_extra=None):
    """Fit the points the planar pass did not claim.

    Those points are precisely the ones a plane could not describe: the holed
    returns off a wrapped top, the cap field of a bottle layer, and the
    fragments a sparse top was shattered into. Grouping them on the raster
    rather than in 3D is what reassembles a fragmented surface, because two
    fragments of one top are adjacent in plan whatever their depths.

    free_extra optionally adds points already claimed by a planar detection
    that was re-opened. A plane fitted to the coherent half of a wrapped top
    consumes the whole footprint it drew, so without this the recovery is
    handed the leftovers of the mistake rather than the surface.
    """
    P = np.asarray(result["points"])
    S = np.asarray(result["sources"])
    labels = np.asarray(result["labels"])
    conv = result.get("conveyor")
    if conv is None or not len(P):
        return [], []
    n, d, ex, ey = frame_axes(conv)
    h = P @ n + d

    unclaimed = labels == -1
    if free_extra is not None:
        unclaimed = unclaimed | np.asarray(free_extra, bool)
    free = (unclaimed & (h > p.min_height) & (h < p.max_height)
            & belt_mask(P, conv, p.belt_margin))
    idx_free = np.flatnonzero(free)
    if len(idx_free) < p.recover_min_points:
        return [], []

    grid = fsx.plan_grid(P[idx_free], n, d, ex, ey, p.film_cell)
    if grid is None:
        return [], []
    _, cnt = fsx.envelope(grid, p.film_env_percentile)
    close_cells = int(round(p.recover_close / grid["cell"]))
    occ = (cnt > 0).astype(np.uint8) * 255
    k = 2 * max(close_cells, 1) + 1
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)

    boxes, rejected = [], []
    cell_lab = lab[grid["iv"], grid["iu"]]
    for c in range(1, n_lab):
        member = cell_lab == c
        n_pts = int(member.sum())
        if n_pts < p.recover_min_points:
            continue
        # Peeled one surface at a time. Two unassigned tops that overlap in
        # plan but sit at different heights, a wrapped parcel beside a pallet
        # for instance, form one raster component; fitting the component once
        # returns the higher of them and loses the other silently.
        remaining = idx_free[member]
        last_h = None
        for level in range(max(1, int(p.recover_max_levels))):
            if len(remaining) < p.recover_min_points:
                break
            pts, src = P[remaining], S[remaining]
            face, rect, why = fsx.fit_surface(pts, src, n, d, ex, ey, p,
                                              cam_names=cam_names)
            if face is None:
                if level == 0:
                    rejected.append({"cluster": f"soft{c}", "n_points": n_pts,
                                     "reason": f"unassigned surface: {why}",
                                     "origin": "recovery"})
                break
            surf = face["surface"]
            h_face = float(np.asarray(rect["centre"]) @ n + d)

            # Everything inside this footprint at or below this surface belongs
            # to it: its own top, its walls and whatever it hides.
            rel = pts - np.asarray(rect["centre"])
            uu = rel @ np.asarray(rect["x_axis"])
            vv = rel @ np.asarray(rect["y_axis"])
            h_r = pts @ n + d
            margin = 2.0 * p.film_cell
            consumed = ((np.abs(uu) <= rect["length_m"] / 2 + margin)
                        & (np.abs(vv) <= rect["width_m"] / 2 + margin)
                        & (h_r <= h_face + surf["relief_mm"] / 1e3 + margin))
            sel = remaining[consumed]
            remaining = remaining[~consumed]

            if last_h is not None and last_h - h_face < p.level_min_gap:
                rejected.append({"cluster": f"soft{c}.{level}",
                                 "n_points": int(len(sel)),
                                 "origin": "recovery",
                                 "reason": f"only {(last_h - h_face) * 1e3:.0f} "
                                           f"mm below the surface already "
                                           f"taken, so it is the same one"})
                continue

            aspect = rect["length_m"] / max(rect["width_m"], 1e-6)
            why = None
            if len(sel) < p.recover_min_points:
                why = (f"only {len(sel)} points fall inside the footprint it "
                       f"was fitted to")
            elif rect["width_m"] < p.min_dim or rect["length_m"] > p.max_dim:
                why = (f"footprint {rect['length_m'] * 1e3:.0f} x "
                       f"{rect['width_m'] * 1e3:.0f} mm outside the accepted "
                       f"range")
            elif aspect > p.max_aspect:
                why = f"footprint aspect {aspect:.1f} above {p.max_aspect:.1f}"
            elif rect["area_m2"] < p.recover_min_area:
                why = (f"top {rect['area_m2'] * 1e4:.0f} cm2, below "
                       f"{p.recover_min_area * 1e4:.0f} cm2")
            elif rect["coverage"] < p.recover_min_coverage:
                why = (f"only {rect['coverage'] * 100:.0f} per cent of the "
                       f"rectangle carries measured cells, below "
                       f"{p.recover_min_coverage * 100:.0f}")
            elif surf["relief_mm"] > p.film_max_relief * 1e3:
                why = (f"relief {surf['relief_mm']:.0f} mm exceeds "
                       f"{p.film_max_relief * 1e3:.0f} mm, so these points are "
                       f"not one surface")
            if why is not None:
                rejected.append({"cluster": f"soft{c}.{level}",
                                 "n_points": int(len(sel)), "reason": why,
                                 "surface_class": surf["class"],
                                 "origin": "recovery"})
                continue

            last_h = h_face
            b = make_box(face, rect, P[sel], S[sel], cam_names, n, d, ex, ey, p,
                         cluster=1000 + c, level=level, origin="recovery")
            b["_label_sel"] = sel
            boxes.append(b)
    return boxes, rejected


def resolve_overlaps(planar, recovered, p, ex, ey):
    """Keep one detection where the two passes found the same top.

    A recovered surface replaces a planar one when it covers substantially more
    of the same footprint, which is the wrapped parcel the planar pass fitted to
    the coherent half of. Otherwise the planar estimate stands, because on a
    bare carton a plane and its inliers measure the top better than a raster
    does.
    """
    for b in planar + recovered:
        b["_uv"] = conv_uv(np.asarray(b["top_face"]["corners_m"]), ex, ey)
    kept = list(planar)
    dropped = []
    for r in recovered:
        clash, iou = None, 0.0
        for k in kept:
            dh = abs(r["top_face"]["height_above_conveyor_m"]
                     - k["top_face"]["height_above_conveyor_m"])
            if dh >= p.level_min_gap * 2:
                continue
            v = bs.rect_iou(r["_uv"], k["_uv"])
            if v > p.max_overlap and v > iou:
                clash, iou = k, v
        if clash is None:
            kept.append(r)
            continue
        ratio = (r["top_face"]["area_m2"]
                 / max(clash["top_face"]["area_m2"], 1e-9))
        if ratio >= p.prefer_soft_area_ratio:
            dropped.append({"cluster": clash["cluster"],
                            "n_points": clash["n_points"],
                            "origin": clash.get("origin", "planar"),
                            "reason": (f"the planar fit covered "
                                       f"{100.0 / ratio:.0f} per cent of the "
                                       f"top the envelope found on the same "
                                       f"footprint, so it was fitted to part "
                                       f"of a {r['surface']['class']} surface")})
            kept[kept.index(clash)] = r
        else:
            dropped.append({"cluster": r["cluster"], "n_points": r["n_points"],
                            "origin": "recovery",
                            "reason": (f"overlaps the planar detection at "
                                       f"{clash['top_face']['height_above_conveyor_m'] * 1e3:.0f} "
                                       f"mm by {iou:.2f} without covering more "
                                       f"of it")})
    return kept, dropped


# --------------------------------------------------------------------------
# artefact screening
# --------------------------------------------------------------------------

def screen(boxes, result, p, cam_names, cam_centres):
    """Apply the see-through, bleed and occlusion tests."""
    P = np.asarray(result["points"])
    S = np.asarray(result["sources"])
    conv = result["conveyor"]
    n, d, ex, ey = frame_axes(conv)
    h_all = P @ n + d
    deck_rms = float(conv.get("inlier_rms_m") or 0.002)
    deck_spread = float(conv.get("per_view_spread_mm") or 0.0) / 1e3
    deck_band = max(p.plane_thresh * 2.0, 0.006)

    prisms = []
    for b in boxes:
        tf = b["top_face"]
        prisms.append({
            "centre": np.asarray(tf["centre_m"], float),
            "x_axis": np.asarray(b["pose_world"], float)[:3, 0],
            "y_axis": np.asarray(b["pose_world"], float)[:3, 1],
            "length": b["dimensions_m"]["length"],
            "width": b["dimensions_m"]["width"],
            "top_h": tf["height_above_conveyor_m"],
            "corners": np.asarray(tf["corners_m"], float),
        })

    kept, dropped = [], []
    for i, b in enumerate(boxes):
        tf = b["top_face"]
        R = np.asarray(b["pose_world"], float)[:3, :3]
        h_top = tf["height_above_conveyor_m"]
        sel = b.get("_label_sel")
        pts = P[sel] if sel is not None and len(sel) else np.empty((0, 3))
        src = S[sel] if sel is not None and len(sel) else np.empty(0, np.int32)
        contributing = [j for j, name in enumerate(cam_names)
                        if b["points_per_camera"].get(name, 0) >= p.min_view_points]

        vc = b.get("view_consensus") or {}
        face_spread = float(vc.get("inter_view_offset_mm") or 0.0) / 1e3
        sigma = float(np.sqrt(deck_rms ** 2 + (deck_spread / 2.0) ** 2
                              + (face_spread / 2.0) ** 2)) + 1e-6
        h_sigma = h_top / sigma

        st, n_deck = fsx.see_through_fraction(
            P, h_all, tf["corners_m"], np.asarray(tf["centre_m"], float),
            R[:, 0], R[:, 1], b["dimensions_m"]["length"],
            b["dimensions_m"]["width"], deck_band, p.see_through_shrink)

        others = [prisms[j] for j in range(len(boxes)) if j != i]
        bleed, per_cam = (0.0, {})
        if len(pts) and cam_centres is not None and contributing:
            bleed, per_cam = fsx.bleed_fraction(
                pts, src, contributing, cam_centres, others, h_top, n, d,
                p.bleed_tol, p.bleed_min_drop)

        blocked = []
        if p.occlusion_check and cam_centres is not None and contributing:
            c = np.asarray(tf["centre_m"], float)
            corners = np.asarray(tf["corners_m"], float)
            samples = [c] + [c + 0.7 * (q - c) for q in corners]
            blocked = fsx.blocked_views(samples, contributing, cam_centres,
                                        others, n, d)

        b["screening"] = {
            "see_through_fraction": round(float(st), 3),
            "deck_points_under": int(n_deck),
            "bleed_fraction": round(float(bleed), 3),
            "bleed_per_camera": {cam_names[k]: v for k, v in per_cam.items()},
            "height_sigma": round(float(h_sigma), 2),
            "noise_used_mm": round(sigma * 1e3, 2),
            "views_blocked": [cam_names[j] for j in blocked],
            "views_contributing": [cam_names[j] for j in contributing],
            "see_through_applies": bool(h_top > deck_band + 3.0 * deck_rms),
        }

        why = None
        separable = h_top > deck_band + 3.0 * deck_rms
        if st > p.max_see_through and separable:
            why = (f"the deck is visible through {st * 100:.0f} per cent of "
                   f"this footprint, above {p.max_see_through * 100:.0f}, so "
                   f"nothing solid is standing there")
        elif bleed > p.max_bleed_fraction:
            why = (f"{bleed * 100:.0f} per cent of its points lie on the sight "
                   f"line past the top edge of a taller parcel, so this is "
                   f"that parcel's interpolated curtain, not an object")
        elif contributing and blocked and len(blocked) == len(contributing):
            why = (f"every camera that contributed points here "
                   f"({', '.join(b['screening']['views_blocked'])}) had its "
                   f"line of sight blocked by another parcel, so those points "
                   f"cannot be measurements of this surface")
        elif h_sigma < p.min_height_sigma:
            why = (f"it stands {h_top * 1e3:.0f} mm above the deck against "
                   f"{sigma * 1e3:.1f} mm of measured noise, a ratio of "
                   f"{h_sigma:.1f}, below the {p.min_height_sigma:.1f} "
                   f"required")
        if why is None:
            kept.append(b)
        else:
            dropped.append({"cluster": b["cluster"], "n_points": b["n_points"],
                            "origin": b.get("origin", "planar"),
                            "reason": why, "screening": b["screening"]})
    return kept, dropped


# --------------------------------------------------------------------------
# relations and pickability, recomputed over the surviving detections
# --------------------------------------------------------------------------

def relations(boxes, p, ex, ey, conveyor):
    for b in boxes:
        b["_uv"] = conv_uv(np.asarray(b["top_face"]["corners_m"]), ex, ey)
        b["_cuv"] = conv_uv(np.asarray(b["top_face"]["centre_m"]), ex, ey)[0]

    for b in boxes:
        gaps, blocked_by, supports = [], [], []
        for o in boxes:
            if o["id"] == b["id"]:
                continue
            gap = bs.polygon_gap(b["_uv"], o["_uv"])
            gaps.append((gap, o["id"]))
            dh = (o["top_face"]["height_above_conveyor_m"]
                  - b["top_face"]["height_above_conveyor_m"])
            if gap < 0.005 and dh > 0.020:
                blocked_by.append(o["id"])
            if dh < -0.020 and bs.rect_contains(o["_uv"], b["_cuv"]):
                supports.append((o["top_face"]["height_above_conveyor_m"],
                                 o["id"]))
        gaps.sort()
        supports.sort(reverse=True)
        support_h = supports[0][0] if supports else 0.0
        b["dimensions_m"]["height"] = float(
            b["top_face"]["height_above_conveyor_m"] - support_h)
        b["support"] = {"on_conveyor": not supports,
                        "supported_by": supports[0][1] if supports else None,
                        "support_height_m": float(support_h),
                        "height_is_inferred": bool(supports)}
        b["nearest_neighbour"] = {"id": gaps[0][1] if gaps else None,
                                  "gap_m": float(gaps[0][0]) if gaps else None}
        b["blocked_by"] = blocked_by

        surf = b.get("surface") or {}
        kind = surf.get("class", "planar")
        # The planar pass's reasons are rebuilt rather than appended to. Its
        # residual and coverage tests do not apply to a surface that was never
        # claimed to be planar, and its relational tests were computed against
        # a set of detections that no longer exists.
        stale = ("plane residual", "own rectangle", "taller parcel is adjacent",
                 "to the nearest parcel", "height inferred", "belt search bound")
        blockers = [r for r in b.get("blocking_reasons", [])
                    if not any(k in r for k in stale)]
        cautions = [r for r in b.get("cautions", [])
                    if not any(k in r for k in stale)]

        if kind in ("film", "rough", "granular"):
            flat = surf.get("flat_area_fraction", 0.0)
            seat = surf.get("pick_flatness_mm")
            if kind == "granular" and not p.allow_granular_pick:
                units = (surf.get("units") or {}).get("count")
                blockers.append(
                    f"the top is a {surf.get('pitch_mm', 0):.0f} mm periodic "
                    f"field with {surf.get('relief_mm', 0):.0f} mm of relief"
                    + (f" and about {units} repeating units" if units else "")
                    + ", so a flat cup has nothing to seal against; pass "
                      "--soft-allow-granular-pick to override")
            elif flat < p.min_flat_fraction or seat is None:
                blockers.append(
                    f"no seat of {p.pick_radius * 2e3:.0f} mm flat to within "
                    f"{p.suction_tol * 1e3:.0f} mm exists anywhere on this "
                    f"{kind} top ({flat * 100:.0f} per cent of it qualifies)")
            else:
                cautions.append(
                    f"{kind} top, {surf.get('relief_mm', 0):.0f} mm relief, "
                    f"picked at the flattest seat found "
                    f"({seat:.1f} mm across the cup) rather than at the centre "
                    f"of the footprint")
            if surf.get("fill_ratio", 1.0) < 0.5:
                cautions.append(
                    f"only {surf.get('fill_ratio', 0) * 100:.0f} per cent of "
                    f"the top returned depth, which is what a specular or "
                    f"transparent wrap does; the footprint is interpolated "
                    f"across the rest")
        else:
            if b["top_face"]["plane_residual_rms_m"] > p.max_residual:
                blockers.append(
                    f"plane residual "
                    f"{b['top_face']['plane_residual_rms_m'] * 1e3:.1f} mm "
                    f"exceeds the {p.max_residual * 1e3:.1f} mm limit")
            if b["top_face"]["coverage"] < p.min_coverage:
                blockers.append(
                    f"the top face fills only "
                    f"{b['top_face']['coverage'] * 100:.0f} per cent of its "
                    f"own rectangle, so it is partly occluded")

        if blocked_by:
            blockers.append(f"a taller parcel is adjacent: {blocked_by}")
        gap0 = b["nearest_neighbour"]["gap_m"]
        if gap0 is not None and 0 <= gap0 < p.clearance_warn:
            cautions.append(f"only {gap0 * 1e3:.0f} mm to the nearest parcel")
        if b["support"]["height_is_inferred"]:
            cautions.append(f"height inferred from parcel "
                            f"{b['support']['supported_by']} beneath it")
        scr = b.get("screening") or {}
        if scr.get("see_through_fraction", 0) > p.max_see_through * 0.6:
            cautions.append(
                f"the deck shows through {scr['see_through_fraction'] * 100:.0f} "
                f"per cent of the footprint, so part of this rectangle is over "
                f"empty belt")
        if scr.get("bleed_fraction", 0) > p.max_bleed_fraction * 0.5:
            cautions.append(
                f"{scr['bleed_fraction'] * 100:.0f} per cent of its points sit "
                f"on a neighbour's sight edge, so the footprint is inflated "
                f"towards that neighbour")
        if scr.get("views_blocked"):
            cautions.append(
                f"points attributed to {', '.join(scr['views_blocked'])} could "
                f"not have been seen from there")

        b["pickable"] = not blockers
        b["blocking_reasons"] = blockers
        b["cautions"] = cautions
        b["reasons"] = blockers + cautions

    for rank, b in enumerate(sorted(
            boxes, key=lambda x: (-x["top_face"]["height_above_conveyor_m"],
                                  -x["n_points"]))):
        b["pick_order"] = rank
    for b in boxes:
        b.pop("_uv", None)
        b.pop("_cuv", None)
    return boxes


# --------------------------------------------------------------------------
# the pass itself
# --------------------------------------------------------------------------

def refine(result, p, cam_names, cam_centres):
    """Recover the non-planar tops, reject the artefacts, rebuild the record."""
    if not result.get("ok") or result.get("conveyor") is None:
        return result
    conv = result["conveyor"]
    n, d, ex, ey = frame_axes(conv)
    labels = np.asarray(result["labels"]).copy()

    planar = result["boxes"]
    for b in planar:
        b.setdefault("origin", "planar")
        b.setdefault("surface", {"class": "planar"})
        b["_label_sel"] = np.flatnonzero(labels == b["id"])

    t0 = time.perf_counter()
    recovered, rej_recover = ([], [])
    reopened = []
    if p.recover:
        free_extra = None
        if p.reopen:
            free_extra = np.zeros(len(labels), bool)
            for b in planar:
                tf = b["top_face"]
                if (tf["coverage"] < p.reopen_coverage
                        or tf["inlier_fraction"] < p.reopen_inlier_fraction):
                    sel = b["_label_sel"]
                    if len(sel):
                        free_extra[sel] = True
                    reopened.append(b["id"])
        recovered, rej_recover = recover_surfaces(result, p, cam_names,
                                                  free_extra)
    t_recover = (time.perf_counter() - t0) * 1e3

    boxes, rej_overlap = resolve_overlaps(planar, recovered, p, ex, ey)
    for i, b in enumerate(boxes):
        b["id"] = i

    t0 = time.perf_counter()
    rej_screen = []
    if p.screen:
        boxes, rej_screen = screen(boxes, result, p, cam_names, cam_centres)
    t_screen = (time.perf_counter() - t0) * 1e3

    for i, b in enumerate(boxes):
        b["id"] = i
    boxes = relations(boxes, p, ex, ey, conv)

    new_labels = np.full(len(labels), -1, np.int32)
    new_labels[labels == -2] = -2
    for b in boxes:
        sel = b.pop("_label_sel", None)
        if sel is not None and len(sel):
            new_labels[sel] = b["id"]
    result["labels"] = new_labels
    result["boxes"] = boxes
    result["rejected"] = (list(result.get("rejected", [])) + rej_recover
                          + rej_overlap + rej_screen)
    result["timings_ms"] = dict(result.get("timings_ms", {}))
    result["timings_ms"]["soft_recover"] = round(t_recover, 2)
    result["timings_ms"]["soft_screen"] = round(t_screen, 2)

    counts = {}
    for b in boxes:
        k = (b.get("surface") or {}).get("class", "planar")
        counts[k] = counts.get(k, 0) + 1
    result["surface_counts"] = counts
    result["soft_params"] = asdict(p.soft) if isinstance(p, Params) else {}

    warn = list(result.get("warnings", []))
    if p.screen and rej_screen:
        warn.append(
            f"{len(rej_screen)} detection(s) were rejected as artefacts rather "
            f"than objects. Each carries the test it failed and by how much; "
            f"read them before loosening --soft-max-see-through or "
            f"--soft-max-bleed, because every one of them is parcel-sized and "
            f"would otherwise be picked at")
    if recovered:
        warn.append(
            f"{len(recovered)} surface(s) were recovered from points the "
            f"planar pass left unassigned. Their dimensions come from the "
            f"height envelope on a {p.film_cell * 1e3:.0f} mm raster, not from "
            f"a plane's inliers, so they are plan extents and under-report a "
            f"tilted top by cos(tilt)")
    if reopened:
        warn.append(
            f"{len(reopened)} planar detection(s) were re-opened before the "
            f"recovery because their own rectangle was mostly empty, which is "
            f"what a plane fitted to part of a wrapped top looks like. Any that "
            f"the envelope did not improve on are still reported as the planar "
            f"pass measured them")
    result["warnings"] = warn
    return result


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

EXTRA_FIELDS = [
    "origin", "surface_class", "relief_mm", "fill_ratio", "envelope_cells",
    "periodic_peak", "pitch_mm", "unit_count",
    "pick_x_m", "pick_y_m", "pick_z_m", "pick_flatness_mm",
    "flat_area_fraction",
    "see_through_frac", "bleed_frac", "height_sigma", "noise_used_mm",
    "views_blocked_geom",
]


def extra_row(b):
    s = b.get("surface") or {}
    scr = b.get("screening") or {}
    pick = s.get("pick_point_m") or [None, None, None]
    units = (s.get("units") or {}).get("count")
    return {
        "origin": b.get("origin", "planar"),
        "surface_class": s.get("class", "planar"),
        "relief_mm": bs._num(s.get("relief_mm")),
        "fill_ratio": bs._num(s.get("fill_ratio")),
        "envelope_cells": bs._num(s.get("cells")),
        "periodic_peak": bs._num(s.get("periodic_peak")),
        "pitch_mm": bs._num(s.get("pitch_mm")),
        "unit_count": bs._num(units),
        "pick_x_m": "" if pick[0] is None else round(float(pick[0]), 5),
        "pick_y_m": "" if pick[1] is None else round(float(pick[1]), 5),
        "pick_z_m": "" if pick[2] is None else round(float(pick[2]), 5),
        "pick_flatness_mm": bs._num(s.get("pick_flatness_mm")),
        "flat_area_fraction": bs._num(s.get("flat_area_fraction")),
        "see_through_frac": bs._num(scr.get("see_through_fraction")),
        "bleed_frac": bs._num(scr.get("bleed_fraction")),
        "height_sigma": bs._num(scr.get("height_sigma")),
        "noise_used_mm": bs._num(scr.get("noise_used_mm")),
        "views_blocked_geom": " ".join(scr.get("views_blocked", [])),
    }


def write_extended_csv(result, path, capture_index, timestamp):
    rows = bs.box_rows(result, capture_index, timestamp)
    by_id = {b["id"]: b for b in result.get("boxes", [])}
    for r in rows:
        r.update(extra_row(by_id[r["id"]]))
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=bs.BOX_CSV_FIELDS + EXTRA_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def summarise(result, p):
    lines = [bs.summarise(result)]
    counts = result.get("surface_counts") or {}
    if counts:
        lines.append("  surfaces  " + "  ".join(f"{k} {v}"
                                                for k, v in sorted(counts.items())))
    for b in sorted(result.get("boxes", []), key=lambda x: x["pick_order"]):
        s = b.get("surface") or {}
        scr = b.get("screening") or {}
        if s.get("class", "planar") == "planar" and not scr:
            continue
        lines.append(
            f"    #{b['id']:<2d} {s.get('class', 'planar'):<8s} "
            f"relief {s.get('relief_mm', 0):5.1f} mm  "
            f"fill {s.get('fill_ratio', 1.0):4.2f}  "
            f"flat seat {('%5.1f mm' % s['pick_flatness_mm']) if s.get('pick_flatness_mm') is not None else '   none'}"
            f" over {s.get('flat_area_fraction', 0) * 100:4.0f} per cent  "
            f"see-through {scr.get('see_through_fraction', 0):4.2f}  "
            f"bleed {scr.get('bleed_fraction', 0):4.2f}  "
            f"h/noise {scr.get('height_sigma', 0):5.1f}"
            + (f"  units {s['units']['count']}" if s.get("units") else ""))
    dropped = [r for r in result.get("rejected", [])
               if r.get("screening") or str(r.get("origin", "")) == "recovery"]
    for r in dropped[:20]:
        lines.append(f"    rejected {r.get('cluster')}: {r.get('reason')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Segment a capture containing wrapped parcels or "
                    "palletised units, and reject the artefacts such scenes "
                    "produce.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, required=True,
                    help="a capture_NNNNN directory holding depth_<cam>.npy, "
                         "K_<cam>.npy, E_<cam>.npy and proc_<cam>.png")
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to a 'segmentation' folder inside the "
                         "capture directory")
    ap.add_argument("--probe", action="store_true",
                    help="list the largest planes and the extent of the cloud, "
                         "then exit")
    ap.add_argument("--probe-planes", type=int, default=6)
    bs.add_arguments(ap)
    add_soft_arguments(ap)
    args = ap.parse_args()

    try:
        from da3_fuse import points_camera, to_world
    except ImportError as exc:
        raise SystemExit(f"cannot import da3_fuse.py: {exc}")

    seg = bs.params_from_args(args)
    soft = soft_params_from_args(args)
    p = Params(seg, soft)

    cap = args.capture_dir
    names = [n for n in args.cameras if (cap / f"depth_{n}.npy").exists()]
    if not names:
        raise SystemExit(f"no depth arrays found in {cap}")

    pts_list, src_list, images, Ks, Es = [], [], [], [], []
    for i, n in enumerate(names):
        depth = np.load(cap / f"depth_{n}.npy").astype(np.float64)
        K = np.load(cap / f"K_{n}.npy")
        E = np.load(cap / f"E_{n}.npy")
        valid = np.isfinite(depth) & (depth > 0)
        world = to_world(points_camera(depth, K)[valid], E)
        pts_list.append(world)
        src_list.append(np.full(len(world), i, dtype=np.int32))
        Ks.append(K)
        Es.append(E)
        proc = cap / f"proc_{n}.png"
        images.append(cv2.cvtColor(cv2.imread(str(proc)), cv2.COLOR_BGR2RGB)
                      if proc.exists()
                      else np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8))

    P = np.concatenate(pts_list)
    S = np.concatenate(src_list)
    if args.probe:
        return bs.probe(P, seg, args.probe_planes)

    centres = [bs.camera_centre(E) for E in Es]
    t0 = time.perf_counter()
    result = bs.segment_boxes(P, S, names, seg, cam_centres=centres)
    n_planar = len(result.get("boxes", []))

    if soft.enabled:
        result = refine(result, p, names, centres)

    out_dir = args.out_dir or (cap / "segmentation")
    written, _ = bs.write_results(
        result, out_dir, capture_index=cap.name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        images=images, K_all=Ks, E_all=Es, p=seg,
        extra_meta={"soft_segment": asdict(soft),
                    "planar_detections": n_planar})
    rows = write_extended_csv(result, out_dir / "boxes.csv", cap.name,
                              time.strftime("%Y-%m-%dT%H:%M:%S"))
    written["boxes_csv"] = out_dir / "boxes.csv"

    print(summarise(result, p))
    for w in result.get("warnings", []):
        print(f"[warn ] {w}")
    print(f"planar pass {n_planar} detections, after recovery and screening "
          f"{len(result.get('boxes', []))}")
    print(f"timings ms: {result.get('timings_ms')}  "
          f"wall {(time.perf_counter() - t0) * 1e3:.0f} ms")
    for k, v in written.items():
        print(f"  {k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())