#!/usr/bin/env python3
"""
box_segment.py

Segmentation of parcel top faces in a fused metric point cloud, and extraction
of the data a pick requires: top-face centre, surface normal, in-plane
orientation, footprint dimensions, height above the conveyor and a per-box
quality record.

The approach is geometric rather than learned. The fused cloud is already
metric and already in a single frame, so a plane fit plus density clustering
recovers the parcels directly and returns a pose in metres without a second
network, a second set of weights and a second failure mode.

Pipeline
--------
1.  optional workspace crop, then voxel subsampling to a uniform density
2.  RANSAC plane fit over the whole cloud, which finds the conveyor because it
    is the largest coplanar surface in view
3.  the plane normal is oriented towards the cameras, so heights above the
    conveyor come out positive
4.  points between --seg-min-height and --seg-max-height above the plane are
    clustered with DBSCAN, one cluster per parcel. Clustering is ANISOTROPIC
5.  the top face of each cluster is fitted in EVERY contributing camera
    separately, the views are reconciled against each other, and only then is
    the combined face fitted. See face_consensus.py
6.  the inliers are projected into that plane and a minimum-area rectangle
    gives the footprint and the in-plane axes
7.  a right-handed pose is built with Z along the approach direction

WHAT THIS VERSION CHANGES, AND WHY IT MATTERS FOR ACCURACY
==========================================================

1  A FAILED RECONCILIATION PRODUCED NO FLAG AT ALL.

   The relations loop read n_views = len(vc.get("views_used", [])) and gated on
   0 < n_views < min_views_pickable. When face_consensus returned a FAILURE
   record it had no views_used key, so n_views came back 0, the guard was
   False, and the parcel was reported PICKABLE on a raw multi-view plane fit.

   That is precisely the blind spot the per-view reconciliation exists to
   close: fit_top_face takes a band of --seg-top-band below the height
   percentile, and an inter-view offset larger than that band puts the second
   view outside it entirely, so the face is fitted to one camera and the record
   claims a clean single-plane fit. --seg-face-grow-scale was also inert on
   that path, because the spread it scales from fell back to zero.

   face_consensus now returns a uniform key set on every path, and this file
   branches explicitly on vc["ok"] is False and blocks.

2  RANSAC WAS UNSEEDED, SO REPEATABILITY MEASURED THE SAMPLER.

   --seg-seed existed and was never used; o3d.utility.random.seed was never
   called. plane_ransac runs for the conveyor, for every top face, inside the
   per-view fits and inside probe. Across a static repeatability sweep that
   mixes RANSAC sampling variance into every reported pose, and it is not
   separable afterwards. The seed is now applied at the start of
   segment_boxes and before each fit.

3  COVERAGE WAS COMPUTED ON CLIPPED COORDINATES.

   rect_on_plane clipped the occupancy grid indices to the trimmed rectangle,
   so every point outside it folded into an edge cell and counted as occupied.
   With --seg-dim-percentile 1.0 that is one per cent of points per edge
   landing in the boundary row, inflating coverage on exactly the ragged faces
   where the metric decides pickability. Points outside the range are now
   masked out rather than clipped in.

4  THE SINGLE-VIEW UNCERTAINTY WAS QUOTED AT THE DECK.

   A face carried by one camera was given an uncertainty equal to that
   camera's bias at the DECK. On this rig the cameras agree at the deck to
   about 2 mm and disagree on a face 260 mm above it by tens of millimetres,
   and the depth field has been measured to compress relief by about a
   quarter, so deck bias does not predict face bias. After the boxes are
   built, face_consensus.measure_bias_ratio derives the empirical face-to-deck
   ratio from this capture's own multi-view faces and rescales every
   single-view uncertainty by it. Where fewer than three multi-view faces
   exist the ratio is refused and the figure stays labelled a lower bound.

5  --seg-min-plane-fraction WAS UNREACHABLE. It sat in SegParams, drove a
   warning, and had no CLI flag and no line in params_from_args, so it was
   permanently 0.15 whatever the caller asked for.

Rig-specific behaviour retained
-------------------------------
  CLUSTERING is anisotropic (--seg-cluster-height-scale). DBSCAN in world
  metres cannot put two views of one face 44 mm apart inside a 20 mm radius,
  so they cluster separately and the reconciliation never sees them together.
  Compressing the height axis costs a depth disagreement a fraction of a
  radius while leaving in-plane distances untouched.

  GROUPING uses intersection over the SMALLER footprint, not IoU. A camera
  seeing half a face scores exactly 0.50 IoU against the camera seeing all of
  it, so partial views were filed as different surfaces and dropped.

  FACE GROWTH is capped at --seg-top-band when only one level is extracted,
  not at --seg-level-gap, which was cutting the tolerance below what the views
  themselves said was needed.

Frames
------
Everything is in the world frame used by the fusion, which is the reference
camera's optical frame. World Z is therefore the reference camera's optical
axis, pointing down at the conveyor, and is not "up". Orientation is reported
twice: as a full 4x4 pose in the world frame, and as a scalar yaw in a
conveyor frame built from the fitted plane.

Metric caveat
-------------
Dimensions and heights inherit whatever scale error the depth carries. Height
above the conveyor is a difference of two depths and so is insensitive to a
common additive offset, but it is fully sensitive to a scale error, and DA3's
relief has been measured on this rig to be compressed by about 26 per cent
over 257 mm while the absolute standoff is correct to under 2 per cent. Treat
the dimensions as a measurement of the corrected cloud, not of the parcels,
until they have been checked against a box of known size.

Note that the consensus shift is applied to the fitted geometry only. The
written point cloud keeps the raw positions, so segmented.ply still shows the
layering that boxes.json quantifies.

Standalone use
--------------
    python box_segment.py --capture-dir runs/live_.../capture_00000

Keep this file beside da3_fuse.py, da3_stream.py and face_consensus.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from statistics import median as statistics_median
from dataclasses import dataclass, field, asdict
from pathlib import Path

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
    from face_consensus import (consensus_shift, apply_consensus,
                                deck_offsets_per_view, measure_bias_ratio)
except ImportError as exc:  # noqa: BLE001
    consensus_shift = None
    apply_consensus = None
    deck_offsets_per_view = None
    measure_bias_ratio = None
    _CONSENSUS_IMPORT_ERROR = exc


BOX_COLOURS = [
    (0, 200, 255), (0, 255, 120), (255, 120, 0), (255, 0, 200),
    (60, 220, 220), (200, 120, 255), (0, 140, 255), (140, 255, 60),
    (255, 200, 0), (120, 120, 255), (0, 255, 255), (180, 60, 255),
]

CONVEYOR_RGB = (0.35, 0.35, 0.35)
UNASSIGNED_RGB = (0.12, 0.12, 0.12)


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class SegParams:
    """Every threshold the segmentation uses, in metres and degrees."""

    voxel: float = 0.006
    roi: tuple | None = None

    plane_thresh: float = 0.008
    plane_iter: int = 2000
    min_plane_fraction: float = 0.15
    plane_distance: tuple | None = None
    max_plane_attempts: int = 5
    max_boxes_warn: int = 12

    min_height: float = 0.030
    max_height: float = 1.000

    belt_only: bool = True
    belt_margin: float = 0.050
    belt_eps: float = 0.050
    belt_voxel: float = 0.015
    belt_close: float = 0.300
    belt_raster_m: float = 0.010
    belt_include_objects: bool = True
    belt_file: str | None = None
    belt_refit: bool = False
    belt_edge_warn: float = 0.020

    cluster_eps: float = 0.020
    cluster_min_points: int = 20
    cluster_height_scale: float = 0.25
    min_points: int = 300

    top_percentile: float = 97.0
    top_band: float = 0.030
    face_thresh: float = 0.006
    min_face_points: int = 150
    max_tilt_deg: float = 25.0
    max_levels: int = 1
    level_min_gap: float = 0.020
    face_grow_scale: float = 0.5
    min_views_pickable: int = 2

    # per-view reconciliation, implemented in face_consensus.py
    face_per_view: bool = True
    min_view_face_points: int = 60
    max_face_incidence: float = 65.0
    max_view_shift: float = 0.060
    view_agree_overlap: float = 0.65
    min_view_area_frac: float = 0.10
    comb_warn: float = 0.35
    view_disagree_warn: float = 0.010
    consensus_estimator: str = "mean"
    consensus_max_iters: int = 4
    bias_ratio: bool = True

    min_dim: float = 0.050
    max_dim: float = 1.200
    max_aspect: float = 4.0
    min_area: float = 0.010
    max_overlap: float = 0.50
    dim_percentile: float = 1.0
    min_coverage: float = 0.55
    max_residual: float = 0.012
    single_view_frac: float = 0.80
    min_view_points: int = 40

    approach_offset: float = 0.100
    clearance_warn: float = 0.015

    overlay_scale: float = 2.0
    draw_points: bool = False
    pick_radius: float = 0.040
    desaturate: bool = True
    label_detail: str = "short"
    views: str = "bev"
    bev_mm_per_px: float = 2.0
    bev_max_px: int = 2000
    seed: int = 0

    metadata: dict = field(default_factory=dict)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Attach the segmentation options to an existing parser."""
    g = ap.add_argument_group("segmentation")
    g.add_argument("--segment", action="store_true",
                   help="on capture, segment the fused cloud, write the parcel "
                        "records and draw the annotated views")
    g.add_argument("--seg-voxel", type=float, default=0.006,
                   help="subsampling before segmentation; sets the effective "
                        "point spacing that DBSCAN sees")
    g.add_argument("--seg-roi", nargs=6, type=float, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="workspace crop in world metres, applied first")

    g.add_argument("--seg-plane-thresh", type=float, default=0.008,
                   help="inlier distance for the conveyor plane fit; set it "
                        "above the per-view flatness, not below")
    g.add_argument("--seg-plane-iter", type=int, default=2000)
    g.add_argument("--seg-min-plane-fraction", type=float, default=0.15,
                   help="fraction of the cloud the fitted conveyor plane must "
                        "hold before it is believed to be the conveyor. This "
                        "was present in the parameter block and unreachable "
                        "from the command line, so it sat at its default "
                        "whatever the caller asked for")
    g.add_argument("--seg-plane-distance", nargs=2, type=float, default=None,
                   metavar=("MIN_M", "MAX_M"),
                   help="accepted distance of the conveyor plane from the "
                        "reference camera. WITHOUT THIS the fit returns the "
                        "largest coplanar surface in the cloud, which in a "
                        "four-camera rig is often the FLOOR: the deck then "
                        "reads as a 0.8 m tall object, fragments into dozens "
                        "of false parcels, and the real ones are cut by "
                        "--seg-max-height. The band is in the MODEL's "
                        "coordinates, not physical metres, because the depth "
                        "is unanchored: read it off --probe rather than off a "
                        "tape measure or a ChArUco standoff")
    g.add_argument("--seg-max-plane-attempts", type=int, default=5,
                   help="planes discarded on distance before giving up")
    g.add_argument("--seg-max-boxes-warn", type=int, default=12,
                   help="parcel count above which a warning is raised that the "
                        "plane is probably wrong")
    g.add_argument("--seg-min-height", type=float, default=0.030,
                   help="a point must stand this far above the conveyor to be "
                        "considered part of a parcel. 30 mm rather than 20 "
                        "because the deck is a single plane through roller "
                        "crowns that are not coplanar. Note this is a floor on "
                        "what can be DETECTED: where the cameras disagree by "
                        "tens of millimetres, a short parcel falls below it in "
                        "the biased view and above it in the others")
    g.add_argument("--seg-max-height", type=float, default=1.000)
    g.add_argument("--seg-no-belt", dest="seg_belt_only", action="store_false",
                   default=True,
                   help="search the whole cloud rather than the conveyor. By "
                        "default the belt is found as the largest connected "
                        "patch of the deck plane and anything standing outside "
                        "its footprint is ignored")
    g.add_argument("--seg-belt-margin", type=float, default=0.050,
                   help="the belt footprint is grown by this much before it is "
                        "used as the search bound, so a parcel overhanging the "
                        "edge is still found")
    g.add_argument("--seg-belt-eps", type=float, default=0.050)
    g.add_argument("--seg-belt-voxel", type=float, default=0.015)
    g.add_argument("--seg-belt-close", type=float, default=0.300,
                   help="gap bridged when working out the belt footprint. The "
                        "deck is only visible between the parcels standing on "
                        "it, so a connectivity test alone returns one strip "
                        "rather than the belt and the footprint SHRINKS as the "
                        "belt fills. Must exceed the widest parcel and stay "
                        "below the distance to any other surface at deck "
                        "height")
    g.add_argument("--seg-belt-raster-m", type=float, default=0.010)
    g.add_argument("--seg-belt-no-objects", dest="seg_belt_include_objects",
                   action="store_false", default=True,
                   help="work the footprint out from bare deck only. Anything "
                        "standing on the belt is on the belt by definition, so "
                        "including it is what lets the footprint survive a "
                        "fully loaded conveyor")
    g.add_argument("--seg-belt-file", type=str, default=None,
                   help="a belt.json holding the footprint. The conveyor does "
                        "not move between captures, so re-solving it every "
                        "time only adds noise. Written on first use, read "
                        "thereafter")
    g.add_argument("--seg-belt-refit", action="store_true",
                   help="re-solve the footprint and overwrite --seg-belt-file")
    g.add_argument("--seg-belt-edge-warn", type=float, default=0.020,
                   help="a parcel reaching this close to the search bound is "
                        "flagged, because its footprint may have been cut by "
                        "the bound rather than measured")

    g.add_argument("--seg-cluster-eps", type=float, default=0.020,
                   help="DBSCAN neighbourhood radius; below the gap between "
                        "adjacent parcels and above the point spacing. It "
                        "applies IN THE CONVEYOR PLANE: the height axis is "
                        "scaled by --seg-cluster-height-scale first")
    g.add_argument("--seg-cluster-height-scale", type=float, default=0.25,
                   help="compression applied to the height axis before "
                        "clustering. Two views of one parcel top that disagree "
                        "by 44 mm cannot fall inside a 20 mm radius in world "
                        "metres, so they cluster separately and the per-view "
                        "reconciliation never sees them together, which "
                        "reports one parcel as several each carried by one "
                        "camera. 1.0 restores isotropic clustering")
    g.add_argument("--seg-cluster-min-points", type=int, default=20,
                   help="neighbours a point needs to be a DBSCAN core point. A "
                        "parcel top is a SHEET, not a volume, so the count "
                        "available within one radius is only about "
                        "pi*(eps/spacing)^2; the value is clamped at runtime")
    g.add_argument("--seg-min-points", type=int, default=300,
                   help="smallest cluster accepted as a parcel")

    g.add_argument("--seg-top-percentile", type=float, default=97.0,
                   help="robust stand-in for the cluster's maximum height")
    g.add_argument("--seg-top-band", type=float, default=0.030,
                   help="depth of the band below that height taken as the "
                        "candidate top face")
    g.add_argument("--seg-face-thresh", type=float, default=0.006,
                   help="inlier distance for the top-face plane fit")
    g.add_argument("--seg-min-face-points", type=int, default=150)
    g.add_argument("--seg-max-tilt", type=float, default=25.0,
                   help="largest angle between a top face and the conveyor "
                        "before the fit is rejected as a side face")
    g.add_argument("--seg-max-levels", type=int, default=1,
                   help="horizontal surfaces extracted per cluster. Raise it "
                        "to 2 or 3 only for stacked parcels")
    g.add_argument("--seg-level-gap", type=float, default=0.020,
                   help="a further level must sit at least this far below the "
                        "one already taken")

    g.add_argument("--seg-face-grow-scale", type=float, default=0.5,
                   help="fraction of the measured inter-view disagreement "
                        "added to --seg-face-thresh when deciding which points "
                        "belong to a top face. Where the views disagree by "
                        "tens of millimetres the rigid consensus shift leaves "
                        "a residual ramp of several millimetres and a fixed "
                        "6 mm band admits only the coherent part of the face. "
                        "0 restores the fixed band")
    g.add_argument("--seg-min-views", type=int, default=2,
                   help="cameras that must each produce a usable top face "
                        "before a parcel is called pickable")
    g.add_argument("--seg-no-view-consensus", dest="seg_face_per_view",
                   action="store_false", default=True,
                   help="fit one plane across all cameras at once, as before")
    g.add_argument("--seg-min-view-face-points", type=int, default=60)
    g.add_argument("--seg-max-face-incidence", type=float, default=65.0,
                   help="angle between a top face and a camera's view ray "
                        "beyond which that camera is dropped from the fit, "
                        "provided another view covers the face")
    g.add_argument("--seg-max-view-shift", type=float, default=0.060,
                   help="largest displacement attributed to per-view depth "
                        "bias. Must exceed the inter-camera disagreement "
                        "actually measured on parcel tops")
    g.add_argument("--seg-view-agree-overlap", type=float, default=0.65,
                   help="plan-view overlap above which two cameras are judged "
                        "to be seeing the same surface, measured as "
                        "intersection over the SMALLER of the two footprints")
    g.add_argument("--seg-min-view-area-frac", type=float, default=0.10,
                   help="a view whose footprint is smaller than this fraction "
                        "of the dominant one's is not admitted to the group "
                        "even when fully contained")
    g.add_argument("--seg-consensus-estimator", choices=["mean", "median"],
                   default="mean",
                   help="how the per-view offsets are combined. MEAN uses "
                        "every retained view; MEDIAN over two or three views "
                        "returns one view's depth exactly and discards the "
                        "others, which is selection rather than robustness. "
                        "Outliers are excluded before the estimate either way")
    g.add_argument("--seg-consensus-iters", type=int, default=4,
                   help="passes of exclude-and-recompute in the consensus")
    g.add_argument("--seg-no-bias-ratio", dest="seg_bias_ratio",
                   action="store_false", default=True,
                   help="leave the single-view height uncertainty at the value "
                        "measured at the DECK. By default the ratio between "
                        "face bias and deck bias is measured over this "
                        "capture's own multi-view faces and used to rescale "
                        "it, because deck bias understates face bias by close "
                        "to an order of magnitude on this rig")
    g.add_argument("--seg-comb-warn", type=float, default=0.35)
    g.add_argument("--seg-view-disagree-warn", type=float, default=0.010,
                   help="inter-view depth disagreement on one top face above "
                        "which a caution is raised, in metres")

    g.add_argument("--seg-dim-percentile", type=float, default=1.0,
                   help="percentile trimmed from each end when measuring the "
                        "footprint; 0 reports the raw minimum-area rectangle, "
                        "which noise inflates by roughly three standard "
                        "deviations per edge")
    g.add_argument("--seg-min-dim", type=float, default=0.050)
    g.add_argument("--seg-max-dim", type=float, default=1.200)
    g.add_argument("--seg-max-aspect", type=float, default=4.0,
                   help="largest length to width ratio accepted")
    g.add_argument("--seg-min-area", type=float, default=0.010,
                   help="smallest top face accepted, in square metres")
    g.add_argument("--seg-max-overlap", type=float, default=0.50,
                   help="two detections overlapping in plan by more than this "
                        "are the same object seen twice")
    g.add_argument("--seg-min-coverage", type=float, default=0.55,
                   help="fraction of the fitted rectangle that must actually "
                        "carry points")
    g.add_argument("--seg-max-residual", type=float, default=0.012,
                   help="largest top-face plane RMS residual still called "
                        "pickable")
    g.add_argument("--seg-single-view-frac", type=float, default=0.80,
                   help="above this fraction from one camera the parcel is "
                        "flagged as single-view")
    g.add_argument("--seg-min-view-points", type=int, default=40)

    g.add_argument("--seg-approach-offset", type=float, default=0.100)
    g.add_argument("--seg-clearance-warn", type=float, default=0.015)

    g.add_argument("--seg-overlay-scale", type=float, default=2.0)
    g.add_argument("--seg-draw-points", dest="seg_draw_points",
                   action="store_true", default=False)
    g.add_argument("--seg-pick-radius", type=float, default=0.040,
                   help="radius of the circle drawn at each pick point. Set it "
                        "to your suction cup so the overlay shows whether the "
                        "cup actually lands on the parcel")
    g.add_argument("--seg-no-desaturate", dest="seg_desaturate",
                   action="store_false", default=True)
    g.add_argument("--seg-label-detail", choices=["none", "short", "full"],
                   default="short")
    g.add_argument("--seg-views", choices=["bev", "cameras", "all"],
                   default="bev")
    g.add_argument("--seg-bev-mm-per-px", type=float, default=2.0)
    g.add_argument("--seg-bev-max-px", type=int, default=2000)
    g.add_argument("--seg-seed", type=int, default=0,
                   help="seed for every RANSAC in this file. It was declared "
                        "and never applied, so a static repeatability sweep "
                        "mixed the sampler's variance into every reported pose "
                        "and the two could not be separated afterwards")


def params_from_args(args: argparse.Namespace) -> SegParams:
    return SegParams(
        voxel=args.seg_voxel,
        roi=tuple(args.seg_roi) if args.seg_roi else None,
        plane_thresh=args.seg_plane_thresh,
        plane_iter=args.seg_plane_iter,
        min_plane_fraction=args.seg_min_plane_fraction,
        plane_distance=(tuple(args.seg_plane_distance)
                        if args.seg_plane_distance else None),
        max_plane_attempts=args.seg_max_plane_attempts,
        max_boxes_warn=args.seg_max_boxes_warn,
        min_height=args.seg_min_height,
        max_height=args.seg_max_height,
        belt_only=args.seg_belt_only,
        belt_margin=args.seg_belt_margin,
        belt_eps=args.seg_belt_eps,
        belt_voxel=args.seg_belt_voxel,
        belt_close=args.seg_belt_close,
        belt_raster_m=args.seg_belt_raster_m,
        belt_include_objects=args.seg_belt_include_objects,
        belt_file=args.seg_belt_file,
        belt_refit=args.seg_belt_refit,
        belt_edge_warn=args.seg_belt_edge_warn,
        cluster_eps=args.seg_cluster_eps,
        cluster_min_points=args.seg_cluster_min_points,
        cluster_height_scale=args.seg_cluster_height_scale,
        min_points=args.seg_min_points,
        top_percentile=args.seg_top_percentile,
        top_band=args.seg_top_band,
        face_thresh=args.seg_face_thresh,
        min_face_points=args.seg_min_face_points,
        max_tilt_deg=args.seg_max_tilt,
        max_levels=args.seg_max_levels,
        level_min_gap=args.seg_level_gap,
        face_grow_scale=args.seg_face_grow_scale,
        min_views_pickable=args.seg_min_views,
        face_per_view=args.seg_face_per_view,
        min_view_face_points=args.seg_min_view_face_points,
        max_face_incidence=args.seg_max_face_incidence,
        max_view_shift=args.seg_max_view_shift,
        view_agree_overlap=args.seg_view_agree_overlap,
        min_view_area_frac=args.seg_min_view_area_frac,
        consensus_estimator=args.seg_consensus_estimator,
        consensus_max_iters=args.seg_consensus_iters,
        bias_ratio=args.seg_bias_ratio,
        comb_warn=args.seg_comb_warn,
        view_disagree_warn=args.seg_view_disagree_warn,
        min_dim=args.seg_min_dim,
        dim_percentile=args.seg_dim_percentile,
        max_dim=args.seg_max_dim,
        max_aspect=args.seg_max_aspect,
        min_area=args.seg_min_area,
        max_overlap=args.seg_max_overlap,
        min_coverage=args.seg_min_coverage,
        max_residual=args.seg_max_residual,
        single_view_frac=args.seg_single_view_frac,
        min_view_points=args.seg_min_view_points,
        approach_offset=args.seg_approach_offset,
        clearance_warn=args.seg_clearance_warn,
        overlay_scale=args.seg_overlay_scale,
        draw_points=args.seg_draw_points,
        pick_radius=args.seg_pick_radius,
        desaturate=args.seg_desaturate,
        label_detail=args.seg_label_detail,
        views=args.seg_views,
        bev_mm_per_px=args.seg_bev_mm_per_px,
        bev_max_px=args.seg_bev_max_px,
        seed=args.seg_seed,
    )


# --------------------------------------------------------------------------
# small geometric helpers
# --------------------------------------------------------------------------

def voxel_indices(points: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one representative point per occupied voxel."""
    if voxel is None or voxel <= 0 or len(points) == 0:
        return np.arange(len(points), dtype=np.int64)
    q = np.floor(points / voxel).astype(np.int64)
    q -= q.min(axis=0)
    dims = [int(v) + 1 for v in q.max(axis=0)]
    if dims[0] * dims[1] * dims[2] > 2 ** 62:
        raise ValueError("voxel grid too large; crop with --seg-roi first")
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def seed_ransac(seed: int) -> None:
    """Pin Open3D's RANSAC sampler.

    Called before every segment_plane in this file. Without it a static
    repeatability sweep measures the sampler as much as the perception chain,
    and the two cannot be separated after the fact.
    """
    try:
        o3d.utility.random.seed(int(seed))
    except Exception:  # noqa: BLE001 - older Open3D builds lack the hook
        pass


def plane_ransac(points: np.ndarray, thresh: float, iters: int, seed: int = 0):
    """Fit a plane, returning the unit normal, the offset and the inliers.

    The plane is n . x + d = 0, so the signed distance of a point is
    n . x + d and is positive on the side the normal points to.
    """
    if len(points) < 3:
        return None, None, np.empty(0, dtype=np.int64)
    seed_ransac(seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    model, inliers = pcd.segment_plane(distance_threshold=thresh,
                                       ransac_n=3,
                                       num_iterations=int(iters))
    n = np.asarray(model[:3], dtype=float)
    d = float(model[3])
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return None, None, np.empty(0, dtype=np.int64)
    return n / norm, d / norm, np.asarray(inliers, dtype=np.int64)


def orient_towards(n: np.ndarray, d: float, point: np.ndarray):
    """Flip the plane so the given point has a positive signed distance."""
    if float(n @ np.asarray(point, dtype=float) + d) < 0:
        return -n, -d
    return n, d


def basis_from_normal(n: np.ndarray, hint: np.ndarray):
    """An orthonormal in-plane basis, with the first axis nearest the hint."""
    n = n / np.linalg.norm(n)
    h = np.asarray(hint, dtype=float)
    ex = h - n * float(h @ n)
    if np.linalg.norm(ex) < 1e-6:
        h = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ex = h - n * float(h @ n)
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(n, ex)
    return ex, ey


def rot_to_quat(R: np.ndarray):
    """Rotation matrix to quaternion, ordered w, x, y, z."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def rot_to_rpy_deg(R: np.ndarray):
    """ZYX Euler angles in degrees, which is the FANUC W, P, R convention."""
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    if abs(R[2, 0]) < 0.999999:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return np.asarray(K, dtype=float)
    Ks = np.array(K, dtype=float, copy=True)
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


def project(points: np.ndarray, K: np.ndarray, E: np.ndarray):
    """World points to pixels using the w2c extrinsics DA3 returned."""
    E = np.asarray(E, dtype=float)
    R, t = E[:3, :3], E[:3, 3]
    cam = np.asarray(points, dtype=float).reshape(-1, 3) @ R.T + t
    z = cam[:, 2]
    ok = z > 1e-6
    uv = np.full((len(cam), 2), np.nan)
    uv[ok, 0] = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
    return uv, ok


def camera_centre(E: np.ndarray) -> np.ndarray:
    """Optical centre of a w2c extrinsic, in world metres."""
    E = np.asarray(E, dtype=float)
    return -E[:3, :3].T @ E[:3, 3]


def as_uint8_image(img: np.ndarray) -> np.ndarray:
    """Processed images arrive as float in [0, 1] or as uint8; accept either."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.copy()
    a = a.astype(np.float64)
    peak = float(np.nanmax(a)) if a.size else 1.0
    a = a / 255.0 if peak > 1.0 + 1e-6 else a
    return (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)


def jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


# --------------------------------------------------------------------------
# top face fitting
# --------------------------------------------------------------------------

def fit_top_face(cluster_pts, n_conv, d_conv, p: SegParams):
    """Fit the plane of the highest surface of one cluster.

    Call this on points that have already been through the per-view
    reconciliation. On raw multi-view points a disagreement larger than
    top_band silently reduces the fit to whichever view sits highest.
    """
    h = cluster_pts @ n_conv + d_conv
    h_top = float(np.percentile(h, p.top_percentile))
    band = cluster_pts[h >= h_top - p.top_band]
    if len(band) < p.min_face_points:
        return None, f"top band holds {len(band)} points, below {p.min_face_points}"

    n_top, d_top, inl = plane_ransac(band, p.face_thresh, p.plane_iter, p.seed)
    if n_top is None or len(inl) < p.min_face_points:
        return None, "top-face plane fit failed"

    if float(n_top @ n_conv) < 0:
        n_top, d_top = -n_top, -d_top

    tilt = float(np.degrees(np.arccos(np.clip(float(n_top @ n_conv), -1.0, 1.0))))
    if tilt > p.max_tilt_deg:
        return None, f"top face tilted {tilt:.1f} deg from the conveyor"

    face = band[inl]
    resid = face @ n_top + d_top
    return {
        "points": face,
        "normal": n_top,
        "offset": d_top,
        "tilt_deg": tilt,
        "band_points": int(len(band)),
        "inliers": int(len(inl)),
        "inlier_fraction": float(len(inl) / len(band)),
        "residual_rms_m": float(np.sqrt(float(np.mean(resid ** 2)))),
        "residual_p95_m": float(np.percentile(np.abs(resid), 95)),
    }, None


def face_growth_tolerance(spread_m, p: SegParams) -> float:
    """How far from the fitted plane a point may sit and still belong to it.

    The rigid per-view shift removes the constant part of an inter-view
    disagreement and leaves the part that varies across the face, so the band
    has to accommodate that remainder or the fit keeps only the coherent half
    of the parcel top.

    The cap is --seg-top-band. It used to be --seg-level-gap unconditionally,
    which is the right bound when several levels are being peeled out of one
    cluster. With --seg-max-levels 1 there is no next level, and a 20 mm cap
    against a 44 mm disagreement cut the tolerance below what the views
    themselves said was needed.
    """
    if p.face_grow_scale <= 0:
        return p.face_thresh
    want = max(p.face_thresh, p.face_grow_scale * float(spread_m))
    cap = p.top_band if p.max_levels <= 1 else min(p.level_min_gap, p.top_band)
    return min(want, cap)


def grow_top_face(cluster_pts, face, n_conv, cx_axis, tol, p: SegParams):
    """Widen the top face from a fixed inlier band to a tolerance the views
    themselves imply, then refit.

    Growth is by plan-view connectivity from the original inliers rather than
    by distance alone, so it crosses a ramp but stops at a genuine step or gap.
    """
    n_top = np.asarray(face["normal"], dtype=float)
    d_top = float(face["offset"])
    sd = cluster_pts @ n_top + d_top
    within = np.abs(sd) <= tol
    seed = np.abs(sd) <= p.face_thresh
    if int(within.sum()) <= int(seed.sum()) or not seed.any():
        return face, None

    ex, ey = basis_from_normal(n_top, cx_axis)
    cell = max(p.voxel * 2.0, 0.008)
    u, v = cluster_pts @ ex, cluster_pts @ ey
    iu = np.floor((u - u.min()) / cell).astype(np.int32)
    iv = np.floor((v - v.min()) / cell).astype(np.int32)
    W, H = int(iu.max()) + 1, int(iv.max()) + 1
    if W * H > 4_000_000 or W < 2 or H < 2:
        return face, None

    occ = np.zeros((H, W), np.uint8)
    occ[iv[within], iu[within]] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, lab = cv2.connectedComponents(occ)
    seed_labels = np.unique(lab[iv[seed], iu[seed]])
    seed_labels = seed_labels[seed_labels > 0]
    if not len(seed_labels):
        return face, None
    keep = within & np.isin(lab[iv, iu], seed_labels)
    if int(keep.sum()) < p.min_face_points:
        return face, None

    n_new, d_new = n_top, d_top
    for _ in range(2):
        pts = cluster_pts[keep]
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        n_new = vt[-1]
        if float(n_new @ n_conv) < 0:
            n_new = -n_new
        d_new = -float(n_new @ c)
        sd = cluster_pts @ n_new + d_new
        keep = within & np.isin(lab[iv, iu], seed_labels) & (np.abs(sd) <= tol)
        if int(keep.sum()) < p.min_face_points:
            return face, None

    tilt = float(np.degrees(np.arccos(
        np.clip(float(n_new @ n_conv), -1.0, 1.0))))
    if tilt > p.max_tilt_deg:
        return face, None

    grown = cluster_pts[keep]
    resid = grown @ n_new + d_new
    record = {
        "tolerance_mm": round(tol * 1e3, 2),
        "points_before": int(face["inliers"]),
        "points_after": int(len(grown)),
        "gain": round(float(len(grown)) / max(face["inliers"], 1), 3),
    }
    return {
        "points": grown,
        "normal": n_new,
        "offset": d_new,
        "tilt_deg": tilt,
        "band_points": int(face["band_points"]),
        "inliers": int(len(grown)),
        "inlier_fraction": float(len(grown) / max(face["band_points"], 1)),
        "residual_rms_m": float(np.sqrt(float(np.mean(resid ** 2)))),
        "residual_p95_m": float(np.percentile(np.abs(resid), 95)),
    }, record


def rect_on_plane(face_pts, n_top, cx_axis, p: SegParams):
    """Footprint of the face, expressed back in the world frame.

    The minimum-area rectangle sets the orientation only. Its extent is driven
    by the most extreme points, so dimensions are taken from percentiles along
    the fitted axes instead.

    COVERAGE is computed over points that actually fall INSIDE the trimmed
    rectangle. The previous version clipped the grid indices, so every point
    outside folded into an edge cell and counted as occupied: with
    --seg-dim-percentile 1.0 that is one per cent of points per edge landing
    in the boundary row, inflating coverage on exactly the ragged faces where
    the metric decides pickability.
    """
    if len(face_pts) < 3:
        return None, "too few points for a rectangle"
    origin = face_pts.mean(axis=0)
    ex, ey = basis_from_normal(n_top, cx_axis)
    rel = face_pts - origin
    uv = np.stack([rel @ ex, rel @ ey], axis=1)
    uv_mm = (uv * 1000.0).astype(np.float32)

    (_, _), (w_mm, h_mm), angle = cv2.minAreaRect(uv_mm)
    if w_mm < 1.0 or h_mm < 1.0:
        return None, "degenerate rectangle"

    a = np.radians(angle)
    ax = ex * np.cos(a) + ey * np.sin(a)
    ay = -ex * np.sin(a) + ey * np.cos(a)

    u = rel @ ax
    v = rel @ ay
    q = float(p.dim_percentile)
    if q > 0:
        u_lo, u_hi = np.percentile(u, [q, 100.0 - q])
        v_lo, v_hi = np.percentile(v, [q, 100.0 - q])
    else:
        u_lo, u_hi = float(u.min()), float(u.max())
        v_lo, v_hi = float(v.min()), float(v.max())

    du, dv = float(u_hi - u_lo), float(v_hi - v_lo)
    if du < dv:                       # keep the long edge as the X axis
        ax, ay = ay, -ax
        u_lo, u_hi, v_lo, v_hi = v_lo, v_hi, -u_hi, -u_lo
        du, dv = dv, du
    if float(ax @ cx_axis) < 0:       # deterministic sign
        ax, ay = -ax, -ay
        u_lo, u_hi = -u_hi, -u_lo
        v_lo, v_hi = -v_hi, -v_lo
    if du < 1e-3 or dv < 1e-3:
        return None, "degenerate rectangle"

    centre = origin + ax * (u_lo + u_hi) / 2.0 + ay * (v_lo + v_hi) / 2.0
    corners = np.array([
        centre - ax * du / 2 - ay * dv / 2,
        centre + ax * du / 2 - ay * dv / 2,
        centre + ax * du / 2 + ay * dv / 2,
        centre - ax * du / 2 + ay * dv / 2,
    ])

    # Occupancy over a grid rather than a convex hull ratio: a hull cannot see
    # the hole left by a parcel standing on top of this one. Points outside the
    # trimmed extent are MASKED OUT, not clipped in.
    cell = max(p.voxel * 2.0, 0.008)
    nu = int(du / cell) + 1
    nv = int(dv / cell) + 1
    fu = (rel @ ax - u_lo) / cell
    fv = (rel @ ay - v_lo) / cell
    inside = (fu >= 0) & (fu < nu) & (fv >= 0) & (fv < nv)
    if int(inside.sum()) == 0:
        coverage = 0.0
    else:
        gu = fu[inside].astype(np.int64)
        gv = fv[inside].astype(np.int64)
        occupied = len(np.unique(gu * nv + gv))
        coverage = float(min(1.0, occupied / max(nu * nv, 1)))

    return {
        "centre": centre,
        "corners": corners,
        "x_axis": ax,
        "y_axis": ay,
        "length_m": du,
        "width_m": dv,
        "area_m2": float(du * dv),
        "coverage": coverage,
        "points_outside_trim": int((~inside).sum()),
    }, None


def polygon_gap(a_uv: np.ndarray, b_uv: np.ndarray) -> float:
    """Smallest in-plane distance between two rectangles, negative if they
    overlap."""
    a = (np.asarray(a_uv) * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    b = (np.asarray(b_uv) * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    dists = [cv2.pointPolygonTest(b, (float(q[0][0]), float(q[0][1])), True) for q in a]
    dists += [cv2.pointPolygonTest(a, (float(q[0][0]), float(q[0][1])), True) for q in b]
    return float(-max(dists) / 1000.0)


def rect_contains(rect_uv: np.ndarray, point_uv: np.ndarray) -> bool:
    poly = (np.asarray(rect_uv) * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point_uv[0] * 1000.0),
                                       float(point_uv[1] * 1000.0)), False) >= 0


def rect_iou(a_uv, b_uv) -> float:
    """Plan-view intersection over union of two rectangles.

    The right metric for asking whether two COMPLETE detections are the same
    object. It is the wrong metric for asking whether two partial views are
    looking at the same surface: see rect_overlap_min.
    """
    a = (np.asarray(a_uv) * 1000.0).astype(np.float32)
    b = (np.asarray(b_uv) * 1000.0).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    area_a = float(cv2.contourArea(a))
    area_b = float(cv2.contourArea(b))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def rect_overlap_min(a_uv, b_uv) -> float:
    """Plan-view intersection over the SMALLER of the two rectangles.

    Used by face_consensus.group_by_footprint to decide whether two cameras
    are looking at the same physical surface. Intersection over UNION cannot
    answer that: a camera seeing half a parcel top scores 0.50 against the
    camera seeing all of it even with perfect containment, so the partial view
    is filed as a different surface and the parcel is reported from one view at
    half its true footprint, with an inter-view spread of 0.0 mm which on a
    multi-camera parcel is zero by construction rather than by agreement.
    """
    a = (np.asarray(a_uv) * 1000.0).astype(np.float32)
    b = (np.asarray(b_uv) * 1000.0).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(float(cv2.contourArea(a)), float(cv2.contourArea(b)))
    return float(inter / smaller) if smaller > 0 else 0.0


# --------------------------------------------------------------------------
# belt and conveyor
# --------------------------------------------------------------------------

def belt_from_raster(P, heights, n_conv, d_conv, ex, ey, p: SegParams):
    """The conveyor footprint, solved on a raster of the deck plane.

    Connectivity alone is the wrong primitive. The deck is visible only between
    the parcels standing on it, so DBSCAN returns each strip as its own cluster
    and the footprint SHRINKS as the belt fills. Since the footprint bounds the
    parcel search, the parcels are then clipped by a bound their own presence
    created.
    """
    on = np.abs(heights) <= p.plane_thresh
    if p.belt_include_objects:
        on = on | ((heights > p.min_height) & (heights < p.max_height))
    idx = np.flatnonzero(on)
    if len(idx) < 500:
        return None, "too few deck-plane points to identify the belt"

    u = P[idx] @ ex
    v = P[idx] @ ey
    cell = max(float(p.belt_raster_m), 0.004)
    u0, v0 = float(u.min()), float(v.min())
    W = int((u.max() - u0) / cell) + 1
    H = int((v.max() - v0) / cell) + 1
    while W * H > 40_000_000 and cell < 0.1:
        cell *= 2.0
        W = int((u.max() - u0) / cell) + 1
        H = int((v.max() - v0) / cell) + 1
    if W < 3 or H < 3:
        return None, "the deck plane spans too little to raster"

    img = np.zeros((H, W), np.uint8)
    img[np.clip(((v - v0) / cell).astype(np.int64), 0, H - 1),
        np.clip(((u - u0) / cell).astype(np.int64), 0, W - 1)] = 255

    k = max(3, int(round(p.belt_close / cell)))
    k += 1 - (k % 2)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n_lab <= 1:
        return None, "the deck plane did not form a connected patch"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    ys, xs = np.nonzero((lab == biggest) & (img > 0))
    if len(xs) < 100:
        return None, "the largest deck component holds too few cells"
    uv_mm = np.stack([xs * cell + u0, ys * cell + v0], axis=1) * 1000.0
    (cu, cv_), (w_mm, h_mm), angle = cv2.minAreaRect(uv_mm.astype(np.float32))
    if w_mm < 1.0 or h_mm < 1.0:
        return None, "degenerate belt rectangle"

    a = np.radians(angle)
    bx = ex * np.cos(a) + ey * np.sin(a)
    by = -ex * np.sin(a) + ey * np.cos(a)
    du, dv = w_mm / 1000.0, h_mm / 1000.0
    if du < dv:
        bx, by = by, -bx
        du, dv = dv, du
    if float(bx @ ex) < 0:
        bx, by = -bx, -by

    centre = (ex * (cu / 1000.0) + ey * (cv_ / 1000.0) + n_conv * (-d_conv))
    corners = np.array([
        centre - bx * du / 2 - by * dv / 2,
        centre + bx * du / 2 - by * dv / 2,
        centre + bx * du / 2 + by * dv / 2,
        centre - bx * du / 2 + by * dv / 2,
    ])
    return {
        "centre_m": centre.tolist(),
        "corners_m": corners.tolist(),
        "x_axis": bx.tolist(),
        "y_axis": by.tolist(),
        "length_m": float(du),
        "width_m": float(dv),
        "n_points": int(len(xs)),
        "n_components": int(n_lab - 1),
        "raster_cell_m": round(cell, 4),
        "closed_m": p.belt_close,
        "included_objects": bool(p.belt_include_objects),
        "margin_m": p.belt_margin,
        "source": "fitted",
    }, None


def load_belt(path):
    """Read a frozen footprint. The conveyor does not move between captures."""
    f = Path(path)
    if not f.exists():
        return None
    try:
        belt = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if not all(k in belt for k in ("centre_m", "corners_m", "x_axis",
                                   "y_axis", "length_m", "width_m")):
        return None
    belt["source"] = f"loaded from {path}"
    return belt


def save_belt(path, belt):
    try:
        Path(path).write_text(json.dumps(belt, indent=2, default=jsonable))
        return True
    except OSError:
        return False


def find_belt(P, heights, n_conv, cx_axis, cy_axis, d_conv, p: SegParams):
    """The conveyor footprint: loaded if frozen, otherwise solved and frozen."""
    if p.belt_file and not p.belt_refit:
        belt = load_belt(p.belt_file)
        if belt is not None:
            belt["margin_m"] = p.belt_margin
            return belt, None

    belt, why = belt_from_raster(P, heights, n_conv, d_conv, cx_axis, cy_axis, p)
    if belt is None:
        return None, why
    if p.belt_file:
        belt["frozen_to"] = str(p.belt_file)
        save_belt(p.belt_file, belt)
    return belt, None


def find_conveyor(points, p: SegParams):
    """Fit the conveyor plane, rejecting candidates at the wrong distance."""
    remaining = np.arange(len(points), dtype=np.int64)
    discarded = []
    for _ in range(max(1, p.max_plane_attempts)):
        n, d, inl = plane_ransac(points[remaining], p.plane_thresh,
                                 p.plane_iter, p.seed)
        if n is None:
            return None, None, discarded
        # The world origin is the reference camera centre, so orienting the
        # normal towards it makes every parcel height positive.
        n, d = orient_towards(n, d, np.zeros(3))
        if p.plane_distance is None or (p.plane_distance[0] <= d <= p.plane_distance[1]):
            return n, d, discarded
        discarded.append({"distance_m": round(float(d), 4),
                          "n_points": int(len(inl)),
                          "normal": [round(float(v), 4) for v in n]})
        keep = np.ones(len(remaining), dtype=bool)
        keep[inl] = False
        remaining = remaining[keep]
        if len(remaining) < 100:
            break
    return None, None, discarded


def cluster_above(P, heights, above, cx_axis, cy_axis, p: SegParams):
    """DBSCAN over the points standing above the conveyor, ANISOTROPICALLY.

    Two views of one top face separated by tens of millimetres along the normal
    cannot both sit inside a 20 mm neighbourhood in world metres, so they never
    enter the same cluster, the reconciliation never sees them together, and
    the partial view is written out as its own parcel. The reconciliation
    cannot fix what clustering already separated, so this has to happen first.
    """
    Q = P[above]
    hs = heights[above] * float(p.cluster_height_scale)
    xyz = np.stack([Q @ cx_axis, Q @ cy_axis, hs], axis=1)

    # A parcel top is a sheet one voxel thick, so the neighbours available
    # inside one radius are about pi*(eps/spacing)^2. Asking for more than a
    # fraction of that leaves no core points and the sheet shatters.
    expected = np.pi * (p.cluster_eps / max(p.voxel, 1e-6)) ** 2
    eff_min_points = min(p.cluster_min_points, max(6, int(0.4 * expected)))

    seed_ransac(p.seed)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    raw = np.asarray(cloud.cluster_dbscan(eps=p.cluster_eps,
                                          min_points=eff_min_points,
                                          print_progress=False))
    return raw, eff_min_points, expected


def _consensus_record(cons, cam_names):
    """The reconciliation, flattened for the record and the CSV.

    Both branches now carry ok, views_used and single_view, because the caller
    gates on them and a failure record that omitted them let an unreconciled
    parcel through as pickable.
    """
    if cons is None:
        return None
    if not cons.get("ok"):
        return {
            "ok": False,
            "reason": cons.get("reason"),
            "views_used": [],
            "views_dropped": cons.get("views_dropped", []),
            "skipped": cons.get("skipped", []),
            "per_view": cons.get("views", []),
            "single_view": False,
            "inter_view_offset_mm": None,
            "height_uncertainty_mm": None,
        }
    return {
        "ok": True,
        "views_used": cons["views_used_names"],
        "views_dropped": cons["views_dropped"],
        "skipped": cons["skipped"],
        "per_view": cons["views"],
        "shift_mm": {cam_names[i]: round(s * 1e3, 2)
                     for i, s in cons["shifts"].items()},
        "consensus_estimator": cons.get("consensus_estimator"),
        "consensus_iterations": cons.get("consensus_iterations"),
        "inter_view_offset_mm": cons["inter_view_offset_mm"],
        "inter_view_tilt_deg": cons["inter_view_tilt_deg"],
        "max_shift_mm": cons["max_shift_mm"],
        "worst_incidence_deg": cons["worst_incidence_deg"],
        "worst_comb_fraction": cons["worst_comb_fraction"],
        "widest_view": cons.get("widest_view"),
        "widest_view_area_m2": cons.get("widest_view_area_m2"),
        "view_area_ratio": cons.get("view_area_ratio"),
        "height_uncertainty_mm": cons["height_uncertainty_mm"],
        "height_uncertainty_lower_bound_mm": cons.get(
            "height_uncertainty_lower_bound_mm"),
        "uncertainty_is_lower_bound": cons.get("uncertainty_is_lower_bound"),
        "single_view": cons["single_view"],
    }


def apply_bias_ratio(boxes, p: SegParams):
    """Rescale every single-view height uncertainty by the measured ratio.

    A face carried by one camera was given an uncertainty equal to that
    camera's bias AT THE DECK. On this rig the cameras agree at the deck to
    about 2 mm and disagree on a face 260 mm above it by tens of millimetres,
    and the depth field compresses relief by roughly a quarter, so deck bias
    does not predict face bias and the figure understates by close to an order
    of magnitude in exactly the case it exists for.

    The ratio is measured from THIS capture's own multi-view faces, where both
    quantities are known, and refused below three samples because a ratio from
    one or two faces says nothing.
    """
    if not p.bias_ratio or measure_bias_ratio is None:
        return None
    ratio = measure_bias_ratio(boxes)
    if ratio is None:
        return None
    for b in boxes:
        vc = b.get("view_consensus") or {}
        if not vc.get("ok") or not vc.get("single_view"):
            continue
        lb = vc.get("height_uncertainty_lower_bound_mm")
        if lb is None:
            continue
        vc["height_uncertainty_mm"] = round(float(lb) * ratio, 2)
        vc["bias_ratio_applied"] = round(float(ratio), 2)
        vc["uncertainty_is_lower_bound"] = False
    return round(float(ratio), 3)


def suppress_overlaps(boxes, p: SegParams):
    """Drop detections that are another detection seen twice.

    Intersection over union is correct here, unlike in the view grouping: both
    candidates are complete detections of their own, so the difference in their
    visible extent is evidence rather than an artefact of occlusion.
    """
    def score(b):
        return (b["top_face"]["coverage"] * b["top_face"]["inlier_fraction"]
                * np.log1p(b["n_points"]))

    ranked = sorted(boxes, key=score, reverse=True)
    kept, dropped = [], []
    for b in ranked:
        clash = None
        for k in kept:
            dh = abs(b["top_face"]["height_above_conveyor_m"]
                     - k["top_face"]["height_above_conveyor_m"])
            if dh >= p.level_min_gap:
                continue
            if rect_iou(b["_conv_uv"], k["_conv_uv"]) > p.max_overlap:
                clash = k
                break
        if clash is None:
            kept.append(b)
        else:
            dropped.append({"cluster": b["cluster"], "level": b["level"],
                            "n_points": b["n_points"],
                            "reason": f"overlaps the stronger detection at "
                                      f"{clash['top_face']['height_above_conveyor_m'] * 1e3:.0f} mm, "
                                      f"so it is the same parcel found twice"})
    kept.sort(key=lambda b: b["id"])
    for new_id, b in enumerate(kept):
        b["id"] = new_id
    return kept, dropped


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

def segment_boxes(points, source, cam_names, p: SegParams,
                  cam_centres=None) -> dict:
    """Segment parcels and extract their pick data.

    points       (N, 3) world-frame metres
    source       (N,) index into cam_names recording which camera each point
                 came from
    cam_centres  optional (M, 3) optical centres in world metres. Without them
                 the views are still reconciled in depth, but the incidence
                 weighting and the comb detection are unavailable.
    """
    t_all = time.perf_counter()
    seed_ransac(p.seed)
    timings = {}
    pts_in = np.asarray(points, dtype=float)
    src_in = (np.asarray(source, dtype=np.int32) if source is not None
              else np.zeros(len(pts_in), dtype=np.int32))

    empty = {"ok": False, "reason": None, "conveyor": None, "boxes": [],
             "rejected": [], "warnings": [], "points": np.empty((0, 3)),
             "sources": np.empty(0, dtype=np.int32),
             "labels": np.empty(0, dtype=np.int32),
             "cam_names": list(cam_names), "timings_ms": {},
             "params": asdict(p), "n_points_segmented": 0}

    t0 = time.perf_counter()
    keep = np.ones(len(pts_in), dtype=bool)
    if p.roi is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = p.roi
        keep &= (pts_in[:, 0] >= xmin) & (pts_in[:, 0] <= xmax)
        keep &= (pts_in[:, 1] >= ymin) & (pts_in[:, 1] <= ymax)
        keep &= (pts_in[:, 2] >= zmin) & (pts_in[:, 2] <= zmax)
    idx_roi = np.flatnonzero(keep)
    if len(idx_roi) < 100:
        return {**empty, "reason": "fewer than 100 points survived the crop"}
    sub = voxel_indices(pts_in[idx_roi], p.voxel)
    P = pts_in[idx_roi[sub]]
    S = src_in[idx_roi[sub]]
    timings["subsample"] = (time.perf_counter() - t0) * 1e3

    # ---- conveyor ------------------------------------------------------
    t0 = time.perf_counter()
    n_conv, d_conv, discarded = find_conveyor(P, p)
    if n_conv is None:
        return {**empty, "reason": "conveyor plane fit failed"
                         + (f" ({len(discarded)} candidate(s) rejected on "
                            f"distance)" if discarded else ""),
                "points": P, "sources": S,
                "labels": np.full(len(P), -1, dtype=np.int32),
                "rejected_planes": discarded}
    heights = P @ n_conv + d_conv
    conv_inl = np.flatnonzero(np.abs(heights) <= p.plane_thresh)
    plane_fraction = float(len(conv_inl) / len(P))
    timings["conveyor_plane"] = (time.perf_counter() - t0) * 1e3

    cx_axis, _ = basis_from_normal(n_conv, np.array([1.0, 0.0, 0.0]))
    # Right-handed about the APPROACH axis, which is the axis the tool rotates
    # about, so a positive yaw here is a positive rotation of the wrist.
    cy_pick = np.cross(-n_conv, cx_axis)
    cy_pick = cy_pick / np.linalg.norm(cy_pick)

    conveyor = {
        "normal": n_conv.tolist(),
        "offset_m": float(d_conv),
        "inlier_fraction": plane_fraction,
        "inlier_rms_m": float(np.sqrt(float(np.mean(heights[conv_inl] ** 2)))),
        "distance_from_reference_camera_m": float(d_conv),
        "planes_rejected_on_distance": discarded,
        "frame_x_axis": cx_axis.tolist(),
        "frame_y_axis": cy_pick.tolist(),
        "ransac_seed": int(p.seed),
        "frame_note": "right-handed about the approach axis, that is about "
                      "-normal, so reported yaw matches the wrist rotation",
    }
    warnings = []

    use_consensus = bool(p.face_per_view) and consensus_shift is not None
    if p.face_per_view and consensus_shift is None:
        warnings.append(f"per-view reconciliation requested but "
                        f"face_consensus.py could not be imported "
                        f"({_CONSENSUS_IMPORT_ERROR}); top faces were fitted "
                        f"across all cameras at once, so inter-view "
                        f"disagreement is neither corrected nor reported")
    if deck_offsets_per_view is not None:
        conveyor["per_view"] = deck_offsets_per_view(heights, S, cam_names,
                                                    conv_inl)
        spread = [v["median_mm"] for v in conveyor["per_view"].values()]
        if len(spread) > 1:
            conveyor["per_view_spread_mm"] = round(max(spread) - min(spread), 2)
            if conveyor["per_view_spread_mm"] > p.view_disagree_warn * 1e3:
                warnings.append(
                    f"the cameras disagree by "
                    f"{conveyor['per_view_spread_mm']:.1f} mm about the deck "
                    f"plane itself, so every height below carries at least "
                    f"that much systematic error. Check the online alignment "
                    f"before reading the parcel figures")
    else:
        conveyor["per_view"] = {}

    if plane_fraction < p.min_plane_fraction:
        warnings.append(
            f"the fitted plane holds only {plane_fraction * 100:.1f} per cent "
            f"of the points, below the {p.min_plane_fraction * 100:.0f} per "
            f"cent required, so it may not be the conveyor")
    if p.plane_distance is None:
        warnings.append(
            f"no --seg-plane-distance given, so the plane is whichever surface "
            f"is largest. It came out {d_conv:.3f} m from the reference camera: "
            f"if that is the floor rather than the deck, every result below is "
            f"meaningless")
    if use_consensus and cam_centres is None:
        warnings.append(
            "no camera centres supplied, so views were reconciled in depth but "
            "not weighted by incidence, and combed surfaces were not detected")

    labels = np.full(len(P), -1, dtype=np.int32)
    labels[conv_inl] = -2

    # ---- belt footprint ------------------------------------------------
    belt, on_belt = None, np.ones(len(P), dtype=bool)
    if p.belt_only:
        belt, why = find_belt(P, heights, n_conv, cx_axis, cy_pick, d_conv, p)
        if belt is None:
            warnings.append(f"{why}; the search was not bounded to the belt, so "
                            f"structure standing at deck height elsewhere in "
                            f"the cell will be reported as parcels")
        else:
            centre = np.asarray(belt["centre_m"])
            bx = np.asarray(belt["x_axis"])
            by = np.asarray(belt["y_axis"])
            rel = P - centre
            on_belt = ((np.abs(rel @ bx) <= belt["length_m"] / 2 + p.belt_margin)
                       & (np.abs(rel @ by) <= belt["width_m"] / 2 + p.belt_margin))
            belt["points_inside"] = int(on_belt.sum())
            if belt.get("source") == "fitted" and not p.belt_file:
                warnings.append(
                    f"the belt footprint was re-solved from this capture "
                    f"({belt['length_m'] * 1e3:.0f} x {belt['width_m'] * 1e3:.0f} "
                    f"mm). It bounds the parcel search, so any variation in it "
                    f"between captures moves the reported dimensions with it. "
                    f"Pass --seg-belt-file to measure it once and freeze it")
    conveyor["belt"] = belt

    # ---- clustering ----------------------------------------------------
    t0 = time.perf_counter()
    above = np.flatnonzero((heights > p.min_height) & (heights < p.max_height)
                           & on_belt)
    if len(above) < p.min_points:
        timings["cluster"] = (time.perf_counter() - t0) * 1e3
        timings["total"] = (time.perf_counter() - t_all) * 1e3
        return {**empty, "ok": True,
                "reason": "nothing stands above the conveyor",
                "conveyor": conveyor, "warnings": warnings, "points": P,
                "sources": S, "labels": labels,
                "n_points_segmented": int(len(P)),
                "timings_ms": {k: round(v, 2) for k, v in timings.items()}}

    raw, eff_min_points, expected = cluster_above(P, heights, above,
                                                  cx_axis, cy_pick, p)
    if eff_min_points < p.cluster_min_points:
        warnings.append(
            f"--seg-cluster-min-points {p.cluster_min_points} exceeds what a "
            f"sheet at {p.voxel * 1e3:.0f} mm spacing can supply within "
            f"{p.cluster_eps * 1e3:.0f} mm (about {expected:.0f} neighbours), "
            f"so it was reduced to {eff_min_points}")
    conveyor["clustering"] = {
        "eps_m": p.cluster_eps,
        "height_scale": p.cluster_height_scale,
        "effective_height_eps_m": round(p.cluster_eps
                                        / max(p.cluster_height_scale, 1e-6), 4),
        "min_points_effective": int(eff_min_points),
        "note": ("the height axis is compressed before clustering, so the "
                 "radius reaches further in depth than in plan and two views "
                 "of one face that disagree in depth still cluster together"),
    }
    timings["cluster"] = (time.perf_counter() - t0) * 1e3

    # ---- per-cluster fitting -------------------------------------------
    t0 = time.perf_counter()
    consensus_ms = 0.0
    boxes, rejected = [], []
    n_clusters = int(raw.max()) + 1 if raw.size and raw.max() >= 0 else 0
    for cid in range(n_clusters):
        sel_all = above[raw == cid]
        if len(sel_all) < p.min_points:
            rejected.append({"cluster": int(cid), "n_points": int(len(sel_all)),
                             "reason": f"only {len(sel_all)} points"})
            continue

        remaining = sel_all.copy()
        levels, attempts, last_h = 0, 0, None
        while (levels < p.max_levels and attempts < p.max_levels * 2
               and len(remaining) >= p.min_face_points):
            attempts += 1
            cpts_raw = P[remaining]
            csrc = S[remaining]

            cons = None
            if use_consensus:
                t_c = time.perf_counter()
                cons = consensus_shift(cpts_raw, csrc, n_conv, d_conv,
                                       cam_names, cam_centres, p,
                                       cx_axis, cy_pick,
                                       conveyor.get("per_view"))
                consensus_ms += (time.perf_counter() - t_c) * 1e3

            if cons is not None and cons["ok"]:
                cpts, idx_use, idx_excl = apply_consensus(
                    cpts_raw, csrc, remaining, n_conv, cons)
            else:
                cpts, idx_use = cpts_raw, remaining
                idx_excl = np.empty(0, dtype=np.int64)

            if len(cpts) < p.min_face_points:
                if levels == 0 and attempts == 1:
                    rejected.append({
                        "cluster": int(cid), "level": levels,
                        "n_points": int(len(remaining)),
                        "reason": (cons or {}).get("reason")
                                  or f"only {len(cpts)} points survived view "
                                     f"reconciliation, below "
                                     f"{p.min_face_points}",
                        "view_consensus": _consensus_record(cons, cam_names)})
                remaining = idx_use
                continue

            h_rem = cpts @ n_conv + d_conv
            h_top = float(np.percentile(h_rem, p.top_percentile))

            face, why = fit_top_face(cpts, n_conv, d_conv, p)

            growth = None
            if face is not None and p.face_grow_scale > 0:
                spread_m = float((cons or {}).get(
                    "inter_view_offset_mm") or 0.0) / 1e3
                tol = face_growth_tolerance(spread_m, p)
                if tol > p.face_thresh * 1.001:
                    face, growth = grow_top_face(cpts, face, n_conv, cx_axis,
                                                 tol, p)

            rect = None
            if face is not None:
                rect, why = rect_on_plane(face["points"], face["normal"],
                                          cx_axis, p)
            if rect is None:
                if levels == 0 and attempts == 1:
                    rejected.append({"cluster": int(cid), "level": levels,
                                     "n_points": int(len(remaining)),
                                     "reason": why,
                                     "view_consensus": _consensus_record(
                                         cons, cam_names)})
                remaining = idx_use[h_rem < h_top - p.top_band]
                continue

            h_face = float(rect["centre"] @ n_conv + d_conv)

            rel = cpts - rect["centre"]
            u = rel @ rect["x_axis"]
            v = rel @ rect["y_axis"]
            margin = p.face_thresh * 2
            consumed = ((np.abs(u) <= rect["length_m"] / 2 + margin)
                        & (np.abs(v) <= rect["width_m"] / 2 + margin)
                        & (h_rem <= h_face + p.face_thresh * 3))
            sel = idx_use[consumed]
            remaining = idx_use[~consumed]

            aspect = rect["length_m"] / max(rect["width_m"], 1e-6)
            shape_why = None
            if rect["width_m"] < p.min_dim or rect["length_m"] > p.max_dim:
                shape_why = (f"footprint {rect['length_m'] * 1e3:.0f} x "
                             f"{rect['width_m'] * 1e3:.0f} mm outside the "
                             f"accepted range")
            elif aspect > p.max_aspect:
                shape_why = (f"footprint {rect['length_m'] * 1e3:.0f} x "
                             f"{rect['width_m'] * 1e3:.0f} mm is a sliver, "
                             f"aspect {aspect:.1f} above {p.max_aspect:.1f}")
            elif rect["area_m2"] < p.min_area:
                shape_why = (f"top face {rect['area_m2'] * 1e4:.0f} cm2, below "
                             f"{p.min_area * 1e4:.0f} cm2")
            if shape_why is not None:
                rejected.append({"cluster": int(cid), "level": levels,
                                 "n_points": int(len(sel)),
                                 "reason": shape_why})
                continue
            if last_h is not None and last_h - h_face < p.level_min_gap:
                rejected.append({"cluster": int(cid), "level": levels,
                                 "n_points": int(len(sel)),
                                 "reason": f"only {(last_h - h_face) * 1e3:.0f} mm "
                                           f"below the level already taken, so "
                                           f"it is the same surface"})
                continue
            if len(sel) < p.min_face_points:
                continue

            idx = len(boxes)
            labels[sel] = idx
            last_h = h_face
            levels += 1

            n_top = face["normal"]
            z_axis = -n_top                       # approach, into the surface
            x_axis = rect["x_axis"]
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)     # re-orthogonalise
            R = np.column_stack([x_axis, y_axis, z_axis])

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = rect["centre"]
            T_pre = T.copy()
            T_pre[:3, 3] = rect["centre"] + n_top * p.approach_offset

            yaw = float(np.degrees(np.arctan2(float(x_axis @ cy_pick),
                                              float(x_axis @ cx_axis))))
            yaw = (yaw + 90.0) % 180.0 - 90.0

            counts = {cam_names[i]: int(np.count_nonzero(S[sel] == i))
                      for i in range(len(cam_names))}
            dominant = max(counts.values()) / max(sum(counts.values()), 1)

            conv_uv = np.stack([rect["corners"] @ cx_axis,
                                rect["corners"] @ cy_pick], axis=1)
            centre_uv = np.array([float(rect["centre"] @ cx_axis),
                                  float(rect["centre"] @ cy_pick)])

            boxes.append({
                "id": idx,
                "cluster": int(cid),
                "level": int(levels - 1),
                "n_points": int(len(sel)),
                "top_face": {
                    "centre_m": rect["centre"].tolist(),
                    "normal": n_top.tolist(),
                    "corners_m": rect["corners"].tolist(),
                    "height_above_conveyor_m": h_face,
                    "tilt_deg": face["tilt_deg"],
                    "plane_residual_rms_m": face["residual_rms_m"],
                    "plane_residual_p95_m": face["residual_p95_m"],
                    "inliers": face["inliers"],
                    "inlier_fraction": face["inlier_fraction"],
                    "coverage": rect["coverage"],
                    "points_outside_trim": rect.get("points_outside_trim"),
                    "area_m2": rect["area_m2"],
                    "growth": growth,
                },
                "dimensions_m": {
                    "length": rect["length_m"],
                    "width": rect["width_m"],
                    "top_height_above_conveyor": h_face,
                },
                "pose_world": T.tolist(),
                "pre_pick_pose_world": T_pre.tolist(),
                "position_m": rect["centre"].tolist(),
                "quaternion_wxyz": rot_to_quat(R).tolist(),
                "rpy_deg_zyx": rot_to_rpy_deg(R),
                "approach_vector": z_axis.tolist(),
                "yaw_in_conveyor_frame_deg": yaw,
                "points_per_camera": counts,
                "dominant_camera_fraction": float(dominant),
                "single_view": bool(dominant >= p.single_view_frac),
                "view_consensus": _consensus_record(cons, cam_names),
                "_conv_uv": conv_uv,
                "_conv_centre_uv": centre_uv,
                "_label_sel": sel,
            })

    kept, dupes = suppress_overlaps(boxes, p)
    if dupes:
        new_labels = np.full(len(P), -1, dtype=np.int32)
        new_labels[conv_inl] = -2
        for b in kept:
            new_labels[b["_label_sel"]] = b["id"]
        labels = new_labels
        rejected.extend(dupes)
    boxes = kept
    for b in boxes:
        b.pop("_label_sel", None)
    timings["fit_faces"] = (time.perf_counter() - t0) * 1e3
    timings["view_consensus"] = consensus_ms

    # The face-to-deck bias ratio can only be measured once every face has been
    # reconciled, so the single-view uncertainties are rescaled here rather than
    # inside face_consensus.
    ratio = apply_bias_ratio(boxes, p)
    conveyor["face_to_deck_bias_ratio"] = ratio
    if ratio is not None:
        warnings.append(
            f"single-view height uncertainties were rescaled by {ratio:.1f}x, "
            f"the ratio between face bias and deck bias measured over this "
            f"capture's own multi-view faces. A camera 2 mm out at the deck is "
            f"therefore quoted at about {2 * ratio:.0f} mm on a parcel top, "
            f"which is the honest figure: deck bias does not predict face bias "
            f"on a depth field whose relief is compressed")
    elif p.bias_ratio:
        warnings.append(
            "fewer than three multi-view faces, so the face-to-deck bias ratio "
            "could not be measured and every single-view height uncertainty "
            "remains a LOWER BOUND taken at the deck. Read "
            "uncertainty_is_lower_bound before quoting it")

    # ---- relations between parcels -------------------------------------
    t0 = time.perf_counter()
    for b in boxes:
        gaps, blocked_by, supports = [], [], []
        for o in boxes:
            if o["id"] == b["id"]:
                continue
            gap = polygon_gap(b["_conv_uv"], o["_conv_uv"])
            gaps.append((gap, o["id"]))
            dh = (o["top_face"]["height_above_conveyor_m"]
                  - b["top_face"]["height_above_conveyor_m"])
            if gap < 0.005 and dh > 0.020:
                blocked_by.append(o["id"])
            if dh < -0.020 and rect_contains(o["_conv_uv"], b["_conv_centre_uv"]):
                supports.append((o["top_face"]["height_above_conveyor_m"], o["id"]))
        gaps.sort()
        supports.sort(reverse=True)

        support_h = supports[0][0] if supports else 0.0
        b["dimensions_m"]["height"] = float(
            b["top_face"]["height_above_conveyor_m"] - support_h)
        b["support"] = {
            "on_conveyor": not supports,
            "supported_by": supports[0][1] if supports else None,
            "support_height_m": float(support_h),
            "height_is_inferred": bool(supports),
        }
        b["nearest_neighbour"] = {
            "id": gaps[0][1] if gaps else None,
            "gap_m": float(gaps[0][0]) if gaps else None,
        }
        b["blocked_by"] = blocked_by

        blockers, cautions = [], []
        if b["top_face"]["coverage"] < p.min_coverage:
            blockers.append(f"the top face fills only "
                            f"{b['top_face']['coverage'] * 100:.0f} per cent of "
                            f"its own rectangle, so it is partly occluded")
        if b["top_face"]["plane_residual_rms_m"] > p.max_residual:
            blockers.append(f"plane residual "
                            f"{b['top_face']['plane_residual_rms_m'] * 1e3:.1f} mm "
                            f"exceeds the {p.max_residual * 1e3:.1f} mm limit")
        if blocked_by:
            blockers.append(f"a taller parcel is adjacent: {blocked_by}")
        if (b["nearest_neighbour"]["gap_m"] is not None
                and 0 <= b["nearest_neighbour"]["gap_m"] < p.clearance_warn):
            cautions.append(f"only {b['nearest_neighbour']['gap_m'] * 1e3:.0f} mm "
                            f"to the nearest parcel")
        if b["single_view"]:
            cautions.append("seen by essentially one camera, so the pose rests "
                            "on that view's bias alone")
        belt_rec = (conveyor.get("belt") or {})
        if belt_rec and p.belt_edge_warn > 0:
            bc = np.asarray(belt_rec["centre_m"])
            bx_ = np.asarray(belt_rec["x_axis"])
            by_ = np.asarray(belt_rec["y_axis"])
            rel_c = np.asarray(b["top_face"]["corners_m"]) - bc
            du_lim = belt_rec["length_m"] / 2 + p.belt_margin
            dv_lim = belt_rec["width_m"] / 2 + p.belt_margin
            reach = max(float(np.max(np.abs(rel_c @ bx_)) - du_lim),
                        float(np.max(np.abs(rel_c @ by_)) - dv_lim))
            b["belt_bound_clearance_mm"] = round(-reach * 1e3, 1)
            if reach > -p.belt_edge_warn:
                cautions.append(
                    f"the footprint reaches the belt search bound to within "
                    f"{max(-reach, 0.0) * 1e3:.0f} mm, so it may have been cut "
                    f"by the bound rather than measured. Widen "
                    f"--seg-belt-margin or freeze the belt with "
                    f"--seg-belt-file")
        if b["support"]["height_is_inferred"]:
            cautions.append(f"height inferred from parcel "
                            f"{b['support']['supported_by']} beneath it, not "
                            f"measured to the conveyor")

        # ---- what the views disagreed about ----------------------------
        vc = b.get("view_consensus")
        if vc is not None and vc.get("ok") is False:
            # THE GUARD THAT WAS MISSING. A failed reconciliation used to
            # produce no flag at all: n_views came back 0 from an absent
            # views_used key and the gate was written as 0 < n_views < required,
            # so the parcel was reported PICKABLE on a raw multi-view plane fit.
            # That is exactly the case the reconciliation exists to catch, since
            # an inter-view offset larger than --seg-top-band silently reduces
            # the fit to whichever camera sits highest and reports it as clean.
            blockers.append(
                f"the per-view reconciliation FAILED here "
                f"({vc.get('reason') or 'no reason given'}), so this face was "
                f"fitted across every camera at once. An inter-view offset "
                f"larger than --seg-top-band "
                f"({p.top_band * 1e3:.0f} mm) reduces that fit to one camera "
                f"without saying so, and the face-growth tolerance falls back "
                f"to a fixed band. The pose is not corroborated")
            for d in vc.get("views_dropped", []):
                cautions.append(f"{d['camera']} excluded: {d['reason']}")
        elif vc and vc.get("ok"):
            n_views = len(vc.get("views_used", []))
            if 0 < n_views < p.min_views_pickable:
                blockers.append(
                    f"only {n_views} camera produced a usable top face here, "
                    f"below the {p.min_views_pickable} required. Nothing "
                    f"corroborates the height, and an uncorroborated detection "
                    f"that reads tall is placed first in the pick order")
            inc = vc.get("worst_incidence_deg")
            if vc.get("single_view") and inc is not None and inc > p.max_face_incidence:
                blockers.append(
                    f"this face is carried by {vc['views_used'][0]} alone at "
                    f"{inc:.0f} deg incidence, above the "
                    f"{p.max_face_incidence:.0f} deg limit, so its depth is "
                    f"quantisation-limited and uncorroborated")
            elif vc.get("single_view"):
                unc = vc.get("height_uncertainty_mm")
                lb = vc.get("uncertainty_is_lower_bound")
                cautions.append(
                    f"only {vc['views_used'][0]} produced a usable top face, so "
                    f"nothing corroborates its depth"
                    + (f"; the height is uncertain to about {unc:.1f} mm"
                       + (" and that figure is a LOWER BOUND measured at the "
                          "deck, which does not predict a face 260 mm above it"
                          if lb else
                          f", scaled from the deck by the measured "
                          f"{vc.get('bias_ratio_applied', 1):.1f}x face-to-deck "
                          f"ratio")
                       if unc is not None else ""))
            spread = vc.get("inter_view_offset_mm")
            if spread is not None and spread > p.view_disagree_warn * 1e3:
                cautions.append(
                    f"the cameras disagreed by {spread:.1f} mm about this "
                    f"surface before reconciliation, so the height is "
                    f"uncertain to about "
                    f"{vc.get('height_uncertainty_mm', 0):.1f} mm")
            ratio_v = vc.get("view_area_ratio")
            if ratio_v is not None and ratio_v < 0.6 and len(vc.get("views_used", [])) > 1:
                cautions.append(
                    f"the contributing views saw very different amounts of "
                    f"this face: the narrowest covers {ratio_v * 100:.0f} per "
                    f"cent of what {vc.get('widest_view')} covers, so the "
                    f"footprint rests mostly on that one view")
            comb = vc.get("worst_comb_fraction")
            if comb is not None and comb > p.comb_warn:
                cautions.append(
                    f"a contributing view is combed over {comb * 100:.0f} per "
                    f"cent of its along-ray extent, so it is sampling this "
                    f"face at the depth quantisation rather than resolving it")
            for d in vc.get("views_dropped", []):
                cautions.append(f"{d['camera']} excluded: {d['reason']}")
        elif vc is None and use_consensus:
            blockers.append(
                "no reconciliation record exists for this face, so nothing is "
                "known about whether the cameras agreed on it")

        b["pickable"] = not blockers
        b["blocking_reasons"] = blockers
        b["cautions"] = cautions
        b["reasons"] = blockers + cautions

    for rank, b in enumerate(sorted(
            boxes, key=lambda x: (-x["top_face"]["height_above_conveyor_m"],
                                  -x["n_points"]))):
        b["pick_order"] = rank
    if len(boxes) > p.max_boxes_warn:
        tops = sorted(b["top_face"]["height_above_conveyor_m"] for b in boxes)
        span = tops[-1] - tops[0] if tops else 0.0
        warnings.append(
            f"{len(boxes)} parcels found, above the {p.max_boxes_warn} expected. "
            f"Their tops span {span * 1e3:.0f} mm at a median of "
            f"{statistics_median(tops) * 1e3:.0f} mm above the plane. A tight "
            f"cluster of many objects at one height is the signature of a "
            f"structure being segmented, usually the roller deck after the "
            f"floor was fitted as the plane; set --seg-plane-distance or "
            f"--seg-roi")

    singles = [b for b in boxes if b.get("single_view")]
    if len(singles) >= 2 and len(singles) >= 0.5 * max(len(boxes), 1):
        cams = {c for b in singles for c, n in b["points_per_camera"].items() if n}
        if len(cams) > 1:
            warnings.append(
                f"{len(singles)} of {len(boxes)} parcels are carried by a "
                f"single camera each, across {len(cams)} different cameras. "
                f"That is what one parcel split into several by the "
                f"inter-camera depth disagreement looks like. Lower "
                f"--seg-cluster-height-scale (currently "
                f"{p.cluster_height_scale}) so the clustering stops separating "
                f"views of the same face, and check "
                f"conveyor.per_view_spread_mm above")
    n_failed = sum(1 for b in boxes
                   if (b.get("view_consensus") or {}).get("ok") is False)
    if n_failed:
        warnings.append(
            f"{n_failed} of {len(boxes)} parcels had their per-view "
            f"reconciliation FAIL and are held. Previously these were reported "
            f"pickable, because the min-views gate read an absent views_used "
            f"key as zero and its lower bound excluded that case")
    timings["relations"] = (time.perf_counter() - t0) * 1e3
    timings["total"] = (time.perf_counter() - t_all) * 1e3

    for b in boxes:
        b.pop("_conv_uv", None)
        b.pop("_conv_centre_uv", None)

    return {
        "ok": True,
        "reason": None,
        "conveyor": conveyor,
        "boxes": boxes,
        "rejected": rejected,
        "warnings": warnings,
        "n_points_segmented": int(len(P)),
        "points": P,
        "sources": S,
        "labels": labels,
        "cam_names": list(cam_names),
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "params": asdict(p),
    }


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def coloured_cloud(result) -> o3d.geometry.PointCloud:
    """The segmented cloud, one colour per parcel, conveyor in grey.

    These are the RAW point positions, not the reconciled ones, deliberately:
    the cloud shows what the cameras measured and boxes.json quantifies how far
    apart they were.
    """
    P = np.asarray(result["points"])
    labels = np.asarray(result["labels"])
    cols = np.tile(np.array(UNASSIGNED_RGB), (len(P), 1))
    cols[labels == -2] = CONVEYOR_RGB
    for b in result["boxes"]:
        bgr = BOX_COLOURS[b["id"] % len(BOX_COLOURS)]
        cols[labels == b["id"]] = np.array(bgr[::-1], dtype=float) / 255.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def draw_overlays(result, images, K_all, E_all, p: SegParams) -> dict:
    """Annotate each camera view with the parcels found in 3D."""
    names = result["cam_names"]
    P = np.asarray(result["points"])
    labels = np.asarray(result["labels"])
    sources = np.asarray(result["sources"])
    n_conv = np.asarray((result.get("conveyor") or {}).get("normal",
                                                           [0.0, 0.0, -1.0]))
    out = {}

    for i, name in enumerate(names):
        img = as_uint8_image(images[i])
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if p.desaturate:
            grey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            canvas = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        K = np.asarray(K_all[i], dtype=float)
        if p.overlay_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=p.overlay_scale,
                                fy=p.overlay_scale,
                                interpolation=cv2.INTER_LINEAR)
            K = scale_K(K, p.overlay_scale)
        h_img, w_img = canvas.shape[:2]
        E = np.asarray(E_all[i], dtype=float)

        if p.draw_points and result["boxes"]:
            layer = canvas.copy()
            for b in result["boxes"]:
                mask = labels == b["id"]
                if not mask.any():
                    continue
                uv, ok = project(P[mask], K, E)
                uv = uv[ok]
                if not len(uv):
                    continue
                u = np.round(uv[:, 0]).astype(int)
                v = np.round(uv[:, 1]).astype(int)
                inside = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
                layer[v[inside], u[inside]] = BOX_COLOURS[b["id"] % len(BOX_COLOURS)]
            canvas = cv2.addWeighted(layer, 0.4, canvas, 0.6, 0)

        cam_centre_i = camera_centre(E)
        order = sorted(result["boxes"],
                       key=lambda b: -float(np.linalg.norm(
                           np.asarray(b["position_m"]) - cam_centre_i)))

        fill = canvas.copy()
        drawn = []
        for b in order:
            colour = BOX_COLOURS[b["id"] % len(BOX_COLOURS)]
            top = np.asarray(b["top_face"]["corners_m"])
            uv, ok = project(top, K, E)
            if not ok.all() or not np.isfinite(uv).all():
                continue
            poly = np.round(uv).astype(np.int32)
            if (poly[:, 0].max() < 0 or poly[:, 1].max() < 0
                    or poly[:, 0].min() > w_img or poly[:, 1].min() > h_img):
                continue
            cv2.fillPoly(fill, [poly.reshape(-1, 1, 2)], colour)
            drawn.append((b, poly))
        canvas = cv2.addWeighted(fill, 0.35, canvas, 0.65, 0)

        for b, poly in drawn:
            colour = BOX_COLOURS[b["id"] % len(BOX_COLOURS)]
            vc = b.get("view_consensus") or {}
            used = vc.get("views_used")
            excluded = any(d.get("camera") == name
                           for d in vc.get("views_dropped", []))
            contributed = b["points_per_camera"].get(name, 0) >= p.min_view_points
            solid = contributed and not excluded and (not used or name in used)
            if excluded:
                edge = (60, 60, 245)          # this view was not trusted here
            elif solid:
                edge = (255, 255, 255)
            else:
                edge = (150, 150, 150)
            top = np.asarray(b["top_face"]["corners_m"])

            base = top - n_conv * float(b["dimensions_m"]["height"])
            buv, bok = project(base, K, E)
            if bok.all() and np.isfinite(buv).all():
                bpoly = np.round(buv).astype(np.int32)
                cv2.polylines(canvas, [bpoly.reshape(-1, 1, 2)], True,
                              edge, 1, cv2.LINE_AA)
                for a, c in zip(poly, bpoly):
                    cv2.line(canvas, tuple(a), tuple(c), edge, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], True, edge,
                          2 if solid else 1, cv2.LINE_AA)

            centre = np.asarray(b["position_m"])
            R = np.asarray(b["pose_world"])[:3, :3]
            ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            ring = (centre + p.pick_radius * (np.outer(np.cos(ang), R[:, 0])
                                              + np.outer(np.sin(ang), R[:, 1])))
            cuv, cok = project(np.asarray([centre]), K, E)
            if not cok[0] or not np.isfinite(cuv[0]).all():
                continue
            cu, cvv = int(round(cuv[0][0])), int(round(cuv[0][1]))

            r_px = 0
            ruv, rok = project(ring, K, E)
            if rok.all() and np.isfinite(ruv).all():
                rpoly = np.round(ruv).astype(np.int32)
                cv2.polylines(canvas, [rpoly.reshape(-1, 1, 2)], True,
                              (60, 60, 245), 2, cv2.LINE_AA)
                r_px = int(np.max(np.linalg.norm(
                    rpoly - np.array([cu, cvv]), axis=1)))
            cv2.drawMarker(canvas, (cu, cvv), (60, 60, 245), cv2.MARKER_CROSS,
                           12, 2, cv2.LINE_AA)

            diag = float(np.linalg.norm(poly.max(axis=0) - poly.min(axis=0)))
            fs = float(np.clip(diag / 160.0, 0.7, 2.4))
            text = str(b["id"])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            org = (cu - tw // 2, max(th + 2, cvv - r_px - 8))
            label_colour = colour if b["pickable"] else (0, 140, 255)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), max(4, int(fs * 4)), cv2.LINE_AA)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                        label_colour, max(2, int(fs * 2)), cv2.LINE_AA)

            if p.label_detail == "none":
                continue
            lines = [f"{b['dimensions_m']['length'] * 1e3:.0f} x "
                     f"{b['dimensions_m']['width'] * 1e3:.0f} x "
                     f"{b['dimensions_m']['height'] * 1e3:.0f} mm"]
            if p.label_detail == "full":
                unc = vc.get("height_uncertainty_mm")
                lines += [f"top {b['top_face']['height_above_conveyor_m'] * 1e3:.0f} mm"
                          + (f" +/- {unc:.0f}" if unc else "")
                          + f"   yaw {b['yaw_in_conveyor_frame_deg']:+.1f} deg",
                          f"order {b['pick_order']}"
                          + ("" if b["pickable"] else "   HOLD")]
            widest = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                         for l in lines)
            box_w = int(poly[:, 0].max() - poly[:, 0].min())
            box_h = int(poly[:, 1].max() - poly[:, 1].min())
            if widest > box_w - 8 or box_h < 20 * len(lines):
                continue
            tx = int(np.clip(poly[:, 0].min() + 4, 2, w_img - widest - 4))
            ty = int(np.clip(poly[:, 1].max() - 6 - 17 * (len(lines) - 1),
                             16, h_img - 17 * len(lines) - 2))
            for j, line in enumerate(lines):
                o = (tx, ty + 17 * j)
                cv2.putText(canvas, line, o, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, line, o, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

        deck_bias = ((result.get("conveyor") or {}).get("per_view") or {}).get(name)
        header = (f"{name}   parcels {len(result['boxes'])}   "
                  f"pick circle {p.pick_radius * 2e3:.0f} mm   "
                  f"points from this view {int(np.count_nonzero(sources == i))}"
                  + (f"   deck bias {deck_bias['median_mm']:+.1f} mm"
                     if deck_bias else ""))
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(canvas, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, th, cv2.LINE_AA)
        out[name] = canvas
    return out


def render_bev(result, p: SegParams) -> np.ndarray:
    """One combined top-down render of the fused cloud, from all cameras.

    Points are projected orthographically onto the conveyor plane and coloured
    by height, highest winning each pixel, so a millimetre is the same number
    of pixels everywhere. Four separate views cannot show whether the cameras
    agree; this can, because every point from every camera lands in the same
    raster.
    """
    P = np.asarray(result["points"])
    conv = result.get("conveyor")
    if conv is None or not len(P):
        return np.zeros((1, 1, 3), np.uint8)

    n = np.asarray(conv["normal"], dtype=float)
    d = float(conv["offset_m"])
    ex = np.asarray(conv["frame_x_axis"], dtype=float)
    ey = np.asarray(conv["frame_y_axis"], dtype=float)

    u = P @ ex
    v = P @ ey
    h = P @ n + d

    belt = conv.get("belt")
    pad = 0.02
    if belt:
        bc = np.asarray(belt["corners_m"])
        bu, bv = bc @ ex, bc @ ey
        u_lo, u_hi = float(bu.min()), float(bu.max())
        v_lo, v_hi = float(bv.min()), float(bv.max())
        stands = (h > p.min_height) & (h < p.max_height)
        if int(stands.sum()) > 100:
            u_lo = min(u_lo, float(np.percentile(u[stands], 0.2)))
            u_hi = max(u_hi, float(np.percentile(u[stands], 99.8)))
            v_lo = min(v_lo, float(np.percentile(v[stands], 0.2)))
            v_hi = max(v_hi, float(np.percentile(v[stands], 99.8)))
        pad = belt["margin_m"] + 0.04
    else:
        interest = (h > -p.plane_thresh * 3) & (h < p.max_height)
        if int(interest.sum()) < 100:
            interest = np.ones(len(h), dtype=bool)
        u_lo, u_hi = np.percentile(u[interest], [0.2, 99.8])
        v_lo, v_hi = np.percentile(v[interest], [0.2, 99.8])
    u_lo, u_hi, v_lo, v_hi = u_lo - pad, u_hi + pad, v_lo - pad, v_hi + pad

    s_m = max(p.bev_mm_per_px, 0.1) / 1000.0
    W = int(np.ceil((u_hi - u_lo) / s_m))
    H = int(np.ceil((v_hi - v_lo) / s_m))
    if max(W, H) > p.bev_max_px:
        s_m *= max(W, H) / p.bev_max_px
        W = int(np.ceil((u_hi - u_lo) / s_m))
        H = int(np.ceil((v_hi - v_lo) / s_m))
    W, H = max(W, 2), max(H, 2)

    def to_px(uu, vv):
        return ((uu - u_lo) / s_m, (vv - v_lo) / s_m)

    col = np.clip(((u - u_lo) / s_m).astype(np.int64), 0, W - 1)
    row = np.clip(((v - v_lo) / s_m).astype(np.int64), 0, H - 1)
    flat = row * W + col
    order = np.argsort(h)              # ascending, so the last write is highest
    top = np.full(W * H, np.nan)
    top[flat[order]] = h[order]

    hmax = float(np.nanpercentile(h, 99.5))
    hmax = max(hmax, 0.05)
    norm = np.clip(np.nan_to_num(top, nan=0.0) / hmax, 0, 1)
    canvas = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(H, W),
                               cv2.COLORMAP_TURBO)
    empty_px = np.isnan(top).reshape(H, W)
    top_img = np.nan_to_num(top, nan=0.0).reshape(H, W)
    canvas[empty_px] = (25, 25, 25)
    canvas[(~empty_px) & (top_img < p.min_height)] = (70, 70, 70)
    canvas[(~empty_px) & (top_img < -p.plane_thresh * 3)] = (40, 40, 40)

    if belt:
        bc = np.asarray(belt["corners_m"])
        bu, bv = to_px(bc @ ex, bc @ ey)
        bpoly = np.stack([bu, bv], axis=1).round().astype(np.int32)
        cv2.polylines(canvas, [bpoly.reshape(-1, 1, 2)], True,
                      (170, 170, 170), 1, cv2.LINE_AA)

    for b in result["boxes"]:
        colour = BOX_COLOURS[b["id"] % len(BOX_COLOURS)]
        corners = np.asarray(b["top_face"]["corners_m"])
        cu, cv_ = to_px(corners @ ex, corners @ ey)
        poly = np.stack([cu, cv_], axis=1).round().astype(np.int32)
        cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], True, colour, 2, cv2.LINE_AA)

        centre = np.asarray(b["position_m"])
        px, py = to_px(float(centre @ ex), float(centre @ ey))
        px, py = int(round(px)), int(round(py))
        cv2.drawMarker(canvas, (px, py), colour, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        axis_end = (centre + np.asarray(b["pose_world"])[:3, 0]
                    * b["dimensions_m"]["length"] * 0.45)
        ax, ay = to_px(float(axis_end @ ex), float(axis_end @ ey))
        cv2.arrowedLine(canvas, (px, py), (int(round(ax)), int(round(ay))),
                        colour, 2, cv2.LINE_AA, tipLength=0.25)

        vc = b.get("view_consensus") or {}
        unc = vc.get("height_uncertainty_mm")
        status = (0, 220, 0) if b["pickable"] else (0, 140, 255)
        lines = [f"#{b['id']} order {b['pick_order']}"
                 + ("" if b["pickable"] else "  HOLD"),
                 f"{b['dimensions_m']['length'] * 1e3:.0f} x "
                 f"{b['dimensions_m']['width'] * 1e3:.0f} x "
                 f"{b['dimensions_m']['height'] * 1e3:.0f} mm",
                 f"top {b['top_face']['height_above_conveyor_m'] * 1e3:.0f} mm"
                 + (f" +/- {unc:.0f}" if unc else "")
                 + f"  yaw {b['yaw_in_conveyor_frame_deg']:+.1f} deg",
                 f"views {'+'.join(vc.get('views_used', [])) or 'FAILED'}"
                 + (f"  spread {vc['inter_view_offset_mm']:.1f} mm"
                    if vc.get("inter_view_offset_mm") is not None else "")]
        ty = int(poly[:, 1].min()) + 16
        tx = int(poly[:, 0].min()) + 4
        widest = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                     for l in lines)
        tx = max(2, min(tx, W - widest - 4))
        ty = max(14, min(ty, H - 17 * len(lines) - 2))
        for j, line in enumerate(lines):
            org = (tx, ty + 17 * j)
            cv2.putText(canvas, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        status if j == 0 else (255, 255, 255), 1, cv2.LINE_AA)

    bar_m = 0.1
    bx, by = 20, H - 24
    cv2.line(canvas, (bx, by), (bx + int(bar_m / s_m), by),
             (255, 255, 255), 3, cv2.LINE_AA)
    for col_, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(canvas, f"{bar_m * 100:.0f} cm", (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_, th, cv2.LINE_AA)

    min_w = 460
    if W < min_w:
        left = (min_w - W) // 2
        canvas = np.hstack([np.full((H, left, 3), 22, np.uint8), canvas,
                            np.full((H, min_w - W - left, 3), 22, np.uint8)])
        W = min_w

    per_view = (result["conveyor"].get("per_view") or {})
    bias = "  ".join(f"{k} {v['median_mm']:+.1f}" for k, v in per_view.items())
    ratio = result["conveyor"].get("face_to_deck_bias_ratio")
    head = [f"fused top-down, all cameras   parcels {len(result['boxes'])}   "
            f"{p.bev_mm_per_px:.1f} mm/px",
            f"deck {result['conveyor']['distance_from_reference_camera_m']:.3f} m "
            f"from the reference camera   colour = height to {hmax * 1e3:.0f} mm"
            + (f"   belt {belt['length_m'] * 1e3:.0f} x "
               f"{belt['width_m'] * 1e3:.0f} mm "
               f"({'frozen' if str(belt.get('source', '')).startswith('loaded') else 'refitted this capture'})"
               if belt else "")]
    if bias:
        head.append(f"deck bias per view, mm   {bias}"
                    + (f"   face/deck ratio {ratio:.1f}x" if ratio else ""))
    scale = 0.5
    widest = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                 for l in head)
    if widest + 24 > W:
        scale = max(0.3, scale * (W - 24) / widest)
    bar = np.full((26 * len(head) + 8, W, 3), 18, np.uint8)
    for j, line in enumerate(head):
        cv2.putText(bar, line, (12, 20 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def mosaic(overlays: dict, names, per_row: int = 2) -> np.ndarray:
    tiles = [overlays[n] for n in names if n in overlays]
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[:t.shape[0], :t.shape[1]] = t
        padded.append(canvas)
    rows = [np.hstack(padded[i:i + per_row]) for i in range(0, len(padded), per_row)]
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
            if r.shape[1] < width else r for r in rows]
    return np.vstack(rows)


BOX_CSV_FIELDS = [
    "capture", "timestamp", "id", "cluster", "level", "pick_order", "pickable",
    "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
    "roll_deg", "pitch_deg", "yaw_deg", "yaw_conveyor_deg",
    "length_mm", "width_mm", "height_mm", "top_height_mm",
    "tilt_deg", "residual_rms_mm", "residual_p95_mm", "coverage",
    "n_points", "face_inliers", "dominant_camera_fraction", "single_view",
    # per-view reconciliation
    "consensus_ok", "n_views", "views_used", "views_dropped",
    "inter_view_offset_mm", "consensus_estimator",
    "face_tolerance_mm", "face_growth_gain",
    "inter_view_tilt_deg", "max_shift_mm", "worst_incidence_deg",
    "worst_comb_frac", "height_uncertainty_mm",
    "height_uncertainty_lower_bound_mm", "uncertainty_is_lower_bound",
    "bias_ratio_applied", "widest_view", "view_area_ratio",
    "belt_clearance_mm",
    "on_conveyor", "height_inferred", "support_height_mm",
    "neighbour_gap_mm", "blocked_by", "blocking_reasons", "cautions",
]


def _num(v):
    return "" if v is None else v


def box_rows(result, capture_index, timestamp):
    """One flat record per parcel, for the running log.

    consensus_ok is now a column of its own. A reconciliation that FAILED used
    to be indistinguishable in the log from one that succeeded with no views
    dropped, because both wrote an empty views_used field.
    """
    rows = []
    for b in result.get("boxes", []):
        pos, q, rpy = b["position_m"], b["quaternion_wxyz"], b["rpy_deg_zyx"]
        d, tf = b["dimensions_m"], b["top_face"]
        gap = b["nearest_neighbour"]["gap_m"]
        vc = b.get("view_consensus") or {}
        rows.append({
            "capture": capture_index, "timestamp": timestamp,
            "id": b["id"], "cluster": b["cluster"], "level": b["level"],
            "pick_order": b["pick_order"],
            "pickable": int(b["pickable"]),
            "x_m": round(pos[0], 5), "y_m": round(pos[1], 5),
            "z_m": round(pos[2], 5),
            "qw": round(q[0], 6), "qx": round(q[1], 6),
            "qy": round(q[2], 6), "qz": round(q[3], 6),
            "roll_deg": round(rpy[0], 2), "pitch_deg": round(rpy[1], 2),
            "yaw_deg": round(rpy[2], 2),
            "yaw_conveyor_deg": round(b["yaw_in_conveyor_frame_deg"], 2),
            "length_mm": round(d["length"] * 1e3, 1),
            "width_mm": round(d["width"] * 1e3, 1),
            "height_mm": round(d["height"] * 1e3, 1),
            "top_height_mm": round(d["top_height_above_conveyor"] * 1e3, 1),
            "tilt_deg": round(tf["tilt_deg"], 2),
            "residual_rms_mm": round(tf["plane_residual_rms_m"] * 1e3, 2),
            "residual_p95_mm": round(tf["plane_residual_p95_m"] * 1e3, 2),
            "coverage": round(tf["coverage"], 3),
            "n_points": b["n_points"], "face_inliers": tf["inliers"],
            "dominant_camera_fraction": round(b["dominant_camera_fraction"], 3),
            "single_view": int(b["single_view"]),
            "consensus_ok": ("" if not vc else int(bool(vc.get("ok")))),
            "n_views": len(vc.get("views_used", [])),
            "views_used": " ".join(vc.get("views_used", [])),
            "consensus_estimator": _num(vc.get("consensus_estimator")),
            "face_tolerance_mm": _num((tf.get("growth") or {}).get(
                "tolerance_mm")),
            "face_growth_gain": _num((tf.get("growth") or {}).get("gain")),
            "views_dropped": " ".join(d_["camera"] for d_
                                      in vc.get("views_dropped", [])),
            "inter_view_offset_mm": _num(vc.get("inter_view_offset_mm")),
            "inter_view_tilt_deg": _num(vc.get("inter_view_tilt_deg")),
            "max_shift_mm": _num(vc.get("max_shift_mm")),
            "worst_incidence_deg": _num(vc.get("worst_incidence_deg")),
            "worst_comb_frac": _num(vc.get("worst_comb_fraction")),
            "height_uncertainty_mm": _num(vc.get("height_uncertainty_mm")),
            "height_uncertainty_lower_bound_mm": _num(
                vc.get("height_uncertainty_lower_bound_mm")),
            "uncertainty_is_lower_bound": (
                "" if vc.get("uncertainty_is_lower_bound") is None
                else int(bool(vc.get("uncertainty_is_lower_bound")))),
            "bias_ratio_applied": _num(vc.get("bias_ratio_applied")),
            "widest_view": _num(vc.get("widest_view")),
            "view_area_ratio": _num(vc.get("view_area_ratio")),
            "belt_clearance_mm": _num(b.get("belt_bound_clearance_mm")),
            "on_conveyor": int(b["support"]["on_conveyor"]),
            "height_inferred": int(b["support"]["height_is_inferred"]),
            "support_height_mm": round(b["support"]["support_height_m"] * 1e3, 1),
            "neighbour_gap_mm": round(gap * 1e3, 1) if gap is not None else "",
            "blocked_by": " ".join(str(i) for i in b["blocked_by"]),
            "blocking_reasons": "; ".join(b["blocking_reasons"]),
            "cautions": "; ".join(b["cautions"]),
        })
    return rows


def write_results(result, out_dir: Path, capture_index, timestamp,
                  images=None, K_all=None, E_all=None, p: SegParams = None,
                  extra_meta: dict = None):
    """Write boxes.json, boxes.csv, the segmented cloud and the overlays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = p or SegParams()
    written = {}

    record = {
        "capture": capture_index,
        "timestamp": timestamp,
        "ok": result.get("ok"),
        "reason": result.get("reason"),
        "warnings": result.get("warnings", []),
        "conveyor": result.get("conveyor"),
        "n_boxes": len(result.get("boxes", [])),
        "n_points_segmented": result.get("n_points_segmented"),
        "boxes": result.get("boxes", []),
        "rejected": result.get("rejected", []),
        "timings_ms": result.get("timings_ms", {}),
        "parameters": result.get("params", {}),
        "metadata": extra_meta or {},
        "frame_note": "poses are in the world frame of the fusion, which is the "
                      "reference camera's optical frame; apply the hand-eye "
                      "transform before sending anything to the robot",
        "scale_note": "dimensions inherit the depth scale error. On this rig "
                      "DA3's relief has been measured to be compressed by about "
                      "26 per cent over 257 mm while the absolute standoff is "
                      "correct to under 2 per cent, so a height is far more "
                      "wrong than a distance. Validate against a parcel of "
                      "known size before quoting them",
        "consensus_note": "each parcel's view_consensus block records what the "
                          "cameras disagreed about before the fit. Check ok "
                          "FIRST: a failed reconciliation means the face was "
                          "fitted across every camera at once and the parcel is "
                          "held. Compare bias_at_face_mm with deck_bias_mm per "
                          "view: matching values are an additive bias the "
                          "online alignment should have removed, differing "
                          "values are a spatially structured bias that no "
                          "per-camera scalar can remove",
    }
    path = out_dir / "boxes.json"
    path.write_text(json.dumps(record, indent=2, default=jsonable))
    written["boxes_json"] = path

    rows = box_rows(result, capture_index, timestamp)
    path = out_dir / "boxes.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BOX_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    written["boxes_csv"] = path

    if len(result.get("points", [])):
        path = out_dir / "segmented.ply"
        o3d.io.write_point_cloud(str(path), coloured_cloud(result))
        written["segmented_ply"] = path

    if result.get("conveyor") is not None and p.views in ("bev", "all"):
        path = out_dir / "segmented.png"
        cv2.imwrite(str(path), render_bev(result, p))
        written["segmented_png"] = path

    if (images is not None and K_all is not None and E_all is not None
            and p.views in ("cameras", "all")):
        overlays = draw_overlays(result, images, K_all, E_all, p)
        for name, img in overlays.items():
            path = out_dir / f"seg_{name}.png"
            cv2.imwrite(str(path), img)
            written[f"seg_{name}"] = path
        name = "segmented.png" if p.views == "cameras" else "segmented_mosaic.png"
        path = out_dir / name
        cv2.imwrite(str(path), mosaic(overlays, result["cam_names"]))
        written["segmented_mosaic"] = path

    return written, rows


def summarise(result) -> str:
    """One console line per parcel, in pick order."""
    if not result.get("ok"):
        return f"  segmentation failed: {result.get('reason')}"
    conv = result.get("conveyor") or {}
    belt = conv.get("belt")
    head = (f"  plane {conv.get('distance_from_reference_camera_m', float('nan')):.3f} m "
            f"from the reference camera, holding "
            f"{conv.get('inlier_fraction', 0) * 100:.0f} per cent of the points, "
            f"residual {conv.get('inlier_rms_m', 0) * 1e3:.1f} mm")
    per_view = conv.get("per_view") or {}
    if per_view:
        head += ("\n  deck bias per view  "
                 + "  ".join(f"{k} {v['median_mm']:+.1f} mm"
                             for k, v in per_view.items()))
        if conv.get("per_view_spread_mm") is not None:
            head += f"   spread {conv['per_view_spread_mm']:.1f} mm"
    if conv.get("face_to_deck_bias_ratio") is not None:
        head += (f"\n  face/deck bias ratio {conv['face_to_deck_bias_ratio']:.1f}x, "
                 f"measured over this capture's multi-view faces")
    cl = conv.get("clustering") or {}
    if cl:
        head += (f"\n  clustering {cl['eps_m'] * 1e3:.0f} mm in plan, "
                 f"{cl['effective_height_eps_m'] * 1e3:.0f} mm in height "
                 f"(scale {cl['height_scale']})")
    if belt:
        head += (f"\n  belt {belt['length_m'] * 1e3:.0f} x "
                 f"{belt['width_m'] * 1e3:.0f} mm, search bounded to it plus "
                 f"{belt['margin_m'] * 1e3:.0f} mm  [{belt.get('source', '?')}]")
    if not result.get("boxes"):
        return head + "\n  no parcels found above it"
    lines = [head]
    for b in sorted(result["boxes"], key=lambda x: x["pick_order"]):
        d = b["dimensions_m"]
        vc = b.get("view_consensus") or {}
        unc = vc.get("height_uncertainty_mm")
        views = ("FAILED" if vc.get("ok") is False
                 else ("+".join(vc.get("views_used", [])) or "combined"))
        lines.append(
            f"  #{b['id']} order {b['pick_order']}  "
            f"{d['length'] * 1e3:6.1f} x {d['width'] * 1e3:6.1f} x "
            f"{d['height'] * 1e3:6.1f} mm  "
            f"top {d['top_height_above_conveyor'] * 1e3:6.1f} mm"
            + (f" +/- {unc:4.1f}" if unc is not None else "")
            + f"  yaw {b['yaw_in_conveyor_frame_deg']:+6.1f} deg  "
            f"res {b['top_face']['plane_residual_rms_m'] * 1e3:4.1f} mm  "
            f"views {views}  "
            + ("pickable" if b["pickable"]
               else "HOLD: " + b["blocking_reasons"][0])
            + (f"  [{len(b['cautions'])} caution(s)]" if b["cautions"] else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# standalone entry point, for tuning against a stored capture
# --------------------------------------------------------------------------

def probe(points, p: SegParams, n_planes: int = 6) -> int:
    """List the largest planes and the extent of the cloud.

    The distances printed here are in the MODEL's coordinates. They will not
    equal a ChArUco or tape measurement of the same surfaces, because the depth
    is unanchored: measured on this rig the absolute standoff reads about 2 per
    cent long while the relief between two surfaces reads about 26 per cent
    short. The number to put in --seg-plane-distance is the one printed here,
    not the physical one.
    """
    P = np.asarray(points, dtype=float)
    sub = voxel_indices(P, max(p.voxel, 0.008))
    Q = P[sub]
    print(f"{len(P)} points, {len(Q)} after subsampling  "
          f"(RANSAC seed {p.seed})\n")

    print(f"{'#':>2} {'distance_m':>11} {'points':>9} {'share':>7} "
          f"{'residual_mm':>12} {'tilt_to_1_deg':>14}")
    remaining = np.arange(len(Q), dtype=np.int64)
    first_n = None
    for i in range(max(1, n_planes)):
        if len(remaining) < 200:
            break
        n, d, inl = plane_ransac(Q[remaining], p.plane_thresh, p.plane_iter,
                                 p.seed)
        if n is None:
            break
        n, d = orient_towards(n, d, np.zeros(3))
        resid = Q[remaining][inl] @ n + d
        tilt = 0.0 if first_n is None else float(np.degrees(np.arccos(
            np.clip(abs(float(n @ first_n)), -1.0, 1.0))))
        if first_n is None:
            first_n = n
        print(f"{i:>2} {d:>11.3f} {len(inl):>9} {len(inl) / len(Q):>6.1%} "
              f"{np.sqrt(float(np.mean(resid ** 2))) * 1e3:>12.2f} "
              f"{tilt:>14.1f}")
        keep = np.ones(len(remaining), dtype=bool)
        keep[inl] = False
        remaining = remaining[keep]

    print("\nworld-frame extent, for --seg-roi XMIN XMAX YMIN YMAX ZMIN ZMAX")
    for axis, name in enumerate("XYZ"):
        v = Q[:, axis]
        lo, q1, q3, hi = np.percentile(v, [0.5, 25, 75, 99.5])
        print(f"  {name}  {lo:+7.3f} .. {hi:+7.3f} m   "
              f"interquartile {q1:+7.3f} .. {q3:+7.3f}")
    print("\nWorld Z is the reference camera's optical axis, so the Z column is "
          "distance\nfrom that camera and should bracket the deck plane above. "
          "The DECK is the plane\nwith the MOST INLIERS at the near end, not "
          "the most distant one.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Segment parcel top faces in a capture written by "
                    "da3_stream.py and extract the pick data.",
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
                    help="list the largest planes in the cloud with their "
                         "distance from the reference camera, and the extent "
                         "of the points, then exit")
    ap.add_argument("--probe-planes", type=int, default=6)
    add_arguments(ap)
    args = ap.parse_args()

    try:
        from da3_fuse import points_camera, to_world
    except ImportError as exc:
        raise SystemExit(f"cannot import da3_fuse.py: {exc}")

    p = params_from_args(args)
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
        if proc.exists():
            images.append(cv2.cvtColor(cv2.imread(str(proc)), cv2.COLOR_BGR2RGB))
        else:
            images.append(np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8))

    if args.probe:
        return probe(np.concatenate(pts_list), p, args.probe_planes)

    centres = [camera_centre(E) for E in Es]
    t0 = time.perf_counter()
    result = segment_boxes(np.concatenate(pts_list),
                           np.concatenate(src_list), names, p,
                           cam_centres=centres)
    out_dir = args.out_dir or (cap / "segmentation")
    written, _ = write_results(result, out_dir, capture_index=cap.name,
                               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               images=images, K_all=Ks, E_all=Es, p=p)
    print(summarise(result))
    for w in result.get("warnings", []):
        print(f"[warn ] {w}")
    print(f"timings ms: {result.get('timings_ms')}  "
          f"wall {(time.perf_counter() - t0) * 1e3:.0f} ms")
    for k, v in written.items():
        print(f"  {k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())