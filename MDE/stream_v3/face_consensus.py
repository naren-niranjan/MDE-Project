#!/usr/bin/env python3
"""
face_consensus.py

Per-view reconciliation of parcel top faces before the combined plane fit.

The problem
-----------
A parcel top seen by two cameras that disagree in depth produces two parallel
sheets. Fitting one RANSAC plane across both either takes the majority view
and discards the other as outliers, or splits the difference and tilts the
plane. Neither outcome is reported, and neither is a measurement.

Worse, the disagreement hides itself. box_segment.fit_top_face takes the band
h >= h_top - top_band with top_band = 30 mm. An inter-view offset larger than
that puts the second view outside the band entirely, so the face is fitted to
one camera and the record claims a clean single-plane fit.

What this module does
---------------------
1.  Fits the top face independently in every camera that contributes enough
    points, each against its OWN height percentile, so a view offset by more
    than top_band still produces a face rather than vanishing.

2.  Groups those per-view faces by plan-view CONTAINMENT. Views whose
    footprints coincide are looking at the same physical surface at different
    depths, which is a bias to be reconciled. Views whose footprints do not
    coincide are looking at different surfaces, which is a stack to be peeled,
    not averaged.

3.  Weights each view by n_points * cos(incidence) * (1 - comb_fraction) and
    takes the weighted MEAN offset as the consensus, excluding outliers first.

4.  Shifts each view's points along the conveyor normal onto that consensus,
    which is a local anchoring on this parcel rather than a further global
    affine. A view needing a shift larger than max_view_shift is dropped
    instead: a displacement that large is not a bias, it is a different
    surface.

5.  Reports what it could not fix. A face carried by one grazing view has no
    corroboration and no correctable bias, only an uncertainty, and that
    uncertainty is quoted rather than suppressed.

WHAT THIS VERSION CHANGES, AND WHY IT MATTERS FOR ACCURACY
==========================================================

1  THE CONSENSUS WAS A WEIGHTED MEDIAN, WHICH ON TWO VIEWS IS SELECTION.

   A weighted median over n samples returns one of the samples. With two views
   it returns whichever carries more weight, EXACTLY, and the other view's
   evidence is discarded entirely; the reported height is then that camera's
   depth including all of that camera's bias. Measured on two views 44 mm
   apart at equal weight, the median returns 0 mm and the mean 22 mm: the whole
   disagreement.

   Robustness was the right instinct but a median over two or three points is
   not robust, it is arbitrary. The consensus is now a weighted MEAN, with
   outliers removed first by an explicit iteration: compute the mean, drop any
   view beyond max_view_shift, recompute over what remains, repeat until
   stable. That uses every retained view's evidence while still refusing a
   view that is looking at a different surface.

   The old behaviour is available as --consensus median for comparison, via
   SegParams.consensus_estimator.

2  GROUPING COMPARED ONLY AGAINST THE HEAVIEST MEMBER, SO A BRIDGING VIEW
   COULD BE LOST.

   Each view was tested against g[0] alone. Three views of one face, one seeing
   all of it and two seeing different halves, group correctly only because the
   full view anchors them. With no full view present the two halves are tested
   against each other, overlap near zero, and become two single-view groups:
   the parcel is reported at half its footprint with an inter-view spread of
   0.0 mm, which on a multi-camera parcel is zero by construction rather than
   by agreement. That is the exact failure this module was written to fix,
   arriving by a different route, and on a rig whose single-view fraction runs
   67 to 91 per cent it is not hypothetical.

   A view now joins a group if it clears the overlap test against ANY member,
   so membership is transitive and two halves are bridged by anything that
   overlaps both.

3  THE SINGLE-VIEW UNCERTAINTY WAS A LOWER BOUND REPORTED AS A VALUE.

   It was |deck_bias| + deck_rms: what that camera is known to be wrong by AT
   THE DECK. On this rig the cameras agree at the deck to about 2 mm and
   disagree on a parcel top 260 mm above it by tens of millimetres, and the
   depth field has since been measured to compress relief by about a quarter,
   so deck bias does not predict face bias at all. The figure understates by
   roughly an order of magnitude in exactly the case it exists for.

   It is now named height_uncertainty_lower_bound_mm, and where other faces in
   the same capture have supplied a measured ratio between face bias and deck
   bias, that ratio scales it and the scaled figure is reported alongside.

Why containment and not IoU
---------------------------
Intersection over UNION cannot separate the two cases it is being asked about.
A camera that sees half a parcel top produces a footprint of half the area;
even with perfect containment inside the full view's footprint its IoU is
0.50, sitting exactly on the old threshold. Intersection over the SMALLER
footprint asks the question actually intended: is one view looking at part of
what the other is looking at. A minimum area ratio keeps a sliver from being
admitted on containment alone.

What it does not do
-------------------
It does not recover geometry the cameras did not see. A parcel visible to one
camera at 75 degrees incidence stays a single-view measurement combed at the
depth quantisation of that view. The module makes that fact appear in the
record instead of in the point cloud.

Parameters are read with getattr so the module can be dropped in and exercised
before SegParams carries the new fields. Defaults below are the fallbacks.

Keep beside box_segment.py.
"""

from __future__ import annotations

import numpy as np


DEFAULTS = {
    "face_per_view": True,
    "min_view_face_points": 60,
    "max_face_incidence": 65.0,
    "max_view_shift": 0.060,
    # Intersection over the SMALLER footprint above which two views are taken
    # to be seeing the same surface.
    "view_agree_overlap": 0.65,
    # A view whose footprint is smaller than this fraction of the dominant
    # one's area is not admitted to the group even if it is fully contained.
    "min_view_area_frac": 0.10,
    "comb_warn": 0.35,
    "view_disagree_warn": 0.010,
    # "mean" uses every retained view's evidence; "median" restores the old
    # behaviour, which on two views returns the heavier view exactly.
    "consensus_estimator": "mean",
    "consensus_max_iters": 4,
}

# Every return from consensus_shift carries these keys, whether it succeeded or
# not. box_segment reads views_used and single_view on both paths, and a
# failure return that omitted them let a parcel through the min-views gate
# unflagged: len(vc.get("views_used", [])) came back 0, and the guard was
# written as 0 < n_views < required.
_EMPTY = {
    "ok": False,
    "reason": None,
    "views": [],
    "skipped": [],
    "views_used": [],
    "views_used_names": [],
    "views_dropped": [],
    "shifts": {},
    "consensus_offset_m": None,
    "inter_view_offset_mm": None,
    "inter_view_tilt_deg": None,
    "max_shift_mm": None,
    "worst_incidence_deg": None,
    "worst_comb_fraction": None,
    "widest_view": None,
    "widest_view_area_m2": None,
    "view_area_ratio": None,
    "single_view": False,
    "height_uncertainty_mm": None,
    "height_uncertainty_lower_bound_mm": None,
    "uncertainty_is_lower_bound": False,
    "bias_ratio_applied": None,
    "consensus_estimator": None,
    "consensus_iterations": None,
}


def _q(p, key):
    return getattr(p, key, DEFAULTS[key])


def _fail(reason, **kw):
    """A failure return carrying the full key set."""
    out = dict(_EMPTY)
    out["reason"] = reason
    out.update(kw)
    return out


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def weighted_median(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


def weighted_mean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    if float(w.sum()) <= 0:
        return float(v.mean())
    return float(np.average(v, weights=w))


def robust_consensus(offsets, weights, max_shift, estimator="mean",
                     max_iters=4):
    """Consensus offset, with views beyond max_shift excluded and refitted.

    The order matters. Computing a consensus over every view and only then
    testing which views are too far leaves the outlier's pull inside the
    number the test is made against, so a view far enough to be excluded has
    already dragged the consensus toward itself. Excluding and recomputing
    until the membership is stable removes that.

    Returns (consensus, keep_mask, n_iterations).
    """
    v = np.asarray(offsets, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.ones(len(v), dtype=bool)
    est = weighted_median if estimator == "median" else weighted_mean

    c = est(v, w)
    for it in range(1, max(1, int(max_iters)) + 1):
        new_keep = np.abs(v - c) <= max_shift
        if not new_keep.any():
            # Everything is far from the consensus, which means the group is
            # not one surface. Keep the heaviest view and let the caller see a
            # single-view result rather than an averaged fiction.
            new_keep = np.zeros(len(v), dtype=bool)
            new_keep[int(np.argmax(w))] = True
        if np.array_equal(new_keep, keep) and it > 1:
            return c, keep, it
        keep = new_keep
        c = est(v[keep], w[keep])
    return c, keep, max(1, int(max_iters))


def incidence_deg(normal, point, cam_centre) -> float:
    """Angle between the surface normal and the ray back to the camera.

    Zero is face-on. Above about 65 degrees the surface is sampled by rays
    separated by more than the depth quantum and the reconstruction combs.
    """
    r = np.asarray(cam_centre, dtype=float) - np.asarray(point, dtype=float)
    nr = float(np.linalg.norm(r))
    if nr < 1e-9:
        return 90.0
    c = abs(float(np.asarray(normal, dtype=float) @ (r / nr)))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def comb_fraction(pts, normal, cam_centre, cell) -> float:
    """How much more of the along-ray extent is empty than the across-ray one.

    A comb is periodic in the direction the camera ray projects into the
    surface and uniform across it, so the difference in empty-bin fraction
    between those two directions isolates it from ordinary sparsity.
    """
    from box_segment import basis_from_normal

    pts = np.asarray(pts, dtype=float)
    if len(pts) < 12 or cam_centre is None:
        return 0.0
    n = np.asarray(normal, dtype=float)
    centre = pts.mean(axis=0)
    r = np.asarray(cam_centre, dtype=float) - centre
    in_plane = r - n * float(r @ n)
    if float(np.linalg.norm(in_plane)) < 1e-6:
        return 0.0                      # viewed square on: no comb direction
    ex, ey = basis_from_normal(n, in_plane)
    rel = pts - centre

    def empty_frac(t):
        span = float(t.max() - t.min())
        if span < 3 * cell:
            return 0.0
        bins = max(4, int(span / cell))
        counts, _ = np.histogram(t, bins=bins)
        return float(np.count_nonzero(counts == 0) / bins)

    return float(np.clip(empty_frac(rel @ ex) - empty_frac(rel @ ey), 0.0, 1.0))


def _plan_uv(corners, ex, ey):
    c = np.asarray(corners, dtype=float)
    return np.stack([c @ ex, c @ ey], axis=1)


def _plan_area(uv) -> float:
    """Area of a plan-view quadrilateral, in square metres."""
    import cv2
    a = (np.asarray(uv, dtype=float) * 1000.0).astype(np.float32)
    return float(abs(cv2.contourArea(a))) / 1e6


# --------------------------------------------------------------------------
# per-view faces
# --------------------------------------------------------------------------

def per_view_faces(pts, src, n_conv, d_conv, cam_names, cam_centres, p,
                   cx_axis, cy_axis, deck_view=None):
    """Fit the top face separately in each contributing camera.

    Each view uses its own height percentile, which is the point: a view
    offset by more than top_band from the majority still yields a face here,
    where the combined band would have discarded it without comment.
    """
    from box_segment import plane_ransac, rect_on_plane

    pts = np.asarray(pts, dtype=float)
    src = np.asarray(src, dtype=np.int32)
    h = pts @ n_conv + d_conv
    min_pts = int(_q(p, "min_view_face_points"))
    cell = max(float(p.voxel) * 1.5, 0.008)

    views, skipped = [], []
    for i, name in enumerate(cam_names):
        sel = np.flatnonzero(src == i)
        if len(sel) < min_pts:
            if len(sel):
                skipped.append({"camera": name, "n_points": int(len(sel)),
                                "reason": f"only {len(sel)} points, below "
                                          f"{min_pts}"})
            continue

        hv = h[sel]
        h_top = float(np.percentile(hv, p.top_percentile))
        band = sel[hv >= h_top - p.top_band]
        if len(band) < min_pts:
            skipped.append({"camera": name, "n_points": int(len(band)),
                            "reason": "top band too thin in this view"})
            continue

        Q = pts[band]
        n_f, d_f, inl = plane_ransac(Q, p.face_thresh, p.plane_iter)
        if n_f is None or len(inl) < min_pts:
            skipped.append({"camera": name, "n_points": int(len(band)),
                            "reason": "plane fit failed in this view"})
            continue
        if float(n_f @ n_conv) < 0:
            n_f, d_f = -n_f, -d_f

        tilt = float(np.degrees(np.arccos(
            np.clip(float(n_f @ n_conv), -1.0, 1.0))))
        if tilt > p.max_tilt_deg:
            skipped.append({"camera": name, "n_points": int(len(inl)),
                            "reason": f"this view's face is tilted "
                                      f"{tilt:.1f} deg from the conveyor"})
            continue

        face = Q[inl]
        centre = face.mean(axis=0)
        offset = float(centre @ n_conv + d_conv)
        resid = face @ n_f + d_f

        rect, why = rect_on_plane(face, n_f, cx_axis, p)
        if rect is None:
            skipped.append({"camera": name, "n_points": int(len(inl)),
                            "reason": f"footprint fit failed: {why}"})
            continue

        cc = cam_centres[i] if cam_centres is not None else None
        inc = incidence_deg(n_f, centre, cc) if cc is not None else float("nan")
        comb = comb_fraction(face, n_f, cc, cell) if cc is not None else 0.0

        w = float(len(inl))
        if np.isfinite(inc):
            w *= max(np.cos(np.radians(min(inc, 89.0))), 0.05)
        w *= max(1.0 - comb, 0.05)

        uv = _plan_uv(rect["corners"], cx_axis, cy_axis)
        deck = (deck_view or {}).get(name) or {}
        views.append({
            "camera": name,
            "index": int(i),
            "n_points": int(len(inl)),
            "band_points": int(len(band)),
            "offset_m": offset,
            "height_mm": round(offset * 1e3, 2),
            "tilt_deg": round(tilt, 3),
            "incidence_deg": (round(float(inc), 2)
                              if np.isfinite(inc) else None),
            "comb_fraction": round(float(comb), 3),
            "residual_rms_mm": round(
                float(np.sqrt(np.mean(resid ** 2))) * 1e3, 2),
            "length_mm": round(rect["length_m"] * 1e3, 1),
            "width_mm": round(rect["width_m"] * 1e3, 1),
            "area_m2": round(float(rect["area_m2"]), 5),
            "coverage": round(rect["coverage"], 3),
            "deck_bias_mm": deck.get("median_mm"),
            "deck_rms_mm": deck.get("rms_mm"),
            "weight": w,
            "_plan_uv": uv,
            "_plan_area": _plan_area(uv),
        })

    return views, skipped


def group_by_footprint(views, p):
    """Split views into surface groups as CONNECTED COMPONENTS of plan overlap.

    Two views of one parcel top overlap in plan however far apart they sit in
    depth, and they do so even when one of them sees only part of the face.
    Two views of different surfaces in a stack do not overlap. Containment,
    not intersection over union, is therefore the test that separates a bias
    from a stack.

    Grouping is a connected-component problem, not a greedy assignment. The
    previous version walked the views in weight order and stopped at the first
    group that accepted, which makes membership depend on arrival order: with
    three views of one face, two seeing opposite halves and a third bridging
    them, the two halves are compared to each other first, score below the
    threshold, and become separate groups; the bridge then joins whichever it
    met first and the other half is reported as its own parcel at half the
    footprint with an inter-view spread of 0.0 mm. Verified on synthetic
    footprints: greedy returns [left, top] and [right]; components return one
    group. On a rig whose single-view fraction runs 67 to 91 per cent, a face
    that no single camera sees in full is the normal case, not the exception.

    The area-fraction test is applied afterwards, against the heaviest member
    of the component the view landed in, so a sliver cannot vote on the
    consensus even though it is genuinely contained.
    """
    from box_segment import rect_overlap_min

    thr = float(_q(p, "view_agree_overlap"))
    min_frac = float(_q(p, "min_view_area_frac"))
    n = len(views)
    if n == 0:
        return []

    # ---- union-find over the pairwise overlap graph ---------------------
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    best_overlap = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            ov = rect_overlap_min(views[i]["_plan_uv"], views[j]["_plan_uv"])
            best_overlap[i] = max(best_overlap[i], ov)
            best_overlap[j] = max(best_overlap[j], ov)
            if ov >= thr:
                union(i, j)

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    groups, evicted = [], []
    for members in comps.values():
        grp = [views[i] for i in members]
        anchor = max(grp, key=lambda x: x["weight"])
        kept = []
        for v in grp:
            frac = v["_plan_area"] / max(anchor["_plan_area"], 1e-9)
            v["group_overlap"] = round(float(best_overlap[views.index(v)]), 3)
            if v is not anchor and frac < min_frac:
                v["_group_note"] = (
                    f"its footprint is {frac * 100:.0f} per cent of "
                    f"{anchor['camera']}'s, below the {min_frac * 100:.0f} "
                    f"per cent needed to contribute to the consensus")
                evicted.append(v)
                continue
            kept.append(v)
        if kept:
            groups.append(kept)

    for v in evicted:
        groups.append([v])
    for v in views:
        if "group_overlap" not in v:
            v["group_overlap"] = 0.0
        if len(comps) > 1 and "_group_note" not in v:
            v["_group_note"] = (
                f"its footprint overlaps the others by at most "
                f"{v['group_overlap'] * 100:.0f} per cent of its own area, "
                f"below the {thr * 100:.0f} per cent needed, so it is a "
                f"different surface in the same cluster")
    return groups


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------

def consensus_shift(pts, src, n_conv, d_conv, cam_names, cam_centres, p,
                    cx_axis, cy_axis, deck_view=None, bias_ratio=None) -> dict:
    """Reconcile the per-view top faces of one cluster level.

    Returns the shift to apply to each view's points along the conveyor normal,
    the views to exclude, and the record of what the views disagreed about.

    bias_ratio, when supplied by the caller, is the measured ratio between a
    camera's bias at a parcel face and its bias at the deck, taken over other
    faces in the same capture. It is used only to scale the single-view
    uncertainty, which is otherwise a lower bound taken at the deck and known
    to understate the face by about an order of magnitude on this rig.
    """
    views, skipped = per_view_faces(pts, src, n_conv, d_conv, cam_names,
                                    cam_centres, p, cx_axis, cy_axis,
                                    deck_view)
    if not views:
        return _fail("no view produced a usable top face", skipped=skipped)

    groups = group_by_footprint(views, p)
    group = max(groups, key=lambda g: sum(v["weight"] for v in g))
    other = [v for g in groups if g is not group for v in g]

    max_shift = float(_q(p, "max_view_shift"))
    max_inc = float(_q(p, "max_face_incidence"))
    estimator = str(_q(p, "consensus_estimator"))

    offsets = np.array([v["offset_m"] for v in group], float)
    weights = np.array([v["weight"] for v in group], float)
    consensus, keep_mask, n_iter = robust_consensus(
        offsets, weights, max_shift, estimator=estimator,
        max_iters=int(_q(p, "consensus_max_iters")))

    shifts, used, dropped = {}, [], []
    for k, v in enumerate(group):
        s = consensus - v["offset_m"]
        if not keep_mask[k]:
            dropped.append({"camera": v["camera"],
                            "height_mm": v["height_mm"],
                            "reason": f"sits {abs(s) * 1e3:.0f} mm from the "
                                      f"consensus, beyond the "
                                      f"{max_shift * 1e3:.0f} mm a bias can "
                                      f"account for, so it is a different "
                                      f"surface rather than the same one "
                                      f"displaced"})
            continue
        if (int(keep_mask.sum()) > 1 and v["incidence_deg"] is not None
                and v["incidence_deg"] > max_inc):
            dropped.append({"camera": v["camera"],
                            "height_mm": v["height_mm"],
                            "reason": f"sees this face at "
                                      f"{v['incidence_deg']:.0f} deg, above "
                                      f"{max_inc:.0f} deg, and other views "
                                      f"cover it"})
            continue
        shifts[v["index"]] = float(s)
        used.append(v)

    for v in other:
        dropped.append({"camera": v["camera"], "height_mm": v["height_mm"],
                        "reason": v.get("_group_note",
                                        "its footprint does not coincide with "
                                        "the dominant one, so it is a "
                                        "different surface in the same "
                                        "cluster"),
                        "plan_overlap": v.get("group_overlap")})

    if not used:
        return _fail("every view was excluded",
                     views=[_strip(v) for v in views], skipped=skipped,
                     views_dropped=dropped)

    # An incidence exclusion can leave the consensus resting on views that no
    # longer include the one it was computed from, so it is recomputed over
    # exactly the retained set and the shifts follow.
    if len(used) != int(keep_mask.sum()):
        est = weighted_median if estimator == "median" else weighted_mean
        consensus = est([v["offset_m"] for v in used],
                        [v["weight"] for v in used])
        shifts = {v["index"]: float(consensus - v["offset_m"]) for v in used}

    raw = [v["offset_m"] for v in used]
    spread_mm = (max(raw) - min(raw)) * 1e3
    tilts = [v["tilt_deg"] for v in used]
    incs = [v["incidence_deg"] for v in used if v["incidence_deg"] is not None]
    combs = [v["comb_fraction"] for v in used]

    # Departure from the consensus at the parcel, against the same camera's
    # departure from the deck plane. Equal means additive and correctable
    # upstream; different means spatially structured and not correctable.
    for v in used:
        at_face = (v["offset_m"] - consensus) * 1e3
        v["bias_at_face_mm"] = round(float(at_face), 2)
        if v.get("deck_bias_mm") is not None:
            v["structured_component_mm"] = round(
                float(at_face - v["deck_bias_mm"]), 2)

    areas = [(v["area_m2"], v["camera"]) for v in used]
    widest = max(used, key=lambda v: v["area_m2"])
    narrowest = min(used, key=lambda v: v["area_m2"])

    single = len(used) == 1
    unc_lower = float("nan")
    unc = float("nan")
    if single:
        # Nothing corroborates this face. The only figure available is what
        # this camera is known to be wrong by AT THE DECK, which is a LOWER
        # BOUND and nothing more: on this rig the cameras agree at the deck to
        # about 2 mm and disagree on a face 260 mm above it by tens of
        # millimetres, and the depth field compresses relief by about a
        # quarter, so deck bias does not predict face bias.
        v = used[0]
        if v.get("deck_bias_mm") is not None:
            unc_lower = abs(v["deck_bias_mm"]) + (v.get("deck_rms_mm") or 0.0)
            unc = (unc_lower * float(bias_ratio)
                   if bias_ratio and np.isfinite(bias_ratio) else unc_lower)
    else:
        # The consensus picked a value out of the disagreement. The truth may
        # be anywhere within it, so half the raw spread is the honest figure.
        unc = 0.5 * spread_mm
        unc_lower = unc

    return {
        "ok": True,
        "reason": None,
        "views": [_strip(v) for v in views],
        "skipped": skipped,
        "views_used": [v["index"] for v in used],
        "views_used_names": [v["camera"] for v in used],
        "views_dropped": dropped,
        "shifts": shifts,
        "consensus_offset_m": float(consensus),
        "consensus_estimator": estimator,
        "consensus_iterations": int(n_iter),
        "inter_view_offset_mm": round(float(spread_mm), 2),
        "inter_view_tilt_deg": round(float(max(tilts) - min(tilts)), 3),
        "max_shift_mm": round(float(max(abs(s) for s in shifts.values())
                                    * 1e3), 2),
        "worst_incidence_deg": round(float(max(incs)), 2) if incs else None,
        "worst_comb_fraction": round(float(max(combs)), 3),
        "widest_view": widest["camera"],
        "widest_view_area_m2": round(float(widest["area_m2"]), 5),
        "view_area_ratio": (round(float(narrowest["area_m2"]
                                        / widest["area_m2"]), 3)
                            if widest["area_m2"] > 0 else None),
        "single_view": bool(single),
        "height_uncertainty_mm": (round(float(unc), 2)
                                  if np.isfinite(unc) else None),
        "height_uncertainty_lower_bound_mm": (round(float(unc_lower), 2)
                                              if np.isfinite(unc_lower)
                                              else None),
        "uncertainty_is_lower_bound": bool(single and not bias_ratio),
        "bias_ratio_applied": (round(float(bias_ratio), 2)
                               if single and bias_ratio else None),
    }


def _strip(v):
    return {k: val for k, val in v.items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def apply_consensus(pts, src, idx, n_conv, cons):
    """Shift each retained view onto the consensus and drop the rest.

    Returns the corrected points and the surviving indices into the parent
    array, so the caller's bookkeeping of which points were consumed stays
    correct.
    """
    src = np.asarray(src, dtype=np.int32)
    keep = np.isin(src, list(cons["views_used"]))
    out = np.asarray(pts, dtype=float).copy()
    for i, s in cons["shifts"].items():
        out[src == i] += np.asarray(n_conv, dtype=float) * float(s)
    return out[keep], np.asarray(idx)[keep], np.flatnonzero(~keep)


def deck_offsets_per_view(heights, sources, cam_names, inliers):
    """Median height of each camera's deck samples about the fitted plane.

    This is the bias each camera carries at deck distance. Comparing it with
    the bias the same camera carries on a parcel top separates an additive
    error, which the online alignment already removes, from a spatially
    structured one, which it cannot.
    """
    out = {}
    h = np.asarray(heights, dtype=float)[inliers]
    s = np.asarray(sources, dtype=np.int32)[inliers]
    for i, name in enumerate(cam_names):
        sel = s == i
        n = int(np.count_nonzero(sel))
        if n < 50:
            continue
        v = h[sel]
        out[name] = {
            "n_points": n,
            "median_mm": round(float(np.median(v)) * 1e3, 2),
            "rms_mm": round(float(np.sqrt(np.mean(v ** 2))) * 1e3, 2),
        }
    return out


def measure_bias_ratio(boxes):
    """Ratio between a camera's bias at a parcel face and its bias at the deck.

    Taken over the multi-view faces already reconciled in this capture, where
    both quantities are known. It is the empirical answer to the question the
    single-view uncertainty needs and cannot ask: given that this camera is
    2 mm out at the deck, how far out is it likely to be on a parcel top.

    Returns None when fewer than three usable pairs exist, because a ratio
    estimated from one or two faces says nothing.
    """
    pairs = []
    for b in boxes or []:
        vc = b.get("view_consensus") or {}
        if not vc.get("ok") or vc.get("single_view"):
            continue
        for v in vc.get("per_view", []) or []:
            face = v.get("bias_at_face_mm")
            deck = v.get("deck_bias_mm")
            if face is None or deck is None or abs(deck) < 0.5:
                continue
            pairs.append(abs(face) / abs(deck))
    if len(pairs) < 3:
        return None
    return float(np.median(pairs))