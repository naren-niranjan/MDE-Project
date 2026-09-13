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

2.  Groups those per-view faces by plan-view overlap. Views whose footprints
    agree are looking at the same physical surface at different depths, which
    is a bias to be reconciled. Views whose footprints do not agree are looking
    at different surfaces, which is a stack to be peeled, not averaged.

3.  Weights each view by n_points * cos(incidence) * (1 - comb_fraction) and
    takes the weighted median offset as the consensus. A grazing view is
    sampled by widely separated rays, so its depth is the least trustworthy
    contribution to a surface it barely sees, and the weight says so.

4.  Shifts each view's points along the conveyor normal onto that consensus,
    which is a local anchoring on this parcel rather than a further global
    affine. A view needing a shift larger than max_view_shift is dropped
    instead: a displacement that large is not a bias, it is a different
    surface.

5.  Reports what it could not fix. A face carried by one grazing view has no
    corroboration and no correctable bias, only an uncertainty, and that
    uncertainty is quoted rather than suppressed.

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
    "view_agree_iou": 0.50,
    "comb_warn": 0.35,
    "view_disagree_warn": 0.010,
}


def _q(p, key):
    return getattr(p, key, DEFAULTS[key])


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
            "coverage": round(rect["coverage"], 3),
            "deck_bias_mm": deck.get("median_mm"),
            "deck_rms_mm": deck.get("rms_mm"),
            "weight": w,
            "_plan_uv": _plan_uv(rect["corners"], cx_axis, cy_axis),
        })

    return views, skipped


def group_by_footprint(views, p):
    """Split views into surface groups by plan-view overlap.

    Two views of one parcel top overlap almost completely in plan however far
    apart they sit in depth. Two views of different surfaces in a stack do not.
    Overlap is therefore the test that separates a bias from a stack, and
    height alone is not.
    """
    from box_segment import rect_iou

    thr = float(_q(p, "view_agree_iou"))
    groups = []
    for v in sorted(views, key=lambda x: -x["weight"]):
        placed = False
        for g in groups:
            if rect_iou(v["_plan_uv"], g[0]["_plan_uv"]) >= thr:
                g.append(v)
                placed = True
                break
        if not placed:
            groups.append([v])
    return groups


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------

def consensus_shift(pts, src, n_conv, d_conv, cam_names, cam_centres, p,
                    cx_axis, cy_axis, deck_view=None) -> dict:
    """Reconcile the per-view top faces of one cluster level.

    Returns the shift to apply to each view's points along the conveyor normal,
    the views to exclude, and the record of what the views disagreed about.
    """
    views, skipped = per_view_faces(pts, src, n_conv, d_conv, cam_names,
                                    cam_centres, p, cx_axis, cy_axis,
                                    deck_view)
    if not views:
        return {"ok": False, "reason": "no view produced a usable top face",
                "views": [], "skipped": skipped, "views_used": [],
                "views_dropped": [], "shifts": {}}

    groups = group_by_footprint(views, p)
    group = max(groups, key=lambda g: sum(v["weight"] for v in g))
    other = [v for g in groups if g is not group for v in g]

    offsets = [v["offset_m"] for v in group]
    weights = [v["weight"] for v in group]
    consensus = weighted_median(offsets, weights)

    max_shift = float(_q(p, "max_view_shift"))
    max_inc = float(_q(p, "max_face_incidence"))

    shifts, used, dropped = {}, [], []
    for v in group:
        s = consensus - v["offset_m"]
        if abs(s) > max_shift:
            dropped.append({"camera": v["camera"],
                            "height_mm": v["height_mm"],
                            "reason": f"sits {abs(s) * 1e3:.0f} mm from the "
                                      f"consensus, beyond the "
                                      f"{max_shift * 1e3:.0f} mm a bias can "
                                      f"account for, so it is a different "
                                      f"surface rather than the same one "
                                      f"displaced"})
            continue
        if (len(group) > 1 and v["incidence_deg"] is not None
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
                        "reason": "its footprint does not overlap the dominant "
                                  "one, so it is a different surface in the "
                                  "same cluster"})

    if not used:
        return {"ok": False, "reason": "every view was excluded",
                "views": [_strip(v) for v in views], "skipped": skipped,
                "views_used": [], "views_dropped": dropped, "shifts": {}}

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

    single = len(used) == 1
    if single:
        # Nothing corroborates this face, so the uncertainty is whatever this
        # camera is known to be wrong by at the deck, which is a lower bound.
        v = used[0]
        if v.get("deck_bias_mm") is not None:
            unc = abs(v["deck_bias_mm"]) + (v.get("deck_rms_mm") or 0.0)
        else:
            unc = float("nan")
    else:
        # The consensus picked one value out of the disagreement. The truth may
        # be anywhere within it, so half the raw spread is the honest figure.
        unc = 0.5 * spread_mm

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
        "inter_view_offset_mm": round(float(spread_mm), 2),
        "inter_view_tilt_deg": round(float(max(tilts) - min(tilts)), 3),
        "max_shift_mm": round(float(max(abs(s) for s in shifts.values()) * 1e3), 2),
        "worst_incidence_deg": round(float(max(incs)), 2) if incs else None,
        "worst_comb_fraction": round(float(max(combs)), 3),
        "single_view": bool(single),
        "height_uncertainty_mm": (round(float(unc), 2)
                                  if np.isfinite(unc) else None),
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