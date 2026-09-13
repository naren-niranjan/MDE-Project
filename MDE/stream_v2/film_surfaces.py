#!/usr/bin/env python3
"""
film_surfaces.py

Top-surface recovery for objects whose top is not a clean plane, and the
geometric tests that separate a real object from an artefact of the depth.

Two failure modes motivate it.

Shrink film, coloured or transparent
------------------------------------
A monocular depth model has no reliable cue on a specular sheet. The returns
on a wrapped parcel come back sparse, holed, laterally displaced where the film
reflects a lamp, and different in every camera because the highlight moves with
the viewpoint. The planar path in box_segment.py then fails in one of three
ways, none of which announces itself: RANSAC locks onto one wrinkle facet and
the footprint comes out at a fraction of the parcel; the holes drop the inlier
count below --seg-min-face-points and the parcel is not reported at all; or
DBSCAN shatters the sparse returns into fragments and each fragment is fitted
as its own small parcel.

Palletised units, wrapped or bare
---------------------------------
A layer of 2.5 l bottles has no top plane. The top is a periodic field of caps
standing 40 to 60 mm above the shoulders, so a plane fit has a residual of
that order by construction and is rejected by --seg-max-residual, which is
correct behaviour against the wrong model. Wrapped, the film drapes over the
caps and the surface becomes a tensioned sheet with the same periodicity
underneath it.

The model used here
-------------------
Instead of a plane and its inliers, the top is represented as a 2.5D upper
envelope on a raster in the conveyor frame: one robust height per cell, taken
as a high percentile of the points that fall in it rather than the maximum, so
a single flyer cannot lift a cell. Small holes are filled by weighted
diffusion, which is what makes a holed film sheet a single surface rather than
a cloud of islands. The footprint is then the outline of that surface, not the
inlier set of a plane, so it survives the holes.

A plane is still fitted, robustly, over the envelope, because the robot needs
an approach direction and a yaw. What changes is that the deviation from it is
reported as relief rather than treated as error, and the surface is classified
so the caller knows which acceptance test applies:

    planar      bare box; the existing planar path is the better estimator
    film        quasi-planar, holed or wrinkled; relief of a few millimetres
                to a few centimetres, low fill
    granular    periodic relief with a detectable pitch; a bottle layer
    rough       non-planar without a periodic structure; a deformed sack, a
                collapsed carton, or depth that has failed outright

For a surface that is not planar the geometric centre of the footprint is not
a pick point: on a bottle layer it lands between four caps, and on a wrinkled
film it lands on a fold. A suction site is therefore searched for explicitly,
by sliding a disc of the cup radius over the envelope and taking the flattest
fully supported position, and the fraction of the top where any such position
exists is reported.

Artefact rejection
------------------
Three tests, all geometric, all using only the camera centres:

    see-through     the deck is visible beneath the detection's own footprint.
                    A solid resting on the belt occludes the deck under it from
                    every camera, so if the deck shows through, nothing solid is
                    there. This is what catches a film ghost floating above the
                    belt.
    edge bleed      the points lie on the plane through a camera centre and the
                    top edge of a taller neighbour, below that neighbour's top.
                    That is the locus of interpolated depth across a
                    discontinuity, so a detection made of such points is the
                    curtain hanging off the box next to it, not an object.
    occlusion       every camera that contributed points to the detection has
                    its line of sight to it blocked by another detection. A
                    camera cannot see through a parcel, so the points it
                    contributed there were never measurements of that surface.

Import this beside box_segment.py, da3_fuse.py and face_consensus.py. It has no
dependency on either and can be exercised on its own arrays.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")


# --------------------------------------------------------------------------
# raster and envelope
# --------------------------------------------------------------------------

def plan_grid(pts, n, d, ex, ey, cell, pad=2, max_cells=12_000_000):
    """Index the points onto a raster of the conveyor plane.

    u and v are the in-plane coordinates and h is the height above the plane,
    so (ex, ey, n) is an orthonormal frame and a world point reconstructs as
    u*ex + v*ey + (h - d)*n.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 8:
        return None
    h = pts @ n + d
    u = pts @ ex
    v = pts @ ey
    cell = float(max(cell, 1e-4))
    u0 = float(u.min()) - pad * cell
    v0 = float(v.min()) - pad * cell
    W = int((float(u.max()) - u0) / cell) + 1 + pad
    H = int((float(v.max()) - v0) / cell) + 1 + pad
    if W < 4 or H < 4 or W * H > max_cells:
        return None
    iu = np.clip(((u - u0) / cell).astype(np.int64), 0, W - 1)
    iv = np.clip(((v - v0) / cell).astype(np.int64), 0, H - 1)
    return {"u0": u0, "v0": v0, "W": W, "H": H, "cell": cell,
            "iu": iu, "iv": iv, "flat": iv * W + iu,
            "h": h, "u": u, "v": v,
            "n": np.asarray(n, float), "d": float(d),
            "ex": np.asarray(ex, float), "ey": np.asarray(ey, float)}


def envelope(grid, percentile=90.0):
    """Upper envelope and point count per cell.

    A percentile rather than the maximum, so one flyer in a cell cannot lift
    the surface by the length of the flyer. With three or four points per cell
    the 90th percentile is the highest ordinary point.
    """
    W, H = grid["W"], grid["H"]
    flat, h = grid["flat"], grid["h"]
    order = np.lexsort((h, flat))
    fs, hs = flat[order], h[order]
    cells, start, cnt = np.unique(fs, return_index=True, return_counts=True)
    q = float(np.clip(percentile, 0.0, 100.0)) / 100.0
    k = np.clip(np.rint(q * (cnt - 1)).astype(np.int64), 0, cnt - 1)
    env = np.full(W * H, np.nan, np.float32)
    env[cells] = hs[start + k]
    count = np.zeros(W * H, np.int32)
    count[cells] = cnt
    return env.reshape(H, W), count.reshape(H, W)


def fill_holes(env, valid, iters=4, min_weight=0.25):
    """Diffuse the envelope into small holes.

    Film returns are holed at the scale of a few cells. Left as holes they
    break the surface into components and the footprint collapses; filled by
    the surrounding heights they stay part of one surface and the fill is
    recorded so the caller can see how much of the top was interpolated.
    """
    filled = np.where(valid, np.nan_to_num(env), 0.0).astype(np.float32)
    m = valid.astype(np.float32).copy()
    kern = (3, 3)
    for _ in range(max(0, int(iters))):
        if m.min() > 0.5:
            break
        num = cv2.blur(filled * m, kern)
        den = cv2.blur(m, kern)
        take = (m < 0.5) & (den > min_weight)
        if not take.any():
            break
        filled[take] = (num[take] / np.maximum(den[take], 1e-6))
        m[take] = 1.0
    return filled, m > 0.5


def largest_component(mask, close_cells=0, min_cells=1):
    """Close the mask, keep its largest component, return it and a count."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if close_cells and close_cells > 0:
        k = int(2 * int(close_cells) + 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n_lab <= 1:
        return None, 0, 0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = lab == biggest
    if int(comp.sum()) < min_cells:
        return None, int(comp.sum()), int(n_lab - 1)
    return comp, int(comp.sum()), int(n_lab - 1)


# --------------------------------------------------------------------------
# surface fitting
# --------------------------------------------------------------------------

def robust_plane_on_cells(env, mask, grid, iters=5):
    """Least-squares plane over the envelope cells, reweighted.

    Reweighting rather than RANSAC: on a bottle layer the caps are a third of
    the cells and RANSAC would happily call the shoulders the surface. A
    reweighted fit settles between them and the relief is then measured about
    it, which is the quantity worth reporting.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 12:
        return None
    u = grid["u0"] + (xs + 0.5) * grid["cell"]
    v = grid["v0"] + (ys + 0.5) * grid["cell"]
    z = env[ys, xs].astype(np.float64)
    A = np.stack([u, v, np.ones_like(u)], axis=1)
    w = np.ones(len(z))
    sol = np.zeros(3)
    for _ in range(max(1, int(iters))):
        sol, *_ = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)
        r = z - A @ sol
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-6
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * s)) ** 2)
    a, b, c = float(sol[0]), float(sol[1]), float(sol[2])
    m = grid["n"] - a * grid["ex"] - b * grid["ey"]
    nn = float(np.linalg.norm(m))
    if nn < 1e-9:
        return None
    resid = z - A @ sol
    res_img = np.full(env.shape, np.nan, np.float32)
    res_img[ys, xs] = resid.astype(np.float32)
    return {"normal": m / nn, "offset": (grid["d"] - c) / nn,
            "coeffs": (a, b, c), "resid": resid, "resid_img": res_img,
            "cells": (ys, xs), "u": u, "v": v}


def periodicity(res_img, mask, cell, pitch_min, pitch_max):
    """Strength and pitch of the strongest periodic component of the relief.

    Normalised autocorrelation over the masked residual. A bottle layer gives a
    clear peak at the bottle pitch; a wrinkled film does not, because folds are
    not periodic. This is what distinguishes the two without needing to know
    what is on the pallet.
    """
    m = (mask & np.isfinite(res_img))
    if int(m.sum()) < 200:
        return 0.0, 0.0
    x = np.where(m, np.nan_to_num(res_img), 0.0).astype(np.float32)
    x = x - (x[m].mean() if m.any() else 0.0) * m
    w = m.astype(np.float32)
    X = np.fft.rfft2(x)
    Wf = np.fft.rfft2(w)
    ac = np.fft.irfft2(X * np.conj(X), s=x.shape)
    nrm = np.fft.irfft2(Wf * np.conj(Wf), s=x.shape)
    ac = ac / np.maximum(nrm, 8.0)
    ac = np.fft.fftshift(ac)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    peak0 = float(ac[cy, cx])
    if peak0 <= 1e-12:
        return 0.0, 0.0
    yy, xx = np.indices(ac.shape)
    rr = np.hypot(yy - cy, xx - cx) * cell
    band = (rr >= pitch_min) & (rr <= pitch_max)
    if not band.any():
        return 0.0, 0.0
    scored = np.where(band, ac, -np.inf)
    i = int(np.argmax(scored))
    best = float(max(0.0, ac.flat[i] / peak0))
    # The strongest peak is often the second harmonic, which would report a
    # bottle layer at twice its true pitch and halve the unit count. The
    # smallest radius that still reaches most of the best peak is the
    # fundamental.
    strong = band & (ac >= 0.85 * ac.flat[i]) & (ac > 0)
    if strong.any():
        return best, float(rr[strong].min())
    return best, float(rr.flat[i])


def classify_surface(resid, fill_ratio, relief, peak_ratio, pitch, p):
    """Name the surface, from the relief and how it is distributed.

    Periodicity alone is not enough to call a top granular. A wrinkled film has
    a correlation length and therefore an autocorrelation peak of its own, and
    calling it granular would hold a parcel that is perfectly pickable. What
    separates a cap field from a fold field is the shape of the height
    distribution: caps are a small fraction of the area standing well above a
    broad base, which is strongly skewed, whereas folds are close to symmetric
    about the mean surface.
    """
    r = np.asarray(resid, float)
    rms = float(np.sqrt(np.mean(r ** 2)))
    s = float(np.std(r))
    skew = float(np.mean(((r - r.mean()) / s) ** 3)) if s > 1e-9 else 0.0
    granular = (peak_ratio >= p.granular_peak_ratio
                and p.granular_pitch_min <= pitch <= p.granular_pitch_max
                and relief >= p.granular_relief
                and abs(skew) >= p.granular_skew)
    if granular:
        return "granular", rms, skew
    if rms <= p.film_planar_rms and fill_ratio >= p.film_dense_fill:
        return "planar", rms, skew
    if relief <= p.film_max_relief:
        return "film", rms, skew
    return "rough", rms, skew


def suction_site(env, mask, grid, radius, tol):
    """Flattest fully supported disc of the cup radius on the envelope.

    Returned in world metres together with the local peak-to-peak height under
    the disc, and the fraction of the surface where such a disc exists at all.
    On a bare bottle layer that fraction is near zero, which is the honest
    answer: there is nowhere on it for a flat cup to seal.
    """
    cell = grid["cell"]
    r_px = int(max(1, round(radius / cell)))
    if r_px * 2 + 1 > min(mask.shape) - 1:
        r_px = max(1, (min(mask.shape) - 2) // 2)
    kern = np.zeros((2 * r_px + 1, 2 * r_px + 1), np.uint8)
    cv2.circle(kern, (r_px, r_px), r_px, 1, -1)
    area = float(kern.sum())

    mf = mask.astype(np.float32)
    cov = cv2.filter2D(mf, -1, kern.astype(np.float32) / area,
                       borderType=cv2.BORDER_CONSTANT)

    # Smoothed first, over about a centimetre, so the measurement is of the
    # shape of the surface and not of the per-point depth noise: a cup seals
    # against the mean surface under it, not against individual samples.
    s_px = int(max(1, round(0.005 / cell)))
    ks = 2 * s_px + 1
    num = cv2.blur(np.where(mask, env, 0.0).astype(np.float32), (ks, ks))
    den = cv2.blur(mf, (ks, ks))
    smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)

    hi = cv2.dilate(np.where(mask, smooth, -1e6).astype(np.float32), kern)
    lo = -cv2.dilate(np.where(mask, -smooth, -1e6).astype(np.float32), kern)
    rng = hi - lo
    ok = mask & (cov > 0.90) & np.isfinite(rng) & (np.abs(rng) < 1.0)
    flat_frac = float((ok & (rng <= tol)).sum()) / float(max(int(mask.sum()), 1))
    if not ok.any():
        return None, 0.0, flat_frac

    ys, xs = np.nonzero(ok)
    cy = float(np.nonzero(mask)[0].mean())
    cx = float(np.nonzero(mask)[1].mean())
    dist = np.hypot(ys - cy, xs - cx) * cell
    score = rng[ys, xs] + 0.02 * dist          # flatness first, centre second
    i = int(np.argmin(score))
    y, x = int(ys[i]), int(xs[i])
    u = grid["u0"] + (x + 0.5) * cell
    v = grid["v0"] + (y + 0.5) * cell
    h = float(env[y, x])
    world = u * grid["ex"] + v * grid["ey"] + (h - grid["d"]) * grid["n"]
    return world, float(rng[y, x]), flat_frac


def detect_units(env, mask, grid, pitch, min_prominence):
    """Count the repeating units on a granular top, such as bottle caps."""
    cell = grid["cell"]
    sep = int(max(1, round(0.45 * pitch / cell)))
    kern = np.ones((2 * sep + 1, 2 * sep + 1), np.uint8)
    e = np.where(mask, env, -1e6).astype(np.float32)
    mx = cv2.dilate(e, kern)
    lo = -cv2.dilate(-np.where(mask, env, -1e6).astype(np.float32), kern)
    peaks = (mask & (e >= mx - 1e-6) & ((e - lo) >= min_prominence))
    n_lab, _, _, cent = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), 8)
    if n_lab <= 1:
        return {"count": 0, "pitch_mm": round(pitch * 1e3, 1),
                "prominence_mm": round(min_prominence * 1e3, 1)}
    return {"count": int(n_lab - 1), "pitch_mm": round(pitch * 1e3, 1),
            "prominence_mm": round(min_prominence * 1e3, 1),
            "centres_uv": [[float(grid["u0"] + c[0] * cell),
                            float(grid["v0"] + c[1] * cell)]
                           for c in cent[1:]]}


def footprint_from_mask(mask, occupied, grid, plane, cx_axis):
    """Outline of the surface, as a rectangle in the world frame.

    Solved on the raster rather than on the points, so a holed film gives the
    same rectangle a solid one would. The extents are plan extents: on a face
    tilted by t they under-report the in-plane size by cos t, which at the 25
    degree limit is four per cent and below the depth scale error anyway.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 12:
        return None
    cell = grid["cell"]
    u = grid["u0"] + (xs + 0.5) * cell
    v = grid["v0"] + (ys + 0.5) * cell
    uv_mm = np.stack([u, v], axis=1).astype(np.float32) * 1000.0
    (_, _), (w_mm, h_mm), angle = cv2.minAreaRect(uv_mm)
    if w_mm < 1.0 or h_mm < 1.0:
        return None

    ex, ey = grid["ex"], grid["ey"]
    a = np.radians(angle)
    au = np.array([np.cos(a), np.sin(a)])
    av = np.array([-np.sin(a), np.cos(a)])
    if w_mm < h_mm:                    # keep the long edge as the first axis
        au, av = av, -au
    du_all = u * au[0] + v * au[1]
    dv_all = u * av[0] + v * av[1]
    u_lo, u_hi = float(du_all.min()), float(du_all.max())
    v_lo, v_hi = float(dv_all.min()), float(dv_all.max())
    du, dv = u_hi - u_lo, v_hi - v_lo
    cu = (u_lo + u_hi) / 2.0
    cv_ = (v_lo + v_hi) / 2.0

    def to_world(pu, pv):
        uu = pu * au[0] + pv * av[0]
        vv = pu * au[1] + pv * av[1]
        a_, b_, c_ = plane["coeffs"]
        hh = a_ * uu + b_ * vv + c_
        return uu * ex + vv * ey + (hh - grid["d"]) * grid["n"]

    centre = to_world(cu, cv_)
    corners = np.array([to_world(cu - du / 2, cv_ - dv / 2),
                        to_world(cu + du / 2, cv_ - dv / 2),
                        to_world(cu + du / 2, cv_ + dv / 2),
                        to_world(cu - du / 2, cv_ + dv / 2)])

    # Axes taken in the fitted plane so the pose is orthonormal about the
    # surface normal rather than about the conveyor normal.
    nrm = np.asarray(plane["normal"], float)
    x_axis = au[0] * ex + au[1] * ey
    x_axis = x_axis - nrm * float(x_axis @ nrm)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(nrm, x_axis)
    if float(x_axis @ cx_axis) < 0:
        x_axis, y_axis = -x_axis, -y_axis
        corners = corners[[2, 3, 0, 1]]

    n_rect = max(1, int(round(du / cell)) * int(round(dv / cell)))
    raw = int((occupied & mask).sum())
    return {"centre": centre, "corners": corners,
            "x_axis": x_axis, "y_axis": y_axis,
            "length_m": float(du), "width_m": float(dv),
            "area_m2": float(du * dv),
            "coverage": float(min(1.0, raw / n_rect)),
            "filled_coverage": float(min(1.0, int(mask.sum()) / n_rect))}


def fit_surface(pts, src, n_conv, d_conv, cx_axis, cy_axis, p,
                cam_names=None, cam_centres=None, _pass=0):
    """Envelope fit of the top surface of one set of points.

    Returns (face, rect, diagnostic) in the shapes box_segment.py already
    consumes, with the surface description carried in face["surface"], or
    (None, None, why) if nothing usable is there.

    Run twice by default. The first pass measures how far apart the cameras are
    on this surface; the second removes that offset before rebuilding the
    envelope. Without it the disagreement between the views is added to the
    relief of the object, so a flat wrapped top reads as a rough one and is
    held for a reason that belongs to the rig rather than to the parcel.
    """
    pts = np.asarray(pts, dtype=float)
    grid = plan_grid(pts, n_conv, d_conv, cx_axis, cy_axis, p.film_cell)
    if grid is None:
        return None, None, "the points do not raster"

    env0, cnt = envelope(grid, p.film_env_percentile)
    occupied = cnt > 0
    close_cells = int(round(p.film_close / grid["cell"]))

    # Only holes that are genuinely enclosed by the surface are filled. A
    # closing would also add a rim of cells around the outside, and once those
    # are given a height by diffusion they enter the footprint: every wrapped
    # parcel then reads a centimetre larger than it is, consistently, which is
    # the kind of error that survives a repeatability test unnoticed.
    bg = (~occupied).astype(np.uint8)
    ff = bg.copy()
    cv2.floodFill(ff, np.zeros((grid["H"] + 2, grid["W"] + 2), np.uint8),
                  (0, 0), 0)
    holes = ff > 0
    env, filled_mask = fill_holes(env0, occupied, p.film_fill_iters)
    valid = filled_mask & (occupied | holes)

    h_all = env[valid]
    if h_all.size < 24:
        return None, None, "too few cells carry a height"
    h_top = float(np.percentile(h_all, p.top_percentile))

    # The band is set by the surface, not fixed. Thirty millimetres is right
    # for a box top and wrong for a bottle layer, where the caps stand a whole
    # band above the shoulders: at a fixed band the caps come back as dozens of
    # disconnected islands and the largest of them is one cap. So the band
    # widens until the top forms a single surface, and then again until it
    # contains the relief that surface actually has.
    band = float(p.top_band)
    cap = float(max(p.film_max_relief, p.top_band))
    comp = None
    last_cells = 0
    for _ in range(6):
        sel = valid & (env >= h_top - band)
        cand, n_cells, _ = largest_component(sel, close_cells, p.film_min_cells)
        last_cells = max(last_cells, n_cells)
        if cand is None:
            if band >= cap - 1e-6:
                return None, None, (f"the largest connected top surface holds "
                                    f"{n_cells} cells, below {p.film_min_cells}")
            band = min(cap, max(band * 2.0, 0.02))
            continue
        # Intersected back with the band. Closing is there to bridge holes
        # inside the surface, and a hole filled by diffusion still passes the
        # band; a cell the closing added outside the surface does not, and
        # admitting it puts the shoulder of the parcel into the top face.
        comp = cand & sel
        if int(comp.sum()) < p.film_min_cells:
            if band >= cap - 1e-6:
                return None, None, (f"the largest connected top surface holds "
                                    f"{int(comp.sum())} cells, below "
                                    f"{p.film_min_cells}")
            band = min(cap, max(band * 2.0, 0.02))
            comp = None
            continue
        r = env[comp]
        relief = float(np.percentile(r, 97) - np.percentile(r, 3))
        want = float(np.clip(1.6 * relief, p.top_band, cap))
        if want <= band * 1.05 or band >= cap - 1e-6:
            break
        band = want
    if comp is None or int(comp.sum()) < p.film_min_cells:
        return None, None, (f"the largest connected top surface holds "
                            f"{last_cells} cells, below {p.film_min_cells}")

    plane = robust_plane_on_cells(env, comp, grid)
    if plane is None:
        return None, None, "the envelope plane fit failed"
    nrm = plane["normal"]
    off = plane["offset"]
    if float(nrm @ n_conv) < 0:
        nrm, off = -nrm, -off
        plane["normal"], plane["offset"] = nrm, off
    tilt = float(np.degrees(np.arccos(np.clip(float(nrm @ n_conv), -1.0, 1.0))))
    if tilt > p.max_tilt_deg:
        return None, None, f"envelope tilted {tilt:.1f} deg from the conveyor"

    resid = plane["resid"]
    relief = float(np.percentile(resid, 97) - np.percentile(resid, 3))
    fill_ratio = float((occupied & comp).sum()) / float(max(int(comp.sum()), 1))
    peak_ratio, pitch = periodicity(plane["resid_img"], comp, grid["cell"],
                                    p.granular_pitch_min, p.granular_pitch_max)
    kind, rms, skew = classify_surface(resid, fill_ratio, relief, peak_ratio,
                                       pitch, p)

    rect = footprint_from_mask(comp, occupied, grid, plane, cx_axis)
    if rect is None:
        return None, None, "degenerate envelope rectangle"

    pick, pick_flat, flat_frac = suction_site(env, comp, grid,
                                              p.pick_radius, p.suction_tol)
    units = None
    if kind == "granular":
        units = detect_units(env, comp, grid, pitch, max(0.5 * relief, 0.008))

    # The points that belong to this surface: inside the component and within a
    # shell below its own envelope, so the side walls stay out of the fit but
    # the whole of a wrinkled top stays in.
    shell = max(2.5 * grid["cell"], relief + p.film_cell)
    in_comp = comp[grid["iv"], grid["iu"]]
    e_at = env[grid["iv"], grid["iu"]]
    member = in_comp & np.isfinite(e_at) & (grid["h"] >= e_at - shell)
    face_pts = pts[member]

    per_view = {}
    spread_mm = None
    shifts = {}
    if src is not None and cam_names is not None:
        s = np.asarray(src)[member]
        hres = (face_pts @ nrm + off)
        meds, weights = {}, {}
        for i, name in enumerate(cam_names):
            m = s == i
            if int(m.sum()) < p.min_view_face_points:
                continue
            meds[i] = float(np.median(hres[m]))
            weights[i] = int(m.sum())
            per_view[name] = {"points": int(m.sum()),
                              "median_offset_mm": round(meds[i] * 1e3, 2),
                              "robust_spread_mm": round(float(
                                  1.4826 * np.median(np.abs(
                                      hres[m] - meds[i]))) * 1e3, 2)}
        if len(meds) > 1:
            spread_mm = round((max(meds.values()) - min(meds.values())) * 1e3, 2)
            tot = float(sum(weights.values()))
            centre = sum(meds[i] * weights[i] for i in meds) / max(tot, 1.0)
            shifts = {i: (centre - meds[i]) for i in meds
                      if abs(centre - meds[i]) <= p.max_view_shift}

    # Second pass on the reconciled points. Only the offsets that a per-camera
    # depth bias can account for are removed; a view further out than
    # --seg-max-view-shift is looking at a different surface and is left where
    # it is, to be reported rather than absorbed.
    if (_pass == 0 and getattr(p, "view_shift", True) and shifts
            and spread_mm is not None
            and spread_mm > p.view_disagree_warn * 1e3):
        moved = np.asarray(pts, float).copy()
        s_all = np.asarray(src)
        for i, sh in shifts.items():
            m = s_all == i
            if m.any():
                moved[m] = moved[m] + nrm * sh
        f2, r2, why2 = fit_surface(moved, src, n_conv, d_conv, cx_axis,
                                   cy_axis, p, cam_names, cam_centres,
                                   _pass=1)
        if f2 is not None:
            f2["surface"]["view_shift_mm"] = {
                cam_names[i]: round(sh * 1e3, 2) for i, sh in shifts.items()}
            f2["surface"]["inter_view_offset_mm"] = spread_mm
            f2["surface"]["shifted"] = True
            return f2, r2, None

    diag = {
        "class": kind,
        "relief_mm": round(relief * 1e3, 2),
        "plane_rms_mm": round(rms * 1e3, 2),
        "fill_ratio": round(fill_ratio, 3),
        "filled_coverage": round(rect["filled_coverage"], 3),
        "band_mm": round(band * 1e3, 1),
        "cells": int(comp.sum()),
        "cell_mm": round(grid["cell"] * 1e3, 1),
        "periodic_peak": round(peak_ratio, 3),
        "pitch_mm": round(pitch * 1e3, 1),
        "skewness": round(skew, 2),
        "view_shift_mm": {},
        "shifted": False,
        "units": units,
        "pick_point_m": None if pick is None else pick.tolist(),
        "pick_flatness_mm": None if pick is None else round(pick_flat * 1e3, 2),
        "flat_area_fraction": round(flat_frac, 3),
        "suction_tolerance_mm": round(p.suction_tol * 1e3, 1),
        "per_view": per_view,
        "inter_view_offset_mm": spread_mm,
        "envelope_percentile": p.film_env_percentile,
    }

    face = {
        "points": face_pts,
        "normal": nrm,
        "offset": off,
        "tilt_deg": tilt,
        "band_points": int(len(pts)),
        "inliers": int(len(face_pts)),
        "inlier_fraction": float(len(face_pts) / max(len(pts), 1)),
        "residual_rms_m": float(rms),
        "residual_p95_m": float(np.percentile(np.abs(resid), 95)),
        "surface": diag,
    }
    return face, rect, None


# --------------------------------------------------------------------------
# artefact tests
# --------------------------------------------------------------------------

def see_through_fraction(P, heights, corners, centre, x_axis, y_axis,
                         length, width, deck_band, shrink=0.85, cell=0.02):
    """Fraction of the detection's own footprint where the deck is visible.

    A solid standing on the belt hides the deck beneath it from every camera.
    If deck-height points come back from inside the footprint, either the
    detection is floating, which nothing on a conveyor does, or the footprint
    belongs to nothing at all. The rectangle is shrunk first so that points
    from the deck just outside a slightly over-sized footprint do not count.
    """
    rel = np.asarray(P) - np.asarray(centre)
    u = rel @ np.asarray(x_axis)
    v = rel @ np.asarray(y_axis)
    lu, lv = length * shrink / 2.0, width * shrink / 2.0
    if lu <= cell or lv <= cell:
        return 0.0, 0
    inside = (np.abs(u) <= lu) & (np.abs(v) <= lv)
    if int(inside.sum()) < 20:
        return 0.0, 0
    gu = ((u[inside] + lu) / cell).astype(np.int64)
    gv = ((v[inside] + lv) / cell).astype(np.int64)
    nv = int(2 * lv / cell) + 1
    key = gu * nv + gv
    deck = np.abs(np.asarray(heights)[inside]) <= deck_band
    occupied = np.unique(key)
    through = np.unique(key[deck])
    if not len(occupied):
        return 0.0, 0
    return float(len(through) / len(occupied)), int(deck.sum())


def bleed_fraction(pts, src, cam_index, cam_centres, others, h_self,
                   n_conv, d_conv, tol, min_drop=0.05):
    """Fraction of a detection's points that lie on a neighbour's sight edge.

    Depth interpolated across the silhouette of a taller object lands on the
    plane through the camera centre and that object's top edge, below the top
    and outside the object in plan. Points satisfying all three are not
    measurements of a surface; they are the ramp the network drew between the
    box top and the deck behind it.
    """
    pts = np.asarray(pts, float)
    src = np.asarray(src)
    total, bled = 0, 0
    per_cam = {}
    for ci in cam_index:
        C = np.asarray(cam_centres[ci], float)
        m = src == ci
        q = pts[m]
        if len(q) < 12:
            continue
        h_q = q @ n_conv + d_conv
        flag = np.zeros(len(q), bool)
        for o in others:
            if o["top_h"] - h_self < min_drop:
                continue
            corners = np.asarray(o["corners"], float)
            for k in range(4):
                A, B = corners[k], corners[(k + 1) % 4]
                mv = np.cross(B - A, C - A)
                nn = float(np.linalg.norm(mv))
                if nn < 1e-9:
                    continue
                mv = mv / nn
                near = np.abs((q - A) @ mv) < tol
                if not near.any():
                    continue
                # only points that lie beyond that edge as seen from C
                M = np.stack([A - C, B - C], axis=1)
                ab = np.linalg.lstsq(M, (q - C).T, rcond=None)[0]
                beyond = ((ab[0] > -0.08) & (ab[1] > -0.08)
                          & (ab[0] + ab[1] > 0.85))
                flag |= near & beyond & (h_q < o["top_h"] - 0.005)
        per_cam[int(ci)] = round(float(flag.mean()), 3)
        total += len(q)
        bled += int(flag.sum())
    if total == 0:
        return 0.0, per_cam
    return float(bled / total), per_cam


def _segment_hits_prism(C, X, prism, n_conv, d_conv, eps=0.02):
    """Does the open segment C to X pass through the prism?

    The prism is the footprint extruded from the deck to the top face, which is
    a convex body, so the segment is clipped against its six half-spaces.
    """
    centre = prism["centre"]
    ax, ay = prism["x_axis"], prism["y_axis"]
    hl, hw = prism["length"] / 2.0, prism["width"] / 2.0
    planes = [(ax, float(ax @ centre) + hl, 1), (ax, float(ax @ centre) - hl, -1),
              (ay, float(ay @ centre) + hw, 1), (ay, float(ay @ centre) - hw, -1)]
    t0, t1 = eps, 1.0 - eps
    D = X - C
    for a, b, sgn in planes:
        # keep sgn * (a.x - b) <= 0
        num = sgn * (float(a @ C) - b)
        den = sgn * float(a @ D)
        if abs(den) < 1e-12:
            if num > 0:
                return False
            continue
        t = -num / den
        if den > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
        if t0 > t1:
            return False
    # height slab: 0 <= h <= top
    hC = float(C @ n_conv + d_conv)
    hD = float(D @ n_conv)
    for lo, hi in ((0.005, prism["top_h"] - 0.005),):
        if abs(hD) < 1e-12:
            if hC < lo or hC > hi:
                return False
            continue
        ta, tb = (lo - hC) / hD, (hi - hC) / hD
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return t1 - t0 > 1e-4


def blocked_views(samples, cam_index, cam_centres, prisms, n_conv, d_conv,
                  frac=0.6):
    """Cameras whose line of sight to the detection is blocked by another one."""
    out = []
    for ci in cam_index:
        C = np.asarray(cam_centres[ci], float)
        hit = 0
        for X in samples:
            if any(_segment_hits_prism(C, np.asarray(X, float), pr,
                                       n_conv, d_conv) for pr in prisms):
                hit += 1
        if hit >= frac * len(samples):
            out.append(int(ci))
    return out