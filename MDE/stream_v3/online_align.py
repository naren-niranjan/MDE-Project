#!/usr/bin/env python3
"""
online_align.py

Per-frame estimation of the inter-camera depth correction, using the scene's
own surfaces as the reference.

What changed, and why
---------------------
The previous version fitted ONE plane, the conveyor deck, and solved ONE
parameter per camera, the additive offset b. That was justified by a
measurement taken through the 8 mm lens: the top camera's bias was -57.56 mm at
4.03 m and -57.62 mm at 3.19 m, so across 840 mm of distance it moved by
0.06 mm and was additive to within the noise.

That justification does not survive the 12 mm lens, and it never explained the
symptom that matters. The four cameras agree on the deck to 1.6 mm and disagree
on a parcel top 270 mm above it by up to 44 mm. A correction with one free
parameter, solved on the deck, is exactly zero at the deck by construction and
says nothing at all about 270 mm above it. Every millimetre of that 44 mm
survives into the parcel heights, which is why the box layer is not flat and
why the heights do not match a tape.

There is also a second contributor the offset cannot touch. The solved focal
lengths on this rig span 0.77 per cent, from 3479.6 px on centre and right to
3506.4 px on top. Under the intrinsic prior a camera receives depth scaled by
roughly that fraction, which is 23 mm at 3 m. An offset removes it at the
distance it was solved at and leaves it everywhere else, growing linearly with
the distance from that plane.

So this version fits TWO parallel planes and solves BOTH terms:

    plane 0   the conveyor deck, the largest surface in the reference view
    plane 1   the dominant surface standing above it, which in a loaded cell is
              the parcel tops

and regresses Z_true = a * Z_pred + b per camera over the samples assigned to
both. The two planes are 100 to 400 mm apart rather than the 825 mm the floor
gave, which is less leverage, but they BRACKET the volume the robot picks from
instead of sitting a metre beneath it. A correction interpolating over 270 mm
beats one extrapolating over 1.1 m even when the second has a longer baseline.

When the upper plane is absent, because the belt is empty or the parcels are
too short to clear --upper-min-height, the solve falls back to offset-only with
the scale taken from --load-affine, and says so in the diagnostic. That
fallback is the OLD behaviour, and the record marks it as degraded rather than
letting it pass as a correction.

What it still cannot do
-----------------------
It corrects RELATIVE disagreement between views. It cannot detect a bias common
to all four cameras, because both reference planes are derived from the
reference camera. Absolute anchoring remains a separate ground-truth
calibration, now expressed as the reference planes in ground_truth.json and
solved by layer_align.py.

It also cannot remove a bias that varies across the frame rather than with
depth. layer_align.py measures that residual per image cell; if
spatial_structure_mm stays large after this correction, the remaining term is
not reachable by any per-camera scalar and needs anchoring at control points.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d is required")

from da3_fuse import points_camera, to_world


_RAY_CACHE: dict = {}


def rays_for(shape, K):
    """Unit-z camera-frame ray directions, cached per (shape, intrinsics)."""
    key = (shape, K.tobytes())
    hit = _RAY_CACHE.get(key)
    if hit is None:
        if len(_RAY_CACHE) > 16:
            _RAY_CACHE.clear()
        hit = points_camera(np.ones(shape, np.float64), K)
        _RAY_CACHE[key] = hit
    return hit


def plane_to_camera(normal_w, offset_w, E):
    """World plane n.X = d expressed in the camera frame of a w2c extrinsic."""
    R, t = E[:3, :3], E[:3, 3]
    n_c = R @ np.asarray(normal_w, np.float64)
    d_c = float(offset_w) + float(n_c @ t)
    return n_c, d_c


def refine_plane(points, normal, offset, thresh, iters=2):
    n = np.asarray(normal, np.float64)
    d = float(offset)
    for _ in range(iters):
        sd = points @ n - d
        inl = points[np.abs(sd) <= thresh]
        if len(inl) < 10:
            break
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n_new = vt[-1]
        if np.dot(n_new, n) < 0:
            n_new = -n_new
        n, d = n_new, float(n_new @ c)
    return n, d


def huber_affine(z_pred, z_true, iters=4, k=1.345):
    """Robust least squares for z_true = a * z_pred + b."""
    A = np.stack([z_pred, np.ones_like(z_pred)], axis=1)
    w = np.ones_like(z_pred)
    a, b = 1.0, 0.0
    for _ in range(max(1, iters)):
        sol, *_ = np.linalg.lstsq(A * w[:, None], z_true * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = z_true - (a * z_pred + b)
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if s <= 1e-9:
            break
        u = np.abs(r) / s
        w = np.where(u <= k, 1.0, k / np.maximum(u, 1e-9))
    return a, b


class OnlineAligner:
    """Track the per-camera depth affine frame by frame from two planes.

    Returns, from update(), a dict mapping camera name to (a, b) and a
    diagnostic. The old interface returned only the offsets; callers that still
    want them can read .b, but they should apply .coefficients() instead, or
    the scale term this class exists to solve is thrown away at the point of
    use.
    """

    def __init__(self, names, reference, base_a=None, stride=3,
                 plane_thresh=0.010, assign_tol=0.08, ema=0.3,
                 min_points=2000, max_step=0.150, seed=0,
                 two_plane=True,
                 upper_min_height=0.040, upper_max_height=0.600,
                 upper_min_points=600, upper_bin=0.005, upper_smooth=0.015,
                 min_span=0.080, min_plane_points=400,
                 max_scale_step=0.010, scale_bounds=(0.90, 1.10)):
        if reference not in names:
            raise ValueError(f"reference {reference!r} not among {names}")
        self.names = list(names)
        self.reference = reference
        self.ref_idx = self.names.index(reference)
        self.base_a = dict(base_a) if base_a else {n: 1.0 for n in self.names}
        self.stride = max(1, int(stride))
        self.plane_thresh = plane_thresh
        # Must exceed the inter-camera disagreement being corrected, or a
        # biased view's samples never reach the plane they belong to and the
        # camera is silently left uncorrected. It must also stay below half the
        # gap between the two planes, or the deck claims the parcel tops.
        self.assign_tol = assign_tol
        self.ema = float(np.clip(ema, 0.0, 1.0))
        self.min_points = min_points
        self.max_step = max_step
        self.seed = seed

        self.two_plane = bool(two_plane)
        self.upper_min_height = upper_min_height
        self.upper_max_height = upper_max_height
        self.upper_min_points = upper_min_points
        self.upper_bin = upper_bin
        self.upper_smooth = upper_smooth
        self.min_span = min_span
        self.min_plane_points = min_plane_points
        self.max_scale_step = max_scale_step
        self.scale_bounds = tuple(scale_bounds)

        self.a = {n: float(self.base_a.get(n, 1.0)) for n in self.names}
        self.b = {n: 0.0 for n in self.names}
        self.initialised = False
        self.frames = 0
        self.frames_two_plane = 0
        self.rejected = 0
        self.last = {}

    # ---- accessors -----------------------------------------------------

    def coefficients(self):
        """Per-camera (a, b), which is what apply_affine expects."""
        return {n: (self.a[n], self.b[n]) for n in self.names}

    # ---- internals -----------------------------------------------------

    def _scaled(self, depth, i):
        """The depth this frame's solve operates on.

        Nothing is pre-scaled here any more. The previous version applied the
        stored a before solving b, which made the solved offset conditional on
        a coefficient the aligner was about to replace. The affine is now
        solved end to end against the raw model output.
        """
        return depth[i].astype(np.float64)

    def _upper_plane(self, pts_w, n, d0):
        """The dominant surface standing above the deck, parallel to it.

        Taken as the tallest peak of the height histogram inside the parcel
        band rather than by a second RANSAC. RANSAC would return a plane at
        whatever tilt fits best, and a parcel top that comes back tilted by a
        few degrees because one corner of it is biased is a worse reference
        than a plane held parallel to the deck by construction. What is wanted
        from this plane is its DISTANCE, not its orientation.
        """
        h = pts_w @ n - d0
        band = (h >= self.upper_min_height) & (h <= self.upper_max_height)
        if int(band.sum()) < self.upper_min_points:
            return None, {"reason": f"only {int(band.sum())} reference points "
                                    f"stand between "
                                    f"{self.upper_min_height * 1e3:.0f} and "
                                    f"{self.upper_max_height * 1e3:.0f} mm "
                                    f"above the deck"}
        hv = h[band]
        lo, hi = float(hv.min()), float(hv.max())
        bins = max(8, int((hi - lo) / max(self.upper_bin, 1e-4)))
        counts, edges = np.histogram(hv, bins=bins, range=(lo, hi))
        centres = 0.5 * (edges[:-1] + edges[1:])
        k = max(1, int(round(self.upper_smooth / max(self.upper_bin, 1e-4))))
        if k > 1:
            counts = np.convolve(counts.astype(np.float64),
                                 np.ones(k) / k, mode="same")
        peak = float(centres[int(np.argmax(counts))])
        near = int(np.count_nonzero(np.abs(hv - peak) <= self.plane_thresh * 2))
        if near < self.upper_min_points // 2:
            return None, {"reason": f"the tallest surface above the deck holds "
                                    f"only {near} points at its own height"}
        if peak < self.min_span:
            return None, {"reason": f"the dominant surface above the deck sits "
                                    f"{peak * 1e3:.0f} mm up, below the "
                                    f"{self.min_span * 1e3:.0f} mm of span "
                                    f"needed to separate scale from offset"}
        return d0 + peak, {"height_m": round(peak, 5), "n_points": near}

    def _assign(self, d_i, rays_i, E_i, n, offset, sub):
        """Samples of one camera that belong to one plane, as (pred, target)."""
        n_c, d_c = plane_to_camera(n, offset, E_i)
        zi = d_i[sub]
        vi = rays_i[sub]
        denom = vi @ n_c
        with np.errstate(divide="ignore", invalid="ignore"):
            z_target = d_c / denom
        ok = (np.isfinite(zi) & (zi > 0) & np.isfinite(z_target)
              & (z_target > 0) & (np.abs(denom) > 1e-6))
        near = ok & (np.abs(zi - z_target) <= self.assign_tol)
        return zi[near], z_target[near]

    # ---- the frame solve -----------------------------------------------

    def update(self, depth, K_out, E_out):
        """Return the per-camera (a, b) for this frame, plus diagnostics."""
        t0 = time.perf_counter()
        s = self.stride
        sub = (slice(None, None, s), slice(None, None, s))

        # ---- deck plane, from the reference view alone -------------------
        i0 = self.ref_idx
        d_ref = self._scaled(depth, i0)
        rays_ref = rays_for(d_ref.shape, K_out[i0])
        z = d_ref[sub]
        good = np.isfinite(z) & (z > 0)
        pts_c = rays_ref[sub][good] * z[good][:, None]
        if len(pts_c) < self.min_points:
            self.rejected += 1
            self.last = {"ok": False, "reason": "too few reference points",
                         "n_reference": int(len(pts_c))}
            return self.coefficients(), self.last

        pts_w = to_world(pts_c, E_out[i0])
        o3d.utility.random.seed(self.seed)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_w)
        model, inliers = pcd.segment_plane(self.plane_thresh, 3, 400)
        a_, bb, c_, dd = model
        n = np.array([a_, bb, c_], np.float64)
        nrm = np.linalg.norm(n)
        if nrm < 1e-9 or len(inliers) < self.min_points // 4:
            self.rejected += 1
            self.last = {"ok": False, "reason": "no usable deck plane",
                         "n_inliers": int(len(inliers))}
            return self.coefficients(), self.last
        n, d0 = n / nrm, float(-dd / nrm)
        if d0 < 0:
            n, d0 = -n, -d0
        n, d0 = refine_plane(pts_w, n, d0, self.plane_thresh)

        # ---- the surface standing above it -------------------------------
        upper, upper_note = (None, {"reason": "two-plane solve disabled"})
        if self.two_plane:
            upper, upper_note = self._upper_plane(pts_w, n, d0)
        planes = [d0] if upper is None else [d0, upper]
        span = 0.0 if upper is None else abs(upper - d0)

        # ---- affine per camera against those planes ----------------------
        proposed_a, proposed_b, detail = {}, {}, {}
        for i, name in enumerate(self.names):
            if i == i0:
                proposed_a[name] = 1.0
                proposed_b[name] = 0.0
                detail[name] = {"model": "reference", "n_assigned":
                                int(len(inliers)), "residual_mm": 0.0}
                continue

            d_i = self._scaled(depth, i)
            rays_i = rays_for(d_i.shape, K_out[i])

            per_plane, preds, trues = [], [], []
            for pi, off in enumerate(planes):
                zp, zt = self._assign(d_i, rays_i, E_out[i], n, off, sub)
                per_plane.append({"plane": pi,
                                  "offset_m": round(float(off), 5),
                                  "n": int(len(zp)),
                                  "median_mm": (round(float(np.median(zt - zp))
                                                      * 1e3, 2)
                                                if len(zp) else None)})
                if len(zp):
                    preds.append(zp)
                    trues.append(zt)

            usable = [p for p in per_plane if p["n"] >= self.min_plane_points]
            if not preds:
                detail[name] = {"model": "none", "per_plane": per_plane,
                                "note": "kept previous coefficients"}
                continue

            zp = np.concatenate(preds)
            zt = np.concatenate(trues)

            if len(usable) >= 2 and span >= self.min_span and \
                    len(zp) >= self.min_points:
                a_fit, b_fit = huber_affine(zp, zt)
                lo, hi = self.scale_bounds
                if not (lo <= a_fit <= hi):
                    # A scale this far from unity is a bad plane assignment,
                    # not a camera. Refuse it rather than let it through.
                    a_fit = float(self.base_a.get(name, 1.0))
                    b_fit = float(np.median(zt - a_fit * zp))
                    model_name = "offset_only_scale_rejected"
                else:
                    model_name = "affine"
            else:
                a_fit = float(self.base_a.get(name, 1.0))
                b_fit = float(np.median(zt - a_fit * zp))
                model_name = "offset_only"

            r = zt - (a_fit * zp + b_fit)
            proposed_a[name] = a_fit
            proposed_b[name] = b_fit
            detail[name] = {
                "model": model_name,
                "a": round(a_fit, 6),
                "b_mm": round(b_fit * 1e3, 3),
                "n_assigned": int(len(zp)),
                "residual_mm": round(float(np.median(r)) * 1e3, 3),
                "mad_mm": round(float(np.median(np.abs(r - np.median(r))))
                                * 1e3, 3),
                "per_plane": per_plane,
            }

        # ---- smooth and clamp ---------------------------------------------
        alpha = 1.0 if not self.initialised else self.ema
        clamped = []
        for name in self.names:
            if name in proposed_b:
                prev = self.b[name]
                val = proposed_b[name]
                if self.initialised and abs(val - prev) > self.max_step:
                    val = prev + np.sign(val - prev) * self.max_step
                    clamped.append(f"{name}:b")
                self.b[name] = (1.0 - alpha) * prev + alpha * val
            if name in proposed_a:
                prev = self.a[name]
                val = proposed_a[name]
                if self.initialised and abs(val - prev) > self.max_scale_step:
                    val = prev + np.sign(val - prev) * self.max_scale_step
                    clamped.append(f"{name}:a")
                self.a[name] = (1.0 - alpha) * prev + alpha * val
        self.initialised = True
        self.frames += 1
        if upper is not None:
            self.frames_two_plane += 1

        self.last = {
            "ok": True,
            "model": "two_plane" if upper is not None else "deck_only",
            "plane_normal": [round(v, 5) for v in n.tolist()],
            "deck_offset_m": round(float(d0), 5),
            "upper_offset_m": (round(float(upper), 5)
                               if upper is not None else None),
            "upper_note": upper_note,
            "span_m": round(float(span), 5),
            "deck_inliers": int(len(inliers)),
            "n_reference_points": int(len(pts_c)),
            "scale": {k: round(v, 6) for k, v in self.a.items()},
            "offsets_mm": {k: round(v * 1e3, 3) for k, v in self.b.items()},
            "per_camera": detail,
            "clamped": clamped,
            "degraded": upper is None,
            "solve_ms": round((time.perf_counter() - t0) * 1e3, 2),
        }
        if upper is None:
            self.last["warning"] = (
                "no second reference surface this frame, so only the OFFSET was "
                "solved and the per-view scale is whatever --load-affine "
                "supplied. The deck will agree and the parcel tops will not. "
                + str(upper_note.get("reason", "")))
        return self.coefficients(), self.last

    def summary(self):
        return {"frames_solved": self.frames,
                "frames_two_plane": self.frames_two_plane,
                "frames_deck_only": self.frames - self.frames_two_plane,
                "frames_rejected": self.rejected,
                "final_scale": {k: round(v, 6) for k, v in self.a.items()},
                "final_offsets_mm": {k: round(v * 1e3, 3)
                                     for k, v in self.b.items()},
                "base_scale": self.base_a,
                "ema": self.ema,
                "stride": self.stride,
                "assign_tol_m": self.assign_tol,
                "min_span_m": self.min_span,
                "max_step_m": self.max_step,
                "max_scale_step": self.max_scale_step}