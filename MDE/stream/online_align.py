#!/usr/bin/env python3
"""
online_align.py

Per-frame estimation of the inter-camera depth offset, using the scene's own
dominant plane as the reference.

Why this exists
---------------
Coefficients solved offline by layer_align.py do not survive a change of
exposure or gain. DA3 infers depth from appearance, so altering brightness,
contrast or noise moves its per-view scale even though the rig geometry is
untouched. A stored correction is therefore only valid for the photometric
conditions it was fitted under, which makes it useless for a workcell where
lighting varies.

The conveyor deck is a large planar surface visible to every camera in every
frame and its geometry never changes. That is a better reference than a file.
Fitting it in the reference view and solving each other camera's offset
against it costs a few milliseconds against a 1.3 s inference, and it tracks
whatever the model does from frame to frame.

What is estimated, and what is not
----------------------------------
Only the additive term b. Measurements on this rig gave top -57.56 mm at
4.03 m and -57.62 mm at 3.19 m, so across 840 mm of distance the bias moved by
0.06 mm: it is additive, not multiplicative. The scale term a is close to
unity, changes slowly, and cannot be separated from b using one plane, so it
is taken from an offline solve and held fixed. One plane, one parameter.

This corrects RELATIVE disagreement between views. It cannot detect a bias
common to all four cameras, because the reference plane is itself derived from
the reference camera. Absolute anchoring remains a separate ground-truth
calibration.
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


class OnlineAligner:
    """Track the per-camera additive depth offset frame by frame."""

    def __init__(self, names, reference, base_a=None, stride=3,
                 plane_thresh=0.010, assign_tol=0.12, ema=0.3,
                 min_points=2000, max_step=0.150, seed=0):
        if reference not in names:
            raise ValueError(f"reference {reference!r} not among {names}")
        self.names = list(names)
        self.reference = reference
        self.ref_idx = self.names.index(reference)
        self.base_a = dict(base_a) if base_a else {n: 1.0 for n in self.names}
        self.stride = max(1, int(stride))
        self.plane_thresh = plane_thresh
        self.assign_tol = assign_tol
        self.ema = float(np.clip(ema, 0.0, 1.0))
        self.min_points = min_points
        # A single frame should never move the offset by more than this. Guards
        # against a spurious plane fit, for example when a pallet covers the
        # deck, dragging the whole cloud sideways.
        self.max_step = max_step
        self.seed = seed

        self.b = {n: 0.0 for n in self.names}
        self.initialised = False
        self.frames = 0
        self.rejected = 0
        self.last = {}

    def _scaled(self, depth, i):
        a = self.base_a[self.names[i]]
        d = depth[i].astype(np.float64)
        if a != 1.0:
            good = np.isfinite(d) & (d > 0)
            d = d.copy()
            d[good] *= a
        return d

    def update(self, depth, K_out, E_out):
        """Return the per-camera offsets for this frame, plus diagnostics."""
        t0 = time.perf_counter()
        s = self.stride

        # ---- reference plane, from the reference view alone --------------
        i0 = self.ref_idx
        d_ref = self._scaled(depth, i0)
        rays_ref = rays_for(d_ref.shape, K_out[i0])
        sub = (slice(None, None, s), slice(None, None, s))
        z = d_ref[sub]
        good = np.isfinite(z) & (z > 0)
        pts_c = rays_ref[sub][good] * z[good][:, None]
        if len(pts_c) < self.min_points:
            self.rejected += 1
            self.last = {"ok": False, "reason": "too few reference points",
                         "n_reference": int(len(pts_c))}
            return dict(self.b), self.last

        pts_w = to_world(pts_c, E_out[i0])
        o3d.utility.random.seed(self.seed)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_w)
        model, inliers = pcd.segment_plane(self.plane_thresh, 3, 400)
        a, bb, c, dd = model
        n = np.array([a, bb, c], np.float64)
        nrm = np.linalg.norm(n)
        if nrm < 1e-9 or len(inliers) < self.min_points // 4:
            self.rejected += 1
            self.last = {"ok": False, "reason": "no usable reference plane",
                         "n_inliers": int(len(inliers))}
            return dict(self.b), self.last
        n, d0 = n / nrm, float(-dd / nrm)
        if d0 < 0:
            n, d0 = -n, -d0
        n, d0 = refine_plane(pts_w, n, d0, self.plane_thresh)

        # ---- offset per camera against that plane -----------------------
        proposed, detail = {}, {}
        for i, name in enumerate(self.names):
            if i == i0:
                proposed[name] = 0.0
                detail[name] = {"n_assigned": int(len(inliers)), "residual_mm": 0.0}
                continue

            d_i = self._scaled(depth, i)
            rays_i = rays_for(d_i.shape, K_out[i])
            n_c, d_c = plane_to_camera(n, d0, E_out[i])

            zi = d_i[sub]
            vi = rays_i[sub]
            denom = vi @ n_c
            with np.errstate(divide="ignore", invalid="ignore"):
                z_target = d_c / denom
            ok = (np.isfinite(zi) & (zi > 0) & np.isfinite(z_target)
                  & (z_target > 0) & (np.abs(denom) > 1e-6))
            near = ok & (np.abs(zi - z_target) <= self.assign_tol)
            cnt = int(near.sum())
            if cnt < self.min_points:
                detail[name] = {"n_assigned": cnt, "residual_mm": None,
                                "note": "kept previous offset"}
                continue

            resid = z_target[near] - zi[near]
            proposed[name] = float(np.median(resid))
            detail[name] = {
                "n_assigned": cnt,
                "residual_mm": round(float(np.median(resid)) * 1e3, 3),
                "mad_mm": round(float(np.median(np.abs(resid - np.median(resid)))) * 1e3, 3),
            }

        # ---- smooth and clamp -------------------------------------------
        alpha = 1.0 if not self.initialised else self.ema
        clamped = []
        for name, val in proposed.items():
            prev = self.b[name]
            if self.initialised and abs(val - prev) > self.max_step:
                val = prev + np.sign(val - prev) * self.max_step
                clamped.append(name)
            self.b[name] = (1.0 - alpha) * prev + alpha * val
        self.initialised = True
        self.frames += 1

        self.last = {
            "ok": True,
            "plane_normal": [round(v, 5) for v in n.tolist()],
            "plane_offset_m": round(float(d0), 5),
            "plane_inliers": int(len(inliers)),
            "n_reference_points": int(len(pts_c)),
            "offsets_mm": {k: round(v * 1e3, 3) for k, v in self.b.items()},
            "clamped": clamped,
            "solve_ms": round((time.perf_counter() - t0) * 1e3, 2),
        }
        return dict(self.b), self.last

    def summary(self):
        return {"frames_solved": self.frames,
                "frames_rejected": self.rejected,
                "final_offsets_mm": {k: round(v * 1e3, 3) for k, v in self.b.items()},
                "base_scale": self.base_a,
                "ema": self.ema,
                "stride": self.stride,
                "assign_tol_m": self.assign_tol,
                "max_step_m": self.max_step}