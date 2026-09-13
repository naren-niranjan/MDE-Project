#!/usr/bin/env python3
"""
field_correct.py

Applies depth_field.py's four-DOF per-camera field inside da3_stream.py.

    d'(u,v) = d(u,v) * ( s + a*x~ + b*y~ ) + c
    x~ = (u - cx)/fx,   y~ = (v - cy)/fy

WHY THIS IS NOT --load-affine
=============================
load_affine applies Z -> a*Z + b, two coefficients, no dependence on pixel
position. Measured on capture_00001 the per-camera apparent tilt against the
measured deck ran 1.2 to 7.1 degrees before correction, which at 3 m sweeps the
plane by tens of millimetres across the frame. No scalar pair reaches that. The
tilt terms do, and they need x~ and y~, so the applier needs K.

x~ AND y~ ARE RESOLUTION INDEPENDENT
------------------------------------
Under an isotropic image rescale fx, fy, cx, cy all scale together, so
(u - cx)/fx is unchanged. The coefficients therefore survive a change of
--input-scale without refitting, which the raw pixel form would not.

INVALID SAMPLES MUST BE EXCLUDED EXPLICITLY
-------------------------------------------
c is up to 180 mm on this rig. Applied blindly it turns a zero or negative
sentinel into a plausible depth and injects a phantom plane at |c|. This is the
same trap apply_affine documents, and it is worse here because c is larger.

ORDERING RELATIVE TO auto-align
-------------------------------
The field goes FIRST and the aligner runs on its output, in OFFSET-ONLY mode.

The field carries the metric scale, fitted against ChArUco planes. The aligner's
scale term is solved from the deck to the parcel tops, 100 to 400 mm, which is
the same short lever arm over which s and c are 99.98 to 99.99 per cent
correlated in the offline fit -- measured, on this rig, running away to
s = 1.058, c = -292 mm on `top` while every anchor residual stayed under 7 mm.
Re-solving scale online on top of a field that already has it invites the same
runaway once per frame, with no holdout to catch it. So the field owns the scale
and the tilt; the aligner tracks only the residual offset drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from da3_fuse import pixel_rays


_GAIN_CACHE: dict = {}


def load_field(path: Path, names):
    """Read depth_field.json and return {name: (s, a, b, c)}.

    Refuses rather than silently substituting the identity, because a camera
    quietly left at s=1, c=0 while its three rig-mates are corrected by 27 to
    180 mm is precisely the failure that puts one sheet in the fused cloud.
    """
    data = json.loads(Path(path).read_text())
    coeffs = data.get("coefficients")
    if not isinstance(coeffs, dict):
        raise SystemExit(f"{path} has no 'coefficients' block")

    missing = [n for n in names if n not in coeffs]
    if missing:
        raise SystemExit(f"{path} lacks coefficients for: {', '.join(missing)}")

    out = {}
    for n in names:
        c = coeffs[n]
        out[n] = (float(c.get("s", 1.0)), float(c.get("a", 0.0)),
                  float(c.get("b", 0.0)), float(c.get("c", 0.0)))

    ident = [n for n, (s, a, b, c) in out.items()
             if abs(s - 1.0) < 1e-9 and abs(a) < 1e-12
             and abs(b) < 1e-12 and abs(c) < 1e-9]
    if ident:
        raise SystemExit(
            f"{path} carries the IDENTITY field for: {', '.join(ident)}. That "
            f"camera was not solved. Applying it alongside corrected cameras "
            f"puts it on its own sheet in the fused cloud, which is the "
            f"symptom this file exists to remove. Re-run MODE=field and read "
            f"the guard notes for that camera before streaming.")

    return out, data


def gain_image(shape, K, s, a, b):
    """s + a*x~ + b*y~ over the grid, cached per (shape, K, coefficients)."""
    key = (tuple(shape), K.tobytes(), round(s, 12), round(a, 12), round(b, 12))
    hit = _GAIN_CACHE.get(key)
    if hit is None:
        if len(_GAIN_CACHE) > 32:
            _GAIN_CACHE.clear()
        rays = pixel_rays(tuple(shape), np.asarray(K, np.float64))
        hit = (s + a * rays[..., 0] + b * rays[..., 1]).astype(np.float64)
        _GAIN_CACHE[key] = hit
    return hit


def apply_field_stack(depth, names, K_out, field):
    """d -> d * gain(u,v) + c per camera, leaving invalid samples untouched.

    depth is the stacked (n_cams, H, W) array da3_stream carries. K_out is the
    per-camera intrinsics ACTUALLY IN USE for that stack, so --input-scale is
    already accounted for.
    """
    out = np.array(depth, dtype=np.float64, copy=True)
    for i, n in enumerate(names):
        s, a, b, c = field[n]
        d = out[i]
        good = np.isfinite(d) & (d > 0)
        if not good.any():
            continue
        g = gain_image(d.shape, np.asarray(K_out[i], np.float64), s, a, b)
        d[good] = d[good] * g[good] + c
        # a large negative c can drive a shallow sample non-positive; such a
        # sample is not a measurement any more, so drop it rather than fuse it
        d[good & ~(d > 0)] = 0.0
        out[i] = d
    return out


def field_summary(field, deck_depth_m=None, parcel_height_m=0.40):
    """The correction in millimetres, and how far it moves off the anchor.

    THIS IS THE NUMBER THE OFFLINE REPORT DOES NOT PRINT. Every residual in
    field_report.json is measured inside the 218 mm span of the reference
    planes, where s and c are ~99.99% correlated and any point along that null
    direction fits equally well. The pair only separates OUTSIDE the span, which
    is where the parcels are. swing_mm is the difference between the correction
    at the deck and the correction at parcel_height_m above it: it is zero for a
    pure offset and grows as parcel_height * (s - 1). Measured on `top` with
    s = 1.0584: 23 mm of swing over 400 mm, all of it unconstrained by the fit.
    """
    rows = {}
    for n, (s, a, b, c) in field.items():
        d0 = deck_depth_m
        if d0 is None:
            rows[n] = {"s": s, "c_mm": c * 1e3}
            continue
        at_deck = d0 * (s - 1.0) + c
        at_top = (d0 - parcel_height_m) * (s - 1.0) + c
        rows[n] = {"s": round(s, 6),
                   "c_mm": round(c * 1e3, 2),
                   "correction_at_deck_mm": round(at_deck * 1e3, 2),
                   "correction_at_parcel_top_mm": round(at_top * 1e3, 2),
                   "swing_mm": round((at_top - at_deck) * 1e3, 2)}
    return rows