#!/usr/bin/env python3
"""
rigkit.py — shared calibration, projection and plane geometry for the
camera-subset study. Imported by every other script in this folder.

Rig frame == center camera frame. All lengths in metres.
"""
import itertools
import json
import os
import re
import numpy as np

# The rig's cameras, in canonical order. Subset directory names are built by
# joining names in THIS order with "+", so changing the order renames every run
# directory. Add cameras at the end unless you are prepared to re-run.
CAMS = ["center", "left", "top", "right"]


def subsets(cams=None, k_min=1, k_max=None):
    """Every subset of `cams` of size k_min..k_max, as "a+b" names in CAMS
    order. 4 cameras gives 15 subsets: 4 singles, 6 pairs, 4 triples, 1 quad."""
    cams = list(cams or CAMS)
    k_max = k_max or len(cams)
    return ["+".join(s)
            for k in range(k_min, k_max + 1)
            for s in itertools.combinations(cams, k)]


def canon(sub, cams=None):
    """Normalise a subset name to CAMS order, so "top+center" and
    "center+top" resolve to the same run directory instead of two."""
    cams = list(cams or CAMS)
    parts = [p for p in str(sub).replace(",", "+").split("+") if p]
    unknown = [p for p in parts if p not in cams]
    if unknown:
        raise ValueError(f"unknown camera(s) {unknown}; known: {cams}")
    return "+".join(c for c in cams if c in set(parts))


SUBSETS = subsets()


# ------------------------------------------------------------------ calib
class Rig:
    """Intrinsics + extrinsics for the rig's cameras.

    `cams` defaults to CAMS. Pass an explicit list to work with a subset, e.g.
    Rig(dir, cams=canon("center+top").split("+")).
    """

    def __init__(self, calib_dir, cams=None):
        self.dir = calib_dir
        self.cams = list(cams or CAMS)
        self.K, self.D, self.size, self.R, self.t = {}, {}, {}, {}, {}
        ext = json.load(open(self._find("extrinsics.json")))
        for c in self.cams:
            j = json.load(open(self._find(f"intrinsics_{c}.json",
                                          f"{c}_intrinsics.json")))
            self.K[c] = np.array(j["camera_matrix"], float)
            self.D[c] = np.array(j["dist_coeffs"], float).ravel()
            self.size[c] = tuple(j.get("image_size", (2448, 2048)))
            if c not in ext:
                raise KeyError(f"{c} missing from extrinsics.json")
            self.R[c] = np.array(ext[c]["R"], float)
            self.t[c] = np.array(ext[c]["t"], float).ravel()

    def _find(self, *names):
        for n in names:
            p = os.path.join(self.dir, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{names[0]} not in {self.dir}")

    def centre(self, c):
        """Camera position in the rig frame."""
        return -self.R[c].T @ self.t[c]

    def focal_mm(self, c, pixel_pitch_mm=3.45e-3):
        return self.K[c][0, 0] * pixel_pitch_mm, self.K[c][1, 1] * pixel_pitch_mm

    def fov_deg(self, c):
        w, h = self.size[c]
        return (2 * np.degrees(np.arctan(w / 2 / self.K[c][0, 0])),
                2 * np.degrees(np.arctan(h / 2 / self.K[c][1, 1])))

    def baseline(self, a, b):
        return float(np.linalg.norm(self.centre(a) - self.centre(b)))

    def project(self, X, c):
        """Rig-frame points (N,3) -> pixels (N,2), full k1k2p1p2k3 distortion."""
        X = np.atleast_2d(np.asarray(X, float))
        Xc = X @ self.R[c].T + self.t[c]
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        xn, yn = Xc[:, 0] / z, Xc[:, 1] / z
        d = np.zeros(5)
        d[:len(self.D[c])] = self.D[c][:5]
        k1, k2, p1, p2, k3 = d
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        K = self.K[c]
        uv = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)
        uv[Xc[:, 2] <= 0] = np.nan
        return uv

    def visible(self, X, c, margin=0):
        w, h = self.size[c]
        uv = self.project(X, c)
        ok = (uv[:, 0] >= margin) & (uv[:, 0] < w - margin) & \
             (uv[:, 1] >= margin) & (uv[:, 1] < h - margin)
        return ok & ~np.isnan(uv[:, 0])

    def n_views(self, X, cams=None, margin=0):
        cams = cams or self.cams
        return sum(self.visible(X, c, margin).astype(int) for c in cams)


# ------------------------------------------------------------------ planes
def plane_z(P, n, d):
    """z on the plane n.X + d = 0 at each point's (x, y)."""
    P = np.atleast_2d(P)
    return -(d + n[0] * P[:, 0] + n[1] * P[:, 1]) / n[2]


def fit_plane_svd(Q):
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[2]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


def fit_plane_ransac(Q, tol=0.006, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    best, bi = None, 0
    for _ in range(trials):
        s = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ s[0]
        k = int((np.abs(Q @ n + d) < tol).sum())
        if k > bi:
            bi, best = k, (n, d)
    if best is None:
        return fit_plane_svd(Q)
    n, d = best
    inl = np.abs(Q @ n + d) < tol
    return fit_plane_svd(Q[inl])


def deck_frame(gt):
    """Orthonormal frame on the deck: ex ~ along belt, ey ~ across, n up."""
    n = np.array(gt["deck_plane"]["n"], float)
    d = float(gt["deck_plane"]["d"])
    ex = np.cross([0, 1, 0], n)
    ex /= np.linalg.norm(ex)
    return n, d, ex, np.cross(n, ex)


def to_deck(P, gt):
    """Rig-frame points -> (along, across, height-above-deck)."""
    n, d, ex, ey = deck_frame(gt)
    return np.stack([P @ ex, P @ ey, -(P @ n + d)], 1)


def from_deck(u, v, h, gt):
    n, d, ex, ey = deck_frame(gt)
    return np.asarray(u)[..., None] * ex + np.asarray(v)[..., None] * ey \
        - d * n - np.asarray(h)[..., None] * n


# ------------------------------------------------------------------ clouds
def load_cloud(path):
    """PLY (binary or ascii) or .npy -> (N,3) float64 xyz."""
    if path.endswith(".npy"):
        return np.load(path)[:, :3].astype(np.float64)
    raw = open(path, "rb").read()
    i = raw.find(b"end_header")
    if i < 0:
        raise ValueError(f"{path}: not a PLY")
    hdr = raw[:i].decode("ascii", "replace")
    i = raw.find(b"\n", i) + 1
    n = int(re.search(r"element vertex (\d+)", hdr).group(1))
    if "ascii" in hdr:
        return np.loadtxt(path, skiprows=hdr.count("\n") + 1,
                          max_rows=n, usecols=(0, 1, 2)).astype(np.float64)
    tymap = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
             "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2"}
    fields = []
    for ln in hdr.splitlines():
        if ln.startswith("property") and "list" not in ln:
            _, ty, nm = ln.split()[:3]
            fields.append((nm, tymap[ty]))
    d = np.frombuffer(raw, dtype=np.dtype(fields), count=n, offset=i)
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


def save_ply(path, P):
    P = np.asarray(P, "<f4")
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(P)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\nend_header\n")
        f.write(P.tobytes())


# --------------------------------------------------------------- depth maps
def scale_K(K, src_wh, dst_wh):
    """Rescale intrinsics between resolutions.

    Uses the pixel-centre convention cx' = (cx + 0.5)*s - 0.5. The naive
    cx' = cx*s leaves a sub-pixel offset that grows with the scale factor.
    """
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K2 = K.copy().astype(float)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K2


def unproject(depth, K, mode="z", mask=None):
    """Depth map (H,W) -> camera-frame points (N,3).

    mode 'z'     : values are depth along the optical axis
    mode 'range' : values are Euclidean distance from the camera centre

    K must already match the depth map's resolution -- use scale_K if the
    map was produced at a different process-res from the calibration.
    """
    depth = np.asarray(depth, float)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    m = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        m &= mask.astype(bool)
    x, y, d = x[m], y[m], depth[m]
    if mode == "range":
        d = d / np.sqrt(1 + x * x + y * y)
    return np.stack([x * d, y * d, d], 1)


def load_depth(path):
    """Load a depth array from .npy or .npz (first 2-D float array found)."""
    if path.endswith(".npz"):
        z = np.load(path)
        for k in z.files:
            a = z[k]
            if a.ndim >= 2 and a.dtype.kind == "f":
                return a, k
        raise ValueError(f"{path}: no float 2-D array among {z.files}")
    return np.load(path), None


def cam_to_rig(X, R, t):
    """Camera-frame points -> rig frame."""
    return (np.asarray(X) - t) @ R