#!/usr/bin/env python3
"""
scan_frame.py

The pieces every scan-referenced tool needs: reading a cloud, finding the deck,
building the conveyor frame, and putting a reference scan into the rig's world
frame.

Why this is a module
--------------------
Three tools need the same registration: scan_register.py solves it,
scan_align.py consumes it to fit the depth correction, and gt_compare.py
consumes it to grade the result. If each carried its own copy of the plane fit
and the belt frame, a correction fitted in one frame would be graded in
another, and the disagreement would look like depth error.

The registration is deliberately RIGID. The scanner and the cameras are both
looking at the same rigid cell, so any scale that would improve the fit is
depth-model error, and letting the registration absorb it would hide the one
number worth reading.

Keep this file beside scan_register.py, scan_align.py, gt_compare.py and
belt_from_scan.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise SystemExit("scipy is required")


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

_PLY_T = {"double": "f8", "float": "f4", "uchar": "u1", "char": "i1",
          "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}


def read_ply(path) -> np.ndarray:
    """A cloud as an (N, 3) float64 array.

    Open3D is used when importable. The fallback exists because the machine
    holding the reference scans is usually not the rig, and a scan should not
    have to be moved before it can be looked at.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"cloud not found: {path}")
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=float)
        if len(pts):
            return pts
    except ImportError:
        pass

    with open(path, "rb") as fh:
        hdr = b""
        while b"end_header" not in hdr:
            line = fh.readline()
            if not line:
                raise SystemExit(f"{path}: no PLY header")
            hdr += line
        text = hdr.decode("ascii", errors="replace").splitlines()
        if not any("binary_little_endian" in t for t in text):
            raise SystemExit(f"{path}: the fallback reader handles only "
                             f"binary_little_endian; install open3d")
        n, props = None, []
        for line in text:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "element" and tok[1] == "vertex":
                n = int(tok[2])
            elif tok[0] == "property" and tok[1] != "list":
                props.append((tok[2], _PLY_T[tok[1]]))
        dt = np.dtype(props)
        arr = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)


def write_ply(path, pts, rgb=None) -> None:
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
            "property float x", "property float y", "property float z"]
    if rgb is not None:
        head += ["property uchar red", "property uchar green",
                 "property uchar blue"]
    head.append("end_header")
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(arr.tobytes())


def voxel(P, v):
    """One representative point per occupied voxel, by linear key.

    np.unique over rows would sort a structured array and dominate the runtime
    on a two-million-point scan.
    """
    P = np.asarray(P, dtype=float)
    if v is None or v <= 0 or not len(P):
        return P
    q = np.floor(P / v).astype(np.int64)
    q -= q.min(axis=0)
    d = q.max(axis=0) + 1
    if float(d[0]) * float(d[1]) * float(d[2]) > 2 ** 62:
        raise SystemExit("voxel grid too large; crop the cloud first")
    key = (q[:, 0] * d[1] + q[:, 1]) * d[2] + q[:, 2]
    _, first = np.unique(key, return_index=True)
    return P[np.sort(first)]


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------

def fit_plane(P, n, d, thr, iters=6):
    """Reweighted least squares about a seed plane, normal towards +Z."""
    n = np.asarray(n, dtype=float)
    for _ in range(iters):
        inl = np.abs(P @ n - d) <= thr
        if int(inl.sum()) < 100:
            break
        Q = P[inl]
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        n = vt[-1] / np.linalg.norm(vt[-1])
        d = float(n @ c)
    if n[2] < 0:
        n, d = -n, -d
    return n, float(d)


def largest_plane(P, thr=0.010, iters=900, seed=0):
    rng = np.random.default_rng(seed)
    S = P if len(P) < 80000 else P[rng.choice(len(P), 80000, replace=False)]
    best = (None, None, -1)
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        nn = np.cross(b - a, c - a)
        m = float(np.linalg.norm(nn))
        if m < 1e-9:
            continue
        nn = nn / m
        dd = float(nn @ a)
        cnt = int((np.abs(S @ nn - dd) <= thr).sum())
        if cnt > best[2]:
            best = (nn, dd, cnt)
    return fit_plane(P, best[0], best[1], thr)


def deck_plane(P, seed_z=None, thr=0.010, window=0.15, label="cloud"):
    """The deck, seeded on its distance rather than taken as the largest plane.

    In any cloud that also sees the floor, the floor is the larger surface.
    Taking it puts the whole conveyor 0.8 m "above the plane". Seeding on the
    expected standoff is the guard, and it is the same guard
    --seg-plane-distance provides in box_segment.py.

    The seed is a guard, not an answer, so the result is checked against it. A
    window that misses the real deck does not fail: it finds SOMETHING, fits a
    plane to whatever sparse points happen to lie in the window, and returns it
    with no sign that anything is wrong. Downstream that plane becomes the
    frame everything else is expressed in, and the first visible symptom is a
    belt measured far too narrow, several steps later. So the share of the
    cloud the fitted plane actually holds is reported, and a thin one says so.

    The window is 150 mm rather than 80 because DA3's absolute scale moves with
    framing: changing --process-res from 504 to 1008 moved the deck on this rig
    by 123 mm, which a tighter window would exclude while a correct seed was
    being passed.
    """
    if seed_z is None:
        return largest_plane(P, thr)
    sel = P[np.abs(P[:, 2] - float(seed_z)) < window]
    if len(sel) < 500:
        raise SystemExit(
            f"{label}: only {len(sel)} points within {window * 1e3:.0f} mm of "
            f"z={seed_z:.3f}. The seed is nowhere near a surface; run --probe "
            f"and read the deck distance off it rather than guessing.")
    # Start from the densest depth inside the window, not from the seed. The
    # first refit only looks 20 mm either side of wherever it starts, so
    # starting at the seed makes the window decorative: a seed 100 mm off finds
    # whatever sparse points sit at the seed and converges on them, and the
    # result is a plane holding almost nothing that is nevertheless returned
    # without complaint. Taking the mode makes the window do what it says.
    counts, edges = np.histogram(sel[:, 2], bins=max(8, int(2 * window / 0.005)))
    d0 = float(0.5 * (edges[int(np.argmax(counts))]
                      + edges[int(np.argmax(counts)) + 1]))
    n, d = fit_plane(sel, np.array([0.0, 0.0, 1.0]), d0, 0.02)
    n, d = fit_plane(P, n, d, thr)
    share = float(np.mean(np.abs(P @ n - d) <= thr))
    if share < 0.03:
        print(f"[WARN ] {label}: the plane fitted at {d:.3f} m holds only "
              f"{share * 100:.1f} per cent of the points, which is too thin to "
              f"be a conveyor deck. The seed {seed_z:.3f} m has probably "
              f"missed it. Run --probe: the deck is the large near-horizontal "
              f"plane nearest the cameras, and in a raw uncorrected cloud it "
              f"appears as SEVERAL parallel sheets a few centimetres apart, "
              f"one per view.")
    elif abs(d - float(seed_z)) > 0.5 * window:
        print(f"[note ] {label}: deck found at {d:.3f} m, {(d - seed_z) * 1e3:+.0f}"
              f" mm from the seed and near the edge of the search window. "
              f"Re-seed nearer it.")
    return n, d


def basis(n):
    h = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = h - n * float(h @ n)
    ex = ex / np.linalg.norm(ex)
    return ex, np.cross(n, ex)


def belt_frame(P, n, d, close_m=0.30, cell=0.01, max_height=0.60,
               below=0.050):
    """Belt centre and axes, from the deck plus whatever stands on it.

    Closed before the components are taken, for the reason
    box_segment.belt_from_raster gives: the deck is visible only between the
    parcels, so connectivity alone returns one strip rather than the belt and
    the footprint shrinks as the belt fills.

    below is how far under the fitted deck a point may sit and still count as
    deck. It has to exceed the per-view layering, which in an UNCORRECTED cloud
    is a couple of centimetres: a view sitting 25 mm low otherwise drops out
    entirely, and since the outer strips of belt are seen by one camera each,
    the footprint then comes back narrow by roughly the width of those strips.
    Measured on this rig that turned a 941 mm belt into a 750 mm one.
    """
    ex, ey = basis(n)
    h = -(P @ n - d)                        # positive above the deck
    Q = P[(h > -abs(below)) & (h < max_height)]
    if len(Q) < 1000:
        raise SystemExit("too few points near the deck to find the belt")
    u, v = Q @ ex, Q @ ey
    u0, v0 = float(u.min()), float(v.min())
    W = int((u.max() - u0) / cell) + 1
    H = int((v.max() - v0) / cell) + 1
    img = np.zeros((H, W), np.uint8)
    img[np.clip(((v - v0) / cell).astype(int), 0, H - 1),
        np.clip(((u - u0) / cell).astype(int), 0, W - 1)] = 255
    k = max(3, int(round(close_m / cell)) | 1)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    nl, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if nl <= 1:
        raise SystemExit("the deck did not form a connected patch")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero((lab == big) & (img > 0))
    pts = (np.stack([xs * cell + u0, ys * cell + v0], 1) * 1000.0)
    (cu, cvv), (w, hgt), ang = cv2.minAreaRect(pts.astype(np.float32))
    a = np.radians(ang)
    bx = ex * np.cos(a) + ey * np.sin(a)
    by = -ex * np.sin(a) + ey * np.cos(a)
    if w < hgt:                             # long edge along X, as a conveyor
        bx, by = by, -bx
    centre = ex * (cu / 1000.0) + ey * (cvv / 1000.0) + n * d
    return {"centre": centre, "x_axis": bx, "y_axis": by, "normal": n,
            "offset": d, "length_m": max(w, hgt) / 1000.0,
            "width_m": min(w, hgt) / 1000.0}


def to_frame(P, bf):
    """Into the conveyor frame: u along the belt, v across, z above the deck."""
    R = np.column_stack([bf["x_axis"], bf["y_axis"], -bf["normal"]])
    return (np.asarray(P, dtype=float) - bf["centre"]) @ R


# --------------------------------------------------------------------------
# rasters
# --------------------------------------------------------------------------

class Grid:
    """A plan-view raster of a cloud already expressed in the conveyor frame."""

    def __init__(self, ext, cell):
        self.u0, self.u1, self.v0, self.v1 = [float(v) for v in ext]
        self.cell = float(cell)
        self.W = max(2, int((self.u1 - self.u0) / self.cell))
        self.H = max(2, int((self.v1 - self.v0) / self.cell))

    def index(self, P):
        iu = ((P[:, 0] - self.u0) / self.cell).astype(np.int64)
        iv = ((P[:, 1] - self.v0) / self.cell).astype(np.int64)
        ok = (iu >= 0) & (iu < self.W) & (iv >= 0) & (iv < self.H)
        return iu, iv, ok

    def surface(self, P, min_points=3):
        """Median height and the 10-90 spread per cell.

        The median rather than the highest point, because the highest point in
        a DA3 cell is often smoothing off the edge of a parcel. The spread is
        the layer thickness, which is what a scanner can be compared against
        directly.
        """
        iu, iv, ok = self.index(P)
        flat = iv[ok] * self.W + iu[ok]
        z = P[ok, 2]
        order = np.argsort(flat, kind="stable")
        flat, z = flat[order], z[order]
        if not len(flat):
            e = np.full((self.H, self.W), np.nan)
            return e, e.copy()
        starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
        ends = np.r_[starts[1:], len(flat)]
        med = np.full(self.W * self.H, np.nan)
        spr = np.full(self.W * self.H, np.nan)
        for a, b in zip(starts, ends):
            if b - a < min_points:
                continue
            zz = z[a:b]
            med[flat[a]] = np.median(zz)
            spr[flat[a]] = np.percentile(zz, 90) - np.percentile(zz, 10)
        return med.reshape(self.H, self.W), spr.reshape(self.H, self.W)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    R = np.eye(4)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R


def coarse_align(A, B, cell=0.005, lo=0.045, hi=0.60,
                 yaw_deg=8.0, yaw_step=1.0, overlay=None):
    """Yaw and the in-plane offset, from the parcel silhouettes.

    The belt rectangle fixes the axes only to the accuracy of a minimum-area
    rectangle fitted to two different lengths of belt, which is a degree or two.
    Trying 0 and 180 alone leaves that residual yaw for ICP to find, and ICP
    given a poor start will take an out-of-plane tilt instead, because tilting
    the whole cloud can match some box faces to some belt edges. So the yaw is
    swept here, where a wrong answer is visible as a low correlation, rather
    than left to a step where a wrong answer looks like a converged fit.

    Swept coarsely on a 20 mm raster, then the offset alone is refined on a
    fine one at the winning angle.
    """
    def build(g, P):
        sel = (P[:, 2] > lo) & (P[:, 2] < hi)
        iu, iv, ok = g.index(P[sel])
        img = np.zeros((g.H, g.W), np.float32)
        img[iv[ok], iu[ok]] = 1.0
        return img

    def match(g, Am, Bm, pad_m=0.9):
        pad = int(pad_m / g.cell)
        Bp = cv2.copyMakeBorder(Bm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
        res = cv2.matchTemplate(Bp, Am, cv2.TM_CCORR_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        return float(mx), (loc[0] - pad) * g.cell, (loc[1] - pad) * g.cell

    ext = (-2.6, 2.6, -0.95, 0.95)
    angles = []
    for base in (0.0, 180.0):
        d = -yaw_deg
        while d <= yaw_deg + 1e-9:
            angles.append(base + d)
            d += yaw_step

    g_c = Grid(ext, 0.02)
    Bc = build(g_c, B)
    best = None
    for th in angles:
        L = apply_T(_rot_z(th), A)
        score, du, dv = match(g_c, build(g_c, L), Bc)
        if best is None or score > best[0]:
            best = (score, th, du, dv)
    _, th, du, dv = best

    g_f = Grid(ext, cell)
    L = apply_T(_rot_z(th), A)
    L[:, 0] += du
    L[:, 1] += dv
    _, ddu, ddv = match(g_f, build(g_f, L), build(g_f, B), pad_m=0.15)

    T = _rot_z(th)
    T[0, 3], T[1, 3] = du + ddu, dv + ddv

    # Count the parcels each cloud shows. A registration cannot invent or
    # remove one, so a difference here is the scene having changed between the
    # scan and the capture, which is the single failure no setting repairs and
    # the one a low correlation is most often hiding.
    Am_f = build(g_f, apply_T(T, A))
    Bm_f = build(g_f, B)
    counts = []
    for m in (Am_f, Bm_f):
        occ = (m > 0).astype(np.uint8)
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        n_lab, _, st, _ = cv2.connectedComponentsWithStats(occ, 8)
        counts.append(int(sum(1 for i in range(1, n_lab)
                              if st[i, cv2.CC_STAT_AREA] * cell ** 2 > 0.02)))

    if overlay is not None:
        # The single most useful artefact when a registration will not settle.
        # A low correlation has two very different causes that no number
        # separates: the clouds are misaligned, or they are of different
        # scenes. One look at this tells them apart, because a misalignment
        # shifts every parcel the same way and a changed scene does not.
        img = np.zeros((g_f.H, g_f.W, 3), np.uint8)
        img[..., 2] = (Am_f * 255).astype(np.uint8)    # rig, red
        img[..., 1] = (Bm_f * 255).astype(np.uint8)    # scan, green
        img = cv2.resize(img, None, fx=2, fy=2,
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, "red rig   green scan   yellow both", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imwrite(str(overlay), img)

    # The coarse score is the one reported and gated on. A finer raster is
    # sparser, so its correlation is systematically lower and would need its
    # own threshold; the coarse figure is what the 0.8 rule of thumb refers to.
    return T, best[0], th, tuple(counts)


def umeyama(X, Y, with_scale=False):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    E = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ E @ Vt
    s = (float((D * np.diag(E)).sum()) * len(X) / float((Xc ** 2).sum())
         if with_scale else 1.0)
    return s, R, my - s * R @ mx


def kabsch_planar(X, Y):
    """Best yaw and 3D translation taking X onto Y.

    Rotation about the deck normal only. Both clouds have already been put
    into their own deck-normal frames, so an out-of-plane rotation here would
    not be a correction to anything: it would be ICP buying a lower residual by
    hinging one cloud away from the other, which is exactly the failure this
    exists to prevent.
    """
    mx, my = X.mean(0), Y.mean(0)
    a, b = X - mx, Y - my
    num = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    den = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    th = np.arctan2(num, den)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R, my - R @ mx


def icp(A, B, iters=40, keep=0.70, planar=True):
    """Trimmed ICP, planar by default.

    Trimming rather than a fixed gate, because the two clouds see different
    parts of the cell and a fixed gate would either admit the parts only one of
    them saw or reject the genuine disagreement being measured.
    """
    tree = cKDTree(B)
    X = np.asarray(A, dtype=float).copy()
    T = np.eye(4)
    for _ in range(iters):
        d, j = tree.query(X, workers=-1)
        m = d < max(0.02, float(np.percentile(d, 100 * keep)))
        if int(m.sum()) < 50:
            break
        if planar:
            R, t = kabsch_planar(X[m], B[j[m]])
        else:
            _, R, t = umeyama(X[m], B[j[m]], False)
        X = X @ R.T + t
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        T = step @ T
    d, _ = tree.query(X, workers=-1)
    return T, d


def out_of_plane_deg(R):
    """How far a rotation tips the vertical axis. Zero for a pure yaw."""
    return float(np.degrees(np.arccos(np.clip(float(np.asarray(R)[2, 2]),
                                              -1.0, 1.0))))


def compose(*Ts):
    out = np.eye(4)
    for T in Ts:
        out = np.asarray(T, dtype=float) @ out
    return out


def apply_T(T, P):
    T = np.asarray(T, dtype=float)
    return np.asarray(P, dtype=float) @ T[:3, :3].T + T[:3, 3]


def solve_scan_to_rig(scan, rig, deck_scan=None, deck_rig=None,
                      voxel_m=0.006, planar=True, verbose=True,
                      overlay=None, below=0.050, window=0.15):
    """The rigid transform putting a reference scan into the rig world frame.

    Both clouds are taken into their own conveyor frame first, which removes
    the two large rotations and leaves only the yaw, the offset along the belt,
    and a small refinement. Registering the raw clouds directly would need a
    global descriptor and would still be seeded from the deck.

    Because the deck frames are built first, the residual transform is planar
    by construction, and the refinement is constrained to be. A 7 degree
    out-of-plane rotation coming out of a step that had no business producing
    one is a diverged fit wearing the costume of a converged one.
    """
    S = voxel(scan, voxel_m)
    R_ = voxel(rig, voxel_m)
    nS, dS = deck_plane(S, deck_scan, window=window, label="scan")
    nR, dR = deck_plane(R_, deck_rig, window=window, label="rig")
    bS = belt_frame(S, nS, dS, below=below)
    bR = belt_frame(R_, nR, dR, below=below)
    if verbose:
        print(f"[frame] scan deck d={dS:.4f} m   belt {bS['length_m'] * 1e3:.0f}"
              f" x {bS['width_m'] * 1e3:.0f} mm visible")
        print(f"[frame] rig  deck d={dR:.4f} m   belt {bR['length_m'] * 1e3:.0f}"
              f" x {bR['width_m'] * 1e3:.0f} mm visible")
        wr = bR["width_m"] / max(bS["width_m"], 1e-6)
        if not 0.85 <= wr <= 1.15:
            print(f"[WARN ] the rig sees the belt {wr * 100:.0f} per cent as "
                  f"wide as the scan does. The two should agree within a few "
                  f"per cent, so its deck plane has probably locked onto one "
                  f"camera's sheet rather than the deck. Try --probe and set "
                  f"--deck-rig from it.")

    Ls, Lr = to_frame(S, bS), to_frame(R_, bR)
    Tc, score, yaw, counts = coarse_align(Ls, Lr, overlay=overlay)
    if verbose:
        print(f"[frame] silhouette correlation {score:.3f} at "
              f"{yaw:+.1f} deg of residual yaw")
        print(f"[frame] parcels standing on the belt: rig {counts[0]}, "
              f"scan {counts[1]}")
        if overlay is not None:
            print(f"[frame] silhouettes written to {overlay}")
    if abs(counts[0] - counts[1]) > 1:
        print(f"[WARN ] the two clouds do not hold the same number of parcels "
              f"({counts[0]} against {counts[1]}). A registration cannot add "
              f"or remove one, so the belt was in a different state when the "
              f"scan was taken than when the capture was. Nothing below is "
              f"worth reading, and no setting repairs it: re-take the scan "
              f"and the captures back to back.")
    if score < 0.75:
        print("[WARN ] the parcel silhouettes correlate poorly. Either these "
              "are not the same arrangement, or one cloud is missing its "
              "parcels. Nothing below can repair that.")
    Ls = apply_T(Tc, Ls)

    def crop(P):
        return P[(np.abs(P[:, 0]) < 1.6) & (np.abs(P[:, 1]) < 0.62)
                 & (P[:, 2] > -0.06) & (P[:, 2] < 0.75)]

    Ti, d = icp(crop(Ls)[::3], crop(Lr), planar=planar)
    tilt = out_of_plane_deg(Ti[:3, :3])
    if verbose:
        print(f"[frame] icp residual median {np.median(d) * 1e3:.1f} mm  "
              f"rms {np.sqrt(float((d ** 2).mean())) * 1e3:.1f} mm  "
              f"yaw {np.degrees(np.arctan2(Ti[1, 0], Ti[0, 0])):+.2f} deg  "
              f"out of plane {tilt:.2f} deg")

    def frame_T(bf):
        R = np.column_stack([bf["x_axis"], bf["y_axis"], -bf["normal"]])
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ bf["centre"]
        return T

    T = compose(frame_T(bS), Tc, Ti, np.linalg.inv(frame_T(bR)))
    return T, {"score": float(score),
               "coarse_yaw_deg": float(yaw),
               "parcels_rig": counts[0], "parcels_scan": counts[1],
               "icp_median_mm": float(np.median(d) * 1e3),
               "icp_rms_mm": float(np.sqrt(float((d ** 2).mean())) * 1e3),
               "icp_out_of_plane_deg": tilt,
               "planar_icp": bool(planar),
               "deck_scan_m": float(dS), "deck_rig_m": float(dR),
               "belt_width_scan_mm": round(bS["width_m"] * 1e3, 1),
               "belt_width_rig_mm": round(bR["width_m"] * 1e3, 1)}


def probe_planes(P, n_planes=6, thr=0.012, voxel_m=0.008):
    """The largest planes and where they sit, for choosing a deck seed.

    The same idea as box_segment.probe: peel the planes off one at a time and
    print the distance to each. The deck is the large, near-horizontal one
    nearest the cameras; the floor is the one roughly 0.8 m beyond it.
    """
    Q = voxel(P, voxel_m)
    print(f"{len(P)} points, {len(Q)} after subsampling\n")
    print(f"{'#':>2} {'distance_m':>11} {'points':>9} {'share':>7} "
          f"{'tilt_to_1_deg':>14}")
    rem = Q
    first = None
    for i in range(max(1, n_planes)):
        if len(rem) < 500:
            break
        n, d = largest_plane(rem, thr)
        inl = np.abs(rem @ n - d) <= thr
        tilt = (0.0 if first is None else
                float(np.degrees(np.arccos(
                    np.clip(abs(float(n @ first)), 0, 1)))))
        if first is None:
            first = n
        print(f"{i:>2} {d:>11.3f} {int(inl.sum()):>9} "
              f"{inl.sum() / len(Q):>6.1%} {tilt:>14.1f}")
        rem = rem[~inl]
    print("\nThe deck is the large near-horizontal plane nearest the cameras. "
          "Pass its\ndistance as --deck-rig. The floor is usually the larger "
          "one about 0.8 m beyond.")


def save_transform(path, T, meta=None):
    Path(path).write_text(json.dumps(
        {"frame": "scan_to_rig_world",
         "note": "x_rig = T[:3,:3] @ x_scan + T[:3,3]; rigid by construction",
         "T": np.asarray(T, dtype=float).tolist(),
         "meta": meta or {}}, indent=2))


def load_transform(path):
    data = json.loads(Path(path).read_text())
    T = np.asarray(data["T"], dtype=float)
    if T.shape != (4, 4):
        raise SystemExit(f"{path}: T is not 4x4")
    det = float(np.linalg.det(T[:3, :3]))
    if abs(det - 1.0) > 1e-4:
        raise SystemExit(f"{path}: rotation determinant {det:.6f} is not 1, so "
                         f"this is not a rigid transform")
    return T, data.get("meta", {})


# --------------------------------------------------------------------------
# rig-side clouds
# --------------------------------------------------------------------------

def pixel_rays(shape, K):
    h, w = shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def load_capture_depths(cap: Path, cameras, raw=True):
    """The arrays da3_stream wrote for one capture.

    raw=True reads depth_raw_<cam>.npy, which is what a correction must be
    fitted against. Fitting against depth_<cam>.npy would fit a correction to
    a cloud that has already had one applied, and the two would compound.
    """
    cap = Path(cap)
    stem = "depth_raw" if raw else "depth"
    out = {}
    for n in cameras:
        d = cap / f"{stem}_{n}.npy"
        if not d.exists():
            if raw and (cap / f"depth_{n}.npy").exists():
                raise SystemExit(
                    f"{cap} holds depth_{n}.npy but no depth_raw_{n}.npy. The "
                    f"correction must be fitted against RAW depth; re-capture "
                    f"with --save-npy, or with the capture key, which writes "
                    f"both.")
            continue
        K = cap / f"K_{n}.npy"
        E = cap / f"E_{n}.npy"
        if not (K.exists() and E.exists()):
            raise SystemExit(f"{cap}: {n} has depth but no K/E arrays")
        out[n] = {"depth": np.load(d).astype(np.float64),
                  "K": np.asarray(np.load(K), dtype=float),
                  "E": np.asarray(np.load(E), dtype=float)}
        c = cap / f"conf_{n}.npy"
        if c.exists():
            out[n]["conf"] = np.load(c).astype(np.float64)
    if not out:
        raise SystemExit(f"no depth arrays found in {cap}")
    return out


def edge_mask(depth, rel_thresh=0.02, dilate=1):
    """Pixels away from a depth discontinuity.

    The same test da3_stream.backproject applies. A depth model smooths across
    a step over several pixels, and those samples land between a parcel top and
    the belt as a hanging veil: points in free space that belong to no surface.
    """
    d = np.asarray(depth, dtype=float)
    finite = np.isfinite(d) & (d > 0)
    filled = np.where(finite, d, np.nanmedian(d[finite]) if finite.any() else 1.0)
    gy, gx = np.gradient(filled)
    smooth = (np.hypot(gx, gy) / np.maximum(filled, 1e-6) <= rel_thresh) & finite
    if dilate > 0:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        smooth = cv2.erode(smooth.astype(np.uint8), k).astype(bool)
    return smooth


def incidence_mask(pts_cam, max_deg):
    du = np.zeros_like(pts_cam)
    dv = np.zeros_like(pts_cam)
    du[:, 1:-1] = pts_cam[:, 2:] - pts_cam[:, :-2]
    dv[1:-1, :] = pts_cam[2:, :] - pts_cam[:-2, :]
    n = np.cross(du, dv)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    r = pts_cam / np.maximum(np.linalg.norm(pts_cam, axis=-1, keepdims=True),
                             1e-12)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(n * r, axis=-1)), 0.0, 1.0)))
    ang[~np.isfinite(ang)] = 90.0
    return ang <= max_deg


def world_from_depth(depth, K, E, conf=None, conf_percentile=20.0,
                     edge_thresh=0.02, edge_dilate=1, max_incidence=70.0):
    """Back-project one depth map, filtered as the streamer filters it.

    Depth is Z along the optical axis, not distance along the ray.

    The filtering is not optional detail. A raw depth map at process_res 1008
    carries roughly 850,000 samples per view, and a large minority of them are
    the smoothing veil around every parcel edge, sitting in free space. ICP
    trims its correspondence set, but once more than about a third of the
    points belong to no surface the trim stops protecting the fit: the veil
    pulls the alignment and inflates the residual. Registering the unfiltered
    union against a scanner cloud is therefore not a harder version of the same
    problem, it is a different and worse-posed one.
    """
    z = np.asarray(depth, dtype=float)
    valid = np.isfinite(z) & (z > 0)
    drops = {}

    if conf is not None and conf_percentile > 0 and valid.any():
        thr = float(np.percentile(np.asarray(conf, dtype=float)[valid],
                                  conf_percentile))
        before = int(valid.sum())
        valid &= np.asarray(conf, dtype=float) >= thr
        drops["conf"] = before - int(valid.sum())

    if edge_thresh > 0:
        before = int(valid.sum())
        valid &= edge_mask(z, edge_thresh, edge_dilate)
        drops["edge"] = before - int(valid.sum())

    pts = pixel_rays(z.shape, K) * np.where(valid, z, 0.0)[..., None]

    if max_incidence < 90:
        before = int(valid.sum())
        valid &= incidence_mask(pts, max_incidence)
        drops["grazing"] = before - int(valid.sum())

    E = np.asarray(E, dtype=float)
    return (pts[valid] - E[:3, 3]) @ E[:3, :3], drops   # R^T (x_cam - t)


def rig_cloud(arrays, verbose=True, **filters):
    """The union of the per-view clouds, filtered, for registration.

    Filtered rather than raw on purpose: what should be registered against a
    scanner is the surfaces the pipeline actually produces, not every sample
    the model emitted. Pass max_incidence=90, edge_thresh=0 and
    conf_percentile=0 to get the old unfiltered behaviour.
    """
    pts, total = [], 0
    for name, a in arrays.items():
        p, drops = world_from_depth(a["depth"], a["K"], a["E"],
                                    a.get("conf"), **filters)
        total += int(np.isfinite(a["depth"]).sum())
        pts.append(p)
        if verbose:
            kept = len(p)
            detail = "  ".join(f"{k} -{v}" for k, v in drops.items())
            print(f"[filt ] {name:<7} kept {kept:>8} of "
                  f"{int((np.isfinite(a['depth']) & (a['depth'] > 0)).sum()):>8}"
                  f"   {detail}")
    return np.concatenate(pts) if pts else np.empty((0, 3))