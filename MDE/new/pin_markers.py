#!/usr/bin/env python3
"""
pin_markers.py

Align the rc_viscore ground truth to the rig frame using the ArUco markers that
appear in both, and read the reconstruction's gauge off the same fit.

WHY THIS REPLACES THE SEARCH
----------------------------
The conveyor constrains five degrees of freedom and leaves position along the
belt free, because a belt looks the same wherever you slide it. Recovering that
last one by sliding against the parcels turned out to be fragile: scaling about
the origin also shifts the cloud along the belt, so scale and slide trade
against each other and the optimum lies in a valley rather than a basin. The
search duly slid to -541 mm against a +/-600 mm limit.

Markers remove the problem rather than manage it. A marker is an exact,
identified correspondence: the same physical square, named by its id, in both
clouds. Six of them over-determine a similarity, and the residual says plainly
whether the fit is any good, which no match-fraction score ever did.

HOW EACH SIDE IS MEASURED
-------------------------
THE RECONSTRUCTION. Markers are detected in proc_<cam>.png, which is the image
DA3 actually saw and which K_<cam>.npy belongs to. Each marker's interior
pixels give a median depth, and that back-projects through the CALIBRATED
extrinsics into the rig frame. The result carries the reconstruction's scale
error, which is the point: that error is what we are here to measure.

THE GROUND TRUTH. There is no image, so one is made: the cloud is rendered
orthographically onto the deck plane, nearest-surface-wins, at a fixed
millimetres per pixel. Detection then runs on that render, and each marker's
pixel maps back to the 3D point that painted it. An orthographic render is the
right choice because it is metric everywhere, so a marker is the same size in
pixels wherever it sits on the belt.

WHAT IT WRITES
--------------
  gt_align.json   the ground truth into the rig frame, rigid, from matched
                  markers rather than from a slide search
  marker_fit.json every matched marker, its residual, and the similarity scale

The scale reported here is independent of the closed-form one that
pin_alignment.py derives from the deck plane. Two estimates that agree are
worth more than either alone; two that disagree mean something is wrong and it
is better to know.

EXAMPLE
-------
    python3 pin_markers.py --gt gt/pointcloud_20260826_095616.ply \\
        --run runs_da3/scene_b/center+left+top+right_res1008 \\
        --belt-corners belt_corners.json --dict auto

Keep this file beside rigkit.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv is required")

try:
    import rigkit
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"rigkit.py must sit beside this file: {exc}")

try:
    import da3_stream as ds
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"da3_stream.py must import cleanly: {exc!r}")


# The rig's cameras in canonical order, matching rigkit.CAMS, so the
# gauge table's columns line up with the other tools' output.
CAMS = ["center", "left", "top", "right"]

DICTS = ["DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250", "DICT_APRILTAG_36h11",
         "DICT_ARUCO_ORIGINAL", "DICT_7X7_250"]


def cams_of(run):
    """Camera names from a run directory called e.g. center+left+top_res1008.

    Split on the literal "_res" rather than by regular expression: an escaping
    slip in the pattern turns it into a match for a backslash followed by d,
    which silently strips nothing and hands "right_res1008" downstream as a
    camera name. Doing it without a regex removes that failure mode.
    """
    name = Path(run).name
    if "_res" in name:
        name = name.rsplit("_res", 1)[0]
    cams = [c for c in name.replace(",", "+").split("+") if c]
    bad = [c for c in cams if c not in CAMS]
    if bad:
        raise SystemExit(
            f"{Path(run).name} parses to camera(s) {bad}, which are not in "
            f"{CAMS}. If one of them still carries a _res suffix, the copy of "
            f"this file being run is not the current one -- see the [file ] "
            f"line above for which one that is.")
    return cams


def load_cloud_rgb(path):
    """xyz and rgb from a binary or ascii PLY.

    rigkit.load_cloud drops colour, and colour is what the detector needs.
    """
    raw = open(path, "rb").read()
    i = raw.find(b"end_header")
    if i < 0:
        raise SystemExit(f"{path}: not a PLY")
    hdr = raw[:i].decode("ascii", "replace")
    j = raw.find(b"\n", i) + 1
    n = int(re.search(r"element vertex (\d+)", hdr).group(1))
    ty = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
          "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
          "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
          "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2"}
    fields = []
    for ln in hdr.splitlines():
        if ln.startswith("property") and "list" not in ln:
            _, t, nm = ln.split()[:3]
            fields.append((nm, ty[t]))
    if "ascii" in hdr:
        arr = np.loadtxt(path, skiprows=hdr.count("\n") + 1, max_rows=n)
        names = [f[0] for f in fields]
        xyz = arr[:, [names.index(k) for k in ("x", "y", "z")]]
        rgb = (arr[:, [names.index(k) for k in ("red", "green", "blue")]]
               if "red" in names else None)
    else:
        d = np.frombuffer(raw, dtype=np.dtype(fields), count=n, offset=j)
        xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
        rgb = (np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8)
               if "red" in d.dtype.names else None)
    if rgb is None:
        raise SystemExit(f"{path} carries no colour; the markers cannot be "
                         f"detected without it")
    return xyz, np.asarray(rgb, np.uint8)


def belt_frame(corners):
    C = np.asarray(corners, float)
    c = C.mean(0)
    _, _, Vt = np.linalg.svd(C - c, full_matrices=False)
    a, b = Vt[0], Vt[1]
    ang = np.arctan2((C - c) @ b, (C - c) @ a)
    C = C[np.argsort(ang)]
    e = [C[(i + 1) % 4] - C[i] for i in range(4)]
    lens = [float(np.linalg.norm(v)) for v in e]
    i0 = int(np.argmax(lens))
    ex = e[i0] / lens[i0] - e[(i0 + 2) % 4] / lens[(i0 + 2) % 4]
    ex /= np.linalg.norm(ex)
    ey = (e[(i0 + 1) % 4] / lens[(i0 + 1) % 4]
          - e[(i0 + 3) % 4] / lens[(i0 + 3) % 4])
    ey /= np.linalg.norm(ey)
    n = np.cross(ex, ey)
    n /= np.linalg.norm(n)
    if float(n @ (np.zeros(3) - c)) < 0:
        n = -n              # towards the sensor, so nearest-wins means nearest
    return c, ex, np.cross(n, ex), n


def sensor_frame(P):
    """Project along the ground truth's own z, which is the sensor's optical
    axis.

    This is the view the markers were actually photographed from, so they are
    face-on and nothing that was behind them can occlude them. Rendering along
    the DECK normal instead looks down on the whole cell, which on this rig
    puts the overhead structure between the camera and the belt, and squashes
    any marker not lying in the deck plane.
    """
    n = np.array([0.0, 0.0, -1.0])          # towards the sensor at the origin
    ex = np.array([1.0, 0.0, 0.0])
    return np.asarray(P, float).mean(0), ex, np.cross(n, ex), n


def ortho_render(P, rgb, frame, mm_per_px, pad=0.25, fill=2):
    """Nearest-surface-wins orthographic render onto the deck plane.

    Orthographic rather than perspective because it is metric everywhere: a
    marker covers the same number of pixels wherever it sits on the belt, so one
    detector setting works across the whole conveyor.
    """
    c, ex, ey, n = frame
    # A left-handed basis renders the scene MIRRORED, and an ArUco code is
    # chirality-sensitive: a mirrored marker simply does not decode, at any
    # resolution. This cost a full sweep returning zero detections.
    if float(ex @ np.cross(ey, n)) < 0:
        ey = -ey
    rel = P - c
    u, v, h = rel @ ex, rel @ ey, rel @ n
    cell = mm_per_px / 1e3
    u0, u1 = u.min() - pad, u.max() + pad
    v0, v1 = v.min() - pad, v.max() + pad
    W = int((u1 - u0) / cell) + 1
    H = int((v1 - v0) / cell) + 1
    if W * H > 80_000_000:
        raise SystemExit("render too large; raise --mm-per-px")
    iu = np.clip(((u - u0) / cell).astype(np.int64), 0, W - 1)
    iv = np.clip(((v - v0) / cell).astype(np.int64), 0, H - 1)
    flat = iv * W + iu
    order = np.argsort(h)                 # ascending, last write is highest
    img = np.zeros((H * W, 3), np.uint8)
    idx = np.full(H * W, -1, np.int64)
    img[flat[order]] = rgb[order]
    idx[flat[order]] = order
    img = img.reshape(H, W, 3)
    idx = idx.reshape(H, W)

    # A cloud sampled at random leaves Poisson holes, and ArUco will not decode
    # a pockmarked square however large it is: six synthetic markers rendered
    # without this gave two detections. Empty pixels are filled from their
    # neighbours, which closes the speckle without touching the marker edges
    # that carry the code.
    for _ in range(max(0, fill)):
        empty = idx < 0
        if not empty.any():
            break
        # MEAN of the valid neighbours, not the maximum. cv2.dilate takes the
        # max, so every pass eats the black modules of a tag into the white
        # ones: on a dense render that is harmless, on one with a third of its
        # pixels empty it turns a 36h11 into a white blob and nothing decodes.
        valid = (~empty).astype(np.float32)
        num = cv2.blur(img.astype(np.float32) * valid[..., None], (3, 3))
        den = cv2.blur(valid, (3, 3))
        ok = empty & (den > 1e-6)
        img[ok] = np.clip(num[ok] / den[ok][:, None], 0, 255).astype(np.uint8)
        # the index map is nearest-neighbour, so max is right for it
        gidx = cv2.dilate(idx.astype(np.float32), np.ones((3, 3), np.uint8))
        idx[ok] = gidx[ok].astype(np.int64)
    return img, idx, (u0, v0, cell)


def detect(img, dicts, verbose=False, upsample=1):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    if upsample > 1:
        # No new information, but a 36h11 rendered at three pixels per module
        # gives the quad detector nothing to refine against. Interpolating up
        # is the standard remedy for a tag that is merely small.
        grey = cv2.resize(grey, None, fx=upsample, fy=upsample,
                          interpolation=cv2.INTER_CUBIC)
    # A render built from a point cloud carries the cloud's own shading, so the
    # local contrast a tag needs may be there while the global range is not.
    variants = [("raw", grey),
                ("clahe", cv2.createCLAHE(3.0, (8, 8)).apply(grey)),
                ("otsu", cv2.threshold(grey, 0, 255,
                                       cv2.THRESH_BINARY
                                       + cv2.THRESH_OTSU)[1])]
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    par.adaptiveThreshWinSizeMax = 53
    best, counts = ({}, None, 0), {}
    for name in dicts:
        if not hasattr(cv2.aruco, name):
            continue
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        det = cv2.aruco.ArucoDetector(dic, par)
        for vname, im in variants:
            corners, ids, _ = det.detectMarkers(im)
            n = 0 if ids is None else len(ids)
            counts[f"{name}/{vname}"] = n
            if n > best[2]:
                best = ({int(i): c.reshape(4, 2) / upsample
                         for i, c in zip(ids.ravel(), corners)}, name, n)
    if verbose and counts:
        print("        " + "  ".join(f"{k.replace('DICT_', '')}:{v}"
                                     for k, v in counts.items() if v))
    return best


def gt_markers(gt, rgb, frame, args):
    keep = np.ones(len(gt), bool)
    if args.height_min is not None or args.height_max is not None:
        h = (gt - frame[0]) @ frame[3]
        if args.height_min is not None:
            keep &= h >= args.height_min
        if args.height_max is not None:
            keep &= h <= args.height_max
        print(f"[gt   ] height band keeps {int(keep.sum())} of {len(gt)} points")
    gt, rgb = gt[keep], rgb[keep]
    img, idx, (u0, v0, cell) = ortho_render(gt, rgb, frame, args.mm_per_px,
                                            fill=args.fill)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = g[idx >= 0]
    filled = float(np.count_nonzero(idx >= 0)) / idx.size
    bright = int(np.count_nonzero(v > 200))
    print(f"[gt   ] {filled * 100:.0f} per cent of the render carries a point")
    print(f"[gt   ] grey p5 {np.percentile(v, 5):.0f}  p50 "
          f"{np.percentile(v, 50):.0f}  p95 {np.percentile(v, 95):.0f}  p99.9 "
          f"{np.percentile(v, 99.9):.0f}  max {v.max():.0f}")
    # A printed tag is white paper: if almost nothing in the render is bright,
    # the tags are not in it and no detector setting will find them.
    tag_px = (0.140 / (args.mm_per_px / 1e3)) ** 2
    print(f"[gt   ] {bright} pixels brighter than 200, which is "
          f"{bright / max(tag_px, 1):.1f} tag-sized patches' worth "
          f"({tag_px:.0f} px per 140 mm sheet at this sampling)")
    if bright < 0.3 * tag_px:
        print("[gt   ] almost nothing in the render is white. Either the tags "
              "are occluded along this view axis, or the cloud's colour is not "
              "reaching them. Try --view deck, or open the render and look.")
    found, dic, n = detect(img, args.dicts, verbose=True,
                           upsample=args.upsample)
    if n == 0:
        # Belt and braces: if the cloud's own frame is unusual the render may
        # still come out mirrored, so try it the other way before giving up.
        c_, ex_, ey_, n_ = frame
        img2, idx2, _ = ortho_render(gt, rgb, (c_, ex_, -ey_, n_),
                                     args.mm_per_px, fill=args.fill)
        found2, dic2, n2 = detect(img2, args.dicts,
                                  upsample=args.upsample)
        if n2 > 0:
            print(f"[gt   ] nothing found, but {n2} in the mirrored render: "
                  f"the frame was left-handed. Using the mirror.")
            img, idx, found, dic, n = img2, idx2, found2, dic2, n2
    print(f"[gt   ] render {img.shape[1]}x{img.shape[0]} at "
          f"{args.mm_per_px:.1f} mm/px -> {n} marker(s)"
          + (f" with {dic}" if dic else ""))
    if args.save_render:
        cv2.imwrite(str(args.save_render), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[gt   ] render written to {args.save_render}")
    out = {}
    H, W = idx.shape
    for mid, c in found.items():
        pts = []
        for px, py in c:
            iu, iv = int(round(px)), int(round(py))
            win = idx[max(iv - 2, 0):iv + 3, max(iu - 2, 0):iu + 3].ravel()
            win = win[win >= 0]
            if len(win):
                pts.append(np.median(gt[win], axis=0))
        if len(pts) == 4:
            out[mid] = np.stack(pts)          # four corners, not one centre
    return out, dic


def calibrated_K(run, cams, args):
    """The rig's own rectified intrinsics, scaled to the processed depth grid.

    K_<cam>.npy holds whatever DA3 returned, and PnP on a tag of known size is a
    direct test of it: if the tag comes back at 1.59 times too close, the focal
    length is 1.59 times too small and the reconstruction's apparent "depth
    scale error" is really a focal length error. da3_stream checks the same
    quantity at warm-up as fx_ratio and warns outside 0.98 to 1.02.

    Rebuilt through build_rect_plan so the rectification matches the one the
    capture used, then scaled by the ratio of the depth grid to the rectified
    width, with the pixel-centre convention scale_K applies.
    """
    intr = {c: ds.load_intrinsics(Path(args.calib_dir), c) for c in cams}
    size = intr[cams[0]]["size"]
    plan = ds.build_rect_plan(intr, cams, size, args.rect_mode, True)
    out = {}
    print("[intr ] calibrated rectified intrinsics against what DA3 returned")
    for c in cams:
        dp = run / f"depth_{c}.npy"
        if not dp.exists():
            continue
        dh, dw = np.load(dp, mmap_mode="r").shape[-2:]
        cw = plan[c]["crop"][2]
        Kc = ds.scale_K(plan[c]["K"], dw / cw)
        out[c] = Kc
        kp = run / f"K_{c}.npy"
        if kp.exists():
            Kd = np.load(kp).astype(float)
            r = float(Kd[0, 0] / Kc[0, 0])
            flag = "" if 0.98 < r < 1.02 else "   <- FOCAL LENGTH IS WRONG"
            print(f"[intr ] {c:<7} calibrated fx {Kc[0, 0]:8.1f}   DA3 "
                  f"{Kd[0, 0]:8.1f}   ratio {r:6.4f}{flag}")
    return out


def recon_markers_pnp(run, cams, args):
    """Tag poses in the rig frame from the IMAGE alone, by solvePnP.

    A 36h11 tag is a printed square of known size, so four image corners plus
    the intrinsics determine its pose outright. No depth is read, which means
    the result carries none of DA3's gauge error and none of its layering: it
    is metric because the tag is 120 mm, not because the reconstruction is.

    That matters for more than convenience. A tag located through depth moves
    with the run's gauge -- scaling a point at y = -0.5 by 1.64 shifts it 320 mm
    along the belt -- so a slide derived that way is per-run. Located by PnP it
    is absolute, one placement of the ground truth serves every run, and each
    run's along-belt error stays in the measurement where it belongs.

    proc_<cam>.png is rectified, so the distortion coefficients are zero: the
    remap already removed them and passing them again would correct twice.
    """
    ext = ds.load_extrinsics(Path(args.calib_dir), cams, args.reference)
    K_cal = calibrated_K(run, cams, args) if args.intrinsics == "calib" else None
    h = args.tag_size_mm / 2e3
    # The order SOLVEPNP_IPPE_SQUARE documents, which is also the order
    # detectMarkers returns corners in: top-left, top-right, bottom-right,
    # bottom-left. Any other ordering silently returns a pose metres away.
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    out, per_cam, rms, rejected = {}, {}, {}, {}
    for cam in cams:
        ip, kp = run / f"proc_{cam}.png", run / f"K_{cam}.npy"
        if not (ip.exists() and kp.exists()):
            continue
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        K_da3 = np.load(kp).astype(float)
        K = K_cal[cam] if K_cal is not None else K_da3
        E = np.asarray(ext[cam]["E"], float)
        found, dic, n = detect(img, args.dicts, upsample=args.recon_upsample)
        per_cam[cam] = n
        for mid, c in found.items():
            ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float64), K,
                                          np.zeros(5),
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                rejected.setdefault(mid, []).append((cam, None, "PnP failed"))
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
            e = float(np.sqrt(np.mean(np.sum(
                (proj.reshape(-1, 2) - c) ** 2, axis=1))))
            if e > args.max_pnp_rms_px:
                rejected.setdefault(mid, []).append((cam, round(e, 2),
                                                     "reprojection"))
                continue
            R, _ = cv2.Rodrigues(rvec)
            cam_pts = obj @ R.T + np.asarray(tvec).ravel()
            out.setdefault(mid, []).append(ds.to_world(cam_pts, E))
            rms.setdefault(mid, []).append(e)
    print("[recon] tags per camera: "
          + "  ".join(f"{k} {v}" for k, v in per_cam.items()))
    # Every id that was seen but did not make it, and why. Without this a tag
    # detected by all four cameras can vanish from the list with no trace, and
    # a second usable correspondence is exactly what settles the yaw when one
    # tag cannot.
    for mid in sorted(set(rejected) - set(out)):
        why = rejected[mid]
        print(f"[recon] tag {mid:>3d} REJECTED in {len(why)} camera(s): "
              + ", ".join(f"{c} ({r} {w})" for c, r, w in why[:4]))
    seen_ids = sorted(set(out) | set(rejected))
    if len(seen_ids) > len(out):
        print(f"[recon] ids seen in the images: {seen_ids}; usable: "
              f"{sorted(out)}. Raise --max-pnp-rms-px to admit more, but a "
              f"tag that reprojects badly is one whose corners were misread")
    for mid in sorted(out):
        P = np.stack(out[mid])
        spread = float(np.linalg.norm(P.mean(1) - P.mean(1).mean(0),
                                      axis=1).max()) * 1e3
        print(f"[recon] tag {mid:>3d} from {len(P)} camera(s), reprojection "
              f"{np.mean(rms[mid]):.2f} px, the cameras place its centre "
              f"within {spread:.1f} mm of each other")
    return ({m: np.stack(v).mean(axis=0) for m, v in out.items()},
            {m: len(v) for m, v in out.items()})


def recon_markers(run, cams, args):
    """Marker centres in the rig frame, through the depth DA3 produced."""
    ext = ds.load_extrinsics(Path(args.calib_dir), cams, args.reference)
    out, per_cam = {}, {}
    for cam in cams:
        ip = run / f"proc_{cam}.png"
        dp = run / f"depth_{cam}.npy"
        kp = run / f"K_{cam}.npy"
        if not (ip.exists() and dp.exists() and kp.exists()):
            continue
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        depth = np.load(dp).astype(float)
        K = np.load(kp).astype(float)
        E = np.asarray(ext[cam]["E"], float)
        found, dic, n = detect(img, args.dicts,
                               upsample=args.recon_upsample)
        per_cam[cam] = n
        for mid, c in found.items():
            mask = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.round(c).astype(np.int32), 1)
            # eroded, so the depth used is the marker's face and not the
            # smoothed ramp the model puts across its border
            if args.erode > 0:
                k = np.ones((2 * args.erode + 1,) * 2, np.uint8)
                mask = cv2.erode(mask, k)
            sel = (mask > 0) & np.isfinite(depth) & (depth > 0)
            if int(sel.sum()) < args.min_marker_px:
                continue
            # One depth for the whole tag: it is a rigid printed square, so
            # its corners share a plane, and reading depth at a corner reads it
            # across the tag's edge where the model has smoothed.
            z = float(np.median(depth[sel]))
            xy = np.stack([(c[:, 0] - K[0, 2]) / K[0, 0] * z,
                           (c[:, 1] - K[1, 2]) / K[1, 1] * z,
                           np.full(4, z)], axis=1)
            out.setdefault(mid, []).append(ds.to_world(xy, E))
    print("[recon] markers per camera: "
          + "  ".join(f"{k} {v}" for k, v in per_cam.items()))
    return {m: np.mean(np.stack(v), axis=0) for m, v in out.items()}, \
           {m: len(v) for m, v in out.items()}


def drop_flipped(cands, gt_all, dst_f, sample=200000, band=(0.06, 0.60)):
    """Discard the candidates whose normal points into the deck.

    A tag lying in the deck plane is almost unmoved by a reflection through that
    plane, so its corners cannot separate the two normal signs: with one tag the
    wrong sign won by a hair and put every check point 800 mm out. The parcels
    settle it without ambiguity, because they stand towards the cameras. Under a
    flipped normal they would have to stand into the conveyor.
    """
    c, n = dst_f["centre"], dst_f["n"]
    rng = np.random.default_rng(0)
    P = gt_all[rng.choice(len(gt_all), min(len(gt_all), sample), replace=False)]
    lo, hi = band
    keep = []
    for label, T in cands:
        h = (apply(P, T) - c) @ n
        up = int(np.count_nonzero((h > lo) & (h < hi)))
        dn = int(np.count_nonzero((h < -lo) & (h > -hi)))
        keep.append((up / max(dn, 1), label, T, up, dn))
    keep.sort(reverse=True)
    best = keep[0][0]
    out = [(l, T) for r, l, T, _, _ in keep if r > 1.0]
    for r, l, T, up, dn in keep:
        print(f"    {l:22s} {up:>8d} above the deck, {dn:>8d} below   "
              f"ratio {r:6.2f}" + ("   kept" if r > 1.0 else "   dropped"))
    return out or [(keep[0][1], keep[0][2])]


def slide_from_markers(g_m, r_m, dst_f, cands, gt_all, scale):
    """The one degree of freedom the conveyor cannot give, taken from a tag.

    The belt fixes the deck plane, the yaw about it and the position across it.
    What is left is position ALONG the belt, and a single AprilTag pins that
    outright: shift the ground truth until the tag sits on the tag.

    The candidates differ by a 180 degree yaw and by the normal's sign, and a
    marker separates them where the parcels could not, because the residual
    PERPENDICULAR to the belt axis is only small for the right one. Corners are
    used rather than centres, so one shared tag is four correspondences and the
    orientation is over-determined.
    """
    ex, ey, n = dst_f["ex"], dst_f["ey"], dst_f["n"]
    both = sorted(set(g_m) & set(r_m))
    R = np.vstack([r_m[m] for m in both]) * scale
    rows = []
    for label, T in cands:
        G = np.vstack([apply(g_m[m], T) for m in both])
        d = R - G
        slide = float(np.median(d @ ex))
        perp = np.stack([d @ ey, d @ n], axis=1)
        rows.append((float(np.sqrt((perp ** 2).sum(1)).mean()), slide,
                     label, T,
                     float(np.std(d @ ex))))
    rows.sort()
    return rows, both


def apply(P, T):
    T = np.asarray(T, float)
    return np.asarray(P) @ T[:3, :3].T + T[:3, 3]


def gauge_from_tags(run, cams, args):
    """The depth gauge per camera, from one tag, two ways.

    A tag's position can be had twice over. solvePnP uses only the image and
    the tag's printed size, so it is metric and knows nothing about DA3's
    depth. Reading the depth map at the same pixels uses DA3's depth and
    nothing else. The ratio of the two ranges from the camera centre is the
    depth gauge:

        s_cam = |P_pnp - C| / |P_depth - C|

    No plane is fitted, no deck is identified, no ground truth is consulted,
    and it is PER CAMERA, so it measures the layering as well as the scale.
    Every previous attempt at this quantity went through finding the conveyor
    in the reconstruction, and four schemes running the deck's median height,
    its largest plane, its width and its cross-camera consistency each picked
    the wrong surface at least once.
    """
    ext = ds.load_extrinsics(Path(args.calib_dir), cams, args.reference)
    K_cal = calibrated_K(run, cams, args) if args.intrinsics == "calib" else None
    h = args.tag_size_mm / 2e3
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    out = {}
    for cam in cams:
        ip, kp, dp = (run / f"proc_{cam}.png", run / f"K_{cam}.npy",
                      run / f"depth_{cam}.npy")
        if not (ip.exists() and kp.exists() and dp.exists()):
            continue
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        K = K_cal[cam] if K_cal is not None else np.load(kp).astype(float)
        depth = np.load(dp).astype(float)
        found, _, _ = detect(img, args.dicts, upsample=args.recon_upsample)
        ratios = []
        for mid, c in found.items():
            ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float64), K,
                                          np.zeros(5),
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            r_pnp = float(np.linalg.norm(np.asarray(tvec).ravel()))
            mask = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.round(c).astype(np.int32), 1)
            if args.erode > 0:
                k = np.ones((2 * args.erode + 1,) * 2, np.uint8)
                mask = cv2.erode(mask, k)
            sel = (mask > 0) & np.isfinite(depth) & (depth > 0)
            if int(sel.sum()) < args.min_marker_px:
                continue
            z = float(np.median(depth[sel]))
            cu, cvv = c.mean(0)
            xy = np.array([(cu - K[0, 2]) / K[0, 0] * z,
                           (cvv - K[1, 2]) / K[1, 1] * z, z])
            ratios.append((mid, r_pnp / float(np.linalg.norm(xy))))
        if ratios:
            out[cam] = {"scale": float(np.median([r for _, r in ratios])),
                        "tags": {int(m): round(float(r), 5) for m, r in ratios}}
    return out


def umeyama(src, dst, with_scale=True):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / n)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1.0
    R = U @ W @ Vt
    s = (float((sig * np.diag(W)).sum() / float((S ** 2).sum() / n))
         if with_scale else 1.0)
    t = mu_d - s * R @ mu_s
    resid = dst - (s * (src @ R.T) + t)
    return s, R, t, np.sqrt((resid ** 2).sum(1))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Align the ground truth to the rig frame using the ArUco "
                    "markers visible in both.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", type=Path, default=None,
                    help="the ground-truth cloud. Not needed by --gauge, "
                         "which measures the depth scale against a tag's own "
                         "printed size and consults nothing else")
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="with --gauge, do every run under this root")
    ap.add_argument("--belt-corners", type=Path,
                    default=Path("belt_corners.json"),
                    help="the picked deck corners, used only to define the "
                         "plane the ground truth is rendered onto")
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/home/jetson/Projects/Calibration_4_5/results"))
    ap.add_argument("--reference", default="center")
    ap.add_argument("--dict", default="auto",
                    help="ArUco dictionary, or auto to try several and keep "
                         "whichever finds the most")
    ap.add_argument("--mm-per-px", type=float, default=5.0,
                    help="ground sampling of the orthographic render. Work it "
                         "out from the tag rather than guessing: an AprilTag "
                         "36h11 is a 6x6 payload inside a one-module border, "
                         "so 8 modules across. A 120 mm tag is then 15 mm per "
                         "module, so 3 px per module at 5.0. But module count "
                         "is only half the trade: the render also needs "
                         "several POINTS per pixel, or it samples the cloud's "
                         "own noise. At this rig's 3 mm spacing the optimum is "
                         "about 5 points per pixel, which is 5.0 mm/px; 2.0 "
                         "finds nothing at all despite having 7.5 px per "
                         "module. Use --sweep rather than reasoning from "
                         "either limit alone")
    ap.add_argument("--gt-tags", type=Path, default=None,
                    help='json of hand-picked tag centres in the GROUND TRUTH '
                         'frame, {"0": [x,y,z], "3": [x,y,z]}. Skips the '
                         'render and its detection entirely. A tag is a sharp '
                         '120 mm fiducial and one click on it is worth more '
                         'than any amount of correlating near-periodic '
                         'cardboard; the reconstruction side finds the same '
                         'tags automatically, so this closes the loop')
    ap.add_argument("--gauge", action="store_true",
                    help="report the depth gauge per camera as the ratio of "
                         "the tag's PnP range to its depth-map range, and "
                         "write metric_fit.json. Needs no ground truth")
    ap.add_argument("--out-gauge", type=Path, default=Path("metric_fit.json"))
    ap.add_argument("--list-recon-tags", action="store_true",
                    help="print the tag centres found in the reconstruction, "
                         "in the rig frame, then exit. Use it to see which "
                         "ids to look for in the ground truth")
    ap.add_argument("--view", choices=("sensor", "deck"), default="sensor",
                    help="which direction to render the ground truth along. "
                         "'sensor' uses its own z, the optical axis, which is "
                         "how the markers were seen. 'deck' looks down the "
                         "conveyor normal, which is metric on the belt but "
                         "puts overhead structure in the way")
    ap.add_argument("--height-min", type=float, default=None,
                    help="drop points nearer than this along the view axis, "
                         "in metres, to clear structure in front of the belt")
    ap.add_argument("--height-max", type=float, default=None)
    ap.add_argument("--upsample", type=int, default=3,
                    help="interpolate the ground-truth render up by this "
                         "factor before detection. A 120 mm tag is only about "
                         "three pixels per module at the sampling this cloud's "
                         "density allows, and a quad detector needs room to "
                         "refine; upsampling adds no information but gives it "
                         "that room")
    ap.add_argument("--recon-upsample", type=int, default=2,
                    help="the same for the proc images, where a tag is seven "
                         "pixels per module at 1008 and half that at 504")
    ap.add_argument("--fill", type=int, default=2,
                    help="passes of neighbour-fill over the render. A cloud "
                         "sampled at random leaves holes and ArUco will not "
                         "decode a pockmarked square")
    ap.add_argument("--sweep", action="store_true",
                    help="try a range of --mm-per-px and report how many "
                         "markers each finds, then exit")
    ap.add_argument("--intrinsics", choices=("calib", "da3"), default="calib",
                    help="whose intrinsics PnP uses. 'calib' rebuilds the rig's "
                         "own rectified K and scales it to the depth grid; "
                         "'da3' uses K_<cam>.npy as returned. If the two "
                         "disagree, that disagreement IS the scale error and "
                         "is worth reporting rather than working around")
    ap.add_argument("--rect-mode", default="common",
                    choices=("common", "roi", "full"))
    ap.add_argument("--tag-size-mm", type=float, default=120.0,
                    help="printed side of the tag's black square. This is what "
                         "makes solvePnP metric, so measure it rather than "
                         "trusting the print dialogue")
    ap.add_argument("--no-pnp", dest="pnp", action="store_false", default=True,
                    help="locate tags through the depth map instead of by "
                         "solvePnP. Only useful for showing what the gauge "
                         "error does to them")
    ap.add_argument("--max-pnp-rms-px", type=float, default=3.0,
                    help="reject a tag whose pose reprojects worse than this")
    ap.add_argument("--erode", type=int, default=2,
                    help="pixels trimmed from a marker before its depth is "
                         "read, keeping the model's edge smoothing out of it")
    ap.add_argument("--min-marker-px", type=int, default=25)
    ap.add_argument("--min-markers", type=int, default=1,
                    help="markers that must match. One is enough when the "
                         "conveyor supplies the other five degrees of freedom, "
                         "because a tag carries four corners")
    ap.add_argument("--belt", type=Path, default=Path("belt.json"),
                    help="the frozen footprint, for the five degrees of "
                         "freedom the conveyor does constrain")
    ap.add_argument("--full-sim3", action="store_true",
                    help="ignore the conveyor and fit all seven degrees of "
                         "freedom from markers alone. Needs three or more")
    ap.add_argument("--max-resid-mm", type=float, default=60.0)
    ap.add_argument("--save-render", type=Path, default=Path("gt_ortho.png"))
    ap.add_argument("--out", type=Path, default=Path("gt_align.json"))
    ap.add_argument("--out-fit", type=Path, default=Path("marker_fit.json"))
    args = ap.parse_args()
    import time
    _f = Path(__file__).resolve()
    print(f"[file ] {_f}  modified "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_f.stat().st_mtime))}")
    args.dicts = DICTS if args.dict == "auto" else [args.dict]
    if not args.gauge and args.gt is None:
        raise SystemExit("--gt is required unless you are using --gauge")
    if args.run is None and args.runs_root is None:
        raise SystemExit("give --run or --runs-root")

    if args.gauge:
        roots = ([args.run] if not args.runs_root
                 else [d for d in sorted(Path(args.runs_root).iterdir())
                       if d.is_dir()])
        table = {}
        print(f"\n{'run':34s} " + "  ".join(f"{c:>8s}" for c in CAMS)
              + f"{'median':>9s}{'spread':>9s}")
        for rd in roots:
            cs = cams_of(rd)
            if not cs:
                continue
            g = gauge_from_tags(rd, cs, args)
            if not g:
                print(f"{rd.name:34s}  no tag located in any camera")
                continue
            v = np.array([g[c]["scale"] for c in g])
            sp = float((v.max() - v.min()) / v.mean()) if len(v) > 1 else 0.0
            cells = "  ".join(f"{g[c]['scale']:8.4f}" if c in g
                              else f"{'-':>8s}" for c in CAMS)
            print(f"{rd.name:34s} {cells}{np.median(v):9.4f}{sp * 100:8.1f}%")
            table[rd.name] = {"root": str(args.runs_root or rd.parent),
                              "scale": round(float(np.median(v)), 6),
                              "per_camera_scale": {c: round(g[c]["scale"], 6)
                                                   for c in g},
                              "scale_spread": round(sp, 4),
                              "solved_against": "AprilTag PnP versus depth"}
        args.out_gauge.write_text(json.dumps(table, indent=2))
        print(f"\nwritten {args.out_gauge} with {len(table)} run(s)")
        print("The spread column is the layering at its source: it is how far "
              "apart the cameras' depth scales are, measured on one physical "
              "tag, before any surface is fitted to anything.")
        return 0

    cams = cams_of(args.run)
    print(f"run {args.run.name}, cameras {cams}")

    r_m0, seen0 = (None, None)
    if args.list_recon_tags:
        r_m0, seen0 = (recon_markers_pnp(args.run, cams, args) if args.pnp
                       else recon_markers(args.run, cams, args))
        print(f"\ntag centres in the rig frame, from {args.run.name}:")
        for m in sorted(r_m0):
            p_ = np.asarray(r_m0[m]).mean(axis=0)
            print(f"  id {m:>3d}  ({p_[0]:+.4f}, {p_[1]:+.4f}, {p_[2]:+.4f})  "
                  f"seen by {seen0[m]} camera(s)")
        # Where to click. The conveyor already fixes five degrees of freedom,
        # so the only thing unknown about a tag's place in the ground truth is
        # how far along the belt it sits. Predicting its position under each
        # surviving candidate turns "find the right tag" into "click near this
        # point", which is the difference between picking the tag the detector
        # found and picking a different one on the opposite rail.
        try:
            import pin_alignment as pa
            belt = json.loads(args.belt.read_text())
            dst_f = pa.rect_frame(np.asarray(belt["corners_m"], float))
            src_f = pa.rect_frame(np.asarray(
                json.loads(args.belt_corners.read_text())["gt"], float))
            gt_all = load_cloud_rgb(str(args.gt))[0] if args.gt else None
            cands = pa.candidates(src_f, dst_f)
            if gt_all is not None:
                cands = drop_flipped(cands, gt_all, dst_f)
            print("\nwhere each tag should be in the GROUND TRUTH's own frame, "
                  "up to the along-belt shift:")
            for label, T in cands:
                Ti = np.eye(4)
                Ti[:3, :3] = np.asarray(T)[:3, :3].T
                Ti[:3, 3] = -Ti[:3, :3] @ np.asarray(T)[:3, 3]
                ex_gt = Ti[:3, :3] @ dst_f["ex"]
                print(f"  under {label}:")
                for m in sorted(r_m0):
                    q = apply(np.asarray(r_m0[m]).mean(axis=0)[None, :], Ti)[0]
                    print(f"    tag {m:>3d} near ({q[0]:+.3f}, {q[1]:+.3f}, "
                          f"{q[2]:+.3f}), free to slide along "
                          f"({ex_gt[0]:+.2f}, {ex_gt[1]:+.2f}, {ex_gt[2]:+.2f})")
            print("\nClick the tag nearest one of those points, allowing for "
                  "movement along the stated direction. The ACROSS-belt and "
                  "height coordinates are already fixed, so if your pick is "
                  "half a metre off in those, it is a different tag.")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[note ] could not predict the tag positions ({exc}); "
                  f"pick by eye and use the perp resid column to check")
        print('\nThen: {"<id>": [x, y, z]} in a json, and re-run with '
              "--gt-tags.")
        return 0

    gt, rgb = load_cloud_rgb(str(args.gt))
    if args.view == "sensor":
        frame = sensor_frame(gt)
        print("rendering along the ground truth's own optical axis")
    else:
        corners = np.asarray(
            json.loads(args.belt_corners.read_text())["gt"], float)
        frame = belt_frame(corners)
        print("rendering along the conveyor normal")
    if args.sweep:
        for mm in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            args.mm_per_px = mm
            found, _ = gt_markers(gt, rgb, frame, args)
            print(f"        {mm:>4.1f} mm/px -> {len(found)} usable")
        return 0
    if args.gt_tags:
        picked = json.loads(args.gt_tags.read_text())
        g_m = {int(k): np.asarray(v, float).reshape(-1, 3)
               for k, v in picked.items()}
        dic = "hand-picked"
        print(f"\n{len(g_m)} tag centre(s) picked by hand in the ground "
              f"truth: {sorted(g_m)}")
    else:
        g_m, dic = gt_markers(gt, rgb, frame, args)
    r_m, seen = (recon_markers_pnp(args.run, cams, args) if args.pnp
                 else recon_markers(args.run, cams, args))

    both = sorted(set(g_m) & set(r_m))
    print(f"\n{len(g_m)} in the ground truth, {len(r_m)} in the "
          f"reconstruction, {len(both)} matched: {both}")
    if len(both) < args.min_markers:
        raise SystemExit(
            f"only {len(both)} markers matched. Run --sweep: detection peaks "
            f"where the render carries about five POINTS per pixel, not where "
            f"it has the most pixels per module, and the two pull opposite "
            f"ways. On the reconstruction side a 120 mm tag is 7 px per module "
            f"at process-res 1008 and only 3.5 at 504, so solve this on the "
            f"1008 run.")

    import pin_alignment as pa
    belt = json.loads(args.belt.read_text())
    dst_f = pa.rect_frame(np.asarray(belt["corners_m"], float))
    src_f = pa.rect_frame(np.asarray(
        json.loads(args.belt_corners.read_text())["gt"], float))
    cands = pa.candidates(src_f, dst_f)

    run_cams = cams
    recon_cloud = rigkit.load_cloud(str(args.run / "fused_calib.ply"))
    s_deck = pa.scale_from_deck(recon_cloud, dst_f)
    print(f"\ngauge from the deck, in closed form: {s_deck:.4f}")

    if args.full_sim3 or len(both) >= 3:
        src = np.vstack([r_m[m] for m in both])
        dstp = np.vstack([g_m[m] for m in both])
        s, R, t, resid = umeyama(src, dstp, with_scale=True)
        print(f"markers alone, all seven degrees of freedom: scale {s:.5f}, "
              f"residual rms {np.sqrt((resid ** 2).mean()) * 1e3:.1f} mm")
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        chosen, slide = "markers only", 0.0
    else:
        print("\n  which candidates stand the parcels the right way up:")
        cands = drop_flipped(cands, gt, dst_f)
        rows, both = slide_from_markers(g_m, r_m, dst_f, cands,
                                        gt, s_deck)
        print(f"\n{'candidate':22s} {'slide mm':>10s} {'perp resid':>12s} "
              f"{'along spread':>14s}")
        for perp, slide_, label, _, spread in rows:
            print(f"{label:22s} {slide_ * 1e3:>10.1f} {perp * 1e3:>11.1f} mm "
                  f"{spread * 1e3:>11.1f} mm")
        perp, slide, chosen, T, spread = rows[0]
        print(f"\ntaking {chosen}: the conveyor gave five degrees of freedom "
              f"and the tag gave the sixth")
        if len(rows) > 1 and rows[1][0] < 1.6 * perp:
            print("[WARN] the next candidate is nearly as good. With one tag "
                  "the yaw is decided by where that tag sits; a second tag "
                  "anywhere else on the belt would settle it.")
        T = np.array(T, float)
        T[:3, 3] += dst_f["ex"] * slide
        s = s_deck
        resid = np.array([perp])

    args.out.write_text(json.dumps(
        {"T": T.tolist(), "gt": str(args.gt), "rigid_only": True,
         "solved_from": f"conveyor for five degrees of freedom, "
                        f"{len(both)} AprilTag(s) for the sixth"
                        if chosen != "markers only"
                        else f"{len(both)} markers, all seven",
         "candidate": chosen, "slide_mm": round(float(slide) * 1e3, 1),
         "dictionary": dic, "recon_scale": round(float(s), 6),
         "marker_resid_rms_mm": round(
             float(np.sqrt((np.asarray(resid) ** 2).mean())) * 1e3, 2)},
        indent=2))
    args.out_fit.write_text(json.dumps(
        {"run": args.run.name, "scale": round(float(s), 6),
         "dictionary": dic, "n_matched": len(both),
         "markers": {str(m): {"gt": np.asarray(g_m[m]).tolist(),
                              "recon": np.asarray(r_m[m]).tolist(),
                              "cameras": seen[m]} for m in both}}, indent=2))
    print(f"\nwritten {args.out} and {args.out_fit}")
    print("Compare this scale with the closed-form one pin_alignment.py gets "
          "from the deck plane. They are independent: one uses identified "
          "correspondences, the other the height of a surface. Agreement is "
          "evidence; disagreement means one of them is wrong and it is worth "
          "finding out which before grading anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())