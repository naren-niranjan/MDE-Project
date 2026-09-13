#!/usr/bin/env python3
"""
Build a metric point cloud by STEREO TRIANGULATION from one calibrated pair
(center + left, or center + right), output in the center frame.

Unlike fusing independent MoGe depth maps, triangulated depth is metrically
consistent across cameras -- run this for both pairs and the two clouds overlap
by construction (same calibration, same reference frame).

This is a SGBM scaffold. SGBM is cheap and runs on Jetson, but is weak on
textureless / specular regions (your conveyor). Box tops with edges/labels
should match fine -- which is what you need for picking. For robustness on
low-texture surfaces, swap the matcher for a learned model (RAFT-Stereo /
FoundationStereo / IGEV-Stereo); the rectify + reproject scaffolding here stays
the same.

Inputs (your existing files):
    calibration/intrinsics_center.json, intrinsics_<side>.json
    calibration/extrinsics.json   (R,t for <side>, mapping center -> side)
    snapshots/center.png, snapshots/<side>.png

Usage:
    python stereo_pair_cloud.py --root /home/jetson/Projects/MultiCamera/Moge2 \
        --side left --scale 0.5 --min-disp 256 --num-disp 256
    # inspect, tune min-disp/num-disp to your working distance, then also run --side right
"""

import argparse
import json
import os
import numpy as np
import cv2


def load_intr(path):
    d = json.load(open(path))
    K = np.ascontiguousarray(np.asarray(d["camera_matrix"], np.float64))
    D = np.ascontiguousarray(np.asarray(d["dist_coeffs"], np.float64).reshape(-1))
    W, H = d["image_size"]
    return K, D, (int(W), int(H))


def load_ext(path, side, invert):
    d = json.load(open(path))[side]
    R = np.asarray(d["R"], float)
    t = np.asarray(d["t"], float).reshape(3)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t   # center -> side
    if invert:
        T = np.linalg.inv(T)
    return np.ascontiguousarray(T[:3, :3]), np.ascontiguousarray(T[:3, 3]).reshape(3, 1)


def write_ply(path, xyz, rgb):
    n = xyz.shape[0]
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.empty(n, dt)
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="downscale factor (wide baseline => big disparities; "
                         "0.5 halves the search)")
    ap.add_argument("--min-disp", type=int, default=256,
                    help="minimum disparity (offset for your working distance)")
    ap.add_argument("--num-disp", type=int, default=256,
                    help="disparity range (multiple of 16)")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--wls", action="store_true",
                    help="WLS post-filter to densify (needs opencv-contrib-python)")
    ap.add_argument("--invert-extrinsics", action="store_true",
                    help="use if extrinsics are stored side->center, not center->side")
    ap.add_argument("--swap", action="store_true",
                    help="make the side camera the rectification reference "
                         "(use when depths come out negative)")
    ap.add_argument("--z-min", type=float, default=0.5)
    ap.add_argument("--z-max", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cal = os.path.join(args.root, "calibration")
    snp = os.path.join(args.root, "snapshots")
    Kc, Dc, size0 = load_intr(os.path.join(cal, "intrinsics_center.json"))
    Ks, Ds, _ = load_intr(os.path.join(cal, f"intrinsics_{args.side}.json"))
    Rcs, Tcs = load_ext(os.path.join(cal, "extrinsics.json"), args.side,
                        args.invert_extrinsics)          # center -> side

    img_c = cv2.imread(os.path.join(snp, "center.png"))
    img_s = cv2.imread(os.path.join(snp, f"{args.side}.png"))
    if img_c is None or img_s is None:
        raise SystemExit("could not read snapshots/center.png or side image")

    # SGBM assumes cam2 is to the RIGHT of cam1; otherwise depths go negative.
    # --swap makes the side camera the reference, then we map the cloud back
    # into the center frame.
    to_center = None
    if not args.swap:
        K1, D1, K2, D2 = Kc, Dc, Ks, Ds
        Rr, Tr = Rcs, Tcs
        img1, img2 = img_c, img_s            # cam1 = center -> already center frame
    else:
        Rsc = np.ascontiguousarray(Rcs.T)                # side -> center
        Tsc = np.ascontiguousarray(-Rcs.T @ Tcs)
        K1, D1, K2, D2 = Ks, Ds, Kc, Dc
        Rr, Tr = Rsc, Tsc
        img1, img2 = img_s, img_c            # cam1 = side
        to_center = (Rsc, Tsc)

    # scale intrinsics + images up front so Q is correct for the scaled disparity
    s = args.scale
    if s != 1.0:
        K1 = K1.copy(); K1[:2, :] *= s
        K2 = K2.copy(); K2[:2, :] *= s
        size = (int(round(size0[0] * s)), int(round(size0[1] * s)))
    else:
        size = size0
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, Rr, Tr, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    r1 = cv2.remap(img1, m1x, m1y, cv2.INTER_LINEAR)
    r2 = cv2.remap(img2, m2x, m2y, cv2.INTER_LINEAR)
    g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)

    nd = args.num_disp - (args.num_disp % 16)
    left = cv2.StereoSGBM_create(
        minDisparity=args.min_disp, numDisparities=nd, blockSize=args.block,
        P1=8 * 3 * args.block ** 2, P2=32 * 3 * args.block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    if args.wls:
        try:
            right = cv2.ximgproc.createRightMatcher(left)
            wls = cv2.ximgproc.createDisparityWLSFilter(left)
            wls.setLambda(8000.0); wls.setSigmaColor(1.5)
            dl = left.compute(g1, g2)
            dr = right.compute(g2, g1)
            disp = wls.filter(dl, g1, disparity_map_right=dr).astype(np.float32) / 16.0
        except AttributeError:
            print("  (ximgproc unavailable -- pip install opencv-contrib-python; "
                  "using plain SGBM)")
            disp = left.compute(g1, g2).astype(np.float32) / 16.0
    else:
        disp = left.compute(g1, g2).astype(np.float32) / 16.0

    good = disp > args.min_disp
    if good.sum() == 0:
        raise SystemExit("no disparities above min_disp -- lower --min-disp")
    print(f"disparity: valid {good.mean()*100:.1f}%  "
          f"range [{disp[good].min():.0f}, {disp[good].max():.0f}]")

    pts = cv2.reprojectImageTo3D(disp, Q)                # cam1 rectified frame, m
    z = pts[:, :, 2]
    fz = z[good & np.isfinite(z)]
    if len(fz):
        print(f"z (cam1 frame): median {np.median(fz):+.3f}  "
              f"[{np.percentile(fz,5):+.3f}, {np.percentile(fz,95):+.3f}] m  |  "
              f"{100*(fz<0).mean():.0f}% negative")
        if (fz < 0).mean() > 0.5:
            hint = "remove --swap" if args.swap else "add --swap"
            print(f"  >>> depths mostly NEGATIVE: wrong left/right order -> {hint}")

    valid = good & np.isfinite(pts).all(2) \
        & (z > args.z_min) & (z < args.z_max)
    P = pts[valid]
    C = cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)[valid]

    P = (R1.T @ P.T).T                                   # rectified -> cam1 frame
    if to_center is not None:                            # side frame -> center frame
        Rsc, Tsc = to_center
        P = (Rsc @ P.T).T + Tsc.reshape(3)

    out = args.out or os.path.join(args.root, f"stereo_{args.side}.ply")
    write_ply(out, P.astype(np.float32), C.astype(np.uint8))
    med = np.median(P[:, 2]) if len(P) else float("nan")
    print(f"wrote {out}  ({len(P):,} pts)  depth median {med:.3f} m")


if __name__ == "__main__":
    main()