#!/usr/bin/env python3
"""
Auto-capture ChArUco frames from a Lucid GigE camera (arena_api) for intrinsic calibration.

Board : calib.io ChArUco, 9x12 squares, square=60 mm, marker=47 mm, DICT_5X5_250
Output: full-resolution Mono8 PNGs into --out, consumed by calibrate_intrinsics.py

Auto mode (default): a frame is saved automatically when ALL hold:
  * >= --min-corners charuco corners detected
  * board held STILL for --hold-frames frames (median corner motion < --motion-thresh px)
  * pose is NOVEL vs already-saved frames (different position / distance / tilt)
Stops at --num frames. SPACE forces a save; q/ESC quits. --manual disables auto.
Use --headless when the GUI is too laggy over VNC (same logic, terminal only).
"""
import argparse, ctypes, os, sys, time
import numpy as np
import cv2

# ---------------- board definition ----------------
SQX, SQY = 12, 9                 # squares (cols, rows) — verified against the physical board
SQUARE_M, MARKER_M = 0.060, 0.047
ARUCO_DICT = cv2.aruco.DICT_5X5_250
MAX_CORNERS = (SQX - 1) * (SQY - 1)   # 88 inner corners when fully visible


def build_detectors():
    """Detectors for both ChArUco patterns; calib.io usually needs legacy=True on >=4.6,
    but we keep whichever finds more corners each frame."""
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    dets = {}
    for legacy in (False, True):
        b = cv2.aruco.CharucoBoard((SQX, SQY), SQUARE_M, MARKER_M, adict)
        b.setLegacyPattern(legacy)
        dets[legacy] = cv2.aruco.CharucoDetector(b)
    return dets


def detect_best(dets, gray):
    best = (None, None, None, None, 0, False)  # cc, ci, mc, mi, n, legacy
    for legacy, det in dets.items():
        cc, ci, mc, mi = det.detectBoard(gray)
        n = 0 if cc is None else len(cc)
        if n > best[4]:
            best = (cc, ci, mc, mi, n, legacy)
    return best


# ---------------- pose / quality descriptors ----------------
def descriptor(cc, ci, W, H):
    """Image-space pose summary used for novelty + stability (no intrinsics needed)."""
    pts = cc.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    area = cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))
    nscale = (area ** 0.5) / W
    # tilt proxy via PCA elongation: 0 = frontal, ->1 = strongly foreshortened
    d = pts - pts.mean(axis=0)
    eig = np.linalg.eigvalsh(np.cov(d.T)) if len(pts) > 2 else np.array([1.0, 1.0])
    eig = np.clip(eig, 1e-9, None)
    tilt = 1.0 - (eig.min() ** 0.5) / (eig.max() ** 0.5)
    id2pt = {int(i): p for i, p in zip(ci.reshape(-1), pts)}
    return {"ncx": cx / W, "ncy": cy / H, "nscale": nscale, "tilt": tilt, "id2pt": id2pt}


def is_novel(d, saved, c_thr, s_thr, t_thr):
    """Novel unless it is close to some saved view in position AND scale AND tilt."""
    for s in saved:
        dc = ((d["ncx"] - s["ncx"]) ** 2 + (d["ncy"] - s["ncy"]) ** 2) ** 0.5
        if dc < c_thr and abs(d["nscale"] - s["nscale"]) < s_thr and abs(d["tilt"] - s["tilt"]) < t_thr:
            return False
    return True


def median_motion(prev, cur):
    if prev is None:
        return None
    common = set(prev["id2pt"]) & set(cur["id2pt"])
    if len(common) < 6:
        return None
    diffs = [np.linalg.norm(cur["id2pt"][i] - prev["id2pt"][i]) for i in common]
    return float(np.median(diffs))


# ---------------- camera ----------------
def open_camera(serial=None, ip=None):
    from arena_api.system import system
    infos = system.device_infos
    cands = ([i for i in infos if str(i.get('serial')) == str(serial)] if serial
             else [i for i in infos if i.get('ip') == ip])
    if not cands:
        print("Discovered:", [(i.get('model'), i.get('serial'), i.get('ip')) for i in infos],
              file=sys.stderr)
        raise RuntimeError(f"No camera matching serial={serial} ip={ip}")

    dev, last_err = None, None
    for k, info in enumerate(cands):
        try:
            dev = system.create_device(device_infos=[info])[0]
            print(f"opened candidate {k + 1}/{len(cands)} "
                  f"(ip={info.get('ip')}, serial={info.get('serial')})")
            break
        except Exception as e:
            print(f"  candidate {k + 1}/{len(cands)} (ip={info.get('ip')}) failed: {e}")
            last_err = e
    if dev is None:
        raise RuntimeError(f"All {len(cands)} candidate(s) failed to open. Ensure exactly one "
                           f"host NIC holds a 172.128.0.0/24 address. Last error: {last_err}")

    nm = dev.nodemap
    for k in ('BinningHorizontal', 'BinningVertical'):
        try: nm[k].value = 1
        except Exception: pass
    for k in ('OffsetX', 'OffsetY'):
        try: nm[k].value = 0
        except Exception: pass
    nm['Width'].value = nm['Width'].max
    nm['Height'].value = nm['Height'].max

    chosen = None
    for fmt in ('Mono8', 'BayerRG8', 'BayerGR8', 'BayerGB8', 'BayerBG8', 'RGB8', 'BGR8'):
        try:
            nm['PixelFormat'].value = fmt; chosen = fmt; break
        except Exception:
            continue
    if chosen is None:
        raise RuntimeError("Could not set any supported PixelFormat")

    for k, v in (('ExposureAuto', 'Continuous'), ('GainAuto', 'Continuous'),
                 ('AcquisitionMode', 'Continuous')):
        try: nm[k].value = v
        except Exception: pass
    try: nm['DeviceStreamChannelPacketSize'].value = nm['DeviceStreamChannelPacketSize'].max
    except Exception: pass
    tl = dev.tl_stream_nodemap
    for k, v in (('StreamBufferHandlingMode', 'NewestOnly'),
                 ('StreamAutoNegotiatePacketSize', True),
                 ('StreamPacketResendEnable', True)):
        try: tl[k].value = v
        except Exception: pass
    return system, dev, chosen


def grab_gray(dev, fmt):
    buf = dev.get_buffer(timeout=2000)
    try:
        if buf.is_incomplete:
            return None
        w, h = buf.width, buf.height
        bpp = max(1, int(len(buf.data) / (w * h)))
        arr = (ctypes.c_ubyte * (w * h * bpp)).from_address(ctypes.addressof(buf.pbytes))
        img = np.frombuffer(arr, dtype=np.uint8).reshape(h, w, bpp).copy()
    finally:
        dev.requeue_buffer(buf)
    if bpp == 1:
        if fmt and fmt.startswith('Bayer'):
            code = {'BayerRG8': cv2.COLOR_BayerRG2GRAY, 'BayerGR8': cv2.COLOR_BayerGR2GRAY,
                    'BayerGB8': cv2.COLOR_BayerGB2GRAY, 'BayerBG8': cv2.COLOR_BayerBG2GRAY}.get(fmt)
            return cv2.cvtColor(img, code) if code else img[:, :, 0]
        return img[:, :, 0]
    if bpp == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if fmt == 'RGB8' else cv2.COLOR_BGR2GRAY)
    return img.reshape(h, w)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--serial', default='260505158', help='unique camera serial (preferred over IP)')
    ap.add_argument('--ip', default='172.128.0.3')
    ap.add_argument('--out', default='./captures')
    ap.add_argument('--num', type=int, default=25, help='target number of frames')
    ap.add_argument('--min-corners', type=int, default=24)
    ap.add_argument('--manual', action='store_true', help='disable auto-capture (SPACE only)')
    ap.add_argument('--headless', action='store_true', help='no GUI window (laggy VNC)')
    ap.add_argument('--preview-width', type=int, default=1100)
    # auto-capture tuning
    ap.add_argument('--hold-frames', type=int, default=5, help='still frames required before save')
    ap.add_argument('--motion-thresh', type=float, default=3.0, help='median corner motion (px) = "still"')
    ap.add_argument('--novel-centroid', type=float, default=0.10, help='min centroid move (frac of frame)')
    ap.add_argument('--novel-scale', type=float, default=0.10, help='min apparent-size change (frac)')
    ap.add_argument('--novel-tilt', type=float, default=0.12, help='min tilt change')
    ap.add_argument('--cooldown', type=float, default=0.8, help='seconds between auto-saves')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dets = build_detectors()
    system, dev, fmt = open_camera(serial=args.serial, ip=args.ip)
    print(f"Camera open, PixelFormat={fmt}, "
          f"{dev.nodemap['Width'].value}x{dev.nodemap['Height'].value}")
    print(f"AUTO-CAPTURE {'OFF (manual)' if args.manual else 'ON'} | target {args.num} frames")

    saved_descs, covered, prev_desc = [], set(), None
    saved, stable, last_save = 0, 0, 0.0

    dev.start_stream(10)
    try:
        while True:
            gray = grab_gray(dev, fmt)
            if gray is None:
                continue
            H, W = gray.shape[:2]
            cc, ci, mc, mi, n, legacy = detect_best(dets, gray)

            status, color = "no board", (0, 0, 255)
            desc = None
            do_save = False
            if n >= args.min_corners:
                desc = descriptor(cc, ci, W, H)
                motion = median_motion(prev_desc, desc)
                still = motion is not None and motion < args.motion_thresh
                stable = stable + 1 if still else 0
                novel = is_novel(desc, saved_descs, args.novel_centroid,
                                 args.novel_scale, args.novel_tilt)
                cooled = (time.time() - last_save) >= args.cooldown
                if not args.manual:
                    if not novel:
                        status, color = "move board (already have this view)", (0, 165, 255)
                    elif stable < args.hold_frames:
                        status, color = f"hold still {stable}/{args.hold_frames}", (0, 215, 255)
                    elif not cooled:
                        status, color = "...", (0, 215, 255)
                    else:
                        do_save = True
                else:
                    status, color = f"{n} corners (SPACE to save)", (0, 255, 0)
            prev_desc = desc

            if do_save:
                fn = os.path.join(args.out, f"cal_{saved:03d}.png")
                cv2.imwrite(fn, gray)
                saved += 1; last_save = time.time(); stable = 0
                saved_descs.append(desc)
                covered.add((min(2, int(desc["ncx"] * 3)), min(2, int(desc["ncy"] * 3))))
                status, color = f"SAVED {saved}/{args.num}", (0, 255, 0)
                print(f"saved {fn}  corners={n}/{MAX_CORNERS} "
                      f"pos=({desc['ncx']:.2f},{desc['ncy']:.2f}) scale={desc['nscale']:.2f} "
                      f"tilt={desc['tilt']:.2f} legacy={legacy}  coverage={len(covered)}/9")

            if saved >= args.num:
                print("Target reached."); break

            # ----- headless: terminal only -----
            if args.headless:
                if not do_save:
                    print(f"\rcorners={n:>3}/{MAX_CORNERS} saved={saved}/{args.num} "
                          f"coverage={len(covered)}/9  {status:<45}", end='', flush=True)
                continue

            # ----- GUI preview -----
            s = args.preview_width / W
            disp = cv2.cvtColor(cv2.resize(gray, None, fx=s, fy=s), cv2.COLOR_GRAY2BGR)
            if mc is not None and len(mc):
                cv2.aruco.drawDetectedMarkers(disp, [m * s for m in mc], mi)
            if cc is not None and n:
                cv2.aruco.drawDetectedCornersCharuco(disp, (cc * s).astype(np.float32), ci)
            # 3x3 coverage grid
            gh, gw = disp.shape[0] // 3, disp.shape[1] // 3
            ov = disp.copy()
            for (gx, gy) in covered:
                cv2.rectangle(ov, (gx * gw, gy * gh), ((gx + 1) * gw, (gy + 1) * gh), (0, 120, 0), -1)
            disp = cv2.addWeighted(ov, 0.18, disp, 0.82, 0)
            for i in (1, 2):
                cv2.line(disp, (i * gw, 0), (i * gw, disp.shape[0]), (60, 60, 60), 1)
                cv2.line(disp, (0, i * gh), (disp.shape[1], i * gh), (60, 60, 60), 1)
            cv2.putText(disp, f"corners {n}/{MAX_CORNERS}  saved {saved}/{args.num}  "
                              f"coverage {len(covered)}/9  legacy={legacy}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(disp, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(disp, "SPACE=force save   q=quit", (10, disp.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow('charuco auto-capture', disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == 32 and desc is not None and n >= args.min_corners:  # manual override
                fn = os.path.join(args.out, f"cal_{saved:03d}.png")
                cv2.imwrite(fn, gray)
                saved += 1; last_save = time.time(); stable = 0
                saved_descs.append(desc)
                covered.add((min(2, int(desc["ncx"] * 3)), min(2, int(desc["ncy"] * 3))))
                print(f"saved (manual) {fn}  corners={n}/{MAX_CORNERS}")
    finally:
        dev.stop_stream()
        system.destroy_device()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved} frames in {args.out}, frustum coverage {len(covered)}/9 cells.")


if __name__ == '__main__':
    main()