#!/usr/bin/env python3
"""
Capture full-resolution COLOR images from a Lucid TRI050S-C (arena_api) for MDE
(Depth Anything 3 / MoGe-2). Saves lossless PNG (BGR; convert to RGB when feeding the model).

Defaults to interactive preview: SPACE saves a frame, q/ESC quits.
  --num N --interval S        timed burst (works with --headless for no-GUI VNC)
  --average K                 average K frames per save to cut sensor noise (cleaner MDE input)
  --undistort                 also write undistorted, MDE-ready frames using --intrinsics
  --rotate180                 flip 180 deg (use for the physically upside-down camera)
Preview shows live focus (Laplacian var) and % saturated pixels so you avoid blur/blown highlights.
"""
import argparse, ctypes, json, os, sys, time
import numpy as np
import cv2


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
        raise RuntimeError(f"All candidate(s) failed to open. Ensure one host NIC holds a "
                           f"172.128.0.0/24 address. Last error: {last_err}")

    nm = dev.nodemap
    for k in ('BinningHorizontal', 'BinningVertical'):
        try: nm[k].value = 1
        except Exception: pass
    for k in ('OffsetX', 'OffsetY'):
        try: nm[k].value = 0
        except Exception: pass
    nm['Width'].value = nm['Width'].max
    nm['Height'].value = nm['Height'].max

    # Native Bayer is most bandwidth-efficient; BufferFactory does sensor-correct debayering.
    chosen = None
    for fmt in ('BayerRG8', 'BayerGR8', 'BayerGB8', 'BayerBG8', 'BGR8', 'RGB8', 'Mono8'):
        try:
            nm['PixelFormat'].value = fmt; chosen = fmt; break
        except Exception:
            continue
    if chosen is None:
        raise RuntimeError("Could not set any supported PixelFormat")

    # Auto exposure/gain for convenience. For repeatable MDE inputs, lock these manually.
    for k, v in (('ExposureAuto', 'Continuous'), ('GainAuto', 'Continuous'),
                 ('AcquisitionMode', 'Continuous'), ('BalanceWhiteAuto', 'Continuous')):
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


def grab_bgr(dev):
    """Always returns an HxWx3 BGR uint8 image (SDK handles debayer / format conversion)."""
    from arena_api.buffer import BufferFactory
    from arena_api.enums import PixelFormat
    buf = dev.get_buffer(timeout=2000)
    try:
        if buf.is_incomplete:
            return None
        conv = BufferFactory.convert(buf, PixelFormat.BGR8)
    finally:
        dev.requeue_buffer(buf)
    try:
        w, h = conv.width, conv.height
        arr = (ctypes.c_ubyte * (w * h * 3)).from_address(ctypes.addressof(conv.pbytes))
        img = np.frombuffer(arr, np.uint8).reshape(h, w, 3).copy()
    finally:
        BufferFactory.destroy(conv)
    return img


def grab_averaged(dev, k):
    if k <= 1:
        return grab_bgr(dev)
    acc, got = None, 0
    for _ in range(k * 3):
        img = grab_bgr(dev)
        if img is None:
            continue
        acc = img.astype(np.float32) if acc is None else acc + img
        got += 1
        if got >= k:
            break
    return None if acc is None else np.clip(acc / got, 0, 255).round().astype(np.uint8)


def quality(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    sat = float((g >= 250).mean() * 100.0)
    return sharp, sat


def build_undistort(intrinsics_path, w, h):
    d = json.load(open(intrinsics_path))
    K = np.array(d['K'], dtype=np.float64)
    dist = np.array(d['dist'], dtype=np.float64)
    if (d['image_width'], d['image_height']) != (w, h):
        print(f"WARN: intrinsics are for {d['image_width']}x{d['image_height']}, "
              f"capturing {w}x{h}")
    # Rectify to the ORIGINAL K so fov_x (55.5 deg) and back-projection stay unchanged.
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
    valid = cv2.remap(np.full((h, w), 255, np.uint8), mapx, mapy, cv2.INTER_NEAREST)
    return mapx, mapy, valid, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--serial', default='260505158')
    ap.add_argument('--ip', default='172.128.0.3')
    ap.add_argument('--out', default='./mde_captures')
    ap.add_argument('--num', type=int, default=0, help='0 = interactive; >0 = save N then exit')
    ap.add_argument('--interval', type=float, default=1.0, help='seconds between saves (--num mode)')
    ap.add_argument('--average', type=int, default=1, help='frames to average per save (denoise)')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--preview-width', type=int, default=1100)
    ap.add_argument('--undistort', action='store_true', help='also write undistorted MDE-ready frames')
    ap.add_argument('--intrinsics', default='intrinsics.json')
    ap.add_argument('--rotate180', action='store_true', help='for the upside-down camera')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    undist_dir = os.path.join(args.out, 'undistorted')
    system, dev, fmt = open_camera(serial=args.serial, ip=args.ip)
    W, H = dev.nodemap['Width'].value, dev.nodemap['Height'].value
    print(f"Camera open, PixelFormat={fmt} -> BGR8, {W}x{H}, average={args.average}")

    undist = None
    if args.undistort:
        os.makedirs(undist_dir, exist_ok=True)
        mapx, mapy, valid, _ = build_undistort(args.intrinsics, W, H)
        cv2.imwrite(os.path.join(undist_dir, 'valid_mask.png'), valid)
        undist = (mapx, mapy)
        print(f"Undistort ON (rectified to original K); wrote {undist_dir}/valid_mask.png")

    saved, last_save = 0, 0.0

    def save(bgr):
        nonlocal saved
        if args.rotate180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        fn = os.path.join(args.out, f"frame_{saved:04d}.png")
        cv2.imwrite(fn, bgr)
        msg = fn
        if undist is not None:
            ud = cv2.remap(bgr, undist[0], undist[1], cv2.INTER_LINEAR)
            ufn = os.path.join(undist_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(ufn, ud)
            msg += f"  (+{ufn})"
        sharp, sat = quality(bgr)
        print(f"saved {msg}  sharpness={sharp:.0f} saturated={sat:.1f}%")
        saved += 1

    dev.start_stream(10)
    try:
        while True:
            bgr = grab_bgr(dev)
            if bgr is None:
                continue

            if args.num > 0:  # timed burst
                if time.time() - last_save >= args.interval:
                    img = grab_averaged(dev, args.average)
                    if img is not None:
                        save(img); last_save = time.time()
                if saved >= args.num:
                    print("Target reached."); break
                if not args.headless:
                    pass  # fall through to preview below
                else:
                    continue

            if args.headless:
                continue

            # ---- preview ----
            sharp, sat = quality(bgr)
            s = args.preview_width / W
            disp = cv2.resize(bgr, None, fx=s, fy=s)
            col = (0, 0, 255) if sat > 2.0 else (0, 255, 0)
            cv2.putText(disp, f"saved {saved}{'/' + str(args.num) if args.num else ''}   "
                              f"focus {sharp:.0f}   saturated {sat:.1f}%",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.putText(disp, f"avg={args.average}  undistort={'on' if undist else 'off'}  "
                              f"SPACE=save  q=quit",
                        (10, disp.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow('mde capture', disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == 32 and args.num == 0:  # manual save (interactive only)
                img = grab_averaged(dev, args.average)
                if img is not None:
                    save(img)
    finally:
        dev.stop_stream()
        system.destroy_device()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved} image(s) in {args.out}")


if __name__ == '__main__':
    main()