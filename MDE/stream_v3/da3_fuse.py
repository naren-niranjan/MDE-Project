#!/usr/bin/env python3
"""
da3_fuse.py

Multi-camera depth fusion using Depth Anything 3 conditioned on calibrated
intrinsics and extrinsics. Takes four snapshot images, runs DA3 once, and
writes per-camera and fused point clouds.

Settled findings baked in
-------------------------
1. NO pad_square WITH UNMODIFIED cy. Padding 2448x2048 to square while passing
   the original cy told the model the vertical FOV was wrong, and the model
   rescaled depth to reconcile the contradiction. Note this is NOT an argument
   against padding as such: da3_stream.py and res_sweep.py pad on the RIGHT AND
   BOTTOM ONLY, which leaves cx and cy untouched and is therefore safe. Mixed
   aspect ratios in one batch still centre-crop to the smallest common
   dimension, so right/bottom padding is the fix, not a risk.

2. NO rot_k. The calibrated extrinsics already encode each camera's physical
   roll. Rotating the image and passing the unmodified extrinsic makes the pose
   disagree with the pixels.

3. NO manual K rescaling. pred.intrinsics and pred.extrinsics come back already
   matched to the depth grid. Use them directly.

4. NO Umeyama scale recovery. pred.is_metric is 1 and pred.depth is in metres.

5. Lens distortion is removed before inference. DA3 assumes a pinhole model.
   With the 12 mm Edmund Optics C Series the measured radial distortion is
   about 2.4 per cent at maximum field.

Artefact filtering
------------------
Two rejections run on the depth map BEFORE back-projection:

  edge     depth discontinuities produce flying pixels; rejected by relative
           depth gradient, then dilated one pixel
  grazing  surfaces seen near-tangentially stretch into visible combs;
           rejected by the angle between the local surface normal and the view
           ray, default 70 deg

THE METRIC ANCHOR -- READ THIS BEFORE USING --apply-absolute
------------------------------------------------------------
Every metric claim the pipeline makes rests on a set of REFERENCE PLANES:
surfaces whose perpendicular distance from the REFERENCE camera's optical
centre has been measured directly, with the ChArUco board, by measure_deck.py.

Through the 8 mm lens there were two of them, the conveyor deck and the floor,
and the anchor was expressed as a deck standoff plus a floor-to-deck
separation. Through the 12 mm lens the field at the deck shrank from
3.15 x 2.64 m to 2.10 x 1.76 m, the floor survives only as a noisy fringe past
the belt edge, and that pair no longer exists as a measurement.

It was also the wrong pair. The floor is 825 mm BELOW the deck; the surfaces
the robot picks from are 30 to 400 mm ABOVE it. A per-view depth correction
solved between the deck and the floor is extrapolated 1.1 m to reach a parcel
top, and extrapolating a bias that is not exactly affine is precisely how four
cameras came to agree on the deck to 1.6 mm and disagree on a parcel top by
44 mm.

ground_truth.json therefore now carries a LIST of reference planes, each with
its measured perpendicular distance and its height above the deck. The set that
matters is one on the deck and at least one on a riser inside the parcel band,
so that the correction interpolates rather than extrapolates. layer_align.py
regresses the metric scale and shift over all of them.

GT_DECK_PERP_M and GT_SEPARATION_M are retained for the older two-surface
files, and are derived from the plane list when it is present. If the anchor is
absent, untagged, marked unusable, or spans too little, every one of these
constants is NaN. That is deliberate: a NaN scale propagates into visibly NaN
depth rather than into a believable wrong dimension.

    python measure_deck.py --calib-dir <results> --camera center \\
        --board-thickness-mm 47 \\
        --plane deck  snaps/deck_a/center.png snaps/deck_b/center.png \\
        --plane riser_270 snaps/riser_a/center.png snaps/riser_b/center.png \\
        --write ground_truth.json

Do not confuse a reference plane's perp_m with --seg-plane-distance or
--deck-distance in box_segment.py and res_sweep.py. Those select a band in the
MODEL's own coordinates, which are unanchored and drift a few per cent with
framing, lens and resolution; perp_m is a physical measurement. A gap of
several centimetres between the two is expected and is itself the scale error.

Example
-------
python da3_fuse.py \
    --snapshot-dir snapshots/20260814_121000_650 \
    --calib-dir    /home/jetson/Projects/Calibration_4_5/results \
    --out-dir      runs/$(date +%Y%m%d_%H%M%S)/prior \
    --mode prior
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import torch
except ImportError:
    torch = None


PALETTE = [(0.90, 0.30, 0.25), (0.25, 0.65, 0.90),
           (0.35, 0.75, 0.40), (0.95, 0.75, 0.20),
           (0.70, 0.45, 0.85), (0.95, 0.55, 0.75)]


# --------------------------------------------------------------------------
# metric anchor
# --------------------------------------------------------------------------

GROUND_TRUTH_FILE = Path(os.environ.get(
    "MDE_GROUND_TRUTH",
    str(Path(__file__).resolve().with_name("ground_truth.json"))))

# Minimum separation between the nearest and furthest reference plane before
# the scale term can be solved against depth noise, and the band above the deck
# at least one of them has to fall in.
MIN_ANCHOR_SPAN_M = 0.150
PARCEL_BAND_M = (0.030, 0.400)


def _load_ground_truth(path=GROUND_TRUTH_FILE):
    """Read the reference planes, or return NaNs with the reason why.

    Never falls back to a previously hardcoded value. An anchor whose
    provenance is unknown is worse than no anchor at all: the pipeline would
    keep producing dimensions and nothing downstream would show that they were
    referred to a lens that is no longer fitted, or solved across a span too
    short to constrain the scale.
    """
    out = {"deck_perp_m": float("nan"), "separation_m": float("nan"),
           "planes": [], "span_m": float("nan"),
           "lens_id": None, "reference_camera": None, "ok": False,
           "source": str(path), "reason": None, "warnings": []}
    if not path.exists():
        out["reason"] = f"{path} does not exist"
        return out
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        out["reason"] = f"{path} could not be read ({exc})"
        return out

    out["lens_id"] = data.get("lens_id")
    out["reference_camera"] = data.get("reference_camera")
    out["measured"] = data.get("measured")
    out["method"] = data.get("method")

    if out["lens_id"] is None:
        out["reason"] = f"{path} records no lens_id, so it cannot be trusted"
        return out
    if data.get("usable") is False:
        out["reason"] = (f"{path} is marked unusable by measure_deck.py: "
                         + "; ".join(data.get("problems", []) or ["no reason"]))
        return out

    planes = []
    for e in data.get("reference_planes", []) or []:
        perp = e.get("perp_m")
        h = e.get("height_above_deck_m")
        if not isinstance(perp, (int, float)):
            continue
        planes.append({"label": e.get("label", "?"),
                       "perp_m": float(perp),
                       "height_above_deck_m": (float(h) if isinstance(
                           h, (int, float)) else float("nan")),
                       "spread_mm": e.get("spread_mm")})
    planes.sort(key=lambda e: e["perp_m"])

    deck = data.get("deck_perp_m")
    sep = data.get("separation_m")

    # ---- the modern form: an explicit list of measured planes -----------
    if len(planes) >= 2:
        out["planes"] = planes
        out["span_m"] = planes[-1]["perp_m"] - planes[0]["perp_m"]
        named_deck = next((p for p in planes if p["label"] == "deck"), None)
        out["deck_perp_m"] = float(named_deck["perp_m"]) if named_deck \
            else float(deck) if isinstance(deck, (int, float)) else planes[-1]["perp_m"]
        floor = next((p for p in planes if p["label"] == "floor"), None)
        out["separation_m"] = (abs(float(floor["height_above_deck_m"]))
                               if floor is not None
                               and np.isfinite(floor["height_above_deck_m"])
                               else float(sep) if isinstance(sep, (int, float))
                               else float("nan"))

        if out["span_m"] < MIN_ANCHOR_SPAN_M:
            out["reason"] = (f"{path} spans only {out['span_m'] * 1e3:.0f} mm "
                             f"between its nearest and furthest reference "
                             f"plane, below the {MIN_ANCHOR_SPAN_M * 1e3:.0f} "
                             f"mm needed to separate scale from offset")
            return out

        lo, hi = PARCEL_BAND_M
        in_band = [p for p in planes if p["label"] != "deck"
                   and np.isfinite(p["height_above_deck_m"])
                   and lo - 0.02 <= p["height_above_deck_m"] <= hi + 0.02]
        if not in_band:
            out["warnings"].append(
                f"no reference plane lies between {lo * 1e3:.0f} and "
                f"{hi * 1e3:.0f} mm above the deck, so the correction solved "
                f"from this anchor is EXTRAPOLATED into the volume the robot "
                f"picks from. Measure a riser inside the band")
        out["ok"] = True
        return out

    # ---- the legacy two-surface form ------------------------------------
    missing = [k for k, v in (("deck_perp_m", deck), ("separation_m", sep))
               if not isinstance(v, (int, float))]
    if missing:
        out["reason"] = (f"{path} carries fewer than two reference planes and "
                         f"has no value for {', '.join(missing)}")
        return out

    out["deck_perp_m"] = float(deck)
    out["separation_m"] = float(sep)
    out["planes"] = [
        {"label": "deck", "perp_m": float(deck),
         "height_above_deck_m": 0.0, "spread_mm": None},
        {"label": "floor", "perp_m": float(deck) + float(sep),
         "height_above_deck_m": -float(sep), "spread_mm": None},
    ]
    out["span_m"] = float(sep)
    out["warnings"].append(
        "this is a legacy deck-plus-floor anchor. Both planes lie at or below "
        "the deck, so any correction solved from it is extrapolated upwards to "
        "reach the parcel tops. Re-measure with a riser inside the parcel band")
    out["ok"] = True
    return out


GT = _load_ground_truth()
GT_DECK_PERP_M = GT["deck_perp_m"]
GT_SEPARATION_M = GT["separation_m"]
GT_LENS_ID = GT["lens_id"]
GT_PLANES = GT["planes"]

if not GT["ok"]:
    print("=" * 74)
    print("  NO METRIC ANCHOR: " + str(GT["reason"]))
    print("  GT_DECK_PERP_M, GT_SEPARATION_M and GT_PLANES are unusable, so")
    print("  layer_align.py's absolute block and da3_stream.py")
    print("  --apply-absolute produce NaN rather than a plausible wrong")
    print("  number. Relative alignment and segmentation still work;")
    print("  DIMENSIONS ARE NOT ANCHORED.")
    print("  Fix: python measure_deck.py --write " + str(GROUND_TRUTH_FILE))
    print("=" * 74)
else:
    labels = ", ".join(f"{p['label']} {p['perp_m']:.4f} m" for p in GT_PLANES)
    print(f"[anchor] {len(GT_PLANES)} reference plane(s) over "
          f"{GT['span_m'] * 1e3:.0f} mm: {labels}")
    for w in GT["warnings"]:
        print(f"[anchor] ** {w} **")

_LENS_WARNED = set()


def _check_lens(name, lens_id):
    """Warn once per camera if its intrinsics were solved through different
    glass from the one the anchor was measured under."""
    if lens_id is None:
        key = ("untagged", name)
        if key not in _LENS_WARNED:
            _LENS_WARNED.add(key)
            print(f"[WARN] intrinsics_{name}.json records no lens_id. If it "
                  f"predates the 12 mm change it describes different optics "
                  f"and every depth below is referred to the wrong focal "
                  f"length.")
        return
    if GT_LENS_ID is not None and lens_id != GT_LENS_ID:
        key = ("mismatch", name, lens_id)
        if key not in _LENS_WARNED:
            _LENS_WARNED.add(key)
            print(f"[WARN] intrinsics_{name}.json was solved for lens "
                  f"{lens_id}, but the metric anchor was measured under "
                  f"{GT_LENS_ID}. Re-measure the reference planes, or the "
                  f"absolute scale is referred to optics that are not fitted.")


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def load_intrinsics(calib_dir, name):
    data = json.loads((Path(calib_dir) / f"intrinsics_{name}.json").read_text())
    _check_lens(name, data.get("lens_id"))
    return {
        "K": np.asarray(data["camera_matrix"], np.float64).reshape(3, 3),
        "dist": np.asarray(data.get("dist_coeffs", [0] * 5), np.float64).ravel(),
        "size": tuple(int(v) for v in data["image_size"]),
        "rms_px": data.get("rms_reprojection_error_px", float("nan")),
        "lens_id": data.get("lens_id"),
        "lens_mm": data.get("lens_mm"),
        "serial": data.get("serial"),
    }


def check_focal_spread(intr, names, warn_frac=0.004):
    """Report the focal spread across the rig and what it costs in depth.

    A camera whose solved fx sits a fraction f from the rig mean receives depth
    scaled by roughly f under the intrinsic prior. On this rig the spread is
    0.77 per cent, which is 23 mm at the 2.99 m deck standoff. The per-view
    affine's SCALE term can absorb it, but only when two reference planes exist
    to solve that term. With one plane the solve drops to offset-only, the
    scale spread survives, and it reappears as inter-camera disagreement that
    grows with height above the surface the offset was solved on.
    """
    fx = {n: float(intr[n]["K"][0, 0]) for n in names}
    mean = float(np.mean(list(fx.values())))
    spread = (max(fx.values()) - min(fx.values())) / mean
    rec = {"fx_px": {n: round(v, 2) for n, v in fx.items()},
           "mean_px": round(mean, 2),
           "spread_fraction": round(spread, 5),
           "depth_error_at_3m_mm": round(spread * 3.0 * 1e3, 1)}
    if spread > warn_frac:
        print(f"[WARN] the solved focal lengths differ by {spread * 100:.2f}% "
              f"across the rig ({rec['depth_error_at_3m_mm']:.0f} mm of depth "
              f"at 3 m).")
        for n, v in fx.items():
            print(f"       {n:<8} {v:8.1f} px  {(v / mean - 1) * 100:+6.3f}%")
        print("       Only the per-view SCALE term removes this, and that term "
              "needs two\n       reference planes. Do not run with an "
              "offset-only correction.")
    return rec


def load_extrinsics(calib_dir, names, reference):
    data = json.loads((Path(calib_dir) / "extrinsics.json").read_text())
    out = {}
    for n in names:
        if n not in data:
            raise SystemExit(f"camera {n!r} absent from extrinsics.json")
        e = data[n]
        E = np.eye(4)
        E[:3, :3] = np.asarray(e["R"], np.float64).reshape(3, 3)
        E[:3, 3] = np.asarray(e["t"], np.float64).ravel()
        out[n] = E
        if e.get("reference") not in (None, reference):
            print(f"[warn] {n} referenced to {e.get('reference')!r}, "
                  f"not {reference!r}")
        lens = e.get("lens_id")
        if lens is not None and GT_LENS_ID is not None and lens != GT_LENS_ID:
            print(f"[WARN] the extrinsic for {n} was solved under lens {lens}, "
                  f"but the anchor was measured under {GT_LENS_ID}")
    return out


def camera_centre(E):
    return -E[:3, :3].T @ E[:3, 3]


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def undistort(img, K, dist):
    if cv2 is None:
        raise SystemExit("opencv required for undistortion; "
                         "install it or pass --no-undistort")
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    out = cv2.undistort(img, K, dist, None, newK)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        out = out[y:y + rh, x:x + rw]
        newK = newK.copy()
        newK[0, 2] -= x
        newK[1, 2] -= y
    return out, newK


# --------------------------------------------------------------------------
# depth-map geometry
# --------------------------------------------------------------------------

def pixel_rays(shape, K):
    """Unit-z camera-frame directions for every pixel."""
    h, w = shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def points_camera(depth, K):
    return pixel_rays(depth.shape, K) * depth[..., None]


def edge_mask(depth, rel_thresh, dilate=1):
    """True where the depth map is locally smooth.

    Relative gradient because absolute gradient scales with distance.
    """
    gy, gx = np.gradient(depth.astype(np.float64))
    grad = np.hypot(gx, gy) / np.maximum(depth, 1e-6)
    smooth = grad <= rel_thresh
    if dilate > 0 and cv2 is not None:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        smooth = cv2.erode(smooth.astype(np.uint8), k).astype(bool)
    return smooth


def normals_from_points(pts_cam):
    """Surface normals via cross products of neighbour differences."""
    dv = np.zeros_like(pts_cam)
    du = np.zeros_like(pts_cam)
    du[:, 1:-1] = pts_cam[:, 2:] - pts_cam[:, :-2]
    dv[1:-1, :] = pts_cam[2:, :] - pts_cam[:-2, :]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def incidence_mask(pts_cam, max_deg):
    """True where the surface faces the camera within max_deg of the ray."""
    n = normals_from_points(pts_cam)
    rays = pts_cam / np.maximum(
        np.linalg.norm(pts_cam, axis=-1, keepdims=True), 1e-12)
    cos = np.abs(np.sum(n * rays, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    ang[~np.isfinite(ang)] = 90.0
    return ang <= max_deg, ang


def to_world(pts_cam, E):
    R, t = E[:3, :3], E[:3, 3]
    return (pts_cam - t) @ R          # R^T @ (X - t)


def to_o3d(points, colors=None, rgb01=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb01 is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(rgb01, (len(points), 1)))
    elif colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, np.float64) / 255.0)
    return pcd


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(model_id, device):
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained(model_id)
    return m.to(device) if hasattr(m, "to") else m


def as_numpy(v):
    if torch is not None and isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="DA3 multi-camera fusion with calibrated conditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--snapshot-dir", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cameras", nargs="+",
                    default=["left", "center", "right", "top"])
    ap.add_argument("--reference", default="center")
    ap.add_argument("--mode", choices=["prior", "noprior"], default="prior")
    ap.add_argument("--ext", default="png")
    ap.add_argument("--model",
                    default="depth-anything/da3nested-giant-large-1.1")
    ap.add_argument("--process-res", type=int, default=504,
                    help="must be a multiple of the ViT patch size 14")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--edge-thresh", type=float, default=0.02)
    ap.add_argument("--edge-dilate", type=int, default=1)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--undistort", dest="undistort", action="store_true",
                    default=True)
    ap.add_argument("--no-undistort", dest="undistort", action="store_false")
    ap.add_argument("--colour-by-camera", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if o3d is None:
        raise SystemExit("open3d is required")
    if args.process_res % 14:
        print(f"[warn] process_res {args.process_res} is not a multiple of 14")

    names = list(args.cameras)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    intr = {n: load_intrinsics(args.calib_dir, n) for n in names}
    ext = load_extrinsics(args.calib_dir, names, args.reference)
    focal = check_focal_spread(intr, names)

    lenses = {intr[n].get("lens_id") for n in names}
    if len(lenses) > 1:
        raise SystemExit(
            f"the intrinsics describe more than one lens "
            f"{sorted(map(str, lenses))}. A rig calibrated across two focal "
            f"lengths cannot be fused; recalibrate every camera under the "
            f"fitted optics.")

    images, Ks_full = [], []
    for n in names:
        p = args.snapshot_dir / f"{n}.{args.ext}"
        if not p.exists():
            raise SystemExit(f"missing image: {p}")
        img = load_image(p)
        K = intr[n]["K"]
        if args.undistort:
            img, K = undistort(img, K, intr[n]["dist"])
        images.append(img)
        Ks_full.append(K)
        print(f"[input] {n:<8} {img.shape[1]}x{img.shape[0]}  "
              f"fx={K[0, 0]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}  "
              f"calib_rms={intr[n]['rms_px']:.3f}px  "
              f"lens={intr[n].get('lens_id')}")

    sizes = {(im.shape[1], im.shape[0]) for im in images}
    if len(sizes) > 1:
        print(f"[WARN] the undistorted images do not share a size "
              f"{sorted(sizes)}. Mixed aspect ratios in one DA3 batch "
              f"centre-crop to the smallest common dimension. Use "
              f"da3_stream.py or res_sweep.py with --pad-square.")

    Es = np.stack([ext[n] for n in names])
    print()
    for n, E in zip(names, Es):
        c = camera_centre(E)
        print(f"[pose ] {n:<8} C={c.round(4)}  |C|={np.linalg.norm(c):.4f} m")
    print()

    # ---- inference ----------------------------------------------------
    model = load_model(args.model, args.device)
    t0 = time.perf_counter()
    if args.mode == "prior":
        pred = model.inference(images, intrinsics=np.stack(Ks_full),
                               extrinsics=Es, align_to_input_ext_scale=True,
                               process_res=args.process_res)
    else:
        pred = model.inference(images, process_res=args.process_res)
    infer_ms = (time.perf_counter() - t0) * 1e3

    depth = as_numpy(pred.depth)
    conf = as_numpy(pred.conf)
    K_out = as_numpy(pred.intrinsics)
    E_out = as_numpy(pred.extrinsics)
    rgb = as_numpy(pred.processed_images)
    is_metric = int(getattr(pred, "is_metric", 0))
    scale_factor = float(getattr(pred, "scale_factor", float("nan")))

    print(f"[pred ] depth {depth.shape}  metric={is_metric}  "
          f"scale_factor={scale_factor:.4f}")
    if not is_metric:
        print("[warn] pred.is_metric is 0 -- depth is NOT in metres.")

    # ---- back-projection ----------------------------------------------
    t0 = time.perf_counter()
    clouds, tinted, per_cam = {}, [], []

    for i, n in enumerate(names):
        d = depth[i].astype(np.float64)
        c = conf[i]
        K_i = K_out[i]

        valid = np.isfinite(d) & (d > 0)
        n_start = int(valid.sum())
        drops = {}

        if args.conf_percentile > 0:
            thr = float(np.percentile(c[valid], args.conf_percentile))
            before = int(valid.sum())
            valid &= c >= thr
            drops["conf"] = before - int(valid.sum())
        else:
            thr = None

        if args.edge_thresh > 0:
            before = int(valid.sum())
            valid &= edge_mask(d, args.edge_thresh, args.edge_dilate)
            drops["edge"] = before - int(valid.sum())

        pts_cam = points_camera(d, K_i)

        inc_med = None
        if args.max_incidence < 90:
            ok, ang = incidence_mask(pts_cam, args.max_incidence)
            inc_med = float(np.median(ang[valid])) if valid.any() else None
            before = int(valid.sum())
            valid &= ok
            drops["grazing"] = before - int(valid.sum())

        pts_world = to_world(pts_cam[valid], E_out[i])
        cols = rgb[i][valid]
        clouds[n] = to_o3d(pts_world, colors=cols)
        if args.colour_by_camera:
            tinted.append(to_o3d(pts_world, rgb01=PALETTE[i % len(PALETTE)]))

        s = d.shape[1] / images[i].shape[1]
        fx_expected = Ks_full[i][0, 0] * s
        dv = d[valid]

        rec = {
            "camera": n,
            "depth_shape": list(d.shape),
            "input_size": [int(images[i].shape[1]), int(images[i].shape[0])],
            "K_out_fx": float(K_i[0, 0]),
            "fx_expected_from_calib": float(fx_expected),
            "fx_ratio": float(K_i[0, 0] / fx_expected),
            "conf_threshold": thr,
            "incidence_median_deg": inc_med,
            "median_depth_m": float(np.median(dv)),
            "depth_p05_m": float(np.percentile(dv, 5)),
            "depth_p95_m": float(np.percentile(dv, 95)),
            "n_points_initial": n_start,
            "n_points_valid": int(valid.sum()),
            "n_points_total": int(d.size),
            "dropped": drops,
        }
        per_cam.append(rec)
        dstr = " ".join(f"{k}={v}" for k, v in drops.items())
        print(f"[cloud] {n:<8} median={rec['median_depth_m']:.4f} m  "
              f"pts={rec['n_points_valid']:>7d}  fx_ratio={rec['fx_ratio']:.4f}"
              + (f"  dropped[{dstr}]" if dstr else ""))

    backproject_ms = (time.perf_counter() - t0) * 1e3

    # ---- fuse ---------------------------------------------------------
    t0 = time.perf_counter()
    fused = o3d.geometry.PointCloud()
    for n in names:
        fused += clouds[n]
    n_before = len(fused.points)
    if args.voxel > 0:
        fused = fused.voxel_down_sample(args.voxel)
    fuse_ms = (time.perf_counter() - t0) * 1e3

    # ---- write --------------------------------------------------------
    t0 = time.perf_counter()
    for i, n in enumerate(names):
        o3d.io.write_point_cloud(str(args.out_dir / f"cloud_{n}.ply"), clouds[n])
        np.save(args.out_dir / f"depth_{n}.npy", depth[i])
        np.save(args.out_dir / f"conf_{n}.npy", conf[i])
        np.save(args.out_dir / f"K_{n}.npy", K_out[i])
        np.save(args.out_dir / f"E_{n}.npy", E_out[i])
    o3d.io.write_point_cloud(str(args.out_dir / "fused.ply"), fused)
    if tinted:
        merged = o3d.geometry.PointCloud()
        for p in tinted:
            merged += p
        o3d.io.write_point_cloud(str(args.out_dir / "fused_by_camera.ply"),
                                 merged)
    write_ms = (time.perf_counter() - t0) * 1e3

    bbox = fused.get_axis_aligned_bounding_box()
    diagnostics = {
        "mode": args.mode, "model": args.model,
        "snapshot_dir": str(args.snapshot_dir),
        "calib_dir": str(args.calib_dir),
        "reference_camera": args.reference, "cameras": names,
        "lens_id": intr[names[0]].get("lens_id"),
        "lens_mm": intr[names[0]].get("lens_mm"),
        "focal_spread": focal,
        "pad_square": False, "rot_k_applied": False,
        "undistorted": args.undistort,
        "process_res": args.process_res,
        "gave_intrinsics": args.mode == "prior",
        "gave_extrinsics": args.mode == "prior",
        "is_metric": is_metric, "scale_factor": scale_factor,
        "filters": {"conf_percentile": args.conf_percentile,
                    "edge_thresh": args.edge_thresh,
                    "edge_dilate": args.edge_dilate,
                    "max_incidence_deg": args.max_incidence},
        "ground_truth": {"reference_planes": GT_PLANES,
                         "span_m": GT.get("span_m"),
                         "separation_m": GT_SEPARATION_M,
                         "deck_perp_m": GT_DECK_PERP_M,
                         "lens_id": GT_LENS_ID,
                         "measured": GT.get("measured"),
                         "source": GT.get("source"),
                         "usable": bool(GT.get("ok")),
                         "reason": GT.get("reason"),
                         "warnings": GT.get("warnings", [])},
        "per_camera": per_cam,
        "fused": {"n_points_before_voxel": int(n_before),
                  "n_points_written": int(len(fused.points)),
                  "voxel_m": args.voxel,
                  "bbox_min": np.asarray(bbox.min_bound).tolist(),
                  "bbox_max": np.asarray(bbox.max_bound).tolist()},
        "timings_ms": {"inference_ms": round(infer_ms, 3),
                       "backproject_ms": round(backproject_ms, 3),
                       "fuse_ms": round(fuse_ms, 3),
                       "write_ms": round(write_ms, 3),
                       "total_ms": round((time.perf_counter() - t_total) * 1e3, 3)},
    }
    if torch is not None and torch.cuda.is_available():
        diagnostics["memory"] = {
            "gpu_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
    (args.out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    # ---- self-checks --------------------------------------------------
    print()
    meds = np.array([r["median_depth_m"] for r in per_cam])
    counts = [r["n_points_valid"] for r in per_cam]
    ratios = np.array([r["fx_ratio"] for r in per_cam])

    print(f"median depth        : {meds.min():.4f} - {meds.max():.4f} m")
    if GT["ok"] and np.isfinite(GT_DECK_PERP_M):
        err = (meds.mean() - GT_DECK_PERP_M) / GT_DECK_PERP_M * 100
        print(f"  ok    against the measured {GT_DECK_PERP_M:.3f} m deck "
              f"standoff the mean sits {err:+.1f}% out; that difference IS the "
              f"unanchored scale error, not a fault")
    else:
        print("  ??    no usable anchor to compare against; run measure_deck.py")

    print(f"fx_ratio            : {ratios.min():.4f} - {ratios.max():.4f}")
    if not np.all((ratios > 0.98) & (ratios < 1.02)):
        print("  FAIL  returned K disagrees with calibrated K rescaled.")
    else:
        print("  ok    returned K matches calibration.")

    print(f"valid points        : {counts}")
    if len(set(counts)) == 1:
        print("  FAIL  identical across cameras -- filtering is doing nothing.")
    else:
        print("  ok    varies per camera as expected.")

    print(f"\nwritten: {args.out_dir}")
    if not GT["ok"]:
        print("\nNOTE: no metric anchor, so anything derived from "
              "--apply-absolute downstream\n      will be NaN by design. Run "
              "measure_deck.py first.")


if __name__ == "__main__":
    main()