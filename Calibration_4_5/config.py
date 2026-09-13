"""Calibration configuration. Edit the values below to match your setup."""

import cv2
import numpy as np

# === ChArUco board ===
# Physical board: calib.io 9x12, checker 60 mm, marker 47 mm, dict ArUco DICT_5X5.
BOARD_SQUARES_X = 12
BOARD_SQUARES_Y = 9        # even row count -> legacy pattern matters (see below)
SQUARE_LENGTH   = 0.060     # metres  (60 mm)
MARKER_LENGTH   = 0.047     # metres  (47 mm)

# A 9x12 board carries (9*12)//2 = 54 markers, so the dictionary MUST hold >= 54.
# That rules out DICT_5X5_50. The bit patterns differ between _100/_250/_1000,
# so this has to match EXACTLY what calib.io used -- confirm with probe_dict.py.
ARUCO_DICT_NAME = "DICT_5X5_250"

# The printed board uses marker IDs starting from this offset.
# calib.io's default start ID is 0 (i.e. the board starts at 0, NOT 1). Leave at
# 0 unless probe_dict.py shows the first detected marker is some other ID.
MARKER_ID_OFFSET = 0

# Outer dimensions of the printed pattern, used to work out how far the board
# has to be from a camera to fill a sensible fraction of the frame. With the
# 12 mm lens this is no longer a detail you can ignore: the board now has to be
# roughly 1.5x further away than it was with the 8 mm lens to frame the same.
BOARD_WIDTH_M  = BOARD_SQUARES_X * SQUARE_LENGTH   # 0.72 m
BOARD_HEIGHT_M = BOARD_SQUARES_Y * SQUARE_LENGTH   # 0.54 m

# === Cameras ===
# Map a friendly name to the camera's serial number (see list_cameras.py).
# Physical arrangement in the rig (dict order below follows it):
#
#        [top]                      <- 261100631, raised above the row
#   [left] [center] ....... [right] <- the horizontal row, left to right
#
# NOTE: the serials moved around when the fourth camera went in -- 261100628
# used to be "left" and 261100627 used to be "center". Any results/*.json or
# .npz written before this change is keyed by the OLD names and is now WRONG.
# Delete or archive the old results directory before re-running.
CAMERAS = {
    "left":   "260505158",   # 1st in row, leftmost
    "center": "261100628",   # 2nd in row, centre bottom          <-- reference
    "top":    "261100631",   # 3rd camera, above the row
    "right":  "261100627",   # 4th in row, rightmost
}

# The reference is the origin of the multi-camera frame: every other camera's
# extrinsics are solved as target-w.r.t-reference, so it should be the camera
# that shares the largest overlapping view volume with ALL the others. The
# centre-bottom camera is the only one with decent baselines to left, right and
# top at once -- picking "left" would make the left->right pair a ~2x longer
# baseline with less common board coverage.
REFERENCE_CAMERA = "center"

# Every camera except the reference gets its own extrinsic solve against it.
TARGET_CAMERAS = [name for name in CAMERAS if name != REFERENCE_CAMERA]

# === Acquisition ===
WIDTH  = 2448
HEIGHT = 2048

# Calibration captures MUST have pinned exposure/gain: auto exposure changing
# between (and during) frames shifts blur and contrast per view, and on the
# extrinsics shoot the four cameras would each pick different settings for the
# same instant.
#
# The 12 mm f/1.8 lens is roughly two stops faster wide open than the previous
# glass, but calibration is NOT shot wide open (see the optics block below --
# depth of field at f/1.8 is far too thin for a tilted board). Shot at the
# recommended f/5.6 the exposure needed lands close to the old value, so 3000 us
# is kept as the fixed-mode default. In practice use auto_capture.py's default
# `--exposure auto`, which converges once against the actual hall lighting and
# then LOCKS the value, and watch the gain it settles on: with the faster lens
# it should now be well under the old 18.7 dB ceiling. If it is not, the iris is
# stopped down further than you think.
EXPOSURE_AUTO    = "Off"
EXPOSURE_TIME_US = 3000.0
GAIN_AUTO        = "Off"
GAIN_DB          = 0.0

# Four 5 MP streams on one link is ~2x the burst the three-camera rig produced.
# Keep the extrinsics shoot triggered/synchronised (PTP or hardware trigger) --
# free-running acquisition de-synchronises the views and that shows up directly
# as inflated stereo reprojection error, not as an obvious capture failure.

# === Optics =================================================================
# LENS CHANGE: the rig moved from 8 mm glass to the Edmund Optics TECHSPEC
# C Series 12 mm f/1.8 (EO model 58-001, LUCID SKU EOC0120) on every camera.
#
# This block is the SINGLE SOURCE OF TRUTH for the focal length. auto_capture.py
# and calibrate_intrinsics.py both used to carry their own copy of the 8 mm
# number; they now import from here, so a future lens swap is one edit.
#
# Everything downstream of a focal length changes with it. The consequences that
# actually matter for this rig:
#
#   * Expected fx moves 2319 px -> 3478 px. The focal guard in
#     calibrate_intrinsics.py is a RATIO against this number, so a solve that
#     lands near 2319 px now (i.e. the old lens' answer, or a stale result file)
#     is correctly rejected instead of silently accepted.
#   * Field of view narrows from 55.7 x 47.7 deg to 38.8 x 32.8 deg. At the
#     3.16 m deck each camera now sees 2.22 x 1.86 m instead of 3.34 x 2.79 m.
#     Verify every camera still covers the belt footprint before shooting, and
#     re-measure belt.json afterwards -- the frozen footprint from the 8 mm rig
#     is wrong.
#   * Ground sample distance improves 1.36 -> 0.91 mm/px at the deck, which is
#     the upside: ~33 % finer sampling on parcel faces.
#   * Pair overlap shrinks by the same 1.5x. The extrinsic shoot must be done
#     pair-by-pair (center+left, center+right, center+top); hunting for a pose
#     all four cameras see was already impractical and is now hopeless.
#   * The sensor diagonal is 11.011 mm and the lens image circle is 2/3" =
#     11.0 mm. There is ZERO margin. Expect some corner falloff, which is
#     exactly where calibration wants corners, so keep the CLAHE detection
#     strategy enabled and check that frames still yield corners at the frame
#     edges.
LENS_NAME = "Edmund Optics TECHSPEC C Series 12 mm f/1.8 (EO 58-001 / EOC0120)"
LENS_ID   = "EO-58-001-12mm"      # written into results/*.json for staleness
LENS_MM   = 12.0
PIXEL_SIZE_MM = 0.00345           # IMX264: 3.45 um
LENS_IMAGE_CIRCLE_MM     = 11.0   # 2/3" max sensor format, per datasheet
LENS_DISTORTION_SPEC_PCT = 2.5    # datasheet "<2.5 %" at max field
LENS_APERTURE_RANGE      = "f/1.8 - f/16"

# The aperture and focus you calibrate at are part of the calibration. Stopping
# down or refocusing shifts the effective focal length, the principal point and
# the distortion curve, so both must be LOCKED (the C Series has recessed set
# screws for exactly this) and left alone between calibration and deployment.
#
# f/5.6 is the recommended setting for this rig, focused at the deck:
#   f/1.8  DOF 2.48 - 4.34 m   too thin for a tilted board at 2 m
#   f/4    DOF 1.97 - 8.01 m   good, Airy disc 1.6 px
#   f/5.6  DOF 1.71 - inf      best board coverage, Airy disc 2.2 px  <-- use
#   f/8    DOF 1.43 - inf      diffraction starts costing corner accuracy
CALIBRATION_APERTURE = "f/5.6"
WORKING_DISTANCE_M   = 3.16       # conveyor deck, the focus target
COC_PX               = 2.0        # circle of confusion used for the DOF numbers

# Nominal focal in pixels. Used as the calibrateCamera seed, as the guard-rail
# reference, and as the K for the capture-time novelty pose.
EXPECTED_FX = LENS_MM / PIXEL_SIZE_MM     # ~3478.3 px

# Accept/reject band for the solved focal. Anything outside this is the planar
# focal/distance degeneracy, not a real lens.
FX_RATIO_LO = 0.80
FX_RATIO_HI = 1.25


def sensor_size_mm():
    """Active sensor size (w, h) in mm at the configured ROI."""
    return WIDTH * PIXEL_SIZE_MM, HEIGHT * PIXEL_SIZE_MM


def sensor_diagonal_mm():
    w, h = sensor_size_mm()
    return float(np.hypot(w, h))


def fov_deg():
    """(horizontal, vertical, diagonal) field of view in degrees."""
    w, h = sensor_size_mm()
    d = sensor_diagonal_mm()
    f = lambda s: float(2 * np.degrees(np.arctan(s / (2 * LENS_MM))))
    return f(w), f(h), f(d)


def footprint_m(distance_m):
    """(width, height) in metres imaged at `distance_m`."""
    w, h = sensor_size_mm()
    return distance_m * w / LENS_MM, distance_m * h / LENS_MM


def gsd_mm(distance_m):
    """Ground sample distance in mm per pixel at `distance_m`."""
    return PIXEL_SIZE_MM * (distance_m * 1000.0 / LENS_MM)


def depth_of_field_m(distance_m, f_number, coc_px=COC_PX):
    """(near, far, hyperfocal) in metres for a circle of confusion of
    `coc_px` pixels. `far` is inf once the focus distance reaches hyperfocal."""
    c = coc_px * PIXEL_SIZE_MM                 # mm
    H = LENS_MM ** 2 / (f_number * c)          # mm
    s = distance_m * 1000.0
    near = H * s / (H + s) / 1000.0
    far = float("inf") if s >= H else H * s / (H - s) / 1000.0
    return near, far, H / 1000.0


def board_distance_range_m(min_frac=0.30, max_frac=0.95):
    """Distances at which the ChArUco board frames sensibly: nearest is where
    the board height fills `max_frac` of the frame height, furthest is where its
    width still covers `min_frac` of the frame width.

    With the 12 mm lens this is ~0.97 m to ~3.41 m. The near end is then clipped
    by depth of field -- at f/5.6 focused on the deck nothing closer than
    ~1.71 m is sharp -- so the usable shooting band is roughly 1.8 - 3.4 m.
    """
    w_s, h_s = sensor_size_mm()
    d_near = (BOARD_HEIGHT_M / max_frac) * LENS_MM / h_s
    d_far = (BOARD_WIDTH_M / min_frac) * LENS_MM / w_s
    return d_near, d_far


def board_shooting_band_m(f_number=5.6, focus_m=WORKING_DISTANCE_M):
    """board_distance_range_m() intersected with the depth of field at the
    locked focus/aperture -- i.e. where the board is both well framed AND
    actually in focus. This is the range to walk the board through."""
    d_near, d_far = board_distance_range_m()
    dof_near, dof_far, _ = depth_of_field_m(focus_m, f_number)
    return max(d_near, dof_near), min(d_far, dof_far)


def describe_optics(f_number=5.6):
    """Human-readable optics summary. Printed at the top of every capture so
    the numbers the shoot depends on are in the log next to the frames."""
    hf, vf, df = fov_deg()
    fw, fh = footprint_m(WORKING_DISTANCE_M)
    near, far, hyper = depth_of_field_m(WORKING_DISTANCE_M, f_number)
    b_lo, b_hi = board_shooting_band_m(f_number)
    diag = sensor_diagonal_mm()
    lines = [
        f"Lens      : {LENS_NAME}",
        f"            {LENS_MM:g} mm on {WIDTH}x{HEIGHT} @ "
        f"{PIXEL_SIZE_MM*1e3:g} um  ->  expected fx = {EXPECTED_FX:.0f} px",
        f"FOV       : {hf:.1f} x {vf:.1f} deg  (diagonal {df:.1f} deg)",
        f"At {WORKING_DISTANCE_M:.2f} m : {fw:.2f} x {fh:.2f} m footprint, "
        f"{gsd_mm(WORKING_DISTANCE_M):.2f} mm/px",
        f"At {f_number:g}    : DOF {near:.2f} - "
        f"{'inf' if far == float('inf') else f'{far:.2f}'} m "
        f"(hyperfocal {hyper:.2f} m), focused at {WORKING_DISTANCE_M:.2f} m",
        f"Board     : shoot between {b_lo:.1f} and {b_hi:.1f} m "
        f"({BOARD_WIDTH_M:.2f} x {BOARD_HEIGHT_M:.2f} m pattern)",
    ]
    if diag > LENS_IMAGE_CIRCLE_MM:
        lines.append(
            f"NOTE      : sensor diagonal {diag:.2f} mm vs {LENS_IMAGE_CIRCLE_MM:.1f} mm "
            f"image circle -- no margin, expect corner falloff")
    return "\n".join("  " + ln for ln in lines)


# === Paths ===
CAPTURE_DIR = "captures"             # per-camera intrinsics shoots: captures/<name>/frame_XXXX.png
# SIMULTANEOUS multi-camera frames for extrinsics. Same index == same instant
# across cameras, board visible to the reference AND the target in that frame.
# With four cameras a single board pose rarely satisfies all pairs at once --
# that's fine, each pair is solved independently, so shoot separate sweeps
# aimed at center<->left, center<->right and center<->top rather than hunting
# for poses all four can see. With the 12 mm lens the per-pair overlap is
# ~1.5x smaller than it was, so this is now the only workable approach.
# Keep this SEPARATE from CAPTURE_DIR -- reusing the intrinsics shoots gives
# unsynchronized views and garbage extrinsics.
EXTRINSIC_CAPTURE_DIR = "captures_extrinsic"
RESULTS_DIR = "results"


def get_aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))


def get_charuco_board():
    """Construct the ChArUco board with the correct marker ID offset.

    For a 9x12 board OpenCV places (9*12)//2 = 54 markers on the white squares.
    By default it labels them 0..53; we offset them by MARKER_ID_OFFSET so the
    IDs match the physically printed board.
    """
    dictionary = get_aruco_dictionary()
    n_markers  = (BOARD_SQUARES_X * BOARD_SQUARES_Y) // 2
    ids        = np.arange(MARKER_ID_OFFSET,
                           MARKER_ID_OFFSET + n_markers,
                           dtype=np.int32)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
        ids,
    )

    # calib.io boards use the pre-4.6 ("legacy") ChArUco layout. For boards with
    # an EVEN number of rows (ours is 12), the modern OpenCV layout differs from
    # what's printed, so the markers detect but ChArUco corner interpolation is
    # sparse/wrong. Opting back into the legacy pattern fixes it.
    # Requires OpenCV >= 4.8 (4.7 has no setter -- upgrade if needed).
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    else:
        import warnings
        warnings.warn(
            "This OpenCV build has no CharucoBoard.setLegacyPattern; upgrade to "
            "OpenCV >= 4.8 or detection of the calib.io board will be unreliable."
        )
    return board


def get_charuco_detector():
    """ChArUco detector built on get_charuco_board() with sub-pixel marker-corner
    refinement. Use this everywhere (intrinsics + extrinsics) so detection is
    identical across the pipeline and corners land at sub-pixel accuracy, which
    is what gets reprojection error below 1 px. The board carries the legacy
    pattern / ID offset, so all of that is inherited here automatically."""
    board = get_charuco_board()
    charuco_params  = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize       = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy   = 0.01
    # 2448 px frames need a wider adaptive-threshold sweep than the 23 px
    # default or large tilted squares stop binarising cleanly. The 12 mm lens
    # makes the board bigger in frame at a given distance, which if anything
    # argues for keeping this generous.
    detector_params.adaptiveThreshWinSizeMax = 53
    refine_params = cv2.aruco.RefineParameters()
    return cv2.aruco.CharucoDetector(board, charuco_params,
                                     detector_params, refine_params)


if __name__ == "__main__":
    print(describe_optics())