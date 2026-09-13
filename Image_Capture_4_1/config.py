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
# same instant. 3000 us at f/~4 under the EC hall lights freezes hand-held
# board motion; raise gain (not exposure) if the image is too dark.
EXPOSURE_AUTO    = "Off"
EXPOSURE_TIME_US = 3000.0
GAIN_AUTO        = "Off"
GAIN_DB          = 0.0

# Four 5 MP streams on one link is ~2x the burst the three-camera rig produced.
# Keep the extrinsics shoot triggered/synchronised (PTP or hardware trigger) --
# free-running acquisition de-synchronises the views and that shows up directly
# as inflated stereo reprojection error, not as an obvious capture failure.

# === Paths ===
CAPTURE_DIR = "captures"             # per-camera intrinsics shoots: captures/<name>/frame_XXXX.png
# SIMULTANEOUS multi-camera frames for extrinsics. Same index == same instant
# across cameras, board visible to the reference AND the target in that frame.
# With four cameras a single board pose rarely satisfies all pairs at once --
# that's fine, each pair is solved independently, so shoot separate sweeps
# aimed at center<->left, center<->right and center<->top rather than hunting
# for poses all four can see.
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
    # default or large tilted squares stop binarising cleanly.
    detector_params.adaptiveThreshWinSizeMax = 53
    refine_params = cv2.aruco.RefineParameters()
    return cv2.aruco.CharucoDetector(board, charuco_params,
                                     detector_params, refine_params)