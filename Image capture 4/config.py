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
# Order doesn't matter, but REFERENCE_CAMERA below should be the center one.
CAMERAS = {
    "left":   "261100628",   # blue marker in your sketch
    "center": "261100627",   # green marker                <-- reference
    "right":  "261100631",   # red marker
}
REFERENCE_CAMERA = "center"

# === Acquisition ===
WIDTH  = 2448
HEIGHT = 2048

# Calibration captures MUST have pinned exposure/gain: auto exposure changing
# between (and during) frames shifts blur and contrast per view, and on the
# extrinsics shoot the three cameras would each pick different settings for the
# same instant. 3000 us at f/~4 under the EC hall lights freezes hand-held
# board motion; raise gain (not exposure) if the image is too dark.
EXPOSURE_AUTO    = "Off"
EXPOSURE_TIME_US = 3000.0
GAIN_AUTO        = "Off"
GAIN_DB          = 0.0

# === Paths ===
CAPTURE_DIR = "captures"             # per-camera intrinsics shoots: captures/<name>/frame_XXXX.png
# SIMULTANEOUS multi-camera frames for extrinsics. Same index == same instant
# across cameras, board visible to the reference AND the target in that frame.
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