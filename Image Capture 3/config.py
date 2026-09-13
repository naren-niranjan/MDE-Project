"""Calibration configuration. Edit the values below to match your setup."""

import cv2
import numpy as np

# === ChArUco board ===
BOARD_SQUARES_X = 5
BOARD_SQUARES_Y = 5
SQUARE_LENGTH   = 0.0965    # metres  (96.5 mm)
MARKER_LENGTH   = 0.07225   # metres  (72.25 mm)

# Set this AFTER running probe_dict.py on a captured board image.
# Common candidates: DICT_4X4_50, DICT_5X5_50, DICT_5X5_100, DICT_6X6_250.
ARUCO_DICT_NAME = "DICT_4X4_50"
# The printed board uses marker IDs starting from this offset.
MARKER_ID_OFFSET = 20

# === Cameras ===
# Map a friendly name to the camera's serial number (see list_cameras.py).
# Order doesn't matter, but REFERENCE_CAMERA below should be the center one.
CAMERAS = {
    "left":   "261100627",   # blue marker in your sketch  (tilts inward from left)
    "center": "261100631",   # green marker                (looks straight down)  <-- reference
    "right":  "261100628",   # red marker                  (tilts inward from right)
}
REFERENCE_CAMERA = "center"

# === Acquisition ===
WIDTH         = 2448
HEIGHT        = 2048
EXPOSURE_AUTO = "Continuous"
GAIN_AUTO     = "Continuous"
 
# === Paths ===
CAPTURE_DIR = "captures"
RESULTS_DIR = "results"
 
 
def get_aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
 
 
def get_charuco_board():
    """Construct the ChArUco board with the correct marker ID offset.
 
    For a 5x5 board OpenCV places (5*5)//2 = 12 markers on the white squares.
    By default it labels them 0..11; we override that to MARKER_ID_OFFSET..(+12).
    """
    dictionary = get_aruco_dictionary()
    n_markers  = (BOARD_SQUARES_X * BOARD_SQUARES_Y) // 2
    ids        = np.arange(MARKER_ID_OFFSET,
                           MARKER_ID_OFFSET + n_markers,
                           dtype=np.int32)
    return cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
        ids,
    )
