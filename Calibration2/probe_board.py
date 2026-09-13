"""Confirm the ChArUco *board* settings on a static frame.

Markers are already known to decode (DICT_5X5_100). This isolates the next
question: do the chessboard CORNERS interpolate? It tries both legacy settings
and both X/Y orderings and reports the corner count for each, so you can lock in
BOARD_SQUARES_X/Y and the legacy flag without touching the camera rig.

Usage:
    python probe_board.py debug_center_<timestamp>.png
    python probe_board.py            # auto-grabs newest debug_*.png
"""

import sys
import glob
import cv2
import numpy as np

DICT_NAME     = "DICT_5X5_100"
SQUARE_LENGTH = 0.060
MARKER_LENGTH = 0.047


def params():
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin    = 3
    p.adaptiveThreshWinSizeMax    = 53
    p.adaptiveThreshWinSizeStep   = 10
    p.minMarkerPerimeterRate      = 0.01
    p.maxMarkerPerimeterRate      = 4.0
    p.polygonalApproxAccuracyRate = 0.08
    p.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("debug_*.png"))[-1]
    img  = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))

    # Raw markers first -> establishes how many are present and the ID offset.
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params()).detectMarkers(gray)
    if ids is None:
        raise SystemExit(f"No markers with {DICT_NAME} -- wrong dictionary.")
    ids = ids.flatten()
    offset = int(ids.min())
    print(f"\n{path}")
    print(f"{DICT_NAME}: {len(ids)} markers, IDs {ids.min()}..{ids.max()}, "
          f"detected offset = {offset}\n")

    print(f"{'dims':<9} {'legacy':<7} charuco_corners")
    print("-" * 35)
    best = (-1, None, None)
    for dims in [(9, 12), (12, 9)]:
        n_markers = (dims[0] * dims[1]) // 2
        board_ids = np.arange(offset, offset + n_markers, dtype=np.int32)
        for legacy in (True, False):
            board = cv2.aruco.CharucoBoard(dims, SQUARE_LENGTH,
                                           MARKER_LENGTH, dictionary, board_ids)
            if hasattr(board, "setLegacyPattern"):
                board.setLegacyPattern(legacy)
            det = cv2.aruco.CharucoDetector(board)
            det.setDetectorParameters(params())
            cc, ci, _, _ = det.detectBoard(gray)
            n = 0 if ci is None else len(ci)
            print(f"{str(dims):<9} {str(legacy):<7} {n}")
            if n > best[0]:
                best = (n, dims, legacy)

    print("\n" + "=" * 50)
    if best[0] <= 0:
        print("Still 0 corners with every combo. Markers decode but the board")
        print("geometry won't interpolate -- re-check SQUARE/MARKER lengths or")
        print("whether MARKER_ID_OFFSET matches the printed first ID.")
    else:
        n, dims, legacy = best
        print(f"BEST: {n} corners with dims={dims}, legacy={legacy}, offset={offset}")
        print("\nSet in config.py:")
        print(f"    BOARD_SQUARES_X  = {dims[0]}")
        print(f"    BOARD_SQUARES_Y  = {dims[1]}")
        print(f"    MARKER_ID_OFFSET = {offset}")
        print(f"    # setLegacyPattern({legacy}) in get_charuco_board()")
    print("=" * 50)


if __name__ == "__main__":
    main()