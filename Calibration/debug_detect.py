"""Run the same detection capture.py uses, on a saved frame, with verbose output."""
import sys, glob, cv2, numpy as np
import config

if len(sys.argv) > 1:
    paths = sys.argv[1:]
else:
    # Default: try one frame from each camera if captures exist
    paths = []
    for name in config.CAMERAS:
        cand = sorted(glob.glob(f"{config.CAPTURE_DIR}/{name}/frame_*.png"))
        if cand:
            paths.append(cand[0])
    if not paths:
        print("No captured frames found. Usage: python debug_detect.py <image1> [image2 ...]")
        sys.exit(1)

board       = config.get_charuco_board()
dictionary  = config.get_aruco_dictionary()
print(f"Dictionary: {config.ARUCO_DICT_NAME}")
print(f"Board    : {config.BOARD_SQUARES_X}x{config.BOARD_SQUARES_Y}, "
      f"square={config.SQUARE_LENGTH*1000:.1f}mm, marker={config.MARKER_LENGTH*1000:.1f}mm\n")

aruco_params   = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
ch_detector    = cv2.aruco.CharucoDetector(board)

for path in paths:
    img = cv2.imread(path)
    if img is None:
        print(f"--- {path} : cannot read ---\n"); continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"--- {path}  ({w}x{h}, mean intensity {gray.mean():.0f}) ---")

    # Step 1: bare ArUco marker detection
    m_corners, m_ids, rejected = aruco_detector.detectMarkers(gray)
    nm = 0 if m_ids is None else len(m_ids)
    nr = 0 if rejected is None else len(rejected)
    print(f"  ArUco markers detected: {nm}   (rejected candidates: {nr})")
    if m_ids is not None and len(m_ids) > 0:
        print(f"    IDs: {sorted(m_ids.flatten().tolist())}")

    # Step 2: ChArUco corner interpolation
    ch_c, ch_ids, _, _ = ch_detector.detectBoard(gray)
    nc = 0 if ch_ids is None else len(ch_ids)
    print(f"  ChArUco corners       : {nc}")
    if ch_ids is not None and len(ch_ids) > 0:
        print(f"    Corner IDs: {sorted(ch_ids.flatten().tolist())}")

    # Step 3: also try with detect on half-res, just to compare
    small = cv2.resize(gray, (w//2, h//2))
    sc, sids, _, _ = ch_detector.detectBoard(small)
    print(f"  ChArUco at half-res   : {0 if sids is None else len(sids)} corners")
    print()
