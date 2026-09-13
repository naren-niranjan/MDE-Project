"""Probe which ArUco dictionary your printed board uses.

Capture (or take a phone photo of) the board so most/all markers are clearly visible,
then run:

    python probe_dict.py <image.jpg>

It tries every standard dictionary and reports how many markers each detects.
The dictionary the board was printed against will detect ~16 markers (for a 5x5
ChArUco board you have 12 markers on a 5x5 board with alternating pattern, actually:
on a 5x5 board with the default colouring, there are ceil(25/2) = 13 markers in white
squares  -- whatever the exact count, the correct dictionary will detect ~all of them
and the wrong ones will detect 0 or 1).
"""
import sys
import cv2

DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


def main(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Image: {path}   shape={gray.shape}\n")

    best = (0, None)
    params = cv2.aruco.DetectorParameters()
    for name in DICT_NAMES:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        det = cv2.aruco.ArucoDetector(d, params)
        _, ids, _ = det.detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        flag = ""
        if n > best[0]:
            best = (n, name)
        if n >= 4:
            flag = "   <-- candidate"
        print(f"  {name:<25} {n:>3} marker(s){flag}")

    print()
    if best[1] is None or best[0] < 4:
        print("No dictionary matched cleanly. Re-shoot the board flat, well lit, in focus.")
    else:
        print(f"Best match: {best[1]} ({best[0]} markers).")
        print(f"Edit config.py:  ARUCO_DICT_NAME = \"{best[1]}\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python probe_dict.py <image>")
        sys.exit(1)
    main(sys.argv[1])
