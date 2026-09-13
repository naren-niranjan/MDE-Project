#!/usr/bin/env bash
# run_da3_subsets.sh — E1: one LIVE da3_stream.py capture per camera subset.
#
# da3_stream.py opens the cameras; there is no replay mode. The rig must be
# powered, the parcels in place, and the scene must not move between the seven
# captures. Take the reference scan in the same session.
#
#   bash run_da3_subsets.sh
#   python3 fuse_subset.py --capture-root runs_da3 --calib "$CALIB" --out runs
#   python3 grade_subsets.py --dir runs --exclude P1 --csv results.csv
#
# Press the capture key (p) once per subset when prompted. A key-forced capture
# writes the npy arrays regardless of --save-npy, but it is passed anyway.
set -euo pipefail

CALIB=${CALIB:-/home/jetson/Projects/Calibration4/results}
OUT=${OUT:-runs_da3}
RES=${RES:-1008}

# --- HOLD THESE CONSTANT ACROSS ALL SEVEN -----------------------------------
# --edge-thresh is per-pixel: the 0.02 default is for process-res 504, so it
#   halves to 0.01 at 1008. Rescale again if you change RES.
# --correction is deliberately absent. grade_subsets.py fits its own alpha/beta
#   per subset from the deck and floor and grades the parcels held out; feeding
#   a pre-fitted correction makes every subset look alike.
# --expect-lens "" because the default is EO-58-001-12mm and this is the 8 mm
#   rig. The Calibration4 records carry no lens_id at all, so an empty string
#   is what accepts them.
COMMON=(
  --calib-dir "$CALIB"
  --expect-lens ""
  --process-res "$RES"
  --edge-thresh 0.01
  --edge-dilate 1
  --conf-percentile 40
  --max-incidence 70
  --fuse-mode union
  --rect-mode common
  --save-npy --save-mode all --save-per-camera
  --no-segment
  --capture-trigger key
  --max-frames 1
)
# ----------------------------------------------------------------------------

run() {
  local cams=$1 name=$2 ref=$3
  if [[ -d "$OUT/$name" ]]; then echo "[skip] $name"; return; fi
  echo
  echo "=============================================================="
  echo " $name   (--cameras $cams, reference $ref)"
  echo " press 'p' to capture. DO NOT MOVE THE SCENE between subsets."
  echo "=============================================================="
  mkdir -p "$OUT/$name"
  python3 da3_stream.py "${COMMON[@]}" \
      --cameras $cams --reference "$ref" --out-dir "$OUT/$name"
}

# reference must be a member of the subset; fuse_subset.py re-derives the rig
# frame from the extrinsics either way, so this only affects da3_stream's own
# internal frame.
run "center"              center             center
run "left"                left               left
run "right"               right              right
run "center left"         center+left        center
run "center right"        center+right       center
run "left right"          left+right         left
run "center left right"   center+left+right  center

cat <<'EOF'

Done. Before grading, check one thing:

  python3 inspect_capture.py --capture-root runs_da3 --compare-k

--rect-mode common crops every view to one SHARED size, so the crop depends on
which cameras are in the subset. If center's K differs between runs_da3/center
and runs_da3/center+left+right, then center did not receive the same pixels in
every subset and part of any difference you measure is the crop, not the
geometry. The check prints them side by side.

Then:
  python3 fuse_subset.py --capture-root runs_da3 --calib "$CALIB" --out runs
  python3 grade_subsets.py --dir runs --exclude P1 --csv results.csv
EOF