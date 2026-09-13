#!/usr/bin/env bash
# run_subsets.sh — box-level pass: box_scene.py once per camera subset.
#
# box_scene.py consumes CLOUDS, not images. So the pipeline is:
#
#   da3_stream.py / capture  ->  per-view clouds in CAPTURE_DIR
#   fuse_subset.py           ->  runs/<subset>.ply      (cloud-level grading)
#   this script              ->  runs/<subset>/         (box-level grading)
#
# Run fuse_subset.py first; it needs no box_scene.py at all and gives you the
# deck-plane numbers, which are the part of this study with statistical power.
#
set -euo pipefail

CAPTURE=${CAPTURE:?set CAPTURE=/path/to/capture-dir}
OUT=${OUT:-runs}

# --- CONSTANT ACROSS SUBSETS -------------------------------------------------
# Every seg parameter must be identical for all seven, or the study measures
# tuning rather than geometry.
#
# --seg-min-views 1 is deliberate and not optional. The production default of 2
# withholds any parcel seen by fewer than two views, so a single-camera subset
# would return zero boxes and the pair subsets would be silently filtered by the
# very thing being measured. Hold it at 1 everywhere and let the grader compare
# geometry; view counts are reported separately by coverage.py.
SEG=(
  --segment
  --seg-min-views 1
  --seg-face-grow-scale 0.3
  --seg-dim-percentile 1.0
  --seg-max-view-shift 0.08
)
# -----------------------------------------------------------------------------

run() {
  local cams=$1 name=$2
  if [[ -d "$OUT/$name" ]]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  (--cameras $cams)"
  mkdir -p "$OUT/$name"
  python3 box_scene.py \
      --capture-dir "$CAPTURE" \
      --cameras $cams \
      --out-dir "$OUT/$name" \
      "${SEG[@]}"
}

run "center"                 center
run "left"                   left
run "right"                  right
run "center left"            center+left
run "center right"           center+right
run "left right"             left+right
run "center left right"      center+left+right

echo
echo "box-level outputs in $OUT/<subset>/"
echo "cloud-level grading:"
echo "  python3 fuse_subset.py --capture-dir $CAPTURE --out $OUT"
echo "  python3 grade_subsets.py --dir $OUT --exclude P1 --csv results.csv"