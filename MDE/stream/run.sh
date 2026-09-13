#!/usr/bin/env bash
# run.sh -- four-camera stream with depth correction and parcel segmentation.
#
# Edit the block below, then: chmod +x run.sh && ./run.sh
#
# Modes, selected with the MODE environment variable. Anything after the
# script name is passed straight through to the underlying tool, so a one-off
# override never needs an edit here.
#
#   MODE=stream   (default) live streaming, capture on p, segment on capture
#   MODE=probe    list the largest planes in the most recent capture with
#                 their distance from the reference camera. This is how DECK
#                 below is read off rather than guessed
#   MODE=align    solve the per-camera depth affine on the most recent
#                 capture, validate it leave-one-plane-out, and write the
#                 residual maps
#   MODE=reseg    re-run segmentation on the most recent capture with the
#                 thresholds below, without touching the cameras
#
#   ./run.sh                          stream
#   MODE=probe ./run.sh               read the deck distance
#   MODE=reseg ./run.sh --seg-views all
#   CAPTURE=runs/live_.../capture_00003 MODE=align ./run.sh
#
# Do not run this under sudo: it redirects ~ to /root and the venv, the
# calibration and the Arena library all disappear.

set -euo pipefail

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

PROJECT=~/Projects/MDE/stream
CALIB=/home/jetson/Projects/Calibration_4_1/results
VENV=~/Projects/MDE/DA3/da3

AFFINE=""          # a depth_affine.json, or empty for auto-align only

# Distance from the REFERENCE CAMERA to the conveyor deck, in metres, and the
# half-width of the band accepted around it. Without this the plane fit returns
# the largest coplanar surface in the cloud, which in a four-camera rig is
# usually the floor. Read the value from MODE=probe; the last measurement on
# this rig was 3.16 m, not 1.00.
DECK=3.16
BAND=0.06

# Fusion. 70 degrees is the incidence beyond which a surface is sampled by
# rays further apart than the depth quantum and reconstructs as a combed
# sheet. 85 rejected almost nothing.
MAX_INCIDENCE=70
CORROBORATE=0.010
MIN_VIEWS=1        # 2 keeps only points a second camera supports

# Segmentation.
# The conveyor does not move, so its footprint is measured once and frozen.
# Delete this file, or run with --seg-belt-refit, after moving the rig or
# recalibrating. Solve it on an EMPTY or lightly loaded belt: whatever is
# standing on it the first time becomes part of the frozen footprint.
BELT=belt.json
MAX_PARCEL_HEIGHT=0.60
SEG_VIEWS=bev      # bev, cameras, or all
SEG_MIN_VIEWS=2    # cameras that must each fit a top face before a pick is issued

# --------------------------------------------------------------------------

MODE="${MODE:-stream}"

cd "$PROJECT"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

for f in da3_stream.py da3_fuse.py online_align.py box_segment.py \
         face_consensus.py layer_align.py; do
    [ -f "$f" ] || { echo "missing $f in $PROJECT" >&2; exit 1; }
done

PMIN=$(awk -v d="$DECK" -v b="$BAND" 'BEGIN{printf "%.4f", d-b}')
PMAX=$(awk -v d="$DECK" -v b="$BAND" 'BEGIN{printf "%.4f", d+b}')

# Thresholds shared by the live run and by MODE=reseg, so tuning offline and
# then streaming does not silently use two different sets of numbers.
SEG_ARGS=(
    --seg-plane-distance "$PMIN" "$PMAX"
    --seg-max-height "$MAX_PARCEL_HEIGHT"
    --seg-views "$SEG_VIEWS"
    --seg-min-views "$SEG_MIN_VIEWS"
    --seg-belt-file "$BELT"
)

latest_capture() {
    if [ -n "${CAPTURE:-}" ]; then
        printf '%s\n' "$CAPTURE"
        return
    fi
    local d
    d=$(ls -d runs/live_*/capture_* 2>/dev/null | sort | tail -n 1 || true)
    [ -n "$d" ] || { echo "no capture found under $PROJECT/runs; stream and press p first, or set CAPTURE=" >&2; exit 1; }
    printf '%s\n' "$d"
}

summarise_log() {
    local log="$1"
    [ -f "$log" ] || return 0
    python - "$log" <<'PY'
import csv, sys
# Short headings: the CSV names are self-documenting but far too wide to
# line up on a terminal, and a table that does not line up is not read.
cols = [("id", "id"), ("order", "pick_order"), ("pick", "pickable"),
        ("len", "length_mm"), ("wid", "width_mm"), ("hgt", "height_mm"),
        ("top", "top_height_mm"), ("views", "n_views"),
        ("spread", "inter_view_offset_mm"), ("unc", "height_uncertainty_mm"),
        ("incid", "worst_incidence_deg"), ("grow", "face_growth_gain")]
width = {h: max(len(h), 6) for h, _ in cols}
with open(sys.argv[1], newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    print("no parcels logged")
    raise SystemExit
print("  ".join(h.rjust(width[h]) for h, _ in cols))
for r in rows[-20:]:
    print("  ".join(str(r.get(k, ""))[:width[h]].rjust(width[h])
                    for h, k in cols))
held = [r for r in rows if r.get("pickable") == "0"]
if held:
    print(f"\n{len(held)} of {len(rows)} parcel(s) on hold. First reasons:")
    for r in held[:5]:
        print(f"  #{r['id']}: {r.get('blocking_reasons', '')[:110]}")
PY
}

# --------------------------------------------------------------------------

case "$MODE" in

probe)
    CAP=$(latest_capture)
    echo "probing $CAP"
    echo "The deck is the large, near-horizontal plane nearest the cameras."
    echo "The floor is roughly 0.84 m beyond it. Put the deck distance in DECK."
    python box_segment.py --capture-dir "$CAP" --probe "$@"
    ;;

align)
    CAP=$(latest_capture)
    echo "solving the per-camera depth affine on $CAP"
    python layer_align.py \
        --capture-dir "$CAP" \
        --mode both \
        --holdout \
        --max-incidence "$MAX_INCIDENCE" \
        --out-dir "$CAP/aligned" \
        "$@"
    echo
    echo "Read the [hold ] block, not the [solve] one: the solve residuals are"
    echo "measured on the planes they were fitted to. Then set"
    echo "AFFINE=$CAP/aligned/depth_affine.json in this script."
    ;;

reseg)
    CAP=$(latest_capture)
    echo "re-segmenting $CAP"
    python box_segment.py --capture-dir "$CAP" "${SEG_ARGS[@]}" "$@"
    ;;

stream)
    OUT=runs/live_$(date +%Y%m%d_%H%M%S)

    CORRECTION=(--auto-align --auto-align-tol 0.08)
    if [ -n "$AFFINE" ] && [ -f "$AFFINE" ]; then
        CORRECTION+=(--load-affine "$AFFINE" --apply-absolute)
        echo "correction: auto-align offsets, scale and anchor from $AFFINE"
    else
        echo "correction: auto-align offsets only, unity scale, no metric anchor."
        echo "            Layering is removed but dimensions are unanchored."
        echo "            Run MODE=align ./run.sh on a capture to produce one."
    fi

    echo "deck band : $PMIN to $PMAX m from the reference camera"
    if [ -f "$BELT" ]; then
        echo "belt      : frozen, from $BELT"
    else
        echo "belt      : not yet measured. The FIRST capture defines it, so"
        echo "            make that one an empty or lightly loaded belt."
    fi
    echo "fusion    : incidence <= ${MAX_INCIDENCE} deg, corroboration at" \
         "$(awk -v v="$CORROBORATE" 'BEGIN{printf "%.0f", v*1000}') mm," \
         "min views $MIN_VIEWS"
    echo

    python da3_stream.py \
        --calib-dir "$CALIB" \
        --out-dir "$OUT" \
        "${CORRECTION[@]}" \
        --max-incidence "$MAX_INCIDENCE" \
        --fuse-corroborate "$CORROBORATE" \
        --fuse-min-views "$MIN_VIEWS" \
        --capture-view-count \
        --segment \
        "${SEG_ARGS[@]}" \
        "$@"

    echo
    echo "output: $OUT"
    summarise_log "$OUT/boxes_log.csv"
    echo
    echo "next:"
    echo "  MODE=align ./run.sh      solve and validate the depth affine"
    echo "  MODE=reseg ./run.sh --seg-views all   retune the thresholds offline"
    echo "  rm $BELT                 re-measure the conveyor footprint"
    ;;

*)
    echo "unknown MODE '$MODE'; expected stream, probe, align or reseg" >&2
    exit 1
    ;;
esac