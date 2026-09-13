#!/usr/bin/env bash
# run.sh -- four-camera stream with depth correction and parcel segmentation.
#
#   MODE=truth    measure the metric reference planes with the ChArUco board.
#                 Do this FIRST after any lens, focus or mount change
#   MODE=anchor   print the anchor and the calibration the rig is running on
#   MODE=probe    list the largest planes in the most recent capture with their
#                 distance from the reference camera, in DA3's unanchored
#                 coordinates. This is how DECK below is read off
#   MODE=align    solve the per-camera depth affine and the metric anchor on
#                 the most recent capture, and validate leave-one-plane-out
#   MODE=list     find every depth_affine.json under runs/ and say which of
#                 them is actually usable. This is how AFFINE below is filled in
#   MODE=reseg    re-run segmentation on the most recent capture
#   MODE=stream   (default) live streaming, capture on p, segment on capture
#
# Do not run this under sudo.

set -euo pipefail

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

PROJECT=~/Projects/MDE/stream_v2
CALIB=/home/jetson/Projects/Calibration_4_5/results
VENV=~/Projects/MDE/DA3/da3
TRUTH=ground_truth.json
CAMS="left center right top"

# ---- the per-camera depth affine -----------------------------------------
# WHAT THIS MAY BE SET TO
#
#   ""                                      no stored coefficients; auto-align
#                                           only, and every camera whose scale
#                                           the live solve refuses falls back to
#                                           a = 1.0
#   path/to/depth_affine.json               the file itself
#   runs/live_.../capture_00002             a capture directory; the aligned/
#                                           subdirectory is found for you
#   runs/live_.../capture_00002/aligned     the output directory
#
# layer_align.py writes to <capture-dir>/aligned/depth_affine.json, NOT to the
# capture directory itself. Pointing AFFINE one level too high is the single
# most common way to get "has no 'coefficients' block" three screens into
# startup, because the old guard tested only that a file existed there. It is
# now parsed and checked before the cameras are touched, and MODE=list will
# tell you which candidates on disk are real.
AFFINE=""

# ---- the metric anchor ---------------------------------------------------
# Three surfaces, of which the RISER is the one that matters. The deck and the
# floor both lie at or below the deck, so a correction solved on that pair
# alone is extrapolated into the volume the robot picks from. The riser sits
# inside the parcel band, which turns the same solve into an interpolation.
#
# THE BOARD THICKNESS IS ADDED, NOT SUBTRACTED. The printed face sits 47 mm
# ABOVE the surface it rests on and the cameras look down, so the surface is
# the printed face PLUS 47 mm. charuco_gt.py subtracted, which put the deck
# 47 mm above the board lying on it and left deck_perp_m 94 mm short. That
# error cancelled in the floor-to-deck separation, which is why it survived.
BOARD_THICKNESS_MM=47
DECK_SNAPS="gt/deck_*.png"
RISER_SNAPS="gt/riser_*.png"
FLOOR_SNAPS=""

# Height of the riser BLOCK's top surface above the deck, taken with a rule.
# Do NOT include the 47 mm here: measure_deck.py removes the board and frame
# from the pose itself, so what it reports is the block top, and this is the
# number it is asserted against. Getting it wrong by exactly 47 or 94 mm is
# the diagnostic the check prints.
RISER_HEIGHT_MM=257
EXPECT_TOL_MM=8

# ---- DA3's own idea of the deck distance ---------------------------------
# THIS IS NOT A PHYSICAL DISTANCE. It selects a band in DA3's UNANCHORED
# coordinates. The measured deck standoff is 3.081 m; the model reads several
# per cent off it and moves when the lens, the resolution or the padding
# changes. The 12 mm lens changed all three, so 3.16 from the 8 mm rig is
# certainly wrong. Run MODE=probe on the first capture and put the deck plane's
# distance column here.
#
# BAND was widened to 0.08 to span layered sheets, which is a symptom being
# accommodated rather than removed. A 160 mm window is wide enough for the
# plane search to lock onto a large parcel top instead of the deck: the last
# capture fitted its "conveyor" at 3.0841 m holding 9.1 per cent of the
# points, below box_segment.py's own 15 per cent floor. Narrow this back
# towards 0.04 once the inter-view spread reported at the end of a stream is
# down to single millimetres.
DECK=3.24
BAND=0.08
DECK_IS_STALE=0

MAX_INCIDENCE=70
CORROBORATE=0.010
MIN_VIEWS=1

# The 12 mm lens narrowed the field at the deck from 3.15 x 2.64 m to
# 2.10 x 1.76 m, so any belt.json carried over describes a footprint the
# cameras can no longer see. Delete it and re-solve on an EMPTY belt. The
# union of four fields plus the 0.86 m left-to-right baseline is about
# 2.96 m, so a footprint reported near 2.97 m long is the CAMERA COVERAGE
# and not the conveyor.
BELT=belt.json

MIN_PARCEL_HEIGHT=0.030
MAX_PARCEL_HEIGHT=0.60

# ---- the alignment bands -------------------------------------------------
# AUTO_ALIGN_TOL is squeezed from both sides and neither bound is optional.
#
#   from BELOW  it must EXCEED the inter-camera disagreement at the reference
#               surface, or a biased camera's samples never reach the plane
#               they belong to and that camera is silently left uncorrected.
#
#   from ABOVE  each plane claims points within AUTO_ALIGN_TOL of itself in
#               BOTH directions, so two such bands must fit inside the gap:
#                   AUTO_ALIGN_TOL < AUTO_ALIGN_UPPER_MIN / 2
#
# Both are satisfiable only when the upper reference stands more than TWICE
# the disagreement above the deck.
#
# MEASURED, AND CURRENTLY IN CONFLICT. The last capture put three views of one
# parcel top at 227.7, 197.7 and 264.9 mm, a spread of 67 mm, with a fourth at
# 137 mm dropped as a different surface. 50 mm does NOT exceed that, so the
# worst camera is outside the assignment band and is not being corrected at
# all. Raising TOL to 0.080 requires UPPER_MIN above 0.160, and to 0.100
# requires UPPER_MIN above 0.200 -- which in turn means no parcel shorter than
# 200 mm can serve as the scale reference. Decide that deliberately; the
# summary at the end of a stream re-measures the spread every run.
#
# These match layer_align.py's --assign-tol and --plane-min-gap, which
# MODE=align passes through explicitly, so the offline solve and the live
# aligner operate on the same population of samples.
AUTO_ALIGN_TOL=0.050
AUTO_ALIGN_UPPER_MIN=0.120
AUTO_ALIGN_UPPER_MAX=0.600

SEG_VIEWS=bev
SEG_MIN_VIEWS=2

# --------------------------------------------------------------------------

MODE="${MODE:-stream}"

cd "$PROJECT"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

for f in da3_stream.py da3_fuse.py online_align.py box_segment.py \
         face_consensus.py layer_align.py measure_deck.py; do
    [ -f "$f" ] || { echo "missing $f in $PROJECT" >&2; exit 1; }
done
[ -d "$CALIB" ] || { echo "calibration directory not found: $CALIB" >&2; exit 1; }

PMIN=$(awk -v d="$DECK" -v b="$BAND" 'BEGIN{printf "%.4f", d-b}')
PMAX=$(awk -v d="$DECK" -v b="$BAND" 'BEGIN{printf "%.4f", d+b}')

# The alignment bands are a STREAMING concern. Checking them before the case
# blocked MODE=truth, which is a ChArUco measurement that never reads them.
check_bands() {
    awk -v t="$AUTO_ALIGN_TOL" -v u="$AUTO_ALIGN_UPPER_MIN" \
        'BEGIN{exit !(t < u/2)}' || {
        echo "AUTO_ALIGN_TOL ($AUTO_ALIGN_TOL) must stay below HALF of" \
             "AUTO_ALIGN_UPPER_MIN ($AUTO_ALIGN_UPPER_MIN). Each plane claims" \
             "points within AUTO_ALIGN_TOL of itself in BOTH directions, so" \
             "two bands of that width have to fit inside the gap between the" \
             "deck and the upper reference. AUTO_ALIGN_TOL must also EXCEED" \
             "the inter-camera disagreement at the parcel tops, so raise" \
             "AUTO_ALIGN_UPPER_MIN rather than lowering the tolerance below" \
             "that disagreement." >&2
        exit 1
    }
    awk -v u="$AUTO_ALIGN_UPPER_MIN" -v x="$AUTO_ALIGN_UPPER_MAX" \
        'BEGIN{exit !(u < x)}' || {
        echo "AUTO_ALIGN_UPPER_MIN must stay below AUTO_ALIGN_UPPER_MAX." >&2
        exit 1
    }
    # This one is a WARNING, not an error, and the previous version had the
    # sense of it backwards: it enforced the overlap while describing the
    # overlap as the harm. Parcels between MIN_PARCEL_HEIGHT and
    # AUTO_ALIGN_TOL are counted as conveyor by the aligner and as parcels by
    # the segmentation at the same time, so their points pull the deck fit.
    # It cannot be configured away on this rig: raising MIN_PARCEL_HEIGHT to
    # AUTO_ALIGN_TOL loses every short parcel, and lowering AUTO_ALIGN_TOL
    # below the measured disagreement leaves the worst camera uncorrected.
    awk -v t="$AUTO_ALIGN_TOL" -v m="$MIN_PARCEL_HEIGHT" \
        'BEGIN{exit !(m < t)}' && {
        echo "[note ] parcels between" \
             "$(awk -v v="$MIN_PARCEL_HEIGHT" 'BEGIN{printf "%.0f", v*1000}')" \
             "and" \
             "$(awk -v v="$AUTO_ALIGN_TOL" 'BEGIN{printf "%.0f", v*1000}')" \
             "mm are inside the aligner's deck band AND above the"
        echo "        segmentation's floor, so they are both conveyor and" \
             "parcel. Their points bias"
        echo "        the deck plane the whole correction is referred to."
    } || true
}

SEG_ARGS=(
    --seg-plane-distance "$PMIN" "$PMAX"
    --seg-min-height "$MIN_PARCEL_HEIGHT"
    --seg-max-height "$MAX_PARCEL_HEIGHT"
    --seg-views "$SEG_VIEWS"
    --seg-min-views "$SEG_MIN_VIEWS"
    --seg-belt-file "$BELT"
)

latest_capture() {
    if [ -n "${CAPTURE:-}" ]; then printf '%s\n' "$CAPTURE"; return; fi
    local d
    d=$(ls -d runs/live_*/capture_* 2>/dev/null | sort | tail -n 1 || true)
    [ -n "$d" ] || { echo "no capture found; stream and press p first, or set CAPTURE=" >&2; exit 1; }
    printf '%s\n' "$d"
}

# --------------------------------------------------------------------------
# the depth affine
# --------------------------------------------------------------------------

# Resolve whatever AFFINE was set to into an actual file, or print nothing.
# layer_align.py writes <capture-dir>/aligned/depth_affine.json, so a capture
# directory, the aligned directory and the file itself all have to work, and a
# path that is missing the aligned/ component has to be repaired rather than
# reported three screens later.
resolve_affine() {
    local p="$1" cand
    [ -n "$p" ] || return 0
    if [ -d "$p" ]; then
        for cand in "$p/aligned/depth_affine.json" "$p/depth_affine.json"; do
            [ -f "$cand" ] && { printf '%s\n' "$cand"; return 0; }
        done
        return 0
    fi
    if [ -f "$p" ]; then printf '%s\n' "$p"; return 0; fi
    cand="$(dirname "$p")/aligned/$(basename "$p")"
    [ -f "$cand" ] && { printf '%s\n' "$cand"; return 0; }
    return 0
}

# Parse a depth_affine.json and report what is in it.
#   exit 0  usable coefficients AND a solved metric anchor
#   exit 2  usable coefficients, but the anchor is refused or absent, so
#           --apply-absolute must be withheld
#   exit 1  not a layer_align.py solve output at all
inspect_affine() {
    # shellcheck disable=SC2086
    python - "$1" $CAMS <<'PY'
import json
import sys
from pathlib import Path

p, names = Path(sys.argv[1]), sys.argv[2:]

try:
    d = json.loads(p.read_text())
except ValueError as exc:
    print(f"  ** {p} is not valid JSON: {exc} **")
    sys.exit(1)
if not isinstance(d, dict):
    print(f"  ** {p} is a {type(d).__name__}, not an object **")
    sys.exit(1)

coeffs = d.get("coefficients")
if not isinstance(coeffs, dict):
    print(f"  ** {p}")
    print( "     has no 'coefficients' block, so it is NOT a layer_align.py")
    print( "     solve output. Top-level keys present:")
    print( "       " + ", ".join(sorted(map(str, d))[:14]))
    for label, keys in (
            ("a per-capture record (capture.json)",
             {"per_camera", "fused", "segmentation", "corroboration"}),
            ("a layer_align report (layer_report.json)",
             {"layering_before", "per_camera_plane_fit", "deck_plane_index"}),
            ("a segmentation record (boxes.json)", {"boxes", "conveyor"}),
            ("a ChArUco measurement (ground_truth.json)",
             {"reference_planes", "deck_perp_m"})):
        if keys & set(d):
            print(f"     It looks like {label}.")
            break
    sib = p.parent / "aligned" / p.name
    if sib.exists():
        print(f"     The file you want is probably:")
        print(f"       {sib}")
    else:
        print( "     layer_align.py writes to <capture-dir>/aligned/. Run")
        print( "       MODE=list ./run.sh")
        print( "     to see every candidate on disk.")
    sys.exit(1)

missing = [n for n in names if n not in coeffs]
if missing:
    print(f"  ** {p} lacks coefficients for: {', '.join(missing)}.")
    print( "     It was solved over a different camera set. Re-run MODE=align")
    print( "     with all four cameras present in the capture. **")
    sys.exit(1)

ref = d.get("reference")
print(f"  affine     : {p}")
print(f"               solved on {d.get('source_capture') or 'an unrecorded capture'}")
span = d.get("plane_span_m")
in_band = d.get("planes_in_parcel_band")
if span is not None:
    print(f"               plane span {float(span) * 1e3:.0f} mm, "
          f"{in_band} inside the parcel band")

unphysical = []
for n in names:
    a, b = float(coeffs[n]["a"]), float(coeffs[n]["b"])
    tag = "   <- reference" if n == ref else ""
    print(f"               {n:<8} a={a:.6f}  b={b * 1e3:+8.2f} mm{tag}")
    if n != ref and abs(a - 1.0) > 0.03:
        unphysical.append(n)

if ref in coeffs:
    a_r, b_r = float(coeffs[ref]["a"]), float(coeffs[ref]["b"])
    if abs(a_r - 1.0) > 1e-9 or abs(b_r) > 1e-9:
        print(f"  ** the reference camera {ref!r} carries a non-identity")
        print( "     correction, which means the coefficients are not relative")
        print( "     to it and every other camera's number is shifted. **")

if unphysical:
    print(f"  ** {', '.join(unphysical)} carry a depth scale more than 3 per")
    print( "     cent from unity, while the fx spread on this rig justifies")
    print( "     0.8 per cent. Check the per-plane sample counts in the")
    print( "     layer_report.json beside this file: where one plane supplied")
    print( "     under a tenth of the samples, 'a' is fitted on the other")
    print( "     plane alone, is degenerate with 'b', and injects error at")
    print( "     the parcel tops rather than removing it. **")

absolute = d.get("absolute")
if not isinstance(absolute, dict):
    print( "  anchor     : ABSENT from this file, so there is no metric scale")
    print( "               here. --apply-absolute withheld.")
    sys.exit(2)
if absolute.get("ok") is not True:
    print( "  anchor     : REFUSED by layer_align.py")
    reason = str(absolute.get("reason") or "no reason recorded")
    for i in range(0, len(reason), 62):
        print(f"               {reason[i:i + 62]}")
    diag = absolute.get("diagnostic")
    if diag:
        print(f"               diagnostic: {json.dumps(diag, default=str)[:300]}")
    print( "               --apply-absolute is withheld. The coefficients still")
    print( "               remove the layering; the heights stay unanchored.")
    print( "               Do NOT raise --anchor-tol to force this through.")
    sys.exit(2)

scale = absolute.get("scale_estimate")
shift = absolute.get("shift_estimate_m")
try:
    shift_mm = f"{float(shift) * 1e3:+.1f} mm"
except (TypeError, ValueError):
    shift_mm = str(shift)
print(f"  anchor     : solved, scale {scale}, shift {shift_mm}, "
      f"{absolute.get('n_planes_matched')} plane(s) matched")
print(f"               fit rms {absolute.get('fit_rms_mm')} mm, worst "
      f"{absolute.get('worst_residual_mm')} mm")
sys.exit(0)
PY
}

check_anchor() {
    python - "$CALIB" "$TRUTH" <<'PY'
import json, sys
from pathlib import Path

calib, truth = Path(sys.argv[1]), Path(sys.argv[2])
lenses, fxs = {}, {}
for p in sorted(calib.glob("intrinsics_*.json")):
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        continue
    name = d.get("camera", p.stem)
    fx = d.get("camera_matrix", [[0]])[0][0]
    lenses[name] = (d.get("lens_id"), d.get("lens_mm"), fx)
    fxs[name] = fx

if not lenses:
    print(f"  intrinsics : NONE FOUND in {calib}")
else:
    for name, (lid, mm, fx) in lenses.items():
        print(f"  {name:<10} lens {str(lid):<16} {mm} mm   fx {fx:8.1f} px")
    if len({v[0] for v in lenses.values()}) > 1:
        print("  ** the cameras were calibrated through DIFFERENT lenses **")
    mean = sum(fxs.values()) / len(fxs)
    spread = (max(fxs.values()) - min(fxs.values())) / mean
    print(f"  fx spread  : {spread*100:.2f}% "
          f"({spread*3.0*1e3:.0f} mm of depth at 3 m)")
    if spread > 0.004:
        print("               Only the per-view SCALE term removes this, and")
        print("               that term needs TWO reference surfaces.")

ext = calib / "extrinsics.json"
if ext.exists():
    e = json.loads(ext.read_text())
    solved = [k for k, v in e.items() if isinstance(v, dict) and "R" in v]
    print(f"  extrinsics : {len(solved)} solved: {', '.join(sorted(solved))}")
else:
    print("  extrinsics : MISSING")

if not truth.exists():
    print(f"  anchor     : MISSING ({truth}). Run MODE=truth ./run.sh")
    sys.exit(0)

t = json.loads(truth.read_text())
planes = t.get("reference_planes") or []
if planes:
    print(f"  anchor     : {len(planes)} plane(s), lens {t.get('lens_id')}, "
          f"measured {t.get('measured')}")
    for p in planes:
        vs = ("" if "expected_error_mm" not in p
              else f"   vs rule {p['expected_error_mm']:+.1f} mm")
        print(f"               {p.get('label','?'):<12} "
              f"{p.get('perp_m'):.4f} m   "
              f"{p.get('height_above_deck_mm', 0):+8.1f} mm above deck   "
              f"spread {p.get('spread_mm')} mm{vs}")
    band = t.get("planes_in_parcel_band") or []
    print(f"               span {t.get('span_m', 0)*1e3:.0f} mm; in the parcel "
          f"band: {', '.join(band) if band else 'NONE'}")
    if not t.get("usable", True):
        for w in t.get("problems", []):
            print(f"  ** {w} **")
    if not band:
        print("  ** no reference plane inside the parcel band, so every")
        print("     correction solved from this anchor is EXTRAPOLATED into")
        print("     the volume the robot picks from. Measure a riser. **")
else:
    print(f"  anchor     : LEGACY deck-plus-floor form "
          f"(deck={t.get('deck_perp_m')}, separation={t.get('separation_m')})")
    print("  ** re-measure with a riser: MODE=truth ./run.sh **")
print( "  note       : perp_m values are PHYSICAL. DECK in run.sh is in DA3's")
print( "               unanchored coordinates. Read it from MODE=probe.")
PY
}

warn_stale_deck() {
    [ "$DECK_IS_STALE" = "1" ] || return 0
    echo
    echo "** DECK=$DECK is flagged stale. It is the value from the 8 mm rig."
    echo "   fx moved from 2319 to 3479 px and the framing changed completely,"
    echo "   so DA3's idea of the deck distance has moved with it. Run"
    echo "   MODE=probe ./run.sh, put the deck plane's distance here, and set"
    echo "   DECK_IS_STALE=0. Until then --seg-plane-distance may be selecting"
    echo "   the wrong surface entirely. **"
    echo
}

summarise_log() {
    local log="$1"
    [ -f "$log" ] || return 0
    python - "$log" "$AUTO_ALIGN_TOL" <<'PY'
import csv, sys
tol_mm = float(sys.argv[2]) * 1e3
cols = [("id", "id"), ("order", "pick_order"), ("pick", "pickable"),
        ("len", "length_mm"), ("wid", "width_mm"), ("hgt", "height_mm"),
        ("top", "top_height_mm"), ("views", "n_views"),
        ("spread", "inter_view_offset_mm"), ("unc", "height_uncertainty_mm"),
        ("incid", "worst_incidence_deg"), ("grow", "face_growth_gain")]
width = {h: max(len(h), 6) for h, _ in cols}
with open(sys.argv[1], newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    print("no parcels logged"); raise SystemExit
print("  ".join(h.rjust(width[h]) for h, _ in cols))
for r in rows[-20:]:
    print("  ".join(str(r.get(k, ""))[:width[h]].rjust(width[h])
                    for h, k in cols))

spreads = [float(r["inter_view_offset_mm"]) for r in rows
           if r.get("inter_view_offset_mm") not in (None, "")]
if spreads:
    med = sorted(spreads)[len(spreads)//2]
    print(f"\ninter-view spread on parcel tops: median {med:.1f} mm, "
          f"max {max(spreads):.1f} mm over {len(spreads)} multi-view face(s)")
    print("This is the number that decides whether the box layer is flat.")
    print("If it is still tens of millimetres, check that auto-align reported")
    print("model=two_plane rather than deck_only: deck_only happens when the")
    print("belt held nothing tall enough to be the second reference.")
    if max(spreads) > tol_mm:
        print(f"\n** the worst spread is {max(spreads):.0f} mm, at or above")
        print(f"   AUTO_ALIGN_TOL ({tol_mm:.0f} mm). Samples from the worst")
        print( "   camera are falling outside the assignment band, so that")
        print( "   camera is not being corrected at all. Raise AUTO_ALIGN_TOL")
        print( "   and raise AUTO_ALIGN_UPPER_MIN above twice the new value. **")

low = [r for r in rows if r.get("top_height_mm")
       and float(r["top_height_mm"]) < 75.0]
if low:
    print(f"\n{len(low)} parcel(s) with a top face below 75 mm. At that height")
    print("the inter-camera disagreement is comparable to the height itself,")
    print("so whether they are found at all depends on which camera saw them.")

held = [r for r in rows if r.get("pickable") == "0"]
if held:
    print(f"\n{len(held)} of {len(rows)} parcel(s) on hold. First reasons:")
    for r in held[:5]:
        print(f"  #{r['id']}: {r.get('blocking_reasons', '')[:110]}")
PY
}

# --------------------------------------------------------------------------

case "$MODE" in

truth)
    echo "Measuring the metric reference planes with the ChArUco board."
    echo "Board plus frame: ${BOARD_THICKNESS_MM} mm, ADDED along the normal."
    echo "Riser expected at ${RISER_HEIGHT_MM} mm above the deck, tolerance" \
         "${EXPECT_TOL_MM} mm."
    echo
    ARGS=(--calib-dir "$CALIB" --camera center
          --board-thickness-mm "$BOARD_THICKNESS_MM"
          --parcel-band "$MIN_PARCEL_HEIGHT" "$MAX_PARCEL_HEIGHT"
          --expect-tol-mm "$EXPECT_TOL_MM"
          --write "$TRUTH")
    # shellcheck disable=SC2086
    ARGS+=(--plane deck $DECK_SNAPS)
    if [ -n "$RISER_SNAPS" ] && compgen -G "$RISER_SNAPS" > /dev/null; then
        # shellcheck disable=SC2086
        ARGS+=(--plane riser $RISER_SNAPS --expect riser "$RISER_HEIGHT_MM")
    else
        echo "** no riser images match '$RISER_SNAPS'. Without a surface"
        echo "   inside the parcel band the anchor will be marked unusable. **"
        echo
    fi
    if [ -n "$FLOOR_SNAPS" ] && compgen -G "$FLOOR_SNAPS" > /dev/null; then
        # shellcheck disable=SC2086
        ARGS+=(--plane floor $FLOOR_SNAPS)
    fi
    python measure_deck.py "${ARGS[@]}" "$@"
    echo
    echo "The [check] line on the riser is the one that matters. It exercises"
    echo "the intrinsics, the board geometry, the marker layout, the thickness"
    echo "sign and the normal projection in a single number."
    ;;

anchor)
    echo "calibration: $CALIB"
    echo "anchor     : $PROJECT/$TRUTH"
    echo
    check_anchor
    if [ -n "$AFFINE" ]; then
        AFFINE_PATH="$(resolve_affine "$AFFINE")"
        echo
        if [ -z "$AFFINE_PATH" ]; then
            echo "  affine     : AFFINE is set to '$AFFINE' but no"
            echo "               depth_affine.json was found there."
        else
            set +e; inspect_affine "$AFFINE_PATH"; set -e
        fi
    else
        echo
        echo "  affine     : none set. Auto-align only; any camera whose scale"
        echo "               the live solve refuses keeps a = 1.0."
    fi
    warn_stale_deck
    ;;

list)
    echo "depth_affine.json candidates under runs/, newest first:"
    echo
    found=0
    while IFS= read -r f; do
        found=1
        set +e
        inspect_affine "$f"
        status=$?
        set -e
        case "$status" in
            0) echo "  -> USABLE with the metric anchor." ;;
            2) echo "  -> usable coefficients, no anchor." ;;
            *) echo "  -> NOT USABLE." ;;
        esac
        echo
    done < <(find runs -name depth_affine.json -type f -printf '%T@\t%p\n' \
                 2>/dev/null | sort -rn | cut -f2-)
    if [ "$found" = "0" ]; then
        echo "  none found. MODE=align ./run.sh writes one to"
        echo "  <capture-dir>/aligned/depth_affine.json, and it needs a capture"
        echo "  with the ${RISER_HEIGHT_MM} mm riser block standing on the belt,"
        echo "  board REMOVED."
    else
        echo "Copy the path of a usable one into AFFINE at the top of run.sh."
    fi
    ;;

probe)
    CAP=$(latest_capture)
    echo "probing $CAP"
    echo "The deck is the large, near-horizontal plane nearest the cameras."
    echo "Put its DISTANCE COLUMN into DECK in this script and set"
    echo "DECK_IS_STALE=0. It will NOT equal the 3.081 m physical standoff:"
    echo "this probe reports DA3's unanchored coordinates, and the gap between"
    echo "the two is the scale error the anchor exists to remove."
    python box_segment.py --capture-dir "$CAP" --probe "$@"
    ;;

align)
    check_bands
    CAP=$(latest_capture)
    echo "solving the depth affine and the metric anchor on $CAP"
    echo
    echo "THE RISER BLOCK MUST BE IN THIS CAPTURE, board REMOVED, standing"
    echo "where it was measured. layer_align.py anchors by matching the planes"
    echo "it extracts from the cloud onto the planes in ground_truth.json, and"
    echo "it can only match a surface present in both. The deck always is; the"
    echo "${RISER_HEIGHT_MM} mm riser is only there if you left the block on"
    echo "the belt. Arbitrary parcels at arbitrary heights match nothing, and"
    echo "the anchor is then refused with one plane matched."
    echo
    python layer_align.py \
        --capture-dir "$CAP" \
        --mode both \
        --holdout \
        --assign-tol "$AUTO_ALIGN_TOL" \
        --plane-min-gap "$AUTO_ALIGN_UPPER_MIN" \
        --parcel-band "$MIN_PARCEL_HEIGHT" "$MAX_PARCEL_HEIGHT" \
        --max-incidence "$MAX_INCIDENCE" \
        --out-dir "$CAP/aligned" \
        "$@"
    echo
    echo "Read in this order:"
    echo "  1. [scale]. If the anchor was REFUSED, stop and fix that first."
    echo "     Do not raise --anchor-tol to force it through."
    echo "  2. [hold ], not [solve]: the solve residuals are measured on the"
    echo "     planes they were fitted to and are small by construction."
    echo "  3. [layer] after correction, the spread on the UPPER plane. That"
    echo "     is the number that decides whether the box layer is flat."
    echo

    OUT_AFFINE="$CAP/aligned/depth_affine.json"
    if [ ! -f "$OUT_AFFINE" ]; then
        echo "** no depth_affine.json was written. layer_align.py only writes"
        echo "   one when the solve produced coefficients, so it came back"
        echo "   empty: fewer than two planes were extracted, or a plane fell"
        echo "   below --min-plane-points (500) or --min-samples (2000)."
        echo "   Read $CAP/aligned/layer_report.json, specifically 'planes',"
        echo "   'planes_rejected' and 'planes_in_parcel_band'. If the riser"
        echo "   was not in this capture, that is the whole answer. **"
        exit 1
    fi
    set +e
    inspect_affine "$OUT_AFFINE"
    AFF_STATUS=$?
    set -e
    echo
    if [ "$AFF_STATUS" = "1" ]; then
        echo "** the file just written does not validate, which should not"
        echo "   happen. Do not point AFFINE at it. **"
        exit 1
    fi
    echo "Set this at the top of run.sh:"
    echo "  AFFINE=$OUT_AFFINE"
    echo
    echo "The full path to the file, or to '$CAP', or to '$CAP/aligned' all"
    echo "work; the aligned/ component is resolved for you either way."
    if [ "$AFF_STATUS" = "2" ]; then
        echo
        echo "The anchor was not solved, so --apply-absolute will be withheld"
        echo "on the next stream. The layering still gets removed; the absolute"
        echo "heights stay floating until a capture with the riser anchors them."
    fi
    ;;

reseg)
    check_bands
    CAP=$(latest_capture)
    warn_stale_deck
    echo "re-segmenting $CAP  (parcels above ${MIN_PARCEL_HEIGHT} m)"
    python box_segment.py --capture-dir "$CAP" "${SEG_ARGS[@]}" "$@"
    ;;

stream)
    check_bands
    OUT=runs/live_$(date +%Y%m%d_%H%M%S)

    # ---- validate the affine BEFORE anything else ----------------------
    # The old guard was [ -f "$AFFINE" ], which passes for any file at all.
    # A capture.json renamed to depth_affine.json, or a path missing the
    # aligned/ component, got through the whole startup banner and then died
    # inside da3_stream.load_affine. It is parsed here instead.
    AFFINE_PATH=""
    AFF_STATUS=0
    if [ -n "$AFFINE" ]; then
        AFFINE_PATH="$(resolve_affine "$AFFINE")"
        if [ -z "$AFFINE_PATH" ]; then
            echo "AFFINE is set to '$AFFINE' but no depth_affine.json was" >&2
            echo "found there. MODE=align writes it to" >&2
            echo "  <capture-dir>/aligned/depth_affine.json" >&2
            echo "Run 'MODE=list ./run.sh' to see the candidates, or clear" >&2
            echo "AFFINE to stream on auto-align alone." >&2
            exit 1
        fi
        set +e
        inspect_affine "$AFFINE_PATH"
        AFF_STATUS=$?
        set -e
        if [ "$AFF_STATUS" = "1" ]; then
            echo >&2
            echo "AFFINE does not point at a layer_align.py solve output." >&2
            echo "Fix it, or clear AFFINE to stream on auto-align alone." >&2
            exit 1
        fi
        echo
    fi

    check_anchor
    warn_stale_deck

    CORRECTION=(--auto-align
                --auto-align-tol "$AUTO_ALIGN_TOL"
                --auto-align-upper-band "$AUTO_ALIGN_UPPER_MIN" \
                                        "$AUTO_ALIGN_UPPER_MAX")
    echo
    if [ -n "$AFFINE_PATH" ]; then
        if [ "$AFF_STATUS" = "0" ] && [ -f "$TRUTH" ]; then
            # Gated on the file's OWN absolute.ok, not merely on
            # ground_truth.json existing. da3_stream.load_affine hard-exits on
            # a refused anchor, so passing --apply-absolute against one is a
            # guaranteed startup failure rather than a degraded run.
            CORRECTION+=(--load-affine "$AFFINE_PATH" --apply-absolute)
            echo "correction: two-plane auto-align, scale and metric anchor"
            echo "            from $AFFINE_PATH"
        elif [ "$AFF_STATUS" = "0" ]; then
            CORRECTION+=(--load-affine "$AFFINE_PATH")
            echo "correction: two-plane auto-align, scale from the affine."
            echo "            --apply-absolute withheld: no $TRUTH."
        else
            CORRECTION+=(--load-affine "$AFFINE_PATH")
            echo "correction: two-plane auto-align, scale from the affine."
            echo "            --apply-absolute withheld: the anchor in that"
            echo "            file is refused or absent."
        fi
    else
        echo "correction: two-plane auto-align only, no metric anchor."
        echo "            Layering is removed and the box layer should be flat,"
        echo "            but the absolute heights are unanchored, and any"
        echo "            camera whose scale the live solve refuses keeps"
        echo "            a = 1.0 with nothing to fall back on. Run"
        echo "            MODE=align ./run.sh on a capture with the riser."
    fi

    echo "deck band : $PMIN to $PMAX m in DA3 coordinates (physical 3.081 m)"
    echo "align     : deck plus the dominant surface between" \
         "$(awk -v v="$AUTO_ALIGN_UPPER_MIN" 'BEGIN{printf "%.0f", v*1000}')" \
         "and $(awk -v v="$AUTO_ALIGN_UPPER_MAX" 'BEGIN{printf "%.0f", v*1000}') mm above it," \
         "band $(awk -v v="$AUTO_ALIGN_TOL" 'BEGIN{printf "%.0f", v*1000}') mm"
    if [ -f "$BELT" ]; then
        echo "belt      : frozen, from $BELT"
    else
        echo "belt      : not yet measured. The FIRST capture defines it, so"
        echo "            make that one an EMPTY belt."
    fi
    echo "parcels   : between $(awk -v v="$MIN_PARCEL_HEIGHT" 'BEGIN{printf "%.0f", v*1000}')" \
         "and $(awk -v v="$MAX_PARCEL_HEIGHT" 'BEGIN{printf "%.0f", v*1000}') mm above the deck," \
         "$SEG_MIN_VIEWS view(s) required to pick"
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
    echo "  MODE=probe ./run.sh      read DECK in DA3 coordinates"
    echo "  MODE=align ./run.sh      solve the affine and the metric anchor"
    echo "  MODE=list  ./run.sh      find a usable depth_affine.json"
    echo "  MODE=reseg ./run.sh --seg-views all   retune offline"
    echo "  rm $BELT                 re-measure the conveyor footprint"
    ;;

*)
    echo "unknown MODE '$MODE'; expected truth, anchor, probe, align, list," \
         "reseg or stream" >&2
    exit 1
    ;;
esac