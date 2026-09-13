#!/usr/bin/env bash
# run.sh -- four-camera stream with depth correction and parcel segmentation.
#
#   MODE=truth    measure the metric reference planes with the ChArUco board.
#                 Do this FIRST after any lens, focus or mount change
#   MODE=anchor   print the anchor and the calibration the rig is running on
#   MODE=probe    list the largest planes in the most recent capture with their
#                 distance from the reference camera, in DA3's unanchored
#                 coordinates. This is how DECK below is read off
#   MODE=audit    per camera per plane, why a reference plane produced no
#                 samples: not in the frustum, removed by the masks, or outside
#                 the assignment window. RUN THIS BEFORE MODE=field
#   MODE=field    solve the four-DOF per-camera depth field against every
#                 measured plane, and validate on a held-out plane. This
#                 SUPERSEDES MODE=align
#   MODE=align    legacy: layer_align.py's two-stage affine plus metric anchor.
#                 Kept for comparison; it has no tilt term
#   MODE=reseg    re-run segmentation on the most recent capture
#   MODE=stream   (default) live streaming, capture on p, segment on capture
#
# Do not run this under sudo.

set -euo pipefail

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

PROJECT=~/Projects/MDE/stream_v3
CALIB=/home/jetson/Projects/Calibration_4_5/results
VENV=~/Projects/MDE/DA3/da3
TRUTH=ground_truth.json

# ONE of these, not both. FIELD is the four-DOF field from MODE=field and is
# what you want; AFFINE is layer_align.py's two-coefficient form, kept only so
# the two can be compared on the same capture. da3_stream.py refuses both.
FIELD=runs/live_20260818_101941/capture_00002/field/depth_field.json           # e.g. runs/live_.../capture_00001/field/depth_field.json
AFFINE=""          # legacy depth_affine.json

# ---- the metric anchor ---------------------------------------------------
# Three surfaces, of which the RISER is the one that matters. The deck and the
# floor both lie at or below the deck, so a correction solved on that pair
# alone is extrapolated into the volume the robot picks from. The riser sits
# inside the parcel band, which turns the same solve into an interpolation.
#
# THE BOARD THICKNESS IS ADDED, NOT SUBTRACTED. The printed face sits 47 mm
# ABOVE the surface it rests on and the cameras look down, so the surface is
# the printed face PLUS 47 mm.
BOARD_THICKNESS_MM=47
DECK_SNAPS="gt/deck_*.png"
RISER_SNAPS="gt/riser_*.png"
FLOOR_SNAPS="gt/floor_*.png"

RISER_HEIGHT_MM=257
EXPECT_TOL_MM=8

# ---- DA3's own idea of the deck distance ---------------------------------
# THIS IS NOT A PHYSICAL DISTANCE. It selects a band in DA3's UNANCHORED
# coordinates. Run MODE=probe on the first capture and put the deck plane's
# distance column here.
DECK=3.14
BAND=0.08
DECK_IS_STALE=0

MAX_INCIDENCE=70

# Corroboration must EXCEED the residual spatial structure, or a point seen by
# two cameras never agrees with itself and every multi-view sample is thrown
# away. Measured after the field on capture_00001: spatial_structure_mm was
# 15.7, 17.2, 17.5, 16.4 across the four cameras. That is the p05-p95 range of
# the per-cell median residual and it is what NO smooth radial field can reach,
# so 10 mm is below the floor. Raise it to sit just above the measured
# structure, or accept that --fuse-min-views 2 will find almost nothing.
CORROBORATE=0.020
MIN_VIEWS=1

BELT=belt.json

MIN_PARCEL_HEIGHT=0.030
MAX_PARCEL_HEIGHT=0.60

# ---- MODE=field: the four-DOF solve -------------------------------------
# FIELD_ASSIGN_TOL is stage one's STARTING window, on the bootstrap plane
# alone. It MUST EXCEED THE ERROR BEING CORRECTED, measured against the
# physical plane. Measured on capture_00001 the per-camera deck offsets are
# left +177, top +74, center +67, right +28 mm, so a 100 mm window found the
# deck for three cameras and found PARCEL TOPS for left: left reads 177 mm long,
# so the surfaces landing near the deck plane in its depth map are boxes 78 to
# 278 mm tall. Its "deck" fit was solved on the boxes, and that camera was the
# extra sheet in the fused cloud. 250 mm covers the measured spread; a single
# plane cannot be confused with another one however wide the window, and the
# deck outnumbers the parcel-top contamination roughly eight to one, so the
# narrowing iterations shed it.
FIELD_ASSIGN_TOL=0.250

# Stage two's STARTING window. MUST STAY BELOW HALF THE GAP BETWEEN THE
# CLOSEST PAIR OF REFERENCE PLANES or a point near one is claimed by both.
# Measured: planes at 0, 130 and 218 mm above the deck, closest pair 87 mm,
# ceiling 43 mm.
FIELD_ASSIGN_TOL_STAGE2=0.025
FIELD_ASSIGN_TOL_FINAL=0.015
FIELD_ITERS=6
FIELD_BOOTSTRAP=deck

# FIX THE SCALE. Across all four cameras and all three planes the residual is
# constant in height to within 1 to 4 mm over the 218 mm anchor span (left
# 178.2/177.0/176.3, right 27.2/28.1/27.2, center 69.1/68.0/65.3). That bounds
# |s-1| below about 1.8 per cent, which a 218 mm lever arm carrying 6 to 7 mm of
# noise cannot resolve: the offline fit reports the scale and the offset as
# 99.98 to 99.99 per cent correlated and says not to quote either alone.
#
# Left free, the pair walks along that null direction. Measured on `top` with
# the guards opened to 10 per cent: c marched -101, -181, -235, -264, -281,
# -292 mm and s marched 1.007 to 1.058 over six iterations, never converging,
# while every anchor residual stayed under 7 mm because the walk stays inside
# the span. Extrapolated 400 mm up to a parcel top that field is 25 mm wrong.
# Fixing s = 1 removes the direction entirely and the correction is then a
# constant offset plus tilt, which is what was measured.
FIELD_FIX_SCALE=1
FIELD_MAX_SCALE_DEV=0.02      # a tripwire now, not a working bound
FIELD_MAX_GAIN_SPAN=0.06
FIELD_MAX_OFFSET=0.30         # must exceed the largest measured offset, 177 mm

# The height at which the extrapolation is checked. The anchor spans 218 mm and
# the parcels run to MAX_PARCEL_HEIGHT, so the correction is EXTRAPOLATED over
# the difference and that is where a scale error shows up.
FIELD_CHECK_HEIGHT=0.40

# ---- the alignment bands -------------------------------------------------
# With a FIELD loaded the aligner runs OFFSET-ONLY: the field owns the scale and
# the tilt, and re-solving scale per frame from the deck-to-parcel lever arm
# reproduces the same degeneracy that walked `top` to s=1.058 offline, once per
# frame and with no holdout to catch it.
#
# AUTO_ALIGN_TOL is still squeezed from both sides:
#   from BELOW  it must EXCEED the residual inter-camera disagreement AFTER the
#               field, or a camera's samples never reach the plane they belong
#               to. Measured after the field: deck spread 1.2 mm, riser1 8.4 mm,
#               riser 1.7 mm, so the residual is under 10 mm and 50 mm is now
#               generous rather than marginal
#   from ABOVE  AUTO_ALIGN_TOL < AUTO_ALIGN_UPPER_MIN / 2
AUTO_ALIGN_TOL=0.050
AUTO_ALIGN_UPPER_MIN=0.120
AUTO_ALIGN_UPPER_MAX=0.600

# The residual the aligner is tracking after the field is single-digit
# millimetres, so a step limit of 150 mm lets a bad frame move the correction
# further than the whole error it is correcting. Tighten it when a field is
# loaded.
AUTO_ALIGN_MAX_STEP_FIELD=0.020
AUTO_ALIGN_MAX_STEP_BARE=0.150

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
         face_consensus.py layer_align.py measure_deck.py \
         depth_field.py field_correct.py mask_audit.py; do
    [ -f "$f" ] || { echo "missing $f in $PROJECT" >&2; exit 1; }
done
[ -d "$CALIB" ] || { echo "calibration directory not found: $CALIB" >&2; exit 1; }

if [ -n "$FIELD" ] && [ -n "$AFFINE" ]; then
    echo "FIELD and AFFINE are both set. They are two different corrections" >&2
    echo "for the same error and da3_stream.py refuses both. The field" >&2
    echo "supersedes the affine: clear AFFINE." >&2
    exit 1
fi

PMIN=$(awk -v d="$DECK" -v b="$BAND" 'BEGIN{printf "%.4f", d-b}')
PMAX=$(awk -v d="$DECK" -v b="$BAND" 'BEGIN{printf "%.4f", d+b}')

check_bands() {
    awk -v t="$AUTO_ALIGN_TOL" -v m="$MIN_PARCEL_HEIGHT" \
        'BEGIN{exit !(m < t)}' || {
        echo "MIN_PARCEL_HEIGHT ($MIN_PARCEL_HEIGHT) must stay below" \
             "AUTO_ALIGN_TOL ($AUTO_ALIGN_TOL), or a parcel at that height is" \
             "counted as conveyor by the aligner and as a parcel by the" \
             "segmentation at the same time." >&2
        exit 1
    }
    awk -v t="$AUTO_ALIGN_TOL" -v u="$AUTO_ALIGN_UPPER_MIN" \
        'BEGIN{exit !(t < u/2)}' || {
        echo "AUTO_ALIGN_TOL ($AUTO_ALIGN_TOL) must stay below HALF of" \
             "AUTO_ALIGN_UPPER_MIN ($AUTO_ALIGN_UPPER_MIN)." >&2
        exit 1
    }
    awk -v u="$AUTO_ALIGN_UPPER_MIN" -v x="$AUTO_ALIGN_UPPER_MAX" \
        'BEGIN{exit !(u < x)}' || {
        echo "AUTO_ALIGN_UPPER_MIN must stay below AUTO_ALIGN_UPPER_MAX." >&2
        exit 1
    }
}

check_corroborate() {
    # The corroboration radius has to clear the residual spatial structure or
    # nothing corroborates. The number is written per camera by MODE=field.
    local rep="$1"
    [ -f "$rep" ] || return 0
    python - "$rep" "$CORROBORATE" <<'PY'
import json, sys
rep, corr = sys.argv[1], float(sys.argv[2]) * 1e3
d = json.loads(open(rep).read())
ss = [(n, c.get("spatial_structure_mm")) for n, c in
      (d.get("per_camera") or {}).items() if c.get("spatial_structure_mm")]
if not ss:
    raise SystemExit
worst = max(v for _, v in ss)
print(f"  spatial structure after the field: "
      + ", ".join(f"{n} {v:.1f}" for n, v in ss) + " mm")
if corr < worst:
    print(f"  ** CORROBORATE is {corr:.0f} mm and the worst residual structure")
    print(f"     is {worst:.1f} mm. A point seen by two cameras cannot agree")
    print(f"     with itself to {corr:.0f} mm, so --fuse-min-views 2 will")
    print(f"     discard almost every multi-view sample. Raise CORROBORATE")
    print(f"     above {worst:.0f} mm, or accept MIN_VIEWS=1. **")
PY
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

warn_stale_deck() {
    [ "$DECK_IS_STALE" = "1" ] || return 0
    echo
    echo "** DECK=$DECK is flagged stale. Run MODE=probe ./run.sh, put the"
    echo "   deck plane's distance here, and set DECK_IS_STALE=0. **"
    echo
}

# The check the offline report does not print: what the field does OUTSIDE the
# span of the planes it was fitted to. Every residual in field_report.json is
# measured inside that span, where the scale and the offset are ~99.99%
# correlated and any point along the null direction fits equally well.
check_field() {
    local fj="$1"
    [ -f "$fj" ] || { echo "no depth_field.json at $fj" >&2; return 1; }
    python - "$fj" "$FIELD_CHECK_HEIGHT" <<'PY'
import json, sys
from pathlib import Path

fj, h = Path(sys.argv[1]), float(sys.argv[2])
d = json.loads(fj.read_text())
coeffs = d.get("coefficients") or {}
planes = d.get("planes") or []
deck = next((p for p in planes if p.get("label") == "deck"), None)
span = 0.0
if len(planes) > 1:
    ds = sorted(float(p["perp_ref_m"]) for p in planes)
    span = ds[-1] - ds[0]

print(f"anchor span {span*1e3:.0f} mm; correction checked {h*1e3:.0f} mm above "
      f"the deck, so {max(0.0, h - span)*1e3:.0f} mm of it is EXTRAPOLATED")
print(f"{'camera':<9}{'s':>10}{'c mm':>10}{'at deck':>10}"
      f"{'at top':>10}{'swing':>9}")

bad = []
detail = d.get("detail") or {}
for n, c in coeffs.items():
    s, cc = float(c.get("s", 1.0)), float(c.get("c", 0.0))
    # the camera's own uncorrected depth to the deck, from the fit record
    dd = None
    for p in ((detail.get(n) or {}).get("per_plane") or []):
        if p.get("label") == "deck" and p.get("target_depth_median_m"):
            dd = float(p["target_depth_median_m"])
    if dd is None and deck is not None:
        dd = float(deck["perp_ref_m"])
    if dd is None:
        continue
    raw = (dd - cc) / s if abs(s) > 1e-9 else dd
    at_deck = raw * (s - 1.0) + cc
    at_top = (raw - h) * (s - 1.0) + cc
    swing = (at_top - at_deck) * 1e3
    print(f"{n:<9}{s:>10.6f}{cc*1e3:>10.2f}{at_deck*1e3:>10.2f}"
          f"{at_top*1e3:>10.2f}{swing:>9.2f}")
    if abs(swing) > 10.0:
        bad.append((n, s, swing))

    stages = (detail.get(n) or {}).get("iterations") or []
    cs = [it.get("c_mm") for it in stages if it.get("c_mm") is not None]
    if len(cs) >= 4:
        mono = all(b < a for a, b in zip(cs, cs[1:])) or \
               all(b > a for a, b in zip(cs, cs[1:]))
        if mono and abs(cs[-1] - cs[0]) > 25.0:
            print(f"  ** {n}: c moved monotonically from {cs[0]:+.0f} to "
                  f"{cs[-1]:+.0f} mm over {len(cs)} iterations without "
                  f"converging. That is the scale-offset null direction, not a "
                  f"solve. Set FIELD_FIX_SCALE=1. **")

for row in (d.get("layering") or []):
    print(f"  layer {row.get('label','?'):<8} spread after "
          f"{row.get('spread_after_mm')} mm "
          f"({len(row.get('after') or {})} of 4 cameras present)")
    if len(row.get("after") or {}) < 4:
        absent = "unknown"
        print(f"    ** a camera absent from a plane was never verified there. "
              f"Check MODE=audit for its coverage. **")

if bad:
    print()
    print("** THE SWING COLUMN IS THE ONE TO READ. It is the difference")
    print("   between the correction at the deck and the correction at parcel")
    print("   height, and it equals parcel_height * (s - 1). It is zero for a")
    print("   pure offset. The measured error on this rig IS a pure offset,")
    print("   flat in height to 1-4 mm over the anchor span, so any large")
    print("   swing is the fit walking along the degeneracy rather than")
    print("   describing the rig:")
    for n, s, sw in bad:
        print(f"     {n}: s={s:.4f} gives {sw:+.0f} mm of swing")
    print("   Set FIELD_FIX_SCALE=1 and re-run MODE=field. **")
    raise SystemExit(2)
PY
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
    print("This is the number that decides whether the box layer is flat, and")
    print("it is the number the FIELD exists to reduce. The field is verified")
    print("only inside the 218 mm anchor span; parcels above that are")
    print("extrapolated, so compare this against the swing column from")
    print("MODE=field before blaming the segmentation.")
    if max(spreads) > tol_mm:
        print(f"\n** the worst spread is {max(spreads):.0f} mm, at or above")
        print(f"   AUTO_ALIGN_TOL ({tol_mm:.0f} mm). Samples from the worst")
        print( "   camera are falling outside the assignment band, so that")
        print( "   camera is not being corrected at all. **")

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
    echo
    echo "PLACE THE RISERS WHERE ALL FOUR CAMERAS SEE THEM. The planes are"
    echo "measured from ${CALIB##*/}'s reference camera, but every camera is"
    echo "CORRECTED against them, and a camera with no samples on a plane"
    echo "cannot be verified there. Measured on capture_00001: left had 9"
    echo "usable pixels within 150 mm of the 218 mm riser against center's"
    echo "11279, so its correction above the deck was never checked."
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
        echo "** no riser images match '$RISER_SNAPS'. **"
        echo
    fi
    if [ -n "$FLOOR_SNAPS" ] && compgen -G "$FLOOR_SNAPS" > /dev/null; then
        # shellcheck disable=SC2086
        ARGS+=(--plane floor $FLOOR_SNAPS)
    fi
    python measure_deck.py "${ARGS[@]}" "$@"
    echo
    echo "The [check] line on the riser is the one that matters."
    echo
    echo "The anchor span also sets how far the field can be trusted. Three"
    echo "planes over 218 mm leave the scale and the offset ~99.99% correlated"
    echo "and parcels above 218 mm extrapolated. A fourth riser near"
    echo "$(awk -v v="$MAX_PARCEL_HEIGHT" 'BEGIN{printf "%.0f", v*1000}') mm"
    echo "would separate them and is the only thing that makes the scale"
    echo "term measurable rather than assumed."
    ;;

anchor)
    echo "calibration: $CALIB"
    echo "anchor     : $PROJECT/$TRUTH"
    echo "correction : ${FIELD:-${AFFINE:-none, auto-align only}}"
    echo
    warn_stale_deck
    [ -n "$FIELD" ] && { echo; check_field "$FIELD" || true; }
    ;;

probe)
    CAP=$(latest_capture)
    echo "probing $CAP"
    python box_segment.py --capture-dir "$CAP" --probe "$@"
    ;;

audit)
    CAP=$(latest_capture)
    echo "auditing $CAP against $TRUTH"
    echo
    echo "Read, per camera per plane, in this order:"
    echo "  1. rays meeting it in front of the camera. Near zero means the"
    echo "     plane is not in that camera's view and NO tuning helps: the"
    echo "     riser has to be moved and MODE=truth re-run."
    echo "  2. within the band before vs after masks, and which mask costs it."
    echo "  3. the cumulative count ladder. A cliff between two windows is the"
    echo "     surface, and FIELD_ASSIGN_TOL has to sit above it. On"
    echo "     capture_00001 left's deck went 10990 at 100 mm to 97285 at"
    echo "     200 mm, which is why a 100 mm window fitted its parcel tops."
    echo
    python mask_audit.py \
        --capture-dir "$CAP" \
        --gt "$TRUTH" \
        --max-incidence "$MAX_INCIDENCE" \
        --min-plane-points 500 \
        "$@"
    ;;

field)
    CAP=$(latest_capture)
    OUTD="$CAP/field"
    echo "solving the four-DOF depth field on $CAP"
    echo
    echo "  d'(u,v) = d(u,v) * ( s + a*x~ + b*y~ ) + c,  per camera"
    echo
    echo "THE RISERS MUST BE IN THIS CAPTURE, board REMOVED, standing where"
    echo "they were measured. Every camera is fitted to the PHYSICALLY MEASURED"
    echo "planes, including the reference, so there is no reference camera"
    echo "whose scale the others inherit and no separate anchor stage."
    echo
    ARGS=(--capture-dir "$CAP"
          --gt "$TRUTH"
          --out-dir "$OUTD"
          --holdout
          --reference center
          --bootstrap-plane "$FIELD_BOOTSTRAP"
          --assign-tol "$FIELD_ASSIGN_TOL"
          --assign-tol-stage2 "$FIELD_ASSIGN_TOL_STAGE2"
          --assign-tol-final "$FIELD_ASSIGN_TOL_FINAL"
          --iters "$FIELD_ITERS"
          --max-scale-dev "$FIELD_MAX_SCALE_DEV"
          --max-gain-span "$FIELD_MAX_GAIN_SPAN"
          --max-offset-m "$FIELD_MAX_OFFSET"
          --max-incidence "$MAX_INCIDENCE")
    if [ "$FIELD_FIX_SCALE" = "1" ]; then
        ARGS+=(--fix-scale)
        echo "scale     : FIXED at 1. The field is tilt plus a constant offset,"
        echo "            which is what was measured: the residual is flat in"
        echo "            height to 1-4 mm over the 218 mm anchor span."
    else
        echo "scale     : FREE. Watch the swing column below; over a 218 mm"
        echo "            lever arm s and c are ~99.99% correlated and the pair"
        echo "            walks rather than converging."
    fi
    echo
    python depth_field.py "${ARGS[@]}" "$@"
    echo
    echo "=== extrapolation check ==="
    check_field "$OUTD/depth_field.json" || {
        echo
        echo "The field was written but it does not pass the extrapolation"
        echo "check. Do NOT set FIELD= to it yet."
        exit 2
    }
    echo
    check_corroborate "$OUTD/field_report.json"
    echo
    echo "Read, in order:"
    echo "  1. the swing column above, then the holdout residual. Everything"
    echo "     else is measured on the planes it was fitted to."
    echo "  2. [layer] spread AFTER at the UPPER plane, and whether all four"
    echo "     cameras are present. An absent camera was never verified there."
    echo "  3. spatial structure, which is what no radial field can reach."
    echo
    echo "A plane sees the standoff and two tilt axes. It is blind to in-plane"
    echo "translation and to yaw, so a clean result here means the DEPTH FIELD"
    echo "is right, not that the cameras are registered."
    echo
    echo "Then set  FIELD=$OUTD/depth_field.json  in this script."
    ;;

align)
    check_bands
    CAP=$(latest_capture)
    echo "LEGACY: layer_align.py has no tilt term. Measured apparent tilt on"
    echo "capture_00001 ran 1.2 to 7.1 degrees per camera, which at 3 m sweeps"
    echo "a plane by tens of millimetres across the frame and no scalar pair"
    echo "reaches it. Use MODE=field. This mode is kept for comparison only."
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

    warn_stale_deck

    CORRECTION=()
    if [ -n "$FIELD" ]; then
        [ -f "$FIELD" ] || { echo "FIELD not found: $FIELD" >&2; exit 1; }
        check_field "$FIELD" || {
            echo "FIELD fails the extrapolation check; refusing to stream" \
                 "with it. Re-run MODE=field with FIELD_FIX_SCALE=1." >&2
            exit 2
        }
        echo
        # The field goes on FIRST and the aligner runs on its output. The field
        # is static, metric and tilt-aware; the aligner is per-frame and now has
        # one job, tracking the residual offset drift. It must NOT re-solve
        # scale: the field already carries it, and the deck-to-parcel lever arm
        # is the same short baseline over which the offline pair walked to
        # s=1.058 on top.
        CORRECTION=(--load-field "$FIELD"
                    --auto-align
                    --auto-align-one-plane
                    --auto-align-tol "$AUTO_ALIGN_TOL"
                    --auto-align-max-step "$AUTO_ALIGN_MAX_STEP_FIELD"
                    --auto-align-upper-band "$AUTO_ALIGN_UPPER_MIN" \
                                            "$AUTO_ALIGN_UPPER_MAX")
        echo "correction: four-DOF field from $FIELD,"
        echo "            then OFFSET-ONLY auto-align on top of it, step"
        echo "            limited to" \
             "$(awk -v v="$AUTO_ALIGN_MAX_STEP_FIELD" 'BEGIN{printf "%.0f", v*1000}') mm."
        echo "            The field owns the scale and the tilt; the aligner"
        echo "            tracks only the per-frame offset drift. That drift is"
        echo "            real: the offsets are a per-VIEW property of the"
        echo "            model, measured at +177 mm on left against +28 mm on"
        echo "            right in the same scene, so they move as the parcel"
        echo "            layout moves and cannot be calibrated once."
    elif [ -n "$AFFINE" ] && [ -f "$AFFINE" ]; then
        CORRECTION=(--auto-align
                    --auto-align-tol "$AUTO_ALIGN_TOL"
                    --auto-align-max-step "$AUTO_ALIGN_MAX_STEP_BARE"
                    --auto-align-upper-band "$AUTO_ALIGN_UPPER_MIN" \
                                            "$AUTO_ALIGN_UPPER_MAX")
        if [ -f "$TRUTH" ]; then
            CORRECTION+=(--load-affine "$AFFINE" --apply-absolute)
            echo "correction: LEGACY two-plane affine from $AFFINE. No tilt"
            echo "            term. Run MODE=field instead."
        else
            CORRECTION+=(--load-affine "$AFFINE")
            echo "correction: LEGACY affine, --apply-absolute withheld: no $TRUTH."
        fi
    else
        CORRECTION=(--auto-align
                    --auto-align-tol "$AUTO_ALIGN_TOL"
                    --auto-align-max-step "$AUTO_ALIGN_MAX_STEP_BARE"
                    --auto-align-upper-band "$AUTO_ALIGN_UPPER_MIN" \
                                            "$AUTO_ALIGN_UPPER_MAX")
        echo "correction: auto-align only, no field and no metric anchor."
        echo "            Relative layering is reduced; the absolute heights"
        echo "            and the per-camera tilt are NOT corrected. Capture"
        echo "            with the risers in frame and run MODE=field."
    fi

    echo "deck band : $PMIN to $PMAX m in DA3 coordinates"
    echo "parcels   : between $(awk -v v="$MIN_PARCEL_HEIGHT" 'BEGIN{printf "%.0f", v*1000}')" \
         "and $(awk -v v="$MAX_PARCEL_HEIGHT" 'BEGIN{printf "%.0f", v*1000}') mm above the deck," \
         "$SEG_MIN_VIEWS view(s) required to pick"
    echo "fusion    : incidence <= ${MAX_INCIDENCE} deg, corroboration at" \
         "$(awk -v v="$CORROBORATE" 'BEGIN{printf "%.0f", v*1000}') mm," \
         "min views $MIN_VIEWS"
    if [ -f "$BELT" ]; then
        echo "belt      : frozen, from $BELT"
    else
        echo "belt      : not yet measured. The FIRST capture defines it, so"
        echo "            make that one an EMPTY belt."
    fi
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
    echo "  MODE=audit ./run.sh      per-camera coverage of each reference plane"
    echo "  MODE=field ./run.sh      re-solve the four-DOF field on this capture"
    echo "  MODE=reseg ./run.sh --seg-views all   retune offline"
    echo "  rm $BELT                 re-measure the conveyor footprint"
    ;;

*)
    echo "unknown MODE '$MODE'; expected truth, anchor, probe, audit, field," \
         "align, reseg or stream" >&2
    exit 1
    ;;
esac