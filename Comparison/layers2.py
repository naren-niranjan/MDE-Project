#!/usr/bin/env python3
"""
layers2.py — layers.py with the four defects found in the 0707/0720 re-run fixed.

Changes vs layers.py (each marked [FIX-n] in the code):

  [FIX-1] Deck/floor plausibility gate. layers.py calls the farthest peak the
          floor unconditionally. When the floor is not reconstructed, or when
          one plane fragments into several peaks, "deck" and "floor" land on the
          SAME physical surface a few tens of mm apart, and every downstream
          number (alpha, beta, gap ratio, "not affine" verdict) is fabricated.
          Now: if the deck->floor separation implies an alpha more than
          FAR_TOL x the scene's rough alpha, the pair is declared unresolved and
          the fit is suppressed rather than printed.

  [FIX-2] layer_area tolerance is derived from the scene, not fixed at 30 mm.
          At 0720 scales, adjacent layers sit ~25 mm apart, so a 30 mm slab
          blends three of them and the area ranking that picks the deck is
          meaningless.

  [FIX-3] Peak budget raised 8 -> 16, and the script says when the budget binds.
          With per-scene suppression (13 mm instead of 35 mm) the budget is
          consumed splitting the dominant plane, which silently dropped the
          445 mm parcel top from 0707/center and turned its scale-free gap
          ratio from -0.14% into -49.5%.

  [FIX-4] Layer assignment is no longer assumed. When fewer layers are resolved
          than the scan contains, every order-preserving assignment is scored
          and the script reports whether the nearest-to-deck default is actually
          the best one, and by how much. If it is not clearly best, the fit is
          labelled ambiguous.

  [FIX-5] per_view picked each view's largest-area layer and called it the deck.
          The floor is normally the largest-area layer, so views that see floor
          were compared against views that see deck, and the "inter-view spread"
          was a deck-to-floor distance. Now per_view excludes the farthest layer
          exactly as describe() does, and warns when views disagree about which
          surface they picked.

Usage is unchanged:
    python3 layers2.py --gt gt_scene.json --capture-root runs_da3 --runs runs
"""
import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np

import rigkit as rk

NMAX = 16          # [FIX-3] was 8
FAR_TOL = 4.0      # [FIX-1] deck->floor alpha may exceed the scene alpha by
                   #         this much before the pair is called unresolved
AMBIG = 1.5        # [FIX-4] best assignment must beat the runner-up by this
                   #         factor in RMS to count as identified


def merge(pk, gap=0.06):
    """Fold peaks closer than `gap` into one; a broad surface splits otherwise."""
    out = []
    for c, n in sorted(pk):
        if out and c - out[-1][0] < gap:
            c0, n0 = out[-1]
            out[-1] = ((c0 * n0 + c * n) / (n0 + n), n0 + n)
        else:
            out.append((c, n))
    return out


def peaks(z, bw=0.004, minfrac=0.01, sup=0.035, nmax=NMAX):
    """`sup` is the peak suppression radius. It MUST be well under the smallest
    layer separation the scene actually contains, or distinct parcel tops get
    folded into one peak and the scene looks less resolved than it is.

    Returns (peaks, budget_bound). budget_bound is True when the loop stopped
    because it hit nmax rather than because it ran out of mass -- in that state
    weak-but-real layers are being crowded out by fine splitting of the
    dominant plane. [FIX-3]
    """
    lo, hi = np.percentile(z, [0.2, 99.8])
    if hi - lo < 1e-3:
        return [], False
    h, e = np.histogram(z, bins=int(max(8, (hi - lo) / bw)), range=(lo, hi))
    c = 0.5 * (e[:-1] + e[1:])
    out, h = [], h.astype(float)
    bound = True
    for _ in range(nmax):
        k = int(h.argmax())
        if h[k] < minfrac * len(z) / 4:
            bound = False
            break
        m = np.abs(c - c[k]) < sup
        out.append((float(c[k]), int(h[m].sum())))
        h[m] = 0
    return sorted(out), bound


def scene_scales(z, gt):
    """Set the peak-resolution scales from the scene, not from a constant.

    Returns (suppression radius, merge gap, smallest predicted gap, alpha) in
    PREDICTED depth units, derived from the smallest true layer separation and
    a rough alpha taken from the span ratio."""
    deck = -gt["deck_plane"]["d"]
    floor = -gt["floor_plane"]["d"]
    hts = sorted(p["H_mm"] / 1000 for p in gt["parcels"])
    lay = merge([(deck - h, 1) for h in hts] + [(deck, 1)], gap=0.015)
    lay = [c for c, _ in lay]
    true_gaps = np.diff(lay) if len(lay) > 1 else np.array([0.1])
    p_lo, p_hi = np.percentile(z, [1.0, 99.0])
    true_span = floor - (deck - hts[-1])
    alpha = true_span / max(p_hi - p_lo, 1e-3)
    g_pred = float(true_gaps.min()) / max(alpha, 1e-3)
    return max(0.008, 0.40 * g_pred), max(0.012, 0.60 * g_pred), g_pred, alpha


def layer_area(P, z0, tol=0.03, cell=0.05):
    """Occupied area of a layer, in m^2. Ranks floor > deck >> parcels, which
    point count does not: a merged parcel band can outnumber the deck.

    `tol` must be passed in from the scene scales, not left at 30 mm. [FIX-2]
    """
    m = np.abs(P[:, 2] - z0) < tol
    if m.sum() < 50:
        return 0.0
    k = np.unique(np.floor(P[m][:, :2] / cell).astype(np.int64), axis=0)
    return len(k) * cell * cell


def true_layers(gt):
    """Physical layers the scan contains: parcel tops (merged at 15 mm) + deck."""
    deck = -gt["deck_plane"]["d"]
    hts = sorted(p["H_mm"] / 1000 for p in gt["parcels"])
    return [c for c, _ in merge([(deck - h, 1) for h in hts] + [(deck, 1)],
                                gap=0.015)]


def score_assignment(x, y):
    A = np.polyfit(x, y, 1)
    res = y - np.polyval(A, x)
    return float(np.sqrt((res ** 2).mean())), A, res


def best_assignments(x, true_band, alo=0.2, ahi=4.0):
    """Every order-preserving assignment of the k resolved layers onto the true
    layers, scored by RMS, keeping only physically sane slopes. [FIX-4]"""
    k = len(x)
    out = []
    for idx in combinations(range(len(true_band)), k):
        y = np.array([true_band[i] for i in idx])
        rms, A, res = score_assignment(x, y)
        if not (alo <= A[0] <= ahi):
            continue
        out.append((rms, idx, A, res, y))
    out.sort(key=lambda t: t[0])
    return out


def gap_ratio(x, y):
    gp, gt_ = np.diff(x), np.diff(y)
    if len(gp) < 2 or gt_.min() <= 0.01:
        return None
    rp, rt = gp[-1] / gp[0], gt_[-1] / gt_[0]
    return rp, rt, 100 * (rp - rt) / rt


def describe(name, z, gt, n_total=None, P=None):
    deck = -gt["deck_plane"]["d"]
    floor = -gt["floor_plane"]["d"]
    sep = floor - deck
    hts = sorted(p["H_mm"] / 1000 for p in gt["parcels"])
    sup, mg, g_pred, a_est = scene_scales(z, gt)
    raw_pk, bound = peaks(z, sup=sup)
    pk = merge(raw_pk, gap=mg)
    tol = max(0.008, 0.45 * mg)                                    # [FIX-2]
    span = float(z.max() - z.min())
    print(f"  {name:22s} n={len(z):8d}"
          + (f" ({100*len(z)/n_total:5.1f}% of unfiltered)" if n_total else "")
          + f"  z[{z.min():.2f},{z.max():.2f}] span={span:.3f}")
    if not pk:
        print("      no layers found")
        return
    areas = [layer_area(P, c, tol=tol) if P is not None else float("nan")
             for c, _ in pk]
    print("      layers (predicted z, points, area m2, gap to next):")
    for i, (c, n) in enumerate(pk):
        gap = f"{pk[i+1][0]-c:+.3f}" if i + 1 < len(pk) else "    -"
        print(f"        {c:6.3f}  {n:8d}  {areas[i]:7.2f}   {gap}")
    print(f"      scan says: floor {floor:.3f}, deck {deck:.3f} "
          f"(gap {sep:.3f}), parcel tops {deck:.3f}-[{hts[0]:.3f}..{hts[-1]:.3f}]")
    print(f"      resolution: smallest scan gap maps to ~{1000*g_pred:.0f} mm "
          f"predicted (alpha~{a_est:.2f}); peak suppression {1000*sup:.0f} mm, "
          f"merge {1000*mg:.0f} mm, area slab {1000*tol:.0f} mm")
    if bound:                                                       # [FIX-3]
        print(f"      *** peak budget ({NMAX}) exhausted -- weak layers may be "
              "crowded out by\n          fine splitting of the dominant plane. "
              "Treat missing layers as unproven.")
    if len(pk) < 3:
        print("      *** fewer than 3 layers: parcel band missing or merged. "
              "Loosen --conf-percentile.")
        return

    floor_i = len(pk) - 1
    if P is not None and max(areas[:floor_i] or [0]) > 0:
        deck_i = max(range(floor_i), key=lambda i: areas[i])
    else:
        deck_i = max(range(floor_i), key=lambda i: pk[i][1])
    d_z, f_z = pk[deck_i][0], pk[floor_i][0]
    print(f"      deck layer = {d_z:.3f} ({pk[deck_i][1]} pts), "
          f"floor = {f_z:.3f} ({pk[floor_i][1]} pts)")

    # ---------------------------------------------------------- [FIX-1] gate
    a_far = sep / max(f_z - d_z, 1e-6)
    if a_far > FAR_TOL * a_est:
        print(f"      *** DECK/FLOOR UNRESOLVED. They sit {1000*(f_z-d_z):.0f} mm "
              f"apart, which implies\n          alpha={a_far:.1f} against a scene "
              f"alpha of ~{a_est:.2f}. The true 783 mm separation would\n"
              f"          need ~{1000*sep/a_est:.0f} mm here. Both peaks are on "
              "one physical surface, or the\n          floor was never "
              "reconstructed. No fit reported -- nothing to fit.")
        return

    band = [pk[i][0] for i in range(deck_i + 1)]
    tb = true_layers(gt)
    if len(band) < 3:
        if len(pk) > deck_i + 2:
            print("      (parcel layers may have merged; they differ by only "
                  f"{1000*(hts[-1]-hts[0]):.0f} mm)")
        print("      *** fewer than 2 parcel layers resolved; "
              "cannot measure relative depth")
        return

    x = np.array(band)
    k = min(len(x), len(tb))
    x = x[-k:]
    y_def = np.array(tb[-k:])                     # nearest-to-deck default
    rms_def, A_def, res_def = score_assignment(x, y_def)

    cands = best_assignments(x, tb)               # [FIX-4]
    default_idx = tuple(range(len(tb) - k, len(tb)))
    if cands:
        rms_b, idx_b, A_b, res_b, y_b = cands[0]
        runner = cands[1][0] if len(cands) > 1 else float("inf")
        identified = runner > AMBIG * rms_b
    else:
        rms_b, idx_b, A_b, res_b, y_b = rms_def, default_idx, A_def, res_def, y_def
        runner, identified = float("inf"), False

    if len(x) != len(tb):
        print(f"      [note] {len(x)} predicted layers vs {len(tb)} in the scan")
    print(f"      DEFAULT assignment (nearest {k} to the deck): "
          f"alpha={A_def[0]:.4f} beta={A_def[1]*1000:+.0f} mm")
    print("        residuals (mm): "
          + "  ".join(f"{r*1000:+.1f}" for r in res_def)
          + f"   RMS {1000*rms_def:.1f}")
    gr = gap_ratio(x, y_def)
    if gr:
        print(f"        scale-free gap ratio: predicted {gr[0]:.4f} vs scan "
              f"{gr[1]:.4f}  ->  {gr[2]:+.2f}%")
    if idx_b != default_idx:
        print(f"      *** a different assignment fits better: layers "
              f"{list(idx_b)} of {len(tb)}, "
              f"alpha={A_b[0]:.4f} RMS {1000*rms_b:.1f} mm "
              f"(default {1000*rms_def:.1f} mm)")
    if not identified:
        print(f"      *** ASSIGNMENT AMBIGUOUS: runner-up RMS {1000*runner:.1f} mm "
              f"vs best {1000*rms_b:.1f} mm.\n          alpha is not identified "
              "from this cell; do not quote it.")

    # span sanity: does alpha map the cloud onto a scene-sized depth range?
    true_span = floor - (deck - hts[-1])
    print(f"      span check: {span:.3f} m predicted x alpha "
          f"= {span*A_def[0]:.3f} m vs scene depth {true_span:.3f} m "
          f"({span*A_def[0]/true_span:.2f}x)")
    print(f"      deck->floor implies alpha={a_far:.3f} vs working-band "
          f"{A_def[0]:.3f}  ({a_far/A_def[0]:.2f}x)")
    if a_far / A_def[0] > 1.3 or a_far / A_def[0] < 0.77:
        print("      *** the response is NOT affine across this range. Do not "
              "fit alpha/beta\n          on deck+floor; the floor is outside "
              "the band the model behaves in.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="gt_scene.json")
    ap.add_argument("--runs", default="runs", help="fused subset clouds")
    ap.add_argument("--capture-root", help="for the unfiltered comparison")
    a = ap.parse_args()
    gt = json.load(open(a.gt))

    def per_view(sub):
        """Where each view puts the DECK. [FIX-5] The deck is the largest-area
        layer that is not the farthest -- the farthest is the floor, and it is
        normally the largest layer of all, so taking a plain max over area
        compares one view's floor against another view's deck."""
        cams = sub.split("+")
        if len(cams) < 2 or not a.capture_root:
            return
        rows = []
        for c in cams:
            p_ = os.path.join(a.capture_root, sub, f"cloud_{c}.ply")
            if not os.path.exists(p_):
                return
            Pv = rk.load_cloud(p_)
            sv, mv, _, _ = scene_scales(Pv[:, 2], gt)
            tolv = max(0.008, 0.45 * mv)
            pv = merge(peaks(Pv[:, 2], sup=sv)[0], gap=mv)
            if len(pv) < 2:
                return
            av = [layer_area(Pv, t[0], tol=tolv) for t in pv]
            fi = len(pv) - 1
            di = max(range(fi), key=lambda i: av[i])
            rows.append((c, pv[di][0], av[di], pv[fi][0], av[fi], len(pv)))
        print("      per view, deck = largest-area layer that is not the farthest:")
        for c, dz, da_, fz, fa, nl in rows:
            print(f"        {c:7s} deck z={dz:.3f} ({da_:4.2f} m2)   "
                  f"far z={fz:.3f} ({fa:4.2f} m2)   {nl} layers")
        spread = max(r[1] for r in rows) - min(r[1] for r in rows)
        # is any view's "deck" really another view's floor?
        suspect = [c for c, dz, da_, fz, fa, nl in rows if da_ < 0.5]
        print(f"        inter-view spread on the deck: {1000*spread:.0f} mm"
              + ("   *** the views are on different sheets; this is what "
                 "--correction removes" if spread > 0.03 else ""))
        if suspect:
            print(f"        *** {', '.join(suspect)} reconstructed under 0.5 m2 "
                  "of deck -- its deck estimate\n            is not trustworthy "
                  "and the spread above is dominated by it.")

    for f in sorted(glob.glob(os.path.join(a.runs, "*.ply"))):
        sub = os.path.splitext(os.path.basename(f))[0]
        Pc = rk.load_cloud(f)
        z = Pc[:, 2]
        n_un = None
        if a.capture_root:
            zs = []
            for c in sub.split("+"):
                d = os.path.join(a.capture_root, sub, f"depth_{c}.npy")
                if os.path.exists(d):
                    arr = np.load(d).ravel()
                    zs.append(arr[np.isfinite(arr) & (arr > 0)])
            if zs:
                zu = np.concatenate(zs)
                n_un = len(zu)
                # NOTE: raw is per-camera depth along the optical axis; filtered
                # is world z. Do not compare their spans directly.
                describe(sub + " [raw]", zu, gt)
        describe(sub + " [filtered]", z, gt, n_un, P=Pc)
        per_view(sub)
        print()


if __name__ == "__main__":
    main()