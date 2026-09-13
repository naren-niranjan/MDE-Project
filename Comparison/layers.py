#!/usr/bin/env python3
"""
layers.py — what horizontal layers are actually in a cloud, and what filtering
removed. Run this when the scale fit reports "floor 0 pts".

    python3 layers.py --gt gt_scene.json --capture-root runs_da3 --runs runs

For each subset it finds the depth layers, matches their spacings against the
scene the scan describes (parcel tops, deck, floor), and says which survived.
If the parcel band is gone, no depth correction can put it back and the filter
settings are the thing to fix -- not the fitter.
"""
import argparse
import glob
import json
import os
import numpy as np

import rigkit as rk


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


def peaks(z, bw=0.004, minfrac=0.01, sup=0.035, nmax=8):
    """`sup` is the peak suppression radius. It MUST be well under the smallest
    layer separation the scene actually contains, or distinct parcel tops get
    folded into one peak and the scene looks less resolved than it is."""
    lo, hi = np.percentile(z, [0.2, 99.8])
    if hi - lo < 1e-3:
        return []
    h, e = np.histogram(z, bins=int(max(8, (hi - lo) / bw)), range=(lo, hi))
    c = 0.5 * (e[:-1] + e[1:])
    out, h = [], h.astype(float)
    for _ in range(nmax):
        k = int(h.argmax())
        if h[k] < minfrac * len(z) / 4:
            break
        m = np.abs(c - c[k]) < sup
        out.append((float(c[k]), int(h[m].sum())))
        h[m] = 0
    return sorted(out)


def scene_scales(z, gt):
    """Set the peak-resolution scales from the scene, not from a constant.

    Returns (suppression radius, merge gap) in PREDICTED depth units, derived
    from the smallest true layer separation and a rough alpha taken from the
    span ratio."""
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
    point count does not: a merged parcel band can outnumber the deck."""
    m = np.abs(P[:, 2] - z0) < tol
    if m.sum() < 50:
        return 0.0
    k = np.unique(np.floor(P[m][:, :2] / cell).astype(np.int64), axis=0)
    return len(k) * cell * cell


def describe(name, z, gt, n_total=None, P=None):
    deck = -gt["deck_plane"]["d"]
    floor = -gt["floor_plane"]["d"]
    sep = floor - deck
    hts = sorted(p["H_mm"] / 1000 for p in gt["parcels"])
    sup, mg, g_pred, a_est = scene_scales(z, gt)
    pk = merge(peaks(z, sup=sup), gap=mg)
    span = float(z.max() - z.min())
    print(f"  {name:22s} n={len(z):8d}"
          + (f" ({100*len(z)/n_total:5.1f}% of unfiltered)" if n_total else "")
          + f"  z[{z.min():.2f},{z.max():.2f}] span={span:.3f}")
    if not pk:
        print("      no layers found")
        return
    areas = [layer_area(P, c) if P is not None else float("nan") for c, _ in pk]
    print("      layers (predicted z, points, area m2, gap to next):")
    for i, (c, n) in enumerate(pk):
        gap = f"{pk[i+1][0]-c:+.3f}" if i + 1 < len(pk) else "    -"
        print(f"        {c:6.3f}  {n:8d}  {areas[i]:7.2f}   {gap}")
    # scene the scan says exists, expressed as gaps from the farthest surface
    print(f"      scan says: floor {floor:.3f}, deck {deck:.3f} "
          f"(gap {sep:.3f}), parcel tops {deck:.3f}-[{hts[0]:.3f}..{hts[-1]:.3f}]")
    # ---- scale-free depth accuracy over the working band -----------------
    # The deck is the largest layer that is not the farthest; the floor is the
    # farthest. Everything nearer than the deck is a parcel top.
    if len(pk) < 3:
        print("      *** fewer than 3 layers: parcel band missing or merged. "
              "Loosen --conf-percentile.")
        return
    floor_i = len(pk) - 1
    if P is not None and max(areas[:floor_i] or [0]) > 0:
        deck_i = max(range(floor_i), key=lambda i: areas[i])
    else:
        deck_i = max(range(floor_i), key=lambda i: pk[i][1])
    band = [pk[i][0] for i in range(deck_i + 1)]          # parcels + deck
    # Parcels whose tops differ by a few mm are ONE physical layer. P2 at
    # 323.0 and P4 at 324.0 are not resolvable, and treating them as two puts
    # a 1 mm interval in the true gaps, which makes any ratio meaningless.
    print(f"      resolution: smallest scan gap maps to ~{1000*g_pred:.0f} mm "
          f"predicted (alpha~{a_est:.2f}); peak suppression {1000*sup:.0f} mm, "
          f"merge {1000*mg:.0f} mm")
    true_band = merge([(deck - h, 1) for h in hts] + [(deck, 1)], gap=0.015)
    true_band = [c for c, _ in true_band]
    print(f"      deck layer = {pk[deck_i][0]:.3f} ({pk[deck_i][1]} pts), "
          f"floor = {pk[floor_i][0]:.3f} ({pk[floor_i][1]} pts)")
    if len(band) < 3 and len(pk) > deck_i + 2:
        print("      (parcel layers may have merged; they differ by only "
              f"{1000*(hts[-1]-hts[0]):.0f} mm)")
    if len(band) < 3:
        print("      *** fewer than 2 parcel layers resolved; "
              "cannot measure relative depth")
        return
    if len(band) != len(true_band):
        print(f"      [note] {len(band)} predicted layers vs "
              f"{len(true_band)} in the scan -- matching the nearest "
              f"{min(len(band), len(true_band))} to the deck")
    k = min(len(band), len(true_band))
    x, y = np.array(band[-k:]), np.array(true_band[-k:])
    A = np.polyfit(x, y, 1)
    res = y - np.polyval(A, x)
    print(f"      working band, {k} layers, local fit alpha={A[0]:.4f} "
          f"beta={A[1]*1000:+.0f} mm")
    print(f"        residuals (mm): "
          + "  ".join(f"{r*1000:+.1f}" for r in res)
          + f"   RMS {1000*np.sqrt((res**2).mean()):.1f}")
    # scale-free: ratio of layer gaps needs no alpha or beta at all
    if k >= 3:
        gp, gt_ = np.diff(x), np.diff(y)
        if gt_.min() > 0.01:
            rp, rt = gp[-1] / gp[0], gt_[-1] / gt_[0]
            print(f"      SCALE-FREE gap ratio: predicted {rp:.4f} vs scan "
                  f"{rt:.4f}  ->  {100*(rp-rt)/rt:+.2f}%")
    a_far = sep / (pk[floor_i][0] - pk[deck_i][0])
    print(f"      deck->floor implies alpha={a_far:.3f} vs working-band "
          f"{A[0]:.3f}  ({a_far/A[0]:.2f}x)")
    if a_far / A[0] > 1.3 or a_far / A[0] < 0.77:
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
        """Where each view puts its surfaces. Two views placing the same deck
        at different depths is the layering the correction exists to remove."""
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
            pv = merge(peaks(Pv[:, 2], sup=sv), gap=mv)
            if not pv:
                return
            big = max(pv, key=lambda t: layer_area(Pv, t[0]))
            rows.append((c, big[0], layer_area(Pv, big[0]), len(pv)))
        print("      per view, largest-area layer (the deck):")
        for c, zc, ar, nl in rows:
            print(f"        {c:7s} z={zc:.3f}  area={ar:5.2f} m2  "
                  f"{nl} layers")
        spread = max(r[1] for r in rows) - min(r[1] for r in rows)
        print(f"        inter-view spread on the deck: {1000*spread:.0f} mm"
              + ("   *** the views are on different sheets; this is what "
                 "--correction removes" if spread > 0.03 else ""))

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
                describe(sub + " [raw]", zu, gt)
        describe(sub + " [filtered]", z, gt, n_un, P=Pc)
        per_view(sub)
        print()


if __name__ == "__main__":
    main()