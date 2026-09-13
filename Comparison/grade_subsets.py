#!/usr/bin/env python3
"""
grade_subsets.py — grade camera-subset reconstructions against the reference
scan, with the depth scale fitted on held-out geometry.

WHY THE TWO-STAGE FIT
    This scene has three usable parcels. Fitting alpha/beta on those three and
    then scoring the same three is circular: two parameters on three points,
    scored on the fit set, and whichever subset has the most absorbable error
    comes out looking best.

    So alpha/beta are fitted on the deck and floor planes -- two large surfaces
    ~780 mm apart, far more than the height spread the parcels offer -- and the
    parcels are graded as held-out data. Two result columns, different weight:

      deck residual / tilt / bias : ~300k points. Statistically real. Isolates
                                    depth quality from segmentation entirely.
      parcel dL/dW/dH/dCentre     : n=3. Per-parcel rows only. Never a sigma.

USAGE
    python3 grade_subsets.py --gt gt_scene.json --selftest
    python3 grade_subsets.py --gt gt_scene.json --dir runs --exclude P1 --csv results.csv
    python3 grade_subsets.py --gt gt_scene.json --cloud runs/left+right.ply \
                             --subset left+right --exclude P1

INPUT
    One cloud per subset in runs/<subset>.ply (or .npy), RIG frame, metres,
    UNCORRECTED -- do not pre-apply any alpha/beta, this script fits its own
    per subset. Subset names: center, left, right, center+left, center+right,
    left+right, center+left+right.
"""
import argparse
import csv
import json
import os
import sys
import numpy as np

try:
    from scipy import ndimage
except ImportError:
    sys.exit("needs scipy:  pip install scipy")

import rigkit as rk


# ------------------------------------------------------- stage 1: scale
def init_scale(P, gt, bwin=0.004, nalpha=400, verbose=True, anchor=None):
    """Consensus search for (alpha, beta), with the window derived from the data.

    Peak-matching on the depth histogram is not safe: a parcel top is a
    flatter, denser band than the tilted deck and wins the argmax. Instead,
    for each candidate alpha every point votes for the beta that would land it
    on the deck plane and the beta that would land it on the floor plane; the
    tallest beta bin across all alphas wins. Parcels and rails vote
    incoherently and wash out.

    The search window is NOT fixed. Without a pose prior DA3 returns whatever
    metric scale it infers from intrinsics alone, and on this rig that lands
    anywhere from 1.3 to 3.1 m against a true 2.9 to 4.1 m -- alpha near 1.5
    and beta near a metre. A hard-coded window silently pins the optimum at
    its edge and returns a fit that recovers no parcels. So alpha is bracketed
    by the ratio of true scene span to predicted span, and beta by the offsets
    that ratio implies."""
    dn, dd = np.array(gt["deck_plane"]["n"]), gt["deck_plane"]["d"]
    if anchor:
        fn, fd, _ = parcel_plane(gt, anchor)
    else:
        fn, fd = np.array(gt["floor_plane"]["n"]), gt["floor_plane"]["d"]
    S = P[:: max(1, len(P) // 60000)]
    z = S[:, 2]
    zt = np.concatenate([rk.plane_z(S, dn, dd), rk.plane_z(S, fn, fd)])
    zz = np.concatenate([z, z])

    # true scene extent: tallest parcel top down to the floor
    tallest = max([p["H_mm"] / 1000 for p in gt["parcels"]] or [0.0])
    t_lo = -dd - tallest
    t_hi = -np.array(gt["floor_plane"]["d"], float).item()
    p_lo, p_hi = np.percentile(z, [1.0, 99.0])
    p_span = max(p_hi - p_lo, 1e-3)
    ratio = (t_hi - t_lo) / p_span
    if anchor:
        # The anchor pair spans only the working band, but the cloud still
        # contains the floor, so the whole-scene ratio is not the local slope.
        # Search broadly and let the scoring pick.
        t_hi = -dd
        a_lo, a_hi = 0.3, 3.5
        ratio = 1.0
    else:
        a_lo = float(np.clip(ratio / 2.5, 0.2, 8.0))
        a_hi = float(np.clip(ratio * 2.5, 0.3, 10.0))
    b_lo = t_lo - a_hi * p_hi
    b_hi = t_hi - a_lo * p_lo
    if verbose:
        print(f"       predicted z {p_lo:.2f}..{p_hi:.2f} m vs true "
              f"{t_lo:.2f}..{t_hi:.2f} m -> alpha~{ratio:.3f}, "
              f"searching alpha [{a_lo:.2f},{a_hi:.2f}] "
              f"beta [{b_lo*1000:+.0f},{b_hi*1000:+.0f}] mm")

    edges = np.arange(b_lo, b_hi + bwin, bwin)
    if len(edges) < 4:
        edges = np.linspace(b_lo - 0.1, b_hi + 0.1, 64)

    # Two planes give two constraints, and a parcel-top layer can impersonate
    # one of them: mapping parcel tops -> deck and floor -> floor is entirely
    # self-consistent, and on a loaded belt it can carry MORE points than the
    # real deck. So the histogram vote only proposes candidates; each is then
    # rescored on what it implies about the rest of the cloud.
    zd = rk.plane_z(S, dn, dd)
    zf = rk.plane_z(S, fn, fd)
    sep = float(abs(fd - dd))
    void_lo, void_hi = (-tol, -sep + tol) if not anchor else (None, None)
    cands = []
    for a in np.linspace(a_lo, a_hi, nalpha):
        hd, _ = np.histogram(zd - a * z, bins=edges)
        hf, _ = np.histogram(zf - a * z, bins=edges)
        both = np.minimum(hd, hf)
        k = int(both.argmax())
        if both[k] > 0:
            cands.append((int(both[k]), float(a),
                          float(0.5 * (edges[k] + edges[k + 1]))))
    if not cands:
        if verbose:
            print("       no (alpha,beta) puts points on BOTH deck and floor. "
                  "Is the floor in this cloud at all?")
        return ratio, float(t_lo - ratio * p_lo)

    cands.sort(key=lambda c: -c[0])
    tol = 0.02
    best, bs = None, -np.inf
    seen = 0
    for _, a, b in cands[:60]:
        h = zd - (a * z + b)          # height above deck after correction
        n_deck = int((np.abs(h) < tol).sum())
        n_floor = int((np.abs(h + sep) < tol).sum())
        # nothing physical lives between the deck and the floor. A solution
        # that parks a large planar population in there has mistaken some
        # other layer for the deck.
        if anchor:
            # nothing lives below the deck except the floor, far away
            n_void = int(((h < -0.02) & (h > -0.60)).sum())
        else:
            n_void = int(((h < -tol) & (h > -sep + tol)).sum())
        # parcels stand ON the deck, within the segmentation band
        n_parcel = int(((h > 0.02) & (h < 0.70)).sum())
        # and nothing should float above the tallest parcel
        n_sky = int((h > 0.90).sum())
        if min(n_deck, n_floor) == 0:
            continue          # both anchor surfaces must be populated
        sc = min(n_deck, n_floor) + 0.25 * n_parcel - 1.5 * n_void - 1.5 * n_sky
        if sc > bs:
            bs, best = sc, (a, b, n_deck, n_floor, n_void, n_parcel, n_sky)
    if best is None:
        if verbose:
            print("       no candidate populates both anchor surfaces")
        return ratio, float(t_lo - ratio * p_lo)
    a, b = best[0], best[1]
    if verbose:
        print(f"       consensus alpha={a:.4f} beta={b*1000:+.1f} mm  "
              f"(deck {best[2]}, floor {best[3]}, parcel band {best[5]}, "
              f"void {best[4]}, sky {best[6]})")
    return a, b


def parcel_plane(gt, pid):
    """The plane of one parcel's top face, in the same n.X+d form as the deck."""
    n = np.array(gt["deck_plane"]["n"], float)
    d = float(gt["deck_plane"]["d"])
    for p in gt["parcels"]:
        if p["id"] == pid:
            return n, d + p["H_mm"] / 1000.0, p
    raise SystemExit(f"{pid} not in gt_scene.json")


def fit_scale(P, gt, tol=0.12, iters=5, deck_only=False, anchor=None):
    """z_true = alpha * z_pred + beta, fitted on deck (+ floor) points."""
    dn, dd = np.array(gt["deck_plane"]["n"]), gt["deck_plane"]["d"]
    if anchor:
        # Deck + one parcel's top face. Spans the working band instead of
        # reaching 783 mm past it to the floor, where DA3's depth response is
        # no longer affine (measured: slope ~1.10 in the parcel band vs ~2.48
        # deck-to-floor, leaving 157 mm residuals on a single linear fit).
        fn, fd, _ = parcel_plane(gt, anchor)
        tol = min(tol, 0.06)
    else:
        fn, fd = np.array(gt["floor_plane"]["n"]), gt["floor_plane"]["d"]
    a, b = init_scale(P, gt, anchor=anchor)
    nd = nf = 0
    for _ in range(iters):
        z = a * P[:, 2] + b
        Pc = P.copy()
        Pc[:, 2] = z
        zd = rk.plane_z(Pc, dn, dd)
        md = np.abs(z - zd) < tol
        if deck_only:
            mf = np.zeros(len(P), bool)
        else:
            mf = np.abs(z - rk.plane_z(Pc, fn, fd)) < tol
        nd, nf = int(md.sum()), int(mf.sum())
        if nd < 5000 or (not deck_only and nf < 5000):
            return None, None, nd, nf
        m = md | mf
        zt = np.where(md, rk.plane_z(P, dn, dd), rk.plane_z(P, fn, fd))[m]
        A = np.stack([P[m, 2], np.ones(int(m.sum()))], 1)
        sol, *_ = np.linalg.lstsq(A, zt, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        tol = max(0.02, tol * 0.6)
    return a, b, nd, nf


def apply_scale(P, a, b):
    """Depth correction on a cloud: scaling z rescales x,y along the same ray."""
    z = P[:, 2]
    z2 = a * z + b
    s = np.divide(z2, z, out=np.ones_like(z), where=np.abs(z) > 1e-6)
    return np.stack([P[:, 0] * s, P[:, 1] * s, z2], 1)


# ------------------------------------------------ stage 2: held-out parcels
def segment(P, gt, res=0.01, min_pts=2000, band=(0.03, 0.70)):
    Q = rk.to_deck(P, gt)
    m = (Q[:, 2] > band[0]) & (Q[:, 2] < band[1])
    B = Q[m]
    if len(B) < min_pts:
        return []
    u = ((B[:, 0] - B[:, 0].min()) / res).astype(int)
    v = ((B[:, 1] - B[:, 1].min()) / res).astype(int)
    g = np.zeros((u.max() + 3, v.max() + 3), bool)
    g[u, v] = True
    g = ndimage.binary_closing(g, np.ones((3, 3)))
    lab, nl = ndimage.label(g, structure=np.ones((3, 3)))
    cl = lab[u, v]
    out = []
    for k in range(1, nl + 1):
        pts = B[cl == k]
        if len(pts) < min_pts:
            continue
        c2 = pts[:, :2].mean(0)
        A = pts[:, :2] - c2
        _, V = np.linalg.eigh(np.cov(A.T))
        pr = A @ V
        L = np.percentile(pr[:, 1], 99) - np.percentile(pr[:, 1], 1)
        W = np.percentile(pr[:, 0], 99) - np.percentile(pr[:, 0], 1)
        if L > 0.8 or W < 0.15:
            continue
        out.append(dict(x=float(c2[0]), y=float(c2[1]), L=float(L), W=float(W),
                        H=float(np.percentile(pts[:, 2], 97)), n=len(pts)))
    return out


def grade(P, gt, subset, alpha, beta, exclude=(), assoc=0.25):
    n = np.array(gt["deck_plane"]["n"])
    d = gt["deck_plane"]["d"]
    z = -(P @ n + d)
    deck = np.abs(z) < 0.03
    dk = {}
    if deck.sum() > 5000:
        pn, pd = rk.fit_plane_svd(P[deck])
        dk = dict(deck_tilt_mdeg=1000 * float(np.degrees(np.arccos(abs(pn @ n)))),
                  deck_resid_mm=1000 * float((P[deck] @ pn + pd).std()),
                  deck_bias_mm=1000 * float(z[deck].mean()),
                  deck_pts=int(deck.sum()))
    det = segment(P, gt)
    rows = []
    for p in gt["parcels"]:
        if p["id"] in exclude:
            continue
        best, bd = None, assoc
        for c in det:
            dd_ = float(np.hypot(c["x"] - p["x"], c["y"] - p["y"]))
            if dd_ < bd:
                best, bd = c, dd_
        if best is None:
            rows.append(dict(subset=subset, parcel=p["id"], found=0,
                             alpha=round(alpha, 5), beta_mm=round(beta * 1000, 2), **dk))
            continue
        dims = sorted([best["L"] * 1000, best["W"] * 1000])
        gdim = sorted([p["L_mm"], p["W_mm"]])
        rows.append(dict(subset=subset, parcel=p["id"], found=1,
                         dL_mm=round(dims[1] - gdim[1], 1),
                         dW_mm=round(dims[0] - gdim[0], 1),
                         dH_mm=round(best["H"] * 1000 - p["H_mm"], 1),
                         dC_mm=round(1000 * bd, 1),
                         alpha=round(alpha, 5), beta_mm=round(beta * 1000, 2),
                         pts=best["n"], **dk))
    return rows


# ----------------------------------------------------------------- report
def report(rows):
    if not rows:
        print("nothing to report")
        return
    print(f"\n{'subset':20s} {'n':>2s} {'|dL|':>6s} {'|dW|':>6s} {'|dH|':>6s} "
          f"{'dC max':>7s} {'deck res':>9s} {'deck bias':>10s} {'deck tilt':>10s} "
          f"{'alpha':>8s} {'beta mm':>8s}")
    print("-" * 108)
    order = {s: i for i, s in enumerate(rk.SUBSETS)}
    for s in sorted({r["subset"] for r in rows}, key=lambda s: order.get(s, 99)):
        g = [r for r in rows if r["subset"] == s and r.get("found")]
        any_ = [r for r in rows if r["subset"] == s]
        if not g:
            print(f"{s:20s}  0   -- no parcels recovered --")
            continue
        f = lambda k: np.mean([abs(r[k]) for r in g])
        r0 = any_[0]
        print(f"{s:20s} {len(g):2d} {f('dL_mm'):6.1f} {f('dW_mm'):6.1f} {f('dH_mm'):6.1f} "
              f"{max(r['dC_mm'] for r in g):7.1f} "
              f"{r0.get('deck_resid_mm', float('nan')):9.2f} "
              f"{r0.get('deck_bias_mm', float('nan')):10.2f} "
              f"{r0.get('deck_tilt_mdeg', float('nan')):10.1f} "
              f"{r0['alpha']:8.5f} {r0['beta_mm']:8.2f}")
    print("\nper-parcel deltas (mm) -- read these, not the means above")
    print(f"{'subset':20s} {'parcel':7s} {'dL':>7s} {'dW':>7s} {'dH':>7s} {'dCentre':>8s}")
    for r in rows:
        if r.get("found"):
            print(f"{r['subset']:20s} {r['parcel']:7s} {r['dL_mm']:7.1f} "
                  f"{r['dW_mm']:7.1f} {r['dH_mm']:7.1f} {r['dC_mm']:8.1f}")
        else:
            print(f"{r['subset']:20s} {r['parcel']:7s}   NOT RECOVERED")
    print("\ndeck columns come from ~1e5 points and carry the statistical weight.")
    print("parcel columns are a handful of samples: compare per parcel, not in aggregate.")


# ---------------------------------------------------------------- selftest
def selftest(gt, exclude=("P1",)):
    rng = np.random.default_rng(0)
    n, d, ex, ey = rk.deck_frame(gt)
    pts = []
    for plane in ("deck", "floor"):
        pn = np.array(gt[f"{plane}_plane"]["n"])
        pd = gt[f"{plane}_plane"]["d"]
        xy = rng.uniform(-1.7, 1.7, (120000, 2))
        z = rk.plane_z(np.c_[xy, np.zeros(len(xy))], pn, pd)
        pts.append(np.stack([xy[:, 0], xy[:, 1], z], 1))
    for p in gt["parcels"]:
        if p["id"] in exclude:
            continue
        L, W, H = p["L_mm"] / 1000, p["W_mm"] / 1000, p["H_mm"] / 1000
        u = p["x"] + rng.uniform(-W / 2, W / 2, 40000)
        v = p["y"] + rng.uniform(-L / 2, L / 2, 40000)
        pts.append(rk.from_deck(u, v, np.full(len(u), H), gt))
    P = np.vstack(pts) + rng.normal(0, 0.002, (sum(len(x) for x in pts), 3))
    A_TRUE, B_TRUE = 1.0426, -0.00673
    Pd = P.copy()
    Pd[:, 2] = (P[:, 2] - B_TRUE) / A_TRUE
    s = Pd[:, 2] / P[:, 2]
    Pd[:, 0] *= s
    Pd[:, 1] *= s
    a, b, nd, nf = fit_scale(Pd, gt)
    print(f"selftest: injected  alpha={A_TRUE}  beta={B_TRUE*1000:.2f} mm")
    print(f"          recovered alpha={a:.5f}  beta={b*1000:.2f} mm  "
          f"(deck {nd}, floor {nf} pts)")
    ok = abs(a - A_TRUE) < 2e-3 and abs(b - B_TRUE) < 3e-3
    report(grade(apply_scale(Pd, a, b), gt, "SELFTEST", a, b, exclude))
    print("\nNote: the 1/99 percentile extent under-reads a uniformly sampled face")
    print("by ~2%, which is most of the dL/dW seen here. On real data that bias")
    print("largely cancels between scan and prediction -- but only when point")
    print("densities are comparable, so distrust cross-subset dimension deltas")
    print("more than deck deltas.")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="gt_scene.json")
    ap.add_argument("--cloud")
    ap.add_argument("--subset")
    ap.add_argument("--dir")
    ap.add_argument("--csv")
    ap.add_argument("--exclude", default="",
                    help="comma-separated parcel ids to skip, e.g. P1")
    ap.add_argument("--anchor-parcel", default=None, metavar="PID",
                    help="fit alpha/beta on the deck plus this parcel's top "
                         "face, and exclude it from grading. Preferred over "
                         "deck+floor: the floor sits 783 mm outside the working "
                         "band and DA3's depth response is not affine that far "
                         "out. Use the tallest clipped parcel -- it is already "
                         "excluded from grading, so nothing is lost.")
    ap.add_argument("--deck-only", action="store_true",
                    help="fit alpha/beta on the deck alone. ILL-CONDITIONED: a "
                         "single near-normal plane spans almost no depth, so "
                         "alpha and beta trade off freely. Last resort only.")
    ap.add_argument("--no-fit", action="store_true",
                    help="grade the cloud as-is, no alpha/beta")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    exclude = tuple(x.strip() for x in a.exclude.split(",") if x.strip())
    if a.anchor_parcel and a.anchor_parcel not in exclude:
        exclude = exclude + (a.anchor_parcel,)
        print(f"[anchor] {a.anchor_parcel} is the fit target, so it is excluded "
              f"from grading")
    if a.selftest:
        sys.exit(selftest(gt, exclude or ("P1",)))

    jobs = []
    if a.dir:
        for s in rk.SUBSETS:
            for ext in (".ply", ".npy"):
                p = os.path.join(a.dir, s + ext)
                if os.path.exists(p):
                    jobs.append((s, p))
                    break
            else:
                print(f"[missing] {s}")
    elif a.cloud:
        jobs = [(a.subset or os.path.splitext(os.path.basename(a.cloud))[0], a.cloud)]
    else:
        sys.exit("give --cloud or --dir")

    if a.deck_only:
        print("WARNING: --deck-only. The deck spans ~90 mm of depth, so alpha and")
        print("beta are barely separable and will absorb each other. Expect alpha")
        print("to drift and beta to reach tens or hundreds of mm. Compare subsets")
        print("on deck residual and tilt, not on the fitted alpha/beta, and get")
        print("the floor into the clouds if you possibly can.\n")

    rows = []
    for s, p in jobs:
        P = rk.load_cloud(p)
        if a.no_fit:
            al, be = 1.0, 0.0
        else:
            al, be, nd, nf = fit_scale(P, gt, deck_only=a.deck_only,
                                       anchor=a.anchor_parcel)
            if al is None:
                lbl = a.anchor_parcel or "floor"
                print(f"[{s}] scale fit failed (deck {nd}, {lbl} {nf} pts). "
                      f"Is the cloud in the rig frame and uncorrected?")
                continue
        if not a.no_fit:
            P = apply_scale(P, al, be)
        print(f"[{s}] {len(P)} pts   alpha={al:.5f}  beta={be*1000:+.2f} mm")
        rows += grade(P, gt, s, al, be, exclude)
    report(rows)
    if a.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()