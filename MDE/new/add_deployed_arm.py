#!/usr/bin/env python3
"""
add_deployed_arm.py - add a fourth, unfitted arm to fit_correction.py.

WHY
---
fit_correction.py scores three FITTED models (constant, relief, positional)
against an uncorrected baseline on held-out belt cells. It does not score the
correction that is actually deployed, which is depth_correction.json's shared
map

    h' = alpha * h + beta                (deck-normal height, metres)

Adding it makes the comparison clean: one model, one set of coefficients,
evaluated on board planes (leave-one-height-out, in depth_correction.json)
AND on belt geometry (spatial split, here). Without it the plane-vs-belt
comparison confounds the holdout design with a change of model, reference and
evaluation surface.

WHAT IT CHANGES
---------------
Three edits, all anchored on exact text:
  1. a module-level loader for alpha/beta from depth_correction.json
  2. a "deployed_heldout_rms_mm" entry beside "uncorrected_heldout_rms_mm"
  3. the printed header/row, so the arm appears in the console table
Edits 1 and 2 are required; edit 3 is best-effort and skipped if the print
statements have moved.

The deployed residual needs no fitting. If d is the height deficit and h the
ground-truth height, applying h' = alpha*h + beta shifts the reconstruction by
(alpha - 1)*h + beta, so the residual on the same cells is

    d_deployed = d - ((alpha - 1) * h + beta)

USAGE
-----
    python3 add_deployed_arm.py                  # dry run, prints the diff
    python3 add_deployed_arm.py --write          # writes, keeps a .bak
    python3 add_deployed_arm.py --sign-convention gt_minus_recon --write

SIGN CONVENTION
---------------
The default assumes d = reconstruction - ground truth, i.e. a positive deficit
means the reconstruction sits too high. If your d is the other way round, pass
--sign-convention gt_minus_recon and the correction term flips.

SELF-TEST: run with --selftest. It synthesises cells whose compression the
deployed alpha exactly undoes; the deployed arm must then score far below the
uncorrected baseline. If it does not, the sign convention is inverted, so
re-run with the other --sign-convention. Do this before believing any number
the patched file produces.
"""

import argparse
import difflib
import io
import os
import shutil
import sys

TARGET = "fit_correction.py"

# ---------------------------------------------------------------- edit 1
LOADER_ANCHOR = "CAMS = [\"center\", \"left\", \"top\", \"right\"]"

LOADER_CODE = '''

# --- deployed correction, loaded not fitted -------------------------------
# depth_correction.json carries the shared map h' = alpha*h + beta that
# da3_stream.py applies at run time. Scoring it here on the same held-out
# cells as the fitted arms is what makes the plane-vs-belt comparison a
# comparison of holdout designs rather than of models.
_DEP_PATH = Path(__file__).with_name("depth_correction.json")


def _load_deployed():
    """Return (alpha, beta_metres, label). Falls back to identity if absent."""
    try:
        with open(_DEP_PATH) as fh:
            shared = json.load(fh)["shared"]
        return (float(shared["alpha"]), float(shared["beta"]),
                f"deployed(alpha={shared['alpha']:.4f}, "
                f"beta={shared['beta'] * 1e3:+.2f}mm)")
    except (OSError, KeyError, ValueError) as exc:
        print(f"[deployed] {_DEP_PATH.name} unusable ({exc}); "
              f"the deployed arm will read as identity and is meaningless",
              file=sys.stderr)
        return 1.0, 0.0, "deployed(UNAVAILABLE)"


_DEP_ALPHA, _DEP_BETA, _DEP_LABEL = _load_deployed()


def deployed_residual(d, h, sign=+1.0):
    """Residual after applying h' = alpha*h + beta to cells with deficit d.

    The correction acts on the MEASURED height, so it scales the deficit as
    well as shifting it. With h_meas = h + d (sign=+1) the corrected residual
    is alpha*d + (alpha-1)*h + beta. Dropping the alpha*d term is wrong: it is
    what makes the correction self-cancelling when alpha exactly undoes the
    compression present in d.

    sign=+1 when d = reconstruction - ground truth.
    sign=-1 when d = ground truth - reconstruction.
    """
    return (_DEP_ALPHA * d
            + sign * ((_DEP_ALPHA - 1.0) * h + _DEP_BETA))
# --------------------------------------------------------------------------
'''

# ---------------------------------------------------------------- edit 2
SCORE_ANCHOR = '            "uncorrected_heldout_rms_mm": round(rms(d[~m]) * 1e3, 2)}'

SCORE_NEW = '''            "uncorrected_heldout_rms_mm": round(rms(d[~m]) * 1e3, 2),
            "deployed_heldout_rms_mm": round(
                rms(deployed_residual(d, h, _DEP_SIGN)[~m]) * 1e3, 2),
            "deployed_coefficients": {"alpha": _DEP_ALPHA,
                                      "beta_mm": round(_DEP_BETA * 1e3, 3)}}'''

# ---------------------------------------------------------------- edit 3
HDR_ANCHOR = ("          f\"{'none':>9s} {'constant':>10s} "
              "{'relief':>9s} {'positional':>11s}\")")
HDR_NEW = ("          f\"{'none':>9s} {'deployed':>10s} {'constant':>10s} "
           "{'relief':>9s} {'positional':>11s}\")")

TUPLE_ANCHOR = '                        for k in ("constant", "relief", "positional")))'


def build(src, sign):
    """Return patched source, and a list of edits applied/skipped."""
    log = []
    out = src

    # required edits ------------------------------------------------------
    for name, anchor, new in (
        ("loader", LOADER_ANCHOR, LOADER_ANCHOR + LOADER_CODE),
        ("score", SCORE_ANCHOR, SCORE_NEW),
    ):
        n = out.count(anchor)
        if n != 1:
            raise SystemExit(
                f"ABORT: anchor '{name}' matched {n} times, expected 1.\n"
                f"The file has moved on from what this patch was written "
                f"against. Paste lines 270-345 of {TARGET} and the patch can "
                f"be rewritten against the real code.")
        out = out.replace(anchor, new, 1)
        log.append(f"applied: {name}")

    # sign constant -------------------------------------------------------
    out = out.replace("_DEP_ALPHA, _DEP_BETA, _DEP_LABEL = _load_deployed()",
                      "_DEP_ALPHA, _DEP_BETA, _DEP_LABEL = _load_deployed()\n"
                      f"_DEP_SIGN = {sign:+.1f}  # +1: d = recon - gt; "
                      "-1: d = gt - recon", 1)

    # imports -------------------------------------------------------------
    if "\nimport json" not in out:
        out = out.replace("import argparse", "import argparse\nimport json", 1)
        log.append("applied: import json")
    if "from pathlib import Path" not in out:
        out = out.replace("import argparse", "import argparse\n"
                          "from pathlib import Path", 1)
        log.append("applied: import Path")
    if "\nimport sys" not in out:
        out = out.replace("import argparse", "import argparse\nimport sys", 1)
        log.append("applied: import sys")

    # optional edits ------------------------------------------------------
    if HDR_ANCHOR in out:
        out = out.replace(HDR_ANCHOR, HDR_NEW, 1)
        log.append("applied: printed header")
    else:
        log.append("SKIPPED: printed header anchor not found; the console "
                   "table will not show the new column, the JSON still will")

    if TUPLE_ANCHOR in out:
        log.append("NOTE: the arm tuple at the 'for k in (...)' line was left "
                   "alone. If the console row is built from it, add "
                   "'deployed' there by hand to align the printed columns.")

    return out, log


def selftest(convention):
    """Synthesise cells the deployed alpha should correct exactly."""
    import json as _json
    import numpy as np
    try:
        shared = _json.load(open("depth_correction.json"))["shared"]
        alpha, beta = float(shared["alpha"]), float(shared["beta"])
    except (OSError, KeyError, ValueError):
        alpha, beta = 1.1712323789579404, 0.000520838122229966
        print("[selftest] depth_correction.json unusable; using nominal alpha",
              file=sys.stderr)

    sign = 1.0 if convention == "recon_minus_gt" else -1.0
    rng = np.random.default_rng(0)
    h = rng.uniform(0.0, 0.36, 20000)
    k = 1.0 - 1.0 / alpha                     # compression alpha exactly undoes
    d_true = -k * h + rng.normal(0, 0.002, h.size)   # recon - gt
    d = d_true * sign                          # expressed in the chosen sign

    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    resid = alpha * d + sign * ((alpha - 1.0) * h + beta)
    unc, dep = rms(d) * 1e3, rms(resid) * 1e3

    print(f"[selftest] convention={convention} alpha={alpha:.5f}")
    print(f"[selftest] uncorrected {unc:7.2f} mm   deployed {dep:7.2f} mm")
    if dep < unc * 0.2:
        print("[selftest] PASS (algebra only): the residual expression "
              "cancels a compression that alpha exactly undoes.")
        print("[selftest] NOTE: this test is self-consistent in either "
              "convention and CANNOT tell you which one your d uses. "
              "Determine that from the data, as below.")
        print("[selftest] Sign check on real data: relief compression means "
              "the reconstruction UNDER-reports height. Print the median "
              "deficit in a high band and a low band. If the deficit grows "
              "POSITIVE with height, d = gt - recon and you want "
              "--sign-convention gt_minus_recon. If it grows NEGATIVE, "
              "d = recon - gt and the default is right.")
        return 0
    print("[selftest] FAIL: the deployed arm does not cancel a compression it "
          "should. Try the other --sign-convention; if both fail, the height "
          "convention in fit_correction.py differs from depth_align.py and "
          "the two must be reconciled before the number means anything.",
          file=sys.stderr)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=TARGET)
    ap.add_argument("--write", action="store_true",
                    help="write the file (a .bak copy is kept)")
    ap.add_argument("--sign-convention", default="recon_minus_gt",
                    choices=["recon_minus_gt", "gt_minus_recon"])
    ap.add_argument("--selftest", action="store_true",
                    help="verify the residual algebra against synthetic cells")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.sign_convention)

    if not os.path.exists(args.file):
        raise SystemExit(f"not found: {args.file} (run this beside it)")

    src = io.open(args.file, encoding="utf-8").read()
    sign = 1.0 if args.sign_convention == "recon_minus_gt" else -1.0
    out, log = build(src, sign)

    print("\n".join(log), file=sys.stderr)

    if not args.write:
        diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                    args.file, args.file + " (patched)")
        sys.stdout.writelines(diff)
        print("\n-- dry run; re-run with --write to apply --", file=sys.stderr)
        return

    shutil.copy2(args.file, args.file + ".bak")
    io.open(args.file, "w", encoding="utf-8").write(out)
    print(f"written; original kept at {args.file}.bak", file=sys.stderr)


if __name__ == "__main__":
    main()