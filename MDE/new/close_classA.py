#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
close_classA.py -- harvest the records the thesis is missing and emit
LaTeX you can paste straight in.

Each block below targets one \gap{} marker in MTPv6.37. The script reads
only files that already exist; it computes nothing new and invents
nothing. Where a record cannot be found it says so and names what it
looked for, so the marker can be converted to a stated limitation
instead of being left open.

    cd ~/Projects/MDE/new
    python3 close_classA.py                 # writes gap_tables.tex
    python3 close_classA.py --root /other/path

Stdlib only. Safe to re-run.
"""

import argparse
import csv
import glob
import io
import json
import os
import re
import sys

FOUND, MISSING = [], []


def load(path):
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def esc(s):
    return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def block(title, marker, body):
    rule = "%% " + "-" * 66
    return ("\n" + rule + "\n%%%% %s\n%%%% closes: \\gap{%s}\n" % (title, marker)
            + rule + "\n" + body + "\n")


# ------------------------------------------------------------------ 1
def run_register(root, out):
    r"""\gap{run register: subset, resolution, arm, correction state,
    gauge outcome, and exclusion reason} and \gap{per-run arm membership}"""
    p = os.path.join(root, 'gauge_probe.json')
    d = load(p)
    if d is None:
        MISSING.append(("run register", p)); return
    rows = d if isinstance(d, list) else d.get('runs') or d.get('entries') or []
    if not rows:
        MISSING.append(("run register (file present, no rows)", p)); return
    lines = [r"\begin{longtable}{@{}lllrrl@{}}",
             r"  \caption{Run register. Every configuration attempted, with"
             r" the gauge outcome that decided admission.}\\",
             r"  \label{tab:res_run_register} \\",
             r"  \toprule",
             r"  \thead{Run} & \thead{Subset} & \thead{Res} & "
             r"\thead{$n$ cams} & \thead{Deck scale} & \thead{Outcome} \\",
             r"  \midrule \endfirsthead",
             r"  \toprule \thead{Run} & \thead{Subset} & \thead{Res} & "
             r"\thead{$n$ cams} & \thead{Deck scale} & \thead{Outcome} \\",
             r"  \midrule \endhead"]
    admitted = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get('run') or r.get('name') or ''
        subset = r.get('subset') or re.sub(r'_res\d+$', '', str(name))
        res = r.get('res') or r.get('resolution') or ''
        m = re.search(r'res(\d+)', str(name))
        if not res and m:
            res = m.group(1)
        ncam = r.get('n_cams') or len(str(subset).split('+'))
        sc = r.get('scale_deck')
        if sc is None:
            outcome, scs = 'excluded: no gauge', '---'
        else:
            ok = abs(float(sc) - 1.0) < 0.10
            admitted += ok
            outcome = 'admitted' if ok else 'excluded: deck scale'
            scs = '%.3f' % float(sc)
        lines.append("  %s & %s & %s & %s & %s & %s \\\\" %
                     (esc(name)[:28], esc(subset)[:26], esc(res),
                      ncam, scs, outcome))
    lines += [r"  \bottomrule", r"\end{longtable}"]
    lines.append("%% %d entries, %d admitted at the 10 per cent deck-scale bound"
                 % (len(rows), admitted))
    out.append(block("Run register", "run register: subset, resolution, arm, "
                     "correction state, gauge outcome, exclusion reason",
                     "\n".join(lines)))
    FOUND.append("run register (%d runs)" % len(rows))


# ------------------------------------------------------------------ 2
def correction_coefficients(root, out):
    r"""\gap{... per-camera field coefficients}"""
    p = os.path.join(root, 'depth_correction.json')
    d = load(p)
    if d is None:
        MISSING.append(("correction coefficients", p)); return
    views = d.get('views') or {}
    shared = d.get('shared') or {}
    lines = [r"\begin{table}[H]", r"  \centering",
             r"  \caption{Deployed correction. The shared map is"
             r" $h' = \alpha h + \beta$ on height above the deck; the"
             r" per-camera terms are the deviations from it.}",
             r"  \label{tab:meth_correction_coeffs}", r"  \small",
             r"  \begin{tabularx}{\textwidth}{@{}Xrrrr@{}}", r"    \toprule",
             r"    \thead{Camera} & \thead{$\alpha$} & \thead{$\beta$ (mm)}"
             r" & \thead{$d_{\mathrm{cam}}$ (m)} & \thead{$k$} \\",
             r"    \midrule"]
    for name, rec in views.items():
        if not isinstance(rec, dict):
            continue
        a = rec.get('alpha'); b = rec.get('beta'); dc = rec.get('d_cam')
        k = (1.0 - 1.0 / a) if a else None
        lines.append("    %s & %s & %s & %s & %s \\\\" % (
            esc(name),
            '%.4f' % a if a else '---',
            '%+.2f' % (b * 1e3) if b is not None else '---',
            '%.4f' % dc if dc is not None else '---',
            '%.3f' % k if k else '---'))
    if shared.get('alpha'):
        lines.append(r"    \midrule")
        lines.append("    shared & %.4f & %+.2f & --- & %.3f \\\\" % (
            shared['alpha'], shared.get('beta', 0) * 1e3,
            1.0 - 1.0 / shared['alpha']))
    lines += [r"    \bottomrule", r"  \end{tabularx}", r"\end{table}"]
    unc = d.get('height_uncertainty') or {}
    if unc:
        lines.append("%% height uncertainty: %s" % json.dumps(unc))
    out.append(block("Correction coefficients",
                     "height-to-depth coefficient derivation, solve "
                     "constraints, and per-camera field coefficients",
                     "\n".join(lines)))
    FOUND.append("correction coefficients (%d cameras)" % len(views))


# ------------------------------------------------------------------ 3
def segmentation_parameters(root, out):
    r"""\gap{complete segmentation parameter table}"""
    cand = sorted(glob.glob(os.path.join(root, 'runs/**/segmentation/boxes.json'),
                            recursive=True))
    src, params = None, None
    for c in reversed(cand):
        d = load(c)
        if d and d.get('parameters'):
            src, params = c, d['parameters']; break
    if params is None:
        d = load(os.path.join(root, 'live_diagnostics.json')) or {}
        seg = (d.get('segmentation') or {}).get('parameters')
        if seg:
            src, params = 'live_diagnostics.json', seg
    if params is None:
        MISSING.append(("segmentation parameters",
                        "runs/**/segmentation/boxes.json -> parameters"))
        return
    lines = [r"\begin{table}[H]", r"  \centering",
             r"  \caption{Segmentation parameters as recorded in the"
             r" capture that produced the reported results.}",
             r"  \label{tab:meth_seg_params}", r"  \small",
             r"  \begin{tabularx}{\textwidth}{@{}Xl@{}}", r"    \toprule",
             r"    \thead{Parameter} & \thead{Value} \\", r"    \midrule"]
    for k in sorted(params):
        v = params[k]
        if isinstance(v, (list, tuple)):
            v = ', '.join(str(x) for x in v)
        lines.append("    %s & %s \\\\" % (esc(k), esc(v)))
    lines += [r"    \bottomrule", r"  \end{tabularx}", r"\end{table}",
              "%% source: %s" % src]
    out.append(block("Segmentation parameters",
                     "complete segmentation parameter table", "\n".join(lines)))
    FOUND.append("segmentation parameters (%d, from %s)" % (len(params), src))


# ------------------------------------------------------------------ 4
def interview_disagreement(root, out):
    r"""\gap{per-parcel inter-view disagreement distribution}"""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, 'runs/**/segmentation/boxes.json'),
                              recursive=True)):
        d = load(f)
        for b in (d or {}).get('boxes', []):
            t = b.get('top_face', {})
            cons = b.get('view_consensus') or b.get('consensus') or {}
            spread = (cons.get('height_spread_mm') if isinstance(cons, dict)
                      else None)
            if spread is None:
                spread = t.get('per_view_spread_mm')
            if spread is not None:
                rows.append((os.path.basename(os.path.dirname(
                    os.path.dirname(f))), t.get('height_above_conveyor_m'),
                    spread, t.get('inlier_fraction')))
    if not rows:
        MISSING.append(("inter-view disagreement per parcel",
                        "boxes.json -> view_consensus / per_view_spread_mm"))
        return
    lines = [r"%% per-parcel inter-view disagreement, one row per detection",
             r"\begin{longtable}{@{}lrrr@{}}",
             r"  \caption{Inter-view disagreement per detected parcel.}\\",
             r"  \label{tab:res_interview} \\", r"  \toprule",
             r"  \thead{Capture} & \thead{Height (mm)} & "
             r"\thead{Spread (mm)} & \thead{Inlier fr.} \\",
             r"  \midrule \endhead"]
    for cap, h, s, inl in rows:
        lines.append("  %s & %s & %s & %s \\\\" % (
            esc(cap)[:26],
            '%.1f' % (h * 1e3) if h else '---',
            '%.1f' % s, '%.2f' % inl if inl else '---'))
    lines += [r"  \bottomrule", r"\end{longtable}"]
    out.append(block("Inter-view disagreement",
                     "per-parcel inter-view disagreement distribution",
                     "\n".join(lines)))
    FOUND.append("inter-view disagreement (%d detections)" % len(rows))


# ------------------------------------------------------------------ 5
def timing_repeats(root, out):
    r"""\gap{per-repeat raw timings, and whether the tabled statistic is a
    mean or a median}"""
    for name in ('bench_prior.csv', 'bench_noprior.csv', 'bench_noprior_quad.csv'):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            MISSING.append(("timing repeats", p)); continue
        with io.open(p, encoding='utf-8') as f:
            rd = list(csv.DictReader(f))
        if not rd:
            continue
        cols = rd[0].keys()
        per_rep = [c for c in cols if re.search(r'(rep|run|trial)_?\d+', c)]
        stat = [c for c in cols if re.search(r'mean|median|p50|p95|std', c)]
        out.append(block("Timing columns in %s" % name,
                         "per-repeat raw timings; mean or median",
                         "%% columns present: %s\n"
                         "%% per-repeat columns: %s\n"
                         "%% statistic columns: %s\n"
                         "%% -> if the per-repeat list is empty the raw values\n"
                         "%%    were never written; re-run the sweep with them\n"
                         "%%    logged, or state this as a limitation."
                         % (', '.join(cols),
                            ', '.join(per_rep) or 'NONE',
                            ', '.join(stat) or 'NONE')))
        FOUND.append("%s: %d rows, per-repeat columns %s"
                     % (name, len(rd), 'yes' if per_rep else 'NO'))


# ------------------------------------------------------------------ 6
def calibration_record(root, out):
    r"""\gap{per-campaign calibration record}"""
    dirs = sorted(glob.glob(os.path.expanduser('~/Projects/Calibration*/results')))
    if not dirs:
        MISSING.append(("calibration records", "~/Projects/Calibration*/results"))
        return
    lines = [r"\begin{table}[H]", r"  \centering",
             r"  \caption{Calibration campaigns and their residuals.}",
             r"  \label{tab:res_calib_record}", r"  \small",
             r"  \begin{tabularx}{\textwidth}{@{}Xllr@{}}", r"    \toprule",
             r"    \thead{Campaign} & \thead{Camera} & \thead{Lens} & "
             r"\thead{RMS (px)} \\", r"    \midrule"]
    n = 0
    for dd in dirs:
        camp = dd.split('/')[-2]
        for f in sorted(glob.glob(os.path.join(dd, 'intrinsics_*.json'))):
            d = load(f) or {}
            cam = os.path.basename(f)[11:-5]
            lines.append("    %s & %s & %s & %s \\\\" % (
                esc(camp), esc(cam), esc(d.get('lens_id', '---')),
                '%.3f' % d['rms_reprojection_error_px']
                if d.get('rms_reprojection_error_px') is not None else '---'))
            n += 1
        e = load(os.path.join(dd, 'extrinsics.json')) or {}
        for cam, rec in e.items():
            if isinstance(rec, dict) and rec.get('rms_error_px') is not None:
                lines.append("    %s & %s (extr.) & %s & %.3f \\\\" % (
                    esc(camp), esc(cam), esc(rec.get('lens_id', '---')),
                    rec['rms_error_px']))
                n += 1
    lines += [r"    \bottomrule", r"  \end{tabularx}", r"\end{table}"]
    if n:
        out.append(block("Calibration record", "per-campaign calibration record",
                         "\n".join(lines)))
        FOUND.append("calibration record (%d rows across %d campaigns)"
                     % (n, len(dirs)))
    else:
        MISSING.append(("calibration record (dirs found, no residuals)",
                        ', '.join(dirs)))


# ------------------------------------------------------------------ 7
def screening_platform(root, out):
    r"""\gap{GPU variant, driver, and memory statistic for the screening
    platform} -- reports what to run if not already recorded."""
    for pat in ('**/screening*.json', '**/benchmark*.json', '**/models*.json'):
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            d = load(f)
            if d and any(k in json.dumps(d)[:4000].lower()
                         for k in ('gpu', 'device_name', 'driver')):
                out.append(block("Screening platform",
                                 "GPU variant, driver, and memory statistic",
                                 "%% found in %s -- transcribe by hand:\n%%   %s"
                                 % (f, json.dumps(d)[:600])))
                FOUND.append("screening platform hints in %s" % f)
                return
    MISSING.append(("screening platform record",
                    "no screening JSON with GPU fields; run "
                    "`nvidia-smi --query-gpu=name,driver_version,memory.total "
                    "--format=csv` on the workstation and paste"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/Projects/MDE/new'))
    ap.add_argument('--out', default='gap_tables.tex')
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        raise SystemExit("not a directory: %s" % a.root)

    out = ["%% Generated by close_classA.py -- paste each block into the\n"
           "%% section named in its comment, then delete the \\gap{} marker."]
    for fn in (run_register, correction_coefficients, segmentation_parameters,
               interview_disagreement, timing_repeats, calibration_record,
               screening_platform):
        try:
            fn(a.root, out)
        except Exception as e:
            MISSING.append((fn.__name__, "failed: %r" % e))

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(out) + "\n")

    print("\nCLOSED (%d):" % len(FOUND), file=sys.stderr)
    for f in FOUND:
        print("   %s" % f, file=sys.stderr)
    print("\nNOT FOUND (%d) -- convert these to stated limitations:"
          % len(MISSING), file=sys.stderr)
    for what, where in MISSING:
        print("   %-42s %s" % (what, where), file=sys.stderr)
    print("\nwritten: %s" % a.out, file=sys.stderr)


if __name__ == '__main__':
    main()