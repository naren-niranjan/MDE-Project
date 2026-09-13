#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
makefig10.py -- Figure 10: the two holdout regimes on one axis.

Closes: \gap{[FIGURE PLACEHOLDER --- F10 ...]}

Section 6.x assesses the deployed correction under two holdout designs
and reports them in different units: the height folds in absolute RMS
(15.81, 23.97, 37.04 mm) and the belt split as a change in RMS. They
cannot be compared in that form. The belt-split records hold both
`uncorrected_heldout_rms_mm` and `deployed_heldout_rms_mm`, so the
absolute figures exist and the two regimes can share an axis.

The script scans for every JSON carrying both keys, pairs them per
camera, and draws the figure. It also prints what it found so the count
in the text can be checked against the records rather than recalled.

    python3 makefig10.py
    python3 makefig10.py --root ~/Projects/MDE/new --out f10_holdout.pdf

Needs matplotlib. Writes the PDF and holdout_summary.md.
"""

import argparse
import glob
import io
import json
import os
import sys

# Height folds, from Table "loho" in the results chapter.
LOHO = [("deck\n(0 mm)", 15.81), ("riser 1\n(130 mm)", 23.97),
        ("riser 2\n(218 mm)", 37.04)]
REQUIREMENT_MM = 5.0


def collect(root):
    """Every (file, camera, uncorrected, deployed) with both values."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '**', '*.json'),
                                 recursive=True)):
        if any(p in path for p in ('site-packages', 'hf_cache', '/venv')):
            continue
        try:
            with io.open(path, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict) or 'cameras' not in d:
            continue
        cams = d.get('cameras') or {}
        if not isinstance(cams, dict):
            continue
        for cam, rec in cams.items():
            if not isinstance(rec, dict):
                continue
            u = rec.get('uncorrected_heldout_rms_mm')
            c = rec.get('deployed_heldout_rms_mm')
            if u is None or c is None:
                continue
            rows.append({'file': os.path.basename(path), 'camera': cam,
                         'uncorrected': float(u), 'deployed': float(c),
                         'delta': float(c) - float(u),
                         'split': d.get('split'),
                         'common_mode_mm': d.get('common_mode_mm'),
                         'run': d.get('run')})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/Projects/MDE/new'))
    ap.add_argument('--out', default='f10_holdout.pdf')
    a = ap.parse_args()

    rows = collect(a.root)
    if not rows:
        raise SystemExit(
            "no record carried both uncorrected_heldout_rms_mm and "
            "deployed_heldout_rms_mm under %s" % a.root)

    print("belt-split records found: %d" % len(rows), file=sys.stderr)
    for r in rows:
        print("  %-26s %-7s %7.2f -> %7.2f  (%+.2f)"
              % (r['file'], r['camera'], r['uncorrected'],
                 r['deployed'], r['delta']), file=sys.stderr)
    worse = sum(1 for r in rows if r['delta'] > 0)
    print("  worsened in %d of %d" % (worse, len(rows)), file=sys.stderr)

    # ---- figure ----------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    GREY, ORANGE, DKOR = '#4D4D4D', '#F47920', '#D4621A'
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # height folds
    xs_l = [0.6, 1.0, 1.4]
    for x, (lab, v) in zip(xs_l, LOHO):
        ax.plot([x], [v], 'o', ms=7, color=DKOR, zorder=3)
        ax.annotate('%.1f' % v, (x, v), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8, color=DKOR)

    # belt split: a segment per camera-run, uncorrected -> deployed
    x0 = 2.3
    step = 0.28
    for i, r in enumerate(rows):
        x = x0 + i * step
        ax.plot([x, x], [r['uncorrected'], r['deployed']], '-',
                color=GREY, lw=1.0, alpha=0.6, zorder=2)
        ax.plot([x], [r['uncorrected']], 'o', ms=4.5, mfc='white',
                mec=GREY, mew=1.2, zorder=3)
        ax.plot([x], [r['deployed']], 'o', ms=4.5, color=ORANGE, zorder=3)

    ax.axhline(REQUIREMENT_MM, color=DKOR, ls='--', lw=1.0, zorder=1)
    ax.annotate('5 mm requirement', (0.45, REQUIREMENT_MM),
                textcoords='offset points', xytext=(0, 5),
                fontsize=8, color=DKOR)

    ax.set_yscale('log')
    ax.set_ylabel('held-out RMS (mm, log scale)')
    ax.set_xticks([1.0, x0 + step * (len(rows) - 1) / 2.0])
    ax.set_xticklabels(['held out by height\n(board planes)',
                        'held out by belt position\n(scene geometry)'])
    ax.set_xlim(0.3, x0 + step * len(rows) + 0.2)
    ax.tick_params(axis='x', length=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GREY)
    ax.spines['bottom'].set_color(GREY)
    ax.grid(axis='y', color=GREY, alpha=0.15, lw=0.6)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', color=DKOR, ms=7,
               label='height fold, corrected'),
        Line2D([], [], marker='o', ls='', mfc='white', mec=GREY, mew=1.2,
               ms=5, label='belt run, uncorrected'),
        Line2D([], [], marker='o', ls='', color=ORANGE, ms=5,
               label='belt run, corrected')],
        frameon=False, fontsize=8, loc='center right')

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches='tight')
    print("\nwrote %s" % a.out, file=sys.stderr)

    # ---- summary ---------------------------------------------------
    md = ["# Two holdout regimes\n",
          "| file | camera | uncorrected | deployed | change |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %.2f | %.2f | %+.2f |"
                  % (r['file'], r['camera'], r['uncorrected'],
                     r['deployed'], r['delta']))
    lo = min(r['uncorrected'] for r in rows)
    hi = max(r['uncorrected'] for r in rows)
    dlo = min(r['delta'] for r in rows)
    dhi = max(r['delta'] for r in rows)
    md += ["", "- records: %d, worsened in %d" % (len(rows), worse),
           "- uncorrected held-out RMS spans %.1f to %.1f mm" % (lo, hi),
           "- the correction changes it by %+.1f to %+.1f mm" % (dlo, dhi),
           "- the height folds sit at 15.8 to 37.0 mm",
           "",
           "Sentence for the text: on belt geometry the held-out RMS is "
           "%.0f to %.0f\\,mm before correction and the deployed map "
           "changes it by %+.1f to %+.1f\\,mm, worsening it in %d of %d "
           "camera-runs. That is an order of magnitude above the 15.8 to "
           "37.0\\,mm the same correction achieves on the board planes it "
           "was fitted against." % (lo, hi, dlo, dhi, worse, len(rows))]
    io.open('holdout_summary.md', 'w', encoding='utf-8').write(
        "\n".join(md) + "\n")
    print("wrote holdout_summary.md", file=sys.stderr)


if __name__ == '__main__':
    main()