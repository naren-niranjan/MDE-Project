#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
probe_tensor_shapes.py -- what shape actually enters the network at each
processing resolution.

Closes: \gap{logged internal tensor shapes above 504\,px}

Section 5.8 states that at a processing resolution of 504 px the tensor
entering the network is 420 x 504, and asks whether that tracks the
processing parameter across the whole ladder. It matters because the
upsampling factor from the returned depth map back to source resolution
-- about 4.9 linear and 24 in area at 504 px -- is quoted in the thesis,
and that factor is only correct if the internal shape follows the
parameter.

The model exposes `input_processor`, which performs exactly the
preparation the network is handed and returns the tensor. This script
calls it at each rung of the ladder and reads the shape off. No forward
pass is run, so it costs seconds and no meaningful GPU memory.

    python3 probe_tensor_shapes.py --image captures/scene_a/center.png

    # sweep the resize method too
    python3 probe_tensor_shapes.py \
        --methods upper_bound_resize lower_bound_resize

Writes tensor_shapes.md (with a LaTeX table ready to paste) and
tensor_shapes.json.
"""

import argparse
import io
import json
import sys

LADDER = [252, 504, 700, 1008, 1400, 2002, 2450]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='frame_0000.png')
    ap.add_argument('--model-dir',
                    default='depth-anything/DA3NESTED-GIANT-LARGE-1.1')
    ap.add_argument('--res', type=int, nargs='+', default=LADDER)
    ap.add_argument('--methods', nargs='+', default=['upper_bound_resize'],
                    help='process_res_method values to sweep')
    ap.add_argument('--out', default='tensor_shapes.md')
    a = ap.parse_args()

    import cv2
    from depth_anything_3.api import DepthAnything3

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit("could not read %s" % a.image)
    H0, W0 = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("source image: %d x %d (WxH)" % (W0, H0), file=sys.stderr)

    print("loading model ...", file=sys.stderr)
    model = DepthAnything3.from_pretrained(a.model_dir)

    results = []
    for meth in a.methods:
        for r in a.res:
            try:
                out = model.input_processor([img_rgb], process_res=r,
                                            process_res_method=meth,
                                            print_progress=False)
            except Exception as e:
                results.append({'method': meth, 'requested': r,
                                'shape': None,
                                'note': "%s: %s" % (type(e).__name__, e)})
                print("  %-20s res %-5d -> FAILED %s"
                      % (meth, r, type(e).__name__), file=sys.stderr)
                continue

            tensor = out[0] if isinstance(out, (tuple, list)) else out
            shp = tuple(tensor.shape)
            h, w = shp[-2], shp[-1]
            results.append({'method': meth, 'requested': r,
                            'shape': list(shp), 'h': h, 'w': w, 'note': ''})
            print("  %-20s res %-5d -> %s   (H=%d W=%d)"
                  % (meth, r, 'x'.join(map(str, shp)), h, w), file=sys.stderr)

    # --- report ------------------------------------------------------
    rpt = ["# Internal tensor shape against processing resolution\n",
           "Source image %d x %d (WxH). Shapes are what `input_processor` "
           "returns, which is what the network is handed.\n" % (W0, H0)]

    for meth in a.methods:
        rows = [x for x in results if x['method'] == meth]
        rpt.append("## `%s`\n" % meth)
        rpt.append("| requested | full shape | H x W | longer side | "
                   "linear to source | area | tracks? |")
        rpt.append("|---|---|---|---|---|---|---|")
        for x in rows:
            if x['shape'] is None:
                rpt.append("| %d | --- | --- | --- | --- | --- | %s |"
                           % (x['requested'], x['note'][:50]))
                continue
            h, w = x['h'], x['w']
            lin = W0 / float(w)
            rpt.append("| %d | %s | %d x %d | %d | %.2f | %.1f | %s |"
                       % (x['requested'], 'x'.join(map(str, x['shape'])),
                          h, w, max(h, w), lin, lin * lin,
                          "yes" if max(h, w) == x['requested'] else "**no**"))
        good = [x['requested'] for x in rows
                if x['shape'] and max(x['h'], x['w']) == x['requested']]
        bad = [x['requested'] for x in rows
               if x['shape'] and max(x['h'], x['w']) != x['requested']]
        rpt.append("\nTracks the parameter at: %s"
                   % (', '.join(map(str, good)) or "none"))
        rpt.append("Does not track at: %s\n"
                   % (', '.join(map(str, bad)) or "none"))

    prim = [x for x in results
            if x['method'] == a.methods[0] and x['shape']]
    rpt += ["## LaTeX table for Section 5.8\n", r"\begin{table}[H]",
            r"  \centering",
            r"  \caption{Tensor shape handed to the network against the",
            r"           requested processing resolution, read from the",
            r"           model's own input processor. The upsampling",
            r"           factor back to the source follows from it.}",
            r"  \label{tab:meth_tensor_shapes}", r"  \small",
            r"  \begin{tabularx}{\textwidth}{@{}Xlrr@{}}", r"    \toprule",
            r"    \thead{Requested (px)} & \thead{Tensor $H \times W$} &"
            r" \thead{Linear} & \thead{Area} \\", r"    \midrule"]
    for x in prim:
        lin = W0 / float(x['w'])
        rpt.append("    %d & $%d \\times %d$ & %.2f & %.1f \\\\"
                   % (x['requested'], x['h'], x['w'], lin, lin * lin))
    rpt += [r"    \bottomrule", r"  \end{tabularx}", r"\end{table}"]

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(rpt) + "\n")
    json.dump({'source': {'width': W0, 'height': H0, 'path': a.image},
               'results': results},
              io.open('tensor_shapes.json', 'w'), indent=1)
    print("\nwritten: %s and tensor_shapes.json" % a.out, file=sys.stderr)


if __name__ == '__main__':
    main()