#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_da3_api.py -- print what the loaded model actually exposes.

The tensor-shape probe guessed at an entry point and guessed wrong. This
prints the real one: every public callable on the model, its signature,
and the first line of its docstring. It loads the model and calls
nothing.

    python3 inspect_da3_api.py

Then tell probe_tensor_shapes.py which method and argument to use:

    python3 probe_tensor_shapes.py --method infer_one --res-kwarg resolution
"""

import inspect
import sys

MODEL_DIR = 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'


def main():
    import torch
    from depth_anything_3.api import DepthAnything3

    print("loading %s ..." % MODEL_DIR, file=sys.stderr)
    model = DepthAnything3.from_pretrained(MODEL_DIR)

    print("\ntype: %s" % type(model).__mro__[0])
    print("bases: %s\n" % ', '.join(c.__name__ for c in type(model).__mro__[1:4]))

    print("=" * 70)
    print("PUBLIC CALLABLES")
    print("=" * 70)
    for name in sorted(dir(model)):
        if name.startswith('_'):
            continue
        try:
            attr = getattr(model, name)
        except Exception:
            continue
        if not callable(attr):
            continue
        try:
            sig = str(inspect.signature(attr))
        except (ValueError, TypeError):
            sig = '(signature unavailable)'
        doc = (inspect.getdoc(attr) or '').strip().splitlines()
        first = doc[0][:70] if doc else ''
        print("  %-26s %s" % (name, sig))
        if first:
            print("  %-26s   %s" % ('', first))

    print("\n" + "=" * 70)
    print("ANYTHING TAKING A RESOLUTION-LIKE ARGUMENT")
    print("=" * 70)
    hits = 0
    for name in sorted(dir(model)):
        if name.startswith('_'):
            continue
        try:
            attr = getattr(model, name)
            if not callable(attr):
                continue
            params = inspect.signature(attr).parameters
        except Exception:
            continue
        matched = [p for p in params
                   if any(k in p.lower()
                          for k in ('res', 'size', 'scale', 'width', 'shape'))]
        if matched:
            hits += 1
            print("  %-26s -> %s" % (name, ', '.join(matched)))
    if not hits:
        print("  none found; the resolution may be set on the object or in a\n"
              "  processor/transform rather than passed per call.")

    print("\n" + "=" * 70)
    print("RESOLUTION-LIKE ATTRIBUTES ON THE OBJECT")
    print("=" * 70)
    for name in sorted(dir(model)):
        if name.startswith('_'):
            continue
        if not any(k in name.lower()
                   for k in ('res', 'size', 'shape', 'patch', 'input')):
            continue
        try:
            v = getattr(model, name)
        except Exception:
            continue
        if callable(v):
            continue
        print("  %-26s = %r" % (name, v))

    if hasattr(model, 'config'):
        print("\n" + "=" * 70)
        print("CONFIG")
        print("=" * 70)
        cfg = model.config
        d = cfg if isinstance(cfg, dict) else getattr(cfg, '__dict__', {})
        for k in sorted(d):
            if any(t in k.lower() for t in
                   ('res', 'size', 'patch', 'input', 'image')):
                print("  %-26s = %r" % (k, d[k]))

    print("\nnothing was executed; no inference was run.", file=sys.stderr)


if __name__ == '__main__':
    main()