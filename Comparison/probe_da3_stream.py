#!/usr/bin/env python3
"""
probe_da3_stream.py — report the internal structure of da3_stream.py so an
offline replay runner can be written against its real API.

    python3 probe_da3_stream.py --src da3_stream.py > da3_api.txt

Parses the source with ast; it does NOT import or execute anything, so it is
safe to run with no cameras attached.

Paste the output back and the offline runner can call the same functions the
live loop calls -- same model load, same rectification, same back-projection,
same npy layout -- instead of reimplementing them and quietly diverging.
"""
import argparse
import ast
import re

# what the offline path has to hook into
INTEREST = {
    "model": r"DepthAnything|depth_anything|from_pretrained|torch\.hub|load_model|"
             r"\.to\(|autocast|torch\.load|build_model|DA3|Da3",
    "inference": r"\.infer|forward\(|predict|with torch\.no_grad|__call__|"
                 r"process_res|patch",
    "rectify": r"undistort|initUndistortRectify|remap|rect_mode|getOptimalNew|"
               r"rectif",
    "backproject": r"unproject|back_?project|depth_to|pixel_to|reproject|"
                   r"o3d\.geometry\.PointCloud|create_from_depth",
    "correction": r"correction|affine|alpha|beta|depth_align",
    "save": r"np\.save|savez|write_point_cloud|save_npy|\.ply",
    "camera": r"Aravis|aravis|Vimba|vmb|pylon|Spinnaker|harvester|GenTL|"
              r"grab|acquisition|Stream",
}


def sig(node):
    a = node.args
    parts = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    parts += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{node.name}({', '.join(parts)})"


def first_doc(node):
    d = ast.get_docstring(node)
    return d.strip().splitlines()[0][:100] if d else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="da3_stream.py")
    a = ap.parse_args()
    src = open(a.src).read()
    lines = src.splitlines()
    tree = ast.parse(src)

    print(f"=== {a.src}: {len(lines)} lines ===\n")

    print("--- imports ---")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            print("  import " + ", ".join(x.name for x in n.names))
        elif isinstance(n, ast.ImportFrom):
            print(f"  from {n.module} import "
                  + ", ".join(x.name for x in n.names)[:90])

    print("\n--- top-level functions ---")
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"  L{n.lineno:<5d} {sig(n)}")
            if first_doc(n):
                print(f"           # {first_doc(n)}")

    print("\n--- classes ---")
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            print(f"  L{n.lineno:<5d} class {n.name}"
                  f"({', '.join(ast.unparse(b) for b in n.bases)})")
            if first_doc(n):
                print(f"           # {first_doc(n)}")
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    print(f"             L{m.lineno:<5d} .{sig(m)}")

    print("\n--- module-level constants ---")
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    try:
                        v = ast.unparse(n.value)[:80]
                    except Exception:
                        v = "?"
                    print(f"  L{n.lineno:<5d} {t.id} = {v}")

    print("\n--- lines by role ---")
    for role, pat in INTEREST.items():
        rx = re.compile(pat)
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines)
                if rx.search(l) and not l.strip().startswith("#")]
        print(f"\n  [{role}]  {len(hits)} hits")
        for ln, txt in hits[:14]:
            print(f"    L{ln:<5d} {txt[:110]}")
        if len(hits) > 14:
            print(f"    ... {len(hits)-14} more")

    print("\n--- what writes the npy arrays ---")
    for i, l in enumerate(lines):
        if re.search(r"np\.savez?|save_npy", l):
            lo, hi = max(0, i - 4), min(len(lines), i + 5)
            print(f"\n  around L{i+1}:")
            for j in range(lo, hi):
                print(f"    {j+1:<5d}{'>' if j == i else ' '} {lines[j][:110]}")

    print("\n--- main entry ---")
    for i, l in enumerate(lines):
        if "__main__" in l:
            print(f"  L{i+1}: {l.strip()}")


if __name__ == "__main__":
    main()