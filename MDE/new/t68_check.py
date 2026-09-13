"""Follow-up on Table 6.8: the three captures found by find_t68.py.
Prints only what is needed: log rows at full precision, processing resolution,
correction on/off, per-camera extrinsic roll, boxes.json, and how da3_stream.py
handles the inverted left camera. Output is small; run as
    python3 t68_check.py > t68b.txt 2>&1
"""
import csv, glob, json, os, re, struct
from datetime import datetime
import numpy as np

MDE = os.path.expanduser('~/Projects/MDE')
NEW = MDE + '/new'
CAPS = [('live_20260831_114629', '0'), ('live_20260831_114629', '1'), ('live_20260831_115241', '2')]
CAMS = ('center', 'left', 'right', 'top')

# 1. log rows for the three captures, every column, as stored
print('=== 1. boxes_log.csv rows (as stored)')
for run in sorted({r for r, _ in CAPS}):
    p = f'{NEW}/runs/{run}/boxes_log.csv'
    rows = list(csv.DictReader(open(p)))
    print(f'\n{run}  columns: {list(rows[0].keys())}')
    for r in rows:
        if (run, r.get('capture')) in CAPS:
            print('  ' + ','.join(r.values()))

# 2. frames.jsonl records (what the streamer logged per frame)
print('\n=== 2. frames.jsonl')
for run in sorted({r for r, _ in CAPS}):
    p = f'{NEW}/runs/{run}/frames.jsonl'
    for i, line in enumerate(open(p) if os.path.exists(p) else []):
        print(f'  {run} line {i}: {line.strip()[:900]}')

def png_size(p):
    with open(p, 'rb') as fh:
        fh.read(16); w, h = struct.unpack('>II', fh.read(8))
    return w, h

def roll(E):
    R = np.asarray(E, float)[:3, :3]
    return np.degrees(np.arctan2(R[0, 1], R[0, 0]))

# 3. per frame: resolution, correction state, extrinsics, layers, boxes
for run, cap in CAPS:
    fd = f'{NEW}/runs/{run}/frame_{int(cap):05d}'
    print(f'\n=== 3. {run} capture {cap} -> {os.path.basename(fd)}')
    if not os.path.isdir(fd):
        print('   frame folder missing:', sorted(os.listdir(os.path.dirname(fd)))); continue
    print('   written', datetime.fromtimestamp(os.path.getmtime(fd + '/depth_center.npy')).isoformat())
    E0 = {}
    for c in CAMS:
        d = np.load(f'{fd}/depth_{c}.npy'); r = np.load(f'{fd}/depth_raw_{c}.npy')
        K = np.load(f'{fd}/K_{c}.npy'); E = np.load(f'{fd}/E_{c}.npy'); E0[c] = E
        diff = (d.astype(np.float64) - r.astype(np.float64)) * 1000.0
        ok = np.isfinite(diff)
        q = np.percentile(diff[ok], [5, 50, 95]) if ok.any() else [np.nan] * 3
        print(f'   {c:6s} depth {d.shape} {d.dtype} raw {r.dtype} proc_png {png_size(f"{fd}/proc_{c}.png")} '
              f'| fx {K[0,0]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f} '
              f'| depth-raw mm p5/p50/p95 {q[0]:+.1f}/{q[1]:+.1f}/{q[2]:+.1f} max|.| {np.nanmax(np.abs(diff)):.1f} '
              f'| E roll {roll(E):+.1f} deg, t {np.round(np.asarray(E)[:3, 3], 4).tolist()}')
    Rc = np.asarray(E0['center'])[:3, :3]
    for c in ('left', 'right', 'top'):
        Rrel = np.asarray(E0[c])[:3, :3] @ Rc.T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        z = np.degrees(np.arccos(np.clip(np.asarray(E0[c])[2, :3] @ Rc[2], -1, 1)))
        print(f'   {c} vs center: total rotation {ang:.1f} deg, optical-axis angle {z:.1f} deg')
    for name in ('layers.json', 'segmentation/boxes.json'):
        p = f'{fd}/{name}'
        if os.path.exists(p):
            print(f'   {name}: {json.dumps(json.load(open(p)), separators=(",", ":"))[:6000]}')

# 4. extrinsics: identical across frames (fixed prior) or re-estimated each frame?
print('\n=== 4. E change between the three captures (deg, mm)')
Es = {c: [np.load(f'{NEW}/runs/{r}/frame_{int(k):05d}/E_{c}.npy') for r, k in CAPS] for c in CAMS}
for c in CAMS:
    a = Es[c]
    for j in (1, 2):
        Rrel = np.asarray(a[j])[:3, :3] @ np.asarray(a[0])[:3, :3].T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        dt = np.linalg.norm(np.asarray(a[j])[:3, 3] - np.asarray(a[0])[:3, 3]) * 1000
        print(f'   {c:6s} capture0 -> {j}: {ang:.3f} deg, {dt:.1f} mm')

# 5. how the streamer treats the inverted left camera, and which config it loads
print('\n=== 5. da3_stream.py: imports, upright/flip, correction, resolution')
pat = r'^\s*(import|from)\s|da3_offline|upright|roll_thresh|flip|\[::-1|rot90|process_res|depth_correction|deck_height|correction'
for i, line in enumerate(open(NEW + '/da3_stream.py', errors='ignore'), 1):
    if re.search(pat, line):
        print(f'   {i}: {line.rstrip()[:160]}')
for p in sorted(glob.glob(NEW + '/*.sh') + glob.glob(NEW + '/*.service') + glob.glob(NEW + '/*stream*.json')):
    print('\n   ---', p)
    print('   ' + open(p, errors='ignore').read()[:1500].replace('\n', '\n   '))