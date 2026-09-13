"""Last three open points for Table 6.8 and Section 5.2.2.
Run in ~/Projects/MDE/new:   python3 t68_check2.py > t68c.txt 2>&1
"""
import glob, json, re, collections

# 1. per-parcel inlier fraction and coverage, as the table's last column
print('=== 1. top-face inlier fraction per box (run, capture, id, inlier_fraction, inliers, coverage)')
for r, f in [('live_20260831_114629', 0), ('live_20260831_114629', 1), ('live_20260831_115241', 2)]:
    for b in json.load(open(f'runs/{r}/frame_{f:05d}/segmentation/boxes.json'))['boxes']:
        t = b['top_face']
        print(r[-6:], f, b['id'], round(t['inlier_fraction'], 4), t['inliers'], round(t['coverage'], 4),
              [v.get('band_points') for v in b['view_consensus'].get('per_view', [])])

# 2. what sync_spread_ms and forced_trigger mean in the streamer
print('\n=== 2. sync_spread_ms / forced_trigger in da3_stream.py')
L = open('da3_stream.py', errors='ignore').read().splitlines()
hits = [i for i, l in enumerate(L) if re.search(r'sync_spread|forced_trigger|Action0|ActionCommand|TriggerSoftware', l)]
shown = set()
for i in hits:
    for j in range(max(0, i - 4), min(len(L), i + 4)):
        if j not in shown:
            print(f'{j + 1}: {L[j][:150]}'); shown.add(j)
    print('   ...')

# 3. did any retained offline run use --upright (and so rotate views before inference)?
print('\n=== 3. upright / roll_thresh recorded in run metadata')
cnt = collections.Counter()
for p in glob.glob('runs*/**/*.json', recursive=True):
    try:
        s = open(p, errors='ignore').read()
    except Exception:
        continue
    for m in re.findall(r'"(upright|roll_thresh)"\s*:\s*([^,}\]]+)', s):
        cnt[(p.split('/')[1] if '/' in p else p, m[0], m[1].strip())] += 1
for k, v in sorted(cnt.items()):
    print('  ', k, v)
if not cnt:
    print('   none found in runs*/**/*.json')