import csv, glob, os, itertools, json, re
from datetime import datetime
ROOT = os.path.expanduser('~/Projects/MDE')
T_H = dict(A=170.1, B=201.1, C=222.5, D=218.7, E=309.3, F=354.1)
T_R = dict(A=16.1, B=14.1, C=14.4, D=33.7, E=25.3, F=47.9)

def f(v):
    try: return float(v)
    except: return None
def ts(v):
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y%m%d_%H%M%S'):
        try: return datetime.strptime(v.strip()[:26], fmt).timestamp()
        except: pass
    try: return float(v)
    except: return None

caps = {}
logs = glob.glob(ROOT + '/**/boxes_log*.csv', recursive=True)
print('logs:', len(logs))
for p in logs:
    for r in csv.DictReader(open(p)):
        k = (p, r.get('capture', ''))
        caps.setdefault(k, []).append(r)
print('captures:', len(caps))

def hts(rows, col): return [f(r.get(col)) for r in rows if f(r.get(col)) is not None]
cand = []
for k, rows in caps.items():
    if len(rows) < 6: continue
    ok = False
    for col in ('height_mm', 'top_height_mm'):
        h = hts(rows, col)
        if h and all(min(abs(x - t) for x in h) < 40 for t in T_H.values()): ok = True
    if ok: cand.append((k, rows, ts(rows[0].get('timestamp', '')) ))
print('captures with six plausible heights:', len(cand))

res = []
for a, b, c in itertools.combinations(cand, 3):
    t = [x[2] for x in (a, b, c)]
    if all(v is not None for v in t) and max(t) - min(t) > 1800: continue
    tracks = []
    for r0 in a[1]:
        x0, y0 = f(r0['x_m']), f(r0['y_m'])
        grp = [r0]
        for other in (b, c):
            best = min(other[1], key=lambda r: (f(r['x_m']) - x0)**2 + (f(r['y_m']) - y0)**2)
            if ((f(best['x_m']) - x0)**2 + (f(best['y_m']) - y0)**2) ** .5 > 0.02: break
            grp.append(best)
        if len(grp) == 3: tracks.append(grp)
    if len(tracks) < 6: continue
    for col in ('height_mm', 'top_height_mm'):
        stats = []
        for g in tracks:
            h = [f(r[col]) for r in g]
            if None in h: continue
            stats.append((sum(h) / 3, max(h) - min(h), g))
        score, used, match = 0, set(), {}
        for lab in T_H:
            j = min((i for i in range(len(stats)) if i not in used),
                    key=lambda i: abs(stats[i][0] - T_H[lab]) + abs(stats[i][1] - T_R[lab]), default=None)
            if j is None: score = 1e9; break
            used.add(j); match[lab] = stats[j]
            score += abs(stats[j][0] - T_H[lab]) + abs(stats[j][1] - T_R[lab])
        res.append((score, col, (a, b, c), match))
res.sort(key=lambda z: z[0])
for score, col, trip, match in res[:3]:
    print('\n=== score %.1f  column %s' % (score, col))
    for k, rows, t in trip:
        print('  ', k[0], 'capture', k[1], rows[0].get('timestamp'))
    for lab, (m, rng, g) in match.items():
        L = [f(r['length_mm']) for r in g]; W = [f(r['width_mm']) for r in g]
        X = [f(r['x_m']) * 1000 for r in g]; Y = [f(r['y_m']) * 1000 for r in g]
        print('  %s mean %.1f range %.1f | L %.0f W %.0f x %.0f y %.0f | inl %s tilt %s rms %s' % (
            lab, m, rng, max(L) - min(L), max(W) - min(W), max(X) - min(X), max(Y) - min(Y),
            [r.get('face_inliers') for r in g], [r.get('tilt_deg') for r in g], [r.get('residual_rms_mm') for r in g]))
if res:
    dirs = {os.path.dirname(k[0]) for k, _, _ in res[0][2]}
    for k, _, _ in res[0][2]:
        if k[1]: dirs |= {p for p in glob.glob(ROOT + '/**/*' + k[1] + '*', recursive=True) if os.path.isdir(p)}
    dirs = sorted(dirs)
    for d in dirs:
        print('\n--- files in', d)
        for p in sorted(glob.glob(d + '/*'))[:40]: print('   ', os.path.basename(p))
        for p in glob.glob(d + '/**/*.json', recursive=True):
            if 'boxes' in os.path.basename(p): continue
            try: j = json.load(open(p))
            except Exception: continue
            def walk(o, pre=''):
                if isinstance(o, dict):
                    for kk, vv in o.items(): walk(vv, pre + '.' + str(kk))
                elif re.search(r'res|size|shape|width|height|process|stream|grid|model|prior', pre, re.I) and not isinstance(o, list):
                    print('   ', os.path.basename(p), pre, '=', o)
            walk(j)