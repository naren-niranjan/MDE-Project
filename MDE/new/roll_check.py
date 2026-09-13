import glob, os, re, subprocess, numpy as np
NEW = os.path.expanduser('~/Projects/MDE/new')
# 1. every place in the live code that could flip or rotate a frame (capture side included)
print('=== flip / rotate / roll handling in all .py files')
pat = r'ReverseX|ReverseY|flip|rot90|rotate|\[::-1|roll|upside|180'
for p in sorted(glob.glob(NEW + '/**/*.py', recursive=True)):
    for i, line in enumerate(open(p, errors='ignore'), 1):
        if re.search(pat, line) and not line.strip().startswith('#'):
            print('%s:%d: %s' % (os.path.relpath(p, NEW), i, line.rstrip()[:140]))
# 2. the offline flip logic for reference
print('\n=== da3_offline.py 270-345')
L = open(NEW + '/da3_offline.py').read().splitlines()
for i in range(269, min(345, len(L))): print(i + 1, L[i])
# 3. which script and imports the streaming run uses
print('\n=== imports in da3_stream.py')
for line in open(NEW + '/da3_stream.py'):
    if line.startswith(('import', 'from')): print('  ', line.rstrip())
 
# 4. empirical test on one streaming capture: does the left depth line up with the
#    left image as saved, or with the image rotated by 180 degrees?
import cv2
runs = [p for p in glob.glob(os.path.expanduser('~/Projects/MDE') + '/**/live_20260831_113807*', recursive=True) if os.path.isdir(p)]
print('\n=== run folders:', runs)
if runs:
    run = runs[0]
    files = sorted(glob.glob(run + '/**/*', recursive=True))
    for p in files[:60]: print('   ', os.path.relpath(p, run), os.path.getsize(p))
    depths = {}
    for p in files:
        if p.endswith('.npy'):
            try: depths[os.path.relpath(p, run)] = np.load(p)
            except Exception: pass
        elif p.endswith('.npz'):
            try:
                z = np.load(p)
                for k in z.files: depths[os.path.relpath(p, run) + ':' + k] = z[k]
            except Exception: pass
    views = {}
    for name, a in depths.items():
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[0] > 100 and a.dtype.kind == 'f': views[name] = a
        elif a.ndim == 3 and a.shape[0] <= 8 and a.shape[1] > 100 and a.dtype.kind == 'f':
            for i in range(a.shape[0]): views['%s[%d]' % (name, i)] = a[i]
    images = {os.path.relpath(p, run): p for p in files if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))}
    print('depth-like arrays:', {k: v.shape for k, v in views.items()})
    print('images:', list(images)[:12])
    def grad(a):
        a = np.nan_to_num(a.astype(np.float32), nan=float(np.nanmedian(a)))
        g = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
        return (g - g.mean()) / (g.std() + 1e-9)
    print('\ncorrelation of edge maps: identity / rotated 180 (the larger one is the true orientation)')
    for iname, ipath in list(images.items())[:8]:
        im = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        for dname, d in views.items():
            gi = grad(cv2.resize(im, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA))
            gd = grad(d)
            c0 = float((gi * gd).mean()); c1 = float((gi * gd[::-1, ::-1]).mean())
            print('  %-40s %-40s %+.3f / %+.3f %s' % (iname[:40], dname[:40], c0, c1, '<- ROTATED' if c1 > c0 else ''))