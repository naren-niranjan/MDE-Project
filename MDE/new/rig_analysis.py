import itertools
import json

import numpy as np

EXT = {
 "center": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
 "left": {"R": [[-0.33177218264435215, 0.9433519346977783, 0.0037876279333225473],
                [-0.9429398758154399, -0.33174166557595874, 0.028493120539027608],
                [0.028135554385251144, 0.005881719378499556, 0.9995868126163868]],
          "t": [-0.2687900224925179, 0.11944214023107762, 0.0007179466149592623]},
 "top": {"R": [[-0.021000046563563145, -0.9997070716670727, 0.012031994978987277],
               [0.9996765704347962, -0.021169028463652397, -0.014093500545600256],
               [0.014344077804165026, 0.011732139308381486, 0.9998282874269941]],
         "t": [-0.211958345967181, -0.08762899458407615, 0.542282050183416]},
 "right": {"R": [[-0.9997706370762844, 0.020919809048150828, 0.004586374327211936],
                 [-0.02083751455158237, -0.9996329911212106, 0.0173112983156758],
                 [0.00494684014225191, 0.017211759103898522, 0.999839629201182]],
           "t": [0.007121054690650974, -0.8591693212689412, -0.0002469653253377267]},
}

INTR = {
 "center": {"K": [[3479.6108409722915, 0, 1256.011651620023],
                  [0, 3479.6108409722915, 1026.385601009735], [0, 0, 1]],
            "d": [-0.18913454751610725, 0.36409735490998046,
                  -0.001048795041801043, -4.43051862224039e-05, 0.0]},
 "left": {"K": [[3490.5647448117074, 0, 1224.3657772586398],
                [0, 3490.5647448117074, 1000.7227304000028], [0, 0, 1]],
          "d": [-0.19353107570779585, 0.3767531555069833,
                -0.00047343571784586426, 0.00026105153700397833, 0.0]},
 "right": {"K": [[3479.6160330176517, 0, 1239.1342687611248],
                 [0, 3479.6160330176517, 1033.1956047669926], [0, 0, 1]],
           "d": [-0.1906567744282987, 0.3807197540103391,
                 0.0009438266732555046, 0.0001699352496518714, 0.0]},
 "top": {"K": [[3506.4130429145375, 0, 1205.7977672626478],
               [0, 3506.4130429145375, 1033.8599969851693], [0, 0, 1]],
         "d": [-0.18751078126869603, 0.36789609713623367,
               -0.00019475933467383774, -0.00022405156613224933, 0.0]},
}

BELT = {
 "centre_m": [0.07155121211803056, -0.2292631652062267, 3.1174503790841306],
 "x_axis": [6.120327790895273e-17, 0.9998908721446398, -0.014773076924992018],
 "y_axis": [-0.9999680398608946, -0.00011811004024069639, -0.007994079482899156],
 "length_m": 2.98999951171875, "width_m": 0.8999998779296875,
}

CAMS = ["center", "left", "top", "right"]
W, H = 2448, 2048
MAX_INC = 70.0          # matches da3_stream --max-incidence


def centre(c):
    R = np.asarray(EXT[c]["R"], float)
    t = np.asarray(EXT[c]["t"], float)
    return -R.T @ t


def project(X, c):
    R = np.asarray(EXT[c]["R"], float)
    t = np.asarray(EXT[c]["t"], float)
    K = np.asarray(INTR[c]["K"], float)
    k1, k2, p1, p2, k3 = INTR[c]["d"]
    Xc = X @ R.T + t
    z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
    xn, yn = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    uv = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)
    return uv, Xc[:, 2]


def sees(X, n_surf, c, gate_incidence=True):
    uv, zc = project(X, c)
    ok = ((zc > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
          & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    if gate_incidence:
        r = centre(c)[None, :] - X
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(np.abs(r @ n_surf), 0, 1)))
        ok &= ang <= MAX_INC
    return ok


def cond(sub, C):
    if len(sub) < 3:
        return 0.0
    P = np.array([C[c] for c in sub])
    P = P - P.mean(0)
    s = np.linalg.svd(P, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


C = {c: centre(c) for c in CAMS}

print("camera centres, rig frame = center camera, +Z = center optical axis")
for c in CAMS:
    print(f"  {c:7s} C = ({C[c][0]:+.4f}, {C[c][1]:+.4f}, {C[c][2]:+.4f})   "
          f"|C| = {np.linalg.norm(C[c]):.4f} m")

print("\nbaselines: lateral triangulates, axial conditions the pose prior")
for a, b in itertools.combinations(CAMS, 2):
    v = C[a] - C[b]
    ax, lat = abs(v[2]), float(np.hypot(v[0], v[1]))
    print(f"  {a:7s}-{b:7s} |b|={np.linalg.norm(v):.4f}  lateral={lat:.4f}  "
          f"axial={ax:.4f}  ({100 * ax / np.linalg.norm(v):4.1f}% axial)")

# ---- belt surface -------------------------------------------------------
ex = np.asarray(BELT["x_axis"], float)
ey = np.asarray(BELT["y_axis"], float)
ex /= np.linalg.norm(ex)
ey /= np.linalg.norm(ey)
n = np.cross(ex, ey)
n /= np.linalg.norm(n)
if float(n @ (np.zeros(3) - np.asarray(BELT["centre_m"]))) < 0:
    n = -n                      # point back towards the cameras
cell = 0.02
us = np.arange(-BELT["length_m"] / 2, BELT["length_m"] / 2, cell) + cell / 2
vs = np.arange(-BELT["width_m"] / 2, BELT["width_m"] / 2, cell) + cell / 2
UU, VV = np.meshgrid(us, vs, indexing="ij")
base = (np.asarray(BELT["centre_m"], float)
        + UU.ravel()[:, None] * ex + VV.ravel()[:, None] * ey)
area = cell * cell
print(f"\nbelt {BELT['length_m']:.3f} x {BELT['width_m']:.3f} m, "
      f"{len(base)} cells, {len(base) * area:.2f} m2, deck normal "
      f"{np.round(n, 4)}")

heights = [0.0, 0.10, 0.20, 0.30, 0.40]
subs = [s for k in range(1, 5) for s in itertools.combinations(CAMS, k)]

for gate in (True, False):
    tag = ("with the 70 deg incidence gate the pipeline actually applies"
           if gate else "field of view only, no incidence gate")
    print(f"\ncoverage m2 at >=2 views (>=1 for singles), {tag}")
    hdr = f"  {'subset':26s} {'cond':>7s} " + "".join(
        f"{'h=' + str(int(h * 1000)):>8s}" for h in heights)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for sub in subs:
        row = []
        for h in heights:
            X = base + h * n
            vis = np.stack([sees(X, n, c, gate) for c in sub], 0)
            need = 1 if len(sub) == 1 else 2
            row.append((vis.sum(0) >= need).sum() * area)
        cd = cond(sub, C)
        cs = f"{cd:.4f}" if len(sub) > 2 else "   -   "
        flag = "  <- collinear" if len(sub) > 2 and cd < 0.05 else ""
        print(f"  {'+'.join(sub):26s} {cs:>7s} "
              + "".join(f"{v:8.2f}" for v in row) + flag)