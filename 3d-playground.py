import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # (import side-effect enables 3D plots)

# ──────────────────────────────────────────────────────────────────────────────
# 3D TRANSFORM MATRICES
# ──────────────────────────────────────────────────────────────────────────────
def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def scale3d(sx, sy, sz):
    return np.diag([sx, sy, sz])

# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────
BASE = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=float)

EDGES = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────
verts = BASE.copy()
ax_deg = ay_deg = az_deg = 0.0
sx = sy = sz = 1.0
tx = ty = tz = 0.0

def apply_transform(P):
    rx, ry, rz = np.deg2rad(ax_deg), np.deg2rad(ay_deg), np.deg2rad(az_deg)
    M = rot_x(rx) @ rot_y(ry) @ rot_z(rz) @ scale3d(sx, sy, sz)
    return (P @ M.T) + np.array([tx, ty, tz])

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE
# ──────────────────────────────────────────────────────────────────────────────
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.34)  # more room for sliders

T = apply_transform(verts)
lines = []
for a, b in EDGES:
    ln, = ax.plot(*zip(T[a], T[b]), lw=2)
    lines.append(ln)

ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
ax.set_title("3D: rotate/scale/translate with sliders • Drag mouse to orbit/zoom")

# ──────────────────────────────────────────────────────────────────────────────
# SLIDERS
# ──────────────────────────────────────────────────────────────────────────────
def add_slider(y, label, vmin, vmax, vinit):
    axsl = plt.axes([0.12, y, 0.76, 0.03])
    return Slider(axsl, label, vmin, vmax, valinit=vinit)

# Rotation sliders (degrees)
s_rx = add_slider(0.29, "rotX (°)", -180, 180, 0)
s_ry = add_slider(0.25, "rotY (°)", -180, 180, 0)
s_rz = add_slider(0.21, "rotZ (°)", -180, 180, 0)

# Scale sliders
s_sx = add_slider(0.17, "scaleX", 0.2, 3.0, 1.0)
s_sy = add_slider(0.13, "scaleY", 0.2, 3.0, 1.0)
# (keep Z-scale fixed at 1.0; add one if you want)
# s_sz = add_slider(0.09, "scaleZ", 0.2, 3.0, 1.0)

# Translation sliders
s_tx = add_slider(0.09, "transX", -4.0, 4.0, 0.0)
s_ty = add_slider(0.05, "transY", -4.0, 4.0, 0.0)
s_tz = add_slider(0.01, "transZ", -4.0, 4.0, 0.0)

def on_change(_):
    global ax_deg, ay_deg, az_deg, sx, sy, sz, tx, ty, tz
    ax_deg, ay_deg, az_deg = s_rx.val, s_ry.val, s_rz.val
    sx, sy = s_sx.val, s_sy.val
    # sz stays 1.0 unless you add s_sz; then: sz = s_sz.val
    tx, ty, tz = s_tx.val, s_ty.val, s_tz.val

    T = apply_transform(verts)
    for (a, b), ln in zip(EDGES, lines):
        ln.set_data_3d(*zip(T[a], T[b]))
    fig.canvas.draw_idle()

for s in (s_rx, s_ry, s_rz, s_sx, s_sy, s_tx, s_ty, s_tz):
    s.on_changed(on_change)

# ──────────────────────────────────────────────────────────────────────────────
# BASIC VERTEX DRAGGING
# ──────────────────────────────────────────────────────────────────────────────
drag_idx = None

def project_points(P):
    xy = []
    for p in P:
        x2, y2, _ = ax.proj_transform(p[0], p[1], p[2], ax.get_proj())
        xy.append([x2, y2])
    return np.array(xy)

def nearest_vertex(event, pts2d, tol=10):
    d = np.hypot(pts2d[:, 0] - event.x, pts2d[:, 1] - event.y)
    i = np.argmin(d)
    return int(i) if d[i] < tol else None

def on_press(event):
    global drag_idx
    if event.inaxes != ax:
        return
    pts2d = project_points(apply_transform(verts))
    drag_idx = nearest_vertex(event, pts2d)

def on_release(event):
    global drag_idx
    drag_idx = None

def on_move(event):
    if drag_idx is None or event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    z0 = verts[drag_idx, 2]
    verts[drag_idx] = [event.xdata, event.ydata, z0]
    on_change(None)

cid1 = fig.canvas.mpl_connect("button_press_event", on_press)
cid2 = fig.canvas.mpl_connect("button_release_event", on_release)
cid3 = fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()
