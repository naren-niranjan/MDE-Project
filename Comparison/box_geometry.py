#!/usr/bin/env python3
"""
box_geometry.py

The parcels of a segmentation result, as geometry: wireframe cuboids, pick
markers and axes in the world frame, written beside the cloud so a 3D viewer
shows the same boxes the camera overlays draw in pixels.

Why this is a module and not part of the driver
-----------------------------------------------
The boxes are wanted in two places: offline, when a stored capture is being
tuned, and live, when da3_stream.py writes a capture. Duplicating the drawing
code in both is how the two drift apart, and a cuboid that differs between the
offline check and the streamed output is worse than no cuboid at all. So the
construction lives here and both callers import it.

Files written
-------------
  boxes_mesh.ply    the cuboids as a triangle mesh, edges as thin bars
  boxes_lines.ply   the same cuboids as a line set, a few kilobytes
  scene_boxed.ply   the coloured cloud and the boxes as one point cloud

The mesh is the one to load beside segmented.ply in Open3D or CloudCompare. The
line set is what to keep in a repository. The combined cloud is for viewers
that will not render a mesh and a cloud together, and it is by far the most
expensive of the three, because the mesh has to be sampled densely enough for
the edges to read as lines.

Cost in a streaming loop
------------------------
Building the mesh for a dozen parcels is a few tens of milliseconds. Sampling
and writing the combined cloud is hundreds, and scales with the point count of
the scene rather than the parcel count. For a live run use

    --geom --geom-no-combined --geom-every 10

which writes the mesh and the line set on every tenth capture and never builds
the combined cloud. The per-call cost is returned so it can go into the same
diagnostic JSON as the rest of the stage timings rather than being guessed at.

Geometry of a cuboid
--------------------
The top face is the fitted rectangle. The four vertical edges run down the
conveyor normal by the parcel's recorded height, which is measured to the deck
for a parcel standing on it and inferred from the parcel beneath for a stacked
one. That is the construction box_segment.draw_overlays uses, so a cuboid that
looks wrong here looks wrong there for the same reason.

Keep this file beside box_segment.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d is required")

try:
    import box_segment as bs
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(f"cannot import box_segment.py: {exc}")


# Status colours matching the BEV render: green for a pickable parcel, orange
# for one held back. RGB here because Open3D takes RGB, while the drawing code
# in box_segment takes BGR.
PICKABLE_RGB = (0.00, 0.86, 0.00)
HOLD_RGB = (1.00, 0.55, 0.00)
BELT_RGB = (0.66, 0.66, 0.66)
CAMERA_RGB = (0.95, 0.95, 0.35)


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class GeomParams:
    """Everything the export needs, in metres."""

    enabled: bool = False
    every: int = 1

    edge_radius: float = 0.003
    edge_step: float = 0.002

    mesh: bool = True
    lines: bool = True
    combined: bool = False

    markers: bool = True
    axes: bool = True
    joints: bool = True
    pick_ring: bool = False
    approach: bool = False
    belt_wire: bool = True
    cameras: bool = False

    pick_radius: float = 0.040
    approach_offset: float = 0.100


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Attach the geometry options to an existing parser.

    Call this after box_segment.add_arguments so --seg-pick-radius and
    --seg-approach-offset are already defined; the ring and the stand-off are
    drawn at those values rather than at a second set that could disagree with
    the overlays.
    """
    g = ap.add_argument_group("box geometry")
    g.add_argument("--geom", action="store_true",
                   help="on every segmented capture, also write the parcels as "
                        "3D geometry: boxes_mesh.ply, boxes_lines.ply and, if "
                        "asked for, scene_boxed.ply")
    g.add_argument("--geom-every", type=int, default=1,
                   help="write the geometry every Nth segmented capture. The "
                        "mesh is cheap and the combined cloud is not, so raise "
                        "this rather than dropping the export when the loop "
                        "will not carry it")
    g.add_argument("--geom-edge-radius", type=float, default=0.003,
                   help="half-thickness of the drawn box edges. It is "
                        "geometry, not a line width, so an edge keeps its size "
                        "in the scene however far the viewer is")
    g.add_argument("--geom-edge-step", type=float, default=0.002,
                   help="point spacing when the mesh is sampled into the "
                        "combined cloud")
    g.add_argument("--geom-no-mesh", dest="geom_mesh", action="store_false",
                   default=True, help="skip boxes_mesh.ply")
    g.add_argument("--geom-no-lines", dest="geom_lines", action="store_false",
                   default=True, help="skip boxes_lines.ply")
    g.add_argument("--geom-combined", action="store_true",
                   help="also write scene_boxed.ply, the cloud and the boxes "
                        "in one file. This is the expensive one: the mesh has "
                        "to be sampled to point density first, so it costs "
                        "hundreds of milliseconds and scales with the size of "
                        "the scene rather than the number of parcels")
    g.add_argument("--geom-no-markers", dest="geom_markers",
                   action="store_false", default=True,
                   help="do not mark the pick point")
    g.add_argument("--geom-no-axes", dest="geom_axes", action="store_false",
                   default=True,
                   help="do not draw the in-plane X axis of each parcel")
    g.add_argument("--geom-no-joints", dest="geom_joints",
                   action="store_false", default=True,
                   help="do not close the cuboid corners with balls")
    g.add_argument("--geom-pick-ring", action="store_true",
                   help="draw the suction footprint at --seg-pick-radius in "
                        "the plane of each top face, so it is visible in 3D "
                        "whether the cup lands on the parcel or over an edge")
    g.add_argument("--geom-approach", action="store_true",
                   help="draw the stand-off vector at --seg-approach-offset")
    g.add_argument("--geom-no-belt", dest="geom_belt_wire",
                   action="store_false", default=True,
                   help="do not outline the belt footprint")
    g.add_argument("--geom-cameras", action="store_true",
                   help="mark the optical centres, which shows at a glance "
                        "which view a grazing face was carried by")


def params_from_args(args: argparse.Namespace,
                     seg_params=None) -> GeomParams:
    """Build the geometry parameters, taking the pick figures from the
    segmentation so the ring and the stand-off match the overlays."""
    pick = getattr(seg_params, "pick_radius", None)
    if pick is None:
        pick = getattr(args, "seg_pick_radius", 0.040)
    stand = getattr(seg_params, "approach_offset", None)
    if stand is None:
        stand = getattr(args, "seg_approach_offset", 0.100)
    return GeomParams(
        enabled=bool(getattr(args, "geom", False)),
        every=max(1, int(getattr(args, "geom_every", 1))),
        edge_radius=float(getattr(args, "geom_edge_radius", 0.003)),
        edge_step=float(getattr(args, "geom_edge_step", 0.002)),
        mesh=bool(getattr(args, "geom_mesh", True)),
        lines=bool(getattr(args, "geom_lines", True)),
        combined=bool(getattr(args, "geom_combined", False)),
        markers=bool(getattr(args, "geom_markers", True)),
        axes=bool(getattr(args, "geom_axes", True)),
        joints=bool(getattr(args, "geom_joints", True)),
        pick_ring=bool(getattr(args, "geom_pick_ring", False)),
        approach=bool(getattr(args, "geom_approach", False)),
        belt_wire=bool(getattr(args, "geom_belt_wire", True)),
        cameras=bool(getattr(args, "geom_cameras", False)),
        pick_radius=float(pick),
        approach_offset=float(stand),
    )


def replace_enabled(gp: GeomParams, enabled: bool) -> GeomParams:
    """A copy with the export switched on or off.

    The offline driver turns it on unconditionally, because a stored capture is
    opened precisely to be looked at and nothing is being timed. The streaming
    driver leaves it under --geom, where the cost has to be paid every frame.
    """
    return dataclasses.replace(gp, enabled=bool(enabled))


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def rot_from_z(d: np.ndarray) -> np.ndarray:
    """Rotation taking +Z onto the given direction.

    Open3D builds cylinders and arrows along +Z, so every bar drawn here needs
    this. The antiparallel case is handled separately because the cross product
    vanishes there and Rodrigues' formula divides by its norm.
    """
    d = np.asarray(d, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(z @ d)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def bar(a, b, radius: float, colour, resolution: int = 8):
    """One edge, as a cylinder from a to b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius),
                                                  height=length,
                                                  resolution=int(resolution))
    m.rotate(rot_from_z(d), center=(0.0, 0.0, 0.0))
    m.translate((a + b) / 2.0)
    m.paint_uniform_color(colour)
    return m


def ball(centre, radius: float, colour, resolution: int = 6):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius),
                                                resolution=int(resolution))
    m.translate(np.asarray(centre, dtype=float))
    m.paint_uniform_color(colour)
    return m


def arrow(origin, direction, length: float, radius: float, colour):
    direction = np.asarray(direction, dtype=float)
    if float(np.linalg.norm(direction)) < 1e-9 or length <= 0:
        return None
    shaft = max(length * 0.72, 1e-4)
    head = max(length - shaft, 1e-4)
    m = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=float(radius),
                                               cone_radius=float(radius) * 2.2,
                                               cylinder_height=shaft,
                                               cone_height=head,
                                               resolution=10)
    m.rotate(rot_from_z(direction), center=(0.0, 0.0, 0.0))
    m.translate(np.asarray(origin, dtype=float))
    m.paint_uniform_color(colour)
    return m


def ring(centre, x_axis, y_axis, radius: float, tube: float, colour,
         segments: int = 32):
    """The suction footprint, drawn in the plane of the top face."""
    ang = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    pts = (np.asarray(centre, dtype=float)
           + radius * (np.outer(np.cos(ang), x_axis)
                       + np.outer(np.sin(ang), y_axis)))
    mesh = o3d.geometry.TriangleMesh()
    for i in range(len(pts)):
        seg = bar(pts[i], pts[(i + 1) % len(pts)], tube, colour, resolution=5)
        if seg is not None:
            mesh += seg
    return mesh


def box_corners(record, n_conv: np.ndarray):
    """The eight corners of one parcel: fitted top face, extruded down.

    A cuboid that appears to float is reporting an inferred height taken from
    the parcel beneath it, which boxes.json flags as height_is_inferred. It is
    not a drawing fault.
    """
    top = np.asarray(record["top_face"]["corners_m"], dtype=float)
    height = float(record["dimensions_m"]["height"])
    base = top - np.asarray(n_conv, dtype=float) * height
    return top, base


def wire_cuboid(top, base, colour, radius: float, joints: bool = True):
    mesh = o3d.geometry.TriangleMesh()
    for i in range(4):
        for a, b in ((top[i], top[(i + 1) % 4]),
                     (base[i], base[(i + 1) % 4]),
                     (top[i], base[i])):
            seg = bar(a, b, radius, colour)
            if seg is not None:
                mesh += seg
    if joints:
        for c in np.vstack([top, base]):
            mesh += ball(c, radius, colour)
    return mesh


def wire_rectangle(corners, colour, radius: float):
    mesh = o3d.geometry.TriangleMesh()
    corners = np.asarray(corners, dtype=float)
    for i in range(len(corners)):
        seg = bar(corners[i], corners[(i + 1) % len(corners)], radius, colour)
        if seg is not None:
            mesh += seg
    return mesh


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_box_mesh(result, gp: GeomParams,
                   cam_centres=None) -> o3d.geometry.TriangleMesh:
    """Every parcel in the result, as one mesh in the world frame.

    Colours follow box_segment.BOX_COLOURS, so a parcel is the same colour in
    the PLY, in the BEV render and in the camera overlays. The pick marker is
    coloured by pickability instead, because that is the property worth reading
    first and it should not have to be looked up.
    """
    conv = result.get("conveyor") or {}
    n_conv = np.asarray(conv.get("normal", [0.0, 0.0, -1.0]), dtype=float)
    mesh = o3d.geometry.TriangleMesh()

    if gp.belt_wire and conv.get("belt"):
        mesh += wire_rectangle(conv["belt"]["corners_m"], BELT_RGB,
                               gp.edge_radius * 0.6)

    for b in result.get("boxes", []):
        bgr = bs.BOX_COLOURS[b["id"] % len(bs.BOX_COLOURS)]
        colour = tuple(float(c) / 255.0 for c in bgr[::-1])
        top, base = box_corners(b, n_conv)
        mesh += wire_cuboid(top, base, colour, gp.edge_radius, gp.joints)

        centre = np.asarray(b["position_m"], dtype=float)
        R = np.asarray(b["pose_world"], dtype=float)[:3, :3]
        status = PICKABLE_RGB if b["pickable"] else HOLD_RGB

        if gp.markers:
            mesh += ball(centre, gp.edge_radius * 2.0, status, resolution=8)
            if gp.pick_ring and gp.pick_radius > 0:
                mesh += ring(centre, R[:, 0], R[:, 1], gp.pick_radius,
                             gp.edge_radius * 0.8, status)
        if gp.axes:
            a = arrow(centre, R[:, 0],
                      float(b["dimensions_m"]["length"]) * 0.45,
                      gp.edge_radius * 0.9, colour)
            if a is not None:
                mesh += a
        if gp.approach:
            # The approach vector points into the surface, so the stand-off is
            # drawn along its negation: that is where the tool waits.
            a = arrow(centre, -np.asarray(b["approach_vector"], dtype=float),
                      gp.approach_offset, gp.edge_radius * 0.7, status)
            if a is not None:
                mesh += a

    if gp.cameras and cam_centres is not None:
        for c in cam_centres:
            mesh += ball(c, gp.edge_radius * 6.0, CAMERA_RGB, resolution=8)

    if len(mesh.vertices):
        mesh.compute_vertex_normals()
    return mesh


def build_line_set(result, gp: GeomParams) -> o3d.geometry.LineSet:
    """The same cuboids as a line set, for viewers that render edges natively."""
    conv = result.get("conveyor") or {}
    n_conv = np.asarray(conv.get("normal", [0.0, 0.0, -1.0]), dtype=float)
    pts, lines, cols = [], [], []

    def add(pairs, colour, block):
        base_i = len(pts)
        pts.extend(np.asarray(block, dtype=float).tolist())
        for i, j in pairs:
            lines.append([base_i + i, base_i + j])
            cols.append(colour)

    edges = ([(i, (i + 1) % 4) for i in range(4)]
             + [(4 + i, 4 + (i + 1) % 4) for i in range(4)]
             + [(i, 4 + i) for i in range(4)])

    if gp.belt_wire and conv.get("belt"):
        add([(i, (i + 1) % 4) for i in range(4)], list(BELT_RGB),
            conv["belt"]["corners_m"])

    for b in result.get("boxes", []):
        bgr = bs.BOX_COLOURS[b["id"] % len(bs.BOX_COLOURS)]
        colour = [float(c) / 255.0 for c in bgr[::-1]]
        top, base = box_corners(b, n_conv)
        add(edges, colour, np.vstack([top, base]))

    ls = o3d.geometry.LineSet()
    if pts:
        ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=float))
    return ls


def combine(cloud, mesh, step: float):
    """The scene and the boxes as one point cloud.

    The mesh is sampled rather than reduced to its vertices, so the edges keep
    their thickness and stay legible at whatever point size a viewer picks.
    Density comes from the surface area, so a scene with many parcels does not
    come out sparser than one with few.
    """
    out = o3d.geometry.PointCloud(cloud)
    if not len(mesh.triangles):
        return out
    area = float(mesh.get_surface_area())
    n = int(max(2000, min(4_000_000, area / max(step, 1e-4) ** 2)))
    out += mesh.sample_points_uniformly(number_of_points=n,
                                        use_triangle_normal=False)
    return out


# --------------------------------------------------------------------------
# the call the drivers make
# --------------------------------------------------------------------------

def write_geometry(result, out_dir, gp: GeomParams, cloud=None,
                   cam_centres=None, capture_index=None):
    """Write the geometry for one segmentation result.

    Returns the paths written and the stage timings in milliseconds, so the
    cost lands in the same diagnostic record as the rest of the pipeline rather
    than being estimated afterwards. Returns empty dictionaries when the export
    is off, when this capture is not on the --geom-every stride, or when there
    is nothing to draw, so the caller needs no guard of its own.
    """
    if not gp.enabled or not result.get("ok") or not result.get("boxes"):
        return {}, {}
    if gp.every > 1 and capture_index is not None:
        try:
            if int(capture_index) % gp.every:
                return {}, {}
        except (TypeError, ValueError):
            pass  # a named capture, so the stride does not apply

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, timings = {}, {}

    t0 = time.perf_counter()
    mesh = build_box_mesh(result, gp, cam_centres)
    timings["geom_build"] = (time.perf_counter() - t0) * 1e3

    if gp.mesh and len(mesh.triangles):
        t0 = time.perf_counter()
        path = out_dir / "boxes_mesh.ply"
        o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
        written["boxes_mesh_ply"] = path
        timings["geom_write_mesh"] = (time.perf_counter() - t0) * 1e3

    if gp.lines:
        t0 = time.perf_counter()
        ls = build_line_set(result, gp)
        if len(ls.lines):
            path = out_dir / "boxes_lines.ply"
            try:
                o3d.io.write_line_set(str(path), ls, write_ascii=False)
                written["boxes_lines_ply"] = path
            except Exception as exc:  # noqa: BLE001
                print(f"[warn ] line set not written: {exc}")
        timings["geom_write_lines"] = (time.perf_counter() - t0) * 1e3

    if gp.combined:
        t0 = time.perf_counter()
        base = cloud if cloud is not None else bs.coloured_cloud(result)
        if len(base.points):
            path = out_dir / "scene_boxed.ply"
            o3d.io.write_point_cloud(str(path), combine(base, mesh,
                                                        gp.edge_step),
                                     write_ascii=False)
            written["scene_boxed_ply"] = path
        timings["geom_combined"] = (time.perf_counter() - t0) * 1e3

    timings = {k: round(v, 2) for k, v in timings.items()}
    return written, timings