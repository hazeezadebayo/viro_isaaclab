# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone USD-to-MP4 video converter with 3D geometry rendering.

Dynamically discovers animated body prims and their associated geometry primitives
(UsdGeom.Capsule, UsdGeom.Sphere) from ANY USD robot stage, reads their actual
dimensions (height, radius, axis), infers skeleton connectivity from USD physics
joints, and renders proper 3D capsule/sphere shapes as MP4.

Works for humanoids, quadrupeds, spiders, cobots, AMRs, or any articulated robot.
Has ZERO dependencies on IsaacLab, IsaacSim, AppLauncher, or GPU Vulkan drivers.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import numpy as np


def _generate_palette(n: int) -> list[tuple[int, int, int]]:
    """Generate N visually distinct BGR colors using golden-ratio HSV hue rotation."""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def _edge_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Compute a lighter edge highlight color from a fill color."""
    return tuple(min(255, int(c + (255 - c) * 0.45)) for c in color)


def _draw_capsule_2d(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    radius_px: int,
    fill: tuple[int, int, int],
    edge: tuple[int, int, int],
    cv2: any,
) -> None:
    """Draw a filled 2D capsule (stadium/discorectangle) between two projected endpoints."""
    radius_px = max(2, radius_px)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = (dx * dx + dy * dy) ** 0.5

    if length < 1:
        cv2.circle(img, p1, radius_px, fill, -1, cv2.LINE_AA)
        cv2.circle(img, p1, radius_px, edge, 1, cv2.LINE_AA)
        return

    # Perpendicular offset vector (capsule width)
    nx = -dy / length * radius_px
    ny = dx / length * radius_px

    # Rectangle body of the capsule
    pts = np.array([
        [p1[0] + nx, p1[1] + ny],
        [p2[0] + nx, p2[1] + ny],
        [p2[0] - nx, p2[1] - ny],
        [p1[0] - nx, p1[1] - ny],
    ], dtype=np.int32)

    cv2.fillConvexPoly(img, pts, fill, cv2.LINE_AA)
    # Hemisphere end caps
    cv2.circle(img, p1, radius_px, fill, -1, cv2.LINE_AA)
    cv2.circle(img, p2, radius_px, fill, -1, cv2.LINE_AA)
    # Edge outline
    cv2.polylines(img, [pts], True, edge, 1, cv2.LINE_AA)
    cv2.circle(img, p1, radius_px, edge, 1, cv2.LINE_AA)
    cv2.circle(img, p2, radius_px, edge, 1, cv2.LINE_AA)


def convert_usd_to_mp4(
    usd_path: str,
    output_mp4_path: str | None = None,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Reads a USD stage and renders animated 3D capsule/sphere geometry as MP4.

    Dynamically discovers all animated body prims and their associated geometry
    (Capsule, Sphere) from the USD file, reads actual dimensions (height, radius,
    axis, local transform), infers skeleton connectivity from physics joints,
    and renders proper depth-sorted 3D shapes.

    Args:
        usd_path: Path to the input USD file.
        output_mp4_path: Optional path for the output .mp4 file.
        fps: Target frames per second for output video.
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        Path to the generated MP4 file.
    """
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError:
        try:
            from pxr import Gf, Usd, UsdGeom
            UsdPhysics = None
        except ImportError as e:
            raise RuntimeError(
                "pxr (usd-core) is required. Install via `pip install usd-core`."
            ) from e

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "opencv-python (cv2) is required. Install via `pip install opencv-python`."
        ) from e

    usd_path = os.path.abspath(usd_path)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    if output_mp4_path is None:
        base, _ = os.path.splitext(usd_path)
        output_mp4_path = f"{base}.mp4"
    output_mp4_path = os.path.abspath(output_mp4_path)
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    start_time = int(stage.GetStartTimeCode())
    end_time = int(stage.GetEndTimeCode())
    n_frames = end_time - start_time + 1

    # -------------------------------------------------------------------------
    # Phase 1: Discover all animated body Xform prims
    # -------------------------------------------------------------------------
    skip_keywords = {"terrain", "groundplane", "ground", "environment", "looks",
                     "light", "camera", "sky", "material", "shader", "geometry"}

    body_xforms: dict[str, any] = {}       # name -> UsdGeom.Xformable
    body_prim_map: dict[str, any] = {}     # name -> Usd.Prim
    body_paths: dict[str, str] = {}        # prim_path_str -> name (for joint matching)

    for prim in stage.Traverse():
        name = prim.GetName()
        if name.lower() in skip_keywords:
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            attr = op.GetAttr()
            if attr and attr.GetNumTimeSamples() > 0:
                body_xforms[name] = xf
                body_prim_map[name] = prim
                body_paths[str(prim.GetPath())] = name
                break

    if not body_xforms:
        print(f"[WARN] No animated body prims found in {usd_path}")
        return output_mp4_path

    body_names = list(body_xforms.keys())
    print(f"[INFO] Discovered {len(body_names)} animated body prims: {body_names}")

    # -------------------------------------------------------------------------
    # Phase 2: Discover geometry primitives (Capsule, Sphere)
    # -------------------------------------------------------------------------
    # Geometry shapes live inside "over Flattened_Prototype_*" sections, which
    # stage.Traverse() skips because "over" prims are not "defined". We use
    # stage.TraverseAll() to reach all prims including prototypes.
    # -------------------------------------------------------------------------
    # Phase 2: Match geometry (Capsule, Cylinder, Sphere, Cube, Mesh) to bodies
    # -------------------------------------------------------------------------
    axis_map = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    geom_info: dict[str, list[dict]] = {}  # body_name -> list of shape dicts

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Gprim):
            continue

        # Find nearest ancestor prim that is an animated body link
        curr = prim
        body_name = None
        while curr and curr.IsValid() and str(curr.GetPath()) != "/":
            cname = curr.GetName()
            if cname in body_xforms:
                body_name = cname
                break
            curr = curr.GetParent()

        if not body_name:
            continue

        # Read local transform if present
        local_mat = Gf.Matrix4d(1.0)
        xform_attr = prim.GetAttribute("xformOp:transform")
        if xform_attr and xform_attr.HasValue():
            val = xform_attr.Get()
            if val is not None:
                local_mat = Gf.Matrix4d(val)

        shape_entry = None

        if prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            h = capsule.GetHeightAttr().Get()
            r = capsule.GetRadiusAttr().Get()
            ax = capsule.GetAxisAttr().Get()
            shape_entry = {
                "type": "capsule",
                "height": float(h) if h else 0.1,
                "radius": float(r) if r else 0.02,
                "axis": str(ax) if ax else "Y",
                "local_mat": local_mat,
            }
        elif prim.IsA(UsdGeom.Cylinder):
            cyl = UsdGeom.Cylinder(prim)
            h = cyl.GetHeightAttr().Get()
            r = cyl.GetRadiusAttr().Get()
            ax = cyl.GetAxisAttr().Get()
            shape_entry = {
                "type": "capsule",
                "height": float(h) if h else 0.1,
                "radius": float(r) if r else 0.03,
                "axis": str(ax) if ax else "Z",
                "local_mat": local_mat,
            }
        elif prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            r = sphere.GetRadiusAttr().Get()
            shape_entry = {
                "type": "sphere",
                "radius": float(r) if r else 0.05,
                "local_mat": local_mat,
            }
        elif prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            sz = cube.GetSizeAttr().Get()
            sz_val = float(sz) if sz else 0.1
            shape_entry = {
                "type": "capsule",
                "height": sz_val,
                "radius": sz_val * 0.4,
                "axis": "Z",
                "local_mat": local_mat,
            }
        elif prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            ext_attr = mesh.GetExtentAttr()
            if ext_attr and ext_attr.HasValue():
                ext = ext_attr.Get()
                if ext and len(ext) == 2:
                    min_p, max_p = ext[0], ext[1]
                    dx = abs(max_p[0] - min_p[0])
                    dy = abs(max_p[1] - min_p[1])
                    dz = abs(max_p[2] - min_p[2])
                    h = max(dz, 0.05)
                    r = max(dx, dy) / 2.0
                    r = max(min(r, 0.25), 0.02)
                    shape_entry = {
                        "type": "capsule",
                        "height": float(h),
                        "radius": float(r),
                        "axis": "Z",
                        "local_mat": local_mat,
                    }

        if shape_entry:
            if body_name not in geom_info:
                geom_info[body_name] = []
            # Avoid duplicate identical shapes
            if not any(s["type"] == shape_entry["type"] and s["height"] == shape_entry.get("height") and s["radius"] == shape_entry["radius"] for s in geom_info[body_name]):
                geom_info[body_name].append(shape_entry)

    total_shapes = sum(len(shapes) for shapes in geom_info.values())
    print(f"[INFO] Matched {total_shapes} geometry shape(s) across {len(geom_info)} body link(s)")

    # -------------------------------------------------------------------------
    # Phase 3: Infer skeleton connectivity
    # -------------------------------------------------------------------------
    # Strategy 1: Read physics joints (PhysicsJoint / PhysicsRevoluteJoint).
    # These have rel physics:body0 and rel physics:body1 defining parent-child links.
    # Strategy 2 (fallback): Walk USD scene graph hierarchy.
    # Strategy 3 (fallback): Connect all bodies sequentially.

    skeleton: list[tuple[str, str]] = []

    # Strategy 1: Physics joints
    for prim in stage.TraverseAll():
        body0_attr = prim.GetRelationship("physics:body0")
        body1_attr = prim.GetRelationship("physics:body1")
        if body0_attr and body1_attr:
            targets0 = body0_attr.GetTargets()
            targets1 = body1_attr.GetTargets()
            if targets0 and targets1:
                path0 = str(targets0[0])
                path1 = str(targets1[0])
                name0 = body_paths.get(path0)
                name1 = body_paths.get(path1)
                if name0 and name1 and (name0, name1) not in skeleton:
                    skeleton.append((name0, name1))

    # Strategy 2 fallback: scene graph hierarchy
    if not skeleton:
        for child_name, child_prim in body_prim_map.items():
            parent_prim = child_prim.GetParent()
            while parent_prim and parent_prim.IsValid() and str(parent_prim.GetPath()) != "/":
                parent_name = parent_prim.GetName()
                if parent_name in body_xforms:
                    skeleton.append((parent_name, child_name))
                    break
                parent_prim = parent_prim.GetParent()

    # Strategy 3 fallback: sequential chain
    if not skeleton and len(body_names) > 1:
        for i in range(len(body_names) - 1):
            skeleton.append((body_names[i], body_names[i + 1]))

    print(f"[INFO] Inferred {len(skeleton)} skeleton bone(s)")

    # -------------------------------------------------------------------------
    # Phase 4: Auto-assign colors by hierarchy depth
    # -------------------------------------------------------------------------
    palette = _generate_palette(len(body_names))
    color_map = {name: palette[i] for i, name in enumerate(body_names)}

    # -------------------------------------------------------------------------
    # Phase 5: Pre-compute per-frame data (positions + geometry shapes)
    # -------------------------------------------------------------------------
    frame_data: list[dict[str, dict]] = []

    for t in range(start_time, end_time + 1):
        tc = Usd.TimeCode(t)
        frame: dict[str, dict] = {}

        for name, xf in body_xforms.items():
            body_mat = xf.ComputeLocalToWorldTransform(tc)
            pos = body_mat.ExtractTranslation()
            entry: dict = {
                "pos": np.array([pos[0], pos[1], pos[2]], dtype=np.float64),
                "shapes": [],
            }

            if name in geom_info:
                for gi in geom_info[name]:
                    if gi["type"] == "capsule":
                        geom_world = gi["local_mat"] * body_mat
                        axis_dir = axis_map.get(gi["axis"], Gf.Vec3d(0, 1, 0))
                        half_h = gi["height"] / 2.0
                        p1 = geom_world.Transform(axis_dir * half_h)
                        p2 = geom_world.Transform(axis_dir * (-half_h))
                        entry["shapes"].append({
                            "type": "capsule",
                            "p1": np.array([p1[0], p1[1], p1[2]], dtype=np.float64),
                            "p2": np.array([p2[0], p2[1], p2[2]], dtype=np.float64),
                            "radius": gi["radius"],
                            "center": np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0], dtype=np.float64),
                        })
                    elif gi["type"] == "sphere":
                        center = body_mat.Transform(Gf.Vec3d(0, 0, 0))
                        entry["shapes"].append({
                            "type": "sphere",
                            "center": np.array([center[0], center[1], center[2]], dtype=np.float64),
                            "radius": gi["radius"],
                        })

            frame[name] = entry

        frame_data.append(frame)

    # -------------------------------------------------------------------------
    # Phase 6: Camera & projection setup
    # -------------------------------------------------------------------------
    all_pts = []
    for frame in frame_data:
        for entry in frame.values():
            all_pts.append(entry["pos"])
    all_pts = np.array(all_pts)
    valid_pts = all_pts[np.isfinite(all_pts).all(axis=1)] if len(all_pts) > 0 else np.array([])
    if len(valid_pts) == 0:
        valid_pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    min_b = valid_pts.min(axis=0)
    max_b = valid_pts.max(axis=0)
    center = (min_b + max_b) / 2.0
    extent = float(np.max(max_b - min_b))
    if not np.isfinite(extent) or extent < 0.5:
        extent = 2.0

    focal = width * 1.3
    cam_dist = max(2.5, extent * 1.6)
    cam_pos = center + np.array([cam_dist * 0.7, -cam_dist * 0.9, cam_dist * 0.5])

    forward = center - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    r_norm = np.linalg.norm(right)
    right = right / r_norm if r_norm > 1e-3 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)

    def project(pt: np.ndarray) -> tuple[int, int] | None:
        rel = pt - cam_pos
        z = np.dot(rel, forward)
        if z <= 0.05:
            return None
        x = np.dot(rel, right)
        y = np.dot(rel, up)
        return (int(width / 2 + (x / z) * focal), int(height / 2 - (y / z) * focal))

    def project_radius(world_pt: np.ndarray, radius: float) -> int:
        """Project a 3D world-space radius to pixel size based on perspective depth."""
        z = np.dot(world_pt - cam_pos, forward)
        if z <= 0.05:
            return 0
        return max(2, int(radius / z * focal))

    # -------------------------------------------------------------------------
    # Phase 7: Render frames
    # -------------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4_path, fourcc, float(fps), (width, height))
    ground_z = float(min_b[2]) - 0.05

    for fi in range(n_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (20, 22, 30)

        # Ground grid
        grid_ext = max(3.0, extent * 2.0)
        grid_step = max(0.25, extent / 8.0)
        for gx in np.arange(center[0] - grid_ext, center[0] + grid_ext, grid_step):
            gp1 = project(np.array([gx, center[1] - grid_ext, ground_z]))
            gp2 = project(np.array([gx, center[1] + grid_ext, ground_z]))
            if gp1 and gp2:
                cv2.line(img, gp1, gp2, (38, 42, 52), 1)
        for gy in np.arange(center[1] - grid_ext, center[1] + grid_ext, grid_step):
            gp1 = project(np.array([center[0] - grid_ext, gy, ground_z]))
            gp2 = project(np.array([center[0] + grid_ext, gy, ground_z]))
            if gp1 and gp2:
                cv2.line(img, gp1, gp2, (38, 42, 52), 1)

        frame = frame_data[fi]

        # Draw ground shadows first
        for name in body_names:
            sp = project(np.array([frame[name]["pos"][0], frame[name]["pos"][1], ground_z]))
            if sp and 0 <= sp[0] < width and 0 <= sp[1] < height:
                cv2.circle(img, sp, 3, (45, 48, 58), -1)

        # Draw skeleton bone lines (behind geometry, for visual connectivity)
        for parent_name, child_name in skeleton:
            p1 = project(frame[parent_name]["pos"])
            p2 = project(frame[child_name]["pos"])
            if p1 and p2:
                cv2.line(img, p1, p2, (50, 55, 70), 2, cv2.LINE_AA)

        # Collect all 3D shapes for this frame and depth-sort back-to-front
        all_shapes = []
        for name in body_names:
            color = color_map[name]
            edge = _edge_color(color)
            shapes = frame[name]["shapes"]
            if shapes:
                for s in shapes:
                    center_pt = s.get("center", frame[name]["pos"])
                    depth = np.dot(center_pt - cam_pos, forward)
                    all_shapes.append({"depth": depth, "shape": s, "color": color, "edge": edge, "body": name})
            else:
                depth = np.dot(frame[name]["pos"] - cam_pos, forward)
                all_shapes.append({"depth": depth, "shape": None, "color": color, "edge": edge, "body": name})

        all_shapes.sort(key=lambda item: -item["depth"])

        # Draw geometry shapes (depth-sorted, back to front)
        for item in all_shapes:
            s = item["shape"]
            color = item["color"]
            edge = item["edge"]

            if s is not None:
                if s["type"] == "capsule":
                    p1_2d = project(s["p1"])
                    p2_2d = project(s["p2"])
                    if p1_2d and p2_2d:
                        r_px = project_radius(s["center"], s["radius"])
                        _draw_capsule_2d(img, p1_2d, p2_2d, r_px, color, edge, cv2)
                elif s["type"] == "sphere":
                    p2d = project(s["center"])
                    if p2d and 0 <= p2d[0] < width and 0 <= p2d[1] < height:
                        r_px = project_radius(s["center"], s["radius"])
                        cv2.circle(img, p2d, r_px, color, -1, cv2.LINE_AA)
                        cv2.circle(img, p2d, r_px, edge, 1, cv2.LINE_AA)
            else:
                # Fallback: no geometry found for this body, draw a joint circle
                p2d = project(frame[item["body"]]["pos"])
                if p2d and 0 <= p2d[0] < width and 0 <= p2d[1] < height:
                    cv2.circle(img, p2d, 6, color, -1, cv2.LINE_AA)
                    cv2.circle(img, p2d, 6, edge, 1, cv2.LINE_AA)

        # HUD
        sim_t = fi * (1.0 / fps) if fps > 0 else 0
        hud_text = (f"Frame {fi + 1}/{n_frames}  |  t = {sim_t:.2f}s  |  "
                    f"Bodies: {len(body_names)}  |  Bones: {len(skeleton)}  |  "
                    f"Shapes: {len(geom_info)}")
        cv2.putText(img, hud_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "USD 3D Trajectory Playback",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 105, 120), 1, cv2.LINE_AA)

        out.write(img)

    out.release()
    print(f"[INFO] Rendered {n_frames} frames ({len(body_names)} bodies, "
          f"{len(geom_info)} shapes, {len(skeleton)} bones) -> {output_mp4_path}")
    return output_mp4_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert USD trajectory (.usda/.usd) to 3D geometry MP4 video."
    )
    parser.add_argument("usd_path", type=str, help="Path to input USD stage file.")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS for output MP4.")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels.")
    args = parser.parse_args()

    convert_usd_to_mp4(args.usd_path, args.out, fps=args.fps, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
