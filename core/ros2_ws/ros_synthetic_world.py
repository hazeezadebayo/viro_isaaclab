#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 synthetic world node (pseudo-HIL).

Single source of truth for the AMR traversability scene. It owns the figure-8 path
geometry and publishes two synchronized streams that IsaacLab consumes:

  * /amr/world/grid   (nav_msgs/OccupancyGrid)  - ground-truth map (white path = 100)
  * /amr/camera/rgb   (sensor_msgs/Image)       - forward-ahead POV rendered from the
                                                  robot pose against the grid

The camera image is derived directly from the occupancy grid, so pixels and map are
guaranteed consistent (what the policy sees == what the rewards enforce).

Closed loop:
  IsaacLab steps -> publishes /amr/robot_pose -> this node renders a fresh camera
  frame at that pose -> IsaacLab reads the frame as its observation.
"""

from __future__ import annotations

import math
import sys

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import Image
except ImportError:
    print("ROS2 (rclpy) is required to run the synthetic world node.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Figure-8 map geometry (same lemniscate as the original IsaacLab task)
# -----------------------------------------------------------------------------

_LEMNISCATE_SCALE = 2.4
_PATH_WIDTH = 0.32  # half-width of the white path strip (m)


def _centerline() -> np.ndarray:
    """Dense figure-8 centerline (M, 2) in local coordinates."""
    t = np.linspace(0.0, 2.0 * math.pi, 800, endpoint=False)
    denom = 1.0 + np.sin(t) ** 2
    x = _LEMNISCATE_SCALE * np.cos(t) / denom
    y = _LEMNISCATE_SCALE * np.sin(t) * np.cos(t) / denom
    return np.stack([x, y], axis=-1)


# -----------------------------------------------------------------------------
# Grid definition
# -----------------------------------------------------------------------------

GRID_RESOLUTION = 0.05  # m/cell
GRID_X_MIN, GRID_X_MAX = -3.0, 3.0
GRID_Y_MIN, GRID_Y_MAX = -1.5, 1.5
GRID_WIDTH = int(round((GRID_X_MAX - GRID_X_MIN) / GRID_RESOLUTION))  # 120
GRID_HEIGHT = int(round((GRID_Y_MAX - GRID_Y_MIN) / GRID_RESOLUTION))  # 60


def build_occupancy_grid() -> tuple[np.ndarray, OccupancyGrid]:
    """Return (cell_grid[H, W] uint8, OccupancyGrid msg)."""
    xs = GRID_X_MIN + (np.arange(GRID_WIDTH) + 0.5) * GRID_RESOLUTION
    ys = GRID_Y_MIN + (np.arange(GRID_HEIGHT) + 0.5) * GRID_RESOLUTION
    xx, yy = np.meshgrid(xs, ys)  # (H, W)

    centerline = _centerline()  # (M, 2)
    # distance from each cell center to the nearest centerline point
    dist = np.linalg.norm(
        np.stack([xx[..., None], yy[..., None]], axis=-1) - centerline[None, None, :, :],
        axis=-1,
    ).min(axis=-1)  # (H, W)

    cell_grid = np.where(dist <= _PATH_WIDTH, 100, 0).astype(np.uint8)

    msg = OccupancyGrid()
    msg.header.frame_id = "world"
    msg.info.resolution = GRID_RESOLUTION
    msg.info.width = GRID_WIDTH
    msg.info.height = GRID_HEIGHT
    msg.info.origin.position.x = GRID_X_MIN
    msg.info.origin.position.y = GRID_Y_MIN
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = cell_grid.ravel().tolist()
    return cell_grid, msg


# -----------------------------------------------------------------------------
# Forward-ahead POV camera (ground-plane pinhole projection of the grid)
# -----------------------------------------------------------------------------

CAM_H, CAM_W = 96, 128
CAM_FOV_H = 90.0  # deg
CAM_FOCAL = (CAM_W / 2.0) / math.tan(math.radians(CAM_FOV_H / 2.0))
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0
CAM_PITCH = math.radians(15.0)  # downward pitch of the camera

# TurtleBot3 camera mount in robot frame (URDF): (x, y, z) meters, camera pitched 15 deg down
_MOUNT_OFFSET = np.array([0.069, -0.047, 0.107])


def render_camera_frame(
    cell_grid: np.ndarray, pose: tuple[float, float, float]
) -> np.ndarray:
    """Render a forward-ahead RGB frame (CAM_H, CAM_W, 3) uint8 from the grid.

    Args:
        cell_grid: occupancy grid (H, W), 100 = white path.
        pose: robot (x, y, yaw) in world coordinates.
    """
    x, y, yaw = pose
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    # camera position in world
    cam_pos = np.array(
        [
            x + _MOUNT_OFFSET[0] * cos_y - _MOUNT_OFFSET[1] * sin_y,
            y + _MOUNT_OFFSET[0] * sin_y + _MOUNT_OFFSET[1] * cos_y,
            _MOUNT_OFFSET[2],
        ]
    )

    # camera basis vectors in world frame (forward = +z_c, right = +x_c, down = +y_c)
    fwd = np.array(
        [math.cos(CAM_PITCH) * cos_y, math.cos(CAM_PITCH) * sin_y, -math.sin(CAM_PITCH)]
    )
    right = np.array([-sin_y, cos_y, 0.0])
    down = np.cross(right, fwd)

    # pixel grid (u, v)
    u = np.arange(CAM_W, dtype=np.float32)
    v = np.arange(CAM_H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    x_n = (uu - CAM_CX) / CAM_FOCAL
    y_n = -(vv - CAM_CY) / CAM_FOCAL

    # ray direction in world: x_n*right + y_n*down + 1*fwd
    ray = (
        x_n[..., None] * right[None, None, :]
        + y_n[..., None] * down[None, None, :]
        + fwd[None, None, :]
    )
    ray_norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(ray_norm, 1e-9)

    # intersect with ground plane z=0: t = -cam_z / ray_z  (only rays pointing down)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -cam_pos[2] / ray[..., 2]
    hit = (t > 0) & (ray[..., 2] < 0)
    t_safe = np.where(hit, t, 1e9)

    world_pt = cam_pos[None, None, :] + t_safe[..., None] * ray  # (H, W, 3)

    # sample occupancy at the ground point
    col = np.floor((world_pt[..., 0] - GRID_X_MIN) / GRID_RESOLUTION).astype(np.int64)
    row = np.floor((world_pt[..., 1] - GRID_Y_MIN) / GRID_RESOLUTION).astype(np.int64)
    in_grid = (col >= 0) & (col < GRID_WIDTH) & (row >= 0) & (row < GRID_HEIGHT) & hit
    occ = np.zeros((CAM_H, CAM_W), dtype=np.float32)
    occ[in_grid] = cell_grid[row[in_grid], col[in_grid]]

    # render: white path -> bright, ground -> dark (keep well clear of the 0.5 mask threshold)
    frame = np.where(occ > 50, 250.0, 5.0)
    # subtle noise so the frame looks like a real sensor (won't flip the threshold)
    frame += np.random.default_rng(0).uniform(-12.0, 12.0, size=frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.stack([frame] * 3, axis=-1)


# -----------------------------------------------------------------------------
# ROS node
# -----------------------------------------------------------------------------

GRID_REPUBLISH_PERIOD = 1.0  # s (transient_local also serves late subscribers)


class SyntheticWorldNode(Node):
    """Publishes the traversability grid and pose-driven forward camera frames."""

    def __init__(self):
        super().__init__("synthetic_world")
        self._cell_grid, self._grid_msg = build_occupancy_grid()

        self._grid_pub = self.create_publisher(OccupancyGrid, "/amr/world/grid", 10)
        self._cam_pub = self.create_publisher(Image, "/amr/camera/rgb", 10)
        self._pose_sub = self.create_subscription(
            PoseStamped, "/amr/robot_pose", self._on_pose, 10
        )

        self._publish_grid()
        self._grid_timer = self.create_timer(GRID_REPUBLISH_PERIOD, self._publish_grid)
        self.get_logger().info(
            f"Synthetic world ready: grid {GRID_WIDTH}x{GRID_HEIGHT} @ {GRID_RESOLUTION} m/cell, "
            f"camera {CAM_W}x{CAM_H}"
        )

    def _publish_grid(self) -> None:
        self._grid_msg.header.stamp = self.get_clock().now().to_msg()
        self._grid_pub.publish(self._grid_msg)

    def _on_pose(self, msg: PoseStamped) -> None:
        # pose is in world frame; yaw from the quaternion (z-rotation)
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        pose = (msg.pose.position.x, msg.pose.position.y, yaw)

        frame = render_camera_frame(self._cell_grid, pose)

        img = Image()
        img.header = msg.header
        img.header.frame_id = "camera"
        img.height = CAM_H
        img.width = CAM_W
        img.encoding = "rgb8"
        img.is_bigendian = False
        img.step = CAM_W * 3
        img.data = frame.tobytes()
        self._cam_pub.publish(img)


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticWorldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
