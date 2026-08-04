# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process ROS2 scene subscriber for the pseudo-HIL traversability task.

Spawns a background ``rclpy`` node that subscribes to the synthetic world node's
topics and caches the latest map and camera frame so IsaacLab observation / reward /
command / event terms can read them synchronously on ``env.device``.

Topics:
  * /amr/world/grid    (nav_msgs/OccupancyGrid)  - ground-truth map
  * /amr/camera/rgb    (sensor_msgs/Image)       - forward-ahead camera frame
  * /amr/robot_pose    (geometry_msgs/PoseStamped) - published by IsaacLab (out)
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch

# Module-level shared handle so every term reuses one node/thread.
_ROS_SCENE: "RosScene" | None = None


class RosScene:
    """Background rclpy subscriber caching the latest map + camera frame."""

    def __init__(self) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import Image

        self._Image = Image
        self._PoseStamped = PoseStamped

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("amr_traversability_ros_scene")

        self._grid_msg: OccupancyGrid | None = None
        self._image_msg: Image | None = None
        self._grid_lock = threading.Lock()
        self._image_lock = threading.Lock()

        self._grid_sub = self._node.create_subscription(OccupancyGrid, "/amr/world/grid", self._on_grid, 10)
        self._cam_sub = self._node.create_subscription(Image, "/amr/camera/rgb", self._on_image, 10)
        self._pose_pub = self._node.create_publisher(PoseStamped, "/amr/robot_pose", 10)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        import rclpy

        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _on_grid(self, msg) -> None:
        with self._grid_lock:
            self._grid_msg = msg

    def _on_image(self, msg) -> None:
        with self._image_lock:
            self._image_msg = msg

    # ------------------------------------------------------------------
    def publish_pose(self, x: float, y: float, yaw: float, stamp=None) -> None:
        """Publish the robot's world pose so the world node renders a fresh frame."""
        msg = self._PoseStamped()
        if stamp is not None:
            msg.header.stamp = stamp
        else:
            msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = float(np.cos(yaw / 2.0))
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(np.sin(yaw / 2.0))
        self._pose_pub.publish(msg)

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Block until the grid (and ideally a camera frame) has arrived."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._grid_lock:
                has_grid = self._grid_msg is not None
            if has_grid:
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    def occupancy_grid_tensor(self, device: str) -> torch.Tensor | None:
        """Return the grid (H, W) uint8 on ``device`` if available, else None."""
        with self._grid_lock:
            msg = self._grid_msg
        if msg is None:
            return None
        arr = np.asarray(msg.data, dtype=np.uint8).reshape(msg.info.height, msg.info.width)
        return torch.from_numpy(arr).to(device)

    def grid_meta(self) -> dict:
        """Return grid resolution and origin in world coordinates."""
        with self._grid_lock:
            msg = self._grid_msg
        return {
            "resolution": msg.info.resolution,
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
            "width": msg.info.width,
            "height": msg.info.height,
        }

    def camera_frame(self, device: str) -> torch.Tensor | None:
        """Return the latest RGB frame (H, W, 3) uint8 on ``device`` if available, else None."""
        with self._image_lock:
            msg = self._image_msg
        if msg is None:
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return torch.from_numpy(arr).to(device)

    def close(self) -> None:
        import rclpy

        if rclpy.ok():
            self._node.destroy_node()
            rclpy.shutdown()


# -----------------------------------------------------------------------------
# Grid ground-truth helpers (world XY <-> occupancy cells)
# -----------------------------------------------------------------------------

#: Occupancy value that counts as "on the white path".
_PATH_OCC_THRESHOLD = 50

#: Cached ground-truth (grid tensor, meta dict, centers tensor) once the map is in.
_GT_CACHE: dict | None = None


def get_ground_truth(device: str) -> dict:
    """Return cached {grid, resolution, origin_x, origin_y, centers} from the ROS map.

    The map is static, so it is fetched once and cached for the process lifetime.
    """
    global _GT_CACHE
    if _GT_CACHE is None:
        scene = get_ros_scene()
        if not scene.wait_until_ready(timeout=15.0):
            raise RuntimeError(
                "No occupancy grid received on /amr/world/grid. "
                "Is the synthetic world node running? (python core/ros2_ws/ros_synthetic_world.py)"
            )
        grid = scene.occupancy_grid_tensor(device)
        meta = scene.grid_meta()
        centers = grid_cell_centers(
            grid, meta["resolution"], meta["origin_x"], meta["origin_y"], device
        )
        _GT_CACHE = {
            "grid": grid,
            "resolution": meta["resolution"],
            "origin_x": meta["origin_x"],
            "origin_y": meta["origin_y"],
            "centers": centers,
        }
    return _GT_CACHE


def world_to_cell(
    xy: torch.Tensor,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert world-local (n, 2) positions to (col, row) integer cell indices."""
    col = torch.floor((xy[:, 0] - origin_x) / resolution).long()
    row = torch.floor((xy[:, 1] - origin_y) / resolution).long()
    return col, row


def grid_cell_centers(
    grid: torch.Tensor,
    resolution: float,
    origin_x: float,
    origin_y: float,
    device: str,
) -> torch.Tensor:
    """World (M, 2) coordinates of the centers of all on-path (white) cells."""
    occ = (grid > _PATH_OCC_THRESHOLD).nonzero(as_tuple=False)  # (M, 2) rows = row, col
    rows = occ[:, 0].float()
    cols = occ[:, 1].float()
    centers = torch.stack(
        [origin_x + (cols + 0.5) * resolution, origin_y + (rows + 0.5) * resolution], dim=1
    ).to(device)
    return centers


def on_path_mask(
    grid: torch.Tensor,
    xy: torch.Tensor,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> torch.Tensor:
    """Boolean (n,) whether each world-local (n, 2) position lies on the white path."""
    col, row = world_to_cell(xy, resolution, origin_x, origin_y)
    h, w = grid.shape
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    occ = torch.zeros(len(xy), dtype=torch.long, device=xy.device)
    occ[valid] = grid[row[valid], col[valid]]
    return occ > _PATH_OCC_THRESHOLD


def distance_to_path(
    grid: torch.Tensor,
    xy: torch.Tensor,
    resolution: float,
    origin_x: float,
    origin_y: float,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Euclidean distance (n,) in meters from each (n, 2) point to the nearest on-path cell center."""
    dist = torch.norm(xy.unsqueeze(1) - centers.unsqueeze(0), dim=2)
    return dist.min(dim=1).values


def path_tangent_at(
    grid: torch.Tensor,
    xy: torch.Tensor,
    resolution: float,
    origin_x: float,
    origin_y: float,
    centers: torch.Tensor,
    radius: float = 0.3,
) -> torch.Tensor:
    """Heading (n,) of the path tangent at the nearest on-path cell.

    Computed with a local PCA over the on-path cell centers within ``radius`` of the
    point: the first principal component is the local contour direction.
    """
    dist = torch.norm(xy.unsqueeze(1) - centers.unsqueeze(0), dim=2)
    nearest = dist.min(dim=1).indices
    base = centers[nearest]  # (n, 2)

    near = dist <= radius  # (n, M) mask of on-path cells within the window
    tangent = torch.empty(len(xy), device=xy.device)
    for i in range(len(xy)):
        cells = centers[near[i]]
        if cells.shape[0] < 2:
            tangent[i] = 0.0
            continue
        c = cells - cells.mean(dim=0, keepdim=True)
        cov = c.T @ c
        # first principal axis (direction of max variance) = local tangent
        _, _, vt = torch.linalg.svd(cov)
        d = vt[0]
        if d[0] < 0:
            d = -d
        tangent[i] = torch.atan2(d[1], d[0])
    return tangent


def get_ros_scene() -> RosScene:
    """Return the shared module-level RosScene, creating it on first call."""
    global _ROS_SCENE
    if _ROS_SCENE is None:
        _ROS_SCENE = RosScene()
    return _ROS_SCENE