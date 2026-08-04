#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 Image Listener: Receives live simulation video stream from IsaacLab headless Docker container."""

from __future__ import annotations

import argparse
import ctypes
import inspect
import os
import sys

import numpy as np

# Patch inspect.getfile to prevent TypeError on namespace packages
_orig_getfile = inspect.getfile
def _safe_getfile(object):
    try:
        return _orig_getfile(object)
    except TypeError:
        return getattr(object, "__file__", None) or "<namespace>"
inspect.getfile = _safe_getfile

# Ensure ROS2 Python site-packages & C shared libraries are loaded into process space
ros_distro = os.getenv("ROS_DISTRO", "humble")
ros_lib_dir = f"/opt/ros/{ros_distro}/lib"

if os.path.exists(ros_lib_dir):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ros_lib_dir not in current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{ros_lib_dir}:{current_ld}"

    for so_name in [
        "librcutils.so",
        "librcpputils.so",
        "librcl_logging_interface.so",
        "librcl_interfaces__rosidl_generator_c.so",
        "librcl_interfaces__rosidl_typesupport_c.so",
        "librmw.so",
        "librmw_implementation.so",
        "librcl.so",
        "librcl_action.so",
        "librcl_yaml_param_parser.so",
    ]:
        so_path = os.path.join(ros_lib_dir, so_name)
        if os.path.exists(so_path):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

for p in [
    f"/opt/ros/{ros_distro}/lib/python3.10/site-packages",
    f"/opt/ros/{ros_distro}/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:
    print("ROS2 (rclpy) is required to run this listener script.")
    sys.exit(1)


class SimulationImageListener(Node):
    """ROS2 node that subscribes to image topic and receives live simulation video frames."""

    def __init__(self, topic: str = "/amr/camera/rgb"):
        super().__init__("simulation_image_listener")
        self.subscription = self.create_subscription(
            Image, topic, self.image_callback, 10
        )
        self.frame_count = 0
        print(f"[INFO] Simulation Image Listener active on ROS2 topic '{topic}'")

    def image_callback(self, msg: Image):
        self.frame_count += 1
        h, w = msg.height, msg.width

        # Convert ROS2 Image bytes into numpy RGB array
        frame_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))

        if self.frame_count % 10 == 0:
            print(
                f"[LIVE SIMULATION STREAM] Received ROS2 frame #{self.frame_count}: {w}x{h} resolution, mean intensity: {frame_data.mean():.1f}"
            )


def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 Image Listener for IsaacLab")
    parser.add_argument("--topic", type=str, default="/amr/camera/rgb", help="ROS2 image topic name")
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SimulationImageListener(topic=cli_args.topic)
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
