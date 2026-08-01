#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 Image Listener: Receives live simulation video stream from IsaacLab headless Docker container."""

import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:
    print("ROS2 (rclpy) is required to run this listener script.")
    sys.exit(1)


class SimulationImageListener(Node):
    """ROS2 node that subscribes to '/camera/rgb/image_raw' and receives live simulation video frames."""

    def __init__(self):
        super().__init__("simulation_image_listener")
        self.subscription = self.create_subscription(
            Image, "/camera/rgb/image_raw", self.image_callback, 10
        )
        self.frame_count = 0
        self.get_logger().info("Simulation Image Listener active on topic '/camera/rgb/image_raw'")

    def image_callback(self, msg: Image):
        self.frame_count += 1
        h, w = msg.height, msg.width

        # Convert ROS2 Image bytes into numpy RGB array
        frame_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"[LIVE SIMULATION STREAM] Received frame #{self.frame_count}: {w}x{h} resolution, mean intensity: {frame_data.mean():.1f}"
            )

        # Optional: Display with OpenCV if GUI desktop environment is available
        try:
            import cv2
            bgr_frame = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
            cv2.imshow("IsaacLab Headless Live Video Stream", bgr_frame)
            cv2.waitKey(1)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SimulationImageListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
