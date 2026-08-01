# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Universal Video Recorder & Optional Live ROS2 Stream Publisher for Headless Docker Simulations."""

from __future__ import annotations

import os
import time
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


class PeriodicVideoRecorderWrapper:
    """Wraps an IsaacLab RL environment to record simulation video clips or publish ROS2 camera streams.

    Supported Modes:
      - 'video' (DEFAULT): Direct MP4 video recording to file system (/workspace/data/videos/).
      - 'ros2': Stream live camera RGB frames over ROS2 network topic (/camera/rgb/image_raw).
      - 'both': Concurrently record MP4 video files AND publish live ROS2 topic stream.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        mode: Literal["video", "ros2", "both"] = "video",  # Default mode: Direct MP4 video recording
        video_folder: str = "/workspace/data/videos",
        record_interval_s: float = 3600.0,  # Record 1 clip every 1 hour
        video_length_s: float = 60.0,        # 1 minute duration per clip
        fps: int = 30,
        ros2_topic: str = "/camera/rgb/image_raw",
    ):
        self.env = env
        self.mode = mode.lower()
        self.video_folder = os.path.abspath(video_folder)
        self.record_interval_s = record_interval_s
        self.video_length_s = video_length_s
        self.fps = fps
        self.ros2_topic = ros2_topic

        # Flags based on selected mode
        self.enable_file_recording = self.mode in ("video", "both")
        self.enable_ros2_stream = self.mode in ("ros2", "both")

        if self.enable_file_recording:
            os.makedirs(self.video_folder, exist_ok=True)

        self.sim_dt = getattr(env, "step_dt", 1 / 60.0)
        self.steps_per_record = int(self.video_length_s / self.sim_dt)
        self.steps_between_records = int(self.record_interval_s / self.sim_dt)

        self.step_counter = 0
        self.is_recording = False
        self.current_video_frames: list[np.ndarray] = []
        self.video_count = 0
        self.last_record_wall_time = time.time()

        # Initialize ROS2 Publisher only if ROS2 stream mode is active
        self._ros2_node = None
        self._ros2_pub = None
        if self.enable_ros2_stream:
            self._init_ros2_publisher()

        logger.info(
            f"PeriodicVideoRecorderWrapper initialized [Mode: '{self.mode.upper()}']: "
            f"File Recording: {self.enable_file_recording} ({video_folder}) | "
            f"ROS2 Topic Stream: {self.enable_ros2_stream} ({self.ros2_topic})"
        )

    def _init_ros2_publisher(self) -> None:
        """Initializes ROS2 publisher when ROS2 mode or both mode is enabled."""
        try:
            import rclpy
            from sensor_msgs.msg import Image

            if not rclpy.ok():
                rclpy.init(args=None)

            self._ros2_node = rclpy.create_node("isaac_simulation_video_streamer")
            self._ros2_pub = self._ros2_node.create_publisher(Image, self.ros2_topic, 10)
            logger.info(f"ROS2 Video Stream Publisher initialized on '{self.ros2_topic}'")
        except Exception as e:
            logger.warning(f"Could not initialize ROS2 video publisher: {e}. Falling back to video file mode.")
            self.enable_ros2_stream = False

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, extras = self.env.reset(**kwargs)
        return obs, extras

    def step(self, action: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, terminated, truncated, extras = self.env.step(action)
        self.step_counter += 1

        # Render frame if active
        frame = None
        if self.enable_file_recording or self.enable_ros2_stream:
            frame = self._capture_rgb_frame()

        # 1. Publish live ROS2 frame if ROS2 stream active
        if self.enable_ros2_stream and self._ros2_pub is not None and frame is not None:
            self._publish_ros2_frame(frame)

        # 2. Process MP4 video recording to disk if file mode active
        if self.enable_file_recording:
            current_wall_time = time.time()
            time_elapsed = current_wall_time - self.last_record_wall_time

            if not self.is_recording:
                if time_elapsed >= self.record_interval_s or (self.step_counter % self.steps_between_records == 0):
                    self._start_recording()

            if self.is_recording and frame is not None:
                self.current_video_frames.append(frame)
                if len(self.current_video_frames) >= self.steps_per_record:
                    self._stop_and_save_recording()

        return obs, rew, terminated, truncated, extras

    def _publish_ros2_frame(self, frame: np.ndarray) -> None:
        """Publishes numpy RGB frame onto ROS2 topic."""
        try:
            from sensor_msgs.msg import Image

            h, w, c = frame.shape
            msg = Image()
            msg.header.stamp = self._ros2_node.get_clock().now().to_msg()
            msg.header.frame_id = "simulation_camera"
            msg.height = h
            msg.width = w
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = w * c
            msg.data = frame.tobytes()

            self._ros2_pub.publish(msg)
        except Exception as e:
            logger.debug(f"ROS2 frame publish notice: {e}")

    def _start_recording(self) -> None:
        self.is_recording = True
        self.current_video_frames = []
        self.last_record_wall_time = time.time()
        logger.info(f"Started direct MP4 video recording clip #{self.video_count + 1} at step {self.step_counter}...")

    def _capture_rgb_frame(self) -> np.ndarray | None:
        """Captures RGB frame from env render or camera sensor."""
        try:
            if hasattr(self.env, "render"):
                frame = self.env.render()
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    return frame

            if hasattr(self.env, "scene") and "tiled_camera" in self.env.scene.sensors:
                cam_data = self.env.scene.sensors["tiled_camera"].data.output["rgb"]
                if isinstance(cam_data, torch.Tensor):
                    return cam_data[0].cpu().numpy()
        except Exception as e:
            logger.debug(f"Frame capture fallback notice: {e}")

        return None

    def _stop_and_save_recording(self) -> None:
        self.is_recording = False
        self.video_count += 1
        video_filename = os.path.join(
            self.video_folder, f"sim_clip_{self.video_count:04d}_step_{self.step_counter}.mp4"
        )

        try:
            import cv2
            if len(self.current_video_frames) > 0:
                h, w, _ = self.current_video_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_filename, fourcc, float(self.fps), (w, h))

                for f in self.current_video_frames:
                    bgr_frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f
                    writer.write(bgr_frame)
                writer.release()
                logger.info(f"Saved direct MP4 video clip: {video_filename} ({len(self.current_video_frames)} frames)")
        except ImportError:
            npy_path = video_filename.replace(".mp4", ".npy")
            np.save(npy_path, np.array(self.current_video_frames))
            logger.info(f"OpenCV not found. Saved raw numpy frames to: {npy_path}")

        self.current_video_frames = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
