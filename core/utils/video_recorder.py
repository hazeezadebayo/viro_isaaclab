# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Universal Video Recorder & Optional Live ROS2 Stream Publisher for Headless Docker Simulations.

Clip scheduling mirrors the built-in IsaacLab ``--video`` recording:
  - A clip is started with the very first simulation step.
  - A new clip is started every ``record_interval_s`` of simulated time.
  - Each clip lasts ``video_length_s`` of simulated time.

MP4 encoding is handled by moviepy + ffmpeg (falling back to OpenCV, imageio, and finally a raw
``.npy`` dump). In ROS2 mode the same RGB frames are streamed on a ROS2 ``Image`` topic.
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


class PeriodicVideoRecorderWrapper:
    """Wraps an IsaacLab RL environment to record simulation video clips or publish ROS2 camera streams.

    Supported Modes:
      - 'video' (DEFAULT): Direct MP4 video recording to file system (/workspace/core/logs/videos/).
      - 'ros2': Stream live camera RGB frames over ROS2 network topic (/camera/rgb/image_raw).
      - 'both': Concurrently record MP4 video files AND publish live ROS2 topic stream.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        mode: Literal["video", "ros2", "both"] = "video",  # Default mode: Direct MP4 video recording
        video_folder: str = "/workspace/core/logs/videos",
        record_interval_s: float = 1800.0,  # Start a new clip every 30 minutes of sim time
        video_length_s: float = 60.0,       # 1 minute duration per clip
        fps: int = 30,
        ros2_topic: str = "/camera/rgb/image_raw",
    ):
        self.env = env
        self.mode = mode.lower()
        if self.mode not in ("video", "ros2", "both"):
            raise ValueError(f"Unknown recorder mode: {self.mode!r} (expected 'video', 'ros2' or 'both')")
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

        # Convert clip length/interval from seconds to simulation steps.
        self.sim_dt = float(getattr(env, "step_dt", 1.0 / 60.0))
        self.steps_per_record = max(1, int(self.video_length_s / self.sim_dt))
        self.steps_between_records = max(self.steps_per_record, int(self.record_interval_s / self.sim_dt))

        self.step_counter = 0
        self.is_recording = False
        self._start_step = 0
        self._next_record_start_step = 0
        self.current_video_frames: list[np.ndarray] = []
        self.video_count = 0
        self._reported_no_frame_source = False

        # Initialize ROS2 Publisher only if ROS2 stream mode is active
        self._ros2_node = None
        self._ros2_pub = None
        if self.enable_ros2_stream:
            self._init_ros2_publisher()

        logger.info(
            f"PeriodicVideoRecorderWrapper initialized [Mode: '{self.mode.upper()}']: "
            f"File Recording: {self.enable_file_recording} ({video_folder}) | "
            f"ROS2 Topic Stream: {self.enable_ros2_stream} ({self.ros2_topic}) | "
            f"sim_dt={self.sim_dt:.5f}s -> clip length {self.steps_per_record} steps, "
            f"interval {self.steps_between_records} steps"
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
            logger.warning(
                f"Could not initialize ROS2 video publisher ({e}). Falling back to file recording only. "
                f"Note: rclpy must be importable by the Python interpreter running the simulation."
            )
            self.enable_ros2_stream = False

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, extras = self.env.reset(**kwargs)
        return obs, extras

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict]:
        obs, rew, terminated, truncated, extras = self.env.step(action)
        self.step_counter += 1

        # Start a new clip when we cross the scheduled window start.
        # The first clip starts with the very first simulation step (step 1).
        if self.enable_file_recording and not self.is_recording and self.step_counter >= self._next_record_start_step:
            self._start_recording()

        # Capture a frame only when something needs it this step.
        frame = None
        if self.enable_file_recording and self.is_recording:
            frame = self._capture_rgb_frame()
            if frame is not None:
                self.current_video_frames.append(frame)
                if len(self.current_video_frames) >= self.steps_per_record:
                    self._stop_and_save_recording()
                    # Schedule the next clip from this clip's start, keeping a constant interval.
                    self._next_record_start_step = self._start_step + self.steps_between_records

        # 1. Publish live ROS2 frame if ROS2 stream active
        if self.enable_ros2_stream and self._ros2_pub is not None:
            if frame is None:
                frame = self._capture_rgb_frame()
            if frame is not None:
                self._publish_ros2_frame(frame)

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
        self._start_step = self.step_counter
        self.current_video_frames = []
        logger.info(f"Started MP4 video recording clip #{self.video_count + 1} at step {self.step_counter}...")

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray | None:
        """Duck-typed conversion of a torch tensor / numpy array to a numpy array."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, "cpu") and hasattr(data, "numpy"):
            try:
                return data.cpu().numpy()
            except Exception:
                return None
        return None

    def _capture_rgb_frame(self) -> np.ndarray | None:
        """Captures an RGB frame [H, W, 3] uint8 from a camera sensor or env render."""
        frame = None

        # 1. Preferred: in-scene TiledCamera sensor (present e.g. in the cobot VLA env variant).
        try:
            if getattr(self.env, "scene", None) is not None:
                sensors = getattr(self.env.scene, "sensors", {})
                if "tiled_camera" in sensors:
                    frame = self._to_numpy(self.env.scene.sensors["tiled_camera"].data.output["rgb"])
                    if frame is not None and frame.ndim == 4:
                        frame = frame[0]
        except Exception as e:
            logger.debug(f"TiledCamera frame capture notice: {e}")

        # 2. Fallback: env.render() (requires render_mode='rgb_array', e.g. when running with --video).
        if frame is None:
            try:
                if hasattr(self.env, "render"):
                    rendered = self._to_numpy(self.env.render())
                    if rendered is not None:
                        if rendered.ndim == 4:
                            rendered = rendered[0]
                        if rendered.ndim == 3 and rendered.shape[-1] in (3, 4):
                            frame = rendered[..., :3]
            except Exception as e:
                logger.debug(f"env.render frame capture notice: {e}")

        if frame is None:
            if not self._reported_no_frame_source:
                self._reported_no_frame_source = True
                logger.warning(
                    "No RGB frame source available. Add a 'tiled_camera' sensor to the scene or create the env "
                    "with render_mode='rgb_array' (built-in --video / --enable_cameras)."
                )
            return None

        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _stop_and_save_recording(self) -> None:
        self.is_recording = False

        if len(self.current_video_frames) == 0:
            logger.warning("Clip ended with no captured frames, skipping save.")
            self.current_video_frames = []
            return

        self.video_count += 1
        video_filename = os.path.join(
            self.video_folder, f"sim_clip_{self.video_count:04d}_step_{self._start_step:05d}.mp4"
        )

        if len(self.current_video_frames) == 0:
            logger.warning(f"Clip #{self.video_count}: no frames captured, skipping save.")
            self.current_video_frames = []
            return

        # Try moviepy + ffmpeg first (same encoder stack as gymnasium RecordVideo).
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip = ImageSequenceClip([f for f in self.current_video_frames], fps=float(self.fps))
            clip.write_videofile(video_filename, codec="libx264", audio=False, preset="ultrafast")
            clip.close()
            logger.info(
                f"Saved MP4 clip: {video_filename} ({len(self.current_video_frames)} frames via moviepy/ffmpeg)"
            )
            self.current_video_frames = []
            return
        except Exception as e:
            logger.debug(f"moviepy save failed ({e}), trying OpenCV...")

        # Fallback 1: OpenCV mp4v.
        try:
            import cv2

            h, w = self.current_video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_filename, fourcc, float(self.fps), (w, h))
            for f in self.current_video_frames:
                bgr_frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f
                writer.write(bgr_frame)
            writer.release()
            logger.info(f"Saved MP4 clip: {video_filename} ({len(self.current_video_frames)} frames via OpenCV)")
            self.current_video_frames = []
            return
        except Exception as e:
            logger.debug(f"OpenCV save failed ({e}), trying imageio...")

        # Fallback 2: imageio.
        try:
            import imageio.v2 as imageio

            imageio.mimsave(video_filename, self.current_video_frames, fps=self.fps)
            logger.info(f"Saved MP4 clip: {video_filename} ({len(self.current_video_frames)} frames via imageio)")
            self.current_video_frames = []
            return
        except Exception as e:
            logger.debug(f"imageio save failed ({e}), falling back to raw numpy dump...")

        # Final fallback: raw numpy dump.
        npy_path = video_filename.replace(".mp4", ".npy")
        np.save(npy_path, np.array(self.current_video_frames))
        logger.warning(f"No video encoder available. Saved raw numpy frames to: {npy_path}")
        self.current_video_frames = []

    def close(self) -> None:
        """Releases the ROS2 node and flushes any in-progress recording."""
        if self.is_recording:
            try:
                self._stop_and_save_recording()
            except Exception as e:
                logger.warning(f"Failed to flush in-progress recording on close: {e}")
            self.is_recording = False

        if self._ros2_node is not None:
            try:
                import rclpy

                self._ros2_node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logger.debug(f"ROS2 shutdown notice: {e}")
            self._ros2_node = None
            self._ros2_pub = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
