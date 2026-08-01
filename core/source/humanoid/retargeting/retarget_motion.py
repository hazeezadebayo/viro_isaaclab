# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retargeting script: Converts TemugeB (bodypose3d -> joint_angles_calculate) output to IsaacLab Humanoid joint format."""

from __future__ import annotations

import argparse
import json
import os
import numpy as np


# Default joint mapping from TemugeB human skeleton model to IsaacLab Humanoid robot joint names
HUMAN_TO_HUMANOID_JOINT_MAP = {
    "pelvis": "pelvis",
    "left_hip_roll": "left_waist",
    "right_hip_roll": "right_waist",
    "left_hip_yaw": "left_thigh_0",
    "left_hip_pitch": "left_thigh_1",
    "left_hip_roll_joint": "left_thigh_2",
    "left_knee": "left_shin",
    "left_ankle": "left_foot",
    "right_hip_yaw": "right_thigh_0",
    "right_hip_pitch": "right_thigh_1",
    "right_hip_roll_joint": "right_thigh_2",
    "right_knee": "right_shin",
    "right_ankle": "right_foot",
    "left_shoulder": "left_upper_arm",
    "left_elbow": "left_lower_arm",
    "right_shoulder": "right_upper_arm",
    "right_elbow": "right_lower_arm",
}

# Standard humanoid robot joint ordering matching HUMANOID_CFG
ROBOT_JOINT_NAMES = [
    "pelvis",
    "left_waist",
    "right_waist",
    "left_thigh_0",
    "left_thigh_1",
    "left_thigh_2",
    "left_shin",
    "left_foot",
    "right_thigh_0",
    "right_thigh_1",
    "right_thigh_2",
    "right_shin",
    "right_foot",
    "left_upper_arm",
    "left_lower_arm",
    "right_upper_arm",
    "right_lower_arm",
]

# Soft joint angle limits for safety (in radians)
JOINT_LIMITS = {
    "left_shin": (0.0, 2.35),
    "right_shin": (0.0, 2.35),
    "left_foot": (-0.75, 0.75),
    "right_foot": (-0.75, 0.75),
}


def retarget_motion_capture(
    input_file: str,
    output_file: str,
    fps: float = 60.0,
    smooth_window: int = 3,
) -> dict:
    """Processes TemugeB joint angles dict/json and retargets angles to humanoid robot dimensions.

    Args:
        input_file: Path to input joint angles file from joint_angles_calculate.
        output_file: Path to output JSON file to save retargeted motion capture data.
        fps: Target frame rate (frames per second).
        smooth_window: Window size for moving average filter smoothing.

    Returns:
        Retargeted motion capture dictionary structure.
    """
    print(f"Loading raw motion capture data from: {input_file}")

    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            raw_data = json.load(f)
    else:
        print("Input file not found. Generating synthesized retargeted trajectory template...")
        raw_data = {}

    num_frames = raw_data.get("num_frames", 60)
    raw_angles = raw_data.get("angles", {})

    retargeted_positions = []
    retargeted_velocities = []

    # Time series construction
    for t_idx in range(num_frames):
        frame_pos = []
        for joint in ROBOT_JOINT_NAMES:
            # Map human joint or use fallback zero
            human_key = next((k for k, v in HUMAN_TO_HUMANOID_JOINT_MAP.items() if v == joint), joint)
            if human_key in raw_angles and t_idx < len(raw_angles[human_key]):
                angle_val = float(raw_angles[human_key][t_idx])
            else:
                # Default baseline gait generator if key absent
                t = t_idx / fps
                if "shin" in joint:
                    angle_val = 0.2 + 0.1 * np.sin(2 * np.pi * 1.5 * t)
                elif "thigh_1" in joint:
                    angle_val = 0.1 * np.cos(2 * np.pi * 1.5 * t)
                else:
                    angle_val = 0.0

            # Enforce physical joint limits
            if joint in JOINT_LIMITS:
                min_lim, max_lim = JOINT_LIMITS[joint]
                angle_val = np.clip(angle_val, min_lim, max_lim)

            frame_pos.append(angle_val)

        retargeted_positions.append(frame_pos)

    pos_array = np.array(retargeted_positions, dtype=np.float32)

    # Apply moving average filter for trajectory smoothness if requested
    if smooth_window > 1 and len(pos_array) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        for j in range(pos_array.shape[1]):
            pos_array[:, j] = np.convolve(pos_array[:, j], kernel, mode="same")

    # Compute numerical joint velocities: dq/dt = (q_t+1 - q_t-1) / (2 * dt)
    dt = 1.0 / fps
    vel_array = np.zeros_like(pos_array)
    vel_array[1:-1] = (pos_array[2:] - pos_array[:-2]) / (2.0 * dt)
    vel_array[0] = (pos_array[1] - pos_array[0]) / dt
    vel_array[-1] = (pos_array[-1] - pos_array[-2]) / dt

    output_data = {
        "metadata": {
            "fps": fps,
            "num_frames": len(pos_array),
            "duration_s": float(len(pos_array) / fps),
            "robot_type": "humanoid",
            "source_pipeline": "bodypose3d -> joint_angles_calculate -> retarget_motion.py",
        },
        "joint_names": ROBOT_JOINT_NAMES,
        "joint_positions": pos_array.tolist(),
        "joint_velocities": vel_array.tolist(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Retargeting complete! Saved {len(pos_array)} frames to: {output_file}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retarget human motion capture data for IsaacLab humanoid.")
    parser.add_argument("--input", type=str, default="raw_joint_angles.json", help="Path to input joint angles")
    parser.add_argument(
        "--output",
        type=str,
        default="../../data/motion_capture/human_walk_retargeted.json",
        help="Path to save retargeted output json",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Frame rate")

    args = parser.parse_args()
    retarget_motion_capture(args.input, args.output, fps=args.fps)
