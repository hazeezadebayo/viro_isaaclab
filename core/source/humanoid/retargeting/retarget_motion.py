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

# TemugeB human joint angle vectors are ZXY Euler angles [thetaz, thetax, thetay].
# Map each humanoid joint to (human joint, Euler component index) with
# 0 = thetaz (yaw), 1 = thetax (roll), 2 = thetay (pitch).
JOINT_ANGLE_MAP = {
    "pelvis": ("hips", 0),
    "left_waist": ("lefthip", 2),
    "right_waist": ("righthip", 2),
    "left_thigh_0": ("lefthip", 0),
    "left_thigh_1": ("lefthip", 2),
    "left_thigh_2": ("lefthip", 1),
    "left_shin": ("leftknee", 2),
    "left_foot": ("leftfoot", 2),
    "right_thigh_0": ("righthip", 0),
    "right_thigh_1": ("righthip", 2),
    "right_thigh_2": ("righthip", 1),
    "right_shin": ("rightknee", 2),
    "right_foot": ("rightfoot", 2),
    "left_upper_arm": ("leftshoulder", 2),
    "left_lower_arm": ("leftelbow", 2),
    "right_upper_arm": ("rightshoulder", 2),
    "right_lower_arm": ("rightelbow", 2),
}


def retarget_motion_capture(
    input_file: str,
    output_file: str,
    fps: float = 60.0,
    smooth_window: int = 3,
) -> dict:
    """Processes TemugeB joint angles json and retargets angles to humanoid robot dimensions.

    Args:
        input_file: Path to raw_angles.json produced by calculate_joint_angles.py.
        output_file: Path to output JSON file to save retargeted motion capture data.
        fps: Target frame rate (frames per second).
        smooth_window: Window size for moving average filter smoothing.

    Returns:
        Retargeted motion capture dictionary structure.

    Raises:
        FileNotFoundError: If the input joint angles file does not exist.
        ValueError: If the input file contains no frames.
    """
    print(f"Loading raw motion capture data from: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input joint angles file not found: {input_file}. "
            "Run the joint angle calculation stage before retargeting."
        )

    with open(input_file, "r") as f:
        raw_data = json.load(f)

    raw_angles = raw_data.get("angles", {})
    num_frames = int(raw_data.get("num_frames", 0))
    if num_frames <= 0:
        raise ValueError(f"No frames in input joint angles file: {input_file}")

    retargeted_positions = []

    # Time series construction
    for t_idx in range(num_frames):
        frame_pos = []
        for joint in ROBOT_JOINT_NAMES:
            angle_val = 0.0
            if joint in JOINT_ANGLE_MAP:
                human_joint, component = JOINT_ANGLE_MAP[joint]
                series = raw_angles.get(human_joint)
                if series and t_idx < len(series) and len(series[t_idx]) > component:
                    angle_val = float(series[t_idx][component])

            # Enforce physical joint limits
            if joint in JOINT_LIMITS:
                min_lim, max_lim = JOINT_LIMITS[joint]
                angle_val = float(np.clip(angle_val, min_lim, max_lim))

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
        default="../../../data/motion_capture/human_walk_retargeted.json",
        help="Path to save retargeted output json",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Frame rate")

    args = parser.parse_args()
    retarget_motion_capture(args.input, args.output, fps=args.fps)
