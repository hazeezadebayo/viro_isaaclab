#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamic Multi-Robot ROS2 State Publisher Launch File.

Supports Humanoid, ANYmal, and AMR mobile robots. Reads the target robot's URDF/Xacro
description file and publishes the complete coordinate transform tree (/tf and /tf_static).

Usage:
    ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=humanoid
    ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=anymal
    ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=amr
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    robot_type = LaunchConfiguration("robot_type").perform(context).lower()
    frame_prefix = LaunchConfiguration("frame_prefix").perform(context)

    # Base source path
    base_source_dir = "/workspace/core/source"
    if not os.path.exists(base_source_dir):
        base_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source"))

    # Resolve URDF file path based on robot type
    if robot_type == "humanoid":
        urdf_path = os.path.join(base_source_dir, "humanoid", "descriptions", "humanoid.urdf")
    elif robot_type == "anymal":
        urdf_path = os.path.join(base_source_dir, "anymal", "descriptions", "anymal_c.urdf")
    elif robot_type == "amr":
        urdf_path = os.path.join(base_source_dir, "amr", "descriptions", "turtlebot3", "model.urdf.xacro")
    elif robot_type == "cobot":
        urdf_path = os.path.join(base_source_dir, "cobot", "descriptions", "urdf", "cobot.xacro")
    else:
        urdf_path = os.path.join(base_source_dir, "amr", "descriptions", "turtlebot3", "model.urdf.xacro")


    # Read URDF XML content if file exists, fallback to basic description
    if os.path.exists(urdf_path):
        with open(urdf_path, "r") as f:
            robot_desc_content = f.read()
    else:
        robot_desc_content = f"<?xml version='1.0'?><robot name='{robot_type}'></robot>"

    robot_description = {"robot_description": robot_desc_content}

    # robot_state_publisher node - publishes URDF kinematic frames to /tf
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=robot_type,
        output="both",
        parameters=[robot_description, {"frame_prefix": frame_prefix, "use_sim_time": True}],
    )

    return [rsp_node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_type",
            default_value="amr",
            description="Target robot model type: 'humanoid', 'anymal', or 'amr'",
        ),
        DeclareLaunchArgument(
            "frame_prefix",
            default_value="",
            description="Optional frame prefix for multi-robot coordinate trees",
        ),
        OpaqueFunction(function=launch_setup),
    ])
