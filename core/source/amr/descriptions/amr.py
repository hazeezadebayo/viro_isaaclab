# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the TurtleBot3 Burger AMR asset.

The robot is loaded from the local self-contained URDF (``descriptions/turtlebot3/model.urdf``)
so the asset is fully offline and independent of the Isaac Lab nuclei assets. The URDF defines
mass/inertia and collision geometry for every link, and wheels are differential-drive joints
(``wheel_left_joint`` / ``wheel_right_joint``) with matching joint origins so positive joint
velocity drives the robot forward along +x.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

_AMR_URDF_DIR = os.path.dirname(os.path.abspath(__file__))
_AMR_URDF_PATH = os.path.join(_AMR_URDF_DIR, "turtlebot3", "model.urdf").replace("\\", "/")

AMR_BURGER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{_AMR_URDF_PATH}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"wheel_left_joint": 0.0, "wheel_right_joint": 0.0},
    ),
    actuators={
        "wheels": sim_utils.IdealPDActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)
