# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.sim import (
    CylinderCfg,
    CuboidCfg,
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
    PreviewSurfaceCfg,
)
from .. import mdp
from .lift3_env_cobot_cfg import BaseLiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

COBOT_UR5E_USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../descriptions/ur5e_allegro_R_A2.usd")
)

COBOT_UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=COBOT_UR5E_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            "joint_0_0": 0.0,
            "joint_1_0": 0.0,
            "joint_2_0": 0.0,
            "joint_3_0": 0.0,
            "joint_4_0": 0.0,
            "joint_5_0": 0.0,
            "joint_6_0": 0.0,
            "joint_7_0": 0.0,
            "joint_8_0": 0.0,
            "joint_9_0": 0.0,
            "joint_10_0": 0.0,
            "joint_11_0": 0.0,
            "joint_12_0": 0.0,
            "joint_13_0": 0.0,
            "joint_14_0": 0.0,
            "joint_15_0": 0.0,
        },
        pos=[0.0, 0.0, 0.0],
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint",
                              "elbow_joint","wrist_1_joint",
                              "wrist_2_joint", "wrist_3_joint"],
            effort_limit_sim={
                "shoulder_(pan|lift)_joint": 330.0,
                "elbow_joint": 150.0,
                "wrist_(1|2|3)_joint": 54.0,
            },
            stiffness={
                "shoulder_(pan|lift)_joint": 13000.0, # 800
                "elbow_joint": 10000.0, # 800
                "wrist_1_joint": 2300.0, # 800
                "wrist_(2|3)_joint": 200.0, # 800
            },
            damping={
                "shoulder_(pan|lift)_joint": 40.0,
                "elbow_joint": 40.0,
                "wrist_(1|2|3)_joint": 40.0,
            },
        ),
        "hand_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit_sim= {"joint_.*": 1.24,},
            stiffness={"joint_.*": 20.0},
            damping={"joint_.*": 2.0},
            armature={"joint_.*": 0.0001},
            friction={"joint_.*": 0.02},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the custom arm."""

@configclass
class CobotUR5eCylinderLift2EnvCfg(BaseLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = COBOT_UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint",
                         "elbow_joint", "wrist_1_joint",
                         "wrist_2_joint", "wrist_3_joint"],
            scale=0.5,
            use_default_offset=True
        )
        self.actions.hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            scale=0.5,
            use_default_offset=True
        )
        # Set the body name for the end effector
        #self.commands.object_pose.body_name = "panda_hand"

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.67, 0.25, 0.2], rot=[1, 0, 0, 0]),
            spawn=CylinderCfg(
                radius=0.025,
                height=0.2,
                axis="Z",  # optional, usually "Z" by default
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=CollisionPropertiesCfg(),
            ),
        )
        
        self.scene.dummy_table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.62, 0.25, 0.05],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=CuboidCfg(
                size=(0.4, 0.4, 0.1),  
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),  # RGB in [0,1]
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(
                    mass=100.0,
                ),
                collision_props=CollisionPropertiesCfg(),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/palm_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.05, 0.0, 0.01],
                    ),
                ),
            ],
        )


@configclass
class CobotUR5eCylinderLift2EnvCfg_PLAY(CobotUR5eCylinderLift2EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
