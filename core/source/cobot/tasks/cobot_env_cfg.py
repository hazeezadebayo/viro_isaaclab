# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Cobot 6-DOF Manipulator Arm Target Reaching Environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as isaac_mdp
from .. import mdp

# Universal UR5 / Cobot Arm Articulation Asset Definition
COBOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR5/ur5.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -1.57,
            "joint_3": 1.57,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
        },
    ),
    actuators={
        "arm_joints": sim_utils.IdealPDActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


##
# Scene definition
##


@configclass
class CobotSceneCfg(InteractiveSceneCfg):
    """Configuration for scene with 6-DOF Cobot arm and camera sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # Cobot Arm Asset
    robot = COBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # In-scene camera sensor for periodic video recording & ROS2 live streaming
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.5, 0.0, 1.2), rot=(0.92388, 0.0, 0.38268, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)
        ),
        width=640,
        height=480,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for Cobot joint position control."""

    arm_joints = mdp.actions.CobotArmActionCfg(
        asset_name="robot",
        joint_names=["joint_.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for Cobot policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for cobot reaching policy."""

        joint_pos_norm = ObsTerm(func=isaac_mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=isaac_mdp.joint_vel_rel, scale=0.1)
        target_displacement = ObsTerm(
            func=mdp.observations.end_effector_target_error,
            params={"target_pos": (0.4, 0.0, 0.4), "ee_body_name": "link_6"},
        )
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for simulation events and joint resets."""

    reset_joints = EventTerm(
        func=mdp.events.reset_cobot_joints_uniform,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-0.05, 0.05)},
    )


@configclass
class RewardsCfg:
    """Reward terms for Cobot 3D target reaching task."""

    target_proximity = RewTerm(
        func=mdp.rewards.end_effector_proximity_reward,
        weight=2.0,
        params={"target_pos": (0.4, 0.0, 0.4), "std": 0.2, "ee_body_name": "link_6"},
    )
    reach_bonus = RewTerm(
        func=mdp.rewards.end_effector_reach_bonus,
        weight=5.0,
        params={"target_pos": (0.4, 0.0, 0.4), "threshold": 0.05, "ee_body_name": "link_6"},
    )
    action_l2 = RewTerm(func=isaac_mdp.action_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for episode resets."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class CobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Cobot 6-DOF Manipulator Target Reaching Environment."""

    scene: CobotSceneCfg = CobotSceneCfg(num_envs=4096, env_spacing=3.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
