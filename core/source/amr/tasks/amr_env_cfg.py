# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Autonomous Mobile Robot (AMR / Turtlebot3) Navigation Environment."""

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

# AMR TurtleBot3 Burger Articulation Asset Definition
AMR_BURGER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="{ISAACLAB_NUCLEUS_DIR}/Robots/Turtlebot/turtlebot3_burger.urdf",
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


##
# Scene definition
##


@configclass
class AmrSceneCfg(InteractiveSceneCfg):
    """Configuration for terrain scene with AMR mobile robot and visualization camera."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # AMR Mobile Robot Asset
    robot = AMR_BURGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # In-scene camera sensor for periodic video recording in headless Docker
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(-1.5, 0.0, 0.8), rot=(0.92388, 0.0, 0.38268, 0.0), convention="world"),
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
    """Action specifications for AMR differential drive twist control."""

    diff_drive = mdp.actions.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_name="wheel_left_joint",
        right_wheel_name="wheel_right_joint",
        wheel_radius=0.033,
        wheel_base=0.160,
        max_wheel_vel=15.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for AMR mobile robot policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for mobile robot navigation policy."""

        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        target_displacement = ObsTerm(func=mdp.observations.target_position_error_b, params={"target_pos": (5.0, 0.0, 0.0)})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for simulation events and resets."""

    reset_amr_pos = EventTerm(
        func=mdp.events.reset_amr_position_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for AMR navigation task."""

    target_proximity = RewTerm(
        func=mdp.rewards.position_tracking_fine_reward,
        weight=2.0,
        params={"target_pos": (5.0, 0.0, 0.0), "std": 2.0},
    )
    reach_bonus = RewTerm(
        func=mdp.rewards.reach_target_reward,
        weight=5.0,
        params={"target_pos": (5.0, 0.0, 0.0), "threshold": 0.3},
    )
    action_l2 = RewTerm(func=isaac_mdp.action_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for AMR episode resets."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class AmrEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Autonomous Mobile Robot Navigation Environment."""

    scene: AmrSceneCfg = AmrSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 0.8
        self.sim.physics_material.dynamic_friction = 0.8
        self.sim.physics_material.restitution = 0.0
