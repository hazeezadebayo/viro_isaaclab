# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYmal-C Quadruped Walking & Locomotion Environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Fallback & combined MDP imports for Quadruped locomotion
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as isaac_mdp
from .. import mdp

# Import ANYmal-C robot asset configuration from IsaacLab assets library
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


##
# Scene definition
##


@configclass
class AnymalSceneCfg(InteractiveSceneCfg):
    """Configuration for terrain scene with an ANYmal-C quadruped robot."""

    # Ground plane terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # ANYmal-C quadruped robot asset (12 actuated joints: LF, RF, LH, RH x HAA, HFE, KFE)
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for ANYmal quadruped MDP."""

    # ANYmal-C joint position targets for HAA, HFE, KFE joints
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*HAA", ".*HFE", ".*KFE"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for ANYmal quadruped policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy network."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=isaac_mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=isaac_mdp.joint_vel_rel, scale=0.1)

        # Incoming wrenches on ANYmal-C foot contacts (LF_FOOT, RF_FOOT, LH_FOOT, RH_FOOT)
        feet_body_forces = ObsTerm(
            func=isaac_mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*FOOT"])},
        )
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for ANYmal simulation events and resets."""

    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for ANYmal-C quadruped locomotion (ANYbotics Gold Standard)."""

    # Primary Locomotion & Heading Tracking
    progress = RewTerm(func=mdp.progress_reward, weight=1.5, params={"target_pos": (1000.0, 0.0, 0.0)})
    alive = RewTerm(func=isaac_mdp.is_alive, weight=2.0)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.3, params={"threshold": 0.90})
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.8, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )

    # Gait Smoothness & Energy Regularization
    action_l2 = RewTerm(func=isaac_mdp.action_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.005)
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.005,
        params={
            "gear_ratio": {
                ".*HAA": 1.0,
                ".*HFE": 1.0,
                ".*KFE": 1.0,
            }
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.95,
            "gear_ratio": {
                ".*HAA": 1.0,
                ".*HFE": 1.0,
                ".*KFE": 1.0,
            },
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for ANYmal episode resets."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    # ANYmal-C nominal standing height is ~0.55m; collapse threshold is 0.35m
    torso_height = DoneTerm(func=isaac_mdp.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class AnymalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ANYmal-C quadruped walking environment."""

    scene: AnymalSceneCfg = AnymalSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0