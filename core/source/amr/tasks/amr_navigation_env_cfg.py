# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the AMR TurtleBot3 Burger local navigation environment.

The high-level navigation policy is trained on top of a frozen pre-trained velocity-tracking
locomotion policy (see :class:`PreTrainedPolicyAction`). Its actions are the low-level velocity
commands (``[v_x, omega_z]``) that the low-level policy must track, so the hierarchical task
mimics the ANYmal navigation architecture with a differential-drive AMR.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import CollisionPropertiesCfg, ConeCfg, PreviewSurfaceCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import core.source.amr.mdp.navigation as mdp
from core.source.amr.mdp.navigation.policy_path import get_locomotion_policy_path
from core.source.amr.tasks.amr_locomotion_env_cfg import AmrLocomotionEnvCfg, MySceneCfg as LocomotionSceneCfg

LOW_LEVEL_ENV_CFG = AmrLocomotionEnvCfg()
_RESOLVED_POLICY_PATH = get_locomotion_policy_path()


@configclass
class NavigationSceneCfg(LocomotionSceneCfg):
    num_envs: int = 2048
    env_spacing: float = 2.5

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    cone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cone",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1.5, 0.0, 0.5], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=ConeCfg(
            radius=0.3,
            height=1.0,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            collision_props=CollisionPropertiesCfg(),
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_cone_pos = EventTerm(
        func=mdp.reset_cone_pos_donut,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cone"),
            "radius_range": (1.5, 2.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=_RESOLVED_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.diff_drive,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        cone_pos = ObsTerm(
            func=mdp.cone_position_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "cone_cfg": SceneEntityCfg("cone"),
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 2.0, "command_name": "pose_command", "asset_cfg": SceneEntityCfg("robot")},
    )

    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.2, "command_name": "pose_command", "asset_cfg": SceneEntityCfg("robot")},
    )

    cone_too_close = RewTerm(
        func=mdp.cone_proximity_penalty,
        weight=-5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cone_cfg": SceneEntityCfg("cone"),
            "threshold": 0.6,
        },
    )
    progress = RewTerm(
        func=mdp.progress_toward_goal,
        weight=0.5,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.ObstacleBlockedPoseCommandCfg(
        asset_name="robot",
        obstacle_cfg=SceneEntityCfg("cone"),
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        goal_distance_behind_obstacle=(1.0, 2.0),
        goal_pose_angle_range=(-1.507, 1.507),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    goal_distance = CurrTerm(
        func=mdp.distance_level,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "pose_command",
        },
    )
    obstacle_angle = CurrTerm(
        func=mdp.obstacle_angle_level,
        params={"command_name": "pose_command"},
    )


@configclass
class AmrNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the AMR local navigation environment."""

    scene: NavigationSceneCfg = NavigationSceneCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 4
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.max_init_terrain_level = None
