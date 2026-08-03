# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import glob

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import ConeCfg, PreviewSurfaceCfg, CollisionPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

import core.source.anymal.mdp.navigation as mdp
from core.source.anymal.tasks.anymal_locomotion_rough_env_cfg import AnymalLocomotionRoughEnvCfg, MySceneCfg as LocomotionSceneCfg
import core.source.anymal.mdp.navigation.commands as obstacle_cmd


def get_locomotion_policy_path(flat: bool = False) -> str:
    """Dynamically resolves the locomotion policy path.
    Searches for exported/policy.pt under recent run folders or falls back to pretrained/policy.pt.
    """
    experiment_name = "anymal_c_flat" if flat else "anymal_c_rough"
    base_log_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "logs",
            "rsl_rl",
            experiment_name,
        )
    )

    if os.path.exists(base_log_dir):
        run_dirs = [
            os.path.join(base_log_dir, d)
            for d in os.listdir(base_log_dir)
            if os.path.isdir(os.path.join(base_log_dir, d)) and d != "pretrained"
        ]
        if run_dirs:
            run_dirs.sort()
            for latest_run_dir in reversed(run_dirs):
                exported_path = os.path.join(latest_run_dir, "exported", "policy.pt")
                if os.path.exists(exported_path):
                    print(f"[INFO] Resolved ANYmal locomotion policy dynamically to: {exported_path}")
                    return exported_path

    fallback_path = os.path.join(base_log_dir, "pretrained", "policy.pt")
    print(f"[INFO] Fallback ANYmal locomotion policy resolved to: {fallback_path}")
    return fallback_path


LOW_LEVEL_ENV_CFG = AnymalLocomotionRoughEnvCfg()
_RESOLVED_POLICY_PATH = get_locomotion_policy_path(flat=False)


@configclass
class NavigationSceneCfg(LocomotionSceneCfg):
    num_envs: int = 2048
    env_spacing: float = 2.5

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
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
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
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
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
        params={"std": 2.0,
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.2,
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot"),
        },
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
    pose_command = obstacle_cmd.ObstacleBlockedPoseCommandCfg(
        asset_name="robot",
        obstacle_cfg=SceneEntityCfg("cone"),
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        goal_distance_behind_obstacle=(1.0, 2.0),
        goal_pose_angle_range=(-1.507, 1.507),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )
    time_remaining = LOW_LEVEL_ENV_CFG.commands.time_remaining


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    goal_distance = CurrTerm(
        func=mdp.distance_level,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "pose_command",
        }
    )
    obstacle_angle = CurrTerm(
        func=mdp.obstacle_angle_level,
        params={"command_name": "pose_command"},
    )


@configclass
class AnymalNavigationRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ANYmal-C rough terrain navigation environment."""
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
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.max_init_terrain_level = None


@configclass
class AnymalNavigationEnvCfg_PLAY(AnymalNavigationRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
