# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and Environment definition for Humanoid Reference Motion Imitation in IsaacLab."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from ..mdp.pd_tracking_action import PDTrackingActionCfg
from ..mdp.motion_loader import ReferenceMotionLoader


##
# Scene definition
##


@configclass
class HumanoidImitationSceneCfg(InteractiveSceneCfg):
    """Configuration for scene with humanoid robot set up for torque PD control."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot configuration - internal actuator stiffness/damping set to 0 for custom PD action term
    robot = mdp.HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # In-scene camera sensor for periodic video recording (headless Docker visualization)
    tiled_camera = sim_utils.TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/front_cam",
        offset=sim_utils.TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 1.8), rot=(0.92388, 0.0, 0.38268, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)
        ),
        width=640,
        height=480,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )



##
# MDP settings
##


@configclass
class ImitationActionsCfg:
    """Action specifications using PDTrackingActionCfg for imitation learning."""

    pd_tracking = PDTrackingActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        kp=100.0,
        kd=10.0,
        action_scale=0.25,
        clip_effort=200.0,
    )


@configclass
class ImitationObservationsCfg:
    """Observation specifications for motion imitation policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for imitation policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        joint_pos_err = ObsTerm(func=mdp.joint_pos_ref_error)
        joint_vel_err = ObsTerm(func=mdp.joint_vel_ref_error, scale=0.1)
        phase = ObsTerm(func=mdp.motion_phase)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ImitationRewardsCfg:
    """Reward terms for motion tracking imitation learning."""

    # Primary Tracking Rewards
    pose_tracking = RewTerm(func=mdp.joint_position_tracking_reward, weight=1.5, params={"tracking_k": 5.0})
    vel_tracking = RewTerm(func=mdp.joint_velocity_tracking_reward, weight=0.5, params={"tracking_k": 0.1})

    # Regularization & Stability
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.2, params={"threshold": 0.90})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    power = RewTerm(
        func=mdp.power_consumption,
        weight=-0.002,
        params={
            "gear_ratio": {
                ".*": 50.0,
            }
        },
    )


@configclass
class ImitationTerminationsCfg:
    """Termination terms for imitation episode management."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_fall = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.7})


@configclass
class HumanoidImitationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Humanoid Motion Tracking Imitation Environment."""

    scene: HumanoidImitationSceneCfg = HumanoidImitationSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    observations: ImitationObservationsCfg = ImitationObservationsCfg()
    actions: ImitationActionsCfg = ImitationActionsCfg()
    rewards: ImitationRewardsCfg = ImitationRewardsCfg()
    terminations: ImitationTerminationsCfg = ImitationTerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 16.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

        # Motion capture reference file setting
        self.motion_file: str = "human_walk_retargeted.json"


class HumanoidImitationEnv(ManagerBasedRLEnv):
    """Environment class for humanoid motion tracking imitation."""

    cfg: HumanoidImitationEnvCfg

    def __init__(self, cfg: HumanoidImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Initialize reference motion loader manager
        motion_file = getattr(cfg, "motion_file", "human_walk_retargeted.json")
        self.motion_loader = ReferenceMotionLoader(
            motion_file=motion_file,
            num_envs=self.num_envs,
            device=self.device,
            loop=True,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, extras = super().reset(seed=seed, options=options)
        if hasattr(self, "motion_loader") and self.motion_loader is not None:
            self.motion_loader.reset()
        return obs, extras

    def step(self, action: torch.Tensor):
        if hasattr(self, "motion_loader") and self.motion_loader is not None:
            self.motion_loader.step(self.step_dt)
        return super().step(action)
