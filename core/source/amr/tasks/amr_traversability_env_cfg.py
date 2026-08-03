# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the AMR TurtleBot3 Burger traversability environment.

The robot navigates on a flat, black ground plane overlaid with a white figure-8 path. A
forward-facing RGB camera is mounted on the robot; the observation is a low-resolution,
thresholded occupancy mask of *where the white path is* (16x12 cells), plus the goal position
in the robot's base frame. The policy must keep the mask "full" (stay on the path) while
driving toward the goal.

This is the vision/teaching task of the AMR suite: it turns raw pixels into a compact binary
mask so an MLP can solve it without a convolutional backbone.
"""

from __future__ import annotations

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

import isaaclab.sim as sim_utils

from core.source.amr.descriptions.amr import AMR_BURGER_CFG

import core.source.amr.mdp.traversability as mdp


@configclass
class AmrTraversabilitySceneCfg(InteractiveSceneCfg):
    """Configuration for the black/white figure-8 path scene."""

    num_envs: int = 1024
    env_spacing: float = 6.5

    # black ground plane (visual color only; physics is a plain plane)
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
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03)),
        debug_vis=False,
    )

    robot: ArticulationCfg = AMR_BURGER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # forward-facing camera mounted on the robot (URDF camera frame at (0.069, -0.047, 0.107),
    # pitched ~15 degrees down so it sees the path ahead of the robot)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera",
        update_period=0.05,
        height=48,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.069, -0.047, 0.107),
            rot=(-0.43046, 0.56099, -0.56099, 0.43046),  # pitch 15 deg down, ROS convention
            convention="ros",
        ),
        return_latest_camera_pose=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{sim_utils.ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    path_goal = mdp.PathGoalCommandCfg(
        resampling_time_range=(5.0, 8.0),
        min_goal_distance=1.0,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    diff_drive = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_name="wheel_left_joint",
        right_wheel_name="wheel_right_joint",
        wheel_radius=0.033,
        wheel_base=0.160,
        max_wheel_vel=11.0,
        max_lin_vel=0.22,
        max_ang_vel=2.84,
        bounding_strategy="clip",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        goal_in_base = ObsTerm(func=mdp.goal_in_base, params={"command_name": "path_goal"})
        path_mask = ObsTerm(
            func=mdp.camera_occupancy_mask,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "mask_height": 16,
                "mask_width": 12,
                "threshold": 0.5,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    reset_base_on_path = EventTerm(
        func=mdp.reset_base_on_path,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_noise": 0.02,
            "yaw_noise": 0.1,
        },
    )
    reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    goal_progress = RewTerm(
        func=mdp.path_goal_progress,
        weight=1.5,
        params={"command_name": "path_goal"},
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached_reward,
        weight=10.0,
        params={"command_name": "path_goal", "threshold": 0.15},
    )
    on_path = RewTerm(
        func=mdp.on_path_reward,
        weight=1.0,
        params={"std": 0.12},
    )

    # -- penalties
    off_path = RewTerm(
        func=mdp.off_path_penalty,
        weight=-1.0,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class AmrTraversabilityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the AMR traversability environment."""

    scene: AmrTraversabilitySceneCfg = AmrTraversabilitySceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 10
        self.episode_length_s = 10.0

        self.sim.dt = 1 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # re-render the cameras right after a reset so the first observation is fresh
        self.rerender_on_reset = True

        # add the visual white path strips to the scene (visual-only assets)
        for i, asset in enumerate(mdp.build_path_assets()):
            setattr(self.scene, f"path_strip_{i:02d}", asset)
