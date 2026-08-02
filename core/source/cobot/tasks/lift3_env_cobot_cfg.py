# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    contact_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_3_0", # /link_3_0_tip
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    contact_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_7_0", # /link_7_0_tip
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    contact_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_11_0", # /link_11_0_tip
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    contact_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_15_0", # /link_15_0_tip
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    #contact_palm = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/Robot/palm_link",
    #    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    #)

    #ground_contact = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/Robot/link_[0-15]_0",
    #    filter_prim_paths_expr=["/World/GroundPlane"],
    #)

    index_tip_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_3_0",
                name="index_tip",
            )
        ],
    )

    middle_tip_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_7_0",
                name="middle_tip",
            )
        ],
    )

    ring_tip_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_11_0",
                name="ring_tip",
            )
        ],
    )

    thumb_tip_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_15_0",
                name="thumb_tip",
            )
        ],
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    hand_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        finger_contacts = ObsTerm(func=mdp.fingertip_contacts, params={"threshold": 0.5},)
        fingertip_cube_vectors = ObsTerm(func=mdp.fingertip_cube_vectors)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=4.0
    )

    finger_contact = RewTerm(
        func=mdp.grasp_reward,
        params={"threshold": 0.5},
        weight=5.0,
    )

    lift_while_grasping = RewTerm(
        func=mdp.lift_while_grasping,
        params={
                "minimal_height": 0.2,
                "lifting_height": 0.1,
                "threshold": 0.5,
                },
        weight=200.0
    )

    cube_is_moved= RewTerm(
        func=mdp.cube_is_moved,
        params={
            "lift_threshold": 0.17,
            "vel_threshold": 0.02
            },
        weight=-5.0
    )

    finger_tip_proximity = RewTerm(
        func=mdp.thumb_opposition_reward2,
        params={"std": 0.03},
        weight=100.0,
    )

    finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint_1_0",
                    "joint_2_0",
                    "joint_3_0",
                    "joint_5_0",
                    "joint_6_0",
                    "joint_7_0",
                    "joint_9_0",
                    "joint_10_0",
                    "joint_11_0",
                    "joint_12_0",
                    "joint_14_0",
                    "joint_15_0",
                ],
            ),
            "std": 0.3
        },
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-4e-5)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.145, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    reaching_object = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "reaching_object", "weight": 2.0, "num_steps": 40000}
    )

    finger_closure = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "finger_closure", "weight": 50, "num_steps": 40000}
    )

##
# Environment configuration
##


@configclass
class BaseLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.friction_correlation_distance = 0.00625

        # Increase patch buffer
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024

        # Often increased together
        self.sim.physx.gpu_max_rigid_contact_count = 4 * 1024 * 1024

        self.sim.physx.gpu_total_aggregate_pairs_capacity = 131072