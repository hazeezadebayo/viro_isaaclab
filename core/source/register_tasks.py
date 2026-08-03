# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import gymnasium as gym

# Monkey-patch gym.make to auto-wrap environments with PeriodicUsdExporterWrapper when USD_EXPORT=1
_original_gym_make = gym.make

def _usd_wrapped_gym_make(*args, **kwargs):
    env = _original_gym_make(*args, **kwargs)
    usd_export_env = os.getenv("USD_EXPORT", "0").lower()
    if usd_export_env in ("1", "true", "yes") or "USD_INTERVAL" in os.environ:
        try:
            from core.utils.usd_exporter import PeriodicUsdExporterWrapper
            interval_s = float(os.getenv("USD_INTERVAL", "1800"))
            length_s = float(os.getenv("USD_LENGTH", "10"))
            out_dir = os.getenv("USD_OUT_DIR", "/workspace/core/logs/usd")
            env = PeriodicUsdExporterWrapper(
                env,
                out_dir=out_dir,
                record_interval_s=interval_s,
                clip_length_s=length_s,
                convert_to_mp4=True,
            )
        except Exception as e:
            print(f"[WARN] Failed to auto-wrap environment with PeriodicUsdExporterWrapper: {e}")
    return env

gym.make = _usd_wrapped_gym_make

# 1. Humanoid Motion Capture Imitation Task
gym.register(
    id="Isaac-Humanoid-Imitation-v0",
    entry_point="core.source.humanoid.tasks.humanoid_imitation_env_cfg:HumanoidImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.humanoid.tasks.humanoid_imitation_env_cfg:HumanoidImitationEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.humanoid.agents.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
    },
)

# 2. ANYmal-C Quadruped Locomotion Tasks
gym.register(
    id="Isaac-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_locomotion_rough_env_cfg:AnymalLocomotionRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCPositionRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Anymal-C-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_locomotion_rough_env_cfg:AnymalLocomotionRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCPositionRoughPPORunnerWithSymmetryCfg",
    },
)

gym.register(
    id="Isaac-Anymal-C-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_locomotion_flat_env_cfg:AnymalLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCPositionFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Anymal-C-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_locomotion_flat_env_cfg:AnymalLocomotionFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCPositionFlatPPORunnerWithSymmetryCfg",
    },
)

# 2b. ANYmal-C Local Navigation Tasks
gym.register(
    id="Isaac-Anymal-C-Navigation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_navigation_rough_env_cfg:AnymalNavigationRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCNavigationRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Anymal-C-Navigation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_navigation_rough_env_cfg:AnymalNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCNavigationRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Anymal-C-Navigation-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_navigation_flat_env_cfg:AnymalNavigationFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCNavigationRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Anymal-C-Navigation-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_navigation_flat_env_cfg:AnymalNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalCNavigationRoughPPORunnerCfg",
    },
)

# 3. AMR TurtleBot3 Burger Differential-Drive Tasks
gym.register(
    id="Isaac-AMR-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.amr.tasks.amr_locomotion_env_cfg:AmrLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.amr.agents.rsl_rl_ppo_cfg:AmrLocomotionPPORunnerCfg",
    },
)

# 3b. AMR Navigation Task (hierarchical: frozen pre-trained locomotion policy below)
gym.register(
    id="Isaac-AMR-Navigation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.amr.tasks.amr_navigation_env_cfg:AmrNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.amr.agents.rsl_rl_ppo_cfg:AmrNavigationPPORunnerCfg",
    },
)

# 3c. AMR Traversability Task (camera-based, figure-8 white path on black terrain)
gym.register(
    id="Isaac-AMR-Traversability-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.amr.tasks.amr_traversability_env_cfg:AmrTraversabilityEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.amr.agents.rsl_rl_ppo_cfg:AmrTraversabilityPPORunnerCfg",
    },
)

# 4. Cobot 6-DOF Manipulator Arm Target Reaching Task
gym.register(
    id="Isaac-Lift-Cylinder-Cobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.cobot.tasks.joint_pos2_cobot_env_cfg:CobotUR5eCylinderLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.cobot.agents.rsl_rl_ppo_cfg:CobotCylinderPPORunnerCfg",
    },
)

# domain randomization is turned off.
gym.register(
    id="Isaac-Lift-Cylinder-Cobot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.cobot.tasks.joint_pos2_cobot_env_cfg:CobotUR5eCylinderLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "core.source.cobot.agents.rsl_rl_ppo_cfg:CobotCylinderPPORunnerCfg",
    },
)

