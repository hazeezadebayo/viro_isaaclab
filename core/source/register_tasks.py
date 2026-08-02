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

# 2. ANYmal-C Quadruped Locomotion Task
gym.register(
    id="Isaac-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.anymal.tasks.anymal_env_cfg:AnymalEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.anymal.agents.rsl_rl_ppo_cfg:AnymalPPORunnerCfg",
    },
)

# 3. AMR Mobile Robot Navigation Task
gym.register(
    id="Isaac-AMR-Navigation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.amr.tasks.amr_env_cfg:AmrEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.amr.agents.rsl_rl_ppo_cfg:AmrPPORunnerCfg",
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

