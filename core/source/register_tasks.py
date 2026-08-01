# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym Task Registration for Core Robot Heads (Humanoid, ANYmal, AMR, Cobot)."""

import gymnasium as gym

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
    id="Isaac-Cobot-Reaching-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "core.source.cobot.tasks.cobot_env_cfg:CobotEnvCfg",
        "rsl_rl_cfg_entry_point": "core.source.cobot.agents.rsl_rl_ppo_cfg:CobotPPORunnerCfg",
    },
)
