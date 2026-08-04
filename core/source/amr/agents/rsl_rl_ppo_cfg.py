# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configurations for the AMR locomotion, navigation and traversability tasks."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AmrLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the AMR velocity-tracking locomotion task."""

    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 50
    experiment_name = "amr_locomotion"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AmrNavigationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the AMR local navigation task."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "amr_navigation"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        noise_std_type="log",
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AmrTraversabilityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the AMR camera-based traversability task."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "amr_traversability"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
