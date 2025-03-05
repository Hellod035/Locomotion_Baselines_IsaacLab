# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from dataclasses import MISSING


@configclass
class Policy(RslRlPpoActorCriticCfg):
    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_hidden_dims = [256, 256, 128]
    critic_hidden_dims = [256, 256, 128]
    activation = "elu"


@configclass
class Algorithm(RslRlPpoAlgorithmCfg):
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = MISSING
    empirical_normalization = False
    policy: Policy = Policy()
    algorithm: Algorithm = Algorithm()
    logger = "wandb"
    wandb_project = MISSING
