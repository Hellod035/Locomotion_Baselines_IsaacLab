# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from unitree_lab.tasks.locomotion.velocity.rsl_rl_ppo_cfg import Algorithm, PPORunnerCfg  # noqa:F401
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class Policy(RslRlPpoActorCriticCfg):
    class_name = "ActorCriticRecurrent"
    init_noise_std = 1.0
    actor_hidden_dims = [256, 256, 128]
    critic_hidden_dims = [256, 256, 128]
    activation = "elu"
    rnn_hidden_size = 256
    rnn_num_layers = 1
    rnn_type = "lstm"


@configclass
class G1FlatPPORunnerCfg(PPORunnerCfg):
    algorithm = Algorithm(entropy_coef=0.008)
    policy: Policy = Policy()
    experiment_name = "g1_flat"
    wandb_project = "g1_flat"


@configclass
class G1RoughPPORunnerCfg(PPORunnerCfg):
    algorithm = Algorithm(entropy_coef=0.008)
    policy: Policy = Policy()
    experiment_name = "g1_rough"
    wandb_project = "g1_rough"
