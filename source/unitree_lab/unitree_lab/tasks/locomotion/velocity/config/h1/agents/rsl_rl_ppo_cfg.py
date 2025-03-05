# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from unitree_lab.tasks.locomotion.velocity.rsl_rl_ppo_cfg import Policy, Algorithm, PPORunnerCfg  # noqa:F401


@configclass
class H1FlatPPORunnerCfg(PPORunnerCfg):
    algorithm = Algorithm(entropy_coef=0.005)
    experiment_name = "h1_flat"
    wandb_project = "h1_flat"


@configclass
class H1RoughPPORunnerCfg(PPORunnerCfg):
    algorithm = Algorithm(entropy_coef=0.005)
    experiment_name = "h1_rough"
    wandb_project = "h1_rough"
