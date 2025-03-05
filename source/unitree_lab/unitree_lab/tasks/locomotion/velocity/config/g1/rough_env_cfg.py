# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from .flat_env_cfg import G1FlatEnvCfg
from unitree_lab.tasks.locomotion.velocity.velocity_env_cfg import SceneCfg, ObservationsCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

import unitree_lab.tasks.locomotion.velocity.mdp as mdp
from unitree_lab.terrains.terrain_generator_cfg import ROUGH_TERRAINS_CFG


@configclass
class RoughSceneCfg(SceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class RoughObservationsCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RoughRewardsCfg:
    """Reward terms for the MDP."""
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll.*"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 500, "max_reward": 400},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"])}
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

##
# Environment configuration
##


@configclass
class G1RoughEnvCfg(G1FlatEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    rewards: RoughRewardsCfg = RoughRewardsCfg()
    scene: RoughSceneCfg = RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RoughObservationsCfg = RoughObservationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False

        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)

        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
