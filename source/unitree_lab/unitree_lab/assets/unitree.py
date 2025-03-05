# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from unitree_lab.assets import ISAAC_ASSET_DIR

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/g1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_.*": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist.*": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                ".*waist.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch.*",
                ".*_shoulder_roll.*",
            ],
            stiffness=100.0,
            damping=2.0,
            armature={
                ".*_shoulder_pitch.*": 0.01,
                ".*_shoulder_roll.*": 0.01,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw.*",
                ".*_elbow_.*",
            ],
            stiffness=50.0,
            damping=2.0,
            armature={
                ".*_shoulder_yaw.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            stiffness=40.0,
            damping=2.0,
            armature={
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)


H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/h1/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch.*": -0.4,  # -16 degrees
            ".*_knee.*": 0.8,  # 45 degrees
            ".*_ankle.*": -0.4,  # -30 degrees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_hip_pitch.*", ".*_knee.*", ".*torso.*"],
            stiffness={
                ".*_hip_yaw.*": 150.0,
                ".*_hip_roll.*": 150.0,
                ".*_hip_pitch.*": 200.0,
                ".*_knee.*": 200.0,
                ".*torso.*": 200.0,
            },
            damping={
                ".*_hip_yaw.*": 5.0,
                ".*_hip_roll.*": 5.0,
                ".*_hip_pitch.*": 5.0,
                ".*_knee.*": 5.0,
                ".*torso.*": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle.*"],
            stiffness={".*_ankle.*": 40.0},
            damping={".*_ankle.*": 2.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*"],
            stiffness={
                ".*_shoulder_pitch.*": 40.0,
                ".*_shoulder_roll.*": 40.0,
                ".*_shoulder_yaw.*": 40.0,
                ".*_elbow.*": 40.0,
            },
            damping={
                ".*_shoulder_pitch.*": 2.0,
                ".*_shoulder_roll.*": 2.0,
                ".*_shoulder_yaw.*": 2.0,
                ".*_elbow.*": 2.0,
            },
        ),
    },
)
"""Configuration for the Unitree H1 Humanoid robot."""
