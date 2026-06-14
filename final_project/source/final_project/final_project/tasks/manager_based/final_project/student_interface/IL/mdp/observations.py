# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the IL pick-place policy.

The end-effector pose is derived directly from the ``panda_hand`` body rather than
a ``FrameTransformer`` sensor, so these terms work unchanged in the TA evaluation
scene (which carries only the robot, cube, and platforms — no extra sensors).
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from . import constants

# Resolved once per process: the index of the panda_hand body in the robot.
_HAND_BODY_NAME = "panda_hand"


def _hand_pose_w(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame pose of the IK-controlled grasp point (panda_hand + local +Z offset)."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(_HAND_BODY_NAME)
    idx = body_ids[0]
    hand_pos_w = robot.data.body_link_pos_w[:, idx, :]      # (N, 3)
    hand_quat_w = robot.data.body_link_quat_w[:, idx, :]    # (N, 4) wxyz
    offset = torch.tensor([0.0, 0.0, constants.EE_OFFSET_Z], device=env.device).expand(env.num_envs, 3)
    ee_pos_w = hand_pos_w + quat_apply(hand_quat_w, offset)
    return ee_pos_w, hand_quat_w


def ee_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector (grasp point) position in env frame. Shape [N, 3]."""
    ee_pos_w, _ = _hand_pose_w(env, robot_cfg)
    return ee_pos_w - env.scene.env_origins


def ee_quat(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector orientation quaternion (w, x, y, z) in world frame. Shape [N, 4]."""
    _, hand_quat_w = _hand_pose_w(env, robot_cfg)
    return hand_quat_w


def object_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Object position in env frame (world pos minus env origin). Shape [N, 3]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :3] - env.scene.env_origins


def object_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Object orientation quaternion (w, x, y, z) in world frame. Shape [N, 4]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Both Franka finger joint positions (gripper width state), shape [N, 2].

    Lets a BC policy observe open/closed directly instead of inferring it from poses.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ids, _ = robot.find_joints(["panda_finger_.*"])
    return robot.data.joint_pos[:, ids]
