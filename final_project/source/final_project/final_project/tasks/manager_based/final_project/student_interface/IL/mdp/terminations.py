# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success termination for IL pick-place L1.

Required for two reasons:
  * the data collector flushes only successful episodes, and
  * Isaac Lab's robomimic play.py reads ``env_cfg.terminations.success`` to score
    rollouts (scripts/imitation_learning/robomimic/play.py).
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import constants


def place_success(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_platform"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    pos_threshold: float = constants.PLACE_POS_THRESHOLD,
    finger_open_threshold: float = constants.FINGER_OPEN_THRESHOLD,
    min_hold_steps: int = constants.HOLD_STEPS,
) -> torch.Tensor:
    """Episode succeeds when the cube rests on the target platform AND the gripper is
    open, held for ``min_hold_steps`` consecutive steps.

    The placement point is one platform-half + one cube-half above the target
    platform's root center — i.e. a cube sitting on top of the platform.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    placement_point = target.data.root_pos_w.clone()
    placement_point[:, 2] += constants.PLATFORM_HALF + constants.CUBE_HALF

    cube_at_goal = torch.norm(cube.data.root_pos_w - placement_point, dim=-1) < pos_threshold

    finger_ids = robot_cfg.joint_ids
    if finger_ids is None or isinstance(finger_ids, slice):
        finger_ids = [i for i, n in enumerate(robot.data.joint_names) if "finger" in n]
    finger_pos = robot.data.joint_pos[:, finger_ids].mean(dim=-1)
    gripper_open = finger_pos > finger_open_threshold

    instant_ok = cube_at_goal & gripper_open

    # Per-env consecutive-success counter, stored on the env across steps.
    if not hasattr(env, "_place_success_counter") or env._place_success_counter.shape[0] != env.num_envs:
        env._place_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    just_reset = env.episode_length_buf == 0
    env._place_success_counter = torch.where(
        just_reset, torch.zeros_like(env._place_success_counter), env._place_success_counter
    )
    env._place_success_counter = torch.where(
        instant_ok, env._place_success_counter + 1, torch.zeros_like(env._place_success_counter)
    )

    return env._place_success_counter >= min_hold_steps
