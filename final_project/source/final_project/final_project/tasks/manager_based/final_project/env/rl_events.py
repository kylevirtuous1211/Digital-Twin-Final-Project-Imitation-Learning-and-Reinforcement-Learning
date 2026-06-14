# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RL event functions for Level 3 smart obstacle placement."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

# Center of the goal position distribution (env-local).
# UniformPose2dCommandCfg samples pos_y in [2.0, 3.0] -> center = 2.5.
_GOAL_LOCAL_Y_CENTER = 2.5

# Half-width of the random y window around the robot-goal midpoint (m).
_Y_HALF_RANGE = 0.5


def reset_obstacle_between_robot_and_goal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_names: tuple[str, ...] = ("obstacle_box", "obstacle_klt"),
) -> None:
    """Randomly pick one USD obstacle per env and place it on the robot's straight-line path.

    The obstacle x is aligned with the robot's current x (read after ``reset_robot``
    fires) so a naive straight-line policy is always blocked.  A small random jitter
    is added to prevent the policy from memorising a fixed offset.

    Multi-env safety
    ----------------
    All local coordinates are converted to world frame via ``env.scene.env_origins``
    before writing to sim, keeping every env's obstacle inside its own arena.

    Inactive obstacle
    -----------------
    The non-chosen obstacle is hidden via ``set_visibility(False)``; its physics
    position is already restored off-arena by ``reset_all`` (reset_scene_to_default).

    Placement (env-local coordinates)
    -----------
        obstacle x  — robot's local x (always blocks straight-line path)
        obstacle y  — midpoint(robot_y, goal_y_center) ± _Y_HALF_RANGE
        obstacle z  — preserved from default_root_state
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    dev = env.device

    # World-frame origins for the resetting envs, shape (n, 3)
    env_origins = env.scene.env_origins[env_ids]

    # Robot x aligned in world frame (reset_robot fires before this event)
    robot_pos_w = env.scene["robot"].data.root_pos_w[env_ids]  # (n, 3)
    local_robot_x = robot_pos_w[:, 0] - env_origins[:, 0]

    # Randomly assign one obstacle variant per env
    choice = torch.randint(0, len(obstacle_names), (n,), device=dev)

    # Obstacle x: exactly on the robot's path (no x jitter — always blocks straight line)
    local_x = local_robot_x

    # Obstacle y: midpoint between robot's actual y and the goal distribution center,
    # with a random window of ±_Y_HALF_RANGE so it's never too close to either end.
    local_robot_y = robot_pos_w[:, 1] - env_origins[:, 1]
    local_y_center = (local_robot_y + _GOAL_LOCAL_Y_CENTER) / 2.0
    local_y = local_y_center + (torch.rand(n, device=dev) - 0.5) * (2 * _Y_HALF_RANGE)

    # Convert to world frame
    world_x = env_origins[:, 0] + local_x
    world_y = env_origins[:, 1] + local_y

    for i, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        active_mask = choice == i

        # Active envs: reposition in corridor + make visible
        if active_mask.any():
            active_env_ids = env_ids[active_mask]
            root_state = obstacle.data.default_root_state[active_env_ids].clone()
            root_state[:, 0] = world_x[active_mask]
            root_state[:, 1] = world_y[active_mask]
            obstacle.write_root_state_to_sim(root_state, env_ids=active_env_ids)
            obstacle.set_visibility(True, env_ids=active_env_ids)

        # Inactive envs: hide (reset_all already restored physics position)
        inactive_mask = ~active_mask
        if inactive_mask.any():
            obstacle.set_visibility(False, env_ids=env_ids[inactive_mask])
