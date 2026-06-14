# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom IL event functions for coupled asset randomization."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

# Geometry from il_eval_env.py:
_PLATFORM_HALF_HEIGHT = 0.025
_CUBE_HALF_HEIGHT = 0.025  # cube is 0.05 m; half-height = 0.025 (was 0.05 -> spawned 2.5 cm too high)


def reset_cube_on_source_platform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    platform_x_range: tuple[float, float],
    platform_y_range: tuple[float, float],
    platform_z_range: tuple[float, float],
    platform_cfg: SceneEntityCfg = SceneEntityCfg("source_platform"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Randomize source platform pose and place the cube exactly on top.

    All ranges are offsets from each asset's default state, matching the convention of
    ``mdp.reset_root_state_uniform``.  Pass a range of ``(0.0, 0.0)`` for any axis that
    should not move (e.g. z for L1/L2, x/y if only height varies).
    """
    platform = env.scene[platform_cfg.name]
    cube = env.scene[cube_cfg.name]
    n = len(env_ids)
    dev = env.device

    def _sample(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=dev) * (hi - lo) + lo

    # default_root_state stores LOCAL positions (from init_state); write_root_state_to_sim
    # expects WORLD positions, so env_origins must be added — same as mdp.reset_root_state_uniform.
    env_origins = env.scene.env_origins[env_ids]  # (n, 3) world-frame origins

    # Platform: local default + env origin + random local offset → world position.
    plat_state = platform.data.default_root_state[env_ids].clone()
    plat_state[:, :3] += env_origins
    plat_state[:, 0] += _sample(*platform_x_range)
    plat_state[:, 1] += _sample(*platform_y_range)
    plat_state[:, 2] += _sample(*platform_z_range)
    platform.write_root_state_to_sim(plat_state, env_ids=env_ids)

    # Cube sits exactly on top of the platform.
    cube_state = cube.data.default_root_state[env_ids].clone()
    cube_state[:, 0] = plat_state[:, 0]
    cube_state[:, 1] = plat_state[:, 1]
    cube_state[:, 2] = plat_state[:, 2] + _PLATFORM_HALF_HEIGHT + _CUBE_HALF_HEIGHT
    cube.write_root_state_to_sim(cube_state, env_ids=env_ids)
