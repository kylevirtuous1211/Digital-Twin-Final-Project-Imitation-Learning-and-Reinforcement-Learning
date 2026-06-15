# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Geometry-REACTIVE pick-place oracle for DAgger relabeling (Isaac-free, torch only).

Maps ANY (eef, cube, goal) geometry to the correct 7-D relative-IK action — robust
to off-manifold states (overshoot, stall) the BC learner visits. Phase is inferred
from geometry (is the cube lifted?), so labels stay coherent with the visited state.
Importable without Isaac Sim (torch + constants only) so it can be unit-tested.
"""
from __future__ import annotations

import torch

APPROACH_CLEAR = 0.10
TRANSIT_CLEAR = 0.08
GRASP_DZ = -0.005
PLACE_CLEAR = 0.005
MAX_STEP = 0.08
MAX_YAW = 0.20
# = constants.PLATFORM_HALF + constants.CUBE_HALF (0.025 + 0.025); hardcoded so this
# module stays Isaac-free (importing mdp.constants pulls in pxr via mdp/__init__).
PLACE_Z_OFFSET = 0.05
GRASP_LIFT_EPS = 0.03  # cube risen this far above its rest z => considered grasped/held


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def reactive_oracle(obs: dict, cube_z0: torch.Tensor, control_yaw: bool) -> torch.Tensor:
    """Correct 7-D action [dx,dy,dz,droll,dpitch,dyaw,gripper] from geometry, per env.

    obs: dict of [N,*] tensors (eef_pos, cube_pos, goal_pos required; *_quat for yaw).
    cube_z0: [N] cube resting z at episode start (to detect lift => grasped).
    """
    eef = obs["eef_pos"]; cube = obs["cube_pos"]; goal = obs["goal_pos"]
    N = eef.shape[0]; dev = eef.device
    grasped = cube[:, 2] > (cube_z0 + GRASP_LIFT_EPS)
    horiz_cube = torch.linalg.vector_norm(eef[:, :2] - cube[:, :2], dim=1)
    horiz_goal = torch.linalg.vector_norm(eef[:, :2] - goal[:, :2], dim=1)
    on_cube_z = cube[:, 2] + GRASP_DZ
    above_cube_z = cube[:, 2] + APPROACH_CLEAR
    place_z = goal[:, 2] + PLACE_Z_OFFSET + PLACE_CLEAR
    transit_z = torch.maximum(cube[:, 2], place_z) + TRANSIT_CLEAR

    target = eef.clone()
    grip = torch.ones(N, device=dev)

    ng = ~grasped
    far = horiz_cube > 0.02
    near = ~far
    descending = near & (eef[:, 2] > on_cube_z + 0.015)
    at_cube = near & (eef[:, 2] <= on_cube_z + 0.015)
    m = ng & far
    target[m] = torch.stack([cube[m, 0], cube[m, 1], above_cube_z[m]], -1); grip[m] = 1.0
    m = ng & descending
    target[m] = torch.stack([cube[m, 0], cube[m, 1], on_cube_z[m]], -1); grip[m] = 1.0
    m = ng & at_cube
    target[m] = torch.stack([cube[m, 0], cube[m, 1], on_cube_z[m]], -1); grip[m] = -1.0

    g = grasped
    lifting = g & (eef[:, 2] < transit_z - 0.02) & (horiz_goal > 0.05)
    over = g & (~lifting) & (horiz_goal > 0.03)
    placing = g & (~lifting) & (horiz_goal <= 0.03) & (eef[:, 2] > place_z + 0.01)
    releasing = g & (~lifting) & (horiz_goal <= 0.03) & (eef[:, 2] <= place_z + 0.01)
    m = lifting
    target[m] = torch.stack([eef[m, 0], eef[m, 1], transit_z[m]], -1); grip[m] = -1.0
    m = over
    target[m] = torch.stack([goal[m, 0], goal[m, 1], transit_z[m]], -1); grip[m] = -1.0
    m = placing
    target[m] = torch.stack([goal[m, 0], goal[m, 1], place_z[m]], -1); grip[m] = -1.0
    m = releasing
    target[m] = torch.stack([goal[m, 0], goal[m, 1], place_z[m]], -1); grip[m] = 1.0

    dpos = target - eef
    n = torch.linalg.vector_norm(dpos, dim=-1, keepdim=True)
    dpos = dpos * torch.where(n > MAX_STEP, MAX_STEP / n.clamp(min=1e-9), torch.ones_like(n))

    act = torch.zeros(N, 7, device=dev)
    act[:, 0:3] = dpos
    act[:, 6] = grip
    if control_yaw and ("goal_quat" in obs) and ("cube_quat" in obs):
        tgt_yaw = yaw_from_quat(obs["goal_quat"]); cube_yaw = yaw_from_quat(obs["cube_quat"])
        half = torch.pi / 4.0
        yaw_err = torch.remainder(tgt_yaw - cube_yaw + half, torch.pi / 2.0) - half
        act[:, 5] = torch.where(grasped, torch.clamp(yaw_err, -MAX_YAW, MAX_YAW), torch.zeros(N, device=dev))
    return act
