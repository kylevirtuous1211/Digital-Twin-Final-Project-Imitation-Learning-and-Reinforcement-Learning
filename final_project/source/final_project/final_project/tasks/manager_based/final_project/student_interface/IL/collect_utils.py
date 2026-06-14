# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure, Isaac-free helpers for demo collection post-processing.

Kept import-light (numpy only) so they are unit-testable without booting Isaac Sim.
"""
from __future__ import annotations

import math

import numpy as np


def trajectory_smoothness(steps: list[dict]) -> dict:
    """Smoothness metrics for an executed episode buffer.

    ee_jerk   = max magnitude of EE position 2nd-difference (acceleration spike).
    cube_jump = max per-step cube displacement (knocked/dropped cube).
    """
    eef = np.stack([s["obs"]["eef_pos"] for s in steps], axis=0)
    cube = np.stack([s["obs"]["cube_pos"] for s in steps], axis=0)
    ee_jerk = 0.0
    if len(eef) >= 3:
        sd = eef[2:] - 2.0 * eef[1:-1] + eef[:-2]
        ee_jerk = float(np.linalg.norm(sd, axis=1).max())
    cube_jump = float(np.linalg.norm(np.diff(cube, axis=0), axis=1).max()) if len(cube) >= 2 else 0.0
    return {"ee_jerk": ee_jerk, "cube_jump": cube_jump}


def is_smooth(steps: list[dict], max_ee_jerk: float, max_cube_jump: float) -> tuple[bool, str, dict]:
    """Return (accept, reason, metrics). A threshold of 0 disables that check."""
    m = trajectory_smoothness(steps)
    if max_ee_jerk > 0 and m["ee_jerk"] > max_ee_jerk:
        return False, f"EE jerk {m['ee_jerk']:.4f} > {max_ee_jerk}", m
    if max_cube_jump > 0 and m["cube_jump"] > max_cube_jump:
        return False, f"cube jump {m['cube_jump']:.4f} > {max_cube_jump}", m
    return True, "", m


def yaw_error_rad(target_yaw: float, cube_yaw: float) -> float:
    """Signed yaw error mapped into (-45deg, 45deg] using the cube's 4-fold (90deg) symmetry."""
    half = math.pi / 4.0
    return ((target_yaw - cube_yaw + half) % (math.pi / 2.0)) - half


def hit_step_cap(buffer_len: int, max_steps: int) -> bool:
    """True if the per-episode buffer reached its cap (trajectory was truncated)."""
    return buffer_len >= max_steps


def failure_record(success: bool, first_obs: dict, last_obs: dict,
                   final_pos_err: float, final_yaw_err_deg: float, last_phase: int) -> dict:
    """One JSONL diagnostics record describing an episode outcome and its config."""
    cube0 = np.asarray(first_obs["cube_pos"]).tolist()
    goal0 = np.asarray(first_obs["goal_pos"]).tolist()
    return {
        "success": bool(success),
        "cube_pos": cube0,
        "goal_pos": goal0,
        "source_z": float(cube0[2]),
        "target_z": float(goal0[2]),
        "final_pos_err": float(final_pos_err),
        "final_yaw_err_deg": float(final_yaw_err_deg),
        "last_phase": int(last_phase),
    }
