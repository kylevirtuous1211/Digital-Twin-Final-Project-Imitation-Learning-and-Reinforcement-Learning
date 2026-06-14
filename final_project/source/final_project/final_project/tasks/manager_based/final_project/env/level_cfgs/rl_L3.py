# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL Level 3 — L2 goal randomization + randomly chosen USD obstacle in corridor."""

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ..rl_events import reset_obstacle_between_robot_and_goal


@configclass
class RL_L3_EventCfg:
    """L2 goal randomization (CommandManager) + one of two USD obstacles placed as chokepoint.

    Event firing order (IsaacLab uses class-body definition order):
      1. reset_all                         — restore scene to init_state
      2. reset_robot                       — sample robot position
      3. reset_obstacle_between_robot_goal — randomly activate obstacle_box or obstacle_klt
                                            in corridor; park the other off-arena
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-4.0, -3.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_obstacle_startup = EventTerm(
        func=reset_obstacle_between_robot_and_goal,
        mode="startup",
        params={"obstacle_names": ("obstacle_box", "obstacle_klt")},
    )

    reset_obstacle_between_robot_goal = EventTerm(
        func=reset_obstacle_between_robot_and_goal,
        mode="reset",
        params={"obstacle_names": ("obstacle_box", "obstacle_klt")},
    )
