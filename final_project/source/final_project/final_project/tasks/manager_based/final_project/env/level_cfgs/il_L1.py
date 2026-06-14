# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IL Level 1 — cube position and target position randomization."""

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..il_events import reset_cube_on_source_platform


@configclass
class IL_L1_EventCfg:
    """Randomizes cube position and target platform position.

    Source platform height is fixed; cube z is explicitly coupled to platform top.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_on_source_platform = EventTerm(
        func=reset_cube_on_source_platform,
        mode="reset",
        params={
            "platform_x_range": (0.1, 0.2),
            "platform_y_range": (-0.2, -0.1),
            "platform_z_range": (0.0, 0.0),
        },
    )

    randomize_target_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.1, 0.2), "y": (0.1, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_platform"),
        },
    )
