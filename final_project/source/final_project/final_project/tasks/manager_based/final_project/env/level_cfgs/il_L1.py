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

    ROBUSTNESS-FIRST RANGES (wide reachable workspace). Grading uses the TA's
    ``EvalSceneCfg`` (we provide only obs+actions+checkpoints, PDF p6), whose target
    placement is unknown-but-reachable. We therefore train over a broad reachable box
    so wherever the TA puts the (reachable) target, the cube_to_goal vector the policy
    sees is in-distribution. Source (pick) on the -y side, target (place) on the +y
    side — non-overlapping; both stay within ~0.71 m of the base (comfortable reach).
      source abs x(0.42,0.60) y(-0.32,-0.05) = base(0.40,0.00) + offset below
      target abs x(0.42,0.62) y( 0.05, 0.35) = base(0.45,0.15) + offset below
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_on_source_platform = EventTerm(
        func=reset_cube_on_source_platform,
        mode="reset",
        params={
            "platform_x_range": (0.02, 0.20),
            "platform_y_range": (-0.32, -0.05),
            "platform_z_range": (0.0, 0.0),
        },
    )

    randomize_target_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.17), "y": (-0.10, 0.20)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_platform"),
        },
    )
