# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared thresholds and geometry for the IL pick-place task.

Imported by both the success termination (env config) and the scripted expert
(scripts/collect_il_demos.py) so the two never disagree on what "success" means.
"""

# Success: cube must come to rest within this distance of the placement point.
# Spec L1 requires position error < 5 cm.
PLACE_POS_THRESHOLD = 0.05  # m

# Gripper must be open (fingers beyond this) to count the cube as released.
FINGER_OPEN_THRESHOLD = 0.035  # m (each finger; max travel 0.04)

# Hold the success condition this many consecutive control steps before flagging
# success — forces the demos to contain extra "released at goal" frames so BC
# doesn't underweight the release.
HOLD_STEPS = 30

# Scene geometry (from il_eval_env.ILSceneCfg): platforms are 0.05 m tall, the
# cube is 0.05 m. The placement point sits one platform-half + one cube-half above
# the target platform's root (its center).
PLATFORM_HALF = 0.025  # m
CUBE_HALF = 0.025  # m

# IK end-effector offset along the hand's local +Z (panda_hand -> grasp point).
# Matches the DifferentialInverseKinematicsActionCfg.body_offset in env_cfg_L1.
EE_OFFSET_Z = 0.107  # m
