# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IL evaluation environment — Franka arm pick-and-place."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from .base_eval_env import BaseEvalEnvCfg
from .level_cfgs import IL_L1_EventCfg, IL_L2_EventCfg, IL_L3_EventCfg


@configclass
class ILSceneCfg(InteractiveSceneCfg):
    """Scene for IL evaluation: Franka + cube + source/target platforms."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.05, 0.02)),
    )

    source_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SourcePlatform",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0)),
    )

    target_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPlatform",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
        ),
        # (0.6, 0.3) put targets at 0.81-0.94 m from the base — beyond the Franka's
        # ~0.85 m reach (mean 0.88 m), making ~77% of placements kinematically
        # impossible. Moved to (0.45, 0.15) so absolute targets land at ~0.60-0.74 m
        # (within reach, like the source platform at ~0.51-0.63 m).
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.15, 0.0)),
    )


@configclass
class ILRewardsCfg:
    """No reward terms needed — IL evaluation discards the reward signal."""

    pass


@configclass
class ILTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ILEvalEnvCfg(BaseEvalEnvCfg):
    """IL evaluation environment base — subclasses add level-specific events."""

    scene: ILSceneCfg = ILSceneCfg(num_envs=64, env_spacing=2.5)
    rewards: ILRewardsCfg = ILRewardsCfg()
    terminations: ILTerminationsCfg = ILTerminationsCfg()
    # events: defined per-level in subclasses below
    # observations/actions: student-defined in their EnvCfg subclass

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation


@configclass
class ILEvalEnvL1Cfg(ILEvalEnvCfg):
    """IL Level 1 — cube+source randomised x/y; fixed target position."""

    events: IL_L1_EventCfg = IL_L1_EventCfg()


@configclass
class ILEvalEnvL2Cfg(ILEvalEnvCfg):
    """IL Level 2 — L1 + target platform yaw randomization."""

    events: IL_L2_EventCfg = IL_L2_EventCfg()


@configclass
class ILEvalEnvL3Cfg(ILEvalEnvCfg):
    """IL Level 3 — L2 + source and target platform height randomization."""

    events: IL_L3_EventCfg = IL_L3_EventCfg()
