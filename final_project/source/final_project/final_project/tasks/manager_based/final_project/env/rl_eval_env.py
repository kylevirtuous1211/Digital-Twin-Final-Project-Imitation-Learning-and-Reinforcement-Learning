# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL evaluation environment — Nova Carter navigation with obstacles."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # RigidObjectCfg used by RLSceneL3Cfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .base_eval_env import BaseEvalEnvCfg
from .level_cfgs import RL_L1_EventCfg, RL_L2_EventCfg, RL_L3_EventCfg
import math


NOVA_CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.03),
        rot=(0.7071, 0.0, 0.0, 0.7071),
    ),
    actuators={
        "wheel_drives": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_left", "joint_wheel_right"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1e5,
        ),
    },
)


@configclass
class RLSceneCfg(InteractiveSceneCfg):
    """Scene for RL evaluation: Nova Carter + goal marker + obstacle slots."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot: ArticulationCfg = NOVA_CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RLSceneL3Cfg(RLSceneCfg):
    """L3 scene — obstacle + 4 boundary walls confining the ±4 m arena."""

    # Two obstacle variants — one is randomly activated per episode, the other
    # is parked off-arena (y=-10 / y=-15) by the reset event.
    obstacle_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Objects/Box/box.usd",
            scale=(4.0, 4.0, 4.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -10.0, 0.25)),
    )
    obstacle_klt: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleKLT",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/KLT_Bin/small_KLT.usd",
            scale=(4.0, 4.0, 4.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -15.0, 0.43)),
    )


@configclass
class RLCommandsL2Cfg:
    """Goal command for L2 — randomised 2D position AND goal heading."""

    goal_pose: UniformPose2dCommandCfg = UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e6, 1e6),   # only resample on episode reset
        simple_heading=False,               # use explicit heading range
        debug_vis=True,
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(3.0, 4.0),
            heading=(math.pi/4.0, math.pi/4.0*3.0),
        ),
    )


@configclass
class RLCommandsL1Cfg:
    """Goal command for L1 — fixed goal at (3, 0)."""

    goal_pose: UniformPose2dCommandCfg = UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e6, 1e6),
        simple_heading=False,  # False = use the heading range directly; True would override with robot→goal angle
        debug_vis=True,
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(2.0, 3.0),
            heading=(math.pi/2.0, math.pi/2.0),  # fixed heading 0 = face +y
        ),
    )


@configclass
class RLRewardsCfg:
    """No reward terms — students define their own rewards for training."""

    pass


@configclass
class RLTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RLEvalEnvCfg(BaseEvalEnvCfg):
    """RL evaluation environment — student obs/actions injected by merger."""

    scene: RLSceneCfg = RLSceneCfg(num_envs=64, env_spacing=15.0)
    rewards: RLRewardsCfg = RLRewardsCfg()
    terminations: RLTerminationsCfg = RLTerminationsCfg()
    # events: defined per-level in subclasses below
    # observations/actions: student-defined in their EnvCfg subclass

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation


@configclass
class RLEvalEnvL1Cfg(RLEvalEnvCfg):
    """RL Level 1 — fixed goal; no obstacles."""

    commands: RLCommandsL1Cfg = RLCommandsL1Cfg()
    events: RL_L1_EventCfg = RL_L1_EventCfg()


@configclass
class RLEvalEnvL2Cfg(RLEvalEnvCfg):
    """RL Level 2 — randomised goal position + heading;"""

    commands: RLCommandsL2Cfg = RLCommandsL2Cfg()
    events: RL_L2_EventCfg = RL_L2_EventCfg()


@configclass
class RLEvalEnvL3Cfg(RLEvalEnvCfg):
    """RL Level 3 — L2 + obstacle placed on robot's straight-line path."""

    scene: RLSceneL3Cfg = RLSceneL3Cfg(num_envs=64, env_spacing=15.0)
    commands: RLCommandsL2Cfg = RLCommandsL2Cfg()
    events: RL_L3_EventCfg = RL_L3_EventCfg()
