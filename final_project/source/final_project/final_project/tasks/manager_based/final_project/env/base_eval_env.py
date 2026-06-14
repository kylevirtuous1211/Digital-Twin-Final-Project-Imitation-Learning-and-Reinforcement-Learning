# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base evaluation environment configuration shared by IL and RL tasks."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class _BaseSceneCfg(InteractiveSceneCfg):
    """Minimal scene anchor; concrete IL/RL subclasses override all slots."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class _BaseTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BaseEvalEnvCfg(ManagerBasedRLEnvCfg):
    """
    Base evaluation environment config.
    """

    scene: _BaseSceneCfg = MISSING          # injected by IL/RL subclass
    events: object = MISSING                # injected by level cfg
    terminations: _BaseTerminationsCfg = _BaseTerminationsCfg()
    rewards: object = MISSING               # injected by IL/RL subclass

    observations: object = MISSING
    actions: object = MISSING

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 0.0, 3.0)
