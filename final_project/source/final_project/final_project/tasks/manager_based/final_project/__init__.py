# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="FinalProject-IL-L1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_interface.IL.env_cfg_L1:EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
        "robomimic_bc_rnn_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn.json",
    },
)

gym.register(
    id="FinalProject-IL-L2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_interface.IL.env_cfg_L2:EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
    },
)

gym.register(
    id="FinalProject-IL-L3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_interface.IL.env_cfg_L3:EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_L3.json",
        "robomimic_bc_rnn_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_L3.json",
    },
)

gym.register(
    id="FinalProject-RL-L1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_interface.RL.env_cfg_L1:EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="FinalProject-RL-L2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_interface.RL.env_cfg_L2:EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="FinalProject-RL-L3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_interface.RL.env_cfg_L3:EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
