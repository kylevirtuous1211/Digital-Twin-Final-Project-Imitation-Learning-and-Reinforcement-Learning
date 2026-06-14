# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Vectorized rollout evaluation for a robomimic BC policy.

Lab's stock ``play.py`` hardcodes ``num_envs=1`` and runs rollouts strictly
sequentially (reloading the checkpoint every trial). This script instead runs
``--num_envs`` rollouts *simultaneously* as parallel GPU environments in a
single process, which is the only meaningful way to parallelize on Isaac Lab
(CPU threads/multiprocessing just spawn extra Isaac Sim instances).

Key difference from play.py: robomimic's ``RolloutPolicy`` wrapper cannot batch
(it flattens N obs into one row), so we call the inner ``algo.get_action`` which
is natively batched. Obs normalization is not applied (configs use
``hdf5_normalize_obs: false``); the script asserts this to stay safe.

Usage:
    python scripts/eval_il_bc_parallel.py --task FinalProject-IL-L1-v0 \
        --num_envs 50 --checkpoint <best.pth> --headless
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Source tree of the final_project package that lives alongside THIS script.
# The shared IsaacLab venv's editable install may map `final_project` to a
# different checkout (e.g. dt_final_project) via a sys.meta_path finder that
# wins over sys.path. Drop it and prepend our own source dir so obs keys /
# env configs come from THIS repo. (Mirrors train_il_bc.py.)
LOCAL_PKG_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "final_project"))


def _force_local_final_project() -> None:
    sys.meta_path = [f for f in sys.meta_path if "final_project" not in type(f).__module__]
    if LOCAL_PKG_SRC not in sys.path:
        sys.path.insert(0, LOCAL_PKG_SRC)

parser = argparse.ArgumentParser(description="Vectorized robomimic BC evaluation.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="robomimic checkpoint (.pth).")
parser.add_argument("--num_envs", type=int, default=50, help="Parallel rollouts run at once on the GPU.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon per rollout.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric / use USD I/O.")
parser.add_argument("--video", action="store_true", default=False, help="Record an mp4 of the rollout.")
parser.add_argument("--video_dir", type=str, default="demo_record", help="Folder for the recorded mp4.")
parser.add_argument("--video_length", type=int, default=0, help="Frames to record (0 = full horizon).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Video capture needs the camera/render pipeline; must be set before AppLauncher boots.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

_force_local_final_project()
import final_project  # noqa: F401  registers FinalProject-* tasks

print(f">>> final_project imported from: {final_project.__file__}", flush=True)

from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    n = args_cli.num_envs

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n, use_fabric=not args_cli.disable_fabric)
    # robomimic wants dict obs, not concatenated
    env_cfg.observations.policy.concatenate_terms = False
    # no auto time-out termination; we run the full horizon and score manually
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    # pull the success checker out of terminations so it doesn't end/reset episodes
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = os.path.abspath(args_cli.video_dir)
        os.makedirs(video_dir, exist_ok=True)
        length = args_cli.video_length or args_cli.horizon
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,  # record from the first step
            video_length=length,
            name_prefix=f"rollout_{args_cli.task}",
            disable_logger=True,
        )
        print(f">>> recording mp4 to {video_dir} (length={length})", flush=True)

    base = env.unwrapped  # success checker / seed need the ManagerBasedRLEnv

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    base.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load once. Use the inner algo (batched); RolloutPolicy wrapper cannot batch.
    rollout_policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
    assert rollout_policy.obs_normalization_stats is None, "obs normalization not supported in this fast path"
    algo = rollout_policy.policy
    algo.set_eval()

    obs_dict, _ = env.reset()

    success = torch.zeros(n, dtype=torch.bool, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)

    for i in range(args_cli.horizon):
        obs = {k: v.to(device).float() for k, v in obs_dict["policy"].items()}
        with torch.no_grad():
            actions = algo.get_action(obs)  # [n, action_dim]

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        succ = success_term.func(base, **success_term.params).to(device).bool()  # [n]
        success |= succ & ~done
        done |= succ | terminated.to(device).bool() | truncated.to(device).bool()

        n_succ = int(success.sum())
        n_done = int(done.sum())
        print(f"[step {i + 1}/{args_cli.horizon}] success={n_succ}/{n}  done={n_done}/{n}", flush=True)
        # When recording, keep stepping so the clip captures the full rollout.
        if bool(done.all()) and not args_cli.video:
            break

    rate = float(success.float().mean())
    print(f"\n===== VECTORIZED EVAL: {args_cli.task} =====")
    print(f"checkpoint: {args_cli.checkpoint}")
    print(f"Successful rollouts: {int(success.sum())} / {n}")
    print(f"Success rate: {rate:.3f}")
    print(f"Per-env results: {success.int().tolist()}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
