# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""DAgger data collection: roll out the LEARNER (BC policy) in-sim, label every
visited state with a geometry-REACTIVE oracle, aggregate with the base dataset.

Why reactive (not the scripted waypoint expert): the BC policy fails by drifting
OFF the expert's razor-thin manifold (overshoots the cube, stalls). DAgger must
label those off-manifold states with the correct *corrective* action. A reactive
oracle computes the right action from ANY (eef, cube, goal) geometry, so its
labels stay coherent with wherever the learner actually goes — including recovery
from overshoot. This is the missing supervision identified by the diagnosis.

    python scripts/dagger_collect.py --task FinalProject-IL-L1-v0 \
        --checkpoint <bc_ckpt.pth> --base_dataset datasets/il_L1.hdf5 \
        --out_dataset datasets/il_L1_dag1.hdf5 --num_demos 400 --num_envs 32 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

LOCAL_PKG_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "final_project"))


def _force_local_final_project() -> None:
    sys.meta_path = [f for f in sys.meta_path if "final_project" not in type(f).__module__]
    if LOCAL_PKG_SRC not in sys.path:
        sys.path.insert(0, LOCAL_PKG_SRC)


parser = argparse.ArgumentParser(description="DAgger collection with a reactive oracle.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Current BC policy (the learner).")
parser.add_argument("--base_dataset", type=str, required=True, help="Aggregate so far (copied into the output).")
parser.add_argument("--out_dataset", type=str, required=True, help="New aggregate = base demos + DAgger rollouts.")
parser.add_argument("--num_demos", type=int, default=400, help="DAgger rollout sequences to add.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--horizon", type=int, default=400, help="Steps per DAgger rollout sequence.")
parser.add_argument("--valid_ratio", type=float, default=0.1)
parser.add_argument("--control_yaw", action="store_true")
parser.add_argument("--noise", type=float, default=0.01, help="Gaussian std (m) added to the LEARNER's dpos to diversify the visited distribution.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_force_local_final_project()
import final_project  # noqa: F401,E402
print(f">>> final_project imported from: {final_project.__file__}", flush=True)
from final_project.tasks.manager_based.final_project.student_interface.IL.dagger_oracle import reactive_oracle  # noqa: E402


def _copy_base_demos(base_path: str, data_grp: h5py.Group) -> int:
    """Copy all demos from the base dataset into the output; return count."""
    n = 0
    with h5py.File(base_path, "r") as bf:
        names = sorted(bf["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for name in names:
            bf.copy(bf["data"][name], data_grp, name=f"demo_{n}")
            n += 1
    return n


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.concatenate_terms = False
    # KEEP time_out: envs auto-reset inside env.step (manual reset across inference_mode
    # corrupts Isaac state buffers). Drop success so a rollout runs full-length and we
    # capture the overshoot/stall recovery states DAgger needs.
    env_cfg.terminations.success = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    rp, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
    algo = rp.policy; algo.set_eval()

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_dataset)), exist_ok=True)
    with h5py.File(args_cli.out_dataset, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": args_cli.task, "type": 2})
        saved = _copy_base_demos(args_cli.base_dataset, data)
        base_clean = saved
        print(f">>> copied {saved} base demos from {args_cli.base_dataset}", flush=True)

        def _flush(idx, steps):
            ep = data.create_group(f"demo_{idx}")
            acts = np.stack([s["action"] for s in steps], 0)
            T = acts.shape[0]
            ep.create_dataset("actions", data=acts)
            ep.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
            dn = np.zeros(T, dtype=np.int64); dn[-1] = 1
            ep.create_dataset("dones", data=dn)
            og = ep.create_group("obs")
            for k in steps[0]["obs"].keys():
                og.create_dataset(k, data=np.stack([s["obs"][k] for s in steps], 0))
            ep.attrs["num_samples"] = T

        dag_target = saved + args_cli.num_demos
        # Continuous rollout with AUTO-RESET (no manual env.reset across waves -> avoids
        # corrupting Isaac state buffers). Flush each env's buffer when the env resets
        # (done) or the buffer reaches --horizon; track per-env cube rest-height for
        # the oracle's grasp detection, refreshing it only when that env resets.
        buffers = [[] for _ in range(env.num_envs)]
        obs_dict, _ = env.reset()
        cube_z0 = obs_dict["policy"]["cube_pos"][:, 2].cpu().numpy()  # numpy [N], normal
        algo.set_eval()
        while saved < dag_target:
            # robomimic get_action returns an INFERENCE tensor; Isaac's auto-reset
            # (reset_scene_to_default) then trips on an in-place write if any inference
            # tensor reached the env. So: compute under no_grad, round-trip the action
            # through numpy, and feed env.step a FRESH NORMAL tensor.
            with torch.no_grad():
                pol = obs_dict["policy"]
                obs = {k: v.to(device).float() for k, v in pol.items()}
                learner = algo.get_action(obs)
                oracle = reactive_oracle(obs, torch.as_tensor(cube_z0, device=device), args_cli.control_yaw)
                obs_np = {k: v.detach().cpu().numpy() for k, v in pol.items()}
                lab_np = oracle.detach().cpu().numpy()
                act_np = np.asarray(learner.detach().cpu().numpy(), dtype=np.float32).copy()
                if args_cli.noise > 0:
                    act_np[:, 0:3] += (np.random.randn(*act_np[:, 0:3].shape) * args_cli.noise).astype(np.float32)
                step = torch.from_numpy(act_np).to(device)              # fresh NORMAL tensor
                next_obs, _, terminated, truncated, _ = env.step(step)
                done_np = (terminated | truncated).detach().cpu().numpy()
                next_cube_z = next_obs["policy"]["cube_pos"][:, 2].detach().cpu().numpy()
            for i in range(env.num_envs):
                buffers[i].append({"obs": {k: obs_np[k][i] for k in obs_np}, "action": lab_np[i]})
                if (done_np[i] or len(buffers[i]) >= args_cli.horizon) and len(buffers[i]) > 1 and saved < dag_target:
                    _flush(saved, buffers[i]); saved += 1
                    buffers[i] = []
                    if saved % 32 == 0 or saved == dag_target:
                        print(f">>> dagger demos: {saved - base_clean}/{args_cli.num_demos} (total {saved})", flush=True)
                if done_np[i]:
                    cube_z0[i] = next_cube_z[i]  # env reset -> new cube rest height
            obs_dict = next_obs

        all_demos = [f"demo_{i}" for i in range(saved)]
        n_valid = max(1, int(saved * args_cli.valid_ratio))
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(all_demos[:-n_valid], dtype=object))
        mask.create_dataset("valid", data=np.array(all_demos[-n_valid:], dtype=object))
    print(f"\nDone. {saved} demos -> {args_cli.out_dataset} "
          f"(base={base_clean}, dagger={saved - base_clean}, train={saved - n_valid}, valid={n_valid})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
