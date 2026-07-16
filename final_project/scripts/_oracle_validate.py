"""Scratch: run the geometry-reactive oracle as the actor in-sim, measure success rate."""
from __future__ import annotations
import argparse, os, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="FinalProject-IL-L1-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--control_yaw", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import numpy as np, torch, gymnasium as gym
LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "final_project"))
sys.meta_path = [f for f in sys.meta_path if "final_project" not in type(f).__module__]
sys.path.insert(0, LOCAL)
import final_project  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from final_project.tasks.manager_based.final_project.student_interface.IL.dagger_oracle import reactive_oracle

cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
cfg.observations.policy.concatenate_terms = False
cfg.terminations.time_out = None
succ = cfg.terminations.success
cfg.terminations.success = None
env = gym.make(args_cli.task, cfg=cfg).unwrapped
dev = env.device
obs_dict, _ = env.reset()
cube_z0 = obs_dict["policy"]["cube_pos"][:, 2].cpu().numpy()
success = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=dev)
print(">>> oracle validate start", flush=True)
for t in range(args_cli.horizon):
    with torch.inference_mode():
        obs = {k: v.float() for k, v in obs_dict["policy"].items()}
        a = reactive_oracle(obs, torch.as_tensor(cube_z0, device=dev), args_cli.control_yaw)
        an = np.asarray(a.detach().cpu().numpy(), dtype=np.float32).copy()
        if args_cli.noise > 0:
            an[:, 0:3] += (np.random.randn(*an[:, 0:3].shape) * args_cli.noise).astype(np.float32)
        step = torch.from_numpy(an).to(dev)
        obs_dict, _, term, trunc, _ = env.step(step)
        s = succ.func(env, **succ.params).to(dev).bool()
        success |= s
    if t % 100 == 0 or t == args_cli.horizon - 1:
        print(f"[step {t+1}/{args_cli.horizon}] success={int(success.sum())}/{args_cli.num_envs}", flush=True)
print(f"\nORACLE SUCCESS RATE: {int(success.sum())}/{args_cli.num_envs} = {float(success.float().mean())*100:.1f}%", flush=True)
env.close(); app.close()
