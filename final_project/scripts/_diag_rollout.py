"""One-off closed-loop diagnostic: run the L1 BC-RNN policy and log where env-0 breaks."""
from __future__ import annotations
import argparse, glob, os, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="FinalProject-IL-L1-v0")
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import numpy as np, torch, gymnasium as gym
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
LOCAL=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","source","final_project"))
sys.meta_path=[f for f in sys.meta_path if "final_project" not in type(f).__module__]
sys.path.insert(0,LOCAL)
import final_project  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
cfg.observations.policy.concatenate_terms=False
cfg.terminations.time_out=None
succ=cfg.terminations.success; cfg.terminations.success=None
env=gym.make(args_cli.task, cfg=cfg).unwrapped
dev=TorchUtils.get_torch_device(try_to_use_cuda=True)
ck=sorted(glob.glob("logs/robomimic/FinalProject-IL-L1-v0/bc_rnn/*/models/model_epoch_*_best_validation_*.pth"))[-1]
rp,_=FileUtils.policy_from_checkpoint(ckpt_path=ck, device=dev); algo=rp.policy; algo.set_eval()
obs,_=env.reset()
print(">>> DIAG start", flush=True)
for t in range(args_cli.steps):
    o={k:v.to(dev).float() for k,v in obs["policy"].items()}
    with torch.no_grad(): a=algo.get_action(o)
    obs,_,term,trunc,_=env.step(a)
    if t%40==0 or t==args_cli.steps-1:
        p=obs["policy"]; eef=p["eef_pos"][0].cpu().numpy(); cube=p["cube_pos"][0].cpu().numpy(); goal=p["goal_pos"][0].cpu().numpy()
        s=bool(succ.func(env,**succ.params)[0])
        print(f"[t{t:3d}] eef={np.round(eef,3)} cube={np.round(cube,3)} "
              f"|eef-cube|={np.linalg.norm(eef-cube):.3f} cube_z={cube[2]:.3f} "
              f"|cube-goal|xy={np.linalg.norm(cube[:2]-goal[:2]):.3f} act_dpos={np.linalg.norm(a[0,:3].cpu().numpy()):.3f} grip={a[0,6].item():.2f} succ={s}", flush=True)
env.close(); app.close()
