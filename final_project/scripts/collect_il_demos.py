# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted pick-place expert + Robomimic-format HDF5 collector for IL Level 1.

Drives ``FinalProject-IL-L1-v0`` with a deterministic state machine
(APPROACH -> DESCEND -> CLOSE -> LIFT -> TRANSLATE -> PLACE -> RELEASE -> DONE)
producing differential-IK delta-pose actions, and streams successful episodes to
disk in the layout expected by Isaac Lab's robomimic train.py:

    data/
      env_args (attr) {"env_name": "<task>", "type": 2}
      demo_N/
        actions   [T, 7]   (dpos x3, drot x3, gripper x1)
        rewards   [T]      (zeros — BC ignores them)
        dones     [T]      (last step = 1)
        obs/
          eef_pos [T,3]  eef_quat [T,4]  cube_pos [T,3]  cube_quat [T,4]  goal_pos [T,3]
        num_samples (attr) = T
    mask/
      train [...]   valid [...]

Run (headless, host venv with Isaac Lab):
    python scripts/collect_il_demos.py --task FinalProject-IL-L1-v0 \
        --num_envs 32 --num_demos 200 --dataset ./datasets/il_L1.hdf5 --headless
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

# ----- CLI must be parsed BEFORE Isaac Sim imports -----
parser = argparse.ArgumentParser(description="Collect Franka pick-place demos for IL L1.")
parser.add_argument("--num_envs", type=int, default=32, help="Parallel envs.")
parser.add_argument("--num_demos", type=int, default=200, help="Successful demos to collect.")
parser.add_argument("--dataset", type=str, default="./datasets/il_L1.hdf5", help="HDF5 output path.")
parser.add_argument("--task", type=str, default="FinalProject-IL-L1-v0", help="Registered task name.")
parser.add_argument("--valid_ratio", type=float, default=0.1, help="Fraction reserved for validation mask.")
parser.add_argument("--max_steps", type=int, default=600, help="Per-episode buffer cap.")
parser.add_argument("--control_yaw", action="store_true",
                    help="Align cube yaw to the target platform yaw (L2/L3). Requires a 'goal_quat' obs term.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- Imports that need Isaac Sim running -----
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import final_project  # noqa: F401,E402  registers FinalProject-* gym envs
from final_project.tasks.manager_based.final_project.student_interface.IL.mdp import constants  # noqa: E402


# ----- Smooth pick-place state machine (NVIDIA PickPlaceController style) -----
# Seven phases. Within each phase the *commanded* Cartesian target is interpolated
# from the phase's start anchor to its end anchor with a sinusoidal ease
# (mix_sin), so the absolute-pose IK tracks a smoothly moving reference -> smooth
# joint motion (no bang-bang). Orientation is held at the natural downward
# ready-pose quaternion captured at reset, so the arm never has to reorient.
#   P0 APPROACH  : move above the cube at safe height
#   P1 DESCEND   : lower onto the cube (gripper open)
#   P2 GRASP     : hold + close gripper
#   P3 LIFT      : raise straight up to safe height
#   P4 TRANSLATE : move over the target platform at safe height
#   P5 PLACE     : lower onto the platform top
#   P6 RELEASE   : open gripper and hold (success termination ends the episode)
_N_PHASES = 7
# Transit height is computed per-env as max(grasp_z, place_z) + TRANSIT_CLEAR so the
# cube clears both (possibly tall, L3) platforms while staying as low as possible —
# a low, extended arm config is what lets the Franka reach the far target platform.
APPROACH_CLEAR = 0.10    # hover this far above the cube before descending (m)
TRANSIT_CLEAR = 0.08     # carry the cube this far above the taller platform top (m)
GRASP_DZ = -0.005        # descend this far below cube center to engulf it (m)
PLACE_CLEAR = 0.005      # release the cube this far above its resting height (m)
MAX_STEP = 0.08          # safety clamp on the per-step position delta (m)
MAX_YAW = 0.10           # safety clamp on the per-step yaw delta (rad)

# Nominal phase durations in control steps — set the sinusoidal easing rate. A
# phase ends when its ease completes AND (for motion phases) the EE has actually
# arrived; a per-phase step cap prevents stalls on unreachable targets.
_PHASE_STEPS = [40, 55, 20, 35, 60, 55, 100000]

# Arrival tolerance per phase (m). Dwell phases (GRASP close, RELEASE hold) are
# time-based only -> tolerance unused (see _DWELL).
_PHASE_TOL = [0.04, 0.02, 0.0, 0.04, 0.04, 0.02, 0.0]

# Hard step cap per phase (safety against stalls on unreachable targets).
_PHASE_CAP = [160, 240, 80, 160, 280, 280, 10**9]

# GRASP (close) and RELEASE (hold) are dwell phases: advance on time alone.
_DWELL = {2, 6}

# Gripper command per phase: +1 open, -1 close. Close starts in GRASP and holds
# through LIFT/TRANSLATE/PLACE, opens in RELEASE.
_PHASE_GRIP = [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0]

# Cube center rests one platform-half + one cube-half above the target root.
PLACE_Z_OFFSET = constants.PLATFORM_HALF + constants.CUBE_HALF


def _mix_sin(t: torch.Tensor) -> torch.Tensor:
    """Smooth 0->1 ease (NVIDIA PickPlaceController._mix_sin)."""
    return 0.5 * (1.0 - torch.cos(t * torch.pi))


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw about world z from a (w, x, y, z) quaternion. Shape [N]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PickPlacePolicy:
    """Vectorized smooth pick-place expert producing 7-D relative-IK actions:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]. The position delta drives the EE
    toward a smoothly interpolated waypoint; transit height adapts to the platform
    heights. With control_yaw, the dyaw channel aligns the (grasped) cube to the
    target platform yaw after the grasp; otherwise rotation is left free for reach."""

    def __init__(self, num_envs: int, device: torch.device | str, control_yaw: bool = False):
        self.num_envs = num_envs
        self.device = device
        self.control_yaw = control_yaw
        self.e = torch.zeros(num_envs, dtype=torch.long, device=device)       # phase index
        self.t = torch.zeros(num_envs, device=device)                          # phase progress [0,1)
        self.start_pos = torch.zeros(num_envs, 3, device=device)               # phase start anchor
        self.pick_pos = torch.zeros(num_envs, 3, device=device)                # cube pos captured at grasp
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)   # steps in current phase
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.dur = torch.tensor(_PHASE_STEPS, dtype=torch.float, device=device)
        self.tol = torch.tensor(_PHASE_TOL, device=device)
        self.cap = torch.tensor(_PHASE_CAP, dtype=torch.float, device=device)
        self.grip = torch.tensor(_PHASE_GRIP, device=device)
        self.is_dwell = torch.zeros(_N_PHASES, dtype=torch.bool, device=device)
        for _p in _DWELL:
            self.is_dwell[_p] = True

    def reset_idx(self, env_ids) -> None:
        self.e[env_ids] = 0
        self.t[env_ids] = 0.0
        self.steps[env_ids] = 0
        self.initialized[env_ids] = False

    def reset(self) -> None:
        self.e[:] = 0
        self.t[:] = 0.0
        self.steps[:] = 0
        self.initialized[:] = False

    def _end_pos(self, cube: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Per-phase end anchor for every env, selected by phase index. (N, 3)."""
        N = cube.shape[0]
        grasp_z = cube[:, 2] + GRASP_DZ
        place_z = goal[:, 2] + PLACE_Z_OFFSET + PLACE_CLEAR
        approach_z = cube[:, 2] + APPROACH_CLEAR
        # Transit height adapts to platform heights (L3): clear the taller of the
        # pick/place heights, but stay as low as possible for far reach.
        transit_z = torch.maximum(self.pick_pos[:, 2] + GRASP_DZ, place_z) + TRANSIT_CLEAR
        cand = torch.stack(
            [
                torch.stack([cube[:, 0], cube[:, 1], approach_z], dim=-1),              # P0 above cube
                torch.stack([cube[:, 0], cube[:, 1], grasp_z], dim=-1),                # P1 on cube
                torch.stack([self.pick_pos[:, 0], self.pick_pos[:, 1],
                             self.pick_pos[:, 2] + GRASP_DZ], dim=-1),                  # P2 hold
                torch.stack([self.pick_pos[:, 0], self.pick_pos[:, 1], transit_z], dim=-1),  # P3 lift
                torch.stack([goal[:, 0], goal[:, 1], transit_z], dim=-1),               # P4 over goal
                torch.stack([goal[:, 0], goal[:, 1], place_z], dim=-1),                 # P5 place
                torch.stack([goal[:, 0], goal[:, 1], place_z], dim=-1),                 # P6 release/hold
            ],
            dim=0,
        )  # (7, N, 3)
        return cand[self.e, torch.arange(N, device=self.device)]

    def compute(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        eef_pos = obs["eef_pos"]    # (N, 3)
        cube = obs["cube_pos"]      # (N, 3)
        goal = obs["goal_pos"]      # (N, 3) target_platform center

        # Lazily initialize freshly reset envs: anchor the interpolation at the
        # current EE pose so the first waypoint starts from where the arm is.
        need = ~self.initialized
        if need.any():
            self.start_pos[need] = eef_pos[need]
            self.pick_pos[need] = cube[need]
            self.e[need] = 0
            self.t[need] = 0.0
            self.steps[need] = 0
            self.initialized[need] = True

        end_pos = self._end_pos(cube, goal)
        alpha = _mix_sin(self.t).unsqueeze(-1)                       # (N, 1)
        waypoint = (1.0 - alpha) * self.start_pos + alpha * end_pos  # (N, 3) smooth target

        # Relative-IK action: position delta toward the smooth waypoint, zero
        # rotation delta (orientation free), plus the gripper command.
        dpos = waypoint - eef_pos
        norms = torch.linalg.vector_norm(dpos, dim=-1, keepdim=True)
        scale = torch.where(norms > MAX_STEP, MAX_STEP / norms.clamp(min=1e-9), torch.ones_like(norms))
        dpos = dpos * scale

        actions = torch.zeros(self.num_envs, 7, device=self.device)
        actions[:, 0:3] = dpos
        actions[:, 6] = self.grip[self.e]

        # Yaw alignment (L2/L3): once the cube is grasped (phase >= LIFT), rotate the
        # wrist about base z to drive the cube yaw toward the target platform yaw.
        # The rotation delta is pre-multiplied (world frame), so dyaw>0 increases the
        # cube's world yaw. Cube has 4-fold symmetry -> align modulo 90 deg.
        if self.control_yaw:
            tgt_yaw = _yaw_from_quat(obs["goal_quat"])
            cube_yaw = _yaw_from_quat(obs["cube_quat"])
            half = torch.pi / 4.0
            yaw_err = torch.remainder(tgt_yaw - cube_yaw + half, torch.pi / 2.0) - half
            dyaw = torch.clamp(yaw_err, -MAX_YAW, MAX_YAW)
            active = self.e >= 3
            actions[:, 5] = torch.where(active, dyaw, torch.zeros_like(dyaw))

        # Advance the phase clock (capped at 1.0 so the waypoint dwells at the end
        # target while the EE converges). A motion phase ends when the ease is done
        # AND the EE has arrived; a dwell phase ends on time; a step cap prevents
        # stalls. On advance, re-anchor at the actual EE pose.
        self.steps += 1
        # In-place so self.t stays a normal tensor (reassignment inside inference_mode
        # would make it an inference tensor that reset_idx can't update afterwards).
        self.t.add_(1.0 / self.dur[self.e]).clamp_(max=1.0)
        reached = torch.linalg.vector_norm(eef_pos - end_pos, dim=-1) < self.tol[self.e]
        time_done = self.t >= 1.0
        dwell = self.is_dwell[self.e]
        cap_hit = self.steps.float() > self.cap[self.e]
        advance = (self.e < _N_PHASES - 1) & (cap_hit | (time_done & (dwell | reached)))
        if advance.any():
            to_grasp = advance & (self.e == 1)
            if to_grasp.any():
                self.pick_pos[to_grasp] = cube[to_grasp]
            self.start_pos[advance] = eef_pos[advance]
            self.e[advance] += 1
            self.t[advance] = 0.0
            self.steps[advance] = 0
        return actions


def _flush_episode(data_grp: h5py.Group, saved_idx: int, steps: list[dict]) -> None:
    ep = data_grp.create_group(f"demo_{saved_idx}")
    actions = np.stack([s["action"] for s in steps], axis=0)
    T = actions.shape[0]
    ep.create_dataset("actions", data=actions)
    ep.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
    dones = np.zeros(T, dtype=np.int64)
    dones[-1] = 1
    ep.create_dataset("dones", data=dones)
    obs_grp = ep.create_group("obs")
    for key in steps[0]["obs"].keys():
        obs_grp.create_dataset(key, data=np.stack([s["obs"][key] for s in steps], axis=0))
    ep.attrs["num_samples"] = T


def run(env, policy: PickPlacePolicy, dataset_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)

    buffers: list[list[dict]] = [[] for _ in range(env.num_envs)]
    saved = 0
    attempted = 0

    with h5py.File(dataset_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": args_cli.task, "type": 2})

        obs_dict, _ = env.reset()
        policy.reset()

        step_counter = 0
        phase_names = ["APP", "DSC", "GRP", "LFT", "TRN", "PLC", "REL"]
        while saved < args_cli.num_demos:
            with torch.inference_mode():
                actions = policy.compute(obs_dict["policy"])
                next_obs_dict, _, terminated, truncated, _ = env.step(actions)

            step_counter += 1
            if step_counter % 120 == 0:
                hist = torch.bincount(policy.e, minlength=_N_PHASES).tolist()
                tagged = " ".join(f"{n}={c}" for n, c in zip(phase_names, hist))
                print(f"[step {step_counter}] saved={saved} attempts={attempted}  {tagged}", flush=True)

            actions_np = actions.cpu().numpy()
            obs_np = {k: v.cpu().numpy() for k, v in obs_dict["policy"].items()}
            for i in range(env.num_envs):
                if len(buffers[i]) < args_cli.max_steps:
                    buffers[i].append({"obs": {k: obs_np[k][i] for k in obs_np}, "action": actions_np[i]})

            done_mask = terminated | truncated
            done_ids = done_mask.nonzero(as_tuple=False).flatten().tolist()
            for i in done_ids:
                attempted += 1
                success = bool(terminated[i].item())  # success term; truncation discarded
                if success and saved < args_cli.num_demos and len(buffers[i]) > 1:
                    _flush_episode(data, saved, buffers[i])
                    saved += 1
                    print(f"[saved {saved}/{args_cli.num_demos}] env={i} len={len(buffers[i])} attempts={attempted}")
                buffers[i] = []
                policy.reset_idx([i])

            obs_dict = next_obs_dict

        all_demos = [f"demo_{i}" for i in range(saved)]
        n_valid = max(1, int(saved * args_cli.valid_ratio))
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(all_demos[:-n_valid], dtype=object))
        mask.create_dataset("valid", data=np.array(all_demos[-n_valid:], dtype=object))

    print(f"\nDone. {saved} demos -> {dataset_path} (train={saved - n_valid}, valid={n_valid}, attempts={attempted})")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Keep obs as a named dict so the expert can index by key.
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    policy = PickPlacePolicy(num_envs=env.num_envs, device=env.device, control_yaw=args_cli.control_yaw)
    try:
        run(env, policy, args_cli.dataset)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
