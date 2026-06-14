# L3 Expert Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the scripted L3 pick-place expert converge on the success condition (instead of timing out on fixed caps) and add gripper state to L3 observations, so re-collected L3 data covers the full L3 distribution (esp. large target yaw and low+far cubes) and trains a better BC policy.

**Architecture:** All controller changes are in `PickPlacePolicy` (in `collect_il_demos.py`): phases advance on *actual arrival* (not a fixed step-cap), PLACE additionally waits for yaw alignment, and `MAX_YAW` is raised. Pure helpers (smoothness, yaw-wrap, truncation, failure-record) move to a new importable `collect_utils.py` so they are unit-testable without booting Isaac Sim. A gripper-width obs term is added to L3. A reusable `coverage_diag.py` quantifies coverage for verification.

**Tech Stack:** Python, PyTorch, NumPy, h5py, Isaac Lab (robomimic), pytest. Venv python: `/home/kyle/Desktop/IsaacLab/.venv/bin/python`.

---

## File Structure

- Create: `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/collect_utils.py` — pure, Isaac-free helpers (smoothness, yaw-wrap, truncation predicate, failure-record). One responsibility: episode post-processing math.
- Create: `final_project/tests/test_collect_utils.py` — unit tests for the pure helpers (run without Isaac).
- Modify: `final_project/scripts/collect_il_demos.py` — import helpers; PickPlacePolicy convergence + yaw gating; raised caps; `MAX_YAW`/`YAW_DONE_TOL`; failure-logging; max_steps discard.
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/env/il_events.py` — fix cube half-height reset bug.
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/mdp/observations.py` — add `gripper_pos` obs func.
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/env_cfg_L3.py` — add `gripper` ObsTerm.
- Create: `final_project/source/final_project/final_project/tasks/manager_based/final_project/agents/robomimic/bc_rnn_L3.json` — BC-RNN config for L3 (poses + goal_quat + gripper).
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/__init__.py` — register `robomimic_bc_rnn_cfg_entry_point` for L3.
- Create: `final_project/scripts/coverage_diag.py` — reusable coverage-thirds diagnostic over an HDF5 dataset.

All paths below are relative to `/home/kyle/Desktop/isaac-sim-quickstart/Final_project`. Run all `git` and `pytest` from that directory. `PY=/home/kyle/Desktop/IsaacLab/.venv/bin/python`.

---

## Task 1: Pure helpers module + unit tests (TDD)

**Files:**
- Create: `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/collect_utils.py`
- Test: `final_project/tests/test_collect_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `final_project/tests/test_collect_utils.py`:

```python
import numpy as np
import importlib.util, os, pathlib

# Import collect_utils directly by path (no Isaac import side effects).
_PKG = pathlib.Path(__file__).resolve().parents[1] / "source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/collect_utils.py"
_spec = importlib.util.spec_from_file_location("collect_utils", _PKG)
cu = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cu)


def _steps(eef, cube):
    return [{"obs": {"eef_pos": np.array(e, dtype=np.float32),
                     "cube_pos": np.array(c, dtype=np.float32)}} for e, c in zip(eef, cube)]


def test_smoothness_smooth_path():
    eef = [[0, 0, t * 0.01] for t in range(10)]   # constant velocity -> ~0 jerk
    cube = [[0.5, 0.5, 0.0]] * 10
    m = cu.trajectory_smoothness(_steps(eef, cube))
    assert m["ee_jerk"] < 1e-5 and m["cube_jump"] < 1e-5


def test_smoothness_detects_jerk_and_jump():
    eef = [[0, 0, 0], [0, 0, 0], [0, 0, 0.1], [0, 0, 0.1]]  # a spike -> nonzero jerk
    cube = [[0, 0, 0], [0, 0, 0], [0.2, 0, 0], [0.2, 0, 0]]  # 0.2 jump
    m = cu.trajectory_smoothness(_steps(eef, cube))
    assert m["ee_jerk"] > 0.05 and m["cube_jump"] > 0.1


def test_is_smooth_thresholds():
    smooth = _steps([[0, 0, t * 0.01] for t in range(5)], [[0, 0, 0]] * 5)
    ok, reason, _ = cu.is_smooth(smooth, max_ee_jerk=0.03, max_cube_jump=0.05)
    assert ok and reason == ""
    rough = _steps([[0, 0, 0], [0, 0, 0], [0, 0, 0.1]], [[0, 0, 0]] * 3)
    ok2, reason2, _ = cu.is_smooth(rough, max_ee_jerk=0.03, max_cube_jump=0.05)
    assert not ok2 and "jerk" in reason2


def test_is_smooth_zero_disables():
    rough = _steps([[0, 0, 0], [0, 0, 0], [0, 0, 0.1]], [[0, 0, 0]] * 3)
    ok, _, _ = cu.is_smooth(rough, max_ee_jerk=0.0, max_cube_jump=0.0)
    assert ok


def test_yaw_error_wraps_modulo_90deg():
    # cube at 0, target at 80 deg -> nearest aligned face is -10 deg (mod 90)
    err = cu.yaw_error_rad(np.radians(80.0), np.radians(0.0))
    assert abs(err - np.radians(-10.0)) < 1e-5
    # target 44 deg -> err 44 deg (within +-45 window)
    err2 = cu.yaw_error_rad(np.radians(44.0), np.radians(0.0))
    assert abs(err2 - np.radians(44.0)) < 1e-5


def test_hit_step_cap():
    assert cu.hit_step_cap(buffer_len=900, max_steps=900) is True
    assert cu.hit_step_cap(buffer_len=500, max_steps=900) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PY=/home/kyle/Desktop/IsaacLab/.venv/bin/python; $PY -m pytest final_project/tests/test_collect_utils.py -q`
Expected: FAIL — `collect_utils.py` does not exist / functions undefined.

- [ ] **Step 3: Write the helpers**

Create `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/collect_utils.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure, Isaac-free helpers for demo collection post-processing.

Kept import-light (numpy only) so they are unit-testable without booting Isaac Sim.
"""
from __future__ import annotations

import math

import numpy as np


def trajectory_smoothness(steps: list[dict]) -> dict:
    """Smoothness metrics for an executed episode buffer.

    ee_jerk   = max magnitude of EE position 2nd-difference (acceleration spike).
    cube_jump = max per-step cube displacement (knocked/dropped cube).
    """
    eef = np.stack([s["obs"]["eef_pos"] for s in steps], axis=0)
    cube = np.stack([s["obs"]["cube_pos"] for s in steps], axis=0)
    ee_jerk = 0.0
    if len(eef) >= 3:
        sd = eef[2:] - 2.0 * eef[1:-1] + eef[:-2]
        ee_jerk = float(np.linalg.norm(sd, axis=1).max())
    cube_jump = float(np.linalg.norm(np.diff(cube, axis=0), axis=1).max()) if len(cube) >= 2 else 0.0
    return {"ee_jerk": ee_jerk, "cube_jump": cube_jump}


def is_smooth(steps: list[dict], max_ee_jerk: float, max_cube_jump: float) -> tuple[bool, str, dict]:
    """Return (accept, reason, metrics). A threshold of 0 disables that check."""
    m = trajectory_smoothness(steps)
    if max_ee_jerk > 0 and m["ee_jerk"] > max_ee_jerk:
        return False, f"EE jerk {m['ee_jerk']:.4f} > {max_ee_jerk}", m
    if max_cube_jump > 0 and m["cube_jump"] > max_cube_jump:
        return False, f"cube jump {m['cube_jump']:.4f} > {max_cube_jump}", m
    return True, "", m


def yaw_error_rad(target_yaw: float, cube_yaw: float) -> float:
    """Signed yaw error mapped into (-45deg, 45deg] using the cube's 4-fold (90deg) symmetry."""
    half = math.pi / 4.0
    return ((target_yaw - cube_yaw + half) % (math.pi / 2.0)) - half


def hit_step_cap(buffer_len: int, max_steps: int) -> bool:
    """True if the per-episode buffer reached its cap (trajectory was truncated)."""
    return buffer_len >= max_steps


def failure_record(success: bool, first_obs: dict, last_obs: dict,
                   final_pos_err: float, final_yaw_err_deg: float, last_phase: int) -> dict:
    """One JSONL diagnostics record describing an episode outcome and its config."""
    cube0 = np.asarray(first_obs["cube_pos"]).tolist()
    goal0 = np.asarray(first_obs["goal_pos"]).tolist()
    return {
        "success": bool(success),
        "cube_pos": cube0,
        "goal_pos": goal0,
        "source_z": float(cube0[2]),
        "target_z": float(goal0[2]),
        "final_pos_err": float(final_pos_err),
        "final_yaw_err_deg": float(final_yaw_err_deg),
        "last_phase": int(last_phase),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest final_project/tests/test_collect_utils.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add final_project/tests/test_collect_utils.py final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/collect_utils.py
git commit -m "feat(il): add Isaac-free collect_utils helpers with unit tests"
```

---

## Task 2: Wire collect_il_demos.py to use collect_utils (no behavior change)

**Files:**
- Modify: `final_project/scripts/collect_il_demos.py`

- [ ] **Step 1: Replace the inlined helpers with imports**

In `collect_il_demos.py`, after the existing `from final_project.tasks...mdp import constants` line, add:

```python
from final_project.tasks.manager_based.final_project.student_interface.IL import collect_utils as cu  # noqa: E402
```

Delete the now-duplicated `_trajectory_smoothness` and `_is_smooth` function definitions (the two functions added earlier in the file).

In `run()`, change the smoothness call from:

```python
                    accept, reason, m = _is_smooth(buffers[i])
```
to:
```python
                    accept, reason, m = cu.is_smooth(buffers[i], args_cli.max_ee_jerk, args_cli.max_cube_jump)
```

- [ ] **Step 2: Smoke-check import resolution (no Isaac boot needed for syntax)**

Run: `$PY -c "import ast; ast.parse(open('final_project/scripts/collect_il_demos.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 3: Commit**

```bash
git add final_project/scripts/collect_il_demos.py
git commit -m "refactor(il): use collect_utils helpers in collector"
```

---

## Task 3: Convergence-gated phase transitions + raised caps (Component 1)

**Files:**
- Modify: `final_project/scripts/collect_il_demos.py`

- [ ] **Step 1: Raise the step caps and yaw constants**

Replace the constants block:

```python
MAX_YAW = 0.10           # safety clamp on the per-step yaw delta (rad)
```
with:
```python
MAX_YAW = 0.20           # per-step yaw delta (rad); higher so large target yaws converge
YAW_DONE_TOL = 0.175     # place/release only completes once |yaw_err| < ~10 deg (margin under 15 deg success)
```

Replace:
```python
_PHASE_CAP = [160, 240, 80, 160, 280, 280, 10**9]
```
with:
```python
# Generous safety backstops only. Phases now advance on ACTUAL arrival (and, for
# PLACE, on yaw convergence); caps just prevent infinite stalls on unreachable goals.
_PHASE_CAP = [320, 480, 80, 320, 560, 700, 10**9]
```

- [ ] **Step 2: Hoist yaw_err and add the convergence/yaw gate in `PickPlacePolicy.compute`**

In `compute()`, replace the yaw-action block:

```python
        if self.control_yaw:
            tgt_yaw = _yaw_from_quat(obs["goal_quat"])
            cube_yaw = _yaw_from_quat(obs["cube_quat"])
            half = torch.pi / 4.0
            yaw_err = torch.remainder(tgt_yaw - cube_yaw + half, torch.pi / 2.0) - half
            dyaw = torch.clamp(yaw_err, -MAX_YAW, MAX_YAW)
            active = self.e >= 3
            actions[:, 5] = torch.where(active, dyaw, torch.zeros_like(dyaw))
```
with:
```python
        if self.control_yaw:
            tgt_yaw = _yaw_from_quat(obs["goal_quat"])
            cube_yaw = _yaw_from_quat(obs["cube_quat"])
            half = torch.pi / 4.0
            yaw_err = torch.remainder(tgt_yaw - cube_yaw + half, torch.pi / 2.0) - half
            dyaw = torch.clamp(yaw_err, -MAX_YAW, MAX_YAW)
            active = self.e >= 3
            actions[:, 5] = torch.where(active, dyaw, torch.zeros_like(dyaw))
            yaw_ok = yaw_err.abs() < YAW_DONE_TOL
        else:
            yaw_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
```

Then replace the advance block:

```python
        reached = torch.linalg.vector_norm(eef_pos - end_pos, dim=-1) < self.tol[self.e]
        time_done = self.t >= 1.0
        dwell = self.is_dwell[self.e]
        cap_hit = self.steps.float() > self.cap[self.e]
        advance = (self.e < _N_PHASES - 1) & (cap_hit | (time_done & (dwell | reached)))
```
with:
```python
        reached = torch.linalg.vector_norm(eef_pos - end_pos, dim=-1) < self.tol[self.e]
        time_done = self.t >= 1.0
        dwell = self.is_dwell[self.e]
        cap_hit = self.steps.float() > self.cap[self.e]
        # Convergence-gated: motion phases advance only once actually arrived. The
        # PLACE phase (e==5) additionally waits for yaw alignment (settle) so the
        # gripper does not open until the cube yaw is within tolerance. cap_hit is a
        # generous backstop against true stalls / unreachable goals.
        arrived = time_done & (dwell | reached)
        place_gate = torch.where(self.e == 5, yaw_ok, torch.ones_like(yaw_ok))
        advance = (self.e < _N_PHASES - 1) & (cap_hit | (arrived & place_gate))
```

- [ ] **Step 3: Syntax check**

Run: `$PY -c "import ast; ast.parse(open('final_project/scripts/collect_il_demos.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Commit**

```bash
git add final_project/scripts/collect_il_demos.py
git commit -m "feat(il): convergence-gated phase transitions + yaw settle in expert"
```

---

## Task 4: Per-episode failure/diagnostics logging (Component 4)

**Files:**
- Modify: `final_project/scripts/collect_il_demos.py`

- [ ] **Step 1: Add a CLI arg for the diagnostics file**

After the `--max_cube_jump` argument, add:

```python
parser.add_argument("--diag", type=str, default="",
                    help="Optional JSONL path to log per-episode outcome+config (diagnostics). Empty disables.")
```

- [ ] **Step 2: Open the diag file and write one record per finished episode**

In `run()`, just after `os.makedirs(...)` at the top, add:

```python
    diag_f = open(args_cli.diag, "w") if args_cli.diag else None
```

Inside the `for i in done_ids:` loop, immediately after `success = bool(terminated[i].item())`, add:

```python
                if diag_f is not None and len(buffers[i]) >= 1:
                    first = buffers[i][0]["obs"]
                    last = buffers[i][-1]["obs"]
                    cube_l = last["cube_pos"]; goal_l = last["goal_pos"]
                    pos_err = float(np.linalg.norm(cube_l[:2] - goal_l[:2]))
                    yaw_err_deg = 0.0
                    if "goal_quat" in last and "cube_quat" in last:
                        gq, cq = last["goal_quat"], last["cube_quat"]
                        ty = float(np.arctan2(2*(gq[0]*gq[3]+gq[1]*gq[2]), 1-2*(gq[2]**2+gq[3]**2)))
                        cyaw = float(np.arctan2(2*(cq[0]*cq[3]+cq[1]*cq[2]), 1-2*(cq[2]**2+cq[3]**2)))
                        yaw_err_deg = abs(np.degrees(cu.yaw_error_rad(ty, cyaw)))
                    rec = cu.failure_record(success, first, last, pos_err, yaw_err_deg, int(policy.e[i].item()))
                    diag_f.write(json.dumps(rec) + "\n"); diag_f.flush()
```

At the end of `run()`, before the final summary `print`, add:

```python
        if diag_f is not None:
            diag_f.close()
```

- [ ] **Step 3: Syntax check**

Run: `$PY -c "import ast; ast.parse(open('final_project/scripts/collect_il_demos.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Commit**

```bash
git add final_project/scripts/collect_il_demos.py
git commit -m "feat(il): per-episode JSONL diagnostics (config + outcome)"
```

---

## Task 5: max_steps discard + raise default (Component 6a)

**Files:**
- Modify: `final_project/scripts/collect_il_demos.py`

- [ ] **Step 1: Raise the buffer cap default**

Change:
```python
parser.add_argument("--max_steps", type=int, default=600, help="Per-episode buffer cap.")
```
to:
```python
parser.add_argument("--max_steps", type=int, default=900,
                    help="Per-episode buffer cap. Episodes that hit it are discarded (truncated, mislabel risk).")
```

- [ ] **Step 2: Discard truncated-at-cap successes**

Add a counter near the other counters in `run()`:
```python
    rejected_capped = 0     # success but buffer hit max_steps -> truncated, discard
```

Change the save branch from:
```python
                elif saved < args_cli.num_demos and len(buffers[i]) > 1:
                    accept, reason, m = cu.is_smooth(buffers[i], args_cli.max_ee_jerk, args_cli.max_cube_jump)
```
to:
```python
                elif cu.hit_step_cap(len(buffers[i]), args_cli.max_steps):
                    rejected_capped += 1
                    print(f"[reject-capped] env={i} hit max_steps={args_cli.max_steps} capped_rejects={rejected_capped}", flush=True)
                elif saved < args_cli.num_demos and len(buffers[i]) > 1:
                    accept, reason, m = cu.is_smooth(buffers[i], args_cli.max_ee_jerk, args_cli.max_cube_jump)
```

Update the final summary print to include the new counter:
```python
    print(f"Quality filter: attempts={attempted}  saved={saved}  "
          f"rejected_fail(not success)={rejected_fail}  rejected_rough(smoothness)={rejected_rough}  "
          f"rejected_capped(truncated)={rejected_capped}")
```

- [ ] **Step 3: Syntax check**

Run: `$PY -c "import ast; ast.parse(open('final_project/scripts/collect_il_demos.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Commit**

```bash
git add final_project/scripts/collect_il_demos.py
git commit -m "fix(il): discard buffer-capped (truncated) episodes; raise max_steps to 900"
```

---

## Task 6: Fix cube reset half-height bug (Component 6b)

**Files:**
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/env/il_events.py:17`

- [ ] **Step 1: Correct the constant**

Change:
```python
_CUBE_HALF_HEIGHT = 0.05
```
to:
```python
_CUBE_HALF_HEIGHT = 0.025  # cube is 0.05 m; half-height = 0.025 (was 0.05 -> spawned 2.5 cm too high)
```

- [ ] **Step 2: Syntax check**

Run: `$PY -c "import ast; ast.parse(open('final_project/source/final_project/final_project/tasks/manager_based/final_project/env/il_events.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 3: Commit**

```bash
git add final_project/source/final_project/final_project/tasks/manager_based/final_project/env/il_events.py
git commit -m "fix(il): correct cube half-height in reset (0.05 -> 0.025)"
```

---

## Task 7: Add gripper-width observation to L3 (Component 5)

**Files:**
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/mdp/observations.py`
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/env_cfg_L3.py`

- [ ] **Step 1: Add a `gripper_pos` observation function**

Append to `mdp/observations.py`:

```python
def gripper_pos(env, robot_cfg=SceneEntityCfg("robot")):
    """Both Franka finger joint positions (gripper width state), shape [N, 2].

    Lets a BC policy observe open/closed directly instead of inferring it from poses.
    """
    robot = env.scene[robot_cfg.name]
    ids, _ = robot.find_joints(["panda_finger_.*"])
    return robot.data.joint_pos[:, ids]
```

(If `SceneEntityCfg` is not already imported in this file, add `from isaaclab.managers import SceneEntityCfg` near the top imports.)

- [ ] **Step 2: Subscribe to it in the L3 observation group**

In `env_cfg_L3.py`, in the policy observation group (next to `goal_quat`), add:

```python
        gripper = ObsTerm(func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg("robot")})
```

- [ ] **Step 3: Verify it loads + emits a [N,2] obs (boots Isaac; ~2-3 min)**

Run:
```bash
cd final_project && OMNI_KIT_ACCEPT_EULA=YES $PY - <<'EOF'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import sys, os
sys.path.insert(0, os.path.abspath("source/final_project"))
import final_project, gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg("FinalProject-IL-L3-v0", device="cuda:0", num_envs=2)
cfg.observations.policy.concatenate_terms = False
env = gym.make("FinalProject-IL-L3-v0", cfg=cfg).unwrapped
obs,_ = env.reset()
print("gripper obs shape:", tuple(obs["policy"]["gripper"].shape))
env.close(); app.close()
EOF
```
Expected: `gripper obs shape: (2, 2)`

- [ ] **Step 4: Commit**

```bash
git add final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/mdp/observations.py final_project/source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/env_cfg_L3.py
git commit -m "feat(il): add gripper-width observation to L3"
```

---

## Task 8: BC-RNN config for L3 + register entry point (Component 5 / training)

**Files:**
- Create: `final_project/source/final_project/final_project/tasks/manager_based/final_project/agents/robomimic/bc_rnn_L3.json`
- Modify: `final_project/source/final_project/final_project/tasks/manager_based/final_project/__init__.py:30-38`

- [ ] **Step 1: Create `bc_rnn_L3.json`**

Copy `bc_rnn.json` (the L1 BC-RNN config) to `bc_rnn_L3.json`, then set the `observation.modalities.obs.low_dim` list to include the L3 + gripper keys:

```json
                "low_dim": [
                    "eef_pos",
                    "eef_quat",
                    "cube_pos",
                    "cube_quat",
                    "goal_pos",
                    "goal_quat",
                    "gripper"
                ],
```

Leave all other fields identical to `bc_rnn.json` (rnn.enabled true, gmm.enabled false, seq_length 10, lr 0.001, etc.). Set `"experiment": { "name": "bc_rnn_L3", ... }`.

- [ ] **Step 2: Register the L3 BC-RNN entry point**

In `__init__.py`, in the `FinalProject-IL-L3-v0` registration kwargs, add after the existing `robomimic_bc_cfg_entry_point` line:

```python
        "robomimic_bc_rnn_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_L3.json",
```

- [ ] **Step 3: Validate JSON + registration**

Run: `$PY -c "import json; json.load(open('final_project/source/final_project/final_project/tasks/manager_based/final_project/agents/robomimic/bc_rnn_L3.json')); print('json OK')"`
Expected: `json OK`

- [ ] **Step 4: Commit**

```bash
git add final_project/source/final_project/final_project/tasks/manager_based/final_project/agents/robomimic/bc_rnn_L3.json final_project/source/final_project/final_project/tasks/manager_based/final_project/__init__.py
git commit -m "feat(il): bc_rnn_L3 config (poses+goal_quat+gripper) and entry point"
```

---

## Task 9: Reusable coverage diagnostic script

**Files:**
- Create: `final_project/scripts/coverage_diag.py`

- [ ] **Step 1: Create the script**

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Report how a success-only demo dataset covers the L3 randomization ranges.

Splits each config axis into thirds; uniform coverage would be ~33/33/33. A low
'high' third means the expert fails on that region. No Isaac Sim needed.

    python scripts/coverage_diag.py --dataset datasets/il_L3.hdf5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def _yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _thirds(name, a, lo, hi, unit=""):
    t1 = lo + (hi - lo) / 3; t2 = lo + 2 * (hi - lo) / 3
    low = (a < t1).mean() * 100; mid = ((a >= t1) & (a < t2)).mean() * 100; high = (a >= t2).mean() * 100
    print(f"{name:22s} range[{lo:.2f},{hi:.2f}]{unit}  thirds%: low={low:4.1f} mid={mid:4.1f} high={high:4.1f}  mean={a.mean():.3f}")
    return {"low": low, "mid": mid, "high": high}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    cz, gz, gyaw = [], [], []
    for k in demos:
        o = f["data"][k]["obs"]
        cube0 = o["cube_pos"][0]; goal0 = o["goal_pos"][0]
        cz.append(cube0[2]); gz.append(goal0[2])
        if "goal_quat" in o:
            gyaw.append(abs(np.degrees(_yaw(o["goal_quat"][0]))))
    cz, gz = np.array(cz), np.array(gz)
    print(f"dataset={args.dataset}  demos={len(demos)}")
    _thirds("source/cube height z", cz, 0.0, 0.35)
    _thirds("target platform z", gz, 0.0, 0.30)
    if gyaw:
        _thirds("target |yaw| deg", np.array(gyaw), 0.0, 45.0, "deg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the baseline (old) data to confirm it reproduces the known numbers**

Run: `cd final_project && $PY scripts/coverage_diag.py --dataset datasets/il_L3.prev.hdf5`
Expected: prints thirds; target |yaw| high third ≈ 12.6 (matches the diagnosis).

- [ ] **Step 3: Commit**

```bash
git add final_project/scripts/coverage_diag.py
git commit -m "feat(il): coverage_diag.py to measure L3 dataset coverage"
```

---

## Task 10: Integration verification — small improved-expert collection

**Files:** none (runs the modified collector). Requires Isaac Sim.

- [ ] **Step 1: Collect a small improved-expert L3 sample headless with diagnostics**

Run:
```bash
cd final_project && OMNI_KIT_ACCEPT_EULA=YES $PY scripts/collect_il_demos.py \
  --task FinalProject-IL-L3-v0 --num_envs 16 --num_demos 100 --control_yaw \
  --dataset datasets/il_L3_smoketest.hdf5 --diag datasets/il_L3_smoketest_diag.jsonl --headless
```
Expected: completes; final line prints `attempts`, `saved`, and the `rejected_*` counts.

- [ ] **Step 2: Check yield improved vs the old ~28%**

Run: read the final `Quality filter:` line from stdout.
Expected (pass): `saved/attempts` materially above ~0.32 (improved coverage of hard configs). Record the number.

- [ ] **Step 3: Check coverage improved on the smoke sample**

Run: `cd final_project && $PY scripts/coverage_diag.py --dataset datasets/il_L3_smoketest.hdf5`
Expected (pass): target |yaw| high-third climbs from 12.6 toward ~33 (target ≥ 25); source-z low-third climbs from 9.9 (target ≥ 20). If not met, inspect `datasets/il_L3_smoketest_diag.jsonl` to see whether remaining failures are far+low (workspace-limited) vs yaw — and adjust `MAX_YAW`/`YAW_DONE_TOL`/caps before the full run.

- [ ] **Step 4: Clean up smoke artifacts**

```bash
cd final_project && rm -f datasets/il_L3_smoketest.hdf5 datasets/il_L3_smoketest_diag.jsonl
```

- [ ] **Step 5: Commit (verification is read-only; nothing to commit — skip if clean)**

```bash
git status --short
```

---

## Task 11: Full re-collection + retrain + eval (the production run)

**Files:** none (long-running). Requires Isaac Sim.

- [ ] **Step 1: Re-collect the full L3 dataset with the improved expert**

```bash
cd final_project && OMNI_KIT_ACCEPT_EULA=YES $PY scripts/collect_il_demos.py \
  --task FinalProject-IL-L3-v0 --num_envs 16 --num_demos 800 --control_yaw \
  --dataset datasets/il_L3.hdf5 --diag datasets/il_L3_collect_diag.jsonl --headless
```
Expected: 800 saved demos to `datasets/il_L3.hdf5`.

- [ ] **Step 2: Confirm coverage on the full dataset**

Run: `cd final_project && $PY scripts/coverage_diag.py --dataset datasets/il_L3.hdf5`
Expected (pass): target |yaw| high-third ≥ 25; source-z low-third ≥ 20.

- [ ] **Step 3: Train L3 BC-RNN on the new data**

```bash
cd final_project && rm -rf logs/robomimic/FinalProject-IL-L3-v0/bc_rnn_L3
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/train_il_bc.py \
  --task FinalProject-IL-L3-v0 --algo bc_rnn \
  --dataset ./datasets/il_L3.hdf5 --epochs 2000
```
Expected: validation loss is a small MSE (~0.01s), NOT exploding (GMM is disabled in bc_rnn_L3).

- [ ] **Step 4: Evaluate the L3 BC-RNN policy (vectorized, horizon 800)**

```bash
cd final_project && OMNI_KIT_ACCEPT_EULA=YES $PY scripts/eval_il_bc_parallel.py \
  --task FinalProject-IL-L3-v0 --num_envs 50 --horizon 800 \
  --checkpoint logs/robomimic/FinalProject-IL-L3-v0/bc_rnn_L3/*/models/model_epoch_2000.pth --headless
```
Expected: success rate materially above the single-frame-BC baseline. Record it.

- [ ] **Step 5: Commit the new dataset diagnostics + any notes**

```bash
git add -A
git commit -m "chore(il): L3 re-collected with robust expert; coverage + eval recorded"
```

---

## Self-Review

**Spec coverage:**
- Component 1 (convergence-gated transitions) → Task 3 ✓
- Component 2 (yaw gating + MAX_YAW + settle) → Task 3 (yaw_ok gate, MAX_YAW 0.20, YAW_DONE_TOL) ✓
- Component 3 (far+low reach via larger step budget) → Task 3 (raised `_PHASE_CAP`) ✓
- Component 4 (failure logging) → Task 4 ✓
- Component 5 (gripper obs + bc_rnn_L3) → Tasks 7, 8 ✓
- Component 6 (max_steps discard + cube half-height) → Tasks 5, 6 ✓
- Verification (re-collect + coverage thirds + yield) → Tasks 9, 10, 11 ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type/name consistency:** `cu.is_smooth(steps, max_ee_jerk, max_cube_jump)`, `cu.trajectory_smoothness`, `cu.yaw_error_rad`, `cu.hit_step_cap`, `cu.failure_record` defined in Task 1 and used consistently in Tasks 2/4/5. `gripper_pos` (Task 7) matches `mdp.gripper_pos` use in Task 7 env_cfg and the `gripper` obs key in Task 8 config. `YAW_DONE_TOL`/`MAX_YAW` defined and used in Task 3. ✓

**Note on TDD scope:** The Isaac-coupled controller (`PickPlacePolicy`) and env configs cannot be unit-tested without booting Isaac Sim; their verification is the integration run (Tasks 10–11) plus the pure-helper unit tests (Task 1). This is an intentional, documented limitation.
