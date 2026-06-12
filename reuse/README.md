# reuse/ — staged scripts to reuse for the Digital Twin Final Project

Copied here from elsewhere in the repo so the final project can reuse them.
Nothing here is wired into the deliverable scaffold yet — see "Adaptation
needed" below.

## What's here

### `run_in_isaac.py`
Sends a `.py` snippet over TCP (port 8226) to an **already-running** Isaac Sim
GUI via the `isaacsim.code_editor.vscode` executor extension.
- Source: `../../run_in_isaac.py`
- Use for: interactive scene inspection / one-off commands in the live viewer
  (e.g. while recording the demo video).
- **Not** for training — the training/collection scripts boot their own Kit app
  via `AppLauncher` and run standalone (`python scripts/...`).

### `il_pipeline/` — complete, working IL pick-place pipeline
Lifted from `../../franka_pickplace_il/` (the iterated successor to the
`111030034_IL_HandsOn` snapshot). This is a full
`Configure Env → collect demos → train BC → play` loop that already produced
trained checkpoints.

| File | Role | Maps to final-project deliverable |
|---|---|---|
| `scripts/pickplace_policy.py` | Scripted state-machine expert (APPROACH→DESCEND→CLOSE→LIFT→TRANSLATE→PLACE→RELEASE) + Robomimic HDF5 collector | **IL data-collection env / expert rollout (20 pts)** |
| `scripts/train_bc.py` | Wrapper that registers the task then runs Isaac Lab's robomimic `train.py` | IL training command (README req.) |
| `scripts/play_bc.py` | Self-contained BC rollout + success-rate eval; per-dim action un-normalization + inference-time release heuristic | IL policy eval / success-rate table |
| `scripts/perdim_normalize_hdf5.py` | Per-dimension action normalization of the demo dataset | IL data tooling |
| `scripts/filter_smooth_demos.py` | Drops jittery demos before training | IL data tooling |
| `source/.../franka_pickplace_env_cfg.py` | Full manager-based env (scene, IK delta actions, obs, events, rewards, terminations) | Reference for IL `EnvCfg` obs/actions |
| `source/.../mdp/observations.py` | `ee_frame_pos`, `ee_frame_quat`, `object_pos_in_env_frame`, `object_quat_w` | **Fills the IL obs `TODO`s** (cube/target pose) |
| `source/.../mdp/terminations.py` | `task_success` (cube-at-goal & gripper-open, held N steps) | IL success detection per level |
| `source/.../mdp/constants.py` | Success thresholds (pos 0.05 m, finger-open 0.035, hold 50 steps) | Note: spec L1 wants pos < **0.05 m**; matches |
| `source/.../agents/robomimic/bc.json` | BC config (actor 1024×1024, 200 epochs) | Compare with scaffold's `bc.json` |

## Adaptation needed (this pipeline ≠ the final-project scaffold)

The final project ships its own scaffold under
`../final_project/source/final_project/...` with a **TA-owns-eval /
student-owns-obs+actions** split. The reused pipeline differs:

1. **Task id**: reused = `Template-Franka-Pickplace-v0`; scaffold =
   `FinalProject-IL-L1/L2/L3-v0`. Obs/action terms must be ported into the
   scaffold's `student_interface/IL/env_cfg_L*.py` (which only stub
   `joint_pos`/`joint_vel`).
2. **Action space**: reused uses **differential IK delta pose** (7-dim:
   3 pos + 3 axis-angle + gripper); the scaffold stub uses **joint-position**
   control. The expert state machine + bc.json obs keys assume the IK action
   layout — pick one and keep collector/env/bc.json consistent.
3. **Scene asset names**: reused calls the place pad `target`; the scaffold uses
   `source_platform` / `target_platform` and randomizes cube-on-platform via
   `il_events.reset_cube_on_source_platform`. Obs/termination `SceneEntityCfg`
   names must be remapped.
4. **Levels**: reused is single-level. Final project needs L1 (pos<5cm),
   L2 (+yaw<15°), L3 bonus (+platform heights) — the expert must place with
   orientation for L2/L3.
5. **Hardcoded path**: `scripts/train_bc.py` has
   `/home/kyle/Desktop/IsaacLab/scripts/imitation_learning/robomimic/train.py` —
   update for the target machine.

## Not copied (kept lean per spec)
Archived datasets (`datasets/*.hdf5`), training logs, and trained checkpoints
(`checkpoints/v*.pth`) remain in `../../franka_pickplace_il/` — pull a checkpoint
from there if you want a warm start, but the final env differs so expect to
retrain.
