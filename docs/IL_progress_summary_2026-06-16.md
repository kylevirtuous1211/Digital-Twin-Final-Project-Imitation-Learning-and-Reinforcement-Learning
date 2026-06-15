# IL Pipeline — Overnight Progress Summary (2026-06-16)

## Headline results (Isaac eval, 50 envs, horizon 800, FINAL checkpoint)

| Level | Before | After (this work) | Best checkpoint | Pass bar |
|---|---|---|---|---|
| **L1** | 0% (clean BC-RNN) | **44%** (obs-redesign + DART) | epoch 500 | 80% |
| **L3** | 0% (clean BC-RNN) | **34%** (obs-redesign + DART) | epoch 800 | 80% |
| **L2** | — | deferred (needs env rewrite) | — | 80% |

Large progress (0 → 44%/34%), but **not yet at the 80% pass threshold.** The recipe
is validated and generalizes across levels; getting to 80% needs more iteration
(see "Path to 80%").

**Checkpoint sweep matters, and the peak differs per level:**
- L1: epoch 500 = **44%** > epoch 800 = 38% (later epochs overfit; `best_validation`
  picked epoch 138 = 0%).
- L3: epoch 800 = **34%** > epoch 500 = 22% (harder task — needs more training).

So: don't trust `best_validation`; eval several late checkpoints per level. (Caveat:
evals run back-to-back starve memory → false 0/low results that finish in ~50 s;
verify each with a clean, memory-settled run — real evals pace over minutes and the
success count climbs gradually.)

## The journey — what was actually wrong

1. **Reach fix (the #1 issue, biggest single win).** The target platform was placed
   0.81–0.94 m from the robot base, but the Franka's practical reach is ~0.85 m, so
   **~77% of placements were kinematically impossible** (expert yield stuck at ~23%).
   Moved `target_platform (0.6,0.3) → (0.45,0.15)` → absolute targets ~0.60–0.74 m
   (reachable) → expert yield jumped to **80–99%**. This fixed feasibility for L1/L2/L3.
   NOTE: grading uses the TA's `EvalSceneCfg`, not ours — confirm the TA's target
   placement matches (we don't have their eval config).

2. **Closed-loop BC failure (the hard part).** Even on feasible data, BC-RNN scored
   **0% closed-loop** despite a tiny val loss (0.0003). A multi-agent diagnosis
   (31 agents, adversarially verified) found: the policy has **~zero corrective
   feedback gain** — it replays an open-loop velocity profile and never corrects
   lateral error, so it overshoots the cube ~15 cm, stalls, and never grasps. Root
   cause: the 800 demos all start from **one fixed pose** with **zero recovery
   examples** → covariate shift.

3. **What we tried:** DAgger round 1 → 0% (recovery data diluted by the clean
   anchor); pure-DART → 10%; **obs-redesign + DART → 38% (L1) / 34% (L3)** ← the fix.

## The fix (the recipe that worked)

- **Object-relative observations:** added `eef_to_cube` (= cube − eef) and
  `cube_to_goal` (= goal − cube) + `gripper` state. These hand the policy the
  *error vector directly* instead of forcing it to infer relative position from
  absolute poses → feedback becomes learnable. (NVIDIA-style obs design.)
- **DART noise injection:** `collect_il_demos.py --noise 0.04` perturbs the
  *stepped* action while recording the *clean* expert label, so the expert visits
  a **tube of states** → the BC sees diverse (state → correct-action) pairs and
  learns a feedback law instead of an open-loop time profile.
- **Use the FINAL checkpoint, not `best_validation`.** robomimic's `on_best_validation`
  picked an underfit early checkpoint (epoch 138 → 0%) while epoch 800 → 38%.
  Val loss is gripper-dominated and an unreliable selector here.

## Gotchas / lessons (so they don't bite again)

- **Eval-after-retrain memory artifact:** booting an Isaac eval immediately after a
  retrain (before its memory frees) starves the sim → false **0/50 in ~50 s**.
  Real evals pace over minutes and their success count climbs gradually. The chains
  now add a memory-settle delay; trust the clean re-run.
- **The offline kinematic gate (`dagger_gate.py`) is unreliable** — it reported 0%
  for a policy that really scored 10–38%. Use the Isaac eval as the metric.
- The **reactive oracle** (geometry-only) deadlocks at grasp (won't lift until the
  cube lifts) → 0% as an actor; the scripted waypoint expert (timed GRASP→LIFT) is
  the one to use. That's why DART uses the scripted expert + noise.

## Path to 80% (next steps for the morning)

- **More DART rounds / tune noise** (0.04 → sweep), and/or **combine with DAgger**
  (visit the policy's *actual* overshoot states, not just a noise tube).
- **L2:** its `env_cfg_L2` is a joint-space stub (`joint_pos`/`joint_vel` obs +
  `JointPositionActionCfg`). Rewrite it to the IK-delta-pose + pose/relative-obs
  setup (essentially L3 minus height randomization), then apply the same recipe.
- **Checkpoint sweep / longer training** to find the peak.

## Artifacts (branch `feat/Kyle/l3-expert-robustness`, all committed)

- Datasets: `datasets/il_L1_dart2.hdf5`, `datasets/il_L3_dart.hdf5`
- Checkpoints: `logs/robomimic/FinalProject-IL-L1-v0/bc_rnn/*/models/model_epoch_800.pth`
  and `.../FinalProject-IL-L3-v0/bc_rnn_L3/*/models/model_epoch_800.pth`
- Code: object-relative obs (`mdp/observations.py`), DART noise (`collect_il_demos.py`),
  vectorized eval (`eval_il_bc_parallel.py`), DAgger pipeline (kept as a tool).

## Reproduce (L1)

```bash
PY=/home/kyle/Desktop/IsaacLab/.venv/bin/python
cd final_project
# collect (DART, obs-redesign is in the env cfg)
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/collect_il_demos.py --task FinalProject-IL-L1-v0 \
  --num_envs 32 --num_demos 800 --noise 0.04 --max_ee_jerk 0 --max_cube_jump 0 \
  --dataset datasets/il_L1_dart2.hdf5 --headless
# train
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/train_il_bc.py --task FinalProject-IL-L1-v0 \
  --algo bc_rnn --dataset ./datasets/il_L1_dart2.hdf5 --epochs 800
# eval (use the FINAL model_epoch_800.pth, not best_validation)
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/eval_il_bc_parallel.py --task FinalProject-IL-L1-v0 \
  --num_envs 50 --horizon 800 --checkpoint <.../models/model_epoch_800.pth> --headless
```
