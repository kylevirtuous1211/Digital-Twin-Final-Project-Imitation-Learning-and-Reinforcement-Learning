# L3 Expert Robustness — Design Spec

**Date:** 2026-06-14
**Status:** Approved design, pending implementation plan
**Owner area:** `final_project/scripts/collect_il_demos.py` (`PickPlacePolicy`)

## Problem

The scripted pick-place expert solves only ~28–32% of L3's randomized configs, so
the success-gated dataset covers only the easy subset of the (fixed) L3
distribution. A BC policy trained on it will generalize poorly at evaluation on the
full L3 distribution. The L3 randomization ranges are **fixed by the task/eval spec**
and must not be narrowed.

### Diagnosis (from saved-demo config coverage vs. uniform 33/33/33)

| Config axis | low third | mid | high third | Verdict |
|---|---|---|---|---|
| Target yaw (0–45°) | 57% | 30% | **12.6%** | #1 failure: large yaw rarely succeeds |
| Source/cube height | 9.9% | 20% | 70% | low platforms (cube near floor) fail |
| Target platform height | 23% | 32% | 45% | OK (tall targets fine) |

- Reach distance (cube→goal xy) is 0.52–0.73 m — near the Franka's practical
  workspace limit regardless of controller.
- Root cause: phases advance on fixed `_PHASE_CAP` step-caps and an ease clock,
  not on actual convergence, so on hard configs the controller "gives up" before
  arriving / before the cube yaw is aligned, then the episode times out and is
  discarded.

## Goal

Make `PickPlacePolicy` converge on the success condition instead of timing out on
fixed caps, raising feasible coverage of the full L3 distribution — especially
large target yaw and low+far cubes. Honest expectation: maximize *feasible*
coverage, not guaranteed 100% (some far+low configs may be physically
workspace-limited).

## Non-goals (YAGNI)

- No motion planner / closed-loop replanning / reachability-aware re-posing
  (that is "Approach B", explicitly deferred).
- No change to action space (7-D rel-IK), HDF5 layout, success term, or the
  smoothness filter. (Obs space *does* gain a gripper term — see Component 5.)
- No change to L3 randomization ranges.

## Considered and deferred (review feedback)

- **#1 Pre-grasp yaw alignment to the cube** — NOT applicable. Verified
  `reset_cube_on_source_platform` leaves cube orientation at default; the cube
  always spawns axis-aligned (yaw 0). Only the *target platform* yaw is
  randomized. The gripper already grasps on opposing faces; all yaw work is
  post-grasp (Component 2).
- **#3 Covariate-shift mitigation (DART action-noise / DAgger)** — valid but
  deferred: action noise conflicts with the smoothness filter (noise → jerk →
  rejected), DAgger needs the trained policy in the loop, and neither addresses
  the coverage gap that is the current bottleneck. Revisit after coverage is fixed.
- **#4 Trajectory diversity (randomized offsets/speeds)** — same smoothness
  tension; lower priority than coverage. Deferred.
- **#6 Absolute full-orientation control** — YAGNI: L3 randomizes yaw only (no
  pitch/roll); relative-yaw suffices. Deferred.
- **#7 Cube-size hardcoding / zero rewards** — cube size is not randomized in L3
  (verified), so grasp constants are safe; zero rewards are fine for pure BC.

## Design

All changes are confined to `PickPlacePolicy` and the per-episode bookkeeping in
`run()` in `collect_il_demos.py`. Data flow (obs → 7-D action → success+smoothness
filter → HDF5) is unchanged.

### Component 1 — Convergence-gated phase transitions
- Motion phases (APPROACH, DESCEND, LIFT, TRANSLATE, PLACE) advance/complete only
  when the EE has actually arrived (`reached`, i.e. position error < `_PHASE_TOL`),
  rather than when the sinusoidal-ease clock alone finishes. The waypoint keeps
  easing; advancement is gated on arrival.
- Replace the premature `cap_hit`-driven advance with a generous global safety cap
  (raise `_PHASE_CAP` ~2–3×). Configs that cannot converge run until the env's
  episode timeout → truncated → discarded (no infinite loops).
- Dwell phases (GRASP, RELEASE) remain time-based.

### Component 2 — Yaw convergence gating (primary fix)
- Keep yaw control active from LIFT onward (`self.e >= 3`), as today.
- Raise the per-step yaw rate: `MAX_YAW` 0.10 → 0.20 rad/step (≈11.5°/step), so a
  45° error aligns in a handful of steps.
- Gate PLACE/RELEASE completion on yaw: do not finish the place/release phase
  until `|yaw_err| < YAW_DONE_TOL` where `YAW_DONE_TOL = 0.175 rad (~10°)` — a
  margin under the 15° (`PLACE_YAW_THRESHOLD = 0.2618`) success bar.
- Add a short settle: hold EE pose while the yaw finishes converging before the
  gripper opens, so the released cube yaw is stable.

### Component 3 — Far + low reach
- Benefit primarily from Component 1's larger step budget (more time to converge
  on far+low cubes).
- Minimal anchor tuning only if measured failures show a systematic gap (e.g.
  small adjustment to `APPROACH_CLEAR` / transit height for low cubes). No
  workspace-reachability logic.

### Component 4 — Reachability / failure logging
- Per finished episode, append one JSONL record to a diagnostics side-file
  (default `datasets/il_L3_collect_diag.jsonl`) with: `success` (bool),
  `cube_pos`, `goal_pos`, source/target heights, `target_yaw_deg`,
  `final_pos_err`, `final_yaw_err_deg`, `last_phase`.
- Purpose: separate *infeasible* failures (position never reached → workspace
  limit) from *fixable* ones (yaw not converged), so the remaining tail is
  understood, not guessed.

### Component 5 — Add gripper state to L3 observations (review feedback #2)
- L3 obs currently carries only poses (`eef_pos/quat`, `cube_pos/quat`,
  `goal_pos/quat`) — no gripper state. The open/closed command is a latent the BC
  policy must infer; including it removes that ambiguity (NVIDIA's manipulation
  obs include `gripper_pos`).
- Add a `gripper` ObsTerm to `env_cfg_L3.py` reading the `panda_finger` joint
  positions (gripper width), and subscribe to it in `bc_rnn_L3.json`'s
  `low_dim` obs list. The collector auto-saves any obs key, so re-collection
  picks it up. Obs dim changes → L3 must be retrained (already planned).

### Component 6 — Correctness fixes (review feedback #5 + verification finding)
- **max_steps truncation:** if an episode's buffer hits `--max_steps` (the cap
  stops appending), do NOT save it — `_flush_episode` would otherwise set
  `dones[-1]=1` on a non-terminal state, mislabeling a truncated trajectory as a
  successful demo. Component 1 lengthens hard-config episodes, making this reachable.
  Discard episodes whose buffer length reached the cap.
- **Cube reset height bug:** `il_events._CUBE_HALF_HEIGHT = 0.05` but the cube is
  0.05 m total (half = 0.025, per `constants.CUBE_HALF`). The cube spawns ~2.5 cm
  too high and drops at reset. Fix to 0.025 for a clean rest (use
  `constants.CUBE_HALF`).

## Tunable parameters (final values)

| Name | Old | New |
|---|---|---|
| `MAX_YAW` (rad/step) | 0.10 | 0.20 |
| `YAW_DONE_TOL` (rad) | — | 0.175 (~10°) |
| `_PHASE_CAP` (motion phases) | [160,240,—,160,280,280,—] | ~2–3× (generous safety) |
| place/release completion | position-only | position **and** yaw converged + settle |

## Verification

1. Re-collect L3 fresh (`il_L3.hdf5`) with the improved expert + existing
   smoothness filter.
2. Re-run the thirds coverage diagnostic. **Pass criteria:**
   - High-yaw third rises from 12.6% toward ~33% (target ≥ 25%).
   - Low-source-height third rises from 9.9% (target ≥ 20%).
   - Attempt yield (saved/attempts) increases above ~32%.
3. Inspect `il_L3_collect_diag.jsonl`: remaining failures should be concentrated
   in far+low (workspace-limited) configs, not large-yaw.

## Risks

- Removing premature caps lengthens per-attempt time on hard configs; net
  collection wall-clock may rise even as yield improves. Acceptable.
- Higher `MAX_YAW` could induce jerk; the smoothness filter (EE-jerk, cube-jump)
  guards dataset quality, and the settle phase mitigates release-time jitter.
- Some far+low configs may remain infeasible (workspace limit); Component 4
  quantifies this so we do not chase impossible coverage.
