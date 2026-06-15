# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""No-GPU offline gate for DAgger rounds (the cheap pre-flight before any Isaac eval).

Optimistic kinematic closed-loop: load a checkpoint, and for each held-out demo,
integrate the eef by the policy's own clamped dpos (cube/goal frozen at the demo
start, IK assumed perfect = best case). Score whether the eef ever reaches the
grasp band (<2.5 cm of the cube) AND the gripper closes. If the policy cannot even
reach the cube in this optimistic harness, it will fail the real Isaac rollout, so
do not spend GPU on it.

    python scripts/dagger_gate.py --checkpoint <ckpt.pth> --dataset datasets/il_L1.hdf5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils

GRASP_BAND = 0.025
MAX_STEP = 0.08
KIN_STEPS = 441


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=30, help="held-out demos to probe")
    args = ap.parse_args()

    rp, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=torch.device("cpu"))
    algo = rp.policy
    algo.set_eval()
    torch.set_grad_enabled(False)

    f = h5py.File(args.dataset, "r")
    valid = [d.decode() if isinstance(d, bytes) else d for d in f["mask/valid"][:]]
    demos = valid[: args.n]
    keys = list(f["data"][demos[0]]["obs"].keys())

    nsucc = 0
    mindists = []
    for demo in demos:
        g = f["data"][demo]["obs"]
        frozen = {k: g[k][0].astype(np.float32) for k in keys}
        eef = frozen["eef_pos"].astype(np.float64).copy()
        cube = frozen["cube_pos"].astype(np.float64)
        algo.reset()
        mind = 1e9
        grasped = False
        for _ in range(KIN_STEPS):
            o = {}
            for k in keys:
                v = eef.astype(np.float32) if k == "eef_pos" else frozen[k]
                o[k] = torch.tensor(v[None], dtype=torch.float32)
            a = algo.get_action(o)[0].numpy().astype(np.float64)
            dp = a[:3]
            n = np.linalg.norm(dp)
            if n > MAX_STEP:
                dp = dp * MAX_STEP / n
            eef = eef + dp
            d = float(np.linalg.norm(eef - cube))
            mind = min(mind, d)
            if d < GRASP_BAND and a[6] < 0:
                grasped = True
        mindists.append(mind)
        if grasped:
            nsucc += 1
    f.close()
    rate = nsucc / len(demos) * 100
    print(f"GATE: reached grasp band AND closed = {nsucc}/{len(demos)} ({rate:.0f}%)  "
          f"median_min_eef_cube_dist={np.median(mindists):.4f}m  (band={GRASP_BAND})")
    print(f"GATE_RATE={rate:.1f}")


if __name__ == "__main__":
    main()
