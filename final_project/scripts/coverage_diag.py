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
