# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around Isaac Lab's robomimic train.py that registers our task first.

Lab's train.py imports a fixed set of isaaclab_tasks modules but doesn't know about
our external ``final_project`` package, so ``gym.spec("FinalProject-IL-L1-v0")``
would fail. This wrapper:
  1. Boots Kit headless (so pxr is available and our task package can import).
  2. Imports final_project -> triggers gym.register(...).
  3. Replaces isaaclab.app.AppLauncher with a no-op so train.py won't boot Kit twice.
  4. Executes train.py as __main__, forwarding all CLI args.

Path to Lab's train.py is taken from $ISAACLAB_PATH (default /home/kyle/Desktop/IsaacLab).

Run exactly the args you'd pass to Lab's train.py, e.g.:
    python scripts/train_il_bc.py --task FinalProject-IL-L1-v0 --algo bc \
        --dataset ./datasets/il_L1.hdf5 --epochs 2000
"""
from __future__ import annotations

import os
import runpy
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", "/home/kyle/Desktop/IsaacLab")
LAB_TRAIN_PY = os.path.join(ISAACLAB_PATH, "scripts/imitation_learning/robomimic/train.py")


def main() -> None:
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app

    import final_project  # noqa: F401  registers FinalProject-* tasks

    import isaaclab.app as app_mod

    class _AlreadyLaunched:
        """Stand-in so Lab's train.py doesn't relaunch Kit."""

        def __init__(self, *args, **kwargs) -> None:
            self._app = app

        @property
        def app(self):
            return self._app

    app_mod.AppLauncher = _AlreadyLaunched

    if not os.path.exists(LAB_TRAIN_PY):
        raise FileNotFoundError(f"Lab train.py not found at {LAB_TRAIN_PY}; set $ISAACLAB_PATH.")

    sys.argv[0] = LAB_TRAIN_PY
    runpy.run_path(LAB_TRAIN_PY, run_name="__main__")


if __name__ == "__main__":
    main()
