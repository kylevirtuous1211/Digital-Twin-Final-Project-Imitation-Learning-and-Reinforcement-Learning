# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around Isaac Lab's robomimic play.py that registers our task first.

Same trick as train_il_bc.py: boot Kit, import final_project (to gym.register),
no-op AppLauncher, then runpy Lab's play.py. play.py scores rollouts using
``env_cfg.terminations.success`` and prints the success rate.

Path to Lab's play.py is taken from $ISAACLAB_PATH (default /home/kyle/Desktop/IsaacLab).

Run with the same args you'd pass to Lab's play.py, e.g.:
    python scripts/play_il_bc.py --task FinalProject-IL-L1-v0 \
        --num_rollouts 50 --checkpoint <best.pth> --headless
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

from isaaclab.app import AppLauncher

ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", "/home/kyle/Desktop/IsaacLab")
LAB_PLAY_PY = os.path.join(ISAACLAB_PATH, "scripts/imitation_learning/robomimic/play.py")


def main() -> None:
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app

    import final_project  # noqa: F401  registers FinalProject-* tasks

    import isaaclab.app as app_mod

    class _AlreadyLaunched:
        """Stand-in so Lab's play.py doesn't relaunch Kit."""

        def __init__(self, *args, **kwargs) -> None:
            self._app = app

        @property
        def app(self):
            return self._app

        @staticmethod
        def add_app_launcher_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            return parser

    app_mod.AppLauncher = _AlreadyLaunched

    if not os.path.exists(LAB_PLAY_PY):
        raise FileNotFoundError(f"Lab play.py not found at {LAB_PLAY_PY}; set $ISAACLAB_PATH.")

    sys.argv[0] = LAB_PLAY_PY
    runpy.run_path(LAB_PLAY_PY, run_name="__main__")


if __name__ == "__main__":
    main()
