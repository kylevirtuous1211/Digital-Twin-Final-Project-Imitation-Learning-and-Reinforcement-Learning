"""Unit tests for the geometry-reactive DAgger oracle (no Isaac)."""
import importlib.util
import pathlib

import torch

# Load the (Isaac-free) oracle module directly by file path — avoids importing the
# package, which would pull in isaaclab/pxr via mdp/__init__.
_PKG = pathlib.Path(__file__).resolve().parents[1] / "source/final_project/final_project/tasks/manager_based/final_project/student_interface/IL/dagger_oracle.py"
_spec = importlib.util.spec_from_file_location("dagger_oracle", _PKG)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
reactive_oracle = _mod.reactive_oracle


def _obs(eef, cube, goal):
    return {
        "eef_pos": torch.tensor([eef], dtype=torch.float32),
        "cube_pos": torch.tensor([cube], dtype=torch.float32),
        "goal_pos": torch.tensor([goal], dtype=torch.float32),
    }


CUBE = [0.56, -0.12, 0.05]
GOAL = [0.45, 0.15, 0.0]


def test_approach_points_toward_cube():
    # eef at home, away from cube, not grasped -> dpos points toward cube xy, gripper open
    o = _obs([0.47, 0.0, 0.38], CUBE, GOAL)
    a = reactive_oracle(o, torch.tensor([0.05]), control_yaw=False)[0]
    assert a[0] > 0 and a[1] < 0          # toward cube (+x, -y)
    assert a[6] > 0.5                      # gripper open


def test_overshoot_recovers_back_toward_cube():
    # THE key test: eef has OVERSHOT the cube (past it in +x, -y), not grasped.
    # Oracle must command BACK toward the cube (the recovery the BC policy lacks).
    o = _obs([0.65, -0.23, 0.18], CUBE, GOAL)
    a = reactive_oracle(o, torch.tensor([0.05]), control_yaw=False)[0]
    assert a[0] < 0, "should move -x back toward cube"
    assert a[1] > 0, "should move +y back toward cube"
    assert a[6] > 0.5, "gripper stays open until at cube"
    assert torch.linalg.vector_norm(a[:3]) > 0.02, "must command a real corrective step, not stall"


def test_at_cube_closes_gripper():
    # eef directly on the cube -> close gripper
    o = _obs([CUBE[0], CUBE[1], CUBE[2] - 0.005], CUBE, GOAL)
    a = reactive_oracle(o, torch.tensor([0.05]), control_yaw=False)[0]
    assert a[6] < -0.5, "gripper should close at the cube"


def test_grasped_transports_toward_goal_closed():
    # cube lifted (grasped) and eef holding it up, away from goal -> move toward goal, gripper closed
    lifted_cube = [0.56, -0.12, 0.20]      # z risen well above rest 0.05 => grasped
    o = _obs([0.56, -0.12, 0.20], lifted_cube, GOAL)
    a = reactive_oracle(o, torch.tensor([0.05]), control_yaw=False)[0]
    assert a[6] < -0.5, "gripper holds closed while carrying"
    # after lift it should head toward goal in xy (goal is +y from cube)
    assert a[1] > 0 or torch.linalg.vector_norm(a[:3]) > 0.0


def test_clamped_to_max_step():
    o = _obs([0.0, 0.0, 1.0], CUBE, GOAL)   # very far
    a = reactive_oracle(o, torch.tensor([0.05]), control_yaw=False)[0]
    assert torch.linalg.vector_norm(a[:3]) <= 0.08 + 1e-6
