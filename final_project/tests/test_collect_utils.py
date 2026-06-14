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
