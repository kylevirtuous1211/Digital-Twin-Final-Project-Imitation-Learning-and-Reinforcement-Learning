"""IL Level 3 (bonus) — student environment config.

Extends L1 with: source & target platform HEIGHT randomization (z in [0, 0.3] m)
and target platform YAW (in [-pi/4, pi/4]). The cube must be picked from a
randomized-height platform and placed on a different-height platform aligned to
the target yaw.

Same action space as L1 (relative-pose IK + binary gripper). Adds a ``goal_quat``
observation so the policy can see the target orientation, and an orientation-aware
success termination.
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from final_project.tasks.manager_based.final_project.env.il_eval_env import ILEvalEnvL3Cfg, ILTerminationsCfg

from . import mdp


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        eef_pos = ObsTerm(func=mdp.ee_pos, params={"robot_cfg": SceneEntityCfg("robot")})
        eef_quat = ObsTerm(func=mdp.ee_quat, params={"robot_cfg": SceneEntityCfg("robot")})
        cube_pos = ObsTerm(func=mdp.object_pos_in_env_frame, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_quat = ObsTerm(func=mdp.object_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        goal_pos = ObsTerm(func=mdp.object_pos_in_env_frame, params={"asset_cfg": SceneEntityCfg("target_platform")})
        goal_quat = ObsTerm(func=mdp.object_quat_w, params={"asset_cfg": SceneEntityCfg("target_platform")})
        gripper = ObsTerm(func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg("robot")})
        # Object-relative error vectors (same lever as L1): direct feedback signal.
        eef_to_cube = ObsTerm(func=mdp.eef_to_cube, params={"robot_cfg": SceneEntityCfg("robot"), "cube_cfg": SceneEntityCfg("cube")})
        cube_to_goal = ObsTerm(func=mdp.cube_to_goal, params={"cube_cfg": SceneEntityCfg("cube"), "goal_cfg": SceneEntityCfg("target_platform")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    # Relative-pose IK: (dx, dy, dz, droll, dpitch, dyaw). The expert uses the
    # position delta to reach waypoints and the dyaw channel to align the cube to
    # the target platform yaw; orientation is otherwise left free for reach.
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class CustomTerminationsCfg(ILTerminationsCfg):
    """Adds an orientation- and height-aware success term on top of time_out."""

    success = DoneTerm(
        func=mdp.place_success_oriented,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target_platform"),
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
    )


@configclass
class EnvCfg(ILEvalEnvL3Cfg):
    """IL Level 3 — L2 + source/target platform height randomized in [0, 0.3] m + target yaw."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: CustomTerminationsCfg = CustomTerminationsCfg()
