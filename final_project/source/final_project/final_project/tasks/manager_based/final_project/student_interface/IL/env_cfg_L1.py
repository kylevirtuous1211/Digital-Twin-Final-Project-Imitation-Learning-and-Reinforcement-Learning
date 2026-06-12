"""IL Level 1 — student environment config (fixed target, arbitrary-yaw pick).

Action space: differential IK delta-pose (task space) + binary gripper. This
overrides the joint-position stub so a scripted state-machine expert can be
written directly in task space (see scripts/collect_il_demos.py). The spec lets
students define their own action space; this file IS the policy_cfg_L1 deliverable.

The same EnvCfg is used for (a) expert data collection and (b) evaluation: it
carries a ``success`` termination, which the collector uses to flush demos and
Isaac Lab's robomimic play.py uses to score rollouts.
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from final_project.tasks.manager_based.final_project.env.il_eval_env import ILEvalEnvL1Cfg, ILTerminationsCfg

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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    # Relative-pose IK: action = (dx, dy, dz, droll, dpitch, dyaw) delta of the EE
    # pose in the base frame. The scripted expert sends only a position delta toward
    # a smoothly interpolated Cartesian waypoint (NVIDIA PickPlaceController-style
    # sinusoidal easing) with zero rotation delta, leaving orientation free so the
    # arm can reach the far target platform; the DLS IK tracks it -> smooth motion.
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
    """Adds a success term on top of the eval env's time_out.

    Required by robomimic play.py (reads env_cfg.terminations.success) and used by
    the data collector to flush only successful episodes.
    """

    success = DoneTerm(
        func=mdp.place_success,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target_platform"),
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
    )


@configclass
class EnvCfg(ILEvalEnvL1Cfg):
    """IL Level 1 — fixed source platform x∈[0.5,0.6] y∈[-0.3,-0.2], fixed target."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: CustomTerminationsCfg = CustomTerminationsCfg()
