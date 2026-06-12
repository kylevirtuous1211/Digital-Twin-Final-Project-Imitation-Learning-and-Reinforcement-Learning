"""IL Level 3 — student environment config (L2 + platform height randomization)."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from final_project.tasks.manager_based.final_project.env.il_eval_env import ILEvalEnvL3Cfg


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # TODO: add observation terms — both platform heights are now randomised

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ── Optional extensions ───────────────────────────────────────────────────────
# Add sensors — subclass ILSceneCfg (has robot, cube, source/target platforms):
# from final_project.tasks.manager_based.final_project.env.il_eval_env import ILSceneCfg
# from isaaclab.sensors import ContactSensorCfg
# @configclass
# class CustomSceneCfg(ILSceneCfg):
#     contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", ...)

# Add termination conditions (time_out always present via ILTerminationsCfg):
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from final_project.tasks.manager_based.final_project.env.il_eval_env import ILTerminationsCfg
# @configclass
# class CustomTerminationsCfg(ILTerminationsCfg):
#     cube_dropped = DoneTerm(func=..., params={...})


@configclass
class EnvCfg(ILEvalEnvL3Cfg):
    """IL Level 3 — L2 + source/target platform height randomized ∈[0, 0.3] m."""

    # scene: CustomSceneCfg = CustomSceneCfg(num_envs=64, env_spacing=2.5)
    # terminations: CustomTerminationsCfg = CustomTerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
