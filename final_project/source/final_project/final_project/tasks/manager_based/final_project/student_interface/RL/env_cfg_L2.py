"""RL Level 2 — student environment config (randomised goal position + heading)."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from final_project.tasks.manager_based.final_project.env.rl_eval_env import RLEvalEnvL2Cfg


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # TODO: add observation terms (goal heading is now randomised — include it!)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_wheel_left", "joint_wheel_right"],
        scale=10.0,
    )


@configclass
class RewardsCfg:
    pass  # TODO: add reward terms


# ── Optional extensions ───────────────────────────────────────────────────────
# Add sensors — subclass RLSceneCfg (has robot, ground, light):
# from final_project.tasks.manager_based.final_project.env.rl_eval_env import RLSceneCfg
# from isaaclab.sensors import ContactSensorCfg
# @configclass
# class CustomSceneCfg(RLSceneCfg):
#     contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", ...)

# Add termination conditions (time_out always present via RLTerminationsCfg):
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from final_project.tasks.manager_based.final_project.env.rl_eval_env import RLTerminationsCfg
# @configclass
# class CustomTerminationsCfg(RLTerminationsCfg):
#     goal_reached = DoneTerm(func=..., params={...})


@configclass
class EnvCfg(RLEvalEnvL2Cfg):
    """RL Level 2 — goal pos_x∈[-1,1], pos_y∈[3,4], heading∈[π/4, 3π/4]."""

    # scene: CustomSceneCfg = CustomSceneCfg(num_envs=64, env_spacing=15.0)
    # terminations: CustomTerminationsCfg = CustomTerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
