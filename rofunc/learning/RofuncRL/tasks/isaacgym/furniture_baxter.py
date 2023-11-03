""" Define baxter environment class FurnitureBaxterEnv. """

from collections import OrderedDict

import numpy as np
import gym.spaces

from rofunc.learning.RofuncRL.tasks import transform_utils as T
from .furniture import FurnitureEnv
from ..util.logger import logger


class FurnitureBaxterEnv(FurnitureEnv):
    """
    Baxter robot environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.agent_type = "Baxter"

        super().__init__(config)

    @property
    def observation_space(self):
        """
        Returns the observation space.
        """
        ob_space = super().observation_space

        if self._robot_ob:
            if self._control_type in ["impedance", "torque"]:
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        (7 + 7 + 2 + 3 + 4 + 3 + 3) * 2,
                    ),  # qpos, qvel, gripper, eefp, eefq, velp, velr
                )
            elif self._control_type in ["ik", "ik_quaternion"]:
                ob_space.spaces["robot_ob"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=((3 + 4 + 3 + 3 + 2) * 2,),  # pos, quat, velp, velr, gripper
                )

        return ob_space

    @property
    def dof(self):
        """
        Returns the DoF of the robot.
        """
        dof = 0  # 'No' Agent
        if self._control_type in ["impedance", "torque"]:
            dof = (7 + 1) * 2 + 1
        elif self._control_type == "ik":
            dof = (3 + 3 + 1) * 2 + 1  # (move, rotate, select) * 2 + connect
        elif self._control_type == "ik_quaternion":
            dof = (3 + 4 + 1) * 2 + 1  # (move, rotate, select) * 2 + connect
        return dof

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        ob, _, done, _ = super()._step(a)

        reward, _done, info = self._compute_reward(a)
        done = done or _done

        info["ac"] = a

        if self._success:
            logger.info("Success!")

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body1 = self.sim.model.body_id2name(id1)
        self._target_body2 = self.sim.model.body_id2name(id2)

    def _get_obs(self):
        """
        Returns the current observation.
        """
        state = super()._get_obs()

        # proprioceptive features
        if self._robot_ob:
            robot_states = OrderedDict()
            if self._control_type in ["impedance", "torque"]:
                for arm in self._arms:
                    robot_states[arm + "_joint_pos"] = np.array(
                        [
                            self.sim.data.qpos[x]
                            for x in self._ref_joint_pos_indexes[arm]
                        ]
                    )
                    robot_states[arm + "_joint_vel"] = np.array(
                        [
                            self.sim.data.qvel[x]
                            for x in self._ref_joint_vel_indexes[arm]
                        ]
                    )
                    robot_states[arm + "_gripper_qpos"] = np.array(
                        [
                            self.sim.data.qpos[x]
                            for x in self._ref_gripper_joint_pos_indexes[arm]
                        ]
                    )
                    robot_states[arm + "_eef_pos"] = np.array(
                        self.sim.data.site_xpos[self.eef_site_id[arm]]
                    )
                    robot_states[arm + "_eef_quat"] = T.convert_quat(
                        self.sim.data.get_body_xquat(arm + "_hand"), to="xyzw"
                    )
                    robot_states[arm + "_eef_velp"] = np.array(
                        self.sim.data.site_xvelp[self.eef_site_id[arm]]
                    )  # 3-dim
                    robot_states[arm + "_eef_velr"] = self.sim.data.site_xvelr[
                        self.eef_site_id[arm]
                    ]  # 3-dim

            else:
                for arm in self._arms:
                    robot_states[arm + "_gripper_qpos"] = np.array(
                        [
                            self.sim.data.qpos[x]
                            for x in self._ref_gripper_joint_pos_indexes[arm]
                        ]
                    )
                    robot_states[arm + "_eef_pos"] = np.array(
                        self.sim.data.site_xpos[self.eef_site_id[arm]]
                    )
                    robot_states[arm + "_eef_quat"] = T.convert_quat(
                        self.sim.data.get_body_xquat(arm + "_hand"), to="xyzw"
                    )
                    robot_states[arm + "_eef_velp"] = np.array(
                        self.sim.data.site_xvelp[self.eef_site_id[arm]]
                    )  # 3-dim
                    robot_states[arm + "_eef_velr"] = self.sim.data.site_xvelr[
                        self.eef_site_id[arm]
                    ]  # 3-dim

            state["robot_ob"] = np.concatenate(
                [x.ravel() for _, x in robot_states.items()]
            )

        return state

    def _get_reference(self):
        """
        Sets up references to robot joints and objects.
        """
        super()._get_reference()

        self.l_finger_geom_ids = {
            "left": [
                self.sim.model.geom_name2id(x)
                for x in self.gripper["left"].left_finger_geoms
            ],
            "right": [
                self.sim.model.geom_name2id(x)
                for x in self.gripper["right"].left_finger_geoms
            ],
        }
        self.r_finger_geom_ids = {
            "left": [
                self.sim.model.geom_name2id(x)
                for x in self.gripper["left"].right_finger_geoms
            ],
            "right": [
                self.sim.model.geom_name2id(x)
                for x in self.gripper["right"].right_finger_geoms
            ],
        }

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes_all = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes_all = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]
        pos_len = len(self._ref_joint_pos_indexes_all)
        vel_len = len(self._ref_joint_vel_indexes_all)
        self._ref_joint_pos_indexes = {
            "right": self._ref_joint_pos_indexes_all[: pos_len // 2],
            "left": self._ref_joint_pos_indexes_all[pos_len // 2 :],
        }
        self._ref_joint_vel_indexes = {
            "right": self._ref_joint_vel_indexes_all[: vel_len // 2],
            "left": self._ref_joint_vel_indexes_all[vel_len // 2 :],
        }

        # indices for grippers in qpos, qvel
        gripper_left_joints = list(self.gripper["left"].joints)
        gripper_right_joints = list(self.gripper["right"].joints)
        self.gripper_joints = gripper_left_joints + gripper_right_joints
        self._ref_gripper_joint_pos_indexes_all = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
        ]
        self._ref_gripper_joint_vel_indexes_all = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
        ]
        self._ref_gripper_joint_pos_indexes = {
            "left": [
                self.sim.model.get_joint_qpos_addr(x) for x in gripper_left_joints
            ],
            "right": [
                self.sim.model.get_joint_qpos_addr(x) for x in gripper_right_joints
            ],
        }
        self._ref_gripper_joint_vel_indexes = {
            "left": [
                self.sim.model.get_joint_qvel_addr(x) for x in gripper_left_joints
            ],
            "right": [
                self.sim.model.get_joint_qvel_addr(x) for x in gripper_right_joints
            ],
        }

        # IDs of sites for gripper visualization
        self.eef_site_id = {
            "left": self.sim.model.site_name2id("l_g_grip_site"),
            "right": self.sim.model.site_name2id("grip_site"),
        }

    def _compute_reward(self, ac):
        """
        Computes reward of the current state.
        """
        return super()._compute_reward(ac)


def main():
    from ..config import create_parser

    parser = create_parser(env="FurnitureBaxterEnv")
    parser.set_defaults(max_episode_steps=2000)

    # settings for VR demos
    parser.set_defaults(alignment_pos_dist=0.15)
    parser.set_defaults(alignment_rot_dist_up=0.8)
    parser.set_defaults(alignment_rot_dist_forward=0.8)
    parser.set_defaults(alignment_project_dist=0.2)
    parser.set_defaults(control_type="ik")
    parser.set_defaults(move_speed=0.05)
    parser.add_argument(
        "--run_mode", type=str, default="manual", choices=["manual", "vr", "demo"]
    )
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterEnv(config)
    if config.run_mode == "manual":
        env.run_manual(config)
    elif config.run_mode == "vr":
        env.run_vr(config)
    elif config.run_mode == "demo":
        env.run_demo_actions(config)


if __name__ == "__main__":
    main()
