# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from rofunc.simulator.base_sim import RobotSim


class FrankaSim(RobotSim):
    def __init__(self, args):
        super().__init__(args)

    def setup_robot_dof_prop(
        self, gym=None, envs=None, robot_asset=None, robot_handles=None
    ):
        from isaacgym import gymapi

        gym = self.gym if gym is None else gym
        envs = self.envs if envs is None else envs
        robot_asset = self.robot_asset if robot_asset is None else robot_asset
        robot_handles = self.robot_handles if robot_handles is None else robot_handles

        # configure robot dofs
        robot_dof_props = gym.get_asset_dof_properties(robot_asset)
        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)

        # default dof states and position targets
        robot_num_dofs = gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        default_dof_pos = robot_mids

        default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # # send to torch
        # default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

        for env, robot in zip(envs, robot_handles):
            # set dof properties
            gym.set_actor_dof_properties(env, robot, robot_dof_props)

            # set initial dof states
            gym.set_actor_dof_states(env, robot, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            gym.set_actor_dof_position_targets(env, robot, default_dof_pos)

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index):
        # TODO: the traj is a little weird, need to be fixed
        from isaacgym import gymutil

        self.gym.clear_lines(self.viewer)
        for i in range(self.num_envs):
            # Update attractor target from current franka state
            attractor_properties = self.gym.get_attractor_properties(
                self.envs[i], attractor_handles[i]
            )
            pose = attractor_properties.target
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            pose.p.x = traj[index, 0] * 0.5
            pose.p.y = traj[index, 2] * 0.5
            pose.p.z = traj[index, 1] * 0.5
            pose.r.w = traj[index, 6]
            pose.r.x = traj[index, 3]
            pose.r.y = traj[index, 5]
            pose.r.z = traj[index, 4]
            self.gym.set_attractor_target(self.envs[i], attractor_handles[i], pose)

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def run_traj(self, traj, attracted_joint="panda_hand"):
        self.run_traj_multi_joints([traj], [attracted_joint])
