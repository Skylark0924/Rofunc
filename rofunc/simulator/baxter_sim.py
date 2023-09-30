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


class BaxterSim(RobotSim):
    def __init__(self, args):
        super().__init__(args)

    def setup_robot_dof_prop(self):
        from isaacgym import gymapi

        gym = self.gym
        envs = self.envs
        robot_asset = self.robot_asset
        robot_handles = self.robot_handles

        # configure robot dofs
        robot_dof_props = gym.get_asset_dof_properties(robot_asset)
        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)

        robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][:].fill(300.0)
        robot_dof_props["damping"][:].fill(30.0)

        # default dof states and position targets
        robot_num_dofs = gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        default_dof_pos[:] = robot_mids[:]

        default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # # send to torch
        # default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

        for env, robot_handle in zip(envs, robot_handles):
            # set dof properties
            gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

            # set initial dof states
            gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            gym.set_actor_dof_position_targets(env, robot_handle, default_dof_pos)
