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
import torch

from rofunc.config.utils import get_sim_config
from rofunc.simulator.base_sim import RobotSim


class HumanoidSim(RobotSim):
    def __init__(self, args):
        super().__init__(args)

        self.num_bodies = self.get_num_bodies(self.robot_asset)
        # self.set_char_color(self.robot_handles, self.num_bodies, [0.54, 0.85, 0.2])
        self.humanoid_asset_infos = get_sim_config(sim_name="Humanoid_info")["Model_type"]
        self.humanoid_info = self._get_humanoid_info(self.robot_asset_file)
        self.parts = ["hands", "upper_body", "lower_body"]
        self.num_parts = len(self.parts)
        self.whole_rb_dict = self.humanoid_info["rigid_bodies"]
        self.wb_decompose_param_rb_ids = [
                [self.whole_rb_dict[rb_name] for rb_name in self.humanoid_info["parts"][part]["rigid_bodies"]]
            for part in self.parts]
        self.set_colors_for_parts(self.robot_handles, self.wb_decompose_param_rb_ids)

    def _get_humanoid_info(self, asset_file):
        return self.humanoid_asset_infos[asset_file.split("/")[-1].split(".")[0]]

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
        robot_mids = 0.5 * (robot_upper_limits + robot_lower_limits)

        robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][:].fill(0.0)
        robot_dof_props["damping"][:].fill(0.0)

        # default dof states and position targets
        robot_num_dofs = gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        default_dof_pos[:] = robot_mids[:]

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

    def add_head_embedded_camera(self, camera_props=None, attached_body=None, local_transform=None):
        from isaacgym import gymapi

        if camera_props is None:
            # Camera Sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 1280

        if attached_body is None:
            attached_body = "head_link2"

        if local_transform is None:
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(0.12, 0, 0.18)
        if self.PlaygroundSim.up_axis == "Y":
            local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(90.0)) * \
                                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0))
        elif self.PlaygroundSim.up_axis == "Z":
            local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(0.0))
        self.add_body_attached_camera(camera_props, attached_body, local_transform)

    def show(self, visual_obs_flag=False):
        """
        Visualize the CURI robot
        :param visual_obs_flag: if True, show visual observation
        :param camera_props: If visual_obs_flag is True, use this camera_props to config the camera
        :param attached_body: If visual_obs_flag is True, use this to refer the body the camera attached to
        :param local_transform: If visual_obs_flag is True, use this local transform to adjust the camera pose
        """
        if visual_obs_flag:
            # Setup a first-person camera embedded in CURI's head
            self.add_head_embedded_camera()
        super().show(visual_obs_flag)

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index, verbose=True):
        from isaacgym import gymutil

        for i in range(self.num_envs):
            # Update attractor target from current franka state
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], attractor_handles[i])
            pose = attractor_properties.target
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            pose.p.x = traj[index, 0]
            pose.p.y = traj[index, 1]
            pose.p.z = traj[index, 2]
            pose.r.w = traj[index, 6]
            pose.r.x = traj[index, 3]
            pose.r.y = traj[index, 4]
            pose.r.z = traj[index, 5]
            self.gym.set_attractor_target(self.envs[i], attractor_handles[i], pose)

            if verbose:
                # Draw axes and sphere at attractor location
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def run_traj(self, traj, attracted_rigid_bodies=None, update_freq=0.001, verbose=True, **kwargs):
        if attracted_rigid_bodies is None:
            attracted_rigid_bodies = ["left_hand", "right_hand"]
        self.run_traj_multi_rigid_bodies(traj, attracted_rigid_bodies, update_freq=update_freq, verbose=verbose,
                                         **kwargs)
