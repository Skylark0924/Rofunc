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

from rofunc.simulator.base.base_sim import RobotSim
from rofunc.utils.logger.beauty_logger import beauty_print
from typing import List
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import numpy as np
import torch
from isaacgym import gymutil


class WalkerSim(RobotSim):
    def __init__(self, args, **kwargs):
        super().__init__(args, robot_name="walker", **kwargs)

    def setup_robot_dof_prop(self, gym=None, envs=None, robot_asset=None, robot_handles=None):
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

        for i in range(len(envs)):
            # set dof properties
            gym.set_actor_dof_properties(envs[i], robot_handles[i], robot_dof_props)

            # set initial dof states
            gym.set_actor_dof_states(envs[i], robot_handles[i], default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            gym.set_actor_dof_position_targets(envs[i], robot_handles[i], default_dof_pos)

    def show(self, visual_obs_flag=False, camera_props=None, attached_body=None, local_transform=None):
        """
        Visualize the CURI robot
        :param visual_obs_flag: if True, show visual observation
        :param camera_props: If visual_obs_flag is True, use this camera_props to config the camera
        :param attached_body: If visual_obs_flag is True, use this to refer the body the camera attached to
        :param local_transform: If visual_obs_flag is True, use this local transform to adjust the camera pose
        """
        from isaacgym import gymapi

        if visual_obs_flag:
            # Setup a first-person camera embedded in CURI's head
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
                local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(90.0)) * \
                                    gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90.0))

        super(WalkerSim, self).show(visual_obs_flag, camera_props, attached_body, local_transform)

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index):
        from isaacgym import gymutil

        for i in range(self.num_envs):
            # Update attractor target from current franka state
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], attractor_handles[i])
            pose = attractor_properties.target
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            pose.p.x = traj[index, 0]
            pose.p.y = traj[index, 2]
            pose.p.z = traj[index, 1]
            pose.r.w = traj[index, 6]
            pose.r.x = traj[index, 3]
            pose.r.y = traj[index, 4]
            pose.r.z = traj[index, 5]
            self.gym.set_attractor_target(self.envs[i], attractor_handles[i], pose)

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.robot_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def run_traj(self, traj, attracted_joints=None, update_freq=0.001):
        if attracted_joints is None:
            attracted_joints = ["left_palm_link", "right_palm_link"]
        self.run_traj_multi_joints_with_prior(traj, attracted_joints, update_freq)

    def run_traj_multi_joints_with_prior(self, traj: List, attracted_joints: List = None, update_freq=0.001):
        """
        Run the trajectory with multiple joints, the default is to run the trajectory with the left and right hand of
        bimanual robot.
        :param traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
        :param attracted_joints: [list], e.g. ["panda_left_hand", "panda_right_hand"]
        :param update_freq: the frequency of updating the robot pose
        :return:
        """
        assert isinstance(traj, list) and len(traj) > 0, "The trajectory should be a list of numpy arrays"

        beauty_print('Execute multi-joint trajectory with the Walker simulator')

        # Create the attractor
        attracted_joints, attractor_handles, axes_geoms, sphere_geoms = self._setup_attractors(traj, attracted_joints)

        # Time to wait in seconds before moving robot
        next_update_time = 0
        index = 0

        # initialize prior parameters
        root_pos_list = []
        root_rot_list = []
        root_vel_list = []
        root_ang_vel_list = []
        dof_pos_list = []
        dof_vel_list = []
        key_body_pos_list = []

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self.num_bodies = super().get_num_bodies()

        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        key_bodies = ["left_limb_l7", "right_limb_l7", "left_leg_l6", "right_leg_l6"]
        self._key_body_ids = self.build_key_body_ids_tensor(key_bodies)

        while index < len(traj[0]):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)
            if t >= next_update_time:
                self.gym.clear_lines(self.viewer)
                for i in range(len(attracted_joints)):
                    self.update_robot(traj[i], attractor_handles[i], axes_geoms[i], sphere_geoms[i], index)

                next_update_time += update_freq
                index += 1

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            # obtain the motion prior
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
            rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
            rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
            rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
            dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.robot_dof, 0]
            dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.robot_dof, 1]
            key_body_pos = rigid_body_pos[:, self._key_body_ids, :]

            # record the motion prior
            root_pos_list.append(rigid_body_pos[:, 0, :])
            root_rot_list.append(rigid_body_rot[:, 0, :])
            root_vel_list.append(rigid_body_vel[:, 0, :])
            root_ang_vel_list.append(rigid_body_ang_vel[:, 0, :])
            dof_pos_list.append(dof_pos)
            dof_vel_list.append(dof_vel)
            key_body_pos_list.append(key_body_pos)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        print("Done")

        prior_dict = {
            "root_pos": root_pos_list,
            "root_rot": root_rot_list,
            "root_vel": root_vel_list,
            "root_ang_vel": root_ang_vel_list,
            "dof_pose": dof_pos_list,
            "dof_vel": dof_vel_list,
            "key_pose": key_body_pos_list}

        print(prior_dict)

        prior_array = np.array(list(prior_dict.items()))
        output_path = "../data/motion_prior_walker/prior_array_09.npy"
        np.save(output_path, prior_array)

    # def generate_motion_prior(self, joint_state):
    #
    #     # initialize prior parameters
    #     root_pose = []
    #     root_rot = [0, 0, 0, 1]
    #     dof_pose = []
    #     root_vel = 0
    #     root_ang_vel = 0
    #     dof_vel = []
    #     key_pose = []
    #
    #     # record the motion prior
    #     joint_state = [joint_state[8:15], joint_state[21:28]]
    #     dof_pose = dof_pose.append(joint_state[0])
    #     dof_vel = dof_vel.append(joint_state[1])
    #
    #     prior_dict = {
    #                 "root_pose": root_pose,
    #                 "root_rot": root_rot,
    #                 "dof_pose": dof_pose,
    #                 "root_vel": root_vel,
    #                 "root_ang_vel": root_ang_vel,
    #                 "dof_vel": dof_vel,
    #                 "key_pose": key_pose}
    #
    #     return prior_dict

# if __name__ == '__main__':
#
#     traj_l = np.load('/home/zhuoli/Rofunc/examples/data/HOTO/mvnx/New Session-012/segment/14_LeftHand.npy')
#     traj_r = np.load('/home/zhuoli/Rofunc/examples/data/HOTO/mvnx/New Session-012/segment/10_RightHand.npy')
#
#     # setup environment args
#     args = gymutil.parse_arguments()
#     args.use_gpu_pipeline = False
#
#     # run the trajectory
#     Walkersim = WalkerSim(args, asset_root="/home/zhuoli/Rofunc/rofunc/simulator/assets", fix_base_link=True)
#     Walkersim.init()
#     # dof_info = Walkersim.get_dof_info()
#     # print("dof_info:", dof_info)
#     Walkersim.run_traj(traj=[traj_l, traj_r], update_freq=0.01)
