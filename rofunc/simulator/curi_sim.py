#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

from typing import List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image as Im

import rofunc as rf
from rofunc.simulator.base_sim import RobotSim
from rofunc.utils.logger.beauty_logger import beauty_print

import torch


def orientation_error(desired, current):
    from isaacgym.torch_utils import quat_conjugate, quat_mul

    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[0:3] * torch.sign(q_r[3]).unsqueeze(-1)


class CURISim(RobotSim):
    def __init__(self, args):
        super().__init__(args)

    def setup_robot_dof_prop(self):
        from isaacgym import gymapi

        gym = self.gym
        envs = self.envs
        robot_asset = self.robot_asset
        robot_handles = self.robot_handles

        robot_dof_info = self.get_dof_info()
        self.left_arm_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                     "panda_left_joint" in key]
        self.right_arm_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                      "panda_right_joint" in key]
        self.summit_wheel_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                         "wheel_joint" in key]
        self.torso_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if "torso" in key]
        self.left_gripper_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                         "panda_left_finger_joint" in key]
        self.right_gripper_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                          "panda_right_finger_joint" in key]
        self.left_softhand_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                          "left_qbhand" in key and "synergy" not in key and (
                                                  "knuckle" not in key or "thumb_knuckle" in key)]
        self.right_softhand_dof_indices = [value for key, value in robot_dof_info["dof_dict"].items() if
                                           "right_qbhand" in key and "synergy" not in key and (
                                                   "knuckle" not in key or "thumb_knuckle" in key)]

        if self.args.env.asset.assetFile in ["urdf/curi/urdf/curi_isaacgym_dual_arm.urdf",
                                             "urdf/curi/urdf/curi_isaacgym.urdf",
                                             "urdf/curi/urdf/curi_isaacgym_dual_arm_w_head.urdf"]:
            self.asset_arm_attracted_link = ["panda_left_hand", "panda_right_hand"]
            self.ee_type = "gripper"
        elif self.args.env.asset.assetFile in ["urdf/curi/urdf/curi_w_softhand_isaacgym.urdf"]:
            self.asset_arm_attracted_link = ["panda_left_link7", "panda_right_link7"]  # 24, 71
            self.ee_type = "softhand"
            self.synergy_action_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                   [2, 2, 2, 1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2]])
            self.useful_right_qbhand_dof_index = sorted(
                [value for key, value in robot_dof_info["dof_dict"].items() if
                 ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
                 ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key) and (
                         "right_qbhand" in key)])
            self.useful_left_qbhand_dof_index = sorted(
                [value for key, value in robot_dof_info["dof_dict"].items() if
                 ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
                 ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key) and (
                         "left_qbhand" in key)])

            self.virtual2real_dof_index_map_dict = {value: robot_dof_info["dof_dict"][key.replace("_virtual", "")] for
                                                    key, value in robot_dof_info["dof_dict"].items() if
                                                    "virtual" in key}

        # configure robot dofs
        robot_dof_props = gym.get_asset_dof_properties(robot_asset)
        self.robot_lower_limits = robot_lower_limits = robot_dof_props["lower"]
        self.robot_upper_limits = robot_upper_limits = robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)

        # use position drive for all dofs
        robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][:].fill(400.0)
        robot_dof_props["damping"][:].fill(40.0)

        # Wheels
        # robot_dof_props["driveMode"][self.summit_wheel_dof_indices] = gymapi.DOF_MODE_POS
        robot_dof_props["stiffness"][self.summit_wheel_dof_indices] = 400 * np.ones_like(self.summit_wheel_dof_indices)
        robot_dof_props["damping"][self.summit_wheel_dof_indices] = 40 * np.ones_like(self.summit_wheel_dof_indices)
        # Torso
        # robot_dof_props["driveMode"][self.torso_dof_indices].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][self.torso_dof_indices] = 10000 * np.ones_like(self.torso_dof_indices)
        robot_dof_props["damping"][self.torso_dof_indices] = 180 * np.ones_like(self.torso_dof_indices)
        # Arms
        if self.robot_controller == "ik":
            robot_dof_props["driveMode"][self.left_arm_dof_indices] = 1 * np.ones_like(self.left_arm_dof_indices)
            robot_dof_props["stiffness"][self.left_arm_dof_indices] = 1000 * np.ones_like(self.left_arm_dof_indices)
            robot_dof_props["damping"][self.left_arm_dof_indices] = 100 * np.ones_like(self.left_arm_dof_indices)
            robot_dof_props["driveMode"][self.right_arm_dof_indices] = 1 * np.ones_like(self.right_arm_dof_indices)
            robot_dof_props["stiffness"][self.right_arm_dof_indices] = 1000 * np.ones_like(self.right_arm_dof_indices)
            robot_dof_props["damping"][self.right_arm_dof_indices] = 100 * np.ones_like(self.right_arm_dof_indices)
        else:  # osc
            robot_dof_props["driveMode"][self.left_arm_dof_indices] = 3 * np.ones_like(self.left_arm_dof_indices)
            robot_dof_props["stiffness"][self.left_arm_dof_indices] = 0 * np.ones_like(self.left_arm_dof_indices)
            robot_dof_props["damping"][self.left_arm_dof_indices] = 0 * np.ones_like(self.left_arm_dof_indices)
            robot_dof_props["driveMode"][self.right_arm_dof_indices] = 3 * np.ones_like(self.right_arm_dof_indices)
            robot_dof_props["stiffness"][self.right_arm_dof_indices] = 0 * np.ones_like(self.right_arm_dof_indices)
            robot_dof_props["damping"][self.right_arm_dof_indices] = 0 * np.ones_like(self.right_arm_dof_indices)
        # grippers
        robot_dof_props["driveMode"][self.left_gripper_dof_indices] = 1 * np.ones_like(self.left_gripper_dof_indices)
        robot_dof_props["stiffness"][self.left_gripper_dof_indices] = 1000 * np.ones_like(self.left_gripper_dof_indices)
        robot_dof_props["damping"][self.left_gripper_dof_indices] = 40 * np.ones_like(self.left_gripper_dof_indices)
        robot_dof_props["driveMode"][self.right_gripper_dof_indices] = 1 * np.ones_like(self.right_gripper_dof_indices)
        robot_dof_props["stiffness"][self.right_gripper_dof_indices] = 1000 * np.ones_like(
            self.right_gripper_dof_indices)
        robot_dof_props["damping"][self.right_gripper_dof_indices] = 40 * np.ones_like(self.right_gripper_dof_indices)
        # softhands
        robot_dof_props["driveMode"][self.left_softhand_dof_indices] = 1 * np.ones_like(self.left_softhand_dof_indices)
        robot_dof_props["stiffness"][self.left_softhand_dof_indices] = 10 * np.ones_like(self.left_softhand_dof_indices)
        robot_dof_props["damping"][self.left_softhand_dof_indices] = 40 * np.ones_like(self.left_softhand_dof_indices)
        robot_dof_props["driveMode"][self.right_softhand_dof_indices] = 1 * np.ones_like(
            self.right_softhand_dof_indices)
        robot_dof_props["stiffness"][self.right_softhand_dof_indices] = 10 * np.ones_like(
            self.right_softhand_dof_indices)
        robot_dof_props["damping"][self.right_softhand_dof_indices] = 40 * np.ones_like(self.right_softhand_dof_indices)

        # default dof states and position targets
        robot_num_dofs = gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        default_dof_pos = robot_mids
        # grippers open
        default_dof_pos[self.left_gripper_dof_indices] = robot_upper_limits[self.left_gripper_dof_indices]
        default_dof_pos[self.right_gripper_dof_indices] = robot_upper_limits[self.right_gripper_dof_indices]
        # softhands open
        default_dof_pos[self.left_softhand_dof_indices] = robot_lower_limits[self.left_softhand_dof_indices]
        default_dof_pos[self.right_softhand_dof_indices] = robot_lower_limits[self.right_softhand_dof_indices]
        self.default_dof_pos = default_dof_pos

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
        super(CURISim, self).show(visual_obs_flag)

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

    def control_ik(self, dpose):
        damping = 0.1
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = (torch.eye(6) * (damping ** 2))
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def control_osc(self, dpose, hand_vel, massmatrix, dof_indices):
        kp = 1500.
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.
        kd_null = 2.0 * np.sqrt(kp_null)
        # default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
        mm_inv = torch.inverse(massmatrix)
        m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (
                kp * dpose - kd * hand_vel.unsqueeze(-1))

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self.j_eef @ mm_inv
        u_null = kd_null * -self.dof_vel + kp_null * (
                (self.default_dof_pos_tensor.view(1, -1, 1) - self.dof_pos + np.pi) % (2 * np.pi) - np.pi)
        u_null = u_null[:, dof_indices]
        u_null = massmatrix @ u_null
        u += (torch.eye(7).unsqueeze(0) -
              torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null
        return u.squeeze(-1)

    def run_traj(self, traj, attracted_rigid_bodies=None, update_freq=0.001, verbose=True, **kwargs):
        if attracted_rigid_bodies is None:
            attracted_rigid_bodies = self.asset_arm_attracted_link
        self.run_traj_multi_rigid_bodies(traj, attracted_rigid_bodies, update_freq=update_freq, verbose=verbose,
                                         **kwargs)

    def run_traj_multi_rigid_bodies_with_interference(self, traj: List, intf_index: List, intf_mode: str,
                                                      intf_forces=None, intf_torques=None, intf_joints: List = None,
                                                      intf_efforts: np.ndarray = None,
                                                      attracted_rigid_bodies: List = None,
                                                      update_freq=0.001, save_name=None):
        """
        Run the trajectory with multiple rigid bodies with interference, the default is to run the trajectory with the left and
        right hand of the CURI robot.
        Args:
            traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
            intf_index: a list of the timing indices of the interference occurs
            intf_mode: the mode of the interference, ["actor_dof_efforts", "body_forces", "body_force_at_pos"]
            intf_forces: a tensor of shape (num_envs, num_bodies, 3), the interference forces applied to the bodies
            intf_torques: a tensor of shape (num_envs, num_bodies, 3), the interference torques applied to the bodies
            intf_joints: [list], e.g. ["panda_left_hand"]
            intf_efforts: array containing the efforts for all degrees of freedom of the actor.
            attracted_rigid_bodies: [list], e.g. ["panda_left_hand", "panda_right_hand"]
            update_freq: the frequency of updating the robot pose
        """
        from isaacgym import gymapi
        from isaacgym import gymtorch
        import torch

        assert isinstance(traj, list) and len(traj) > 0, "The trajectory should be a list of numpy arrays"
        assert intf_mode in ["actor_dof_efforts", "body_forces", "body_force_at_pos"], \
            "The interference mode should be one of ['actor_dof_efforts', 'body_forces', 'body_force_at_pos']"

        if attracted_rigid_bodies is None:
            attracted_rigid_bodies = self.asset_arm_attracted_link

        beauty_print('Execute multi rigid bodies trajectory with interference with the CURI simulator')

        device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'
        num_bodies = self.get_num_bodies()
        if intf_forces is not None:
            assert intf_forces.shape == torch.Size(
                [self.num_envs, num_bodies, 3]), "The shape of forces should be (num_envs, num_bodies, 3)"
            intf_forces = intf_forces.to(device)
        if intf_torques is not None:
            assert intf_torques.shape == torch.Size(
                [self.num_envs, num_bodies, 3]), "The shape of torques should be (num_envs, num_bodies, 3)"
            intf_torques = intf_torques.to(device)

        # Create the attractor
        attracted_rigid_bodies, attractor_handles, axes_geoms, sphere_geoms = self._setup_attractors(traj,
                                                                                                     attracted_rigid_bodies)

        # Time to wait in seconds before moving robot
        next_curi_update_time = 1
        index = 0
        states = []
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)
            if t >= next_curi_update_time:
                self.gym.clear_lines(self.viewer)
                for i in range(len(attracted_rigid_bodies)):
                    self.update_robot(traj[i], attractor_handles[i], axes_geoms[i], sphere_geoms[i], index)
                next_curi_update_time += update_freq
                index += 1
                if index >= len(traj[0]):
                    index = 0

                # Create the interference
                if index in intf_index:
                    if intf_mode == "actor_dof_efforts":
                        # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(intf_efforts))
                        for i in range(len(self.envs)):
                            self.gym.apply_actor_dof_efforts(self.envs[i], self.robot_handles[i], intf_efforts)
                    elif intf_mode == "body_forces":
                        # set intf_forces and intf_torques for the specific bodies
                        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(intf_forces),
                                                                gymtorch.unwrap_tensor(intf_torques), gymapi.ENV_SPACE)

                # Get current robot state
                state = self.get_robot_state(mode='dof_state')
                states.append(np.array(state))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        print("Done")

        with open('{}.npy'.format(save_name), 'wb') as f:
            np.save(f, np.array(states))
        beauty_print('{}.npy saved'.format(save_name), type="info")

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def run_hand_reach_target_pose(self, target_pose, attracted_hand=None, update_freq=0.001, verbose=True):
        from isaacgym import gymapi, gymtorch, gymutil
        import math

        # Create helper geometry used for visualization
        # Create a wireframe axis
        axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

        curi_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        curi_hand_index = curi_link_dict[attracted_hand[0]]

        self.gym.prepare_sim(self.sim)

        # get jacobian tensor
        # for fixed-base curi, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "CURI")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_pos = dof_states[:, 0].view(self.num_envs, 18, 1)

        # jacobian entries corresponding to curi hand
        self.j_eef = jacobian[:, curi_hand_index - 1, :, ]

        pos_action = torch.zeros_like(dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        controller = "ik"

        step = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.clear_lines(self.viewer)

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

            pose = gymapi.Transform()
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            pose.p.x = target_pose[0][step, 0]
            pose.p.y = target_pose[0][step, 1]
            pose.p.z = target_pose[0][step, 2]
            pose.r.w = target_pose[0][step, 6]
            pose.r.x = target_pose[0][step, 3]
            pose.r.y = target_pose[0][step, 4]
            pose.r.z = target_pose[0][step, 5]
            if verbose:
                # Draw axes and sphere at attractor location
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], pose)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], pose)

            hand_pos = rb_states[curi_hand_index, :3]
            hand_rot = rb_states[curi_hand_index, 3:7]
            hand_vel = rb_states[curi_hand_index, 7:]

            # compute goal position and orientation
            goal_pos = torch.tensor(target_pose[0][step, :3], dtype=torch.float32)
            goal_rot = torch.tensor(target_pose[0][step, 3:], dtype=torch.float32)

            # compute position and orientation error
            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(goal_rot, hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            if dpose.norm() < 0.01:
                step += 1
                if step >= len(target_pose[0]):
                    step = 0

            rf.logger.beauty_print("pos_err: {}".format(pos_err), type="info")
            rf.logger.beauty_print("orn_err: {}".format(orn_err), type="info")

            # Deploy control based on type
            if controller == "ik":
                pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
            else:  # osc
                effort_action[:, :7] = self.control_osc(dpose)

            # Deploy actions
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))

            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def run_with_text_commands(self, verbose=True):
        from isaacgym import gymapi, gymtorch, gymutil
        from isaacgym.torch_utils import to_torch
        from scipy.spatial.transform import Rotation as R

        self.add_tracking_target_sphere_axes()
        self.add_head_embedded_camera()
        fig = plt.figure("Visual observation", figsize=(8, 8))
        # use rcParams to control the plot window not on the top
        if matplotlib.rcParams['figure.raise_window']:
            matplotlib.rcParams['figure.raise_window'] = False

        self.gym.prepare_sim(self.sim)
        self.monitor_rigid_body_states()
        self.monitor_dof_states()
        self.monitor_robot_jacobian()
        self.monitor_robot_mass_matrix()

        self.robot_dof_info = self.get_dof_info()
        curi_link_dict = self.get_actor_rigid_body_info(self.robot_handles[0])
        beauty_print("curi_link_dict: {}".format(curi_link_dict), type="info")
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, self.robot_dof_info["dof_count"], 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, self.robot_dof_info["dof_count"], 1)

        if self.args.env.object_asset is not None:
            object_names = self.args.env.object_asset.object_names
            assert len(object_names) == len(
                self.object_handles), "The number of object names should be the same as the number of object handles"
            for j in range(len(self.object_handles)):
                object_dict = self.get_actor_rigid_body_info(self.object_handles[j][0])
                beauty_print("object_name: {}, object_dict: {}".format(object_names[j], object_dict), type="info")

        # Define keyboard and mouse event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "KEY_1")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "KEY_2")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "KEY_3")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_4, "KEY_4")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_5, "KEY_5")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_6, "KEY_6")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_7, "KEY_7")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_8, "KEY_8")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_9, "KEY_9")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_0, "KEY_0")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "KEY_MINUS")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_EQUAL, "KEY_EQUAL")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "grasp")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "open_camera")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "KEY_D")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "KEY_A")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "KEY_W")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "KEY_S")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "KEY_UP")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "KEY_DOWN")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "KEY_LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "KEY_RIGHT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_PAGE_UP, "KEY_PAGE_UP")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_PAGE_DOWN, "KEY_PAGE_DOWN")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "keep_arm_dof")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "homing_arm_dof")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "query_rigid_body_poses")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "query_object_pose")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "attach_object")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ENTER, "confirm_target")
        beauty_print("====Keyboard and mouse event===\n"
                     "-----Arms-----\n"
                     "   Space: control dual arms of CURI\n"
                     "   K: keep arm dof\n"
                     "   H: homing arm dof\n"
                     "-----End-effectors-----\n"
                     "   G: Grasp or un-grasp\n"
                     "-----Head-----\n"
                     "   D: head turn right\n"
                     "   A: head turn left\n"
                     "   W: head turn up\n"
                     "   S: head turn down\n"
                     "-----Vision-----\n"
                     "   C: open or close head-embedded camera\n"
                     "-----Mobile base-----\n"
                     "   UP: mobile base forward\n"
                     "   DOWN: mobile base backward\n"
                     "   LEFT: mobile base left\n"
                     "   RIGHT: mobile base right\n"
                     "-----Infos-----\n"
                     "   B: query key rigid body poses of the robot\n"
                     "   O: query the object pose\n"
                     "-----Attach object-----\n"
                     "   M: attach object\n"
                     "-----Others-----\n"
                     "   R: reset\n"
                     "   Q: quit\n", type="info")

        self.default_dof_pos_tensor = to_torch(self.default_dof_pos, device="cpu")
        pos_action = self.default_dof_pos_tensor.reshape(self.dof_pos.shape).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        attracted_link_index = None
        visual_obs_flag = False
        homing_flag = False
        keep_arm_dof_flag = False
        left_grasp_flag = False
        right_grasp_flag = False
        left_synergy_action = [1.0, 0.0]
        right_synergy_action = [1.0, 0.0]
        attached_objects = {}
        attracted_link_index = None  # 当前控制的 link
        target_pose = None  # 目标位姿
        self.attached_objects = {}

        def compute_relative_pose(object_pose, link_pose):
            """ 修复后的相对位姿计算 """
            # 提取位置和四元数 (IsaacGym xyzw 格式)
            link_pos, link_quat = link_pose[:3], link_pose[3:]
            obj_pos, obj_quat = object_pose[:3], object_pose[3:]

            # 转换为 Rotation 对象（保持xyzw顺序）
            link_rot = R.from_quat(link_quat)  # 直接传入 [x,y,z,w]
            obj_rot = R.from_quat(obj_quat)

            # 计算相对旋转
            relative_rot = link_rot.inv() * obj_rot

            # 计算相对位置（转换到 link 坐标系）
            relative_pos = link_rot.inv().apply(obj_pos - link_pos)

            return np.concatenate([relative_pos, relative_rot.as_quat()])

        def apply_relative_pose(link_pose, relative_pose):
            """ 修复后的绝对位姿计算 """
            # 提取数据
            link_pos, link_quat = link_pose[:3], link_pose[3:]
            rel_pos, rel_quat = relative_pose[:3], relative_pose[3:]

            # 转换为 Rotation 对象
            link_rot = R.from_quat(link_quat)
            rel_rot = R.from_quat(rel_quat)

            # 计算绝对位姿
            new_pos = link_pos + link_rot.apply(rel_pos)
            new_rot = link_rot * rel_rot

            return np.concatenate([new_pos, new_rot.as_quat()])

        def update_object_pose(object_index, new_pose):
            """ 更新仿真中的物体位姿 """
            state = self.gym.get_actor_rigid_body_states(self.envs[0], self.object_handles[object_index][0],
                                                         gymapi.STATE_ALL)
            state['pose']['p'].fill(tuple(new_pose[:3]))  # 更新位置
            state['pose']['r'].fill(tuple(new_pose[3:]))  # 更新旋转
            state['vel']['linear'].fill((0, 0, 0))  # 清零线速度
            state['vel']['angular'].fill((0, 0, 0))  # 清零角速度
            self.gym.set_actor_rigid_body_states(self.envs[0], self.object_handles[object_index][0], state,
                                                 gymapi.STATE_ALL)

        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.clear_lines(self.viewer)

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.value == 0:  # 按键松开时不处理
                    continue
                # **按 SPACE 选择目标 link**
                if evt.action == "space_shoot":
                    try:
                        attracted_link_index = int(input("Input attracted index:\n"))
                    except:
                        continue
                    current_pose = self.rb_states[attracted_link_index].clone().detach().cpu().numpy()
                    x, y, z = current_pose[:3]  # 位置
                    qx, qy, qz, qw = current_pose[3:7]  # 旋转（四元数）

                    # **转换四元数为欧拉角**
                    euler_angles = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)
                    rx, ry, rz = euler_angles  # roll, pitch, yaw

                    # **初始化目标 pose**
                    target_pose = torch.tensor([x, y, z, qx, qy, qz, qw])
                    beauty_print(f"Controlling link {attracted_link_index} (Starting Pose: {x, y, z, qx, qy, qz, qw}).",
                                 type="info")
                    beauty_print("Use 1-6 to rotate, 7-+ to move. Press Enter to confirm.", type="info")
                # **如果已经选择了 link，监听 1-+ 进行调整**
                if attracted_link_index is not None:
                    if evt.action == "KEY_1":
                        rx += 5  # 绕 X 轴 +5°
                    elif evt.action == "KEY_2":
                        rx -= 5  # 绕 X 轴 -5°
                    elif evt.action == "KEY_3":
                        ry += 5  # 绕 Y 轴 +5°
                    elif evt.action == "KEY_4":
                        ry -= 5  # 绕 Y 轴 -5°
                    elif evt.action == "KEY_5":
                        rz += 5  # 绕 Z 轴 +5°
                    elif evt.action == "KEY_6":
                        rz -= 5  # 绕 Z 轴 -5°

                    # **位置控制**
                    elif evt.action == "KEY_7":
                        x += 0.01  # X 轴 +
                    elif evt.action == "KEY_8":
                        x -= 0.01  # X 轴 -
                    elif evt.action == "KEY_9":
                        y += 0.01  # Y 轴 +
                    elif evt.action == "KEY_0":
                        y -= 0.01  # Y 轴 -
                    elif evt.action == "KEY_MINUS":
                        z += 0.01  # Z 轴 +
                    elif evt.action == "KEY_EQUAL":  # `+` 键在 `KEY_EQUAL`
                        z -= 0.01  # Z 轴 -

                    quat = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_quat()
                    qx, qy, qz, qw = quat
                    target_pose = torch.tensor([x, y, z, qx, qy, qz, qw])
                    beauty_print(f"Target Pose: {x, y, z, rx, ry, rz}", type="info")

                # if evt.action == "space_shoot" and evt.value > 0:
                #     attracted_link_index = input("Input attracted index:\n")
                #     target_pose = input("Input target pose:\n x, y, z, qx, qy, qz, qw\n")
                #     try:
                #         attracted_link_index = int(attracted_link_index)
                #         target_pose = [([float(i) for i in target_pose.split(", ")])]
                #     except ValueError:
                #         beauty_print("Invalid input!", type="error")
                if evt.action == "grasp" and evt.value > 0:
                    ee_link = input("Left/Right/Both End-effector: [L/R/B]\n")

                    if self.ee_type == "gripper":
                        left_gripper_dof_index1 = self.robot_dof_info["dof_dict"]["panda_left_finger_joint1"]
                        left_gripper_dof_index2 = self.robot_dof_info["dof_dict"]["panda_left_finger_joint2"]
                        right_gripper_dof_index1 = self.robot_dof_info["dof_dict"]["panda_right_finger_joint1"]
                        right_gripper_dof_index2 = self.robot_dof_info["dof_dict"]["panda_right_finger_joint2"]

                        def execute_gripper_grasp(grasp_flag, gripper_dof_index1, gripper_dof_index2, pos_action):
                            if not grasp_flag:
                                pos_action[:, gripper_dof_index1] = (
                                        self.dof_pos.squeeze(-1)[:, gripper_dof_index1]
                                        - torch.tensor([0.03]))
                                pos_action[:, gripper_dof_index2] = (
                                        self.dof_pos.squeeze(-1)[:, gripper_dof_index2]
                                        - torch.tensor([0.03]))
                            else:
                                pos_action[:, gripper_dof_index1] = (
                                        self.dof_pos.squeeze(-1)[:, gripper_dof_index1]
                                        + torch.tensor([0.03]))
                                pos_action[:, gripper_dof_index2] = (
                                        self.dof_pos.squeeze(-1)[:, gripper_dof_index2]
                                        + torch.tensor([0.03]))
                            grasp_flag = not grasp_flag
                            return grasp_flag, pos_action

                        if ee_link.upper() == "L":
                            left_grasp_flag, pos_action = execute_gripper_grasp(left_grasp_flag,
                                                                                left_gripper_dof_index1,
                                                                                left_gripper_dof_index2,
                                                                                pos_action)
                        elif ee_link.upper() == "R":
                            right_grasp_flag, pos_action = execute_gripper_grasp(right_grasp_flag,
                                                                                 right_gripper_dof_index1,
                                                                                 right_gripper_dof_index2,
                                                                                 pos_action)
                        elif ee_link.upper() == "B":
                            left_grasp_flag, pos_action = execute_gripper_grasp(left_grasp_flag,
                                                                                left_gripper_dof_index1,
                                                                                left_gripper_dof_index2,
                                                                                pos_action)
                            right_grasp_flag, pos_action = execute_gripper_grasp(right_grasp_flag,
                                                                                 right_gripper_dof_index1,
                                                                                 right_gripper_dof_index2,
                                                                                 pos_action)
                    elif self.ee_type == "softhand":
                        def execute_softhand_grasp(grasp_flag, qbhand_dof_index, pos_action):
                            synergy_action = input("Input synergy:\n")
                            synergy_action = [float(i) for i in synergy_action.split(" ")]
                            dof_action = self._get_dof_action_from_synergy(synergy_action,
                                                                           qbhand_dof_index)
                            for i, index in enumerate(qbhand_dof_index):
                                pos_action[:, index] = torch.tensor(dof_action[i])
                            grasp_flag = not grasp_flag
                            return grasp_flag, pos_action,

                        if ee_link.upper() == "L":
                            left_grasp_flag, pos_action = execute_softhand_grasp(left_grasp_flag,
                                                                                 self.useful_left_qbhand_dof_index,
                                                                                 pos_action)
                        elif ee_link.upper() == "R":
                            right_grasp_flag, pos_action = execute_softhand_grasp(right_grasp_flag,
                                                                                  self.useful_right_qbhand_dof_index,
                                                                                  pos_action)
                        if ee_link.upper() == "B":
                            left_grasp_flag, pos_action = execute_softhand_grasp(left_grasp_flag,
                                                                                 self.useful_left_qbhand_dof_index,
                                                                                 pos_action)
                            right_grasp_flag, pos_action = execute_softhand_grasp(right_grasp_flag,
                                                                                  self.useful_right_qbhand_dof_index,
                                                                                  pos_action)

                        for key, value in self.virtual2real_dof_index_map_dict.items():
                            pos_action[:, key] = pos_action[:, value]
                if evt.action == "open_camera" and evt.value > 0:
                    visual_obs_flag = not visual_obs_flag
                    beauty_print("Open camera" if visual_obs_flag else "Close camera", type="info")
                if evt.action == "KEY_D" and evt.value > 0:
                    head_righ_left_dof_index = self.robot_dof_info["dof_dict"]["head_actuated_joint1"]
                    pos_action[:, head_righ_left_dof_index] = (self.dof_pos.squeeze(-1)[:, head_righ_left_dof_index]
                                                               - torch.tensor([0.1]))
                if evt.action == "KEY_A" and evt.value > 0:
                    head_righ_left_dof_index = self.robot_dof_info["dof_dict"]["head_actuated_joint1"]
                    pos_action[:, head_righ_left_dof_index] = (self.dof_pos.squeeze(-1)[:, head_righ_left_dof_index]
                                                               + torch.tensor([0.1]))
                if evt.action == "KEY_W" and evt.value > 0:
                    head_up_down_dof_index = self.robot_dof_info["dof_dict"]["head_actuated_joint2"]
                    pos_action[:, head_up_down_dof_index] = (self.dof_pos.squeeze(-1)[:, head_up_down_dof_index]
                                                             - torch.tensor([0.1]))
                if evt.action == "KEY_S" and evt.value > 0:
                    head_up_down_dof_index = self.robot_dof_info["dof_dict"]["head_actuated_joint2"]
                    pos_action[:, head_up_down_dof_index] = (self.dof_pos.squeeze(-1)[:, head_up_down_dof_index]
                                                             + torch.tensor([0.1]))
                if evt.action == "KEY_UP" and evt.value > 0:
                    pos_action[:, self.summit_wheel_dof_indices] = (
                            self.dof_pos.squeeze(-1)[:, self.summit_wheel_dof_indices]
                            + torch.tensor([0.5, 0.5, 0.5, 0.5]))
                if evt.action == "KEY_DOWN" and evt.value > 0:
                    pos_action[:, self.summit_wheel_dof_indices] = (
                            self.dof_pos.squeeze(-1)[:, self.summit_wheel_dof_indices]
                            - torch.tensor([0.5, 0.5, 0.5, 0.5]))
                if evt.action == "KEY_LEFT" and evt.value > 0:
                    pos_action[:, self.summit_wheel_dof_indices] = (
                            self.dof_pos.squeeze(-1)[:, self.summit_wheel_dof_indices]
                            + torch.tensor([0.3, -0.3, -0.3, 0.3]))
                if evt.action == "KEY_RIGHT" and evt.value > 0:
                    pos_action[:, self.summit_wheel_dof_indices] = (
                            self.dof_pos.squeeze(-1)[:, self.summit_wheel_dof_indices]
                            - torch.tensor([0.3, -0.3, -0.3, 0.3]))
                if evt.action == "query_rigid_body_poses" and evt.value > 0:
                    beauty_print("Left hand pose: {}".format(
                        self.rb_states[curi_link_dict[self.asset_arm_attracted_link[0]]][:7]), type="info")
                    beauty_print("Right hand pose: {}".format(
                        self.rb_states[curi_link_dict[self.asset_arm_attracted_link[1]]][:7]), type="info")
                    beauty_print("Robot base pose: {}".format(self.rb_states[0][:7]), type="info")
                if evt.action == "query_object_pose" and evt.value > 0:
                    for j in range(len(self.object_handles)):
                        beauty_print(
                            "object_name: {}, object pose: {}".format(self.args.env.object_asset.object_names[j],
                                                                      self.rb_states[self.object_idxs[j][0]][:7]),
                            type="info")
                if evt.action == "attach_object" and evt.value > 0:
                    object_names = self.args.env.object_asset.object_names

                    print("\nAvailable objects:")
                    for idx, name in enumerate(object_names):
                        print(f"  [{idx}] {name}")
                    object_index = input("Select object index to attach (0-{}):\n".format(len(self.object_handles) - 1))
                    attach_link_index = input("Select robot link index to attach to:\n")

                    try:
                        object_index = int(object_index)
                        attach_link_index = int(attach_link_index)
                        if object_index < 0 or object_index >= len(self.object_handles):
                            raise ValueError("Invalid object index!")

                        # **如果已经 attach，则解除 attach**
                        if object_index in attached_objects:
                            del attached_objects[object_index]
                            beauty_print(f"Detached object {object_index} from link {attach_link_index}", type="info")
                        else:
                            # **计算物体相对 link 的位姿**
                            object_pose = self.rb_states[self.object_idxs[object_index][0]][:7]
                            link_pose = self.rb_states[attach_link_index][:7]
                            relative_pose = compute_relative_pose(object_pose, link_pose)

                            # **存储 attach 信息**
                            attached_objects[object_index] = (attach_link_index, relative_pose)
                            beauty_print(f"Attached object {object_index} to link {attach_link_index}", type="info")

                    except ValueError as e:
                        beauty_print(f"Error: {e}", type="error")
                if evt.action == "reset" and evt.value > 0:
                    pos_action = torch.tensor(self.default_dof_pos).reshape(self.dof_pos.shape).squeeze(-1)
                    effort_action = torch.zeros_like(pos_action)
                    attracted_link_index = None
                    visual_obs_flag = False
                    left_grasp_flag = False
                    right_grasp_flag = False
                    left_synergy_action = [1.0, 0.0]
                    right_synergy_action = [1.0, 0.0]
                    for j in range(len(self.object_handles)):
                        state = self.gym.get_actor_rigid_body_states(self.envs[0], self.object_handles[j][0],
                                                                     gymapi.STATE_ALL)
                        init_pose = self.args.env.object_asset.init_poses[j]
                        state['pose']['p'].fill((init_pose[0], init_pose[1], init_pose[2]))
                        state['pose']['r'].fill((init_pose[3], init_pose[4], init_pose[5], init_pose[6]))
                        state['vel']['linear'].fill((0, 0, 0))
                        state['vel']['angular'].fill((0, 0, 0))
                        self.gym.set_actor_rigid_body_states(self.envs[0], self.object_handles[j][0], state,
                                                             gymapi.STATE_ALL)
                    beauty_print("Reset", type="info")
                if evt.action == "quit" and evt.value > 0:
                    break
                if evt.action == "keep_arm_dof" and evt.value > 0:
                    keep_arm_dof_flag = not keep_arm_dof_flag
                    keep_dof_pos = self.dof_pos.squeeze(-1).clone()
                    beauty_print("Keep arm dof" if keep_arm_dof_flag else "Release arm dof", type="info")
                if evt.action == "homing_arm_dof" and evt.value > 0:
                    homing_flag = not homing_flag
                    beauty_print("Homing arm dof" if homing_flag else "Release arm dof", type="info")

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if visual_obs_flag:
                # digest image
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                cam_img = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handle,
                                                    gymapi.IMAGE_COLOR).reshape(1280, 1280, 4)
                cam_img = Im.fromarray(cam_img)
                plt.imshow(cam_img)
                plt.axis('off')
                plt.pause(1e-9)
                fig.clf()

                self.gym.end_access_image_tensors(self.sim)

            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

            if attracted_link_index is not None:
                # jacobian entries corresponding to curi hand
                if curi_link_dict[self.asset_arm_attracted_link[0]] == attracted_link_index:
                    if self.args.env.asset.fix_base_link:
                        self.j_eef = self.jacobian[:, attracted_link_index - 1, :, self.left_arm_dof_indices]
                    else:
                        self.j_eef = self.jacobian[:, attracted_link_index, :,
                                     [i + 6 for i in self.left_arm_dof_indices]]
                elif curi_link_dict[self.asset_arm_attracted_link[1]] == attracted_link_index:
                    if self.args.env.asset.fix_base_link:
                        self.j_eef = self.jacobian[:, attracted_link_index - 1, :, self.right_arm_dof_indices]
                    else:
                        self.j_eef = self.jacobian[:, attracted_link_index, :,
                                     [i + 6 for i in self.right_arm_dof_indices]]
                else:
                    beauty_print("Only support left and right hand now!", type="error")
                    attracted_link_index = None
                    continue

                pose = gymapi.Transform()
                # pose.p: (x, y, z), pose.r: (w, x, y, z)
                pose.p.x = target_pose[0]
                pose.p.y = target_pose[1]
                pose.p.z = target_pose[2]
                pose.r.x = target_pose[3]
                pose.r.y = target_pose[4]
                pose.r.z = target_pose[5]
                pose.r.w = target_pose[6]
                if verbose:
                    # Draw axes and sphere at attractor location
                    gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[0], pose)
                    gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[0], pose)

                if self.args.env.asset.fix_base_link:
                    hand_pos = self.rb_states[attracted_link_index, :3]
                    hand_rot = self.rb_states[attracted_link_index, 3:7]
                    hand_vel = self.rb_states[attracted_link_index, 7:]
                else:
                    hand_pos = self.rb_states[attracted_link_index + 1, :3]
                    hand_rot = self.rb_states[attracted_link_index + 1, 3:7]
                    hand_vel = self.rb_states[attracted_link_index + 1, 7:]

                # compute goal position and orientation
                goal_pos = torch.tensor(target_pose[:3], dtype=torch.float32)
                goal_rot = torch.tensor(target_pose[3:], dtype=torch.float32)

                # compute position and orientation error
                pos_err = goal_pos - hand_pos
                orn_err = orientation_error(goal_rot, hand_rot)
                dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
                # if dpose.norm() < 1:
                #     attracted_link_index = None
                #     continue

                # rf.logger.beauty_print("pos_err: {}".format(pos_err), type="info")
                # rf.logger.beauty_print("orn_err: {}".format(orn_err), type="info")

                # Deploy control based on type
                if self.robot_controller == "ik":
                    if curi_link_dict[self.asset_arm_attracted_link[0]] == attracted_link_index:
                        pos_action[:, self.left_arm_dof_indices] = self.dof_pos.squeeze(-1)[:,
                                                                   self.left_arm_dof_indices] + self.control_ik(dpose)
                    elif curi_link_dict[self.asset_arm_attracted_link[1]] == attracted_link_index:
                        pos_action[:, self.right_arm_dof_indices] = self.dof_pos.squeeze(-1)[:,
                                                                    self.right_arm_dof_indices] + self.control_ik(dpose)
                else:  # osc
                    if curi_link_dict[self.asset_arm_attracted_link[0]] == attracted_link_index:
                        massmatrix = self.massmatrix[:, self.left_arm_dof_indices][:, :, self.left_arm_dof_indices]
                        effort_action[:, self.left_arm_dof_indices] = self.control_osc(dpose, hand_vel, massmatrix,
                                                                                       self.left_arm_dof_indices)
                    elif curi_link_dict[self.asset_arm_attracted_link[1]] == attracted_link_index:
                        massmatrix = self.massmatrix[:, self.right_arm_dof_indices][:, :, self.right_arm_dof_indices]
                        effort_action[:, self.right_arm_dof_indices] = self.control_osc(dpose, hand_vel, massmatrix,
                                                                                        self.right_arm_dof_indices)

                if homing_flag:
                    default_dof_pos_tensor = self.default_dof_pos_tensor.reshape(self.dof_pos.shape).squeeze(-1)
                    pos_action[:, self.left_arm_dof_indices] = default_dof_pos_tensor[:, self.left_arm_dof_indices]
                    pos_action[:, self.right_arm_dof_indices] = default_dof_pos_tensor[:, self.right_arm_dof_indices]
                elif keep_arm_dof_flag:
                    pos_action[:, self.left_arm_dof_indices] = keep_dof_pos[:, self.left_arm_dof_indices]
                    pos_action[:, self.right_arm_dof_indices] = keep_dof_pos[:, self.right_arm_dof_indices]

            # **更新附着物体的 pose**
            for object_index, (attached_link_index, relative_pose) in attached_objects.items():
                link_pose = self.rb_states[attached_link_index][:7]  # 获取 link 当前世界坐标
                new_object_pose = apply_relative_pose(link_pose, relative_pose)  # 计算新位姿
                update_object_pose(object_index, new_object_pose)  # 更新物体位姿

            # Deploy actions
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))

            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _get_dof_action_from_synergy(self, synergy_action, useful_joint_index):
        # the first synergy is 0~1, the second is -1~1
        synergy_action[0] = abs(synergy_action[0])
        dof_action = np.matmul(synergy_action, self.synergy_action_matrix)
        dof_action = np.clip(dof_action, 0, 1.0)
        dof_action = dof_action * 2 - 1  # -1~1

        tmp = np.zeros_like(dof_action)
        # Thumb
        tmp[12:] = dof_action[:3]
        # Index
        tmp[0:3] = dof_action[3:6]
        # Middle
        tmp[6:9] = dof_action[6:9]
        # Ring
        tmp[9:12] = dof_action[9:12]
        # Little
        tmp[3:6] = dof_action[12:15]
        dof_action = tmp

        dof_action = scale_np(dof_action,
                              self.robot_upper_limits[useful_joint_index],
                              self.robot_lower_limits[useful_joint_index])
        return dof_action


def scale_np(value, upper, lower):
    """
    将 `value` 从 `[-1, 1]` 线性缩放到 `[lower, upper]`
    """
    return lower + (value + 1) * (upper - lower) / 2
