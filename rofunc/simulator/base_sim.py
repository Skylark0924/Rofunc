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

import os
from typing import List

import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Im

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print


class PlaygroundSim:
    def __init__(self, args):
        self.args = args
        self.up_axis = None
        self.init_sim()
        self.init_viewer()
        self.init_plane()

    def init_sim(self):
        from isaacgym import gymapi

        self.up_axis = self.args.sim.up_axis.upper()

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = self.args.sim.dt
        self.sim_params.substeps = self.args.sim.substeps
        self.sim_params.gravity = gymapi.Vec3(*self.args.sim.gravity)
        self.sim_params.up_axis = gymapi.UP_AXIS_Y if self.up_axis == "Y" else gymapi.UP_AXIS_Z

        if self.args.physics_engine == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            self.sim_params.flex.solver_type = self.args.get("sim_flex_solver_type", 5)
            self.sim_params.flex.num_outer_iterations = self.args.get("sim_flex_num_outer_iterations", 4)
            self.sim_params.flex.num_inner_iterations = self.args.get("sim_flex_num_inner_iterations", 15)
            self.sim_params.flex.relaxation = self.args.get("sim_flex_relaxation", 0.75)
            self.sim_params.flex.warm_start = self.args.get("sim_flex_warm_start", 0.8)
        elif self.args.physics_engine == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            self.sim_params.physx.solver_type = self.args.sim.physx.solver_type
            self.sim_params.physx.num_position_iterations = self.args.sim.physx.num_position_iterations
            self.sim_params.physx.num_velocity_iterations = self.args.sim.physx.num_velocity_iterations
            self.sim_params.physx.rest_offset = self.args.sim.physx.rest_offset
            self.sim_params.physx.contact_offset = self.args.sim.physx.contact_offset
            self.sim_params.physx.friction_offset_threshold = self.args.sim.physx.friction_offset_threshold
            self.sim_params.physx.friction_correlation_distance = self.args.sim.physx.friction_correlation_distance
            self.sim_params.physx.num_threads = self.args.sim.physx.num_threads
            self.sim_params.physx.use_gpu = self.args.sim.physx.use_gpu
        else:
            raise ValueError("The physics engine should be in [flex, physx]")

        self.sim_params.use_gpu_pipeline = self.args.sim.use_gpu_pipeline
        if self.sim_params.use_gpu_pipeline:
            beauty_print("WARNING: Forcing CPU pipeline.", type="warning")

        split_device = self.args.sim_device.split(":")
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0
        self.sim = self.gym.create_sim(self.device_id, self.args.graphics_device_id,
                                       self.physics_engine, self.sim_params)

        if self.sim is None:
            beauty_print("Failed to create sim", type="warning")
            quit()

    def init_viewer(self):
        from isaacgym import gymapi

        if self.up_axis == "Y":
            cam_pos = self.args.get("cam_pos", (3.0, 2.0, 0.0))
            cam_target = self.args.get("cam_target", (0.0, 0.0, 0.0))
        elif self.up_axis == "Z":
            cam_pos = self.args.get("cam_pos", (3.0, 0.0, 2.0))
            cam_target = self.args.get("cam_target", (0.0, 0.0, 0.0))

        # Create viewer
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = self.args.get("camera_horizontal_fov", 75.0)
        camera_props.width = self.args.get("camera_width", 1920)
        camera_props.height = self.args.get("camera_height", 1080)
        camera_props.use_collision_geometry = self.args.get("camera_use_collision_geometry", False)
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            beauty_print("Failed to create viewer", type="warning")
            quit()

        # Point camera at environments
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))

    def init_plane(self):
        from isaacgym import gymapi
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        if self.up_axis == "Y":
            plane_params.normal = gymapi.Vec3(0, 1, 0)
        elif self.up_axis == "Z":
            plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)


class RobotSim:
    def __init__(self, args):
        """
        Initialize the robot-centered simulator

        :param args: arguments
        """
        self.args = args
        self.num_envs = self.args.env.numEnvs

        # Initial gym, sim, and viewer
        self.PlaygroundSim = PlaygroundSim(self.args)
        self.gym = self.PlaygroundSim.gym
        self.sim = self.PlaygroundSim.sim
        self.viewer = self.PlaygroundSim.viewer

        self.create_env()
        self.setup_robot_dof_prop()

        if self.args.env.object_asset is not None:
            self.add_object()

    def create_env(self):
        from isaacgym import gymapi

        # Load robot asset
        asset_root = self.args.env.asset.assetRoot or os.path.join(rf.oslab.get_rofunc_path(), "simulator/assets")
        asset_file = self.args.env.asset.assetFile

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.args.env.asset.fix_base_link
        asset_options.flip_visual_attachments = self.args.env.asset.flip_visual_attachments
        asset_options.armature = self.args.env.asset.armature

        init_pose = self.args.env.asset.init_pose
        self.robot_name = self.args.env.asset.robot_name

        beauty_print("Loading robot asset {} from {}".format(asset_file, asset_root), type="info")
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Set up the env grid
        spacing = self.args.env.envSpacing
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        envs = []
        robot_handles = []

        # configure env grid
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*init_pose[:3])
        pose.r = gymapi.Quat(*init_pose[3:7])
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)

            # add robot
            robot_handle = self.gym.create_actor(env, self.robot_asset, pose, self.robot_name, i, 2)
            self.gym.enable_actor_dof_force_sensors(env, robot_handle)
            robot_handles.append(robot_handle)

        self.envs = envs
        self.robot_handles = robot_handles

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

    def add_object(self):
        # TODO
        from isaacgym import gymapi

        asset_root = self.args.env.object_asset.assetRoot or os.path.join(rf.oslab.get_rofunc_path(),
                                                                          "simulator/assets")
        asset_file = self.args.env.object_asset.assetFile

        asset_options = gymapi.AssetOptions()
        init_pose = self.args.env.object_asset.init_pose
        object_name = self.args.env.object_asset.object_name

        beauty_print("Loading object asset {} from {}".format(asset_file, asset_root), type="info")
        self.object_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.object_idxs = []
        self.object_handles = []
        for i, env_ptr in enumerate(self.envs):
            if isinstance(object_poses, list):
                object_pose = init_pose[i]
            else:
                object_pose = init_pose

            object_handle = self.gym.create_actor(env_ptr, self.object_asset, init_pose, object_name, collision_group,
                                                  filter_mask, seg_id)
            self.object_handles.append(object_handle)

            if object_color is not None:
                self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              object_color)

            object_idx = self.gym.get_actor_rigid_body_index(self.envs[i], object_handle, 0, gymapi.DOMAIN_SIM)
            self.object_idxs.append(object_idx)
            self.current_poses = self.object_poses
        return self.object_handles, self.object_idxs

    def _monitor_rigid_body_states(self):
        from isaacgym import gymtorch
        self.gym.prepare_sim(self.sim)
        self.rigid_body_states_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(self.rigid_body_states_tensor)

    def _init_attractor(self, attracted_joint, verbose=True):
        """
        Initialize the attractor for tracking the trajectory using the embedded Isaac Gym PID controller

        :param attracted_joint: the joint to be attracted
        :param verbose: if True, visualize the attractor spheres
        :return:
        """
        from isaacgym import gymapi
        from isaacgym import gymutil

        # Attractor setup
        attractor_handles = []
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3

        # Make attractor in all axes
        attractor_properties.axes = gymapi.AXIS_ALL

        # Create helper geometry used for visualization
        # Create a wireframe axis
        axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

        for i in range(len(self.envs)):
            env = self.envs[i]
            handle = self.robot_handles[i]

            body_dict = self.gym.get_actor_rigid_body_dict(env, handle)
            props = self.gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
            attracted_joint_handle = body = self.gym.find_actor_rigid_body_handle(env, handle, attracted_joint)

            # Initialize the attractor
            attractor_properties.target = props['pose'][:][body_dict[attracted_joint]]
            attractor_properties.target.p.y -= 0.1
            attractor_properties.target.p.z = 0.1
            attractor_properties.rigid_handle = attracted_joint_handle

            if verbose:
                # Draw axes and sphere at attractor location
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, env, attractor_properties.target)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env, attractor_properties.target)

            attractor_handle = self.gym.create_rigid_body_attractor(env, attractor_properties)
            attractor_handles.append(attractor_handle)
        return attractor_handles, axes_geom, sphere_geom

    def _setup_attractors(self, traj, attracted_joints, verbose=True):
        assert isinstance(attracted_joints, list), "The attracted joints should be a list"
        assert len(attracted_joints) > 0, "The length of the attracted joints should be greater than 0"
        assert len(attracted_joints) == len(traj), "The first dimension of trajectory should equal to attracted_joints"

        attractor_handles, axes_geoms, sphere_geoms = [], [], []
        for i in range(len(attracted_joints)):
            attractor_handle, axes_geom, sphere_geom = self._init_attractor(attracted_joints[i], verbose=verbose)
            attractor_handles.append(attractor_handle)
            axes_geoms.append(axes_geom)
            sphere_geoms.append(sphere_geom)
        return attracted_joints, attractor_handles, axes_geoms, sphere_geoms

    def add_marker(self, marker_pose):
        from isaacgym import gymapi

        asset_file = "mjcf/location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options)

        self.marker_handles = []
        for i in range(self.num_envs):
            marker_handle = self.gym.create_actor(self.envs[i], self._marker_asset, marker_pose, "marker", i, 2, 0)
            self.gym.set_rigid_body_color(self.envs[i], marker_handle, 0, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(0.8, 0.0, 0.0))
            self.marker_handles.append(marker_handle)

    def show(self, visual_obs_flag=False, camera_props=None, attached_body=None, local_transform=None):
        """
        Visualize the robot in an interactive viewer
        :param visual_obs_flag: If True, show the visual observation
        :param camera_props: If visual_obs_flag is True, use this camera_props to config the camera
        :param attached_body: If visual_obs_flag is True, use this to refer the body the camera attached to
        :param local_transform: If visual_obs_flag is True, use this local transform to adjust the camera pose
        """
        from isaacgym import gymapi

        beauty_print("Show the {} simulator in the interactive mode".format(self.robot_name), type="module")

        if visual_obs_flag:
            fig = plt.figure("Visual observation", figsize=(8, 8))
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
            body_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], attached_body)
            self.gym.attach_camera_to_body(camera_handle, self.envs[0], body_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

        while not self.gym.query_viewer_has_closed(self.viewer):
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)

            if visual_obs_flag:
                # digest image
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                cam_img = self.gym.get_camera_image(self.sim, self.envs[0], camera_handle, gymapi.IMAGE_COLOR).reshape(
                    1280, 1280, 4)
                cam_img = Im.fromarray(cam_img)
                plt.imshow(cam_img)
                plt.axis('off')
                plt.pause(1e-9)
                fig.clf()

                self.gym.end_access_image_tensors(self.sim)

            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_num_bodies(self):
        from isaacgym import gymapi

        robot_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, gymapi.AssetOptions())
        num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        beauty_print("The number of bodies in the robot asset is {}".format(num_bodies), 2)
        return num_bodies

    def get_dof_info(self):
        # Gets number of Degree of Freedom for an actor
        dof_count = self.gym.get_actor_dof_count(self.envs[0], self.robot_handles[0])
        # maps degree of freedom names to actor-relative indices
        dof_dict = self.gym.get_actor_dof_dict(self.envs[0], self.robot_handles[0])
        # Gets forces for the actor’s degrees of freedom
        # dof_forces = self.gym.get_actor_dof_forces(self.envs[0], self.robot_handles[0])
        # Gets Frames for Degrees of Freedom of actor
        # dof_frames = self.gym.get_actor_dof_frames(self.envs[0], self.robot_handles[0])
        # Gets names of all degrees of freedom on actor
        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.robot_handles[0])
        # Gets target position for the actor’s degrees of freedom.
        # dof_position_targets = self.gym.get_actor_dof_position_targets(self.envs[0], self.robot_handles[0])
        # Gets properties for all Dofs on an actor.
        dof_properties = self.gym.get_actor_dof_properties(self.envs[0], self.robot_handles[0])
        # Gets state for the actor’s degrees of freedom
        # dof_states = self.gym.get_actor_dof_states(self.envs[0], self.robot_handles[0], gymapi.STATE_ALL)
        # Gets target velocity for the actor’s degrees of freedom
        # dof_velocity_targets = self.gym.get_actor_dof_velocity_targets(self.envs[0], self.robot_handles[0])

        return {'dof_count': dof_count, 'dof_dict': dof_dict, 'dof_names': dof_names, 'dof_properties': dof_properties}

    def get_robot_state(self, mode):
        from isaacgym import gymtorch

        if mode == 'dof_force':
            # One force value per each DOF
            robot_dof_force = np.array(gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)))
            beauty_print('DOF forces:\n {}'.format(robot_dof_force), 2)
            return robot_dof_force
        elif mode == 'dof_state':
            # Each DOF state contains position and velocity and force sensor value
            for i in range(len(self.envs)):
                # TODO: multi envs
                robot_dof_force = np.array(self.gym.get_actor_dof_forces(self.envs[i], self.robot_handles[i])).reshape(
                    (-1, 1))
            robot_dof_pose_vel = np.array(gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))
            robot_dof_state = np.hstack((robot_dof_pose_vel, robot_dof_force))
            beauty_print('DOF states:\n {}'.format(robot_dof_state), 2)
            return robot_dof_state
        elif mode == 'dof_pose_vel':
            # Each DOF state contains position and velocity
            robot_dof_pose_vel = np.array(gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))
            beauty_print('DOF poses and velocities:\n {}'.format(robot_dof_pose_vel), 2)
            return robot_dof_pose_vel
        elif mode == 'dof_pose':
            # Each DOF pose contains position
            robot_dof_pose_vel = np.array(gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))
            beauty_print('DOF poses:\n {}'.format(robot_dof_pose_vel[:, 0]), 2)
            return robot_dof_pose_vel[:, 0]
        elif mode == 'dof_vel':
            # Each DOF velocity contains velocity
            robot_dof_pose_vel = np.array(gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)))
            beauty_print('DOF velocities:\n {}'.format(robot_dof_pose_vel[:, 1]), 2)
            return robot_dof_pose_vel[:, 1]
        elif mode == 'dof_force_np':
            for i in range(len(self.envs)):
                # TODO: multi envs
                robot_dof_force = self.gym.get_actor_dof_forces(self.envs[i], self.robot_handles[i])
                beauty_print('DOF force s:\n {}'.format(robot_dof_force), 2)
            return robot_dof_force
        else:
            raise ValueError("The mode {} is not supported".format(mode))

    def get_robot_jacobian(self):
        from isaacgym import gymtorch

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        return jacobian

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index):
        raise NotImplementedError

    def update_object(self, object_handles, object_poses, state_type):
        """
        Update the object pose

        :param object_handles:
        :param object_poses:
        :param state_type: gymapi.STATE_ALL, gymapi.STATE_POS, gymapi.STATE_VEL
        :return:
        """
        from isaacgym import gymapi

        for i in range(len(self.envs)):
            state = self.gym.get_actor_rigid_body_states(self.envs[i], object_handles[i], gymapi.STATE_NONE)
            state['pose']['p'].fill((object_poses[i][0], object_poses[i][1], object_poses[i][2]))
            state['pose']['r'].fill((object_poses[i][3], object_poses[i][4], object_poses[i][5], object_poses[i][6]))
            state['vel']['linear'].fill((0, 0, 0))
            state['vel']['angular'].fill((0, 0, 0))
            self.gym.set_actor_rigid_body_states(self.envs[i], object_handles[i], state, state_type)

    def ik_controller(self, joint_index, dpose, damping=0.05):
        jacobian = self.get_robot_jacobian()

        # jacobian entries corresponding to curi hand
        j_eef = jacobian[:, joint_index - 1, :, 7:14]

        import torch

        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def run_traj_multi_joints(self, traj: List, attracted_joints: List = None,
                              object_start_pose: List = None, object_end_pose: List = None,
                              object_related_joints: List = None,
                              update_freq=0.001, verbose=True):
        """
        Run the trajectory with multiple joints, the default is to run the trajectory with the left and right hand of
        bimanual robot.

        :param traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
        :param attracted_joints: [list], e.g. ["panda_left_hand", "panda_right_hand"]
        :param update_freq: the frequency of updating the robot pose
        :param verbose: if True, visualize the attractor spheres
        :return:
        """
        assert isinstance(traj, list) and len(traj) > 0, "The trajectory should be a list of numpy arrays"

        beauty_print('Execute multi-joint trajectory with the CURI simulator')

        # Create the attractor
        attracted_joints, attractor_handles, axes_geoms, sphere_geoms = self._setup_attractors(traj, attracted_joints,
                                                                                               verbose=verbose)

        # Time to wait in seconds before moving robot
        next_update_time = 1
        index = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            if t >= next_update_time:
                self.gym.clear_lines(self.viewer)
                for i in range(len(attracted_joints)):
                    self.update_robot(traj[i], attractor_handles[i], axes_geoms[i], sphere_geoms[i], index, verbose)

                # if self.object_handles is not None:
                #     if index <= 1:
                #         self.object_poses = object_start_pose
                #         self.update_object(self.object_handles, object_start_pose, gymapi.STATE_ALL)
                #
                #     object_poses = self.object_poses
                #     # get global index of hand in rigid body state tensor
                #     left_hand_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0],
                #                                                          object_related_joints[0], gymapi.DOMAIN_SIM)
                #     right_hand_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0],
                #                                                           object_related_joints[1], gymapi.DOMAIN_SIM)
                #     left_hand_pos = self.rigid_body_states[left_hand_idx, :3]
                #     left_hand_rot = self.rigid_body_states[left_hand_idx, 3:7]
                #     left_hand_vel = self.rigid_body_states[left_hand_idx, 7:]
                #     right_hand_pos = self.rigid_body_states[right_hand_idx, :3]
                #     right_hand_rot = self.rigid_body_states[right_hand_idx, 3:7]
                #     right_hand_vel = self.rigid_body_states[right_hand_idx, 7:]
                #
                #     center_pos = (left_hand_pos + right_hand_pos) / 2
                #     euclidean_dist = np.linalg.norm(np.array(center_pos) - self.current_poses[0][:3])
                #     # euclidean_dist = np.linalg.norm(left_hand_pos - right_hand_pos)
                #     if euclidean_dist < 0.1:
                #         left_hand_rot_euler = rf.robolab.euler_from_quaternion(left_hand_rot)
                #         right_hand_rot_euler = rf.robolab.euler_from_quaternion(right_hand_rot)
                #         # tmp = left_hand_rot_euler[1]
                #         # object_rot = rf.robolab.quaternion_from_euler(0, left_hand_rot_euler[1], 0)
                #         # object_rot = rf.robolab.quaternion_multiply(left_hand_rot, [0.707, 0, 0.707, 0])
                #         object_rot = left_hand_rot
                #
                #         object_pos = (left_hand_pos + right_hand_pos) / 2
                #         object_poses = np.array([[*object_pos, *object_rot]])
                #
                #         done_euclidean_dist = np.linalg.norm(
                #             np.array(self.current_poses[0][:3]) - object_end_pose[:3])
                #         if done_euclidean_dist < 0.05:
                #             object_poses = self.current_poses
                #             self.object_poses = object_poses
                #
                #    self.current_poses = object_poses
                #    self.update_object(self.object_handles, object_poses, gymapi.STATE_ALL)

                next_update_time += update_freq
                index += 1
                if index >= len(traj[i]):
                    index = 0

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
