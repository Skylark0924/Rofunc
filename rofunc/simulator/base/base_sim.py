"""
 Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Im
from typing import List

from rofunc.utils.logger.beauty_logger import beauty_print

'''TODO: Make this page configurable'''


class PlaygroundSim:
    def __init__(self, args):
        self.args = args
        self.up_axis = None
        self.init_sim()
        self.init_viewer()
        self.init_plane()

    def init_sim(self, up_axis="Y"):
        from isaacgym import gymapi

        if hasattr(self.args, "up_axis"):
            up_axis = self.args.up_axis.upper()
        self.up_axis = up_axis

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        if up_axis == "Y":
            self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
            self.sim_params.up_axis = gymapi.UP_AXIS_Y
        elif up_axis == "Z":
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            raise ValueError("The up_axis should be in [Y, Z]")

        if self.args.physics_engine == gymapi.SIM_FLEX:
            self.sim_params.flex.solver_type = 5
            self.sim_params.flex.num_outer_iterations = 4
            self.sim_params.flex.num_inner_iterations = 15
            self.sim_params.flex.relaxation = 0.75
            self.sim_params.flex.warm_start = 0.8
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 4
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise ValueError("The physics engine should be in [SIM_FLEX, SIM_PHYSX]")

        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id,
                                       self.args.physics_engine, self.sim_params)

        if self.sim is None:
            beauty_print("Failed to create sim", type="warning")
            quit()

    def init_viewer(self):
        from isaacgym import gymapi

        if self.up_axis == "Y":
            cam_pos = (3.0, 2.0, 0.0)
            cam_target = (0.0, 0.0, 0.0)
        elif self.up_axis == "Z":
            cam_pos = (3.0, 0.0, 2.0)
            cam_target = (0.0, 0.0, 0.0)

        if hasattr(self.args, 'cam_pos') and hasattr(self.args, 'cam_target'):
            cam_pos = self.args.cam_pos
            cam_target = self.args.cam_target

        # Create viewer
        self.viewer = None
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 75.0 if not hasattr(self.args,
                                                          'camera_horizontal_fov') else self.args.camera_horizontal_fov
        camera_props.width = 1920 if not hasattr(self.args, 'camera_width') else self.args.camera_width
        camera_props.height = 1080 if not hasattr(self.args, 'camera_height') else self.args.camera_height
        # camera_props.use_collision_geometry = True
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            beauty_print("Failed to create viewer", type="warning")
            quit()

        # Point camera at environments
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2]),
                                       gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2]))

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
    def __init__(self, args, robot_name, asset_root=None, asset_file=None, fix_base_link=False,
                 flip_visual_attachments=True, init_pose_vec=None, num_envs=1, device="cpu"):
        """
        Initialize the robot simulator
        :param args: arguments
        :param robot_name: name of the robot
        :param asset_root: path to the assets, 
                           e.g., /home/ubuntu/anaconda3/lib/python3.7/site-packages/rofunc/simulator/assets
        """
        from isaacgym import gymapi

        self.args = args
        self.robot_name = robot_name
        self.fix_base_link = fix_base_link
        self.flip_visual_attachments = flip_visual_attachments
        self.init_pose_vec = init_pose_vec
        self.num_envs = num_envs
        self.device = device
        if self.robot_name == "CURI":
            self.asset_file = "urdf/curi/urdf/curi_isaacgym_dual_arm.urdf"
            self.init_pose_vec = (0., 0., 0., -0.707107, 0., 0., 0.707107) if init_pose_vec is None else init_pose_vec
        elif self.robot_name == "walker":
            self.asset_file = "urdf/walker/urdf/walker_cartesio.urdf"
            self.fix_base_link = True
            self.flip_visual_attachments = False
            self.init_pose_vec = (0., 1.3, 0., -0.707107, 0., 0., 0.707107) if init_pose_vec is None else init_pose_vec
        elif self.robot_name == "CURI-mini":
            self.asset_file = "urdf/curi_mini/urdf/diablo_simulation.urdf"
            self.flip_visual_attachments = False
        elif self.robot_name == "franka":
            self.asset_file = "urdf/franka_description/robots/franka_panda.urdf"
            self.fix_base_link = True
            self.init_pose_vec = (0., 0., 0., -0.707107, 0., 0., 0.707107) if init_pose_vec is None else init_pose_vec
        elif self.robot_name == "baxter":
            self.asset_file = "urdf/baxter/robot.xml"
            self.init_pose_vec = (0., 1., 0., -0.707107, 0., 0., 0.707107) if init_pose_vec is None else init_pose_vec
        elif self.robot_name == "sawyer":
            self.asset_file = "urdf/sawyer/robot.xml"
        elif self.robot_name == "gluon":
            self.asset_file = "urdf/gluon/gluon.urdf"
            self.flip_visual_attachments = False
            self.fix_base_link = True
            self.init_pose_vec = (0., 0., 0., -0.707107, 0., 0., 0.707107) if init_pose_vec is None else init_pose_vec
        elif self.robot_name == "human":
            self.asset_file = asset_file
            self.flip_visual_attachments = False
            self.fix_base_link = True
            pos_y, pos_z = 2., 0.
        else:
            raise ValueError(
                "The robot {} is not supported. Please choose a robot in [CURI, walker, CURI-mini, baxter, sawyer]".format(
                    self.robot_name))

        if hasattr(self.args, "up_axis"):  # TODO: suit for z-up setting
            up_axis = self.args.up_axis.upper()
            if up_axis == "Z":
                pos_z, pos_y = pos_y, pos_z
        self.init_pose_vec = (0., pos_y, pos_z, -0.707107, 0., 0., 0.707107) if self.init_pose_vec is None else self.init_pose_vec

        # Find the asset root folder
        if asset_root is None:
            import site
            pip_root_path = site.getsitepackages()[0]
            self.asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
        else:
            self.asset_root = asset_root

        self.asset_options = gymapi.AssetOptions()
        self.asset_options.fix_base_link = self.fix_base_link
        self.asset_options.flip_visual_attachments = self.flip_visual_attachments
        self.asset_options.armature = 0.01

    def init(self):
        # Initial gym, sim, viewer and env
        self.PlaygroundSim = PlaygroundSim(self.args)
        self.gym = self.PlaygroundSim.gym
        self.sim = self.PlaygroundSim.sim
        self.viewer = self.PlaygroundSim.viewer
        self.init_env()

        self.setup_robot_dof_prop()
        self.robot_dof = self.gym.get_actor_dof_count(self.envs[0], self.robot_handles[0])

    def init_env(self, spacing=3.0):
        from isaacgym import gymapi

        beauty_print("Loading robot asset {} from {}".format(self.asset_file, self.asset_root), type="info")
        self.robot_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, self.asset_options)

        # Set up the env grid
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        envs = []
        robot_handles = []

        # configure env grid
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))
        pose = gymapi.Transform()
        init_pose = self.init_pose_vec
        pose.p = gymapi.Vec3(init_pose[0], init_pose[1], init_pose[2])
        pose.r = gymapi.Quat(init_pose[3], init_pose[4], init_pose[5], init_pose[6])
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)

            # add robot
            robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "robot", i, 2)
            self.gym.enable_actor_dof_force_sensors(env, robot_handle)
            robot_handles.append(robot_handle)

        self.envs = envs
        self.robot_handles = robot_handles

    def _init_attractor(self, attracted_joint):
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

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, env, attractor_properties.target)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env, attractor_properties.target)

            attractor_handle = self.gym.create_rigid_body_attractor(env, attractor_properties)
            attractor_handles.append(attractor_handle)
        return attractor_handles, axes_geom, sphere_geom

    def setup_robot_dof_prop(self, **kwargs):
        raise NotImplementedError

    def _setup_attractors(self, traj, attracted_joints):
        assert isinstance(attracted_joints, list), "The attracted joints should be a list"
        assert len(attracted_joints) > 0, "The length of the attracted joints should be greater than 0"
        assert len(attracted_joints) == len(traj), "The first dimension of trajectory should equal to attracted_joints"

        attractor_handles, axes_geoms, sphere_geoms = [], [], []
        for i in range(len(attracted_joints)):
            attractor_handle, axes_geom, sphere_geom = self._init_attractor(attracted_joints[i])
            attractor_handles.append(attractor_handle)
            axes_geoms.append(axes_geom)
            sphere_geoms.append(sphere_geom)
        return attracted_joints, attractor_handles, axes_geoms, sphere_geoms

    def add_object(self, object_asset, object_poses, object_name, object_color=None, collision_group=0, filter_mask=-1,
                   seg_id=0):
        """

        :param object_asset:
        :param object_poses:
        :param object_name:
        :param object_color:
        :param collision_group:
        :param filter_mask:
        :param seg_id:
        :return:
        """
        from isaacgym import gymapi

        object_idxs = []
        object_handles = []
        for i in range(self.num_envs):
            if isinstance(object_poses, list):
                object_pose = object_poses[i]
            else:
                object_pose = object_poses

            object_handle = self.gym.create_actor(self.envs[i], object_asset, object_pose, object_name, collision_group,
                                                  filter_mask, seg_id)
            object_handles.append(object_handle)

            if object_color is not None:
                self.gym.set_rigid_body_color(self.envs[i], object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              object_color)

            object_idx = self.gym.get_actor_rigid_body_index(self.envs[i], object_handle, 0, gymapi.DOMAIN_SIM)
            object_idxs.append(object_idx)
        return object_handles, object_idxs

    def show(self, visual_obs_flag=False, camera_props=None, attached_body=None, local_transform=None):
        """
        Visualize the robot in an interactive viewer
        :param visual_obs_flag: If True, show the visual observation
        :param camera_props: If visual_obs_flag is True, use this camera_props to config the camera
        :param attached_body: If visual_obs_flag is True, use this to refer the body the camera attached to
        :param local_transform: If visual_obs_flag is True, use this local transform to adjust the camera pose
        """
        from isaacgym import gymapi

        beauty_print("Show the {} simulator in the interactive mode".format(self.robot_name), 1)

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

    def ik_controller(self, joint_index, dpose, damping=0.05):
        jacobian = self.get_robot_jacobian()

        # jacobian entries corresponding to curi hand
        j_eef = jacobian[:, joint_index - 1, :, 7:14]

        import torch

        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def run_traj_multi_joints(self, traj: List, attracted_joints: List = None, update_freq=0.001):
        """
        Run the trajectory with multiple joints, the default is to run the trajectory with the left and right hand of
        bimanual robot.
        :param traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
        :param attracted_joints: [list], e.g. ["panda_left_hand", "panda_right_hand"]
        :param update_freq: the frequency of updating the robot pose
        :return:
        """
        assert isinstance(traj, list) and len(traj) > 0, "The trajectory should be a list of numpy arrays"

        beauty_print('Execute multi-joint trajectory with the CURI simulator')

        # Create the attractor
        attracted_joints, attractor_handles, axes_geoms, sphere_geoms = self._setup_attractors(traj, attracted_joints)

        # Time to wait in seconds before moving robot
        next_update_time = 1
        index = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)
            if t >= next_update_time:
                self.gym.clear_lines(self.viewer)
                for i in range(len(attracted_joints)):
                    self.update_robot(traj[i], attractor_handles[i], axes_geoms[i], sphere_geoms[i], index)
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
