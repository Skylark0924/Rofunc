import os
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Im
from typing import List

from rofunc.utils.logger.beauty_logger import beauty_print


class RobotSim:
    def __init__(self, args, robot_name, asset_root=None, fix_base_link=False, flip_visual_attachments=True,
                 init_pose_vec=(0, 0.5, 0.0), num_envs=1):
        """
        Initialize the robot simulator
        :param args: arguments
        :param robot_name: name of the robot
        :param asset_root: path to the assets, 
                           e.g., /home/ubuntu/anaconda3/lib/python3.7/site-packages/rofunc/simulator/assets
        """
        self.args = args
        self.robot_name = robot_name
        self.fix_base_link = fix_base_link
        self.flip_visual_attachments = flip_visual_attachments
        self.init_pose_vec = init_pose_vec
        self.num_envs = num_envs
        if self.robot_name == "CURI":
            self.asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
            self.init_pose_vec = (0, 0.0, 0.0)
        elif self.robot_name == "walker":
            self.asset_file = "urdf/walker/urdf/walker.urdf"
            self.fix_base_link = True
            self.flip_visual_attachments = False
            self.init_pose_vec = (0, 2.0, 0.0)
        elif self.robot_name == "CURI-mini":
            self.asset_file = "urdf/curi_mini/urdf/diablo_simulation.urdf"
            self.flip_visual_attachments = False
        elif self.robot_name == "franka":
            self.asset_file = "urdf/franka_description/robots/franka_panda.urdf"
            self.fix_base_link = True
            self.init_pose_vec = (0, 0.0, 0.0)
        elif self.robot_name == "baxter":
            self.asset_file = "urdf/baxter/robot.xml"
            self.init_pose_vec = (0, 1.0, 0.0)
        elif self.robot_name == "sawyer":
            self.asset_file = "urdf/sawyer/robot.xml"
        else:
            raise ValueError(
                "The robot {} is not supported. Please choose a robot in [CURI, walker, CURI-mini, baxter, sawyer]".format(
                    self.robot_name))

        if asset_root is None:
            import site
            pip_root_path = site.getsitepackages()[0]
            self.asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")

        # Initial gym, sim and env
        self._init_sim()
        self._init_env()

    def _init_sim(self, cam_pos=(3.0, 2.0, 0.0), cam_target=(0.0, 0.0, 0.0), up_axis="Y"):
        from isaacgym import gymapi

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        if up_axis == "Y":
            self.sim_params.gravity.y = -9.80
            self.sim_params.up_axis = gymapi.UP_AXIS_Y
        elif up_axis == "Z":
            self.sim_params.gravity.y = 0.0
            self.sim_params.gravity.z = -9.80
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
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
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu

        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id,
                                       self.args.physics_engine, self.sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # Create viewer
        self.viewer = None
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 75.0
        camera_props.width = 1920
        camera_props.height = 1080
        # camera_props.use_collision_geometry = True
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # Point camera at environments
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2]),
                                       gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2]))

    def _init_env(self, num_env=None, spacing=3.0, plane_vec=None):
        from isaacgym import gymapi

        if num_env is not None:
            self.num_envs = num_env

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        if plane_vec is not None:
            plane_params.normal = plane_vec  # z-up! gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.fix_base_link
        asset_options.flip_visual_attachments = self.flip_visual_attachments
        asset_options.armature = 0.01

        beauty_print("Loading robot asset {} from {}".format(self.asset_file, self.asset_root), type="info")
        robot_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

        # Set up the env grid
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        envs = []
        handles = []

        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.init_pose_vec[0], self.init_pose_vec[1], self.init_pose_vec[2])
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)

            # add robot
            handle = self.gym.create_actor(env, robot_asset, pose, "robot", i, 2)
            self.gym.enable_actor_dof_force_sensors(env, handle)
            handles.append(handle)

        dof_props = self.gym.get_actor_dof_properties(envs[0], handles[0])

        # override default stiffness and damping values
        # TODO: make this configurable
        dof_props['stiffness'].fill(100000.0)
        dof_props['damping'].fill(100000.0)

        # Give a desired pose for first 2 robot joints to improve stability
        dof_props["driveMode"][0:2] = gymapi.DOF_MODE_EFFORT

        dof_props["driveMode"][7:] = gymapi.DOF_MODE_EFFORT
        dof_props['stiffness'][7:] = 1e10
        dof_props['damping'][7:] = 1

        for i in range(self.num_envs):
            self.gym.set_actor_dof_properties(envs[i], handles[i], dof_props)

        self.envs = envs
        self.robot_handles = handles

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

    def _setup_robot(self):
        from isaacgym import gymapi

        # get joint limits and ranges for the robot
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.robot_handles[0])
        lower_limits = dof_props['lower']
        upper_limits = dof_props['upper']
        mids = 0.5 * (upper_limits + lower_limits)
        num_dofs = len(dof_props)

        for i in range(len(self.envs)):
            # Set updated stiffness and damping properties
            self.gym.set_actor_dof_properties(self.envs[i], self.robot_handles[i], dof_props)

            # Set robot pose so that each joint is in the middle of its actuation range
            dof_states = self.gym.get_actor_dof_states(self.envs[i], self.robot_handles[i], gymapi.STATE_NONE)
            for j in range(num_dofs):
                dof_states['pos'][j] = mids[j]
            self.gym.set_actor_dof_states(self.envs[i], self.robot_handles[i], dof_states, gymapi.STATE_POS)

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

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index):
        raise NotImplementedError

    def run_traj_multi_joints(self, traj: List, attracted_joints: List = None, update_freq=0.001):
        """
        Run the trajectory with multiple joints, the default is to run the trajectory with the left and right hand of the
        CURI robot.
        Args:
            traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
            attracted_joints: [list], e.g. ["panda_left_hand", "panda_right_hand"]
            update_freq: the frequency of updating the robot pose
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
