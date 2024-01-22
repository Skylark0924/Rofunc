import collections
from rofunc.learning.RofuncRL.tasks.isaacgymenv.grasp.base.base_task import Task
from rofunc.learning.RofuncRL.tasks.isaacgymenv.grasp.base.camera_task import CameraTaskMixin
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from typing import *
from rofunc.utils.oslab import get_rofunc_path


class GraspingTask(Task, CameraTaskMixin):
    def __init__(
            self,
            cfg: Dict[str, Any],
            sim_device: str,
            graphics_device_id: int,
            headless: bool
    ) -> None:
        """Initializes the control class for all grasping/ dexterous hand
        manipulation tasks.

        Args:
            cfg: Configuration dictionary that contains parameters for
                how the simulation should run and defines properties of the
                task.
            sim_device: Device on which the physics simulation is run
                (e.g. "cuda:0" or "cpu").
            graphics_device_id: ID of the device to run the rendering.
            headless:  Whether to run the simulation without a viewer.
        """
        self.cfg = cfg
        self._parse_cfg(cfg)
        super().__init__(cfg, sim_device, graphics_device_id, headless)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.sim_initialized = False
        self.sim = self.create_sim()
        self.create_envs(self.num_envs, self.cfg["env"]["envSpacing"],
                         int(np.sqrt(self.num_envs)))

        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.viewer = self.create_viewer()
        self.allocate_buffers()
        self._init_buffers()
        self.obs_dict = {}
        self.obs_separate = {}

        self.initial_reset = True

        if not headless:
            self._set_camera(cfg["viewer"]["pos"],
                             cfg["viewer"]["lookat"])

        if "cameras" in cfg.keys():
            self._create_cameras(cfg["cameras"])

    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        # TEMPORARY
        self.force_torque_obs_scale = 1.0

        # General simulation parameters
        self.dt = cfg["sim"]["dt"]
        self.max_episode_length = cfg["reset"]["maxEpisodeLength"]
        self.aggregate_mode = cfg["env"]["aggregateMode"]
        self.up_axis = "z"

        # number of actions for UR5-SIH robot
        self.cfg["env"]["numActions"] = 11

        # Reset object pose and robot joint angles
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)

        # Specify observation-space of environment
        self.cfg["env"]["numObservations"] = 0
        self.cfg["env"]["numStates"] = 0

        # Overwrite episode length if reset_time is specified
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time /
                      (self.cfg["control"]["controlFreqInv"] * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # quick and dirty fix for old configs
        if "returnObsDict" not in self.cfg["task"].keys():
            self.cfg["task"]["returnObsDict"] = False
            self.cfg["reward"]["returnRewardsDict"] = False

        self._init_summary_writer(cfg)

    def _init_summary_writer(self, cfg: Dict[str, Any]) -> None:
        train_dir = cfg.get("train_dir", "runs")
        experiment_name = cfg["experiment_name"]
        experiment_dir = os.path.join(train_dir, experiment_name)
        summaries_dir = os.path.join(experiment_dir, "summaries")
        log_dir = os.path.join(summaries_dir, "env_stats")
        self.writer = SummaryWriter(log_dir)

    def create_viewer(self):
        self.enable_viewer_sync = True
        viewer = None

        if not self.headless:
            viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_V, "toggle_viewer_sync")
        return viewer

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        if "cameras" in self.cfg.keys():
            assert "save_recordings" in self.cfg["cameras"].keys()
            for cam in self.cfg["cameras"].keys():
                if cam != "save_recordings":
                    camera_name = cam
            num_cameras = len(self.cfg["cameras"].keys()) - 3
            height = self.cfg["cameras"][camera_name]["height"]
            width = self.cfg["cameras"][camera_name]["width"]
            camera_type = self.cfg["cameras"][camera_name]["type"]

            if camera_type == "rgb":
                num_channels = 3
            elif camera_type == "rgbd":
                num_channels = 4
            else:
                assert False

            self.image_buf = torch.zeros(
                (self.num_envs, num_cameras, height, width, num_channels),
                device=self.device, dtype=torch.float)

            if self.cfg["cameras"]["convert_to_pointcloud"]:
                self.pointcloud_buf = torch.zeros(
                    (self.num_envs, num_cameras, height * width, 6),
                    device=self.device, dtype=torch.float)

            if self.cfg["cameras"]["convert_to_voxelgrid"]:
                self.voxelgrid_buf = torch.zeros(
                    (self.num_envs, 101, 101, 101, 4), device=self.device,
                    dtype=torch.float)

    def create_sim(self):
        """Create the Isaac Gym simulation

        Returns:
            Isaac Gym simulation object.
        """

        sim = self.gym.create_sim(self.device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self._create_ground_plane(sim)
        return sim

    def _create_ground_plane(self, sim) -> None:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(sim, plane_params)

    def create_envs(self, num_envs: int, spacing: float, num_per_row: int
                     ) -> None:
        """Creates the environments:
            1. Loads assets from URDFs (robot, table, task-assets),
            2. For each environment
                - Creates the environment,
                - Adds the loaded assets as actors
            3. Stores handles and indices to be used later
        """

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, 2 * spacing)

        # Load assets to be added
        assets_dict = self._load_assets()
        self.envs, self.robots = [], []
        self.robot_indices, self.fingertip_indices = [], []

        for i in range(num_envs):
            # Create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            agg_info = self._begin_aggregate(env_ptr, i)

            # Add actors
            self._add_robot(env_ptr, i, assets_dict["robot"]["asset"],
                            assets_dict["robot"]["start_pose"])
            self._add_table(env_ptr, i, assets_dict["table"]["asset"],
                            assets_dict["table"]["start_pose"])
            self._add_task(env_ptr, i, assets_dict["task"], agg_info)

            self._end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        self._process_handles()
        self._process_task_handles()

    def _load_assets(self) -> Dict[str, Any]:
        assets_dict = collections.defaultdict(dict)

        # Load robot asset
        robot_asset, robot_start_pose = self._load_robot_asset()
        self._process_robot_props(robot_asset)
        assets_dict["robot"]["asset"] = robot_asset
        assets_dict["robot"]["start_pose"] = robot_start_pose

        # Load table asset
        table_asset, table_start_pose = self._load_table_asset()

        table_start_pose.r = gymapi.Quat(0, 0, 0.707, 0.707)
        table_start_pose.p = gymapi.Vec3(0, -0.05, 0)
        assets_dict["table"]["asset"] = table_asset
        assets_dict["table"]["start_pose"] = table_start_pose

        # Load task-specific assets
        assets_dict["task"].update(self._load_task_assets())
        return assets_dict

    @property
    def asset_root(self):
        rofunc_path = get_rofunc_path()
        asset_root = os.path.join(rofunc_path, "simulator/assets")
        if "asset" in self.cfg:
            return os.path.normpath(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), self.cfg["asset"].get(
                "assetRoot", asset_root)))
        return asset_root

    @property
    def robot_asset_file(self) -> os.path:
        robot_asset_file = os.path.normpath(
            "urdf/sih/ur5e_schunk_sih_right.urdf")
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "robotAssetFile", robot_asset_file))
        return robot_asset_file

    @property
    def table_asset_file(self) -> os.path:
        table_asset_file = "urdf/table.urdf"
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "tableAssetFile", table_asset_file))
        return table_asset_file

    def _load_robot_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.use_mesh_materials = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.override_com = True
        asset_options.override_inertia = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, self.asset_root,
                                          self.robot_asset_file, asset_options)
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0, -0.7, 1.1)
        return robot_asset, robot_start_pose

    def _process_robot_props(self, robot_asset):
        """Processes some robot specific properties, such as DOF props and
        limits and the coupling of the joints in the SIH hand."""
        self.fingertips = ["th_distal", "if_distal", "mf_distal", "rf_distal",
                           "lf_distal"]
        self.num_fingertips = len(self.fingertips)
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(
            robot_asset, name) for name in self.fingertips]

        # Get counts
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        self._set_robot_dof_props(robot_asset)
        self._set_robot_dof_limits(robot_asset)
        self._couple_robot_joints(robot_asset)

    def _set_robot_dof_props(self, robot_asset) -> None:
        # Set the DOF properties of robot arm and hand
        self.robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        for idx in range(self.num_robot_dofs):
            if idx < 6:
                self.robot_dof_props["stiffness"][idx] = \
                    self.cfg["asset"]["robotProps"]["armStiffness"]
                self.robot_dof_props["damping"][idx] = \
                    self.cfg["asset"]["robotProps"]["armDamping"]
                if "armMaxEffort" in self.cfg["asset"]["robotProps"]:
                    self.robot_dof_props["effort"][idx] = \
                        self.cfg["asset"]["robotProps"]["armMaxEffort"]
            else:
                self.robot_dof_props["stiffness"][idx] = \
                    self.cfg["asset"]["robotProps"]["handStiffness"]
                self.robot_dof_props["damping"][idx] = \
                    self.cfg["asset"]["robotProps"]["handDamping"]
                self.robot_dof_props["armature"][idx] = 0.005

        # Specify actuated DOFs of arm and hand
        arm_actuated_dof_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                                  'elbow_joint', 'wrist_1_joint',
                                  'wrist_2_joint', 'wrist_3_joint']
        hand_actuated_dof_names = ['palm_to_if_proximal', 'palm_to_mf_proximal',
                                   'palm_to_rf_proximal', 'palm_to_th_proximal',
                                   'th_proximal_to_th_inter']
        self.arm_actuated_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in arm_actuated_dof_names]
        self.hand_actuated_dof_indices = [
            self.gym.find_asset_dof_index(robot_asset, name)
            for name in hand_actuated_dof_names]
        self.actuated_dof_indices = self.arm_actuated_dof_indices + \
                                    self.hand_actuated_dof_indices
        self.arm_actuated_dof_indices = to_torch(
            self.arm_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.hand_actuated_dof_indices = to_torch(
            self.hand_actuated_dof_indices, dtype=torch.long,
            device=self.device)
        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device)

    def _set_robot_dof_limits(self, robot_asset) -> None:
        # Set DOF limits and default positions and velocities
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.robot_dof_default_vel = []
        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(self.robot_dof_props["lower"][i])
            self.robot_dof_upper_limits.append(self.robot_dof_props["upper"][i])
            self.robot_dof_default_vel.append(0.0)
        arm_dof_default_pos = [1.3143, -0.61410,  1.7623, -1.1482,  1.3143,
                               -3.1416]
        arm_dof_away_pos = [1.3143 - 0.5 * np.pi, -0.61410,  1.7623, -1.1482,  1.3143,
                            -3.1416]
        hand_dof_default_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.robot_dof_default_pos = arm_dof_default_pos + hand_dof_default_pos

        self.robot_dof_away_pos = arm_dof_away_pos + hand_dof_default_pos
        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits,
                                               device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits,
                                               device=self.device)

        # enforce direction the arm in bend
        self.robot_dof_lower_limits[2] = 0.025  # elbow must stay up
        self.robot_dof_lower_limits[3] = -np.pi + 0.1 * np.pi
        self.robot_dof_upper_limits[3] = 0.5 * np.pi - 0.1 * np.pi
        self.robot_dof_lower_limits[4] = 0 + 0.01 * np.pi
        self.robot_dof_upper_limits[4] = np.pi - 0.01 * np.pi

        self.robot_dof_default_pos = to_torch(self.robot_dof_default_pos,
                                              device=self.device)
        self.robot_dof_default_vel = to_torch(self.robot_dof_default_vel,
                                              device=self.device)
        self.robot_dof_away_pos = to_torch(self.robot_dof_away_pos,
                                           device=self.device)

        # Set EEF-Pose limits and default
        self.hand_default_pose = to_torch(
            [0, 0, 1.1, 0, 0, 0], device=self.device)
        self.hand_pose_limits = to_torch(
            [0.3, 0.3, 0.3, (1 / 2) * np.pi, (1 / 2) * np.pi, (1 / 2) * np.pi],
            device=self.device)

        self.hand_pose_lower_limits = self.hand_default_pose - self.hand_pose_limits
        self.hand_pose_upper_limits = self.hand_default_pose + self.hand_pose_limits
        self.hand_pose_max_deviation = to_torch(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device)
        self.hand_angle_max_deviation = to_torch(
            [0.33, 0.33, 0.33, 0.33, 0.33], device=self.device)

    def _couple_robot_joints(self, robot_asset) -> None:
        # Couple DOFs of SIH hand
        hand_coupled_joint_names = [
            ["palm_to_if_proximal", "if_proximal_to_if_distal"],
            ["palm_to_mf_proximal", "mf_proximal_to_mf_distal"],
            ["palm_to_rf_proximal", "rf_proximal_to_rf_distal"],
            ["palm_to_rf_proximal", "palm_to_lf_proximal"],
            ["palm_to_lf_proximal", "lf_proximal_to_lf_distal"],
            ["th_proximal_to_th_inter", "th_inter_to_th_distal"]]
        hand_coupled_joint_vals = [[0, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 1, 0, 0]]
        self.hand_coupled_joint_indices = []
        for row in hand_coupled_joint_names:
            self.hand_coupled_joint_indices.append(
                [self.gym.find_asset_dof_index(robot_asset, name)
                 for name in row])
        self.hand_coupled_joint_indices = to_torch(
            self.hand_coupled_joint_indices, dtype=torch.long,
            device=self.device)
        self.hand_coupled_joint_vals = to_torch(hand_coupled_joint_vals,
                                                device=self.device)

    def _load_table_asset(self):
        """Loads the table and robot stand and keeps track of the number of
        rigid bodies and shapes they add to the environment."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        table_asset = self.gym.load_asset(self.sim, self.asset_root,
                                          self.table_asset_file, asset_options)
        table_start_pose = gymapi.Transform()
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_table_shapes = self.gym.get_asset_rigid_shape_count(
            table_asset)
        return table_asset, table_start_pose

    def _load_task_assets(self) -> Dict[str, Any]:
        """Loads task specific assets"""
        raise NotImplementedError

    def _begin_aggregate(self, env_ptr, env_idx: int) -> None:
        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies
        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes
        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)

    def _end_aggregate(self, env_ptr) -> None:
        if self.aggregate_mode > 0:
            self.gym.end_aggregate(env_ptr)

    def _add_table(self, env_ptr, i, table_asset, table_start_pose) -> None:
        table_actor = self.gym.create_actor(env_ptr, table_asset,
                                            table_start_pose, "table", i, -1, 1)

    def _add_robot(self, env_ptr, i, robot_asset, robot_start_pose) -> None:
        robot_actor = self.gym.create_actor(env_ptr, robot_asset,
                                            robot_start_pose, "robot", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, robot_actor,
                                          self.robot_dof_props)
        robot_idx = self.gym.get_actor_index(env_ptr, robot_actor,
                                             gymapi.DOMAIN_SIM)

        self.robot_indices.append(robot_idx)
        self.robots.append(robot_actor)

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any], agg_info) -> None:
        raise NotImplementedError

    def _process_handles(self) -> None:
        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device)
        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device)
        self.fingertip_actor_rb_handle = [
            self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.robots[0], name) for name in self.fingertips]
        self.fingertip_actor_rb_handle = to_torch(
            self.fingertip_actor_rb_handle, dtype=torch.long,
            device=self.device)

    def _process_task_handles(self) -> None:
        raise NotImplementedError

    def _init_buffers(self) -> None:
        """Initialize tensors that contain the state of the simulation."""
        self.env_steps = 0

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.robot_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13)

        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(
            actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs),
                                        dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs),
                                       dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1)

        # Acquire unit tensors
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1))

        if self.cfg["sim"]["useContactForces"]:
            self._acquire_contact_force_tensor()

        if self.cfg["sim"]["useForceSensors"]:
            self._acquire_force_sensor_tensor()

        actor_count = self.gym.get_actor_count(self.envs[0])
        actor_names = [self.gym.get_actor_name(self.envs[0], i)
                       for i in range(actor_count)]
        actor_rigid_body_counts = [self.gym.get_actor_rigid_body_count(
            self.envs[0], i) for i in range(actor_count)]
        actor_dof_counts = [self.gym.get_actor_dof_count(
            self.envs[0], i) for i in range(actor_count)]

        self.gym.refresh_jacobian_tensors(self.sim)

        dof_state = self.dof_state.view(
            self.num_envs, sum(actor_dof_counts), 2)

        self.rigid_body_state_dict, self.jacobian_dict, self.dof_dict = \
            {}, {}, {}
        dof_count, rb_count = 0, 0
        for i, name in enumerate(actor_names):
            # Rigid body states
            rigid_body_names = self.gym.get_actor_rigid_body_names(
                self.envs[0], i)
            self.rigid_body_state_dict[name] = {}
            self.dof_dict[name] = {}

            self.dof_dict[name]["position"] = \
                dof_state[:, dof_count:dof_count + actor_dof_counts[i], 0]
            self.dof_dict[name]["velocity"] = \
                dof_state[:, dof_count:dof_count + actor_dof_counts[i], 1]
            dof_count += actor_dof_counts[i]

            for j, body in enumerate(rigid_body_names):
                self.rigid_body_state_dict[name][body] = {}
                t = self.rigid_body_states
                self.rigid_body_state_dict[name][body]["position"] = \
                    self.rigid_body_states[:, j + rb_count, 0:3]
                self.rigid_body_state_dict[name][body]["rotation"] = \
                    self.rigid_body_states[:, j + rb_count, 3:7]
                self.rigid_body_state_dict[name][body]["linear_velocity"] = \
                    self.rigid_body_states[:, j + rb_count, 7:10]
                self.rigid_body_state_dict[name][body][
                    "rotational_velocity"] = self.rigid_body_states[
                                             :, j + rb_count, 10:13]
            rb_count += actor_rigid_body_counts[i]

            # Jacobian matrices
            if actor_dof_counts[i] > 0:
                jacobian_tensor = self.gym.acquire_jacobian_tensor(
                    self.sim, name)
                jacobian = gymtorch.wrap_tensor(jacobian_tensor)
                self.jacobian_dict[name] = jacobian

        self._objects_dropped = False

        self.successes = torch.zeros(self.num_envs, dtype=torch.float,
                                     device=self.device)
        self.success_rate = torch.zeros(1, dtype=torch.float,
                                        device=self.device)

        self.actions = torch.zeros(self.num_envs, self.cfg["env"]["numActions"],
                                   device=self.device)
        self.prev_actions = self.actions.clone()

        # haptics
        self.prev_force_magnitude = torch.zeros(self.num_envs, 5).to(self.device)
        self.low_pass_horizon_len = int(self.cfg["haptics"]["buzz"]["low_pass_horizon"] / (self.dt))
        self.past_force_magnitudes = torch.zeros(5, self.low_pass_horizon_len - 1).to(self.device)
        self.past_force_directions = torch.zeros(5, 3, self.low_pass_horizon_len - 1).to(self.device)
        self.prev_effective_force = torch.zeros(self.num_envs, 5).to(self.device)
        self.prev_smoothed_force_increase = torch.zeros(self.num_envs, 5).to(self.device)
        self.finger_collision = torch.zeros(self.num_envs, 5).to(self.device)  # reports finger collision forces
        self.finger_in_contact_for = torch.zeros((self.num_envs, 5), dtype=torch.long).to(self.device)  # counts the time-steps for how long a finger has been in contact
        self.collision_vibration_time_steps = int(
            self.cfg["haptics"]["buzz"]["collision_vibration_time"] / (
                    self.dt * self.control_freq_inv))

        # buffers for prev targets needed for relative control
        if self.cfg["control"]["useRelativeControl"]:
            self._init_relative_targets()

    def _acquire_contact_force_tensor(self) -> None:
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(
            self.num_envs, -1, 3) # (num_envs, num_bodies, xyz)

    def _acquire_force_sensor_tensor(self) -> None:
        num_force_sensors = self.num_fingertips
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, num_force_sensors, 6)
        self.sensor_forces = self.force_sensor_readings[..., :3]
        self.sensor_torques = self.force_sensor_readings[..., 3:]

    def _init_relative_targets(self) -> None:
        self.prev_hand_pose_targets = self.tracker_pose
        self.prev_hand_finger_targets = self.prev_targets[
                              :, self.hand_actuated_dof_indices]

    def _reset_relative_targets(self, env_ids) -> None:
        if len(env_ids) > 0:
            self.prev_hand_pose_targets[env_ids] = \
                self.initial_tracker_pose[env_ids]

            self.prev_hand_finger_targets[env_ids] = \
                self.initial_finger_angles[env_ids]

    @property
    def eef_pos(self) -> torch.Tensor:
        return self.rigid_body_state_dict["robot"]["wrist_3_link"]["position"]

    @property
    def tracker_pos(self) -> torch.Tensor:
        tracker_pos = self.eef_pos + quat_apply(
            self.tracker_rot, torch.Tensor([[0, 0.075, 0.05]]).repeat(
                self.num_envs, 1).to(self.device))
        return tracker_pos

    @property
    def tracker_rot(self) -> torch.Tensor:
        tracker_rot = quat_mul(self.eef_rot, torch.Tensor([[0.707, 0., 0., 0.707]]).repeat(self.num_envs, 1).to(self.device))
        return tracker_rot

    @property
    def tracker_rot_euler(self) -> torch.Tensor:
        return torch.stack(get_euler_xyz(self.tracker_rot)).transpose(0, 1)

    @property
    def tracker_pose(self) -> torch.Tensor:
        return torch.cat([self.tracker_pos, self.tracker_rot_euler], dim=-1)

    @property
    def eef_rot(self) -> torch.Tensor:
        return self.rigid_body_state_dict["robot"]["wrist_3_link"]["rotation"]

    @property
    def eef_rot_euler(self) -> torch.Tensor:
        return torch.stack(get_euler_xyz(self.eef_rot)).transpose(0, 1)

    @property
    def eef_pose(self) -> torch.Tensor:
        return torch.cat([self.eef_pos, self.eef_rot_euler], dim=-1)

    def _reset_robot(self, env_ids, default_dof_pos: torch.Tensor = None,
                     apply_dof_resets: bool = True, initial_sampling: bool = False) -> None:
        """Resets the robot DOFs to the initial position plus randomized
        delta position and delta velocity."""
        if default_dof_pos is None:
            default_dof_pos = self.robot_dof_default_pos.unsqueeze(0).repeat(len(env_ids), 1)

        if initial_sampling:
            # Generate random values
            rand_floats = torch_rand_float(
                -1.0, 1.0, (self.num_envs, self.num_robot_dofs),
                device=self.device)

            # Reset position and velocity of robot DOFs
            delta_max = self.robot_dof_upper_limits - default_dof_pos
            delta_min = self.robot_dof_lower_limits - default_dof_pos
            rand_delta = delta_min + (delta_max - delta_min) * \
                         rand_floats[:, :self.num_robot_dofs]
            pos = default_dof_pos + \
                  self.cfg["initState"]["noise"]["robotDofPos"] * rand_delta

            self.initial_robot_dof_pos = pos.clone()
            self.initial_robot_dof_vel = self.robot_dof_default_vel.unsqueeze(0).repeat(self.num_envs, 1).clone()

        self.robot_dof_pos[env_ids] = self.initial_robot_dof_pos[env_ids]
        self.robot_dof_vel[env_ids] = self.initial_robot_dof_vel[env_ids]


        self.prev_targets[env_ids, :self.num_robot_dofs] = self.initial_robot_dof_pos[env_ids]
        self.cur_targets[env_ids, :self.num_robot_dofs] = self.initial_robot_dof_pos[env_ids]

        # Set DOF state and target tensors
        robot_indices = self.robot_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices), len(env_ids))

        if apply_dof_resets:
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(robot_indices), len(env_ids))

        self.prev_actions[env_ids, :] = 0.

    def pre_physics_step(self, actions):
        self.env_steps += self.num_envs
        overwrite_actions = False
        if overwrite_actions:
            if self.progress_buf[0] < 150:
                actions = torch.Tensor(
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(
                    self.num_envs, 1).to(self.device)
            else:
                actions = torch.Tensor(
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(
                   self.num_envs, 1).to(self.device)

        actions = actions.to(self.device)

        self._reset_terminated_envs()

        hand_pose_targets, hand_joint_targets = self._process_actions(actions)

        # Compute robot arm joint targets via inverse kinematics
        arm_joint_targets = self._ik_hand_pose(hand_pose_targets)

        if self.cfg["control"]["useRelativeControl"]:
            self._apply_relative_controls(arm_joint_targets, hand_joint_targets)
        else:
            self._apply_absolute_controls(arm_joint_targets, hand_joint_targets)

        # Update previous targets
        self.prev_targets[:, self.actuated_dof_indices] = \
            self.cur_targets[:, self.actuated_dof_indices]

        # Enforce joint couplings of SIH hand
        self.cur_targets = self._couple_joints(self.cur_targets)

        # Set joint targets in the simulation
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def _reset_terminated_envs(self) -> None:
        # Reset any environments that terminated
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            if "cameras" in self.cfg.keys():
                if self.cfg["cameras"]["save_recordings"]:
                    self.reset_recordings_idx(env_ids)

            # Reset relative control targets if needed
            if self.cfg["control"]["useRelativeControl"]:
                self._reset_relative_targets(env_ids)

    def _process_actions(self, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.actions[:] = actions.clone().to(self.device)

        assert torch.all(actions <= 1.) and torch.all(actions >= -1.), \
            "Expected all actions to be in [-1, 1]."

        assert actions.shape[1] == 11, \
            "Expected actions to be 11 dimensional with 6 dimensions " \
            "controlling the EEF-pose and 5 dimensions controlling the " \
            "SIH-hand joint targets."

        hand_pose_targets = actions[:, 0:6].clone()

        hand_joint_targets = actions[:, 6:11]
        if self.cfg["control"]["useRelativeControl"]:
            # hand_pose_targets = prev_targets + Δt * β * hand_pose_targets
            hand_pose_targets *= self.dt
            hand_pose_targets[:, 0:3] *= \
                self.cfg["control"]["scale"]["relativePosSpeed"]
            hand_pose_targets[:, 3:6] *= \
                self.cfg["control"]["scale"]["relativeRotSpeed"]
            hand_pose_targets += self.prev_hand_pose_targets

            if self.cfg["control"]["clampActionDeviation"]:
                # Clamp hand_pose_targets not to deviate to far from the
                # current pose
                hand_pos_targets = hand_pose_targets[:, 0:3]
                hand_rot_targets = hand_pose_targets[:, 3:6]

                # Clamp target_pose around current_pose
                hand_pos_targets = tensor_clamp(
                    hand_pos_targets,
                    self.tracker_pos - self.hand_pose_max_deviation[0:3],
                    self.tracker_pos + self.hand_pose_max_deviation[0:3])

                euler_rot_err = hand_rot_targets - self.tracker_rot_euler

                euler_rot_err = torch.where(euler_rot_err < -np.pi,
                                            euler_rot_err + 2*np.pi,
                                            euler_rot_err)
                euler_rot_err = torch.where(euler_rot_err > np.pi,
                                            euler_rot_err - 2*np.pi,
                                            euler_rot_err)
                # clamp euler rotation error
                euler_rot_err = tensor_clamp(euler_rot_err,
                                             -self.hand_pose_max_deviation[3:6],
                                             self.hand_pose_max_deviation[3:6])
                hand_rot_targets = self.tracker_rot_euler + euler_rot_err

                # put hand_rot_targets in [-pi, pi]
                hand_rot_targets = torch.where(hand_rot_targets < -np.pi,
                                               hand_rot_targets + 2 * np.pi,
                                               hand_rot_targets)
                hand_rot_targets = torch.where(hand_rot_targets > np.pi,
                                               hand_rot_targets - 2 * np.pi,
                                               hand_rot_targets)

                hand_pose_targets = torch.cat(
                    [hand_pos_targets, hand_rot_targets], dim=1)

            # Clamp hand pose to remain in allowed workspace
            hand_pose_targets = tensor_clamp(hand_pose_targets,
                                             self.hand_pose_lower_limits,
                                             self.hand_pose_upper_limits)

            self.prev_hand_pose_targets = hand_pose_targets

            # hand_joint_targets = prev_targets + Δt * β * hand_joint_targets
            hand_joint_targets *= \
                self.cfg["control"]["scale"]["handJointSpeed"] * self.dt
            hand_joint_targets += self.prev_targets[
                                  :, self.hand_actuated_dof_indices]
            self.prev_hand_finger_targets = hand_joint_targets

            if self.cfg["control"]["clampActionDeviation"]:
                # Clamp hand_joint_targets not to deviate to far from current pos
                hand_joint_pos = self.robot_dof_pos[:,
                                 self.hand_actuated_dof_indices]
                hand_joint_targets = tensor_clamp(
                    hand_joint_targets,
                    hand_joint_pos - self.hand_angle_max_deviation,
                    hand_joint_pos + self.hand_angle_max_deviation)
        else:
            assert False
            hand_pose_targets = scale(hand_pose_targets,
                                      self.hand_pose_lower_limits,
                                      self.hand_pose_upper_limits)

            if self.cfg["control"]["clampActionDeviation"]:
                hand_pos_targets = hand_pose_targets[:, 0:3]
                hand_rot_targets = hand_pose_targets[:, 3:6]

                # Clamp position deviation
                hand_pos_targets = tensor_clamp(
                    hand_pos_targets,
                    self.hand_pos - self.hand_pose_max_deviation[0:3],
                    self.hand_pos + self.hand_pose_max_deviation[0:3])

                hand_pose_targets = torch.cat([hand_pos_targets, hand_rot_targets], dim=1)

                import time
                #time.sleep(1000)
                # Clamp hand_pose_targets not to deviate to far from current pose
                #hand_pose_targets = tensor_clamp(
                #    hand_pose_targets,
                #    self.eef_pose - self.hand_pose_max_deviation,
                #    self.eef_pose + self.hand_pose_max_deviation)

            hand_joint_targets = scale(
                hand_joint_targets,
                self.robot_dof_lower_limits[self.hand_actuated_dof_indices],
                self.robot_dof_upper_limits[self.hand_actuated_dof_indices])

            if self.cfg["control"]["clampActionDeviation"]:
                # Clamp hand_joint_targets not to deviate to far from current pos
                hand_joint_pos = self.robot_dof_pos[:, self.hand_actuated_dof_indices]
                hand_joint_targets = tensor_clamp(
                    hand_joint_targets,
                    hand_joint_pos - self.hand_angle_max_deviation,
                    hand_joint_pos + self.hand_angle_max_deviation)
        return hand_pose_targets, hand_joint_targets

    def _ik_hand_pose(self, hand_pose_targets: torch.Tensor):
        hand_target_pos = hand_pose_targets[:, 0:3]
        hand_target_rot_euler = hand_pose_targets[:, 3:6]
        hand_target_rot = quat_from_euler_xyz(
            hand_target_rot_euler[:, 0], hand_target_rot_euler[:, 1],
            hand_target_rot_euler[:, 2])

        pos_err = 0.5 * (hand_target_pos - self.tracker_pos)
        rot_err = self.quat_rot_error(hand_target_rot, self.tracker_rot)

        delta_pose = torch.cat([pos_err, rot_err], dim=1).unsqueeze(2)
        self.gym.refresh_jacobian_tensors(self.sim)
        j_eef = self.jacobian_dict["robot"][:, 5]

        j_eef_T = torch.transpose(j_eef, 1, 2)
        ls_dmp = 0.05  # least squares damping
        lmbda = torch.eye(6, device=self.device) * (ls_dmp ** 2)
        eef_joint_targets = (j_eef_T @ torch.inverse(
            j_eef @ j_eef_T + lmbda) @ delta_pose).view(
            self.num_envs, self.num_robot_dofs)

        eef_joint_pos = self.dof_dict["robot"]["position"]
        eef_joint_targets += eef_joint_pos
        return eef_joint_targets[:, 0:6]

    @staticmethod
    def quat_rot_error(target, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(target, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def _apply_relative_controls(self, arm_joint_targets: torch.Tensor,
                                 hand_joint_targets: torch.Tensor) -> None:
        # Make sure joint targets obey the joint limits
        joint_targets = torch.cat([arm_joint_targets, hand_joint_targets],
                                  dim=1)
        self.cur_targets[:, self.actuated_dof_indices] = \
            tensor_clamp(joint_targets,
                         self.robot_dof_lower_limits[self.actuated_dof_indices],
                         self.robot_dof_upper_limits[self.actuated_dof_indices])

    def _apply_absolute_controls(self, arm_joint_targets: torch.Tensor,
                                 hand_joint_targets: torch.Tensor) -> None:
        # Set current targets to joint targets inferred from actions
        joint_targets = torch.cat([arm_joint_targets, hand_joint_targets],
                                  dim=1)
        self.cur_targets[:, self.actuated_dof_indices] = joint_targets

        # Apply moving average on joint targets
        self.cur_targets[:, self.actuated_dof_indices] = \
            self.cfg["control"]["actionMovingAverage"] * \
            self.cur_targets[:, self.actuated_dof_indices] + \
            (1.0 - self.cfg["control"]["actionMovingAverage"]) * \
            self.prev_targets[:, self.actuated_dof_indices]

        # Make sure joint targets obey the joint limits
        self.cur_targets[:, self.actuated_dof_indices] = \
            tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                         self.robot_dof_lower_limits[self.actuated_dof_indices],
                         self.robot_dof_upper_limits[self.actuated_dof_indices])

    def _couple_joints(self, targets: torch.Tensor) -> torch.Tensor:
        parent_idxs = self.hand_coupled_joint_indices[:, 0]
        child_idxs = self.hand_coupled_joint_indices[:, 1]
        x = targets[:, parent_idxs]
        c0, c1, c2, c3 = self.hand_coupled_joint_vals[:, 0], \
                         self.hand_coupled_joint_vals[:, 1], \
                         self.hand_coupled_joint_vals[:, 2], \
                         self.hand_coupled_joint_vals[:, 3]
        targets[:, child_idxs] = c0 + c1 * x + c2 * x.pow(2) + c3 * x.pow(3)
        return targets

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward()

        if "cameras" in self.cfg.keys():
            self._render_camera_sensors()
            if self.convert_to_pointcloud:
                self._convert_rgbd_to_pointcloud()
            if self.convert_to_voxelgrid:
                self._convert_pointcloud_to_voxelgrid()
            if self.cfg["cameras"]["save_recordings"]:
                self._write_recordings()

        if self.cfg["debug"]["visualization"]:
            self.gym.clear_lines(self.viewer) and (self.viewer or self.cfg["debug"].get("showInCamera", False))
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self._draw_debug_visualization(i)

    def compute_observations(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def _refresh_state_tensors(self) -> None:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.cfg["sim"]["useContactForces"]:
            self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.cfg["sim"]["useForceSensors"]:
            self.gym.refresh_force_sensor_tensor(self.sim)

    def _process_state_tensors(self) -> None:
        self.fingertip_state = self.rigid_body_states[
                               :, self.fingertip_handles][:, :, 0:13]
        self.fingertip_rigid_body_pos = self.rigid_body_states[
                                        :, self.fingertip_handles][:, :, 0:3]
        self.fingertip_rot = self.rigid_body_states[
                             :, self.fingertip_handles][:, :, 3:7]
        self.fingertip_pos = self.calculate_fingertip_pos()

    def calculate_fingertip_pos(self) -> torch.Tensor:
        """Converts the default positions of the rigid bodies of the fingertips
        to the actual positions to be used in the observation."""
        fingertip_pos = self.fingertip_rigid_body_pos.clone()
        fingertip_pos += quat_apply(
            self.fingertip_rot, torch.Tensor([[[-0.0200, 0.0000, 0.0150],
                                               [-0.0870, 0.0175, 0.0400],
                                               [-0.0375, 0.0175, 0.0490],
                                               [ 0.0140, 0.0175, 0.0410],
                                               [ 0.0640, 0.0170, 0.0340]]]
                                             ).repeat(self.num_envs, 1, 1
                                                      ).to(self.device))
        return fingertip_pos
    
    def add_separate_obs(self, key, data):
        self.obs_separate[key] = data

    def add_fingertip_contact_forces(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx + 15] = \
            self.contact_forces[:, self.fingertip_actor_rb_handle].reshape(
            self.num_envs, 3 * self.num_fingertips)
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs(
                "fingertipContactForces",
                self.contact_forces[:, self.fingertip_actor_rb_handle].reshape(self.num_envs, 3 * self.num_fingertips))

    def add_fingertip_pos(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx+15] = self.fingertip_pos.reshape(
            self.num_envs, 3 * self.num_fingertips)
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs(
                "fingertipPos",
                self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips))

    def add_hand_pose(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx+3] = self.eef_pos
        self.obs_buf[:, start_idx+3:start_idx+7] = self.eef_rot
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("handPose", torch.cat([self.eef_pos, self.eef_rot], dim=1))

    def add_joint_pos(self, start_idx: int,
                      use_unscale: bool = True) -> None:
        if use_unscale:
            data = \
                unscale(self.robot_dof_pos, self.robot_dof_lower_limits,
                        self.robot_dof_upper_limits)
        else:
            data = \
                self.robot_dof_pos
        self.obs_buf[:, start_idx:start_idx + self.num_robot_dofs] = data
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("jointPos", data)
    
    def add_joint_vel(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx + self.num_robot_dofs] = \
            self.robot_dof_vel
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("jointVel", self.robot_dof_vel)

    def add_hand_dof_pos(self, start_idx: int,
                      use_unscale: bool = True) -> None:
        # Only adds the 5 degrees of freedom of the SIH hand to the observation
        if use_unscale:
            data = \
                unscale(self.robot_dof_pos[:, self.hand_actuated_dof_indices],
                        self.robot_dof_lower_limits[self.hand_actuated_dof_indices],
                        self.robot_dof_upper_limits[self.hand_actuated_dof_indices])
        else:
            data = \
                self.robot_dof_pos[:, self.hand_actuated_dof_indices]
        self.obs_buf[:, start_idx:start_idx + 5] = data
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("handDofPos", data) 

    def add_previous_action(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx+self.cfg["env"]["numActions"]] = \
            self.actions
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("previousAction", self.actions)

    def get_state(self):
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(
            self.rl_device)

    def step(
            self,
            actions: torch.Tensor
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any]]:
        """Steps the environment.

        Args:
            actions: Actions to apply to the environments
        Returns:
            obs, rew, done, info
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.step_simulation(self.gym, self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # fill time out buffer
        self.timeout_buf = torch.where \
            (self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if "cameras" in self.cfg.keys():
            self.obs_dict["image"] = self.image_buf.to(self.rl_device)
        
        if self.cfg["task"]["returnObsDict"]:
            self.obs_dict["obs_separate"] = self.obs_separate
        
        if self.cfg["reward"]["returnRewardsDict"]:
            self.obs_dict["rewards"] = self.rewards_dict

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to \
            (self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def zero_actions(self) -> torch.Tensor:
        actions = torch.zeros([self.num_envs, self.num_actions],
                              dtype=torch.float32, device=self.rl_device)
        return actions

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments.

        Returns:
            obs
        """
        zero_actions = self.zero_actions()

        self._refresh_state_tensors()
        self._process_state_tensors()

        # step the simulator
        self.step(zero_actions)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs,
                                           self.clip_obs).to(self.rl_device)
        if "cameras" in self.cfg.keys():
            self.obs_dict["image"] = self.image_buf.to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        self.initial_reset = False
        return self.obs_dict

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

    def _draw_debug_visualization(self, i: int) -> None:
        if self.cfg["debug"]["drawEefPose"]:
            self._draw_eef_pose(i)
        if self.cfg["debug"]["drawTrackerPose"]:
            self._draw_tracker_pose(i)
        if self.cfg["debug"]["drawFingertipPose"]:
            self._draw_fingertip_pose(i)
        if self.cfg["debug"]["drawFingertipContactForces"]:
            self._draw_fingertip_contact_forces(i)
        if self.cfg["debug"]["colorFingertipContactForce"]:
            self._color_fingertip_contact_force(i)

    def _draw_coordinate_system(self, env, pos, rot, axis_length: float = 0.15
                                ) -> None:
        targetx = (pos + quat_apply(rot, to_torch(
            [1, 0, 0], device=self.device) * axis_length)).cpu().numpy()
        targety = (pos + quat_apply(rot, to_torch(
            [0, 1, 0], device=self.device) * axis_length)).cpu().numpy()
        targetz = (pos + quat_apply(rot, to_torch(
            [0, 0, 1], device=self.device) * axis_length)).cpu().numpy()
        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1,
                           [p0[0], p0[1], p0[2], targetx[0], targetx[1],
                            targetx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1,
                           [p0[0], p0[1], p0[2], targety[0], targety[1],
                            targety[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1,
                           [p0[0], p0[1], p0[2], targetz[0], targetz[1],
                            targetz[2]], [0.1, 0.1, 0.85])

    def _draw_arrow(self, env, base, head,
                    color: List[float] = [0.9, 0., 0.],
                    num_arrowhead_lines: int = 16) -> None:
        base = base.cpu().numpy()
        head = head.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1,
                           [base[0], base[1], base[2],
                            head[0], head[1], head[2]], color)
        arrow = np.array(head - base)

        ah_base = base + 0.8 * arrow

        arrow_dir = arrow / np.linalg.norm(arrow)

        v = np.array([1., 0, 0])
        if np.linalg.norm(v - arrow_dir) < 0.1:
            v = np.array([0., 1., 0.])

        n1 = np.cross(v, arrow_dir)
        n2 = np.cross(n1, arrow_dir)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        ah_width = 0.01
        prev_ah_target = None
        for i in range(num_arrowhead_lines):

            rot_angle = (2 * np.pi) * (i / num_arrowhead_lines)
            ah_target = ah_base + ah_width * (np.sin(rot_angle) * n1 + np.cos(rot_angle) * n2)
            if i == 0:
                first_ah_target = ah_target
            self.gym.add_lines(self.viewer, env, 1,
                               [ah_target[0], ah_target[1], ah_target[2],
                                head[0], head[1], head[2]], color)
            if prev_ah_target is not None:
                self.gym.add_lines(self.viewer, env, 1,
                                   [ah_target[0], ah_target[1], ah_target[2],
                                    prev_ah_target[0], prev_ah_target[1],
                                    prev_ah_target[2]], color)
            prev_ah_target = ah_target

            if i == num_arrowhead_lines - 1:
                self.gym.add_lines(self.viewer, env, 1,
                               [ah_target[0], ah_target[1], ah_target[2],
                                first_ah_target[0], first_ah_target[1],
                                first_ah_target[2]], color)

    def _draw_eef_pose(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.eef_pos[i],
                                     self.eef_rot[i])

    def _draw_hand_pose(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.hand_pos[i],
                                     self.eef_rot[i])

    def _draw_tracker_pose(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.tracker_pos[i],
                                     self.tracker_rot[i])

    def _draw_fingertip_pose(self, i: int) -> None:
        for ft_idx in range(self.num_fingertips):
            self._draw_coordinate_system(self.envs[i],
                                         self.fingertip_pos[i, ft_idx],
                                         self.fingertip_rot[i, ft_idx],
                                         axis_length=0.05)

    def _draw_fingertip_contact_forces(self, i: int,
                                       force_scaling: float = 0.1,
                                       sqrt_scaling: bool = True) -> None:
        ft_pos = self.fingertip_pos[i].cpu().numpy()
        ft_contact_force = self.contact_forces[
            i, self.fingertip_actor_rb_handle].cpu().numpy()

        if sqrt_scaling:
            for ft_idx in range(self.num_fingertips):
                force_magnitude = np.linalg.norm(ft_contact_force[ft_idx])
                ft_contact_force[ft_idx] *= (np.sqrt(force_magnitude) / (force_magnitude + 1e-8))

        target_pos = ft_pos + force_scaling * ft_contact_force
        for ft_idx in range(self.num_fingertips):
            self.gym.add_lines(self.viewer, self.envs[i], 1,
                               [ft_pos[ft_idx, 0], ft_pos[ft_idx, 1], ft_pos[ft_idx, 2],
                                target_pos[ft_idx, 0], target_pos[ft_idx, 1], target_pos[ft_idx, 2]],
                               [0.55, 0., 0.])

    def _color_fingertip_contact_force(self, i: int,
                                       start_color: np.ndarray = np.array([0., 0.8, 0.]),
                                       end_color: np.ndarray = np.array([0.8, 0., 0.]),
                                       max_force: float = 5.,
                                       sqrt_scaling: bool = True) -> None:
        ft_contact_force = self.contact_forces[
            i, self.fingertip_actor_rb_handle].cpu().numpy()
        for ft_idx in range(self.num_fingertips):
            force_magnitude = np.linalg.norm(ft_contact_force[ft_idx])
            if sqrt_scaling:
                force_magnitude = np.sqrt(force_magnitude)
            force_ratio = min(force_magnitude / max_force, 1)
            color = start_color + force_ratio * (end_color - start_color)
            self.gym.set_rigid_body_color(
                self.envs[i], self.robots[i],
                self.fingertip_handles[ft_idx], gymapi.MeshType.MESH_VISUAL,
                gymapi.Vec3(*tuple(color)))
