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

import math
import os

from PIL import Image as Im
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from rofunc.learning.RofuncRL.tasks.isaacgymenv.base.vec_task import VecTask
from rofunc.learning.RofuncRL.tasks.utils.torch_jit_utils import *
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.utils.oslab import get_rofunc_path


class CURIQbSoftHandSynergyGraspTask(VecTask):
    """
    This class corresponds to the GraspAndPlace task. This environment consists of dual-hands, an
    object and a bucket that requires us to pick up the object and put it into the bucket.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture,
                 force_render, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        # assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            # "pot": "mjcf/pot.xml",
            "pot": "mjcf/bucket/100454/mobility.urdf",
            "power_drill": "urdf/ycb/035_power_drill/035_power_drill.urdf",
            "hammer": "urdf/ycb/048_hammer/048_hammer.urdf",
            "large_clamp": "urdf/ycb/051_large_clamp/051_large_clamp.urdf",
            "spatula": "urdf/ycb/033_spatula/033_spatula.urdf",
            "wine_glass": "urdf/ycb/023_wine_glass/023_wine_glass.urdf",
            "mug": "urdf/ycb/025_mug/025_mug.urdf",
            "knife": "urdf/ycb/032_knife/032_knife.urdf",
            "scissors": "urdf/ycb/037_scissors/037_scissors.urdf",
            "phillips_screw_driver": "urdf/ycb/043_phillips_screwdriver/043_phillips_screwdriver.urdf",
            "large_marker": "urdf/ycb/040_large_marker/040_large_marker.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock",
                                                                          self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg",
                                                                        self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen",
                                                                        self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)

        # <editor-fold desc="obs space">
        self.num_point_cloud_feature_dim = 768

        if self.cfg["env"]["useSynergy"]:
            self.num_hand_obs = 95 * 3 + 6 + 8
        else:
            self.num_hand_obs = 95 * 3 + 6 + 21
        num = 13 + self.num_hand_obs
        self.num_obs_dict = {
            "point_cloud": num + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": num + self.num_point_cloud_feature_dim * 3,
            "full_state": num
        }
        self.up_axis = 'z'
        # </editor-fold>

        self.fingertips = ["left_qbhand_thumb_distal_link", "left_qbhand_index_distal_link",
                           "left_qbhand_middle_distal_link", "left_qbhand_ring_distal_link",
                           "left_qbhand_little_distal_link"]
        # self.fingertips = ["qbhand_thumb_distal_link", "qbhand_index_distal_link",
        #                    "qbhand_middle_distal_link", "qbhand_ring_distal_link",
        #                    "qbhand_little_distal_link"]

        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        self.num_agents = 1
        if self.cfg["env"]["useSynergy"]:
            self.cfg["env"]["numActions"] = 2 + 7  # 2-dim synergy for controlling each hand
        else:
            self.cfg["env"]["numActions"] = 15 + 7  # 15-dim dof for controlling each hand
        self.num_action = self.cfg["env"]["numActions"]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # <editor-fold desc="dof reduction">
        actor_joint_dict = self.gym.get_actor_joint_dict(self.envs[0], 0)
        self.useful_joint_index = sorted(
            [value for key, value in actor_joint_dict.items() if "left_qbhand" in key if
             ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
             ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key)])
        self.real_virtual_joint_index_map_dict = {value: actor_joint_dict[key.replace("_virtual", "")] for
                                                  key, value in actor_joint_dict.items() if "virtual" in key
                                                  if "left_qbhand" in key}

        # </editor-fold>
        actor_rigid_body_dict = self.gym.get_actor_rigid_body_dict(self.envs[0], 0)
        self.fingertips_index = sorted([actor_rigid_body_dict[fingertip] for fingertip in self.fingertips])

        if self.obs_type in ["point_cloud"]:
            from PIL import Image as Im
            # from pointnet2_ops import pointnet2_utils

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 93 = 91 + 1 + 1

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                                            self.num_hand_dofs + self.num_object_dofs * 2)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "hand")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.hand_default_dof_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        # self.hand_default_dof_pos[:7] = to_torch(self.hand_dof_lower_limits[:7] + self.hand_dof_upper_limits[:7],
        #                                          dtype=torch.float, device=self.device) / 2.0

        if self.object_type in ["power_drill", "mug", "large_marker"]:
            self.hand_default_dof_pos[:7] = torch.tensor([0.0905, 0.5326, 0.0486, -1.5469, -0.9613, 2.2102, 1.5221]).to(
                self.device)
        elif self.object_type in ["hammer", "large_clamp", "spatula", "phillips_screw_driver", "scissors", "knife"]:
            self.hand_default_dof_pos[:7] = torch.tensor(
                [-0.6654, 0.5872, 0.9190, -1.4767, -2.4621, 3.2423, 2.8480]).to(
                self.device)
        self.hand_default_dof_pos[33 + 7:33 + 14] = to_torch(self.hand_dof_lower_limits[33 + 7:33 + 14]
                                                             + self.hand_dof_upper_limits[33 + 7:33 + 14],
                                                             dtype=torch.float, device=self.device) / 2.0
        # self.shadow_hand_default_dof_pos = to_torch([0.0, 0.0, -0,  -0,  -0,  -0, -0, -0,
        #                                     -0,  -0, -0,  -0,  -0,  -0, -0, -0,
        #                                     -0,  -0, -0,  -1.04,  1.2,  0., 0, -1.57], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_hand_dofs]
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:,
                                self.num_hand_dofs:self.num_hand_dofs + self.num_object_dofs]
        self.object_dof_pos = self.object_dof_state[..., 0]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_synergy_actions = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.prev_goal_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.prev_dof_action = -torch.ones((self.num_envs, 15)).to(self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,
                                                                                                          -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

        # self.attractor_handles, self.axes_geoms, self.sphere_geoms = self._create_attractor("panda_left_link7")
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

    def create_sim(self):
        """
        Allocates which device will simulate and which device will render the scene. Defines the simulation type to be used
        """
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        l_color = gymapi.Vec3(1, 1, 1)
        l_ambient = gymapi.Vec3(0.3, 0.3, 0.3)
        l_direction = gymapi.Vec3(-1, 0, 1)
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

    def _create_ground_plane(self):
        """
        Adds ground plane to simulation
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Create multiple parallel isaacgym environments

        :param num_envs: The total number of environment
        :param spacing: Specifies half the side length of the square area occupied by each environment
        :param num_per_row: Specify how many environments in a row
        :return:
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # get rofunc path from rofunc package metadata
        rofunc_path = get_rofunc_path()
        asset_root = os.path.join(rofunc_path, "simulator/assets")
        hand_asset_file = "urdf/curi/urdf/curi_isaacgym_dual_arm_w_softhand.urdf"
        table_texture_files = os.path.join(asset_root, "textures/texture_stone_stone_texture_0.jpg")
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)

        object_asset_file = self.asset_files_dict[self.object_type]

        # <editor-fold desc="load hand asset and set hand dof properties">
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)

        # print asset info
        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        self.num_hand_actuators = self.gym.get_asset_actuator_count(hand_asset)
        self.num_hand_tendons = self.gym.get_asset_tendon_count(hand_asset)
        beauty_print(f"self.num_hand_bodies: {self.num_hand_bodies}")
        beauty_print(f"self.num_hand_shapes: {self.num_hand_shapes}")
        beauty_print(f"self.num_hand_dofs: {self.num_hand_dofs}")
        beauty_print(f"self.num_hand_actuators: {self.num_hand_actuators}")
        beauty_print(f"self.num_hand_tendons: {self.num_hand_tendons}")

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(hand_asset, i) for i in
                              range(self.num_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(hand_asset, name) for name in
                                     actuated_dof_names]

        # set shadow_hand dof properties
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        # hand_dof_props['driveMode'] = gymapi.DOF_MODE_POS
        # hand_dof_props['stiffness'] = 1000000.0
        # hand_dof_props['damping'] = 1000.0

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []
        self.hand_dof_default_pos = []
        self.hand_dof_default_vel = []
        self.sensors = []

        for i in range(self.num_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props['lower'][i])
            self.hand_dof_upper_limits.append(hand_dof_props['upper'][i])
            self.hand_dof_default_pos.append(0.0)
            self.hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)
        self.hand_dof_default_pos = to_torch(self.hand_dof_default_pos, device=self.device)
        self.hand_dof_default_vel = to_torch(self.hand_dof_default_vel, device=self.device)
        # </editor-fold>

        # <editor-fold desc="load manipulated object and goal assets">
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 100
        object_asset_options.fix_base_link = False
        # object_asset_options.collapse_fixed_joints = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.disable_gravity = False
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # object_asset_options.override_com = True
        # object_asset_options.override_inertia = True
        # # Enable convex decomposition
        # object_asset_options.vhacd_enabled = True
        # object_asset_options.vhacd_params = gymapi.VhacdParams()
        # object_asset_options.vhacd_params.resolution = 1000000

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        # block_asset_file = "urdf/objects/cube_multicolor1.urdf"
        # block_asset = self.gym.load_asset(self.sim, asset_root, block_asset_file, object_asset_options)

        # object_asset_options.disable_gravity = True

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        # set object dof properties
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
        # </editor-fold>

        # <editor-fold desc="create table asset">
        table_dims = gymapi.Vec3(1.0, 1.0, 0.6)
        if self.object_type in ["mug", "large_marker"]:
            table_dims = gymapi.Vec3(1.0, 1.0, 0.65)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        # </editor-fold>

        # block_asset = self.gym.create_box(self.sim, table_dims.x / 10, table_dims.y / 10, 0.05, gymapi.AssetOptions())

        # <editor-fold desc="set initial poses">
        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(1.25, 0.0, 0.0)
        hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 3.14)  # xyzw (0.0, 0.0, 1.0, 0.0)

        object_start_pose = gymapi.Transform()
        if self.object_type in ["power_drill"]:
            object_start_pose.p = gymapi.Vec3(0.1, -0.05, 0.7)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, -1.57)  # xyzw (0.5, -0.5, -0.5, 0.5)
        elif self.object_type in ["hammer"]:
            object_start_pose.p = gymapi.Vec3(0., -0.15, 0.6)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # xyzw (0.5, -0.5, -0.5, 0.5)
        elif self.object_type in ["large_clamp"]:
            object_start_pose.p = gymapi.Vec3(0.05, -0.2, 0.6)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 3.14)
        elif self.object_type in ["spatula"]:
            object_start_pose.p = gymapi.Vec3(0.15, -0.15, 0.65)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(3.14, 0, 1.57)
        elif self.object_type in ["phillips_screw_driver"]:
            object_start_pose.p = gymapi.Vec3(0., -0.20, 0.6)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.57)
        elif self.object_type in ["scissors"]:
            object_start_pose.p = gymapi.Vec3(0.05, -0.18, 0.6)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        elif self.object_type in ["knife"]:
            # object_start_pose.p = gymapi.Vec3(-0.05, -0.17, 0.6)
            # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)
            object_start_pose.p = gymapi.Vec3(0.05, -0.17, 0.6)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.57)
        elif self.object_type in ["mug"]:
            object_start_pose.p = gymapi.Vec3(0.07, 0.03, 0.65)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.57)
        elif self.object_type in ["large_marker"]:
            object_start_pose.p = gymapi.Vec3(0.14, -0.03, 0.65)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        if self.object_type == "pen":
            object_start_pose.p.z = hand_start_pose.p.z + 0.02

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)
        # </editor-fold>

        # compute aggregate size
        max_agg_bodies = self.num_hand_bodies + 3 * self.num_object_bodies + 1
        max_agg_shapes = self.num_hand_shapes + 3 * self.num_object_shapes + 1

        self.hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.object_indices = []
        self.table_indices = []

        if self.obs_type in ["point_cloud"]:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 768
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

            if self.point_cloud_debug:
                import open3d as o3d
                from bidexhands.utils.o3dviewer import PointcloudVisualizer
                self.pointCloudVisualizer = PointcloudVisualizer()
                self.pointCloudVisualizerInitialized = False
                self.o3d_pc = o3d.geometry.PointCloud()
            else:
                self.pointCloudVisualizer = None

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, 1, 0)

            self.hand_start_states.append(
                [hand_start_pose.p.x, hand_start_pose.p.y, hand_start_pose.p.z,
                 hand_start_pose.r.x, hand_start_pose.r.y, hand_start_pose.r.z,
                 hand_start_pose.r.w, 0, 0, 0, 0, 0, 0])

            hand_dof_props = self.gym.get_actor_dof_properties(env_ptr, hand_actor)
            hand_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            hand_dof_props["stiffness"].fill(10000.0)
            hand_dof_props["damping"].fill(20.0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.gym.set_actor_scale(env_ptr, hand_actor, 1)

            # <editor-fold desc="randomize colors and textures for rigid body">
            # num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, hand_actor)
            # hand_rigid_body_index = [[0], [i for i in range(1, 6)], [i for i in range(6, 13)],
            #                          [i for i in range(13, 20)],
            #                          [i for i in range(20, 27)], [i for i in range(27, 34)]]

            # for n in self.agent_index[0]:
            #     colorx = random.uniform(0, 1)
            #     colory = random.uniform(0, 1)
            #     colorz = random.uniform(0, 1)
            #     for m in n:
            #         for o in hand_rigid_body_index[m]:
            #             self.gym.set_rigid_body_color(env_ptr, hand_actor, o, gymapi.MESH_VISUAL,
            #                                           gymapi.Vec3(colorx, colory, colorz))
            # </editor-fold>

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            if self.object_type in ["mug"]:
                self.gym.set_actor_scale(env_ptr, object_handle, 1.5)
            elif self.object_type in ["power_drill", "hammer", "spatula", "phillips_screw_driver", "knife", "scissors",
                                      "large_marker"]:
                self.gym.set_actor_scale(env_ptr, object_handle, 1.2)
            elif self.object_type in ["large_clamp"]:
                self.gym.set_actor_scale(env_ptr, object_handle, 1)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # block_pose = gymapi.Transform()
            # block_pose.p = gymapi.Vec3(-0.1, 0.0, 0.6)
            # block_pose.r = gymapi.Quat().from_euler_zyx(0., 0., 0.)
            # block_handle = self.gym.create_actor(env_ptr, block_asset, block_pose, "block", i, -1, 0)
            # block_idx = self.gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_SIM)

            # if self.object_type != "block":
            #     self.gym.set_rigid_body_color(
            #         env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.obs_type in ["point_cloud"]:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.25, -0., 1.0),
                                             gymapi.Vec3(-0.24, -0., 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse(
                    (torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),
                                        device=self.device)

                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.hands.append(hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        """
        Compute the reward of all environment. The core function is compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.block_right_handle_pos, self.block_left_handle_pos, 
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        , which we will introduce in detail there

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        if self.object_type in ["power_drill", "hammer", "spatula", "phillips_screw_driver", "knife"]:
            synergy_target = torch.tensor([1, 0]).to(self.device)
        elif self.object_type in ["large_clamp", "scissors", "pen", "mug", "large_marker"]:
            synergy_target = torch.tensor([0.44, 1]).to(self.device)

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[
                                                                                          :], self.consecutive_successes[
                                                                                              :] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot,
            self.rigid_body_states[:, 36, 0:3], self.hand_ff_pos, self.hand_mf_pos,
            self.hand_rf_pos, self.hand_lf_pos, self.hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.prev_synergy_actions,
            synergy_target
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        """
        Compute the observations of all environment. The core function is self.compute_full_state(True), 
        which we will introduce in detail there

        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # <editor-fold desc="hand poses">
        self.hand_pos = self.rigid_body_states[:, 7, 0:3]
        self.hand_rot = self.rigid_body_states[:, 7, 3:7]
        self.hand_pos = self.hand_pos + quat_apply(self.hand_rot,
                                                   to_torch([0, 0, 1], device=self.device).repeat(
                                                       self.num_envs, 1) * 0.08)
        self.hand_pos = self.hand_pos + quat_apply(self.hand_rot,
                                                   to_torch([0, 1, 0], device=self.device).repeat(
                                                       self.num_envs, 1) * -0.02)
        # </editor-fold>

        # <editor-fold desc="right hand finger">
        self.hand_th_pos = self.rigid_body_states[:, self.fingertips_index[0], 0:3]
        self.hand_th_rot = self.rigid_body_states[:, self.fingertips_index[0], 3:7]
        self.hand_th_pos = self.hand_th_pos + quat_apply(self.hand_th_rot,
                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                             self.num_envs, 1) * 0.02)
        self.hand_ff_pos = self.rigid_body_states[:, self.fingertips_index[1], 0:3]
        self.hand_ff_rot = self.rigid_body_states[:, self.fingertips_index[1], 3:7]
        self.hand_ff_pos = self.hand_ff_pos + quat_apply(self.hand_ff_rot,
                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                             self.num_envs, 1) * 0.02)
        self.hand_mf_pos = self.rigid_body_states[:, self.fingertips_index[2], 0:3]
        self.hand_mf_rot = self.rigid_body_states[:, self.fingertips_index[2], 3:7]
        self.hand_mf_pos = self.hand_mf_pos + quat_apply(self.hand_mf_rot,
                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                             self.num_envs, 1) * 0.02)
        self.hand_rf_pos = self.rigid_body_states[:, self.fingertips_index[3], 0:3]
        self.hand_rf_rot = self.rigid_body_states[:, self.fingertips_index[3], 3:7]
        self.hand_rf_pos = self.hand_rf_pos + quat_apply(self.hand_rf_rot,
                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                             self.num_envs, 1) * 0.02)
        self.hand_lf_pos = self.rigid_body_states[:, self.fingertips_index[4], 0:3]
        self.hand_lf_rot = self.rigid_body_states[:, self.fingertips_index[4], 3:7]
        self.hand_lf_pos = self.hand_lf_pos + quat_apply(self.hand_lf_rot,
                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                             self.num_envs, 1) * 0.02)

        # </editor-fold>

        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        """
        Compute the observations of all environment. The observation is composed of three parts:
        the state values of the left and right hands, and the information of objects and target.
        The state values of the left and right hands were the same for each task, including hand
        joint and finger positions, velocity, and force information. The detail 361-dimensional
        observational space as shown in below:

        Index       Description
        0 - 14	    right shadow hand dof position
        15 - 29	    right shadow hand dof velocity
        30 - 44	    right shadow hand dof force
        45 - 109	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        110 - 139	right shadow hand fingertip force, torque (5 x 6)
        140 - 142	right shadow hand base position
        143 - 145	right shadow hand base rotation
        146 - 166	right shadow hand actions
        167 - 181	left shadow hand dof position
        182 - 196	left shadow hand dof velocity
        197 - 211	left shadow hand dof force
        212 - 276	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        277 - 306	left shadow hand fingertip force, torque (5 x 6)
        307 - 309	left shadow hand base position
        310 - 312	left shadow hand base rotation
        313 - 333	left shadow hand actions
        334 - 340	object pose
        341 - 343	object linear velocity
        344 - 346	object angle velocity
        347 - 349	block right handle position
        350 - 353   block right handle rotation
        354 - 356	block left handle position
        357 - 360	block left handle rotation
        """
        # num_ft_states = 13 * int(self.num_fingertips)  # 65 = 13 * (10 / 2)
        # num_ft_force_torques = 6 * int(self.num_fingertips)  # 30
        num_act_per_hand = int(self.num_action)  # 8

        # 0 -14 right hand dof position
        self.obs_buf[:, 0:self.num_hand_dofs] = unscale(self.hand_dof_pos,
                                                        self.hand_dof_lower_limits,
                                                        self.hand_dof_upper_limits)
        # 15 - 29 right hand dof velocity
        self.obs_buf[:, self.num_hand_dofs:2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel
        # 30 - 44 right hand dof force
        self.obs_buf[:, 2 * self.num_hand_dofs:3 * self.num_hand_dofs] \
            = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_hand_dofs]

        # 140 - 142 right hand base position
        hand_pose_start = 3 * self.num_hand_dofs  # 140 = 45 + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.hand_pos
        # 143 - 145 right hand base rotation
        self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = \
            get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = \
            get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = \
            get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        # 146 - 167 right hand actions
        action_obs_start = hand_pose_start + 6  # 146 = 140 + 6
        self.obs_buf[:, action_obs_start:action_obs_start + num_act_per_hand] = self.actions[:, :num_act_per_hand]

        obj_obs_start = action_obs_start + num_act_per_hand  # 334
        # 334 - 340 object pose
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        # 341 - 343 object linear velocity
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        # 344 - 346 object angle velocity
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

    # def compute_point_cloud_observation(self, collect_demonstration=False):
    #     """
    #     Compute the observations of all environment. The observation is composed of three parts:
    #     the state values of the left and right hands, and the information of objects and target.
    #     The state values of the left and right hands were the same for each task, including hand
    #     joint and finger positions, velocity, and force information. The detail 361-dimensional
    #     observational space as shown in below:
    #
    #     Index       Description
    #     0 - 23	    right shadow hand dof position
    #     24 - 47	    right shadow hand dof velocity
    #     48 - 71	    right shadow hand dof force
    #     72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
    #     137 - 166	right shadow hand fingertip force, torque (5 x 6)
    #     167 - 169	right shadow hand base position
    #     170 - 172	right shadow hand base rotation
    #     173 - 198	right shadow hand actions
    #     199 - 222	left shadow hand dof position
    #     223 - 246	left shadow hand dof velocity
    #     247 - 270	left shadow hand dof force
    #     271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
    #     336 - 365	left shadow hand fingertip force, torque (5 x 6)
    #     366 - 368	left shadow hand base position
    #     369 - 371	left shadow hand base rotation
    #     372 - 397	left shadow hand actions
    #     398 - 404	object pose
    #     405 - 407	object linear velocity
    #     408 - 410	object angle velocity
    #     411 - 413	block right handle position
    #     414 - 417	block right handle rotation
    #     418 - 420	block left handle position
    #     421 - 424	block left handle rotation
    #     """
    #     num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
    #     num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30
    #
    #     self.obs_buf[:, 0:self.num_hand_dofs] = unscale(self.right_hand_dof_pos,
    #                                                     self.shadow_hand_dof_lower_limits,
    #                                                     self.shadow_hand_dof_upper_limits)
    #     self.obs_buf[:,
    #     self.num_hand_dofs:2 * self.num_hand_dofs] = self.vel_obs_scale * self.right_hand_dof_vel
    #     self.obs_buf[:,
    #     2 * self.num_hand_dofs:3 * self.num_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[
    #                                                                                    :, :24]
    #
    #     fingertip_obs_start = 72  # 168 = 157 + 11
    #     self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.right_fingertip_state.reshape(
    #         self.num_envs, num_ft_states)
    #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
    #                                                         num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[
    #                                                                                                               :,
    #                                                                                                               :30]
    #
    #     right_hand_pose_start = fingertip_obs_start + 95
    #     self.obs_buf[:, right_hand_pose_start:right_hand_pose_start + 3] = self.right_hand_pos
    #     self.obs_buf[:, right_hand_pose_start + 3:right_hand_pose_start + 4] = \
    #         get_euler_xyz(self.hand_orientations[self.right_hand_indices, :])[0].unsqueeze(-1)
    #     self.obs_buf[:, right_hand_pose_start + 4:right_hand_pose_start + 5] = \
    #         get_euler_xyz(self.hand_orientations[self.right_hand_indices, :])[1].unsqueeze(-1)
    #     self.obs_buf[:, right_hand_pose_start + 5:right_hand_pose_start + 6] = \
    #         get_euler_xyz(self.hand_orientations[self.right_hand_indices, :])[2].unsqueeze(-1)
    #
    #     right_action_obs_start = right_hand_pose_start + 6
    #     self.obs_buf[:, right_action_obs_start:right_action_obs_start + 26] = self.actions[:, :26]
    #
    #     # left_hand
    #     left_hand_start = right_action_obs_start + 26
    #     self.obs_buf[:, left_hand_start:self.num_hand_dofs + left_hand_start] = unscale(
    #         self.left_hand_dof_pos,
    #         self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
    #     self.obs_buf[:,
    #     self.num_hand_dofs + left_hand_start:2 * self.num_hand_dofs + left_hand_start] = self.vel_obs_scale * self.left_hand_dof_vel
    #     self.obs_buf[:,
    #     2 * self.num_hand_dofs + left_hand_start:3 * self.num_hand_dofs + left_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[
    #                                                                                                                        :,
    #                                                                                                                        24:48]
    #
    #     left_fingertip_obs_start = left_hand_start + 72
    #     self.obs_buf[:,
    #     left_fingertip_obs_start:left_fingertip_obs_start + num_ft_states] = self.left_fingertip_state.reshape(
    #         self.num_envs, num_ft_states)
    #     self.obs_buf[:, left_fingertip_obs_start + num_ft_states:left_fingertip_obs_start + num_ft_states +
    #                                                              num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[
    #                                                                                                                    :,
    #                                                                                                                    30:]
    #
    #     left_hand_pose_start = left_fingertip_obs_start + 95
    #     self.obs_buf[:, left_hand_pose_start:left_hand_pose_start + 3] = self.left_hand_pos
    #     self.obs_buf[:, left_hand_pose_start + 3:left_hand_pose_start + 4] = \
    #         get_euler_xyz(self.hand_orientations[self.left_hand_indices, :])[0].unsqueeze(-1)
    #     self.obs_buf[:, left_hand_pose_start + 4:left_hand_pose_start + 5] = \
    #         get_euler_xyz(self.hand_orientations[self.left_hand_indices, :])[1].unsqueeze(-1)
    #     self.obs_buf[:, left_hand_pose_start + 5:left_hand_pose_start + 6] = \
    #         get_euler_xyz(self.hand_orientations[self.left_hand_indices, :])[2].unsqueeze(-1)
    #
    #     left_right_action_obs_start = left_hand_pose_start + 6
    #     self.obs_buf[:, left_right_action_obs_start:left_right_action_obs_start + 26] = self.actions[:, 26:]
    #
    #     obj_obs_start = left_right_action_obs_start + 26  # 144
    #     self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
    #     self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
    #     self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
    #     self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.block_right_handle_pos
    #     self.obs_buf[:, obj_obs_start + 16:obj_obs_start + 20] = self.block_right_handle_rot
    #     self.obs_buf[:, obj_obs_start + 20:obj_obs_start + 23] = self.block_left_handle_pos
    #     self.obs_buf[:, obj_obs_start + 23:obj_obs_start + 27] = self.block_left_handle_rot
    #     # goal_obs_start = obj_obs_start + 13  # 157 = 144 + 13
    #     # self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
    #     # self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
    #     point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
    #
    #     if self.camera_debug:
    #         import matplotlib.pyplot as plt
    #         self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
    #         camera_rgba_image = self.camera_visulization(is_depth_image=False)
    #         plt.imshow(camera_rgba_image)
    #         plt.pause(1e-9)
    #
    #     for i in range(self.num_envs):
    #         # Here is an example. In practice, it's better not to convert tensor from GPU to CPU
    #         points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i],
    #                                                 self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2,
    #                                                 self.camera_props.width, self.camera_props.height, 10, self.device)
    #
    #         if points.shape[0] > 0:
    #             selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum,
    #                                                  sample_mathed='random')
    #         else:
    #             selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
    #
    #         point_clouds[i] = selected_points
    #
    #     if self.pointCloudVisualizer != None:
    #         import open3d as o3d
    #         points = point_clouds[0, :, :3].cpu().numpy()
    #         # colors = plt.get_cmap()(point_clouds[0, :, 3].cpu().numpy())
    #         self.o3d_pc.points = o3d.utility.Vector3dVector(points)
    #         # self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])
    #
    #         if self.pointCloudVisualizerInitialized == False:
    #             self.pointCloudVisualizer.add_geometry(self.o3d_pc)
    #             self.pointCloudVisualizerInitialized = True
    #         else:
    #             self.pointCloudVisualizer.update(self.o3d_pc)
    #
    #     self.gym.end_access_image_tensors(self.sim)
    #     point_clouds -= self.env_origin.view(self.num_envs, 1, 3)
    #
    #     point_clouds_start = obj_obs_start + 27
    #     self.obs_buf[:, point_clouds_start:].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))

    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        Reset and randomize the goal pose

        Args:
            env_ids (tensor): The index of the environment that needs to reset goal pose

            apply_reset (bool): Whether to reset the goal directly here, usually used
            when the same task wants to complete multiple goals

        """
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids],
                                     self.y_unit_tensor[env_ids])

        # # self.goal_states[env_ids, 3:7] = new_rot
        # self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids,
        #                                                                  0:3] + self.goal_displacement_tensor
        # self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        # self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
        #     self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        """
        Reset and randomize the environment

        Args:
            env_ids (tensor): The index of the environment that needs to reset

            goal_env_ids (tensor): The index of the environment that only goals need reset

        """
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(0, 0, (len(env_ids), self.num_hand_dofs + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids],
                                            self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
                                                    self.z_unit_tensor[env_ids])

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        delta_max = self.hand_dof_upper_limits - self.hand_dof_default_pos
        delta_min = self.hand_dof_lower_limits - self.hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_hand_dofs]

        pos = self.hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.hand_dof_pos[env_ids, :] = pos

        self.hand_dof_vel[env_ids, :] = self.hand_dof_default_vel + \
                                        self.reset_dof_vel_noise * rand_floats[:, 5:5 + self.num_hand_dofs]

        self.prev_targets[env_ids, :self.num_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_hand_dofs] = pos
        self.prev_synergy_actions[env_ids, :] = torch.tensor([0.0, 0.0], device=self.device)

        # self.prev_dof_action[env_ids, :] = -torch.ones((self.num_envs, 15)).to(self.device)

        if self.object_type in ["power_drill"]:
            self.prev_targets[env_ids, :7] = torch.tensor(
                [0.0905, 0.5326, 0.0486, -1.5469, -0.9613, 2.2102, 1.5221]).to(
                self.device)
            # self.prev_goal_pose[env_ids, :3] = torch.tensor([0.30, -0.15, 0.68]).to(self.device)
            self.prev_goal_pose[env_ids, :3] = torch.tensor([0.2616, -0.2173, 0.7121]).to(self.device)
            self.prev_goal_pose[env_ids, 3:7] = torch.tensor([0, 0.707, 0.707, 0]).to(self.device)
        elif self.object_type in ["mug", "large_marker"]:
            self.prev_targets[env_ids, :7] = torch.tensor(
                [0.0905, 0.5326, 0.0486, -1.5469, -0.9613, 2.2102, 1.5221]).to(
                self.device)
            # self.prev_goal_pose[env_ids, :3] = torch.tensor([0.30, -0.15, 0.68]).to(self.device)
            self.prev_goal_pose[env_ids, :3] = torch.tensor([0.2616, -0.2173, 0.6321]).to(self.device)
            self.prev_goal_pose[env_ids, 3:7] = torch.tensor([0, 0.707, 0.707, 0]).to(self.device)
        elif self.object_type in ["hammer", "large_clamp", "spatula", "phillips_screw_driver", "scissors", "knife"]:
            self.prev_targets[env_ids, :7] = torch.tensor(
                [-0.6654, 0.5872, 0.9190, -1.4767, -2.4621, 3.2423, 2.8480]).to(
                self.device)
            self.prev_goal_pose[env_ids, :3] = torch.tensor([0.2616, -0.2173, 0.8]).to(self.device)
            self.prev_goal_pose[env_ids, 3:7] = torch.tensor([0, 1., 0., 0]).to(self.device)
        self.cur_targets[env_ids, :7] = self.prev_targets[env_ids, :7]

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        all_hand_indices = torch.unique(hand_indices).to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.table_indices[env_ids]]).to(torch.int32))

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        """
        The pre-processing of the physics step. Determine whether the reset environment is needed,
        and calculate the next movement of Shadowhand through the given action. The 52-dimensional
        action space as shown in below:

        Index   Description
        0 - 2	right shadow hand base translation
        3 - 5	right shadow hand base rotation
        6 - 20 	right shadow hand actuated joint
        21 - 23 left shadow hand base translation
        24 - 26 left shadow hand base rotation
        27 - 41 left shadow hand actuated joint

        Args:
            actions (tensor): Actions of agents in the all environment
        """

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:,
                      self.actuated_dof_indices] + self.hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.hand_dof_lower_limits[
                                                                              self.actuated_dof_indices],
                                                                          self.hand_dof_upper_limits[
                                                                              self.actuated_dof_indices])
        else:
            self.gym.clear_lines(self.viewer)

            if self.cfg["env"]["useSynergy"]:
                synergy_action = self.actions[:, 6:8]
                # synergy_action = torch.ones_like(self.actions[:, 6:8]).to(self.device)
                # synergy_action[:, 0] = 0.44 * synergy_action[:, 0]

                synergy_action[:, 0] = torch.abs(synergy_action[:, 0])
                synergy_action = self.prev_synergy_actions * 0.9 + 0.1 * synergy_action
                synergy_action[:, 0] = torch.abs(synergy_action[:, 0])
                # synergy_action = torch.zeros_like(self.actions[:, 6:8]).to(self.device)
                # synergy_action[:, 0] = torch.ones_like(self.actions[:, 6]).to(self.device)
                # synergy_action[:, 0] = self.actions[:, 6]
                self.prev_synergy_actions = synergy_action
                synergy_action_matrix = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                      [2, 2, 2, 1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2]],
                                                     device=self.device, dtype=torch.float32)
                dof_action = torch.matmul(synergy_action, synergy_action_matrix)
                dof_action = torch.clamp(dof_action, 0, 1.0)
                dof_action = dof_action * 2 - 1

                tmp = torch.zeros_like(dof_action)
                # Thumb
                tmp[:, 12:] = dof_action[:, :3]
                # Index
                tmp[:, 0:3] = dof_action[:, 3:6]
                # Middle
                tmp[:, 6:9] = dof_action[:, 6:9]
                # Ring
                tmp[:, 9:12] = dof_action[:, 9:12]
                # Little
                tmp[:, 3:6] = dof_action[:, 12:15]
                dof_action = tmp
            else:
                dof_action = self.actions[:, 6:21]

            self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
            thumb_pos = self.rigid_body_states[:, 36, 0:3]
            hand_dist = torch.norm(self.object_pos - thumb_pos, p=2, dim=-1)
            # dof_action = torch.where(self.object_pos[:, 2].unsqueeze(-1) < 0.7, torch.where(hand_dist.unsqueeze(-1) > 0.05,
            #                          -torch.ones_like(dof_action), dof_action), dof_action)
            # self.prev_dof_action = dof_action

            # dof_action = torch.where(hand_dist.unsqueeze(-1) > 0.06, -torch.ones_like(dof_action), dof_action)

            self.cur_targets[:, self.useful_joint_index] = scale(dof_action,
                                                                 self.hand_dof_lower_limits[
                                                                     self.useful_joint_index],
                                                                 self.hand_dof_upper_limits[
                                                                     self.useful_joint_index])
            self.cur_targets[:, self.useful_joint_index] = self.act_moving_average * self.cur_targets[:,
                                                                                     self.useful_joint_index] + (
                                                                   1.0 - self.act_moving_average) * self.prev_targets[
                                                                                                    :,
                                                                                                    self.useful_joint_index]
            self.cur_targets[:, self.useful_joint_index] = tensor_clamp(
                self.cur_targets[:, self.useful_joint_index],
                self.hand_dof_lower_limits[self.useful_joint_index],
                self.hand_dof_upper_limits[self.useful_joint_index])
            for key, value in self.real_virtual_joint_index_map_dict.items():
                self.cur_targets[:, key] = self.cur_targets[:, value]

            goal_pos = self.prev_goal_pose[:, :3] + self.actions[:, 0:3] * 0.01
            # goal_pos = torch.where(hand_dist.unsqueeze(-1) > 0.2, self.prev_goal_pose[:, :3],
            #                        self.prev_goal_pose[:, :3] + self.actions[:, 0:3] * 0.005)
            # goal_rot = torch.where(hand_dist.unsqueeze(-1) > 0.2, self.prev_goal_pose[:, 3:7],
            #             slerp(self.prev_goal_pose[:, 3:7], self.actions[:, 3:7], torch.tensor(10)))
            # goal_rot = torch.nn.functional.normalize(goal_rot, dim=1)
            goal_rot = self.prev_goal_pose[:, 3:7]

            # goal_pos[:, 2] = torch.where(goal_pos[:, 2] < 0.65, torch.tensor([0.65]).to(self.device), goal_pos[:, 2])
            if self.object_type in ["power_drill"]:
                goal_pos[:, 2] = torch.where(hand_dist < 0.05, goal_pos[:, 2] + 0.01, goal_pos[:, 2])
            elif self.object_type in ["mug", "large_marker"]:
                goal_pos[:, 2] = torch.where(hand_dist < 0.1, goal_pos[:, 2] + 0.01, goal_pos[:, 2])
            # elif self.object_type in ["mug"]:
            #     goal_pos[:, 2] = torch.where(hand_dist < 0.15, goal_pos[:, 2] + 0.01, goal_pos[:, 2])
            elif self.object_type in ["large_clamp", "spatula", "knife", "hammer"]:
                goal_pos[:, 2] = torch.where((thumb_pos[:, 2] - self.object_pos[:, 2]) < 0.05, goal_pos[:, 2] + 0.01,
                                             goal_pos[:, 2])
            elif self.object_type in ["scissors", "phillips_screw_driver"]:
                goal_pos[:, 2] = torch.where((thumb_pos[:, 2] - self.object_pos[:, 2]) < 0.02, goal_pos[:, 2] + 0.01,
                                             goal_pos[:, 2])

            self.prev_goal_pose[:, :3] = goal_pos
            self.prev_goal_pose[:, 3:7] = goal_rot
            hand_pos = self.rigid_body_states[:, 7, 0:3]
            hand_rot = self.rigid_body_states[:, 7, 3:7]
            pos_err = goal_pos - hand_pos
            orn_err = self.orientation_error(goal_rot, hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            self.j_eef = self.jacobian[:, 7 - 1, :, 0:7]
            self.cur_targets[:, :7] = self.hand_dof_pos[:, :7] + control_ik(self.j_eef, dpose).view(self.num_envs, 7)

            for i in range(self.num_envs):
                # Draw axes and sphere at attractor location
                pose = gymapi.Transform()
                # pose.p: (x, y, z), pose.r: (w, x, y, z)
                pose.p.x = goal_pos[i, 0]
                pose.p.y = goal_pos[i, 1]
                pose.p.z = goal_pos[i, 2]
                pose.r.x = goal_rot[i, 0]
                pose.r.y = goal_rot[i, 1]
                pose.r.z = goal_rot[i, 2]
                pose.r.w = goal_rot[i, 3]
                # gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], pose)
                # gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], pose)

            # hand_dist = torch.norm(self.object_pos - hand_pos, p=2, dim=-1)
            # self.apply_forces[:, 0, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            # self.apply_torque[:, 0, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            #
            # self.apply_forces[:, 0, :] = torch.zeros_like(self.actions[:, 0:3])
            # self.apply_torque[:, 0, :] = torch.zeros_like(self.actions[:, 3:6])
            #
            # self.apply_forces[:, 7, :] = torch.where(hand_dist.unsqueeze(-1) < 0.05,
            #                                          torch.zeros_like(self.actions[:, 0:3]) + torch.tensor(
            #                                              [0., 0., 1000]).to(self.device),
            #                                          self.actions[:, 0:3] * self.dt * self.transition_scale * 100000)
            # # self.apply_torque[:, 0, :] = torch.where(hand_dist.unsqueeze(-1) < 0.2,
            # #                                          torch.zeros_like(self.actions[:, 3:6]),
            # #                                          self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000)
            # self.apply_torque[:, 7, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
            #                                         gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

            # hand_target_pos = self.actions[:, 0:3] * 0.01
            # hand_target_rot = rf.robolab.quaternion_from_euler(*self.actions[:, 3:6])
            # # Update attractor target from current franka state
            # attractor_properties = self.gym.get_attractor_properties(self.envs, self.attractor_handles)
            # pose = attractor_properties.target
            # # pose.p: (x, y, z), pose.r: (w, x, y, z)
            # pose.p.x = hand_target_pos[:, 0]
            # pose.p.y = hand_target_pos[:, 1]
            # pose.p.z = hand_target_pos[:, 2]
            # pose.r.w = hand_target_rot[:, 0]
            # pose.r.x = hand_target_rot[:, 1]
            # pose.r.y = hand_target_rot[:, 2]
            # pose.r.z = hand_target_rot[:, 3]
            # self.gym.set_attractor_target(self.envs, self.attractor_handles, pose)

            # # Draw axes and sphere at attractor location
            # gymutil.draw_lines(self.axes_geoms, self.gym, self.viewer, self.envs, pose)
            # gymutil.draw_lines(self.sphere_geoms, self.gym, self.viewer, self.envs, pose)

        # self.prev_targets[:, :7] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        #                                          0.0], device=self.device).repeat(self.num_envs, 1)
        self.prev_targets[:, 7:39] = self.cur_targets[:, 7:39]
        self.prev_targets[:, :7] = self.cur_targets[:, :7]

        # self.prev_targets = torch.zeros_like(self.prev_targets)

        # self.prev_targets[:, 49] = self.cur_targets[:, 49]
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        all_hand_indices = torch.unique(self.hand_indices).to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        # self.prev_targets_vel = torch.zeros_like(self.prev_targets)
        # dof_state = torch.cat([self.prev_targets, self.prev_targets_vel], dim=-1)
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(dof_state),
        #                                       gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def post_physics_step(self):
        """
        The post-processing of the physics step. Compute the observation and reward, and visualize auxiliary
        lines for debug when needed

        """
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # for i in range(self.num_envs):
            # self.add_debug_lines(self.envs[i], self.block_handle_pos[i], self.block_handle_rot[i])
            # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.block_left_handle_rot[i])
            # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
            # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
            # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
            # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
            # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

            # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
            # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
            # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
            # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
            # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def rand_row(self, tensor, dim_needed):
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)), :]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2] > 0.04]
        if eff_points.shape[0] < sample_num:
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape),
                                                                      sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0],
                                                                       gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                       to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)

        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0],
                                                                      gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)

        return camera_image


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width: float,
                                   height: float, depth_bar: float, device: torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection

    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points


@torch.jit.script
def compute_hand_reward(
        rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
        max_episode_length: float, object_pos, object_rot,
        hand_pos, hand_ff_pos, hand_mf_pos, hand_rf_pos, hand_lf_pos,
        hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool,
        prev_synergy_actions, synergy_target):
    """
    Compute the reward of all environment.

    Args:
        rew_buf (tensor): The reward buffer of all environments at this time

        reset_buf (tensor): The reset buffer of all environments at this time

        reset_goal_buf (tensor): The only-goal reset buffer of all environments at this time

        progress_buf (tensor): The porgress buffer of all environments at this time

        successes (tensor): The successes buffer of all environments at this time

        consecutive_successes (tensor): The consecutive successes buffer of all environments at this time

        max_episode_length (float): The max episode length in this environment

        object_pos (tensor): The position of the object

        object_rot (tensor): The rotation of the object

        target_pos (tensor): The position of the target

        target_rot (tensor): The rotate of the target

        block_right_handle_pos (tensor): The position of the right block handle

        right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos (tensor): The position of the five fingers 
            of the right hand
            
        dist_reward_scale (float): The scale of the distance reward

        rot_reward_scale (float): The scale of the rotation reward

        rot_eps (float): The epsilon of the rotation calculate

        actions (tensor): The action buffer of all environments at this time

        action_penalty_scale (float): The scale of the action penalty reward

        success_tolerance (float): The tolerance of the success determined

        reach_goal_bonus (float): The reward given when the object reaches the goal

        fall_dist (float): When the object is far from the Shadowhand, it is judged as falling

        fall_penalty (float): The reward given when the object is fell

        max_consecutive_successes (float): The maximum of the consecutive successes

        av_factor (float): The average factor for calculate the consecutive successes

        ignore_z_rot (bool): Is it necessary to ignore the rot of the z-axis, which is usually used 
            for some specific objects (e.g. pen)
    """
    hand_dist = torch.norm(object_pos - hand_pos, p=2, dim=-1)

    hand_finger_dist = (torch.norm(object_pos - hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_pos - hand_mf_pos, p=2, dim=-1)
                        + torch.norm(object_pos - hand_rf_pos, p=2, dim=-1) + torch.norm(
                object_pos - hand_lf_pos, p=2, dim=-1)
                        + torch.norm(object_pos - hand_th_pos, p=2, dim=-1))
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    hand_dist_rew = torch.exp(-10 * hand_finger_dist)

    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    synergy = prev_synergy_actions
    # synergy_target = torch.tensor([1, 0]).to(synergy.device)
    synergy_dist = torch.norm(synergy - synergy_target, p=2, dim=-1) * 10

    # # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
    # # up_rew = torch.zeros_like(right_hand_dist_rew)
    # # up_rew = torch.where(right_hand_finger_dist < 0.6,
    # #                 torch.where(left_hand_finger_dist < 0.4,
    # up_rew = torch.zeros_like(right_hand_dist_rew)
    # up_rew = torch.exp(-10 * torch.norm(block_right_handle_pos - block_left_handle_pos, p=2, dim=-1)) * 2
    # # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)
    #
    # # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))

    up_rew = object_pos[:, 2] - 0.7
    up_rew = torch.where(up_rew > 0, up_rew, torch.zeros_like(up_rew)) * 100
    # print("Max height: {}, Avg height: {}".format(torch.max(object_pos[:, 2]), torch.mean(object_pos[:, 2])))
    reward = up_rew - synergy_dist - hand_dist
    # print("up_rew: {}, synergy_rew: {}, hand_dist: {}".format(float(torch.mean(up_rew).cpu()),
    #                                                           float(torch.mean(synergy_dist).cpu()),
    #                                                           float(torch.mean(hand_dist).cpu())))

    # resets = torch.where(hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(object_rot[:, 3] > 0.9, torch.ones_like(resets), resets)
    # resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
    resets = torch.where(hand_dist >= 0.3, torch.ones_like(reset_buf), reset_buf)

    # Find out which envs hit the goal and update successes count
    successes = torch.where(successes == 0, torch.where(object_pos[:, 2] > 1.0,
                                                        torch.ones_like(successes), successes), successes)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


@torch.jit.script
def control_ik(j_eef, dpose):
    damping = 0.0000001
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = (torch.eye(6) * (damping ** 2)).to(j_eef.device)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose)
    return u
