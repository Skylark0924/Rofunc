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
from enum import Enum

import rofunc as rf
from gym import spaces
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from rofunc.learning.RofuncRL.tasks.isaacgymenv.base.vec_task import VecTask
from rofunc.learning.RofuncRL.tasks.isaacgymenv.hotu.humanoid import dof_to_obs
from rofunc.learning.RofuncRL.tasks.isaacgymenv.hotu.motion_lib import MotionLib, ObjectMotionLib
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils
from rofunc.utils.oslab.path import get_rofunc_path


class Humanoid(VecTask):
    """
    This class is a wrapper of the Isaac Gym environment for the Humanoid task.
    """

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture,
                 force_render):
        # Load the config
        self.cfg = config
        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)

        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        # Set the dimensions of the observation and action spaces
        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # Acquiring the state tensors from the simulator
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        # Update state tensor buffers
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[...,
                                     0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        # Get the actor ids for the humanoid and the objects
        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        if self.object_names is not None:
            self._object_actor_ids = {
                object_name: torch.tensor(
                    [self.gym.get_actor_index(self.envs[i], self.object_handles[object_name][i], gymapi.DOMAIN_SIM) for
                     i in range(self.num_envs)], dtype=torch.int32, device=self.device) for object_name in
                self.object_names}

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._build_termination_heights()

        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        if self.viewer is not None:
            self._init_camera()

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self, **kwargs):
        # self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._reset_env_tensors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _setup_character_props(self, key_bodies):
        """
        dof_body_ids records the ids of the bodies that are connected to their parent bodies with joints
        The order of these ids follows the define order of the body in the MJCF. The id start from 0, and
        the body with id:0 is pelvis, which is not considered in the list.

        dof_offset's length is always len(dof_body_ids) + 1, and it always start from 0.
        Each 2 values' minus in the list represents how many dofs that corresponding body have.

        dof_observation_size is equal to dof * 6, where 6 stands for position and rotation observations, dof is the
        number of actuated dofs, it equals to the length of dof_body_ids

        num_actions is equal to the number of actuatable joints' number in the character. It does not include the
        joint connecting the character to the world.
        dof_observation_size

        num_observations is composed by 3 parts, the first observation is the height of the CoM of the character; the
        second part is the observations for all bodies. The body number is multiplied by (3 position values, 6
        orientation values, 3 linear velocity, and 3 angular velocity); finally, -3 stands for

        :param key_bodies:
        """
        # asset_body_num = self.cfg["env"]["asset"]["assetBodyNum"]
        # asset_joint_num = self.cfg["env"]["asset"]["assetJointNum"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        # num_key_bodies = len(key_bodies)

        # The following are: body_name (body_id/body's joint num to its parent/offset pair)
        # if asset_body_num == 15:
        #     if asset_joint_num == 28:
        #         # torso (1/3/0 3), head (2/3/3 6), right_upper_arm (3/3/6 9), right_lower_arm (4/1/9 10),
        #         # right_hand (5/0, omitted as no joint to parent)
        #         # left_upper_arm (6/3/10 13), left_lower_arm (7/1/13 14), left_hand (8/0), right_thigh (9/3/14 17),
        #         # right_shin (10/1/17 18), right_foot (11/3/18 21), left_thigh (12/3/21 24), left_shin (13/1/24 25),
        #         # left_foot (14/3/25 28)
        #         self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]  # len=12
        #         self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]  # len=12+1
        #         self._dof_obs_size = 72
        #         self._num_actions = 28
        #         self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        #     elif asset_joint_num == 34:
        #         # torso (1/3/0 3), head (2/3/3 6),
        #         # right_upper_arm (3/3/6 9), right_lower_arm (4/1/9 10), right_hand (5/3/10 13)
        #         # left_upper_arm (6/3/13 16), left_lower_arm (7/1/16 17), left_hand (8/3/17 20),
        #         # right_thigh (9/3/20 23), right_shin (10/1/23 24), right_foot (11/3/24 27),
        #         # left_thigh (12/3/27 30), left_shin (13/1/30 31), left_foot (14/3/31 34)
        #         self._dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # len=14
        #         self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 23, 24, 27, 30, 31, 34]  # len=14+1
        #         self._dof_obs_size = 84
        #         self._num_actions = 34
        #         self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        #     else:
        #         raise NotImplementedError
        # elif asset_body_num == 16:
        #     self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
        #     self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
        #     self._dof_obs_size = 78
        #     self._num_actions = 31
        #     self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        # elif asset_body_num == 17:
        #     if asset_joint_num == 34:
        #         # The following are: body_name (body_id/body's joint num to its parent/offset pair)
        #         # torso (1/3/0 3), head (2/3/3 6), right_upper_arm (3/3/6 9), right_lower_arm (4/1/9 10),
        #         # right_hand (5/3/10 13), spoon (6/0), left_upper_arm (7/3/13 16), left_lower_arm (8/1/16 17),
        #         # left_hand (9/3/17 20), pan (10/0), right_thigh (11/3/20 23), right_shin (12/1/23 24),
        #         # right_foot (13/3/24 27), left_thigh (14/3/27 30), left_shin (15/1/30 31), left_foot (16/3/31 34)
        #         self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16]
        #         self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 23, 24, 27, 30, 31, 34]
        #         self._dof_obs_size = 84
        #         self._num_actions = 34
        #         self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        #     elif asset_joint_num == 38:
        #         # torso (1/3/0 3), head (2/3/3 6), right_upper_arm (3/3/6 9), right_lower_arm (4/3/9 12),
        #         # right_hand (5/3/12 15), spoon (6/0), left_upper_arm (7/3/15 18), left_lower_arm (8/3/18 21),
        #         # left_hand (9/3/21 24), pan (10/0), right_thigh (11/3/24 27), right_shin (12/1/27 28),
        #         # right_foot (13/3/28 31), left_thigh (14/3/31 34), left_shin (15/1/34 35), left_foot (16/3/35 38)
        #         self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16]
        #         self._dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 31, 34, 35, 38]
        #         self._dof_obs_size = 84
        #         self._num_actions = 38
        #         self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        # elif asset_body_num == 19:
        #     if asset_joint_num == 44:
        #         # torso (1/3/0 3), head (2/3/3 6),
        #         # right_shoulder(3/3/6 9), right_upper_arm (4/3/9 12), right_lower_arm (5/3/12 15),
        #         # right_hand (6/3/15 18), spoon (7/0),
        #         # left_shoulder(8/3/18 21), left_upper_arm (9/3/21 24), left_lower_arm (10/3/24 27),
        #         # left_hand (11/3/27 30), pan (12/0), right_thigh (13/3/30 33), right_shin (14/1/33 34),
        #         # right_foot (15/3/34 37), left_thigh (16/3/37 40), left_shin (17/1/40 41), left_foot (18/3/41 44)
        #         self._dof_body_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
        #         self._dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 34, 37, 40, 41, 44]
        #         self._dof_obs_size = 96
        #         self._num_actions = 44
        #         self._num_obs = 1 + 19 * (3 + 6 + 3 + 3) - 3
        if asset_file == "mjcf/amp_humanoid.xml":
            # torso (1/3/0 3), head (2/3/3 6), right_upper_arm (3/3/6 9), right_lower_arm (4/1/9 10),
            # right_hand (5/0, omitted as no joint to parent)
            # left_upper_arm (6/3/10 13), left_lower_arm (7/1/13 14), left_hand (8/0), right_thigh (9/3/14 17),
            # right_shin (10/1/17 18), right_foot (11/3/18 21), left_thigh (12/3/21 24), left_shin (13/1/24 25),
            # left_foot (14/3/25 28)
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]  # len=12
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]  # len=12+1
            self._dof_obs_size = 72  # 12 * 6 (joint_obs_size) = 72
            self._num_actions = 28
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            self._dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31]
            self._dof_obs_size = 78
            self._num_actions = 31
            self._num_obs = 1 + 17 * (3 + 6 + 3 + 3) - 3
        elif asset_file in ["mjcf/amp_humanoid_spoon_pan_fixed.xml", "mjcf/hotu_humanoid.xml"]:
            self._dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # len=14
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 16, 17, 20, 23, 24, 27, 30, 31, 34]  # len=14+1
            self._dof_obs_size = 84  # 14 * 6 (joint_obs_size) = 72
            self._num_actions = 34
            self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
        elif asset_file in "mjcf/hotu_humanoid_w_qbhand.xml":
            self._dof_body_ids = [*[i for i in range(1, 6)], 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39,
                                  *[i for i in range(40, 43)], 44, 46, 48, 51, 53, 55, 58, 60, 62, 65, 67, 69, 72, 74,
                                  76, *[i for i in range(77, 83)]]  # len=44
            self._dof_offsets = [0, 3, 6, 9, 10, *[i for i in range(13, 28)], 28, 31, 32, *[i for i in range(35, 50)],
                                 50, 53, 54, 57, 60, 61, 64]  # len=44+1
            self._dof_obs_size = 264  # 44 * 6 (joint_obs_size) = 264
            self._num_actions = 64
            self._num_obs = 1 + 83 * (3 + 6 + 3 + 3) - 3  # 1243
        elif asset_file in ["mjcf/hotu_humanoid_w_qbhand_no_virtual.xml",
                            "mjcf/hotu_humanoid_w_qbhand_no_virtual_no_quat.xml"]:
            self._dof_body_ids = [*[i for i in range(1, 45)]]  # len=44
            self._dof_offsets = [0, 3, 6, 9, 10, *[i for i in range(13, 28)], 28, 31, 32, *[i for i in range(35, 50)],
                                 50, 53, 54, 57, 60, 61, 64]  # len=44+1
            self._dof_obs_size = 264  # 44 * 6 (joint_obs_size) = 264
            self._num_actions = 64
            self._num_obs = 1 + 45 * (3 + 6 + 3 + 3) - 3  # 673
        else:
            raise rf.logger.beauty_print(f"Unsupported character config file: {asset_file}")

    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            left_arm_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.humanoid_handles[0], "left_lower_arm"
            )
            self._termination_heights[left_arm_id] = max(shield_term_height, self._termination_heights[left_arm_id])

        self._termination_heights = to_torch(self._termination_heights, device=self.device)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # get rofunc path from rofunc package metadata
        rofunc_path = get_rofunc_path()
        asset_root = os.path.join(rofunc_path, "simulator/assets")

        # Load humanoid asset
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.disable_gravity = True
        # asset_options.fix_base_link = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        # Load object assets
        object_asset_files = self.cfg["env"]["object_asset"]["assetFileName"]
        self.object_names = self.cfg["env"]["object_asset"]["assetName"]
        if self.object_names is not None:
            object_assets = {}
            for i in range(len(self.object_names)):
                asset_options = gymapi.AssetOptions()
                asset_options.angular_damping = 0.01
                asset_options.max_angular_velocity = 100.0
                asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                # asset_options.disable_gravity = True
                # asset_options.fix_base_link = True

                if 'box' in self.object_names[i].lower():
                    object_size = self.cfg["env"]["object_asset"]["assetSize"][i]
                    object_asset = self.gym.create_box(self.sim, *object_size, asset_options)
                elif self.object_names[i].lower() == 'sphere':
                    object_radius = self.cfg["env"]["object_asset"]["assetSize"][i]
                    object_asset = self.gym.create_sphere(self.sim, object_radius, asset_options)
                else:
                    object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_files[i], asset_options)
                object_assets[self.object_names[i]] = object_asset
        else:
            object_assets = None

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.object_handles = {}
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, humanoid_asset, object_assets)
            self.envs.append(env_ptr)

        # Set humanoid dof
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if self._pd_control:
            self._build_pd_action_offset_scale()

    def _build_env(self, env_id, env_ptr, humanoid_asset, object_assets=None):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(
            env_ptr,
            humanoid_asset,
            start_pose,
            "humanoid",
            col_group,
            col_filter,
            segmentation_id,
        )

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.54, 0.85, 0.2),
            )

        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        if object_assets is not None:
            for object_name, object_asset in object_assets.items():
                self._add_object(env_id, env_ptr, object_name, object_asset)

    def _add_object(self, env_id, env_ptr, object_name, object_asset):
        start_pose = gymapi.Transform()
        char_h = 0.5

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        col_group = 100000 if object_name is 'base' else env_id

        object_handle = self.gym.create_actor(
            env_ptr,
            object_asset,
            start_pose,
            object_name,
            col_group,
            1,
            0,
        )
        if object_name not in self.object_handles:
            self.object_handles[object_name] = []
        self.object_handles[object_name].append(object_handle)
        self.gym.set_rigid_body_color(
            env_ptr,
            object_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.5, 0.2, 0.0),
        )

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if dof_size == 3:
                curr_low = lim_low[dof_offset: (dof_offset + dof_size)]
                curr_high = lim_high[dof_offset: (dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset: (dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset: (dof_offset + dof_size)] = curr_scale

                # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

    def _get_humanoid_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            self._rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_heights,
        )

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

    def _compute_humanoid_obs(self, env_ids=None):
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        obs = compute_humanoid_observations_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            self._local_root_obs,
            self._root_height_obs,
        )
        return obs

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if self._pd_control:
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

    def render(self, sync_frame_time=False):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render(sync_frame_time)

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0
        )
        if self.cfg["sim"]["up_axis"] == "y":
            cam_pos = gymapi.Vec3(
                self._cam_prev_char_pos[0],
                1.0,
                self._cam_prev_char_pos[1] - 3.0,
            )
            cam_target = gymapi.Vec3(
                self._cam_prev_char_pos[0],
                1.0,
                self._cam_prev_char_pos[1],
            )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2]
        )

        if self.cfg["sim"]["up_axis"] == "y":
            new_cam_target = gymapi.Vec3(char_root_pos[0], 1.0, char_root_pos[1])
            new_cam_pos = gymapi.Vec3(
                char_root_pos[0] + cam_delta[0],
                cam_pos[2],
                char_root_pos[1] + cam_delta[1],
            )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)


class HumanoidHOTU(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidHOTU.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Load motion file
        motion_file = cfg["env"].get("motion_file", None)
        if rf.oslab.is_absl_path(motion_file):
            motion_file_path = motion_file
        elif motion_file.split("/")[0] == "examples":
            motion_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../../../../" + motion_file,
            )
        else:
            raise ValueError("Unsupported motion file path")
        self._load_motion(motion_file_path)

        # Load object motion file
        object_motion_file = cfg["env"].get("object_motion_file", None)
        if object_motion_file is not None:
            if rf.oslab.is_absl_path(object_motion_file):
                object_motion_file_path = object_motion_file
            elif object_motion_file.split("/")[0] == "examples":
                object_motion_file_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../../../../../" + object_motion_file,
                )
            else:
                raise ValueError("Unsupported object motion file path")
            self._load_object_motion(object_motion_file_path)

        # Set up the observation space for AMP
        self._amp_obs_space = spaces.Box(np.ones(self.get_num_amp_obs()) * -np.Inf,
                                         np.ones(self.get_num_amp_obs()) * np.Inf)
        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                        device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert self._amp_obs_demo_buf.shape[0] == num_samples

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, _, _ \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                             device=self.device, dtype=torch.float32)

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if asset_file == "mjcf/amp_humanoid.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, d
        elif asset_file in ["mjcf/amp_humanoid_spoon_pan_fixed.xml", "mjcf/hotu_humanoid.xml"]:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 34 + 3 * num_key_bodies
        elif asset_file == "mjcf/hotu_humanoid_w_qbhand.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 64 + 3 * num_key_bodies
        elif asset_file in ["mjcf/hotu_humanoid_w_qbhand_no_virtual.xml",
                            "mjcf/hotu_humanoid_w_qbhand_no_virtual_no_quat.xml"]:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 64 + 3 * num_key_bodies
        else:
            print(f"Unsupported humanoid body num: {asset_file}")
            assert False

    def _load_motion(self, motion_file):
        assert self._dof_offsets[-1] == self.num_dof
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            dof_body_ids=self._dof_body_ids,
            dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
        )

    def _load_object_motion(self, object_motion_file):
        self._object_motion_lib = ObjectMotionLib(
            object_motion_file=object_motion_file,
            object_names=self.cfg["env"]["object_asset"]["assetName"],
            device=self.device,
            height_offset=0
        )

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidHOTU.StateInit.Default:
            self._reset_default(env_ids)
        elif (
                self._state_init == HumanoidHOTU.StateInit.Start
                or self._state_init == HumanoidHOTU.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidHOTU.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[
            env_ids
        ]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (
                self._state_init == HumanoidHOTU.StateInit.Random
                or self._state_init == HumanoidHOTU.StateInit.Hybrid
        ):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HumanoidHOTU.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (
                False
            ), f"Unsupported state initialization strategy: {self._state_init}"

        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(
            np.array([self._hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(torch.tensor(ref_init_mask))]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
            )

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
                torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(
            self._hist_amp_obs_buf[env_ids].shape
        )

    def _set_env_state(
            self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        # self._dof_pos[env_ids] = dof_pos + self.init_dof_pose.to('cuda:0')
        # self._dof_pos[env_ids] = torch.zeros_like(dof_pos).to('cuda:0')
        self._dof_pos[env_ids] = dof_pos
        # self._dof_pos[env_ids, 6] = -1
        # self._dof_pos[env_ids, 28] = 1
        self._dof_vel[env_ids] = dof_vel

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )


@torch.jit.script
def build_amp_observations(
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        key_body_pos,
        local_root_obs,
        root_height_obs,
        dof_obs_size,
        dof_offsets,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset: (dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif dof_size == 1:
            axis = torch.tensor(
                [0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device
            )
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size): ((j + 1) * joint_obs_size)] = joint_dof_obs

    assert (num_joints * joint_obs_size) == dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        key_body_pos,
        local_root_obs,
        root_height_obs,
        dof_obs_size,
        dof_offsets,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_observations_max(
        body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_humanoid_reset(
        reset_buf,
        progress_buf,
        contact_buf,
        contact_body_ids,
        rigid_body_pos,
        max_episode_length,
        enable_early_termination,
        termination_heights,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated
