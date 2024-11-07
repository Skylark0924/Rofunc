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

import os
import random
from enum import Enum

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import rofunc as rf
from rofunc.learning.RofuncRL.tasks.isaacgymenv.base.vec_task import VecTask
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils
from rofunc.utils.oslab import get_rofunc_path

PERTURB_OBJS = [
    ["small", 60],
    # ["large", 60],
]


class Humanoid_SMPLX(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.ref_hoi_obs_size = 324 + len(self.cfg["env"]["keyBodies"]) * 3
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)

        # self.max_episode_length = self.cfg["env"]["episodeLength"]

        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        # self.cfg["device_type"] = device_type
        # self.cfg["device_id"] = device_id
        # self.cfg["headless"] = headless

        # super().__init__(cfg=self.cfg)
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

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

        if self.viewer != None:
            self._init_camera()

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self.reset_idx(env_ids)

        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)

        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    # def reset(self):
    #     actions = 0.01 * (1 - 2 * np.random.rand(self.num_envs, self.num_actions)).astype('f')
    #     actions = to_torch(actions, device=self.rl_device, dtype=torch.float)
    #
    #     # step the simulator
    #     obs, rewards, resets, extras = self.step(actions)
    #
    #     return torch.clamp(obs["obs"], -self.clip_obs, self.clip_obs)

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

    def reset_idx(self, env_ids):
        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.progress_buf[env_ids] = self.motion_times.clone()
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "smplx/smplx_capsule.xml"):
            self._dof_obs_size = (51) * 3
            self._num_actions = (51) * 3
            obj_obs_size = 15
            self._num_obs = 1 + (52) * (3 + 6 + 3 + 3) - 3 + 10 * 3 + obj_obs_size + self.ref_hoi_obs_size
        else:
            raise rf.logger.beauty_print(f"Unsupported character config file: {asset_file}")

    def _build_termination_heights(self):
        self._termination_heights = 0.3
        self._termination_heights = to_torch(self._termination_heights, device=self.device)

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # get rofunc path from rofunc package metadata
        rofunc_path = get_rofunc_path()
        asset_root = os.path.join(rofunc_path, "simulator/assets")

        # Load humanoid asset
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.fix_base_link = True
        # asset_options.disable_gravity = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

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

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        max_agg_bodies = self.num_humanoid_bodies + 2
        max_agg_shapes = self.num_humanoid_shapes + 2

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._build_env(i, env_ptr, humanoid_asset)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

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

    def _build_env(self, env_id, env_ptr, humanoid_asset):
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

    def _build_pd_action_offset_scale(self):

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

    def _get_humanoid_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(
            self._curr_ref_obs,
            self._curr_obs,
            self._contact_forces,
            self._tar_contact_forces,
            len(self._key_body_ids),
            self.reward_weights
            # self.reward_weights_p,
            # self.reward_weights_r,
            # self.reward_weights_pv,
            # self.reward_weights_rv,
            # self.reward_weights_op,
            # self.reward_weights_or,
            # self.reward_weights_opv,
            # self.reward_weights_orv,
            # self.reward_weights_ig,
            # self.reward_weights_cg1,
            # self.reward_weights_cg2,
        )

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces,
                                                                           self._rigid_body_pos,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights,
                                                                           self._curr_ref_obs, self._curr_obs,
                                                                           )

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]

        obs = compute_obj_observations(root_states, tar_states)
        return obs

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        task_obs = self._compute_task_obs(env_ids)
        obs = torch.cat([obs, task_obs], dim=-1)

        if env_ids is None:
            ts = self.progress_buf.clone()

            self._curr_ref_obs = self.hoi_data_dict[0]['hoi_data'][ts].clone()
            next_ts = torch.clamp(ts + 1, max=self.max_episode_length - 1)
            ref_obs = self.hoi_data_dict[0]['hoi_data'][next_ts].clone()
            self.obs_buf[:] = torch.cat((obs, ref_obs), dim=-1)

        else:
            ts = self.progress_buf[env_ids].clone()
            self._curr_ref_obs[env_ids] = self.hoi_data_dict[0]['hoi_data'][ts].clone()
            next_ts = torch.clamp(ts + 1, max=self.max_episode_length - 1)
            ref_obs = self.hoi_data_dict[0]['hoi_data'][next_ts].clone()
            self.obs_buf[env_ids] = torch.cat((obs, ref_obs), dim=-1)

        self.obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def _compute_humanoid_obs(self, env_ids=None):
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]

        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids)

        return obs

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

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

        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        # extra calc of self._curr_hoi_obs_buf, for correct calculate of imitation reward
        self._compute_hoi_observations()

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

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos

        # # # fixed camera
        # new_cam_target = gymapi.Vec3(0, 0.5, 1.0)
        # new_cam_pos = gymapi.Vec3(1, -1, 1.6)
        # self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)


class HumanoidPhysHOITask(Humanoid_SMPLX):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidPhysHOITask.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.projtype = cfg['env']['projtype']
        self.robot_type = cfg["env"]["asset"]["assetFileName"]
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.reward_weights_p = cfg["env"]["rewardWeights"]["p"]
        self.reward_weights_r = cfg["env"]["rewardWeights"]["r"]
        self.reward_weights_pv = cfg["env"]["rewardWeights"]["pv"]
        self.reward_weights_rv = cfg["env"]["rewardWeights"]["rv"]
        self.reward_weights_op = cfg["env"]["rewardWeights"]["op"]
        self.reward_weights_or = cfg["env"]["rewardWeights"]["or"]
        self.reward_weights_opv = cfg["env"]["rewardWeights"]["opv"]
        self.reward_weights_orv = cfg["env"]["rewardWeights"]["orv"]
        self.reward_weights_ig = cfg["env"]["rewardWeights"]["ig"]
        self.reward_weights_cg1 = cfg["env"]["rewardWeights"]["cg1"]
        self.reward_weights_cg2 = cfg["env"]["rewardWeights"]["cg2"]

        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']

        super().__init__(cfg=cfg,
                         rl_device=rl_device,
                         sim_device=sim_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless,
                         virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

        if rf.oslab.is_absl_path(self.motion_file):
            motion_file_path = self.motion_file
        elif self.motion_file.split("/")[0] == "examples":
            motion_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../../../../" + self.motion_file,
            )
        else:
            raise ValueError("Unsupported motion file path")
        self.motion_file = motion_file_path
        self._load_motion(self.motion_file)

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._build_target_tensors()
        # self._build_marker_state_tensors()

        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj_tensors()

    def post_physics_step(self):
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._update_proj()

        super().post_physics_step()

        self._update_hist_hoi_obs()
        self._compute_hoi_observations()

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

    def _load_motion(self, motion_file):

        # '''load HOI dataset'''
        self.num_motions = 1
        data_path = motion_file
        self.hoi_data_dict = {}

        loaded_dict = {}
        hoi_data = torch.load(data_path)
        loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

        # '''change the data framerate'''
        # NOTE: this is used for temporary testing, and is not rigorous that may yield incorrect rotations.
        dataFramesScale = self.cfg["env"]["dataFramesScale"]
        scale_hoi_data = torch.nn.functional.interpolate(loaded_dict['hoi_data'].unsqueeze(
            1).transpose(0, 2), scale_factor=dataFramesScale, mode='linear', align_corners=True)
        loaded_dict['hoi_data'] = scale_hoi_data.transpose(0, 2).squeeze(1).clone().contiguous()

        self.max_episode_length = loaded_dict['hoi_data'].shape[0]
        self.fps_data = 30.
        # self.fps_data = self.cfg["env"]["dataFPS"]*dataFramesScale

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:, :].clone() -
                                       loaded_dict['root_pos'][:-1, :].clone()) * self.fps_data
        loaded_dict['root_pos_vel'] = torch.cat(
            (torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'), loaded_dict['root_pos_vel']), dim=0)

        loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:6].clone()
        loaded_dict['root_rot_data'] = loaded_dict['root_rot'].clone()
        loaded_dict['root_rot_vel'] = (loaded_dict['root_rot'][1:, :].clone() -
                                       loaded_dict['root_rot'][:-1, :].clone()) * self.fps_data
        loaded_dict['root_rot_vel'] = torch.cat(
            (torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'), loaded_dict['root_rot_vel']), dim=0)
        loaded_dict['root_rot'] = torch_utils.exp_map_to_quat(loaded_dict['root_rot']).clone()

        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9 + 153].clone()
        loaded_dict['dof_pos_vel'] = (loaded_dict['dof_pos'][1:, :].clone() -
                                      loaded_dict['dof_pos'][:-1, :].clone()) * self.fps_data
        loaded_dict['dof_pos_vel'] = torch.cat(
            (torch.zeros((1, loaded_dict['dof_pos_vel'].shape[-1])).to('cuda'), loaded_dict['dof_pos_vel']), dim=0)

        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162 + 52 * 3].clone().view(self.max_episode_length,
                                                                                             52, 3)
        loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:,
                                      self._key_body_ids, :].view(self.max_episode_length, -1).clone()
        loaded_dict['key_body_pos_vel'] = (loaded_dict['key_body_pos'][1:, :].clone() -
                                           loaded_dict['key_body_pos'][:-1, :].clone()) * self.fps_data
        loaded_dict['key_body_pos_vel'] = torch.cat(
            (torch.zeros((1, loaded_dict['key_body_pos_vel'].shape[-1])).to('cuda'), loaded_dict['key_body_pos_vel']),
            dim=0)

        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()
        loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:, :].clone() -
                                      loaded_dict['obj_pos'][:-1, :].clone()) * self.fps_data
        if self.init_vel:
            loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1], loaded_dict['obj_pos_vel']), dim=0)
        else:
            loaded_dict['obj_pos_vel'] = torch.cat(
                (torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to('cuda'), loaded_dict['obj_pos_vel']), dim=0)

        loaded_dict['obj_rot'] = -loaded_dict['hoi_data'][:, 321:324].clone()
        loaded_dict['obj_rot_vel'] = (loaded_dict['obj_rot'][1:, :].clone() -
                                      loaded_dict['obj_rot'][:-1, :].clone()) * self.fps_data
        loaded_dict['obj_rot_vel'] = torch.cat(
            (torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to('cuda'), loaded_dict['obj_rot_vel']), dim=0)
        loaded_dict['obj_rot'] = torch_utils.exp_map_to_quat(-loaded_dict['hoi_data'][:, 321:324]).clone()

        loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())

        loaded_dict['hoi_data'] = torch.cat((
            loaded_dict['root_pos'].clone(),
            loaded_dict['root_rot'].clone(),
            loaded_dict['dof_pos'].clone(),
            loaded_dict['dof_pos_vel'].clone(),
            loaded_dict['obj_pos'].clone(),
            loaded_dict['obj_rot'].clone(),
            loaded_dict['obj_pos_vel'].clone(),
            loaded_dict['key_body_pos'][:, :].clone(),
            loaded_dict['contact'].clone()
        ), dim=-1)

        assert (self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])

        self.hoi_data_dict[0] = loaded_dict

    def _update_marker(self):

        self._marker_states[:, :3] = self.hoi_data_dict[0]['obj_pos'][self.progress_buf, :]  # .clone()
        self._marker_states[:, 3:7] = self.hoi_data_dict[0]['obj_rot'][self.progress_buf, :]  # .clone() #rand_rot
        self._marker_states[:, 7:10] = self.hoi_data_dict[0]['obj_pos_vel'][self.progress_buf, :]  # .clone()
        self._marker_states[:, 10:13] = self.hoi_data_dict[0]['obj_rot_vel'][self.progress_buf, :]  # .clone()

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids),
                                                     len(self._marker_actor_ids))

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        # self._marker_handles = []
        # self._load_marker_asset()
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._proj_handles = []
            self._load_proj_asset()
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        # self._build_marker(env_id, env_ptr)
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)

    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose,
                                                "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)
            self.gym.set_actor_scale(env_ptr, proj_handle, 1)

    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40

        num_actors = self.get_num_actors_per_env()
        num_objs = len(PERTURB_OBJS)
        self._proj_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - num_objs):, :]

        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + \
                               np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        self._calc_perturb_times()

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")

    def _load_proj_asset(self):
        asset_root = "physhoi/data/assets/mjcf/"

        small_asset_file = "block_projectile.urdf"
        # small_asset_file = "ball_medium.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

    def _load_marker_asset(self):
        asset_root = "physhoi/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _load_target_asset(self):  # smplx
        rofunc_path = get_rofunc_path()
        asset_root = os.path.join(rofunc_path, "simulator/assets")
        asset_file = "mjcf/basketball/ball.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 1
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.disable_gravity = True
        # asset_options.fix_base_link = True

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose,
                                              "target", col_group, col_filter, segmentation_id)

        # set ball color
        # if self.cfg["headless"] == False:
        # self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.5, 1.5, 1.5))
        # # gymapi.Vec3(0., 1.0, 1.5))
        #
        # rofunc_path = get_rofunc_path()
        # asset_root = os.path.join(rofunc_path, "simulator/assets")
        # texture_file = "mjcf/basketball/basketball.png"
        # h = self.gym.create_texture_from_file(self.sim, os.path.join(asset_root, texture_file))
        # self.gym.set_rigid_body_texture(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, h)

        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()

        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose,
                                              "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_actor_ids = to_torch(num_actors * np.arange(self.num_envs),
                                          device=self.device, dtype=torch.int32) + 2

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.hoi_data_dict[0]['obj_pos'][self.motion_times, :]  # .clone()+0.5
        self._target_states[env_ids, 3:7] = self.hoi_data_dict[0]['obj_rot'][self.motion_times, :]  # .clone() #rand_rot
        self._target_states[env_ids, 7:10] = self.hoi_data_dict[0]['obj_pos_vel'][self.motion_times, :]  # .clone()
        self._target_states[env_ids, 10:13] = self.hoi_data_dict[0]['obj_rot_vel'][self.motion_times, :]  # .clone()

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().reset_idx(env_ids)

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidPhysHOITask.StateInit.Default:
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidPhysHOITask.StateInit.Start
              or self._state_init == HumanoidPhysHOITask.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidPhysHOITask.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self._reset_target(env_ids)

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        if (self._state_init == HumanoidPhysHOITask.StateInit.Random
                or self._state_init == HumanoidPhysHOITask.StateInit.Hybrid):
            motion_times = torch.randint(
                0, self.hoi_data_dict[0]['hoi_data'].shape[0] - 2, (num_envs,), device=self.device, dtype=torch.long)
        elif (self._state_init == HumanoidPhysHOITask.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)  # .int()

        self.motion_times = motion_times.clone()

        # TODO: i should has shape of env_ids
        i = random.randint(0, self.num_motions - 1)

        self._set_env_state(env_ids=env_ids,
                            root_pos=self.hoi_data_dict[i]['root_pos'][motion_times, :].clone(),
                            root_rot=self.hoi_data_dict[i]['root_rot'][motion_times, :].clone(),
                            dof_pos=self.hoi_data_dict[i]['dof_pos'][motion_times, :].clone(),
                            root_vel=self.hoi_data_dict[i]['root_pos_vel'][motion_times, :].clone(),
                            root_ang_vel=self.hoi_data_dict[i]['root_rot_vel'][motion_times, :].clone(),
                            dof_vel=self.hoi_data_dict[i]['dof_pos_vel'][motion_times, :].clone(),
                            )

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        # diffvel, set 0 for the first frame
        hist_dof_pos = self._hist_obs[:, 7:7 + 153]
        dof_diffvel = (self._dof_pos - hist_dof_pos) * self.fps_data
        dof_diffvel = dof_diffvel * (self.progress_buf != 1).to(float).unsqueeze(dim=-1)

        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                       self._rigid_body_rot[:, 0, :],
                                                       self._rigid_body_vel[:, 0, :],
                                                       self._rigid_body_ang_vel[:, 0, :],
                                                       self._dof_pos, self._dof_vel, key_body_pos,
                                                       self._local_root_obs, self._root_height_obs,
                                                       self._dof_obs_size, self._target_states,
                                                       dof_diffvel)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                             self._rigid_body_rot[env_ids][:, 0, :],
                                                             self._rigid_body_vel[env_ids][:, 0, :],
                                                             self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                             self._dof_pos[env_ids], self._dof_vel[env_ids],
                                                             key_body_pos[env_ids],
                                                             self._local_root_obs, self._root_height_obs,
                                                             self._dof_obs_size, self._target_states[env_ids],
                                                             dof_diffvel[env_ids])

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

    def _update_proj(self):

        if self.projtype == 'Auto':
            curr_timestep = self.progress_buf.cpu().numpy()[0]
            curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
            perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]

            if (len(perturb_step) > 0):
                perturb_id = perturb_step[0]
                n = self.num_envs
                humanoid_root_pos = self._humanoid_root_states[..., 0:3]

                rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
                rand_theta *= 2 * np.pi
                rand_dist = (self._proj_dist_max - self._proj_dist_min) * \
                            torch.rand([n], dtype=self._proj_states.dtype,
                                       device=self._proj_states.device) + self._proj_dist_min
                pos_x = rand_dist * torch.cos(rand_theta)
                pos_y = -rand_dist * torch.sin(rand_theta)
                pos_z = (self._proj_h_max - self._proj_h_min) * \
                        torch.rand([n], dtype=self._proj_states.dtype,
                                   device=self._proj_states.device) + self._proj_h_min

                self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + pos_x
                self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + pos_y
                self._proj_states[..., perturb_id, 2] = pos_z
                self._proj_states[..., perturb_id, 3:6] = 0.0
                self._proj_states[..., perturb_id, 6] = 1.0

                tar_body_idx = np.random.randint(self.num_bodies)
                tar_body_idx = 1

                launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
                launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
                launch_dir += 0.1 * torch.randn_like(launch_dir)
                launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
                launch_speed = (self._proj_speed_max - self._proj_speed_min) * \
                               torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
                launch_vel = launch_speed * launch_dir
                launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
                self._proj_states[..., perturb_id, 7:10] = launch_vel
                self._proj_states[..., perturb_id, 10:13] = 0.0

                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                             gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                             len(self._proj_actor_ids))

        elif self.projtype == 'Mouse':
            # mouse control
            for evt in self.gym.query_viewer_action_events(self.viewer):

                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(self.sim, self._proj_states, gymapi.STATE_ALL)

                elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
                    if evt.action == "mouse_shoot":
                        pos = self.gym.get_viewer_mouse_position(self.viewer)
                        window_size = self.gym.get_viewer_size(self.viewer)
                        xcoord = round(pos.x * window_size.x)
                        ycoord = round(pos.y * window_size.y)
                        print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

                    cam_pose = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

                    spawn = cam_pose.p
                    speed = 25
                    vel = cam_fwd * speed

                    angvel = 1.57 - 3.14 * np.random.random(3)

                    self._proj_states[..., 0] = spawn.x
                    self._proj_states[..., 1] = spawn.y
                    self._proj_states[..., 2] = spawn.z
                    self._proj_states[..., 7] = vel.x
                    self._proj_states[..., 8] = vel.y
                    self._proj_states[..., 9] = vel.z

                    self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                                 gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                                 len(self._proj_actor_ids))

    def play_dataset_step(self, time):

        t = time

        ### update object ###
        self._target_states[:, :3] = self.hoi_data_dict[0]['obj_pos'][t, :]
        self._target_states[:, 3:7] = self.hoi_data_dict[0]['obj_rot'][t, :]
        self._target_states[:, 7:10] = torch.zeros_like(self._target_states[:, 7:10])
        self._target_states[:, 10:13] = torch.zeros_like(self._target_states[:, 10:13])

        ### update subject ###
        _humanoid_root_pos = self.hoi_data_dict[0]['root_pos'][t, :].clone()
        _humanoid_root_rot = self.hoi_data_dict[0]['root_rot'][t, :].clone()
        self._humanoid_root_states[:, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[:, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])

        self._dof_pos[:] = self.hoi_data_dict[0]['dof_pos'][t, :].clone()
        self._dof_vel[:] = torch.zeros_like(self._dof_vel[:])

        contact = self.hoi_data_dict[0]['contact'][t, :]

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._refresh_sim_tensors()

        # ### draw contact label ###
        obj_contact = torch.any(contact > 0.1, dim=-1)
        for env_id, env_ptr in enumerate(self.envs):
            env_ptr = self.envs[env_id]
            handle = self._target_handles[env_id]

            if obj_contact == True:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(1., 0., 0.))
            else:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(0., 1., 0.))

        self.render(t=t)
        self.gym.simulate(self.sim)

    def get_dataset_step(self, time):
        t = time

        ### update object ###
        self._target_states[:, :3] = self.hoi_data_dict[0]['obj_pos'][t, :]
        self._target_states[:, 3:7] = self.hoi_data_dict[0]['obj_rot'][t, :]
        self._target_states[:, 7:10] = torch.zeros_like(self._target_states[:, 7:10])
        self._target_states[:, 10:13] = torch.zeros_like(self._target_states[:, 10:13])

        ### update subject ###
        _humanoid_root_pos = self.hoi_data_dict[0]['root_pos'][t, :].clone()
        _humanoid_root_rot = self.hoi_data_dict[0]['root_rot'][t, :].clone()
        self._humanoid_root_states[:, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[:, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])

        self._dof_pos[:] = self.hoi_data_dict[0]['dof_pos'][t, :].clone()
        self._dof_vel[:] = torch.zeros_like(self._dof_vel[:])

        contact = self.hoi_data_dict[0]['contact'][t, :]

        return self._dof_state

    def _draw_task_play(self, t):
        # self._update_marker()

        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # color

        self.gym.clear_lines(self.viewer)

        starts = self.hoi_data_dict[0]['hoi_data'][t, :3]

        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self.hoi_data_dict[0]['key_body_pos'][t, j * 3:j * 3 + 3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            # self._draw_task()

            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]
                dataname = self.motion_file[len('physhoi/data/motions/BallPlay/'):-3]
                rgb_filename = "physhoi/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("physhoi/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)

    # def _draw_task(self):
    #     # self._update_marker()
    #
    #     # # draw obj contact
    #     # obj_contact = torch.any(torch.abs(self._tar_contact_forces[..., 0:2]) > 0.1, dim=-1)
    #     # for env_id, env_ptr in enumerate(self.envs):
    #     #     env_ptr = self.envs[env_id]
    #     #     handle = self._target_handles[env_id]
    #
    #     #     if obj_contact[env_id] == True:
    #     #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
    #     #                                     gymapi.Vec3(1., 0., 0.))
    #     #     else:
    #     #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
    #     #                                     gymapi.Vec3(0., 1., 0.))


#####################################################################
### =========================jit functions=========================###
#####################################################################

#@torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                           local_root_obs, root_height_obs, dof_obs_size, target_states, dof_diffvel):
    contact = torch.zeros(key_body_pos.shape[0], 1).cuda()
    obs = torch.cat((root_pos, root_rot, dof_pos, dof_diffvel, target_states[:, :10], key_body_pos.contiguous(
    ).view(-1, key_body_pos.shape[1] * key_body_pos.shape[2]), contact), dim=-1)
    return obs


@torch.jit.script
def compute_obj_observations(root_states, tar_states):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs,
                                      contact_forces, contact_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                  heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone().view(contact_forces.shape[0], -1)

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs,
                     local_body_vel, local_body_ang_vel, body_contact_buf), dim=-1)
    return obs


def compute_humanoid_reward(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, len_keypos, w):
    ### data preprocess ###

    # simulated states
    root_pos = hoi_obs[:, :3]
    root_rot = hoi_obs[:, 3:3 + 4]
    dof_pos = hoi_obs[:, 7:7 + 51 * 3]
    dof_pos_vel = hoi_obs[:, 160:160 + 51 * 3]
    obj_pos = hoi_obs[:, 313:313 + 3]
    obj_rot = hoi_obs[:, 316:316 + 4]
    obj_pos_vel = hoi_obs[:, 320:320 + 3]
    key_pos = hoi_obs[:, 323:323 + len_keypos * 3]
    contact = hoi_obs[:, -1:]  # fake one
    key_pos = torch.cat((root_pos, key_pos), dim=-1)
    body_rot = torch.cat((root_rot, dof_pos), dim=-1)
    ig = key_pos.view(-1, len_keypos + 1, 3).transpose(0, 1) - obj_pos[:, :3]
    ig = ig.transpose(0, 1).view(-1, (len_keypos + 1) * 3)

    # reference states
    ref_root_pos = hoi_ref[:, :3]
    ref_root_rot = hoi_ref[:, 3:3 + 4]
    ref_dof_pos = hoi_ref[:, 7:7 + 51 * 3]
    ref_dof_pos_vel = hoi_ref[:, 160:160 + 51 * 3]
    ref_obj_pos = hoi_ref[:, 313:313 + 3]
    ref_obj_rot = hoi_ref[:, 316:316 + 4]
    ref_obj_pos_vel = hoi_ref[:, 320:320 + 3]
    ref_key_pos = hoi_ref[:, 323:323 + len_keypos * 3]
    ref_obj_contact = hoi_ref[:, -1:]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos), dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos), dim=-1)
    ref_ig = ref_key_pos.view(-1, len_keypos + 1, 3).transpose(0, 1) - ref_obj_pos[:, :3]
    ref_ig = ref_ig.transpose(0, 1).view(-1, (len_keypos + 1) * 3)

    ### body reward ###

    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos) ** 2, dim=-1)
    rp = torch.exp(-ep * w['p'])

    # body rot reward
    er = torch.mean((ref_body_rot - body_rot) ** 2, dim=-1)
    rr = torch.exp(-er * w['r'])

    # body pos vel reward
    epv = torch.zeros_like(ep)
    rpv = torch.exp(-epv * w['pv'])

    # body rot vel reward
    erv = torch.mean((ref_dof_pos_vel - dof_pos_vel) ** 2, dim=-1)
    rrv = torch.exp(-erv * w['rv'])

    rb = rp * rr * rpv * rrv

    ### object reward ###

    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos) ** 2, dim=-1)
    rop = torch.exp(-eop * w['op'])

    # object rot reward
    eor = torch.zeros_like(ep)  # torch.mean((ref_obj_rot - obj_rot)**2,dim=-1)
    ror = torch.exp(-eor * w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel) ** 2, dim=-1)
    ropv = torch.exp(-eopv * w['opv'])

    # object rot vel reward
    eorv = torch.zeros_like(ep)  # torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv * w['orv'])

    ro = rop * ror * ropv * rorv

    ### interaction graph reward ###

    eig = torch.mean((ref_ig - ig) ** 2, dim=-1)
    rig = torch.exp(-eig * w['ig'])

    ### simplified contact graph reward ###

    # Since Isaac Gym does not yet provide API for detailed collision detection in GPU pipeline,
    # we use force detection to approximate the contact status.
    # In this case we use the CG node istead of the CG edge for imitation.
    # TODO: update the code once collision detection API is available.

    ## body ids
    # Pelvis, 0
    # L_Hip, 1
    # L_Knee, 2
    # L_Ankle, 3
    # L_Toe, 4
    # R_Hip, 5
    # R_Knee, 6
    # R_Ankle, 7
    # R_Toe, 8
    # Torso, 9
    # Spine, 10
    # Chest, 11
    # Neck, 12
    # Head, 13
    # L_Thorax, 14
    # L_Shoulder, 15
    # L_Elbow, 16
    # L_Wrist, 17
    # L_Hand, 18-32
    # R_Thorax, 33
    # R_Shoulder, 34
    # R_Elbow, 35
    # R_Wrist, 36
    # R_Hand, 37-51

    # body contact
    contact_body_ids = [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35]
    body_contact_buf = contact_buf[:, contact_body_ids, :].clone()
    body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)
    body_contact = torch.all(body_contact, dim=-1).to(float)  # =1 when no contact happens to the body

    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(
        float)  # =1 when contact happens to the object

    ref_body_contact = torch.ones_like(ref_obj_contact)  # no body contact for all time
    ecg1 = torch.abs(body_contact - ref_body_contact[:, 0])
    rcg1 = torch.exp(-ecg1 * w['cg1'])
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:, 0])
    rcg2 = torch.exp(-ecg2 * w['cg2'])

    rcg = rcg1 * rcg2

    ### task-agnostic HOI imitation reward ###
    reward = rb * ro * rig * rcg

    return reward

# @torch.jit.script
# def compute_humanoid_reward(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, len_keypos, w_p, w_r, w_pv, w_rv, w_op,
#                             w_or, w_opv, w_orv, w_ig, w_cg1, w_cg2):
#     # type: (Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, float, float, float) -> Tensor
#
#     ### data preprocess ###
#
#     # simulated states
#     root_pos = hoi_obs[:, :3]
#     root_rot = hoi_obs[:, 3:3 + 4]
#     dof_pos = hoi_obs[:, 7:7 + 51 * 3]
#     dof_pos_vel = hoi_obs[:, 160:160 + 51 * 3]
#     obj_pos = hoi_obs[:, 313:313 + 3]
#     obj_rot = hoi_obs[:, 316:316 + 4]
#     obj_pos_vel = hoi_obs[:, 320:320 + 3]
#     key_pos = hoi_obs[:, 323:323 + len_keypos * 3]
#     contact = hoi_obs[:, -1:]  # fake one
#     key_pos = torch.cat((root_pos, key_pos), dim=-1)
#     body_rot = torch.cat((root_rot, dof_pos), dim=-1)
#     ig = key_pos.view(-1, len_keypos + 1, 3).transpose(0, 1) - obj_pos[:, :3]
#     ig = ig.transpose(0, 1).view(-1, (len_keypos + 1) * 3)
#
#     # reference states
#     ref_root_pos = hoi_ref[:, :3]
#     ref_root_rot = hoi_ref[:, 3:3 + 4]
#     ref_dof_pos = hoi_ref[:, 7:7 + 51 * 3]
#     ref_dof_pos_vel = hoi_ref[:, 160:160 + 51 * 3]
#     ref_obj_pos = hoi_ref[:, 313:313 + 3]
#     ref_obj_rot = hoi_ref[:, 316:316 + 4]
#     ref_obj_pos_vel = hoi_ref[:, 320:320 + 3]
#     ref_key_pos = hoi_ref[:, 323:323 + len_keypos * 3]
#     ref_obj_contact = hoi_ref[:, -1:]
#     ref_key_pos = torch.cat((ref_root_pos, ref_key_pos), dim=-1)
#     ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos), dim=-1)
#     ref_ig = ref_key_pos.view(-1, len_keypos + 1, 3).transpose(0, 1) - ref_obj_pos[:, :3]
#     ref_ig = ref_ig.transpose(0, 1).view(-1, (len_keypos + 1) * 3)
#
#     ### body reward ###
#
#     # body pos reward
#     ep = torch.mean((ref_key_pos - key_pos) ** 2, dim=-1)
#     rp = torch.exp(-ep * w_p)
#
#     # body rot reward
#     er = torch.mean((ref_body_rot - body_rot) ** 2, dim=-1)
#     rr = torch.exp(-er * w_r)
#
#     # body pos vel reward
#     epv = torch.zeros_like(ep)
#     rpv = torch.exp(-epv * w_pv)
#
#     # body rot vel reward
#     erv = torch.mean((ref_dof_pos_vel - dof_pos_vel) ** 2, dim=-1)
#     rrv = torch.exp(-erv * w_rv)
#
#     rb = rp * rr * rpv * rrv
#
#     ### object reward ###
#
#     # object pos reward
#     eop = torch.mean((ref_obj_pos - obj_pos) ** 2, dim=-1)
#     rop = torch.exp(-eop * w_op)
#
#     # object rot reward
#     eor = torch.zeros_like(ep)  # torch.mean((ref_obj_rot - obj_rot)**2,dim=-1)
#     ror = torch.exp(-eor * w_or)
#
#     # object pos vel reward
#     eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel) ** 2, dim=-1)
#     ropv = torch.exp(-eopv * w_opv)
#
#     # object rot vel reward
#     eorv = torch.zeros_like(ep)  # torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
#     rorv = torch.exp(-eorv * w_orv)
#
#     ro = rop * ror * ropv * rorv
#
#     ### interaction graph reward ###
#
#     eig = torch.mean((ref_ig - ig) ** 2, dim=-1)
#     rig = torch.exp(-eig * w_ig)
#
#     ### simplified contact graph reward ###
#
#     # Since Isaac Gym does not yet provide API for detailed collision detection in GPU pipeline,
#     # we use force detection to approximate the contact status.
#     # In this case we use the CG node istead of the CG edge for imitation.
#     # TODO: update the code once collision detection API is available.
#
#     # body ids
#     # Pelvis, 0
#     # L_Hip, 1
#     # L_Knee, 2
#     # L_Ankle, 3
#     # L_Toe, 4
#     # R_Hip, 5
#     # R_Knee, 6
#     # R_Ankle, 7
#     # R_Toe, 8
#     # Torso, 9
#     # Spine, 10
#     # Chest, 11
#     # Neck, 12
#     # Head, 13
#     # L_Thorax, 14
#     # L_Shoulder, 15
#     # L_Elbow, 16
#     # L_Wrist, 17
#     # L_Hand, 18-32
#     # R_Thorax, 33
#     # R_Shoulder, 34
#     # R_Elbow, 35
#     # R_Wrist, 36
#     # R_Hand, 37-51
#
#     # body contact
#     contact_body_ids = [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35]
#     body_contact_buf = contact_buf[:, contact_body_ids, :].clone()
#     body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)
#     body_contact = torch.all(body_contact, dim=-1)  # =1 when no contact happens to the body
#
#     # object contact
#     # =1 when contact happens to the object
#     obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1)
#
#     ref_body_contact = torch.ones_like(ref_obj_contact)  # no body contact for all time
#     ecg1 = torch.abs(body_contact - ref_body_contact[:, 0])
#     rcg1 = torch.exp(-ecg1 * w_cg1)
#     ecg2 = torch.abs(obj_contact - ref_obj_contact[:, 0])
#     rcg2 = torch.exp(-ecg2 * w_cg2)
#
#     rcg = rcg1 * rcg2
#
#     ### task-agnostic HOI imitation reward ###
#     reward = rb * ro * rig * rcg
#
#     return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2]  # root height
        body_fall = body_height < termination_heights  # [4096]
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
