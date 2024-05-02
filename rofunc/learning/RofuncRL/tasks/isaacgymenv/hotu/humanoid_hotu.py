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
import torch
from gym import spaces
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from rofunc.config.utils import get_sim_config
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils

from .humanoid import Humanoid, dof_to_obs
from .motion_lib import MotionLib, ObjectMotionLib


class HumanoidHOTUTask(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.override_unuse_actions = True

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidHOTUTask.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        self.humanoid_asset_infos = get_sim_config(sim_name="Humanoid_info")["Model_type"]
        self.humanoid_asset_file = self.cfg["env"]["asset"]["assetFileName"]
        self.mf_humanoid_asset_file = self.cfg["env"]["motion_file_asset"]
        self.humanoid_info = self._get_humanoid_info(self.humanoid_asset_file)
        self.mf_humanoid_info = self._get_humanoid_info(self.mf_humanoid_asset_file)

        self.wb_decompose = cfg["task"].get("wb_decompose", False)
        self.use_synergy = cfg["task"].get("use_synergy", False)
        if self.wb_decompose:
            self.parts = cfg["task"]["wb_decompose_parameter"]["parts"]
            self.num_parts = len(self.parts)
            self.whole_rb_dict = self.humanoid_info["rigid_bodies"]
            self.whole_rb_names = list(self.whole_rb_dict.keys())
            self.del_rb_names = self.humanoid_info["del_rb"]
            self.whole_rb_names_a_del = [rb_name for rb_name in self.whole_rb_names if rb_name not in self.del_rb_names]

            # wb_decompose_param_rb_ids: list of tensor, each tensor is the rigid body id of the part
            self.wb_decompose_param_rb_ids = [
                torch.tensor(
                    [self.whole_rb_dict[rb_name] for rb_name in self.humanoid_info["parts"][part]["rigid_bodies"]],
                    device=sim_device, dtype=torch.long)
                for part in self.parts]
            # wb_decompose_param_dof_ids: list of tensor, each tensor is the dof id of the part
            self.wb_decompose_param_dof_ids = [
                torch.tensor(
                    [self.humanoid_info["dofs"][dof_name] for dof_name in self.humanoid_info["parts"][part]["dofs"]],
                    device=sim_device, dtype=torch.long)
                for part in self.parts]
            # wb_decompose_param_rb_names: list of list of str, each list is the rigid body name of the part
            self.wb_decompose_param_rb_names = [
                [rb_name for rb_name in self.humanoid_info["parts"][part]["rigid_bodies"]] for part in self.parts]

            # Delete del_rb from wb_decompose_param_rb_names and get the new wb_decompose_param_rb_index
            self.wb_decompose_param_rb_names_a_del = [
                [rb_name for rb_name in rb_names if rb_name not in self.del_rb_names] for rb_names in
                self.wb_decompose_param_rb_names]
            # wb_decompose_param_rb_index: list of tensor, each tensor is the rigid body index of the part
            # Used for dof_obs
            self.wb_decompose_param_rb_index = [
                torch.tensor([self.whole_rb_names_a_del.index(rb_name) for rb_name in rb_names],
                             device=sim_device, dtype=torch.long) for rb_names in
                self.wb_decompose_param_rb_names_a_del]
            # # wb_decompose_param_rb_index_wo_pelvis: list of tensor, each tensor is the rigid body index of the part without pelvis
            # self.wb_decompose_param_rb_index_wo_pelvis = [
            #     torch.tensor([self.whole_rb_names.index(rb_name)-1 for rb_name in rb_names if rb_name != "pelvis" ],
            #                  device=sim_device, dtype=torch.long) for rb_names in self.wb_decompose_param_rb_names]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self._check_humanoid_info(self.humanoid_asset_file, self.humanoid_info)
        self._check_humanoid_info(self.mf_humanoid_asset_file, self.mf_humanoid_info)

        if self.use_synergy:
            self.synergy_action_matrix = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                       [2, 2, 2, 1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2]],
                                                      device=sim_device, dtype=torch.float32)
            self.useful_right_qbhand_dof_index = sorted(
                [value for key, value in self.humanoid_info["dofs"].items() if
                 ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
                 ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key) and (
                         "right_qbhand" in key)])
            self.useful_left_qbhand_dof_index = sorted(
                [value for key, value in self.humanoid_info["dofs"].items() if
                 ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
                 ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key) and (
                         "left_qbhand" in key)])

            self.virtual2real_dof_index_map_dict = {value: self.humanoid_info["dofs"][key.replace("_virtual", "")] for
                                                    key, value in self.humanoid_info["dofs"].items() if
                                                    "virtual" in key}

        # Load motion file
        self._load_motion(cfg["env"].get("motion_file", None))

        # Load object motion file
        self._load_object_motion(cfg["env"].get("object_motion_file", None))

        # Set up amp observation
        if self.wb_decompose:
            self._amp_obs_space = []
            self._amp_obs_buf = []
            self._curr_amp_obs_buf = []
            self._hist_amp_obs_buf = []
            for part_i in range(self.num_parts):
                self._amp_obs_space.append(spaces.Box(np.ones(self.get_num_amp_obs()[part_i]) * -np.Inf,
                                                      np.ones(self.get_num_amp_obs()[part_i]) * np.Inf))
                self._amp_obs_buf.append(
                    torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step[part_i]),
                                device=self.device, dtype=torch.float))
                self._curr_amp_obs_buf.append(self._amp_obs_buf[part_i][:, 0])
                self._hist_amp_obs_buf.append(self._amp_obs_buf[part_i][:, 1:])
            self._set_colors_for_parts()
        else:
            self._amp_obs_space = spaces.Box(np.ones(self.get_num_amp_obs()) * -np.Inf,
                                             np.ones(self.get_num_amp_obs()) * np.Inf)
            self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                            device=self.device, dtype=torch.float)
            self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
            self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

    def _set_colors_for_parts(self):
        colors = [[255 / 255., 165 / 255., 0 / 255.], [0.54, 0.85, 0.2], [0.5, 0.5, 0.5], [0.35, 0.35, 0.35]]
        for part_i in range(self.num_parts):
            self.set_char_color(self.wb_decompose_param_rb_ids[part_i], colors[part_i])

    def _get_dof_action_from_synergy(self, synergy_action, useful_joint_index):
        # the first synergy is 0~1, the second is -1~1
        synergy_action[:, 0] = torch.abs(synergy_action[:, 0])
        dof_action = torch.matmul(synergy_action, self.synergy_action_matrix)
        dof_action = torch.clamp(dof_action, 0, 1.0)
        dof_action = dof_action * 2 - 1  # -1~1

        dof_action = scale(dof_action,
                           self.dof_limits_lower[useful_joint_index],
                           self.dof_limits_upper[useful_joint_index])
        return dof_action

    def pre_physics_step(self, actions):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if self.use_synergy and asset_file == "mjcf/hotu_humanoid_w_qbhand_full.xml":
            right_synergy_action = actions[:, -2 * 2:-2].clone()
            right_dof_action = self._get_dof_action_from_synergy(right_synergy_action,
                                                                 self.useful_right_qbhand_dof_index)
            left_synergy_action = actions[:, -2:].clone()
            left_dof_action = self._get_dof_action_from_synergy(left_synergy_action,
                                                                self.useful_left_qbhand_dof_index)

            expanded_actions = torch.zeros((actions.shape[0], 100), device=self.device)

            j = 0
            for dof, index in self.humanoid_info["dofs"].items():
                if "qbhand" not in dof:
                    expanded_actions[:, index] = actions[:, j]
                    j += 1

            for i, index in enumerate(self.useful_right_qbhand_dof_index):
                expanded_actions[:, index] = right_dof_action[:, i]
            for i, index in enumerate(self.useful_left_qbhand_dof_index):
                expanded_actions[:, index] = left_dof_action[:, i]

            for key, value in self.virtual2real_dof_index_map_dict.items():
                expanded_actions[:, key] = expanded_actions[:, value]
            self.actions = expanded_actions.to(self.device).clone()
        else:
            self.actions = actions.to(self.device).clone()

        if self.override_unuse_actions:
            tmp_action = torch.zeros_like(self.actions).to(self.device)
            tmp_action[:, self.wb_decompose_param_dof_ids[0]] = self.actions[:, self.wb_decompose_param_dof_ids[0]]
            self.actions = tmp_action

        if self._pd_control:
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        if self.wb_decompose:
            amp_obs_flat_list = []
            for part_i in range(self.num_parts):
                amp_obs_flat_list.append(self._amp_obs_buf[part_i].view(-1, self.get_num_amp_obs()[part_i]))
                self.extras[f"amp_obs_{part_i}"] = amp_obs_flat_list[part_i]
        else:
            amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
            self.extras["amp_obs"] = amp_obs_flat
            self.extras["amp_obs_0"] = amp_obs_flat

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * np.array(self._num_amp_obs_per_step)

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            if self.wb_decompose:
                assert self._amp_obs_demo_buf[0].shape[0] == num_samples
            else:
                assert self._amp_obs_demo_buf.shape[0] == num_samples

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)  # (num_envs*num_amp_steps, num_amp_obs)

        if self.wb_decompose:
            amp_obs_demo_flat_list = []
            for part_i in range(self.num_parts):
                self._amp_obs_demo_buf[part_i][:] = amp_obs_demo[part_i].view(self._amp_obs_demo_buf[part_i].shape)
                amp_obs_demo_flat_list.append(self._amp_obs_demo_buf[part_i].view(-1, self.get_num_amp_obs()[part_i]))

            return amp_obs_demo_flat_list
        else:
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
                                              self._dof_obs_size, self._dof_offsets,
                                              )
        if self.wb_decompose:
            amp_obs_demo = amp_obs_decompose(amp_obs_demo, self.wb_decompose_param_rb_index,
                                             self.wb_decompose_param_dof_ids, self.parts[0] == "body")
        else:
            amp_obs_demo = amp_obs_concat(amp_obs_demo)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        if self.wb_decompose:
            self._amp_obs_demo_buf = []
            for part_i in range(self.num_parts):
                self._amp_obs_demo_buf.append(
                    torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step[part_i]),
                                device=self.device, dtype=torch.float32))

        else:
            self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
                                                 device=self.device, dtype=torch.float32)

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        if asset_file == "mjcf/amp_humanoid.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies
        elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies
        elif asset_file in ["mjcf/amp_humanoid_spoon_pan_fixed.xml", "mjcf/hotu_humanoid.xml"]:
            if self.wb_decompose:
                num_amp_obs_per_step_list = []
                for i, part in enumerate(self.parts):
                    num_rb = len(self.wb_decompose_param_rb_index[i])
                    num_dof = len(self.wb_decompose_param_dof_ids[i])
                    if part == "body":  # body
                        num_amp_obs_per_step = 13 + num_rb * 6 + num_dof + 3 * num_key_bodies
                    else:
                        num_amp_obs_per_step = num_rb * 6 + num_dof
                    num_amp_obs_per_step_list.append(num_amp_obs_per_step)
                self._num_amp_obs_per_step = num_amp_obs_per_step_list
            else:
                self._num_amp_obs_per_step = 13 + self._dof_obs_size + 34 + 3 * num_key_bodies
        elif asset_file == "mjcf/hotu_humanoid_w_qbhand.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 64 + 3 * num_key_bodies
        elif asset_file in ["mjcf/hotu_humanoid_w_qbhand_no_virtual.xml",
                            "mjcf/hotu_humanoid_w_qbhand_no_virtual_no_quat.xml"]:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 64 + 3 * num_key_bodies
        elif asset_file == "mjcf/hotu_humanoid_w_qbhand_full.xml":
            if self.wb_decompose:
                num_amp_obs_per_step_list = []
                for i, part in enumerate(self.parts):
                    num_rb = len(self.wb_decompose_param_rb_index[i])
                    num_dof = len(self.wb_decompose_param_dof_ids[i])
                    if part == "body":  # body
                        num_amp_obs_per_step = 13 + num_rb * 6 + num_dof + 3 * num_key_bodies
                    else:
                        num_amp_obs_per_step = num_rb * 6 + num_dof
                    num_amp_obs_per_step_list.append(num_amp_obs_per_step)
                self._num_amp_obs_per_step = num_amp_obs_per_step_list
            else:
                self._num_amp_obs_per_step = 13 + self._dof_obs_size + 100 + 3 * num_key_bodies

        else:
            print(f"Unsupported humanoid body num: {asset_file}")
            assert False

    def _load_motion(self, motion_file_path):
        if rf.oslab.is_absl_path(motion_file_path):
            motion_file = motion_file_path
        elif motion_file_path.split("/")[0] == "examples":
            motion_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../../../../../../" + motion_file_path)
        else:
            raise ValueError("Unsupported motion file path")

        assert self._dof_offsets[-1] == self.num_dof

        self._motion_lib = MotionLib(
            cfg=self.cfg,
            motion_file=motion_file,
            asset_infos=self.humanoid_asset_infos,
            key_body_names=self.key_bodies,
            device=self.device,
            humanoid_type=self.humanoid_info["name"],
            mf_humanoid_type=self.mf_humanoid_info["name"],
        )

    def _load_object_motion(self, object_motion_file_path):
        if object_motion_file_path is not None:
            if rf.oslab.is_absl_path(object_motion_file_path):
                object_motion_file = object_motion_file_path
            elif object_motion_file_path.split("/")[0] == "examples":
                object_motion_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "../../../../../../" + object_motion_file_path)
            else:
                raise ValueError("Unsupported object motion file path")

            self._object_motion_lib = ObjectMotionLib(
                object_motion_file=object_motion_file,
                object_names=self.cfg["env"]["object_asset"]["assetName"],
                device=self.device,
                height_offset=self._motion_lib.humanoid_height_offsets[0]  # TODO: make it for multiple motions
            )

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

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidHOTUTask.StateInit.Default:
            self._reset_default(env_ids)
        elif (
                self._state_init == HumanoidHOTUTask.StateInit.Start
                or self._state_init == HumanoidHOTUTask.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidHOTUTask.StateInit.Hybrid:
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

        if (self._state_init == HumanoidHOTUTask.StateInit.Random
                or self._state_init == HumanoidHOTUTask.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HumanoidHOTUTask.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert False, f"Unsupported state initialization strategy: {self._state_init}"

        (root_pos,
         root_rot,
         dof_pos,
         root_vel,
         root_ang_vel,
         dof_vel,
         key_pos,
         _, _) = self._motion_lib.get_motion_state(motion_ids, motion_times)
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
        if self.wb_decompose:
            for part_i in range(self.num_parts):
                curr_amp_obs = self._curr_amp_obs_buf[part_i][env_ids].unsqueeze(-2)
                self._hist_amp_obs_buf[part_i][env_ids] = curr_amp_obs
        else:
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
            _, _
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
        if self.wb_decompose:
            amp_obs_demo = amp_obs_decompose(amp_obs_demo, self.wb_decompose_param_rb_index,
                                             self.wb_decompose_param_dof_ids, self.parts[0] == "body")
        else:
            amp_obs_demo = amp_obs_concat(amp_obs_demo)

        if self.wb_decompose:
            for part_i in range(len(self.parts)):
                self._hist_amp_obs_buf[part_i][env_ids] = amp_obs_demo[part_i].view(
                    self._hist_amp_obs_buf[part_i][env_ids].shape
                )
        else:
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
        if self.wb_decompose:
            for part_i in range(self.num_parts):
                if env_ids is None:
                    for i in reversed(range(self._amp_obs_buf[part_i].shape[1] - 1)):
                        self._amp_obs_buf[part_i][:, i + 1] = self._amp_obs_buf[part_i][:, i]
                else:
                    for i in reversed(range(self._amp_obs_buf[part_i].shape[1] - 1)):
                        self._amp_obs_buf[part_i][env_ids, i + 1] = self._amp_obs_buf[part_i][env_ids, i]
        else:
            if env_ids is None:
                for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                    self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
            else:
                for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                    self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if env_ids is None:
            amp_obs = build_amp_observations(
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
                self._dof_offsets
            )
        else:
            amp_obs = build_amp_observations(
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
        if self.wb_decompose:
            amp_obs = amp_obs_decompose(amp_obs, self.wb_decompose_param_rb_index,
                                        self.wb_decompose_param_dof_ids, self.parts[0] == "body")
        else:
            amp_obs = amp_obs_concat(amp_obs)

        if self.wb_decompose:
            if env_ids is None:
                for part_i in range(len(self.parts)):
                    self._curr_amp_obs_buf[part_i][:] = amp_obs[part_i]
            else:
                for part_i in range(len(self.parts)):
                    self._curr_amp_obs_buf[part_i][env_ids] = amp_obs[part_i]

        else:
            if env_ids is None:
                self._curr_amp_obs_buf[:] = amp_obs
            else:
                self._curr_amp_obs_buf[env_ids] = amp_obs

    def amp_obs_decompose(self, amp_obs, wb_decompose_param_rb_index, wb_decompose_param_dof_ids, first_is_body):
        root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos = amp_obs
        amp_obs_list = []
        for part_i in range(len(wb_decompose_param_dof_ids)):
            dof_obs_size_this_part = len(wb_decompose_param_dof_ids[part_i])
            dof_obs_this_part = torch.zeros(root_h_obs.shape[:-1] + (dof_obs_size_this_part,),
                                            device=root_h_obs.device)
            i = 0
            for j in wb_decompose_param_dof_ids[part_i]:
                try:
                    dof_obs_this_part[:, i: i + 1] = dof_obs[:, j: (j + 1)]
                except:
                    pass
                i += 1
            dof_vel_this_part = dof_vel[:, wb_decompose_param_dof_ids[part_i]]
            if part_i == 0 and first_is_body:
                amp_obs_this_part = torch.cat(
                    (
                        root_h_obs,  # (num_envs, 1)
                        root_rot_obs,  # (num_envs, 6)
                        local_root_vel,  # (num_envs, 3)
                        local_root_ang_vel,  # (num_envs, 3)
                        dof_obs_this_part,  # (num_envs, 84)
                        dof_vel_this_part,  # (num_envs, 34)
                        flat_local_key_pos,  # (num_envs, 3 * num_key_body)
                    ),
                    dim=-1,
                )
            else:
                amp_obs_this_part = torch.cat(
                    (
                        dof_obs_this_part,  # (num_envs, 180)
                        dof_vel_this_part,  # (num_envs, 30)
                    ),
                    dim=-1,
                )
            amp_obs_list.append(amp_obs_this_part)
        return amp_obs_list


@torch.jit.script
def amp_obs_concat(amp_obs):
    # type: (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor
    root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos = amp_obs
    return torch.cat(
        (
            root_h_obs,  # (num_envs, 1)
            root_rot_obs,  # (num_envs, 6)
            local_root_vel,  # (num_envs, 3)
            local_root_ang_vel,  # (num_envs, 3)
            dof_obs,  # (num_envs, 264)
            dof_vel,  # (num_envs, 64)
            flat_local_key_pos,  # (num_envs, 3 * num_key_body)
        ),
        dim=-1,
    )


@torch.jit.script
def amp_obs_decompose(amp_obs, wb_decompose_param_rb_index, wb_decompose_param_dof_ids, first_is_body):
    # type: (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], List[Tensor], List[Tensor], bool) -> List[Tensor]
    root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos = amp_obs
    amp_obs_list = []
    for part_i in range(len(wb_decompose_param_rb_index)):
        dof_obs_size_this_part = len(wb_decompose_param_rb_index[part_i]) * 6
        dof_obs_this_part = torch.zeros(root_h_obs.shape[:-1] + (dof_obs_size_this_part,),
                                        device=root_h_obs.device)
        i = 0
        for j in wb_decompose_param_rb_index[part_i]:
            dof_obs_this_part[:, (i * 6): ((i + 1) * 6)] = dof_obs[:, (j * 6): ((j + 1) * 6)]
            i += 1
        dof_vel_this_part = dof_vel[:, wb_decompose_param_dof_ids[part_i]]
        if part_i == 0 and first_is_body:
            amp_obs_this_part = torch.cat(
                (
                    root_h_obs,  # (num_envs, 1)
                    root_rot_obs,  # (num_envs, 6)
                    local_root_vel,  # (num_envs, 3)
                    local_root_ang_vel,  # (num_envs, 3)
                    dof_obs_this_part,  # (num_envs, 84)
                    dof_vel_this_part,  # (num_envs, 34)
                    flat_local_key_pos,  # (num_envs, 3 * num_key_body)
                ),
                dim=-1,
            )
        else:
            amp_obs_this_part = torch.cat(
                (
                    dof_obs_this_part,  # (num_envs, 180)
                    dof_vel_this_part,  # (num_envs, 30)
                ),
                dim=-1,
            )
        amp_obs_list.append(amp_obs_this_part)
    return amp_obs_list


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
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> List[Tensor]

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

    return root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos
