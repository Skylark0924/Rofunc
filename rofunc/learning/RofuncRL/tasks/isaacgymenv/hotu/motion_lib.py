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
import time
from collections import OrderedDict

import numpy as np
import tqdm
import yaml

import rofunc as rf
from rofunc.learning.RofuncRL.tasks.utils import torch_jit_utils as torch_utils
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonMotion

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy


    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)


    torch.Tensor.numpy = Patch.numpy


class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib:
    def __init__(self, cfg, motion_file, asset_infos, key_body_names, device, humanoid_type, mf_humanoid_type):
        """
        Motion library for the IsaacGym humanoid series environments, process the motion data exported from Optitrack
        or Xsens.

        :param cfg: the configuration file for the task
        :param motion_file: motion file path, should be .yaml files or .npy files
        :param asset_infos: the asset information dict for the humanoid models
        :param key_body_names: list of key body ids
        :param device: device same as the env/task device
        :param humanoid_type: target humanoid model type, e.g., hotu_humanoid, hotu_humanoid_w_qbhand_full, etc.
        :param mf_humanoid_type: source humanoid type in the motion file, e.g., hotu_humanoid, hotu_humanoid_w_qbhand_full, etc.
        """
        self.cfg = cfg
        self.humanoid_asset_infos = asset_infos
        self.humanoid_type = humanoid_type
        self.mf_humanoid_type = mf_humanoid_type

        self.humanoid_rb_dict = self.humanoid_asset_infos[self.humanoid_type]["rigid_bodies"]
        self.humanoid_dof_dict = self.humanoid_asset_infos[self.humanoid_type]["dofs"]
        self.humanoid_del_rb = self.humanoid_asset_infos[self.humanoid_type]["del_rb"]
        self.humanoid_rb_dict_a_del = OrderedDict(
            {k: v for k, v in self.humanoid_rb_dict.items() if k not in self.humanoid_del_rb}
        )  # no pelvis and no root links, but still keep the id
        self.mf_humanoid_rb_dict = self.humanoid_asset_infos[self.mf_humanoid_type]["rigid_bodies"]
        self.mf_humanoid_dof_dict = self.humanoid_asset_infos[self.mf_humanoid_type]["dofs"]
        self.mf_humanoid_del_rb = self.humanoid_asset_infos[self.mf_humanoid_type]["del_rb"]
        self.mf_humanoid_rb_dict_a_del = OrderedDict(
            {k: v for k, v in self.mf_humanoid_rb_dict.items() if k not in self.mf_humanoid_del_rb}
        )
        self._get_skeleton_rb_index()
        self._get_skeleton_dof_index()

        if "w_qbhand" in self.humanoid_type:
            self.useful_right_qbhand_dof_index = sorted(
                [value for key, value in self.humanoid_dof_dict.items() if
                 ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
                 ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key) and (
                         "right_qbhand" in key)])
            self.useful_left_qbhand_dof_index = sorted(
                [value for key, value in self.humanoid_dof_dict.items() if
                 ("virtual" not in key) and ("index_knuckle" not in key) and ("middle_knuckle" not in key) and
                 ("ring_knuckle" not in key) and ("little_knuckle" not in key) and ("synergy" not in key) and (
                         "left_qbhand" in key)])

            self.virtual2real_dof_index_map_dict = {value: self.humanoid_dof_dict[key.replace("_virtual", "")] for
                                                    key, value in self.humanoid_dof_dict.items() if
                                                    "virtual" in key}

        self.use_extra_dof_states_motion = cfg["env"]["use_extra_dof_states_motion"]
        self.extra_rewrite_dof_names = cfg["env"].get("extra_rewrite_dof_names", None)
        if self.use_extra_dof_states_motion and self.extra_rewrite_dof_names is not None:
            if self.extra_rewrite_dof_names == "all":
                self.extra_rewrite_dof_id = [id for id in self.humanoid_dof_dict.values()]
                self.mf_extra_rewrite_dof_id = [id for id in self.mf_humanoid_dof_dict.values()]
            else:
                self.extra_rewrite_dof_id = [self.humanoid_dof_dict[dof_name] for dof_name in
                                             self.extra_rewrite_dof_names]
                self.mf_extra_rewrite_dof_id = [self.mf_humanoid_dof_dict[dof_name] for dof_name in
                                                self.extra_rewrite_dof_names]

        # no pelvis and no root links, start from 1
        self._mf_dof_body_ids = [v for k, v in self.mf_humanoid_rb_dict_a_del.items()]
        self._mf_dof_offsets = self.humanoid_asset_infos[self.mf_humanoid_type]["dof_offsets"]
        self._mf_num_dof = self._mf_dof_offsets[-1]
        self._device = device
        self._key_body_ids = self._get_key_body_ids(key_body_names, self.humanoid_rb_dict)

        self._object_poses = torch.zeros((), device=device)

        # Load motion from motion files
        self._load_motions(motion_file)
        self.mf_gts = torch.cat([m.global_translation for m in self._motions], dim=0).float()
        self.mf_grs = torch.cat([m.global_rotation for m in self._motions], dim=0).float()
        self.mf_lrs = torch.cat([m.local_rotation for m in self._motions], dim=0).float()

        if self.use_extra_dof_states_motion:
            self.mf_dps = torch.cat([m[:, :, 0] for m in self._motion_dof_states], dim=0).float()
            self.mf_dvs = torch.cat([m[:, :, 1] for m in self._motion_dof_states], dim=0).float()

        # Transfer motion from mf_humanoid_type to humanoid_type
        if self.mf_humanoid_type != self.humanoid_type:
            self.transfer_motion_to_humanoid_type()

        # Adjust humanoid height to let its feet on the floor
        self.humanoid_height_offsets = []
        for motion in self._motions:
            tar_global_pos = motion.global_translation
            min_h = torch.min(tar_global_pos[..., 2])
            motion.global_translation[..., 2] -= min_h
            self.humanoid_height_offsets.append(min_h)

        # Get motion values based on target humanoid model type
        self.gts = torch.cat([m.global_translation for m in self._motions], dim=0).float()
        self.grs = torch.cat([m.global_rotation for m in self._motions], dim=0).float()
        self.lrs = torch.cat([m.local_rotation for m in self._motions], dim=0).float()
        self.grvs = torch.cat([m.global_root_velocity for m in self._motions], dim=0).float()
        self.gravs = torch.cat([m.global_root_angular_velocity for m in self._motions], dim=0).float()
        self.dvs = torch.cat([m.dof_vels for m in self._motions], dim=0).float()
        # self.dps = torch.cat([m.dof_pos for m in self._motions], dim=0).float()

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        self.right_left_ids_except_thumb_and_knuckle = [rb_id for rb_name, rb_id in self.mf_humanoid_rb_dict.items() if
                                                        "qbhand" in rb_name and "thumb" not in rb_name and
                                                        "knuckle" not in rb_name and "root_link" not in rb_name]
        self.right_left_thumb_knuckle_ids = [rb_id for rb_name, rb_id in self.mf_humanoid_rb_dict.items() if
                                             "thumb_knuckle" in rb_name]

    def _num_motions(self):
        return len(self._motions)

    def _get_key_body_ids(self, key_body_names, rb_dict):
        key_body_ids = []
        for key_body_name in key_body_names:
            if key_body_name in rb_dict.keys():
                key_body_ids.append(rb_dict[key_body_name])
            else:
                raise ValueError("The key_body_name is not in the mf_humanoid_rb_dict")
        return torch.tensor(key_body_ids, device=self._device, dtype=torch.long)

    def _get_skeleton_rb_index(self):
        self.mfrb2rb_dict = {}  # rigid body id mapping from motion file humanoid model to humanoid model we need
        unexpected_rb_names = []
        for rb_name in self.mf_humanoid_rb_dict.keys():
            if rb_name in self.humanoid_rb_dict.keys():
                self.mfrb2rb_dict[self.mf_humanoid_rb_dict[rb_name]] = self.humanoid_rb_dict[rb_name]
            else:
                # raise ValueError(f"The {rb_name} is not in the humanoid_rb_dict")
                unexpected_rb_names.append(rb_name)
        if len(unexpected_rb_names) > 0:
            rf.logger.beauty_print(f"The following rigid bodies are not in the humanoid_rb_dict: {unexpected_rb_names}",
                                   "warning")
        self.rb2mfrb_dict = {v: k for k, v in self.mfrb2rb_dict.items()}

    def _get_skeleton_dof_index(self):
        self.mfdof2dof_dict = {}
        unexpected_dof_names = []
        for dof_name in self.mf_humanoid_dof_dict.keys():
            if dof_name in self.humanoid_dof_dict.keys():
                self.mfdof2dof_dict[self.mf_humanoid_dof_dict[dof_name]] = self.humanoid_dof_dict[dof_name]
            else:
                # raise ValueError(f"The {dof_name} is not in the humanoid_dof_dict")
                unexpected_dof_names.append(dof_name)
        if len(unexpected_dof_names) > 0:
            rf.logger.beauty_print(f"The following dofs are not in the humanoid_dof_dict: {unexpected_dof_names}",
                                   "warning")
        self.dof2mfdof_dict = {v: k for k, v in self.mfdof2dof_dict.items()}

    def transfer_motion_to_humanoid_type(self):
        """
        Transfer the motion data with mf_humanoid_type to humanoid_type we need.
        Joints do not exist in the mf_humanoid_type will be regarded as 0.
        Global translation and rotation for these joints are meaningless.
        """
        transferred_motions = []
        for i in range(self.num_motion_files):
            motion = self._motions[i]
            num_frames = motion.global_angular_velocity.shape[0]

            old_global_translation = motion.global_translation.clone()
            old_global_rotation = motion.global_rotation.clone()
            old_local_rotation = motion.local_rotation.clone()
            old_dof_vels = motion.dof_vels.clone()
            # old_dof_pos = motion.dof_pos.clone()

            motion.global_translation = torch.zeros(
                (num_frames, len(self.humanoid_rb_dict), 3), device=self._device, dtype=torch.float32)
            motion.global_rotation = torch.zeros(
                (num_frames, len(self.humanoid_rb_dict), 4), device=self._device, dtype=torch.float32)
            motion.local_rotation = torch.zeros(
                (num_frames, len(self.humanoid_rb_dict), 4), device=self._device, dtype=torch.float32)
            motion.dof_vels = torch.zeros((num_frames, len(self.humanoid_dof_dict)),
                                          device=self._device, dtype=torch.float32)

            motion.global_rotation[:, -1] = 1
            motion.local_rotation[:, -1] = 1

            for index, mf_index in self.rb2mfrb_dict.items():
                motion.global_translation[:, index] = old_global_translation[:, mf_index]
                motion.global_rotation[:, index] = old_global_rotation[:, mf_index]
                motion.local_rotation[:, index] = old_local_rotation[:, mf_index]

            for index, mf_index in self.dof2mfdof_dict.items():
                motion.dof_vels[:, index] = old_dof_vels[:, mf_index]
                # motion.dof_pos[:, index] = old_dof_pos[:, mf_index]

            transferred_motions.append(motion)
        self._motions = transferred_motions

    def _get_total_length(self):
        return sum(self._motion_lengths)

    def _get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._motion_weights, num_samples=n, replacement=True
        )

        # m = self._num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_dof_states = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        self.motion_files = motion_files
        self.num_motion_files = len(motion_files)

        with tqdm.trange(self.num_motion_files, ncols=100, colour="green") as t_bar:
            for f in t_bar:
                curr_file = motion_files[f]
                t_bar.set_postfix_str("Loading: {:s}".format(curr_file.split("/")[-1]))
                curr_motion = SkeletonMotion.from_file(curr_file)

                motion_fps = curr_motion.fps
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.tensor.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)

                self._motion_fps.append(float(motion_fps))
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
                curr_motion.dof_vels = curr_dof_vels

                # Moving motion tensors to the GPU
                if USE_CACHE:
                    curr_motion = DeviceCache(curr_motion, self._device)
                else:
                    curr_motion.tensor = curr_motion.tensor.to(self._device)
                    curr_motion._skeleton_tree._parent_indices = (
                        curr_motion._skeleton_tree._parent_indices.to(self._device)
                    )
                    curr_motion._skeleton_tree._local_translation = (
                        curr_motion._skeleton_tree._local_translation.to(self._device)
                    )
                    curr_motion._rotation = curr_motion._rotation.to(self._device)

                self._motions.append(curr_motion)
                self._motion_lengths.append(curr_len)

                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)
                self._motion_files.append(curr_file)

                if self.use_extra_dof_states_motion:
                    curr_dof_states_file = curr_file.replace(".npy", "_dof_states.npy")
                    curr_dof_states = np.load(curr_dof_states_file)
                    curr_dof_states = torch.tensor(curr_dof_states, device=self._device, dtype=torch.float32)
                    self._motion_dof_states.append(curr_dof_states)

        for motion in self._motions:
            if motion.object_poses is not None:
                self._object_poses = motion.object_poses
                self._object_poses = torch.tensor(self._object_poses, device=self._device, dtype=torch.float32)
                break

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        self.num_motions = self._num_motions()
        total_len = self._get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(self.num_motions, total_len))

    @staticmethod
    def _fetch_motion_files(motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = motion_entry["file"]
                curr_weight = motion_entry["weight"]
                assert curr_weight >= 0

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def get_object_pose(self, frame_id):
        """
        Return object pose at frame=id, where id is recorded in motion_ids

        :param frame_id: Same frame id * num_envs, where num_envs is the environment number.
        :return:
        """
        if self._object_poses.ndim == 0:
            return None
        object_pose = self._object_poses[frame_id][0]
        return object_pose

    def get_motion_state(self, motion_ids, motion_times):
        """
        Get the motion state at the given time for the given motion ids

        :param motion_ids: the list of motion ids, useful for yaml file
        :param motion_times:
        :return:
        """
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        mf_local_rot0 = self.mf_lrs[f0l]
        mf_local_rot1 = self.mf_lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]

        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [
            root_pos0,
            root_pos1,
            mf_local_rot0,
            mf_local_rot1,
            root_vel,
            root_ang_vel,
            key_pos0,
            key_pos1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1

        mf_local_rot = torch_utils.slerp(mf_local_rot0, mf_local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(mf_local_rot)

        if self.use_extra_dof_states_motion:
            dof_pos[:, self.extra_rewrite_dof_id] = self.mf_dps[f1l][:, self.mf_extra_rewrite_dof_id]
            dof_vel[:, self.extra_rewrite_dof_id] = self.mf_dvs[f1l][:, self.mf_extra_rewrite_dof_id]

        if "w_qbhand" in self.humanoid_type:
            for virtual_id, real_id in self.virtual2real_dof_index_map_dict.items():
                dof_pos[:, virtual_id] = dof_pos[:, real_id].clone()
                dof_vel[:, virtual_id] = dof_vel[:, real_id].clone()

        root_pos[:, 2] += 0.2

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, f0l, f1l

    def _calc_frame_blend(self, time, len, num_frames, dt):
        """
        
        :param time: 
        :param len: 
        :param num_frames: 
        :param dt: 
        :return: 
        """
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self._get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_pos(self, motion):
        """
        Compute joint positions from local rotation value
        
        :param motion: 
        :return: 
        """
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_pos = []

        # for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[[i for i in range(num_frames - 1)]]
        local_rot1 = motion.local_rotation[[i for i in range(1, num_frames)]]

        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.tensor(dt))
        dof_pos = self._local_rotation_to_dof(local_rot)
        # dof_pos.append(frame_dof_pos)
        #
        # dof_pos.append(dof_pos[-1])
        # dof_pos = torch.stack(dof_pos, dim=0)

        return dof_pos

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._mf_dof_body_ids
        dof_offsets = self._mf_dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros(
            (n, self._mf_num_dof), dtype=torch.float, device=self._device
        )

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            # if self.mf_humanoid_type == "hotu_humanoid_w_qbhand_no_virtual":
            #     right_hand_id = 5
            #     right_hand_ori = [0, 1, 0, 0]
            #     left_hand_id = 23
            #     left_hand_ori = [1, 0, 0, 0]
            #     right_left_ids_except_thumb = [*[i for i in range(10, 21)], *[i for i in range(28, 39)]]
            #     right_left_thumb_knuckle_ids = [6, 24]
            # elif self.mf_humanoid_type == "hotu_humanoid_w_qbhand_full":
            #     right_left_ids_except_thumb = [*[i for i in range(11, 39)], *[i for i in range(47, 75)]]
            #     right_left_thumb_knuckle_ids = [6, 42]
            # elif self.mf_humanoid_type == "h1_w_qbhand":
            #     right_hand_id = 55
            #     right_hand_ori = [0, 0, 0, 1]
            #     left_hand_id = 16
            #     left_hand_ori = [0, 0, 0, 1]
            #     right_left_ids_except_thumb = [*[i for i in range(23, 51)], *[i for i in range(62, 90)]]
            #     right_left_thumb_knuckle_ids = [18, 57]
            #     left_shoulder_pitch_id = 12
            #     left_shoulder_roll_id = 13
            #     left_shoulder_yaw_id = 14
            #     right_shoulder_pitch_id = 51
            #     right_shoulder_roll_id = 52
            #     right_shoulder_yaw_id = 53
            #     left_hip_pitch_id = 3
            #     left_hip_roll_id = 2
            #     left_hip_yaw_id = 1
            #     right_hip_pitch_id = 8
            #     right_hip_roll_id = 7
            #     right_hip_yaw_id = 6
            #     torso_id = 11
            # elif self.mf_humanoid_type == "curi_w_softhand_isaacgym":
            #     right_hand_id = 54
            #     right_hand_ori = [0, 0, 0, 1]
            #     left_hand_id = 14
            #     left_hand_ori = [0, 0, 0, 1]
            #     right_left_ids_except_thumb = [*[i for i in range(20, 48)], *[i for i in range(60, 88)]]
            #     right_left_thumb_knuckle_ids = [15, 55]
            #
            #     torso_id = 5
            #     panda_left_link1_id = 8
            #     panda_left_link2_id = 9
            #     panda_left_link3_id = 10
            #     panda_left_link4_id = 11
            #     panda_left_link5_id = 12
            #     panda_left_link6_id = 13
            #     panda_left_link7_id = 14
            #     panda_right_link1_id = 48
            #     panda_right_link2_id = 49
            #     panda_right_link3_id = 50
            #     panda_right_link4_id = 51
            #     panda_right_link5_id = 52
            #     panda_right_link6_id = 53
            #     panda_right_link7_id = 54
            # elif self.mf_humanoid_type == "walker":
            #     right_left_ids_except_thumb = [*[i for i in range(9, 19)], *[i for i in range(26, 36)]]
            #     # right_left_ids = [*[i for i in range(9, 19)], *[i for i in range(26, 36)]]
            #     right_left_thumb_knuckle_ids = []
            #
            #     head_l1_id = 36
            #     head_l2_id = 37
            #
            #     right_limb_l1_id = 2
            #     right_limb_l2_id = 3
            #     right_limb_l3_id = 4
            #     right_limb_l4_id = 5
            #     right_limb_l5_id = 6
            #     right_limb_l6_id = 7
            #     right_limb_l7_id = 8
            #     left_limb_l1_id = 19
            #     left_limb_l2_id = 20
            #     left_limb_l3_id = 21
            #     left_limb_l4_id = 22
            #     left_limb_l5_id = 23
            #     left_limb_l6_id = 24
            #     left_limb_l7_id = 25
            #
            #     right_leg_l1_id = 38
            #     right_leg_l2_id = 39
            #     right_leg_l3_id = 40
            #     right_leg_l4_id = 41
            #     right_leg_l5_id = 42
            #     right_leg_l6_id = 43
            #     left_leg_l1_id = 44
            #     left_leg_l2_id = 45
            #     left_leg_l3_id = 46
            #     left_leg_l4_id = 47
            #     left_leg_l5_id = 48
            #     left_leg_l6_id = 49
            #
            #     torso_id = 1
            # else:
            #     raise ValueError("Unsupported humanoid type")
            #     # rf.logger.beauty_print("Unsupported humanoid type", "warning")
            #     # pass

            if joint_size == 3:
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset: (joint_offset + joint_size)] = joint_exp_map
            elif joint_size == 1:
                if body_id in self.right_left_ids_except_thumb_and_knuckle:  # Right and left fingers except thumbs
                    joint_q = local_rot[:, body_id]
                    joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                    joint_theta = -(joint_theta * joint_axis[..., 2])  # assume joint is always along z axis
                elif body_id in self.right_left_thumb_knuckle_ids:  # right and left thumbs knuckles link
                    joint_q = local_rot[:, body_id]
                    joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                    joint_theta = -(joint_theta * joint_axis[..., 0])  # assume joint is always along x axis
                # elif self.humanoid_type == "mjcf/unitreeH1/h1_w_qbhand.xml":
                #     if body_id == left_shoulder_pitch_id:  # Pitch joint
                #         joint_q = local_rot[:, left_shoulder_pitch_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == left_shoulder_roll_id:  # Roll joint
                #         joint_q = local_rot[:, left_shoulder_pitch_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0.707, 0, 0, 0.707], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])  # assume joint is always along x axis
                #     elif body_id == left_shoulder_yaw_id:  # Yaw joint
                #         joint_q = local_rot[:, left_shoulder_pitch_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])  # assume joint is always along z axis
                #     elif body_id == right_shoulder_roll_id:  # Roll joint
                #         joint_q = local_rot[:, right_shoulder_pitch_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([-0.707, 0, 0, 0.707], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])  # assume joint is always along x axis
                #     elif body_id == right_shoulder_yaw_id:  # Yaw joint
                #         joint_q = local_rot[:, right_shoulder_pitch_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])  # assume joint is always along z axis
                #     elif body_id == torso_id:  # Yaw joint
                #         joint_q = local_rot[:, body_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])  # assume joint is always along z axis
                #     elif body_id == left_hip_roll_id:  # Roll joint
                #         joint_q = local_rot[:, left_hip_yaw_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])  # assume joint is always along x axis
                #     elif body_id == left_hip_pitch_id:  # Pitch joint
                #         joint_q = local_rot[:, left_hip_yaw_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])  # assume joint is always along y axis
                #     elif body_id == left_hip_yaw_id:  # Yaw joint
                #         joint_q = local_rot[:, left_hip_yaw_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])  # assume joint is always along z axis
                #     elif body_id == right_hip_roll_id:  # Roll joint
                #         joint_q = local_rot[:, right_hip_yaw_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])  # assume joint is always along x axis
                #     elif body_id == right_hip_pitch_id:  # Pitch joint
                #         joint_q = local_rot[:, right_hip_yaw_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])  # assume joint is always along y axis
                #     elif body_id == right_hip_yaw_id:  # Yaw joint
                #         joint_q = local_rot[:, right_hip_yaw_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])  # assume joint is always along z axis
                #     else:
                #         joint_q = local_rot[:, body_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])  # assume joint is always along y axis
                #
                # elif self.humanoid_type == "mjcf/curi/curi_w_softhand_isaacgym.xml":
                #     if body_id == panda_left_link1_id:
                #         joint_q = local_rot[:, panda_left_link1_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0, 0, 0.382, 0.92], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == panda_left_link2_id:
                #         joint_q = local_rot[:, panda_left_link1_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0, 0, 0.382, 0.92], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     # elif body_id == panda_left_link3_id:
                #     #     joint_q = local_rot[:, panda_left_link1_id]
                #     #     joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0, 0, 0.382, 0.92], joint_q)
                #     #     # joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0.707107, 0, 0, 0.707107], joint_q)
                #     #     # joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([-0.436865, 0.49775, 0.054428, 0.747283], joint_q)
                #     #     joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #     #     joint_theta = (joint_theta * joint_axis[..., 0])
                #     elif body_id == panda_left_link4_id:
                #         joint_q = local_rot[:, panda_left_link4_id]
                #         # joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0.707107, 0, 0, 0.707107], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == panda_left_link5_id:
                #         joint_q = local_rot[:, panda_left_link7_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0, -0.707107, 0, 0.707107], joint_q)
                #         # joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([-0.707107, 0, 0, 0.707107], joint_q)
                #         # joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([-0.707107, 0, 0, 0.707107], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])
                #     elif body_id == panda_left_link6_id:
                #         joint_q = local_rot[:, panda_left_link7_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0, -0.707107, 0, 0.707107], joint_q)
                #
                #         # joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0.707107, 0, 0, 0.707107], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == panda_left_link7_id:
                #         joint_q = local_rot[:, panda_left_link7_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0, -0.707107, 0, 0.707107], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == torso_id:  # Yaw joint
                #         joint_q = local_rot[:, body_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])  # assume joint is always along z axis
                #     else:
                #         joint_q = local_rot[:, body_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])  # assume joint is always along y axis
                #
                # elif self.humanoid_type == "mjcf/walker/walker.xml":
                #     if body_id == head_l1_id:
                #         joint_q = local_rot[:, head_l1_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == head_l2_id:
                #         joint_q = local_rot[:, head_l2_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == right_limb_l1_id:
                #         joint_q = local_rot[:, right_limb_l1_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == right_limb_l2_id:
                #         joint_q = local_rot[:, right_limb_l1_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0,-0.707,  0, 0.707], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == right_limb_l3_id:
                #         joint_q = local_rot[:, right_limb_l1_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])
                #     elif body_id == right_limb_l4_id:
                #         joint_q = local_rot[:, right_limb_l4_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == right_limb_l5_id:
                #         joint_q = local_rot[:, right_limb_l7_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = -(joint_theta * joint_axis[..., 0])
                #     elif body_id == right_limb_l6_id:
                #         joint_q = local_rot[:, right_limb_l7_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == right_limb_l7_id:
                #         joint_q = local_rot[:, right_limb_l7_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == left_limb_l1_id:
                #         joint_q = local_rot[:, left_limb_l1_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == left_limb_l2_id:
                #         joint_q = local_rot[:, left_limb_l1_id]
                #         joint_q = rf.robolab.quaternion_multiply_tensor_multirow2([0,-0.707,  0, 0.707], joint_q)
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == left_limb_l3_id:
                #         joint_q = local_rot[:, left_limb_l1_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])
                #     elif body_id == left_limb_l4_id:
                #         joint_q = local_rot[:, left_limb_l4_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     elif body_id == left_limb_l5_id:
                #         joint_q = local_rot[:, left_limb_l7_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 0])
                #     elif body_id == left_limb_l6_id:
                #         joint_q = local_rot[:, left_limb_l7_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                #     elif body_id == left_limb_l7_id:
                #         joint_q = local_rot[:, left_limb_l7_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 2])
                #     else:
                #         joint_q = local_rot[:, body_id]
                #         joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                #         joint_theta = (joint_theta * joint_axis[..., 1])
                else:
                    joint_q = local_rot[:, body_id]
                    joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                    joint_theta = (joint_theta * joint_axis[..., 1])  # assume joint is always along y axis

                joint_theta = torch_utils.normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta
            else:
                print("Unsupported joint type")
                assert False

        if self.mf_humanoid_type != self.humanoid_type:
            mf_dof_pos = dof_pos
            dof_pos = torch.zeros((n, len(self.humanoid_dof_dict)), dtype=torch.float, device=self._device)
            for humanoid_index, mf_index in self.dof2mfdof_dict.items():
                dof_pos[:, humanoid_index] = mf_dof_pos[:, mf_index]

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._mf_dof_body_ids
        dof_offsets = self._mf_dof_offsets

        dof_vel = torch.zeros([self._mf_num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset
            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset: (joint_offset + joint_size)] = joint_vel
            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[
                    1
                ]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert False

        return dof_vel

    # def spherical_joint_decompose(self, source_spherical_joint_name, target_revolute_joint_name):
    #     """
    #     Decompose and represent 3DoF spherical joint with three 1 DoF revolution joint
    #
    #     :param source_spherical_joint_name:
    #     :param target_revolute_joint_name:
    #     :return:
    #     """


class ObjectMotionLib:
    def __init__(self, object_motion_file, object_names, device, humanoid_start_time=None, height_offset=0.0):
        """
        Object motion library for the IsaacGym humanoid series environments and process the motion data from Optitrack.

        :param object_motion_file: optitrack motion file path, should be .csv files, can be a list of files or a single file
        :param object_names: list of object names you want to load
        :param device: device same as the env/task device
        :param humanoid_start_time: start time of the humanoid motion, for the temporal synchronization
        :param height_offset: height offset of the object, for the spatial alignment
        """
        self.object_motion_file = object_motion_file
        self.object_names = object_names
        self.object_poses_w_time = []
        self.device = device
        self.humanoid_start_time = humanoid_start_time  # in second, for the motion sync
        if self.humanoid_start_time is not None:
            assert len(self.humanoid_start_time) == len(self.object_motion_file)
        self.height_offset = height_offset

        if isinstance(self.object_motion_file, list):  # Multiple files
            self.num_motions = len(self.object_motion_file)
        elif isinstance(self.object_motion_file, str):  # Single file
            self.num_motions = 1
        else:
            raise NotImplementedError

        self._load_motions()

    def _load_motions(self):
        objs_list, meta_list = rf.optitrack.get_objects(self.object_motion_file)
        self.scales = self._get_scale(meta_list)
        self.dts = self._get_dt(meta_list)
        if self.humanoid_start_time is not None:
            self.tds = self._get_time_difference(meta_list)
        else:
            self.tds = [0] * self.num_motions

        for i in range(self.num_motions):
            # data is a numpy array of shape (n_samples, n_features)
            # labels is a list of strings corresponding to the name of the features
            data, labels = rf.optitrack.data_clean(self.object_motion_file, legacy=False, objs=objs_list[i])[i]

            # Accessing the position and attitude of an object over all samples:
            # Coordinates names and order: ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            object_poses_dict = {}
            for object_name in self.object_names:
                data_ptr = labels.index(f'{object_name}.pose.x')
                assert data_ptr + 6 == labels.index(f'{object_name}.pose.qw')
                pose = data[:, data_ptr:data_ptr + 7]
                pose[:, :3] *= self.scales[i]  # convert to meter
                pose = self._motion_transform(torch.tensor(pose, dtype=torch.float))

                pose_w_time = torch.hstack((torch.tensor(data[:, 1], dtype=torch.float).unsqueeze(-1), pose))
                object_poses_dict[object_name] = pose_w_time.to(self.device)
            self.object_poses_w_time.append(object_poses_dict)  # [num_motions, num_objects, num_samples, 7]

    def _get_scale(self, meta_list):
        """
        Get the scale of the motion data, convert to meter

        :param meta_list: list of meta data obtained from motion files
        :return: list of scale for each motion file
        """
        scales = []
        for i in range(len(meta_list)):
            if meta_list[i]["Length Units"] == "Centimeters":
                scales.append(0.01)
            elif meta_list[i]["Length Units"] == "Meters":
                scales.append(1.0)
            elif meta_list[i]["Length Units"] == "Millimeters":
                scales.append(0.001)
            else:
                raise NotImplementedError
        return scales

    def _get_dt(self, meta_list):
        """
        Get the time interval of the motion data, convert to second

        :param meta_list: list of meta data obtained from motion files
        :return: list of dt for each motion file
        """
        dts = []
        for i in range(len(meta_list)):
            dts.append(1.0 / float(meta_list[i]["Export Frame Rate"]))
        return dts

    def _get_time_difference(self, meta_list):
        """
        Get the time difference between the motion data and the humanoid motion by comparing their start time, convert to second

        :param meta_list: list of meta data obtained from motion files
        :return: list of time difference for each motion file
        """
        time_diffs = []
        for i in range(len(meta_list)):
            object_start_time = meta_list[i]["Capture Start Time"]
            y, t = object_start_time.split(' ')[0], object_start_time.split(' ')[1]
            if meta_list[i]["Take Name"].split(" ")[-1] == "PM":
                t = t.split(".")
                t[0] = str(int(t[0]) + 12)
                t = ":".join(t[:3])
            timestamp = time.mktime(time.strptime(y + " " + t, "%Y-%m-%d %H:%M:%S"))
            time_diffs.append(float(timestamp - self.humanoid_start_time[i]))
        return time_diffs

    def _motion_transform(self, pose):
        """
        Coordinate transformation from optitrack to IsaacGym

        :param pose: [num_samples, 7]
        :return: pose: [num_samples, 7]
        """
        # y-up in the optitrack to z-up in IsaacGym
        # pose[:, 0] = pose[:, 0]
        # tmp = pose[:, 1].clone()
        # pose[:, 1] = -pose[:, 2]
        # pose[:, 2] = tmp - self.height_offset
        # pose[:, 3:] = rf.robolab.quaternion_multiply_tensor_multirow2(torch.tensor([0.5, 0.5, 0.5, 0.5]),
        #                                                               torch.tensor(pose[:, 3:]))
        homo_matrix = torch.tensor(rf.robolab.homo_matrix_from_quaternion([0.5, 0.5, 0.5, 0.5]), dtype=torch.float32)
        num_samples = len(pose)
        raw_position = pose[:, :3]
        new_position = torch.ones((num_samples, 4, 1))
        new_position[:, :3] = raw_position.resize(num_samples, 3, 1)
        new_position = torch.bmm(homo_matrix.expand(num_samples, 4, 4), new_position)

        new_pose = torch.zeros((num_samples, 7))
        # new_pose[:, 3:] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        new_pose[:, :3] = new_position[:, :3].resize(num_samples, 3)
        new_pose[:, 2] -= self.height_offset.cpu()

        # pose[:, :3] = new_position[:, :3].resize(num_samples, 3)
        new_pose[:, 3:] = rf.robolab.quaternion_multiply_tensor_multirow2(torch.tensor([0.5, 0.5, 0.5, 0.5]),
                                                                          torch.tensor(pose[:, 3:]))
        # new_pose[:, 3:] = rf.robolab.quaternion_multiply_tensor_multirow2(torch.tensor([0, 0, -0.383, 0.924]),
        new_pose[:, 3:] = rf.robolab.quaternion_multiply_tensor_multirow2(torch.tensor([0, 0, 0.924, 0.383]),
                                                                          torch.tensor(new_pose[:, 3:]))

        return new_pose

    def get_motion_state(self, motion_ids, motion_times):
        """
        Get the object pose at the given time

        :param motion_ids: the id indicating which motion file to use
        :param motion_times: the time to get the object pose
        :return:
        """
        approx_index = torch.round(motion_times[motion_ids] / self.dts[motion_ids] + self.tds[motion_ids])[0].long()
        if approx_index < 0:
            approx_index = 0
        elif approx_index >= self.object_poses_w_time[motion_ids][self.object_names[0]].shape[0]:
            approx_index = self.object_poses_w_time[motion_ids][self.object_names[0]].shape[0] - 1
        object_poses = {}
        for object_name in self.object_poses_w_time[motion_ids].keys():
            object_poses[object_name] = self.object_poses_w_time[motion_ids][object_name][approx_index][1:]
        return object_poses
