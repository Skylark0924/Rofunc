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

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *

from rofunc.learning.RofuncRL.tasks.isaacgymenv.base.curi_base_task import CURIBaseTask
from rofunc.utils.oslab.path import get_rofunc_path


class CURICabinetTask(CURIBaseTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.visual_obs_flag = cfg['task']['visual_obs_flag']

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # set default pos (seven links and left and right gripper, [:9] for the left arm, [9:] for the right arm)
        self.curi_default_dof_pos = to_torch(
            [0.3863, 0.5062, -0.1184, -0.6105, 0.023, 1.6737, 0.9197, 0.04, 0.04, -0.5349, 0, 0.1401, -1.7951, 0.0334,
             3.2965, 0.6484, 0.04, 0.04], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.curi_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_curi_dofs]
        self.curi_dof_pos = self.curi_dof_state[..., 0]
        self.curi_dof_vel = self.curi_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_curi_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.curi_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # get rofunc path from rofunc package metadata
        rofunc_path = get_rofunc_path()
        asset_root = os.path.join(rofunc_path, "simulator/assets")

        curi_asset_file = "urdf/curi/urdf/curi_isaacgym_dual_arm.urdf"
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        # load curi asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        curi_asset = self.gym.load_asset(self.sim, asset_root, curi_asset_file, asset_options)

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        curi_dof_stiffness = to_torch(
            [400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6, 400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6],
            dtype=torch.float, device=self.device)
        curi_dof_damping = to_torch(
            [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float,
            device=self.device)

        self.num_curi_bodies = self.gym.get_asset_rigid_body_count(curi_asset)
        self.num_curi_dofs = self.gym.get_asset_dof_count(curi_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num env: ", num_envs)
        print("num curi bodies: ", self.num_curi_bodies)
        print("num curi dofs: ", self.num_curi_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

        # set curi dof properties
        curi_dof_props = self.gym.get_asset_dof_properties(curi_asset)
        self.curi_dof_lower_limits = []
        self.curi_dof_upper_limits = []
        for i in range(self.num_curi_dofs):
            curi_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                curi_dof_props['stiffness'][i] = curi_dof_stiffness[i]
                curi_dof_props['damping'][i] = curi_dof_damping[i]
            else:
                curi_dof_props['stiffness'][i] = 7000.0
                curi_dof_props['damping'][i] = 50.0

            self.curi_dof_lower_limits.append(curi_dof_props['lower'][i])
            self.curi_dof_upper_limits.append(curi_dof_props['upper'][i])

        self.curi_dof_lower_limits = to_torch(self.curi_dof_lower_limits, device=self.device)
        self.curi_dof_upper_limits = to_torch(self.curi_dof_upper_limits, device=self.device)
        self.curi_dof_speed_scales = torch.ones_like(self.curi_dof_lower_limits)
        self.curi_dof_speed_scales[[7, 8]] = 0.1
        self.curi_dof_speed_scales[[16, 17]] = 0.1
        curi_dof_props['effort'][7] = 200
        curi_dof_props['effort'][8] = 200
        curi_dof_props['effort'][16] = 200
        curi_dof_props['effort'][17] = 200

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        # create prop assets
        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        curi_start_pose = gymapi.Transform()
        curi_start_pose.p = gymapi.Vec3(1.5, 0.0, 0.0)
        curi_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # compute aggregate size
        num_curi_bodies = self.gym.get_asset_rigid_body_count(curi_asset)
        num_curi_shapes = self.gym.get_asset_rigid_shape_count(curi_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_curi_bodies + num_cabinet_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_curi_shapes + num_cabinet_shapes + self.num_props * num_prop_shapes

        self.curis = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            curi_actor = self.gym.create_actor(env_ptr, curi_asset, curi_start_pose, "curi", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, curi_actor, curi_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            cabinet_pose = cabinet_start_pose
            cabinet_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cabinet_pose.p.y += self.start_position_noise * dy
            cabinet_pose.p.z += self.start_position_noise * dz
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
                drawer_pose = self.gym.get_rigid_transform(env_ptr, drawer_handle)

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = drawer_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = drawer_pose.p.y + propy
                        prop_state_pose.p.z = drawer_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose,
                                                            "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z,
                                                         prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.curis.append(curi_actor)
            self.cabinets.append(cabinet_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, "panda_left_link7")
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, "panda_left_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, "panda_left_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(
            self.num_envs, self.num_props, 13)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curis[0], "panda_left_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curis[0], "panda_left_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.curis[0], "panda_left_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        curi_local_grasp_pose = hand_pose_inv * finger_pose
        curi_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.curi_local_grasp_pos = to_torch([curi_local_grasp_pose.p.x, curi_local_grasp_pose.p.y,
                                              curi_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.curi_local_grasp_rot = to_torch([curi_local_grasp_pose.r.x, curi_local_grasp_pose.r.y,
                                              curi_local_grasp_pose.r.z, curi_local_grasp_pose.r.w],
                                             device=self.device).repeat((self.num_envs, 1))

        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                                drawer_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                                drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w],
                                               device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.curi_grasp_pos = torch.zeros_like(self.curi_local_grasp_pos)
        self.curi_grasp_rot = torch.zeros_like(self.curi_local_grasp_rot)
        self.curi_grasp_rot[..., -1] = 1  # xyzw
        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., -1] = 1
        self.curi_lfinger_pos = torch.zeros_like(self.curi_local_grasp_pos)
        self.curi_rfinger_pos = torch.zeros_like(self.curi_local_grasp_pos)
        self.curi_lfinger_rot = torch.zeros_like(self.curi_local_grasp_rot)
        self.curi_rfinger_rot = torch.zeros_like(self.curi_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_curi_reward(
            self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
            self.curi_grasp_pos, self.drawer_grasp_pos, self.curi_grasp_rot, self.drawer_grasp_rot,
            self.curi_lfinger_pos, self.curi_rfinger_pos,
            self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        self.curi_grasp_rot[:], self.curi_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.curi_local_grasp_rot, self.curi_local_grasp_pos,
                                     drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
                                     )

        self.curi_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.curi_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.curi_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.curi_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.curi_dof_pos - self.curi_dof_lower_limits)
                          / (self.curi_dof_upper_limits - self.curi_dof_lower_limits) - 1.0)
        to_target = self.drawer_grasp_pos - self.curi_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.curi_dof_vel * self.dof_vel_scale, to_target,
                                  self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)),
                                 dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset curi
        pos = tensor_clamp(
            self.curi_default_dof_pos.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_curi_dofs), device=self.device) - 0.5),
            self.curi_dof_lower_limits, self.curi_dof_upper_limits)
        self.curi_dof_pos[env_ids, :] = pos
        self.curi_dof_vel[env_ids, :] = torch.zeros_like(self.curi_dof_vel[env_ids])
        self.curi_dof_targets[env_ids, :self.num_curi_dofs] = pos

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        # reset props
        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.curi_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.curi_grasp_pos[i] + quat_apply(self.curi_grasp_rot[i], to_torch([1, 0, 0],
                                                                                           device=self.device) * 0.2)).cpu().numpy()
                py = (self.curi_grasp_pos[i] + quat_apply(self.curi_grasp_rot[i], to_torch([0, 1, 0],
                                                                                           device=self.device) * 0.2)).cpu().numpy()
                pz = (self.curi_grasp_pos[i] + quat_apply(self.curi_grasp_rot[i], to_torch([0, 0, 1],
                                                                                           device=self.device) * 0.2)).cpu().numpy()

                p0 = self.curi_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                   [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                   [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                   [0.1, 0.1, 0.85])

                px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1],
                                                                                               device=self.device) * 0.2)).cpu().numpy()

                p0 = self.drawer_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.curi_lfinger_pos[i] + quat_apply(self.curi_lfinger_rot[i], to_torch([1, 0, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                py = (self.curi_lfinger_pos[i] + quat_apply(self.curi_lfinger_rot[i], to_torch([0, 1, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                pz = (self.curi_lfinger_pos[i] + quat_apply(self.curi_lfinger_rot[i], to_torch([0, 0, 1],
                                                                                               device=self.device) * 0.2)).cpu().numpy()

                p0 = self.curi_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.curi_rfinger_pos[i] + quat_apply(self.curi_rfinger_rot[i], to_torch([1, 0, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                py = (self.curi_rfinger_pos[i] + quat_apply(self.curi_rfinger_rot[i], to_torch([0, 1, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                pz = (self.curi_rfinger_pos[i] + quat_apply(self.curi_rfinger_rot[i], to_torch([0, 0, 1],
                                                                                               device=self.device) * 0.2)).cpu().numpy()

                p0 = self.curi_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_curi_reward(
        reset_buf, progress_buf, actions, cabinet_dof_pos,
        curi_grasp_pos, drawer_grasp_pos, curi_grasp_rot, drawer_grasp_rot,
        curi_lfinger_pos, curi_rfinger_pos,
        gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
        num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
        finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(curi_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(curi_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    axis3 = tf_vector(curi_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
        -1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if left finger is above the drawer handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(curi_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                       torch.where(curi_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                   around_handle_reward + 0.5, around_handle_reward),
                                       around_handle_reward)
    # reward for distance of each finger from the drawer
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(curi_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    rfinger_dist = torch.abs(curi_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    finger_dist_reward = torch.where(curi_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                     torch.where(curi_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward),
                                     finger_dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # how far the cabinet has been opened out
    open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
              + around_handle_reward_scale * around_handle_reward + 10 * open_reward_scale * open_reward \
              + finger_dist_reward_scale * finger_dist_reward \
        # - action_penalty_scale * action_penalty

    # bonus for opening drawer properly
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # prevent bad style in opening drawer
    rewards = torch.where(curi_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    rewards = torch.where(curi_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)

    # reset if drawer is open or max length reached
    reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards * 0.01, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, curi_local_grasp_rot, curi_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_curi_rot, global_curi_pos = tf_combine(
        hand_rot, hand_pos, curi_local_grasp_rot, curi_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_curi_rot, global_curi_pos, global_drawer_rot, global_drawer_pos
