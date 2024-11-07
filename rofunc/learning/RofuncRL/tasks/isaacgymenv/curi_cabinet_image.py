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

import tqdm
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import rofunc as rf
from rofunc.learning.RofuncRL.tasks.isaacgymenv.base.curi_base_task import CURIBaseTask
from rofunc.learning.RofuncRL.tasks.isaacgymenv.curi_cabinet import CURICabinetTask
from rofunc.utils.oslab.path import get_rofunc_path


class CURICabinetImageTask(CURICabinetTask, CURIBaseTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        CURIBaseTask.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture,
                              force_render)
        self.obs_buf = torch.zeros((self.num_envs, 128, 128, 4), device=self.device, dtype=torch.float)

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
            [0., 0., 0.3863, 0.5062, -0.1184, -0.6105, 0.023, 1.6737, 0.9197, 0.04, 0.04, -0.5349, 0, 0.1401, -1.7951,
             0.0334, 3.2965, 0.6484, 0.04, 0.04], device=self.device)
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

        curi_asset_file = "urdf/curi/urdf/curi_isaacgym_dual_arm_w_head.urdf"
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        # load curi asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        # asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.use_mesh_materials = True
        curi_asset = self.gym.load_asset(self.sim, asset_root, curi_asset_file, asset_options)

        """
        Rigid body dict for curi_isaacgym_dual_arm_w_head.urdf (since we use `collapse_fixed_joints` here)
        {'head_link1': 2,
         'head_link2': 3,
         'panda_left_leftfinger': 11,
         'panda_left_link1': 4,
         'panda_left_link2': 5,
         'panda_left_link3': 6,
         'panda_left_link4': 7,
         'panda_left_link5': 8,
         'panda_left_link6': 9,
         'panda_left_link7': 10,
         'panda_left_rightfinger': 12,
         'panda_right_leftfinger': 20,
         'panda_right_link1': 13,
         'panda_right_link2': 14,
         'panda_right_link3': 15,
         'panda_right_link4': 16,
         'panda_right_link5': 17,
         'panda_right_link6': 18,
         'panda_right_link7': 19,
         'panda_right_rightfinger': 21,
         'summit_xls_base_footprint': 1,
         'world': 0}
         """

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        curi_dof_stiffness = to_torch(
            [400, 400, 400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6, 400, 400, 400, 400, 400, 400, 400, 1.0e6,
             1.0e6], dtype=torch.float, device=self.device)
        curi_dof_damping = to_torch(
            [80, 80, 80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2],
            dtype=torch.float,
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

        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cfg['env']['image_size']
        camera_props.height = self.cfg['env']['image_size']
        attached_body = "head_link2"
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.12, 0, 0.18)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(90.0)) * \
                            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90.0))

        self.curis = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []

        rf.logger.beauty_print("Creating {} environments".format(num_envs), type="info")
        for i in tqdm.trange(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            curi_actor = self.gym.create_actor(env_ptr, curi_asset, curi_start_pose, "curi", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, curi_actor, curi_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Camera Sensor
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, attached_body)
            self.gym.attach_camera_to_body(camera_handle, env_ptr, body_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)
            # obtain camera tensor
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            # wrap camera tensor in a pytorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(cam_tensor)

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
            self.camera_handles.append(camera_handle)
            self.camera_tensors.append(torch_camera_tensor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, "panda_left_link7")
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, "panda_left_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, curi_actor, "panda_left_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(
            self.num_envs, self.num_props, 13)

        # self.create_cameras()
        self.init_data()

    def create_cameras(self):
        # Camera Sensor
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = 128
        camera_props.height = 128

        attached_body = "head_link2"
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.12, 0, 0.18)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(90.0)) * \
                            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90.0))

        self.camera_handles = []
        self.camera_tensors = []
        rf.logger.beauty_print("Attaching {} cameras".format(self.num_envs), type="info")

        # def parallel(i):
        #     camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
        #     body_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.curis[i], attached_body)
        #     self.gym.attach_camera_to_body(camera_handle, self.envs[i], body_handle, local_transform,
        #                                    gymapi.FOLLOW_TRANSFORM)
        #
        #     # obtain camera tensor
        #     cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], camera_handle, gymapi.IMAGE_COLOR)
        #     # wrap camera tensor in a pytorch tensor
        #     torch_camera_tensor = gymtorch.wrap_tensor(cam_tensor)
        #
        #     self.camera_handles.append(camera_handle)
        #     self.camera_tensors.append(torch_camera_tensor)
        #
        # from multiprocessing.pool import ThreadPool
        # # multiprocessing.set_start_method('fork')
        # pool = ThreadPool()
        # env_list = np.arange(self.num_envs)
        # pool.map(parallel, env_list)
        # pool.close()

        for i in tqdm.trange(self.num_envs):
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            body_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.curis[i], attached_body)
            self.gym.attach_camera_to_body(camera_handle, self.envs[i], body_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

            # obtain camera tensor
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], camera_handle, gymapi.IMAGE_COLOR)
            # wrap camera tensor in a pytorch tensor
            # torch_camera_tensor = gymtorch.wrap_tensor(cam_tensor)

            self.camera_handles.append(camera_handle)
            # self.camera_tensors.append(torch_camera_tensor)

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Obtain image observations from camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.obs_buf = torch.permute(torch.stack(self.camera_tensors).cuda()[:, :, :, :3], (0, 3, 1, 2))

        self.gym.end_access_image_tensors(self.sim)

        # hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        # hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        # drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        # drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        # self.curi_grasp_rot[:], self.curi_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
        #     compute_grasp_transforms(hand_rot, hand_pos, self.curi_local_grasp_rot, self.curi_local_grasp_pos,
        #                              drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
        #                              )
        #
        # self.curi_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        # self.curi_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        # self.curi_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        # self.curi_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]
        #
        # dof_pos_scaled = (2.0 * (self.curi_dof_pos - self.curi_dof_lower_limits)
        #                   / (self.curi_dof_upper_limits - self.curi_dof_lower_limits) - 1.0)
        # to_target = self.drawer_grasp_pos - self.curi_grasp_pos
        # self.obs_buf = torch.cat((dof_pos_scaled, self.curi_dof_vel * self.dof_vel_scale, to_target,
        #                           self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)),
        #                          dim=-1)
        return self.obs_buf
