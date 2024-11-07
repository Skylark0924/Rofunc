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
import os.path

import matplotlib.pyplot as plt
import numpy as np

from rofunc.simulator.base_sim import PlaygroundSim
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.utils.visualab.image import overlay_seg_w_img


class ObjectSim:
    def __init__(self, args, asset_file, asset_root=None):
        """

        :param args:
        :param asset_file: can be a list of asset files
        :param asset_root:
        """
        self.args = args
        self.asset_file = asset_file
        self.asset_root = asset_root
        if asset_root is None:
            from rofunc.utils.oslab import get_rofunc_path
            self.asset_root = os.path.join(get_rofunc_path(), "simulator/assets")

        # Initial gym, sim, and viewer
        self.PlaygroundSim = PlaygroundSim(self.args)
        self.gym = self.PlaygroundSim.gym
        self.sim = self.PlaygroundSim.sim
        self.viewer = self.PlaygroundSim.viewer

        self.num_envs = 1
        self.visual_obs_flag = False
        self.create_env()

    def create_env(self):
        from isaacgym import gymapi

        # Set up the env grid
        spacing = 5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))
        print("Creating %d environments" % self.num_envs)

        table_dims = gymapi.Vec3(1, 2, 0.5)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*[0.0, 0.0, 0.25])
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        envs = []
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)
            envs.append(env)
        self.envs = envs

        if isinstance(self.asset_file, list):
            for i, asset_file in enumerate(self.asset_file):
                self.create_object_single(asset_file, i + 1)
        elif isinstance(self.asset_file, str):
            self.create_object_single(self.asset_file)

    def create_object_single(self, asset_file, index=1):
        from isaacgym import gymapi

        # Load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = self.args.env.asset.armature
        asset_options.convex_decomposition_from_submeshes = self.args.env.asset.convex_decomposition_from_submeshes
        asset_options.disable_gravity = self.args.env.asset.disable_gravity
        asset_options.fix_base_link = self.args.env.asset.fix_base_link
        asset_options.flip_visual_attachments = self.args.env.asset.flip_visual_attachments
        asset_options.use_mesh_materials = self.args.env.asset.use_mesh_materials
        asset_options.vhacd_enabled = self.args.env.asset.vhacd_enabled
        for vhacd_param in self.args.env.asset.vhacd_params:
            setattr(asset_options.vhacd_params, vhacd_param, self.args.env.asset.vhacd_params[vhacd_param])
        beauty_print("Loading robot asset {} from {}".format(asset_file, self.asset_root), type="info")
        object_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options)
        object_name = os.path.basename(asset_file).split(".")[0]

        object_handles = []
        init_pose = self.args.env.asset.init_pose
        init_position = np.array(init_pose[:3]) + np.array([0, 0, 0.3]) * index
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*init_position)
        pose.r = gymapi.Quat(*init_pose[3:7])

        for i in range(self.num_envs):
            env = self.envs[i]
            # add robot
            object_handle = self.gym.create_actor(env, object_asset, pose, object_name, i, 0)
            self.gym.set_rigid_body_segmentation_id(env, object_handle, 0, index)
            object_handles.append(object_handle)

    def show(self, mode="rgb"):
        """
        Show the simulation

        :param mode: visual mode, can be "rgb", "depth", "seg"
        :return:
        """
        from isaacgym import gymapi

        # create a local copy of initial state, which we can send back for reset
        initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

        # subscribe to R event for reset
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        if self.visual_obs_flag:
            fig = plt.figure(mode.upper(), figsize=(16, 8))

        while not self.gym.query_viewer_has_closed(self.viewer):
            # Get input actions from the viewer and handle them appropriately
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(self.sim, initial_state, gymapi.STATE_ALL)

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if self.visual_obs_flag:
                # digest image
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                cam_img0 = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0],
                                                     gymapi.IMAGE_COLOR).reshape(1280, 1280, 4)
                cam_img0_depth = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0],
                                                           gymapi.IMAGE_DEPTH).reshape(1280, 1280)
                cam_img0_seg = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0],
                                                         gymapi.IMAGE_SEGMENTATION).reshape(1280, 1280)
                cam_img1 = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[1],
                                                     gymapi.IMAGE_COLOR).reshape(1280, 1280, 4)
                cam_img1_depth = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[1],
                                                           gymapi.IMAGE_DEPTH).reshape(1280, 1280)
                cam_img1_seg = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[1],
                                                         gymapi.IMAGE_SEGMENTATION).reshape(1280, 1280)

                if mode == "rgb":
                    cam_img = np.concatenate((cam_img0[:, :, :3], cam_img1[:, :, :3]), axis=1)
                    plt.imshow(cam_img)
                elif mode == "depth":
                    cam_img_depth = np.concatenate((cam_img0_depth, cam_img1_depth), axis=1)
                    cam_img_depth = cam_img_depth / np.abs(cam_img_depth).max() * 255
                    plt.imshow(cam_img_depth, 'gray')
                elif mode == "seg":
                    cam_img = np.concatenate((cam_img0[:, :, :3], cam_img1[:, :, :3]), axis=1)
                    cam_img_seg = np.concatenate((cam_img0_seg, cam_img1_seg), axis=1)
                    image_with_masks = overlay_seg_w_img(cam_img, cam_img_seg, alpha=0.5)
                    plt.imshow(image_with_masks)
                else:
                    raise NotImplementedError

                plt.axis('off')
                plt.pause(1e-9)
                fig.clf()

                self.gym.end_access_image_tensors(self.sim)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        beauty_print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def create_track_cameras(self):
        from isaacgym import gymapi

        # track cameras
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = 1280
        camera_props.height = 1280

        for env_idx in range(self.num_envs):
            env_ptr = self.envs[env_idx]
            camera_handle0 = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_location(camera_handle0, env_ptr, gymapi.Vec3(0.5, -0.5, 1.3), gymapi.Vec3(0, 0, 0))

        for env_idx in range(self.num_envs):
            env_ptr = self.envs[env_idx]
            camera_handle1 = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.set_camera_location(camera_handle1, env_ptr, gymapi.Vec3(0.5, 0.5, 1.3), gymapi.Vec3(0, 0, 0))

        self.gym.render_all_camera_sensors(self.sim)
        self.visual_obs_flag = True
        self.camera_handles = [camera_handle0, camera_handle1]
