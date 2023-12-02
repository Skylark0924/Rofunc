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

import math
import os.path
import numpy as np

from rofunc.simulator.base_sim import PlaygroundSim
from rofunc.utils.logger.beauty_logger import beauty_print


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
                self.create_object_single(asset_file, i)
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
            object_handles.append(object_handle)

    def show(self):
        from isaacgym import gymapi

        # create a local copy of initial state, which we can send back for reset
        initial_state = np.copy(self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL))

        # subscribe to R event for reset
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

        while not self.gym.query_viewer_has_closed(self.viewer):
            # Get input actions from the viewer and handle them appropriately
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(self.sim, initial_state, gymapi.STATE_ALL)

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        beauty_print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
