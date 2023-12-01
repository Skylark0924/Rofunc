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

        envs = []
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)
        self.envs = envs

        if isinstance(self.asset_file, list):
            for asset_file in self.asset_file:
                self.create_env_single(asset_file)
        elif isinstance(self.asset_file, str):
            self.create_env_single(self.asset_file)

    def create_env_single(self, asset_file):
        from isaacgym import gymapi

        # Load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = self.args.env.asset.flip_visual_attachments
        asset_options.armature = self.args.env.asset.armature
        asset_options.vhacd_enabled = True
        beauty_print("Loading robot asset {} from {}".format(asset_file, self.asset_root), type="info")
        object_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file, asset_options)
        object_name = os.path.basename(asset_file).split(".")[0]

        object_handles = []
        init_pose = self.args.env.asset.init_pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*init_pose[:3])
        pose.r = gymapi.Quat(*init_pose[3:7])

        for i in range(self.num_envs):
            env = self.envs[i]
            # add robot
            object_handle = self.gym.create_actor(env, object_asset, pose, object_name, i, 0)
            object_handles.append(object_handle)

    def show(self):
        from isaacgym import gymapi

        while not self.gym.query_viewer_has_closed(self.viewer):
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)

            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        beauty_print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
