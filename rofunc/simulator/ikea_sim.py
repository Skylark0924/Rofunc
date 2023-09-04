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

import isaacgym
import os
from github import Github

from rofunc.simulator.franka_sim import FrankaSim
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.utils.downloader.github_downloader import download_folder, download_file


class IkeaSim:
    def __init__(self, args, furniture_name, **kwargs):
        self.args = args
        self.furniture_name = furniture_name
        self.robot_sim = FrankaSim(args)
        self.robot_sim.setup_robot_dof_prop()

        self.asset_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets"
        )
        self._texture_asset_folder = os.path.join(self.asset_root, "mjcf/textures")
        self.furniture_asset_folder = os.path.join(self.asset_root, "mjcf/ikea")
        self._init_env_w_furniture()

    def _init_env_w_furniture(self):
        from isaacgym import gymapi

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005

        furniture_model_folder = os.path.join(
            self.furniture_asset_folder, f"{self.furniture_name}"
        )
        furniture_description = os.path.join(
            self.furniture_asset_folder, f"{self.furniture_name}.xml"
        )

        repo = Github().get_repo("clvrai/furniture")
        if not os.path.exists(furniture_description):
            download_file(
                repo,
                f"furniture/env/models/assets/objects/{self.furniture_name}.xml",
                furniture_description,
            )

        if not os.path.exists(furniture_model_folder):
            os.makedirs(furniture_model_folder)
            download_folder(
                repo,
                f"furniture/env/models/assets/objects/{self.furniture_name}",
                furniture_model_folder,
                False,
            )

        if not os.path.exists(self._texture_asset_folder):
            os.makedirs(self._texture_asset_folder)
            download_folder(
                repo,
                f"furniture/env/models/assets/textures",
                self._texture_asset_folder,
                False,
            )

        beauty_print(
            "Loading furniture asset {}".format(furniture_description),
            type="info",
        )

        furniture_assets = []
        furniture_poses = []

        # If the return asset is None, pay attention to the output:
        # Worldbody with multiple direct child bodies not currently supported!
        # You need create a parent body for all child bodies
        furniture_asset = self.robot_sim.gym.load_asset(
            self.robot_sim.sim, self.asset_root, f"mjcf/ikea/{self.furniture_name}.xml", asset_options
        )
        furniture_assets.append(furniture_asset)

        furniture_pose = gymapi.Transform()
        furniture_pose.p = gymapi.Vec3(0.7, 0.3, 0)
        furniture_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        furniture_poses.append(furniture_pose)

        furniture_handles = []
        for i in range(self.robot_sim.num_envs):
            # add furniture
            for j in range(len(furniture_assets)):
                handle = self.robot_sim.gym.create_actor(
                    self.robot_sim.envs[i],
                    furniture_assets[j],
                    furniture_poses[j],
                    "furniture",
                    i,
                    i,
                    1,
                )
                actor_handle = self.robot_sim.gym.find_actor_handle(
                    self.robot_sim.envs[i], "furniture"
                )

                rigid_body_index = self.robot_sim.gym.find_actor_rigid_body_index(
                    self.robot_sim.envs[i], actor_handle, "furniture", gymapi.DOMAIN_ENV
                )
                texture_handle = self.robot_sim.gym.get_rigid_body_texture(
                    self.robot_sim.envs[i],
                    handle,
                    rigid_body_index,
                    gymapi.MESH_VISUAL,
                )
                self.robot_sim.gym.set_rigid_body_texture(
                    self.robot_sim.envs[i],
                    handle,
                    rigid_body_index,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    texture_handle,
                )
                # self.robot_sim.gym.set_rigid_body_color(self.robot_sim.envs[i], handle, rigid_body_index,
                #                                         gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8, 0.8, 0.1))
                self.robot_sim.gym.enable_actor_dof_force_sensors(
                    self.robot_sim.envs[i], handle
                )
                furniture_handles.append(handle)

    def show(self):
        self.robot_sim.show()


if __name__ == "__main__":
    from isaacgym import gymutil

    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False

    furniture_name = "box_ivar_0666"
    ikea_sim = IkeaSim(args, furniture_name)
    ikea_sim.show()
