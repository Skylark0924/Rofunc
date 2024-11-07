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

import isaacgym
import os
from github import Github

from isaacgym import gymapi

from rofunc.simulator.franka_sim import FrankaSim
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.config.utils import load_ikea_config
from rofunc.utils.downloader.github_downloader import download_folder, download_file


class IkeaSim:
    def __init__(self, args, furniture_ids, **kwargs):
        self.args = args
        self.robot_sim = FrankaSim(args)
        self.robot_sim.setup_robot_dof_prop()

        self.asset_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets"
        )
        self._texture_asset_folder = os.path.join(self.asset_root, "mjcf/textures")
        self._furniture_asset_folder = os.path.join(self.asset_root, "mjcf/ikea")

        for furniture_id in furniture_ids:
            ikea_config = load_ikea_config(furniture_id)
            self._init_furniture(ikea_config)

    def _init_furniture(self, ikea_config):
        furniture_name = ikea_config["name"]

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = ikea_config["flip_visual_attachments"]
        asset_options.fix_base_link = ikea_config["fix_base_link"]
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005

        furniture_model_folder = os.path.join(
            self._furniture_asset_folder, f"{furniture_name}"
        )
        furniture_description = os.path.join(
            self._furniture_asset_folder, f"{furniture_name}.xml"
        )

        repo = Github().get_repo("clvrai/furniture")
        if not os.path.exists(furniture_description):
            download_file(
                repo,
                f"furniture/env/models/assets/objects/{furniture_name}.xml",
                furniture_description,
            )

        if not os.path.exists(furniture_model_folder):
            os.makedirs(furniture_model_folder)
            download_folder(
                repo,
                f"furniture/env/models/assets/objects/{furniture_name}",
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
            self.robot_sim.sim,
            self.asset_root,
            ikea_config["description_file_path"],
            asset_options,
        )
        furniture_assets.append(furniture_asset)

        pose = ikea_config["initial_base_pose"]
        furniture_pose = gymapi.Transform()
        furniture_pose.p = gymapi.Vec3(pose[0], pose[1], pose[2])
        furniture_pose.r = gymapi.Quat(pose[-4], pose[-3], pose[-2], pose[-1])
        furniture_poses.append(furniture_pose)

        furniture_handles = []
        for i in range(self.robot_sim.num_envs):
            # add furniture
            for j in range(len(furniture_assets)):
                actor_handle = self.robot_sim.gym.create_actor(
                    self.robot_sim.envs[i],
                    furniture_assets[j],
                    furniture_poses[j],
                    furniture_name,
                    i,
                    i,
                    1,
                )
                for body_name in ikea_config["body_names"]:
                    self._apply_texture(
                        self.robot_sim.envs[i],
                        actor_handle,
                        body_name,
                        ikea_config["texture_file_name"],
                    )

                # self.robot_sim.gym.set_rigid_body_color(self.robot_sim.envs[i], actor_handle, rigid_body_index,
                #                                         gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.1, 0.1))
                self.robot_sim.gym.enable_actor_dof_force_sensors(
                    self.robot_sim.envs[i], actor_handle
                )
                furniture_handles.append(actor_handle)

    def _apply_texture(self, env, actor_handle, body_name, texture_file_name=None):
        rigid_body_index = self.robot_sim.gym.find_actor_rigid_body_index(
            env, actor_handle, body_name, gymapi.DOMAIN_ACTOR
        )
        texture_handle = self.robot_sim.gym.create_texture_from_file(
            self.robot_sim.sim,
            os.path.join(self._texture_asset_folder, texture_file_name),
        )
        self.robot_sim.gym.set_rigid_body_texture(
            env,
            actor_handle,
            rigid_body_index,
            gymapi.MESH_VISUAL,
            texture_handle,
        )

    def show(self):
        self.robot_sim.show()


if __name__ == "__main__":
    from isaacgym import gymutil

    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False

    # The list hosts all objects to be added into the scene,
    # To add multiple the same type of objects, multiple config files should be created
    # The order is not restricted
    ikea_sim = IkeaSim(args, ["shelf_liden_1", "box_ivar_1"])
    ikea_sim.show()
