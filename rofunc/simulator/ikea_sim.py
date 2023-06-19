"""
 Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import isaacgym
import os
from rofunc.simulator.franka_sim import FrankaSim
from rofunc.utils.logger.beauty_logger import beauty_print


class IkeaSim:
    def __init__(self, args, furniture_name, **kwargs):
        self.args = args
        self.furniture_name = furniture_name
        self.robot_sim = FrankaSim(args)
        self.asset_root = self.robot_sim.asset_root
        self._init_env_w_furniture()

    def _init_env_w_furniture(self):
        from isaacgym import gymtorch, gymapi

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005



        if self.furniture_name == "bed_dalsev":
            self.furniture_asset_folder = "urdf/objects/bed_dalsev/"
        elif self.furniture_name == "bench_bjoderna":
            self.furniture_asset_folder = "urdf/objects/bench_bjoderna/"
        elif self.furniture_name == "bench_bjursta":
            self.furniture_asset_folder = "urdf/objects/bench_bjursta/"
        elif self.furniture_name == "blocks":
            self.furniture_asset_folder = "urdf/objects/blocks/"
        elif self.furniture_name == "bookcase_agerum":
            self.furniture_asset_folder = "urdf/objects/bookcase_agerum/"
        elif self.furniture_name == "bookcase_besta_0165":
            self.furniture_asset_folder = "urdf/objects/bookcase_besta_0165/"
        elif self.furniture_name == "bookcase_besta_0170":
            self.furniture_asset_folder = "urdf/objects/bookcase_besta_0170/"
        elif self.furniture_name == "bookcase_besta_0172":
            self.furniture_asset_folder = "urdf/objects/bookcase_besta_0172/"
        elif self.furniture_name == "bookcase_billy_0190":
            self.furniture_asset_folder = "urdf/objects/bookcase_billy_0190/"
        elif self.furniture_name == "bookcase_billy_0191":
            self.furniture_asset_folder = "urdf/objects/bookcase_billy_0191/"
        elif self.furniture_name == "bookcase_expedit_0373":
            self.furniture_asset_folder = "urdf/objects/bookcase_expedit_0373/"
        elif self.furniture_name == "bookcase_expedit_0374":
            self.furniture_asset_folder = "urdf/objects/bookcase_expedit_0374/"
        elif self.furniture_name == "bookcase_expedit_0376":
            self.furniture_asset_folder = "urdf/objects/bookcase_expedit_0376/"
        elif self.furniture_name == "bookcase_expedit_0385":
            self.furniture_asset_folder = "urdf/objects/bookcase_expedit_0385/"
        elif self.furniture_name == "bookcase_flaerke_0404":
            self.furniture_asset_folder = "urdf/objects/bookcase_flaerke_0404/"
        elif self.furniture_name == "bookcase_grevback_0484":
            self.furniture_asset_folder = "urdf/objects/bookcase_grevback_0484/"
        elif self.furniture_name == "bookcase_hensvik_0565":
            self.furniture_asset_folder = "urdf/objects/bookcase_hensvik_0565/"
        elif self.furniture_name == "box_ivar_0666":
            self.furniture_asset_folder = "urdf/objects/box_ivar_0666/"
        elif self.furniture_name == "box_lekman_0858":
            self.furniture_asset_folder = "urdf/objects/box_lekman_0858/"
        elif self.furniture_name == "cabinet_akurum_0011":
            self.furniture_asset_folder = "urdf/objects/cabinet_akurum_0011/"
        elif self.furniture_name == "cabinet_akurum_0014":
            self.furniture_asset_folder = "urdf/objects/cabinet_akurum_0014/"
        elif self.furniture_name == "cabinet_akurum_0019":
            self.furniture_asset_folder = "urdf/objects/cabinet_akurum_0019/"
        elif self.furniture_name == "cabinet_akurum_0021":
            self.furniture_asset_folder = "urdf/objects/cabinet_akurum_0021/"
        elif self.furniture_name == "cabinet_bjorken_0203":
            self.furniture_asset_folder = "urdf/objects/cabinet_bjorken_0203/"
        elif self.furniture_name == "cabinet_lillagen_0933":
            self.furniture_asset_folder = "urdf/objects/cabinet_lillagen_0933/"
        elif self.furniture_name == "chair_agam_0005":
            self.furniture_asset_folder = "urdf/objects/chair_agam_0005/"
        elif self.furniture_name == "chair_agne_0007":
            self.furniture_asset_folder = "urdf/objects/chair_agne_0007/"
        elif self.furniture_name == "chair_agne_0010":
            self.furniture_asset_folder = "urdf/objects/chair_agne_0010/"
        elif self.furniture_name == "chair_balser_0115":
            self.furniture_asset_folder = "urdf/objects/chair_balser_0115/"
        elif self.furniture_name == "chair_bernhard_0146":
            self.furniture_asset_folder = "urdf/objects/chair_bernhard_0146/"
        elif self.furniture_name == "chair_ingolf_0650":
            self.furniture_asset_folder = "urdf/objects/chair_ingolf_0650/"
        elif self.furniture_name == "chair_ivar_0668":
            self.furniture_asset_folder = "urdf/objects/chair_ivar_0668/"
        elif self.furniture_name == "desk_fredrik_0430":
            self.furniture_asset_folder = "urdf/objects/desk_fredrik_0430/"
        elif self.furniture_name == "desk_hannes_0529":
            self.furniture_asset_folder = "urdf/objects/desk_hannes_0529/"
        elif self.furniture_name == "desk_mikael_1064":
            self.furniture_asset_folder = "urdf/objects/desk_mikael_1064/"
        elif self.furniture_name == "shelf_ivar_0678":
            self.furniture_asset_folder = "urdf/objects/shelf_ivar_0678/"
        elif self.furniture_name == "shelf_liden_0922":
            self.furniture_asset_folder = "urdf/objects/shelf_liden_0922/"
        elif self.furniture_name == "swivel_chair_0700":
            self.furniture_asset_folder = "urdf/objects/swivel_chair_0700/"
        elif self.furniture_name == "shelf_lillagen_0927":
            self.furniture_asset_folder = "urdf/objects/shelf_lillagen_0927/"
        elif self.furniture_name == "table_benno_0141":
            self.furniture_asset_folder = "urdf/objects/table_benno_0141/"
        elif self.furniture_name == "table_billsta_round_0189":
            self.furniture_asset_folder = "urdf/objects/table_billsta_round_0189/"
        elif self.furniture_name == "table_bjorkudden_0206":
            self.furniture_asset_folder = "urdf/objects/table_bjorkudden_0206/"
        elif self.furniture_name == "table_bjorkudden_0207":
            self.furniture_asset_folder = "urdf/objects/table_bjorkudden_0207/"
        elif self.furniture_name == "table_dalom_0267":
            self.furniture_asset_folder = "urdf/objects/table_dalom_0267/"
        elif self.furniture_name == "table_dockstra_0279":
            self.furniture_asset_folder = "urdf/objects/table_dockstra_0279/"
        elif self.furniture_name == "table_expedit_0387":
            self.furniture_asset_folder = "urdf/objects/table_expedit_0387/"
        elif self.furniture_name == "table_hemnes_0539":
            self.furniture_asset_folder = "urdf/objects/table_hemnes_0539/"
        elif self.furniture_name == "table_hemnes_0541":
            self.furniture_asset_folder = "urdf/objects/table_hemnes_0541/"
        elif self.furniture_name == "table_jokkmokk_0694":
            self.furniture_asset_folder = "urdf/objects/table_jokkmokk_0694/"
        elif self.furniture_name == "table_klubbo_0740":
            self.furniture_asset_folder = "urdf/objects/table_klubbo_0740/"
        elif self.furniture_name == "table_klubbo_0743":
            self.furniture_asset_folder = "urdf/objects/table_klubbo_0743/"
        elif self.furniture_name == "table_lack_0825":
            self.furniture_asset_folder = "urdf/objects/table_lack_0825/"
        elif self.furniture_name == "table_liden_0919":
            self.furniture_asset_folder = "urdf/objects/table_liden_0919/"
        elif self.furniture_name == "table_liden_0920":
            self.furniture_asset_folder = "urdf/objects/table_liden_0920/"
        elif self.furniture_name == "table_liden_0921":
            self.furniture_asset_folder = "urdf/objects/table_liden_0921/"
        elif self.furniture_name == "table_torsby_1549":
            self.furniture_asset_folder = "urdf/objects/table_torsby_1549/"
        elif self.furniture_name == "three_blocks":
            self.furniture_asset_folder = "urdf/objects/three_blocks/"
        elif self.furniture_name == "toy_table":
            self.furniture_asset_folder = "urdf/objects/toy_table/"
        elif self.furniture_name == "tvunit_0406":
            self.furniture_asset_folder = "urdf/objects/tvunit_0406/"
        elif self.furniture_name == "tvunit_lack_0829":
            self.furniture_asset_folder = "urdf/objects/tvunit_lack_0829/"
        elif self.furniture_name == "tvunit_lack_0830":
            self.furniture_asset_folder = "urdf/objects/tvunit_lack_0830/"

        beauty_print("Loading furniture asset {} from {}".format(self.furniture_asset_folder, self.asset_root),
                     type="info")

        furniture_assets = []
        furniture_poses = []

        for asset_file in os.listdir(os.path.join(self.asset_root, self.furniture_asset_folder)):
            if asset_file.endswith(".xml"):
                asset_file = os.path.join(self.furniture_asset_folder, asset_file)
                furniture_asset = self.robot_sim.gym.load_asset(self.robot_sim.sim, self.asset_root, asset_file,
                                                                asset_options)
                furniture_pose = gymapi.Transform()
                furniture_pose.p = gymapi.Vec3(0, 1, 0)
                furniture_assets.append(furniture_asset)
                furniture_poses.append(furniture_pose)

        furniture_handles = []
        for i in range(self.robot_sim.num_envs):
            # add furniture
            for j in range(len(furniture_assets)):
                handle = self.robot_sim.gym.create_actor(self.robot_sim.envs[i], furniture_assets[j],
                                                         furniture_poses[j], "furniture", i, i, 1)
                actor_handle = self.robot_sim.gym.find_actor_handle(self.robot_sim.envs[i], "furniture")

                rigid_body_index = self.robot_sim.gym.find_actor_rigid_body_index(self.robot_sim.envs[i], actor_handle, "furniture", 0)
                print(rigid_body_index)
                texture_handle = self.robot_sim.gym.get_rigid_body_texture(self.robot_sim.envs[i], handle, rigid_body_index, gymapi.MESH_VISUAL(0.8, 0.8, 0.8))
                self.robot_sim.gym.set_rigid_body_texture(self.robot_sim.envs[i], handle, rigid_body_index, gymapi.MESH_VISUAL_AND_COLLISION(0.8, 0.8, 0.8), texture_handle)
                # self.robot_sim.gym.set_rigid_body_color(self.robot_sim.envs[i], handle, rigid_body_index,
                #                                         gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8, 0.8, 0.8))
                self.robot_sim.gym.enable_actor_dof_force_sensors(self.robot_sim.envs[i], handle)
                furniture_handles.append(handle)

    def show(self):
        self.robot_sim.show()


if __name__ == '__main__':
    from isaacgym import gymutil

    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False

    furniture_name = "shelf_liden_0922"
    ikea_sim = IkeaSim(args, furniture_name)
    ikea_sim.show()
