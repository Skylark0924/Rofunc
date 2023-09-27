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

from rofunc.simulator.base_sim import RobotSim


class BaxterSim(RobotSim):
    def __init__(self, args, robot_name, asset_root=None, asset_file=None, fix_base_link=None,
                 flip_visual_attachments=True, init_pose_vec=None, num_envs=1, device="cpu"):
        super().__init__(args, robot_name, asset_root, asset_file, fix_base_link, flip_visual_attachments,
                         init_pose_vec, num_envs, device)
        self.asset_file = "urdf/baxter/robot.xml"
        self.init_pose = (0., 1., 0., -0.707107, 0., 0., 0.707107) if init_pose_vec is None else init_pose_vec
