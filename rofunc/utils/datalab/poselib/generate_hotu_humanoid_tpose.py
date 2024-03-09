# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
This scripts imports a MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

import os

import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from rofunc.utils.datalab.poselib.poselib.visualization.common import plot_skeleton_state


def get_hotu_tpose(xml_path, save_path, verbose=True):
    skeleton = SkeletonTree.from_mjcf(xml_path)
    import numpy as np
    np.save("local_orientation.npy", skeleton.local_orientation)
    # generate zero rotation pose
    zero_pose = SkeletonState.zero_pose_wo_qbhand(skeleton)
    # plot_skeleton_state(zero_pose, verbose=False)

    # adjust pose into a T Pose
    local_rotation = zero_pose.local_rotation
    local_rotation[skeleton.index("left_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
        local_rotation[skeleton.index("left_upper_arm")]
    )
    local_rotation[skeleton.index("right_upper_arm")] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
        local_rotation[skeleton.index("right_upper_arm")]
    )
    # local_rotation[skeleton.index("right_hand")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([180.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    #     local_rotation[skeleton.index("right_hand")]
    # )
    # local_rotation[skeleton.index("left_hand")] = quat_mul(
    #     quat_from_angle_axis(angle=torch.tensor([180.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
    #     local_rotation[skeleton.index("left_hand")]
    # )

    # finger_tune_list = ["right_qbhand_thumb_knuckle_link", "right_qbhand_index_knuckle_link",
    #                     "right_qbhand_middle_knuckle_link", "right_qbhand_ring_knuckle_link",
    #                     "right_qbhand_little_knuckle_link"]
    # for finger_tune in finger_tune_list:
    #     local_rotation[skeleton.index(finger_tune)] = quat_mul(
    #         quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    #         local_rotation[skeleton.index(finger_tune)]
    #     )
    # finger_tune_list = ["left_qbhand_thumb_knuckle_link", "left_qbhand_index_knuckle_link",
    #                     "left_qbhand_middle_knuckle_link", "left_qbhand_ring_knuckle_link",
    #                     "left_qbhand_little_knuckle_link"]
    # for finger_tune in finger_tune_list:
    #     local_rotation[skeleton.index(finger_tune)] = quat_mul(
    #         quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([0.0, 1.0, 0.0]), degree=True),
    #         local_rotation[skeleton.index(finger_tune)]
    #     )
    # translation = zero_pose.root_translation
    # translation += torch.tensor([0, 0, 0.9])

    # save and visualize T-pose
    zero_pose.to_file(save_path)
    if verbose:
        plot_skeleton_state(zero_pose, verbose=True)


if __name__ == '__main__':
    rofunc_path = rf.oslab.get_rofunc_path()
    xml_path = os.path.join(rofunc_path, "simulator/assets/mjcf/hotu_humanoid_w_qbhand_no_virtual.xml")
    save_path = os.path.join(rofunc_path, "utils/datalab/poselib/data/target_hotu_humanoid_w_qbhand_tpose.npy")
    get_hotu_tpose(xml_path, save_path)
