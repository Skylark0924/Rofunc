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

import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from rofunc.utils.datalab.poselib.poselib.visualization.common import plot_skeleton_state


def get_hotu_tpose(xml_path, save_path, verbose=True):
    skeleton = SkeletonTree.from_mjcf(xml_path)
    # import numpy as np
    # np.save("local_orientation.npy", skeleton.local_orientation)
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
    # xml_path = os.path.join(rofunc_path, "simulator/assets/mjcf/hotu_humanoid_w_qbhand_no_virtual.xml")
    # save_path = os.path.join(rofunc_path, "utils/datalab/poselib/data/target_hotu_humanoid_w_qbhand_tpose.npy")

    xml_path = os.path.join(rofunc_path, "simulator/assets/mjcf/hotu/hotu_humanoid_w_qbhand_full_new.xml")
    save_path = os.path.join(rofunc_path, "utils/datalab/poselib/data/target_hotu_humanoid_w_qbhand_full_tpose.npy")
    get_hotu_tpose(xml_path, save_path)
