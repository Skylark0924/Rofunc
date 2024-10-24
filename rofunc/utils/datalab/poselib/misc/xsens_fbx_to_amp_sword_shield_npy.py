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

"""
Attention: Since the Autodesk FBX SDK just supports Python 3.7, this script should be run with Python 3.7.
"""

import multiprocessing
import os.path
import click
import glob

import numpy as np

import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonMotion,
)
from rofunc.utils.datalab.poselib.poselib.visualization.common import (
    plot_skeleton_motion_interactive,
    plot_skeleton_state,
)


def motion_from_fbx(fbx_file_path, root_joint, fps=60, visualize=True):
    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file_path, root_joint=root_joint, fps=fps
    )
    # visualize motion
    if visualize:
        plot_skeleton_motion_interactive(motion)
    return motion


def motion_retargeting(retarget_cfg, source_motion, visualize=False):
    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(retarget_cfg["source_tpose"])
    if visualize:
        plot_skeleton_state(source_tpose, "source_tpose")

    target_tpose = SkeletonState.from_file(retarget_cfg["target_tpose"])
    if visualize:
        plot_skeleton_state(target_tpose, "target_tpose")

    # parse data from retarget config
    rotation_to_target_skeleton = torch.tensor(retarget_cfg["rotation"])

    # run retargeting
    target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=retarget_cfg["joint_mapping"],
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=retarget_cfg["scale"],
    )

    # plot_skeleton_motion_interactive(target_motion)

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    frame_beg = retarget_cfg["trim_frame_beg"]
    frame_end = retarget_cfg["trim_frame_end"]
    if frame_beg == -1:
        frame_beg = 0

    if frame_end == -1:
        frame_end = target_motion.local_rotation.shape[0]

    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
    )
    target_motion = SkeletonMotion.from_skeleton_state(
        new_sk_state, fps=target_motion.fps
    )

    # need to convert some joints from 3D to 1D (e.g. elbows and knees)
    # target_motion = _project_joints(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_cfg["root_height_offset"]
    root_translation[:, 2] += root_height_offset

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
    )
    target_motion = SkeletonMotion.from_skeleton_state(
        new_sk_state, fps=target_motion.fps
    )

    # save retargeted motion
    target_motion.to_file(retarget_cfg["target_motion_path"])

    if visualize:
        # visualize retargeted motion
        plot_skeleton_motion_interactive(target_motion)


def amp_npy_from_fbx(fbx_file, tpose_file, amp_tpose_file, verbose=True):
    """
    This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
    Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
      - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the
                       same skeleton as the source T-Pose skeleton.
      - target_motion_path: path to save the retargeted motion to
      - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
      - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state
                      (pose should match source T-Pose)
      - joint_mapping: mapping of joint names from source to target
      - rotation: root rotation offset from source to target skeleton (for transforming across different orientation
                  axes), represented as a quaternion in XYZW order.
      - scale: scale offset from source to target skeleton
    """
    target_motion_path = fbx_file[:-4] + ".npy"
    config = {
        "target_motion_path": target_motion_path,
        "source_tpose": tpose_file,
        "target_tpose": amp_tpose_file,
        "joint_mapping": {
            "Hips": "pelvis",
            "LeftUpLeg": "left_thigh",
            "LeftLeg": "left_shin",
            "LeftToeBase": "left_foot",
            "RightUpLeg": "right_thigh",
            "RightLeg": "right_shin",
            "RightToeBase": "right_foot",
            "Spine": "torso",
            "Head": "head",
            "LeftShoulder": "left_shoulder",
            "LeftArm": "left_upper_arm",
            "LeftForeArm": "left_lower_arm",
            "LeftHand": "left_hand",
            "RightShoulder": "right_shoulder",
            "RightArm": "right_upper_arm",
            "RightForeArm": "right_lower_arm",
            "RightHand": "right_hand",
        },
        # "rotation": [0.707, 0, 0, 0.707], xyzw
        "rotation": [0.5, 0.5, 0.5, 0.5],
        "scale": 0.001,
        "root_height_offset": 0.0,
        "trim_frame_beg": 0,
        "trim_frame_end": -1,
    }

    motion = motion_from_fbx(fbx_file, root_joint="Hips", fps=60, visualize=verbose)
    config["target_motion_path"] = fbx_file.replace(".fbx", "_amp.npy")
    motion_retargeting(config, motion, visualize=verbose)


@click.command()
@click.option(
    "--is_parallel",
    default=False,
    help="Whether using parallel conversion.",
)
@click.option(
    "--verbose",
    default=True,
    help="Whether visualize the conversion.",
)
def main(is_parallel, verbose):
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data"
    )
    fbx_files = sorted(glob.glob(os.path.join(data_dir, '*.fbx')))
    tpose_file = os.path.join(data_dir, "new_tpose.npy")

    amp_humanoid_tpose_file = os.path.join(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/amp_humanoid_generated_new_tpose.npy"
    ))

    if is_parallel:
        pool = multiprocessing.Pool()
        pool.map(amp_npy_from_fbx, fbx_files)
    else:
        for fbx_file in fbx_files:
            amp_npy_from_fbx(fbx_file, tpose_file, amp_humanoid_tpose_file, verbose)


if __name__ == "__main__":
    main()
