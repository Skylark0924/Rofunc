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
import os
import sys

import numpy as np
from isaacgym import gymapi

import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from rofunc.utils.datalab.poselib.poselib.visualization.common import plot_skeleton_motion_interactive, \
    plot_skeleton_state

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_sim(motion):
    body_links = {"torso": gymapi.AXIS_ROTATION,
                  "right_limb_l4": gymapi.AXIS_ROTATION, "left_limb_l4": gymapi.AXIS_ROTATION,
                  "right_limb_l7": gymapi.AXIS_ALL, "left_limb_l7": gymapi.AXIS_ALL,
                  "base_link": gymapi.AXIS_ROTATION,
                  "left_leg_l1": gymapi.AXIS_ROTATION, "right_leg_l1": gymapi.AXIS_ROTATION,
                  "left_limb_l1": gymapi.AXIS_ROTATION, "right_limb_l1": gymapi.AXIS_ROTATION,
                  "left_leg_l4": gymapi.AXIS_ALL, "right_leg_l4": gymapi.AXIS_ALL,
                  "left_leg_l6": gymapi.AXIS_ALL, "right_leg_l6": gymapi.AXIS_ALL,
                  "head_l2": gymapi.AXIS_ROTATION,}
    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links.keys()]

    all_links = list(body_links.keys())
    all_ids = body_ids
    all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    motion_rb_states_pos[:, :, 2] += 0.15
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("Walker")
    Walkersim = rf.sim.RobotSim(args)
    dof_states = Walkersim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        update_freq=0.001,
        root_state=motion_root_states,
        attr_types=all_types,
        verbose=False
    )
    return dof_states


def motion_from_fbx(fbx_file_path, root_joint, fps=60, visualize=True):
    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file_path,
        root_joint=root_joint,
        fps=fps
    )
    # visualize motion
    if visualize:
        rf.logger.beauty_print("Plot Optitrack skeleton motion", type="module")
        plot_skeleton_motion_interactive(motion)
    return motion


def motion_retargeting(retarget_cfg, source_motion, visualize=False):
    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(retarget_cfg["source_tpose"])
    if visualize:
        rf.logger.beauty_print("Plot Optitrack T-pose", type="module")
        plot_skeleton_state(source_tpose)

    target_tpose = SkeletonState.from_file(retarget_cfg["target_tpose"])
    if visualize:
        rf.logger.beauty_print("Plot Walker T-pose", type="module")
        plot_skeleton_state(target_tpose, verbose=True)

    # parse data from retarget config
    rotation_to_target_skeleton = torch.tensor(retarget_cfg["rotation"])

    # run retargeting
    # target_motion = source_motion.retarget_to_by_tpose(
    target_motion = source_motion.retarget_to_hotu_qbhand_by_tpose(
        joint_mapping=retarget_cfg["joint_mapping"],
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=retarget_cfg["scale"]
    )

    # state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, target_motion.rotation[0],
    #                                                          target_motion.root_translation[0], is_local=True)
    # plot_skeleton_state(state, verbose=True)
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

    # move the human to the origin
    # avg_root_translation = root_translation.mean(axis=0)
    # root_translation[1:] -= avg_root_translation

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
                                                                    root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # need to convert some joints from 3D to 1D (e.g. elbows and knees)
    # target_motion = _project_joints(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation

    # Set the human foot on the ground
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_cfg["root_height_offset"]
    root_translation[:, 2] += root_height_offset

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation,
                                                                    root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(retarget_cfg["target_motion_path"])

    if visualize:
        # visualize retargeted motion
        rf.logger.beauty_print("Plot Walker skeleton motion", type="module")
        plot_skeleton_motion_interactive(target_motion, verbose=False)

    dof_states = _run_sim(target_motion)
    dof_states = np.array(dof_states.cpu().numpy())
    np.save(retarget_cfg["target_dof_states_path"], dof_states)
    rf.logger.beauty_print(f"Saved Walker dof_states to {retarget_cfg['target_motion_path']}", type="module")


def npy_from_fbx(fbx_file):
    """
    This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
    Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
      - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
      - target_motion_path: path to save the retargeted motion to
      - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
      - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
      - joint_mapping: mapping of joint names from source to target
      - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
      - scale: scale offset from source to target skeleton
    """

    rf.logger.beauty_print(f"Processing {fbx_file}", type="module")

    rofunc_path = rf.oslab.get_rofunc_path()
    config = {
        "target_motion_path": fbx_file.replace('_optitrack.fbx', '_optitrack2walker.npy'),
        "target_dof_states_path": fbx_file.replace('_optitrack.fbx', '_optitrack2walker_dof_states.npy'),
        "source_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/source_optitrack_w_gloves_tpose.npy"),
        # "target_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/target_hotu_humanoid_w_qbhand_tpose.npy"),
        "target_tpose": os.path.join(rofunc_path, args.target_tpose),
        "joint_mapping": {  # Left: Optitrack, Right: MJCF
            # hotu_humanoid.xml
            "Skeleton_Hips": "base_link",
            "Skeleton_LeftUpLeg": "left_leg_l1",
            "Skeleton_LeftLeg": "left_leg_l4",
            "Skeleton_LeftFoot": "left_leg_l6",
            "Skeleton_RightUpLeg": "right_leg_l1",
            "Skeleton_RightLeg": "right_leg_l4",
            "Skeleton_RightFoot": "right_leg_l6",
            "Skeleton_Spine1": "torso",
            "Skeleton_Neck": "head_l1",
            # "Skeleton_LeftShoulder": "left_shoulder_pitch_link",
            "Skeleton_LeftArm": "left_limb_l1",
            "Skeleton_LeftForeArm": "left_limb_l4",
            "Skeleton_LeftHand": "left_limb_l7",
            # "Skeleton_RightShoulder": "right_shoulder_pitch_link",
            "Skeleton_RightArm": "right_limb_l1",
            "Skeleton_RightForeArm": "right_limb_l4",
            "Skeleton_RightHand": "right_limb_l7",
            # extra mapping for hotu_humanoid_w_qbhand.xml
            "Skeleton_LeftHandThumb1": "left_thumb_l1",
            "Skeleton_LeftHandThumb2": "left_thumb_l2",
            # "Skeleton_LeftHandThumb3": "left_qbhand_thumb_distal_link",
            "Skeleton_LeftHandIndex1": "left_index_l1",
            "Skeleton_LeftHandIndex2": "left_index_l2",
            # "Skeleton_LeftHandIndex3": "left_qbhand_index_distal_link",
            "Skeleton_LeftHandMiddle1": "left_middle_l1",
            "Skeleton_LeftHandMiddle2": "left_middle_l2",
            # "Skeleton_LeftHandMiddle3": "left_qbhand_middle_distal_link",
            "Skeleton_LeftHandRing1": "left_ring_l1",
            "Skeleton_LeftHandRing2": "left_ring_l2",
            # "Skeleton_LeftHandRing3": "left_qbhand_ring_distal_link",
            "Skeleton_LeftHandPinky1": "left_pinky_l1",
            "Skeleton_LeftHandPinky2": "left_pinky_l2",
            # "Skeleton_LeftHandPinky3": "left_qbhand_little_distal_link",
            "Skeleton_RightHandThumb1": "right_thumb_l1",
            "Skeleton_RightHandThumb2": "right_thumb_l2",
            # "Skeleton_RightHandThumb3": "right_qbhand_thumb_distal_link",
            "Skeleton_RightHandIndex1": "right_index_l1",
            "Skeleton_RightHandIndex2": "right_index_l2",
            # "Skeleton_RightHandIndex3": "right_qbhand_index_distal_link",
            "Skeleton_RightHandMiddle1": "right_middle_l1",
            "Skeleton_RightHandMiddle2": "right_middle_l2",
            # "Skeleton_RightHandMiddle3": "right_qbhand_middle_distal_link",
            "Skeleton_RightHandRing1": "right_ring_l1",
            "Skeleton_RightHandRing2": "right_ring_l2",
            # "Skeleton_RightHandRing3": "right_qbhand_ring_distal_link",
            "Skeleton_RightHandPinky1": "right_pinky_l1",
            "Skeleton_RightHandPinky2": "right_pinky_l2",
            # "Skeleton_RightHandPinky3": "right_qbhand_little_distal_link",
        },
        # "rotation": [0.707, 0, 0, 0.707], # xyzw
        "rotation": [0.5, 0.5, 0.5, 0.5],
        "scale": 0.001,  # Export millimeter to meter
        "root_height_offset": 0.0,
        "trim_frame_beg": 0,
        "trim_frame_end": -1
    }

    source_motion = motion_from_fbx(fbx_file, root_joint="Skeleton_Hips", fps=120, visualize=False)
    motion_retargeting(config, source_motion, visualize=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx_dir", type=str, default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/20240509")
    # parser.add_argument("--fbx_dir", type=str, default=None)
    # parser.add_argument("--fbx_file", type=str,
    #                     default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/test_data_05_optitrack.fbx")
    parser.add_argument("--fbx_file", type=str,
                        default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/20240509/Waving hand_Take 2024-05-09 04.20.29 PM_optitrack.fbx")
    parser.add_argument("--parallel", action="store_true")
    # Available asset:
    #                   1. mjcf/amp_humanoid_spoon_pan_fixed.xml
    #                   2. mjcf/amp_humanoid_sword_shield.xml
    #                   3. mjcf/hotu/hotu_humanoid.xml
    #                   4. mjcf/hotu_humanoid_w_qbhand_no_virtual.xml
    #                   5. mjcf/hotu/hotu_humanoid_w_qbhand_full.xml
    parser.add_argument("--humanoid_asset", type=str, default="mjcf/walker/walker.xml")
    parser.add_argument("--target_tpose", type=str, default="utils/datalab/poselib/data/target_walker_tpose.npy")
    args = parser.parse_args()

    rofunc_path = rf.oslab.get_rofunc_path()

    if args.fbx_dir is not None:
        fbx_dir = args.fbx_dir
        fbx_files = rf.oslab.list_absl_path(fbx_dir, suffix='.fbx')
    elif args.fbx_file is not None:
        fbx_files = [args.fbx_file]
    else:
        raise ValueError("Please provide a valid fbx_dir or fbx_file.")
    # fbx_dir = os.path.join(rofunc_path, "../examples/data/hotu")
    # fbx_dir = "/home/ubuntu/Data/2023_11_15_HED/has_gloves"
    # fbx_files = rf.oslab.list_absl_path(fbx_dir, suffix='.fbx')
    # fbx_files = ["/home/ubuntu/Data/2023_11_15_HED/has_gloves/New Session-009.fbx"]
    # fbx_files = [os.path.join(rofunc_path, "../examples/data/hotu/test_data_01_xsens.fbx")]

    if args.parallel:
        pool = multiprocessing.Pool()
        pool.map(npy_from_fbx, fbx_files)
    else:
        for fbx_file in fbx_files:
            # if os.path.exists(fbx_file.replace('_optitrack.fbx', '_optitrack2walker_dof_states.npy')):
            #     continue
            npy_from_fbx(fbx_file)
