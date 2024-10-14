"""
Visualize robots and objects
============================================================

This example shows how to visualize robots and objects in the Isaac Gym simulator in an interactive viewer.
"""

import isaacgym
import torch
from isaacgym import gymapi
import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion


def hotu_random():
    file = "../examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2hotu.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {"right_hand": gymapi.AXIS_ALL, "left_hand": gymapi.AXIS_ALL,
                  "right_foot": gymapi.AXIS_ROTATION, "left_foot": gymapi.AXIS_ROTATION,
                  "torso": gymapi.AXIS_ALL, "pelvis": gymapi.AXIS_ALL, "head": gymapi.AXIS_ROTATION,
                  "right_upper_arm": gymapi.AXIS_ROTATION, "left_upper_arm": gymapi.AXIS_ROTATION,
                  "right_lower_arm": gymapi.AXIS_ROTATION, "left_lower_arm": gymapi.AXIS_ROTATION,
                  "right_thigh": gymapi.AXIS_ROTATION, "left_thigh": gymapi.AXIS_ROTATION,
                  "right_shin": gymapi.AXIS_ALL, "left_shin": gymapi.AXIS_ALL,
                  }
    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links]

    hand_links = {"left_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION}
    hand_ids = [motion.skeleton_tree._node_indices[link] for link in hand_links]
    all_links = list(body_links.keys()) + list(hand_links.keys())
    all_ids = body_ids + hand_ids
    all_types = list(body_links.values()) + list(hand_links.values())
    # all_links = list(body_links.keys())
    # all_ids = body_ids
    # all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    motion_rb_states_pos[:, :, 2] -= 0.1
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("Humanoid")
    Humanoidsim = rf.sim.HumanoidSim(args)
    dof_states = Humanoidsim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        attr_types=all_types,
        update_freq=0.001,
        # root_state=motion_root_states,
        verbose=False,
        index_list=[1200, 1200, 1200, 1200, 1200, 600, 1250, 1330, 1820, 1970, 2400, 2700, 3000, 3200, 1700, 2100],
        recursive_play=True
    )


def h1_random():
    file = "../examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2h1.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {"torso_link": gymapi.AXIS_ALL,
                  "right_elbow_link": gymapi.AXIS_ROTATION, "left_elbow_link": gymapi.AXIS_ROTATION,
                  "right_hand": gymapi.AXIS_ROTATION, "left_hand": gymapi.AXIS_ROTATION,
                  "pelvis": gymapi.AXIS_ALL,
                  "left_hip_pitch_link": gymapi.AXIS_ROTATION, "right_hip_pitch_link": gymapi.AXIS_ROTATION,
                  "left_shoulder_yaw_link": gymapi.AXIS_ROTATION, "right_shoulder_yaw_link": gymapi.AXIS_ROTATION,
                  "left_knee_link": gymapi.AXIS_ALL, "right_knee_link": gymapi.AXIS_ALL,
                  "left_ankle_link": gymapi.AXIS_ROTATION, "right_ankle_link": gymapi.AXIS_ROTATION}
    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links]
    hand_links = {"left_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION}
    hand_ids = [motion.skeleton_tree._node_indices[link] for link in hand_links]

    all_links = list(body_links.keys()) + list(hand_links.keys())
    all_ids = body_ids + hand_ids
    all_types = list(body_links.values()) + list(hand_links.values())
    # all_links = list(body_links.keys())
    # all_ids = body_ids
    # all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_rot[:, hand_ids] = quat_mul(
    #     torch.tensor([0, 0.707, 0, 0.707]),
    #     motion_rb_states_rot[:, hand_ids]
    # )

    motion_rb_states_pos[:, :, 2] -= 0.06
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("UnitreeH1")
    UnitreeH1sim = rf.sim.RobotSim(args)
    dof_states = UnitreeH1sim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        update_freq=0.001,
        # root_state=motion_root_states,
        attr_types=all_types,
        verbose=False,
        index_list=[1200, 1200, 1200, 1200, 1200, 600, 1250, 1330, 1820, 1970, 2400, 2700, 3000, 3200, 1700, 2100],
        recursive_play=True
    )


def zju_random():
    file = "../examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2zju.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {"pelvis": gymapi.AXIS_ROTATION,
                  "FOREARM_R": gymapi.AXIS_ROTATION, "FOREARM_L": gymapi.AXIS_ROTATION,
                  "HAND_R": gymapi.AXIS_ALL, "HAND_L": gymapi.AXIS_ALL,
                  "SACRUM": gymapi.AXIS_ROTATION,
                  "THIGH_L": gymapi.AXIS_ROTATION, "THIGH_R": gymapi.AXIS_ROTATION,
                  "UPPERARM_L": gymapi.AXIS_ROTATION, "UPPERARM_R": gymapi.AXIS_ROTATION,
                  "SHANK_L": gymapi.AXIS_ALL, "SHANK_R": gymapi.AXIS_ALL,
                  "FOOT_L": gymapi.AXIS_ROTATION, "FOOT_R": gymapi.AXIS_ROTATION}
    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links.keys()]
    hand_links = {"left_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION}
    hand_ids = [motion.skeleton_tree._node_indices[link] for link in hand_links]

    all_links = list(body_links.keys()) + list(hand_links.keys())
    all_ids = body_ids + hand_ids
    all_types = list(body_links.values()) + list(hand_links.values())
    # all_links = list(body_links.keys())
    # all_ids = body_ids
    # all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_rot[:, hand_ids] = quat_mul(
    #     torch.tensor([0, 0.707, 0, 0.707]),
    #     motion_rb_states_rot[:, hand_ids]
    # )

    motion_rb_states_pos[:, :, 2] -= 0.06
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("ZJUHumanoid")
    ZJUHumanoidsim = rf.sim.RobotSim(args)
    dof_states = ZJUHumanoidsim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        update_freq=0.001,
        # root_state=motion_root_states,
        attr_types=all_types,
        verbose=False,
        index_list=[1200, 1200, 1200, 1200, 1200, 600, 1250, 1330, 1820, 1970, 2400, 2700, 3000, 3200, 1700, 2100],
        recursive_play=True
    )


def walker_random():
    file = "../examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2walker.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {"torso": gymapi.AXIS_ROTATION,
                  "right_limb_l4": gymapi.AXIS_ROTATION, "left_limb_l4": gymapi.AXIS_ROTATION,
                  "right_limb_l7": gymapi.AXIS_ALL, "left_limb_l7": gymapi.AXIS_ALL,
                  "base_link": gymapi.AXIS_ROTATION,
                  "left_leg_l1": gymapi.AXIS_ROTATION, "right_leg_l1": gymapi.AXIS_ROTATION,
                  "left_limb_l1": gymapi.AXIS_ROTATION, "right_limb_l1": gymapi.AXIS_ROTATION,
                  "left_leg_l4": gymapi.AXIS_ALL, "right_leg_l4": gymapi.AXIS_ALL,
                  "left_leg_l6": gymapi.AXIS_ALL, "right_leg_l6": gymapi.AXIS_ALL,
                  "head_l2": gymapi.AXIS_ROTATION, }

    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links.keys()]

    all_links = list(body_links.keys())
    all_ids = body_ids
    all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_rot[:, hand_ids] = quat_mul(
    #     torch.tensor([0, 0.707, 0, 0.707]),
    #     motion_rb_states_rot[:, hand_ids]
    # )

    motion_rb_states_pos[:, :, 2] -= 0.06
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
        # root_state=motion_root_states,
        attr_types=all_types,
        verbose=False,
        index_list=[1200, 1200, 1200, 1200, 1200, 600, 1250, 1330, 1820, 1970, 2400, 2700, 3000, 3200, 1700, 2100],
        recursive_play=True
    )


def bruce_random():
    file = "../examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2bruce.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {
        "hand_l": gymapi.AXIS_TRANSLATION, "hand_r": gymapi.AXIS_TRANSLATION,
        "elbow_pitch_link_l": gymapi.AXIS_ROTATION, "elbow_pitch_link_r": gymapi.AXIS_ROTATION,
        "pelvis": gymapi.AXIS_ROTATION,
        "hip_pitch_link_l": gymapi.AXIS_ALL, "hip_pitch_link_r": gymapi.AXIS_ALL,
        # "shoulder_roll_link_l": gymapi.AXIS_ROTATION, "shoulder_roll_link_r": gymapi.AXIS_ROTATION,
        "knee_pitch_link_l": gymapi.AXIS_ROTATION, "knee_pitch_link_r": gymapi.AXIS_ROTATION,
        "ankle_pitch_link_l": gymapi.AXIS_ROTATION, "ankle_pitch_link_r": gymapi.AXIS_ROTATION
    }

    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links.keys()]

    all_links = list(body_links.keys())
    all_ids = body_ids
    all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_rot[:, hand_ids] = quat_mul(
    #     torch.tensor([0, 0.707, 0, 0.707]),
    #     motion_rb_states_rot[:, hand_ids]
    # )

    motion_rb_states_pos[:, :, 2] -= 0.27
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("Bruce")
    Brucesim = rf.sim.RobotSim(args)
    dof_states = Brucesim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        update_freq=0.001,
        # root_state=motion_root_states,
        attr_types=all_types,
        verbose=False,
        index_list=[1200, 1200, 1200, 1200, 1200, 600, 1250, 1330, 1790, 1970, 2400, 2700, 3000, 3200, 1700, 2100],
        recursive_play=True
    )


def curi_random():
    file = "../examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2curi.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {
        # "torso_base2": gymapi.AXIS_ROTATION, "root": gymapi.AXIS_ROTATION,
                  "head_link1": gymapi.AXIS_ROTATION,
                  "panda_right_link1": gymapi.AXIS_ROTATION, "panda_left_link1": gymapi.AXIS_ROTATION,
                  "panda_right_link4": gymapi.AXIS_ROTATION, "panda_left_link4": gymapi.AXIS_ROTATION,
                  "panda_right_link7": gymapi.AXIS_ROTATION, "panda_left_link7": gymapi.AXIS_ROTATION
                  }
    body_ids = [motion.skeleton_tree._node_indices[link] for link in body_links.keys()]
    hand_links = {"left_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "left_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_knuckle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_thumb_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_index_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_middle_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_ring_distal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_proximal_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_middle_link": gymapi.AXIS_TRANSLATION,
                  "right_qbhand_little_distal_link": gymapi.AXIS_TRANSLATION}
    hand_ids = [motion.skeleton_tree._node_indices[link] for link in hand_links]
    # all_links = list(body_links.keys()) + list(hand_links.keys())
    # all_ids = body_ids + hand_ids
    # all_types = list(body_links.values()) + list(hand_links.values())

    all_links = list(body_links.keys())
    all_ids = body_ids
    all_types = list(body_links.values())

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_rot[:, hand_ids] = quat_mul(
    #     torch.tensor([0, 0.707, 0, 0.707]),
    #     motion_rb_states_rot[:, hand_ids]
    # )

    # motion_rb_states_pos[:, :, 2] += 0.06
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("CURI")
    CURIsim = rf.sim.RobotSim(args)
    dof_states = CURIsim.run_traj_multi_rigid_bodies(
        traj=[motion_rb_states[:, id] for id in all_ids],
        attr_rbs=all_links,
        update_freq=0.001,
        # root_state=motion_root_states,
        attr_types=all_types,
        verbose=False,
        index_list=[1200, 1200, 1200, 1200, 1200, 600, 1250, 1330, 1820, 1970, 2400, 2700, 3000, 3200, 1700, 2100],
        recursive_play=True
    )


if __name__ == '__main__':
    curi_random()
