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


# # CURI
# args = rf.config.get_sim_config("CURI")
# CURIsim = rf.sim.CURISim(args)
# CURIsim.show(visual_obs_flag=False)

# walker
# args = rf.config.get_sim_config("Walker")
# walkersim = rf.sim.WalkerSim(args)
# walkersim.show()

# CURI-mini
# args = rf.config.get_sim_config("CURImini")
# CURIminisim = rf.sim.RobotSim(args)
# CURIminisim.show(visual_obs_flag=True)

# franka
# args = rf.config.get_sim_config("Franka")
# frankasim = rf.sim.FrankaSim(args)
# frankasim.show()

# baxter
# args = rf.config.get_sim_config("Baxter")
# baxtersim = rf.sim.RobotSim(args)
# baxtersim.show()

# sawyer
# args = rf.config.get_sim_config("Sawyer")
# sawyersim = rf.sim.RobotSim(args)
# sawyersim.show()

# gluon
# args = rf.config.get_sim_config("Gluon")
# Gluonsim = rf.sim.GluonSim(args)
# Gluonsim.show()

# # qbsofthand
# args = rf.config.get_sim_config("QbSoftHand")
# QbSoftHandsim = rf.sim.QbSoftHandSim(args)
# QbSoftHandsim.show()

# # HOTU
def hotu_random():
    file = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2hotu.npy"
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
    file = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack2h1.npy"
    motion = SkeletonMotion.from_file(file)
    body_links = {"torso_link": gymapi.AXIS_ROTATION,
                  "right_elbow_link": gymapi.AXIS_ROTATION, "left_elbow_link": gymapi.AXIS_ROTATION,
                  "right_hand": gymapi.AXIS_ALL, "left_hand": gymapi.AXIS_ALL,
                  "pelvis": gymapi.AXIS_ROTATION,
                  "left_hip_pitch_link": gymapi.AXIS_ROTATION, "right_hip_pitch_link": gymapi.AXIS_ROTATION,
                  "left_shoulder_yaw_link": gymapi.AXIS_ROTATION, "right_shoulder_yaw_link": gymapi.AXIS_ROTATION,
                  "left_knee_link": gymapi.AXIS_ALL, "right_knee_link": gymapi.AXIS_ALL,
                  "left_ankle_link": gymapi.AXIS_ALL, "right_ankle_link": gymapi.AXIS_ALL}
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

    motion_rb_states_pos[:, :, 2] += 0.06
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
        verbose=False
    )


if __name__ == '__main__':
    h1_random()
