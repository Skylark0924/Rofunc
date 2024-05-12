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

"""
Attention: Since the Autodesk FBX SDK just supports Python 3.7, this script should be run with Python 3.7.
"""

import isaacgym
import multiprocessing
import os
import sys
import numpy as np

import rofunc as rf
from rofunc.utils.datalab.poselib.poselib.core.rotation3d import *
from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from rofunc.utils.datalab.poselib.poselib.visualization.common import plot_skeleton_motion_interactive, \
    plot_skeleton_state

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_joints(motion):
    """
    For 1 DoF joints like elbow and knee, new elbow_q is abstracted from default 3 DoF elbow rotation along the
     joint rotation axis and project the rest rotations in other directions to shoulder and hip that have 3 DoF

    :param motion:
    :return:
    """
    right_thigh_id = motion.skeleton_tree._node_indices["hip_yaw_link_r"]
    right_shin_id = motion.skeleton_tree._node_indices["knee_pitch_link_r"]
    right_foot_id = motion.skeleton_tree._node_indices["ankle_pitch_link_r"]
    left_thigh_id = motion.skeleton_tree._node_indices["hip_yaw_link_l"]
    left_shin_id = motion.skeleton_tree._node_indices["knee_pitch_link_l"]
    left_foot_id = motion.skeleton_tree._node_indices["ankle_pitch_link_l"]

    device = motion.global_translation.device

    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]

    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]),
                                                                                  device=device,
                                                                                  dtype=torch.float32))

    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)

    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]

    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]),
                                                                                device=device, dtype=torch.float32))

    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)

    new_local_rotation = motion.local_rotation.clone()

    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation,
                                                                    motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion


def control_ik(j_eef, dpose):
    damping = 0.1
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = (torch.eye(6) * (damping ** 2))
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(1, 7)
    return u


def _decompose_spherical_joints(motion):
    """
    Decompose and represent 3 DoF spherical joint with three 1 DoF revolution joint

    :param motion:
    :return:
    """
    from isaacgym import gymapi, gymtorch, gymutil
    right_shoulder_pitch_id = motion.skeleton_tree._node_indices["right_shoulder_pitch_link"]
    right_shoulder_yaw_id = motion.skeleton_tree._node_indices["right_shoulder_yaw_link"]
    right_shoulder_roll_id = motion.skeleton_tree._node_indices["right_shoulder_roll_link"]
    left_shoulder_pitch_id = motion.skeleton_tree._node_indices["left_shoulder_pitch_link"]
    left_shoulder_yaw_id = motion.skeleton_tree._node_indices["left_shoulder_yaw_link"]
    left_shoulder_roll_id = motion.skeleton_tree._node_indices["left_shoulder_roll_link"]

    right_hip_yaw_id = motion.skeleton_tree._node_indices["right_hip_yaw_link"]
    right_hip_pitch_id = motion.skeleton_tree._node_indices["right_hip_pitch_link"]
    right_hip_roll_id = motion.skeleton_tree._node_indices["right_hip_roll_link"]
    left_hip_yaw_id = motion.skeleton_tree._node_indices["left_hip_yaw_link"]
    left_hip_pitch_id = motion.skeleton_tree._node_indices["left_hip_pitch_link"]
    left_hip_roll_id = motion.skeleton_tree._node_indices["left_hip_roll_link"]

    verbose = True

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    rofunc_path = rf.oslab.get_rofunc_path()
    asset_root = os.path.join(rofunc_path, "simulator/assets")

    camera_props = gymapi.CameraProperties()
    cam_pos = (3.0, 0.0, 2.0)
    cam_target = (0.0, 0.0, 0.0)
    camera_props.horizontal_fov = 75
    camera_props.width = 1920
    camera_props.height = 1080
    camera_props.use_collision_geometry = False
    viewer = gym.create_viewer(sim, camera_props)
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load humanoid asset
    asset_file = args.humanoid_asset
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.disable_gravity = False
    # asset_options.fix_base_link = True
    humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    pose = gymapi.Transform()
    spacing = 10
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    humanoid_handle = gym.create_actor(env, humanoid_asset, pose, "humanoid", 0, -1, 0)

    robot_dof_props = gym.get_asset_dof_properties(humanoid_asset)
    robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
    robot_dof_props["stiffness"][:] = 300
    robot_dof_props["damping"][:] = 30
    gym.set_actor_dof_properties(env, humanoid_handle, robot_dof_props)

    gym.prepare_sim(sim)
    _jacobian = gym.acquire_jacobian_tensor(sim, "humanoid")
    jacobian = gymtorch.wrap_tensor(_jacobian)  # [1, 90, 6, 97]

    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)  # [90, 13]

    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_pos = dof_states[:, 0].view(1, -1, 1)  # [1, 91, 1]

    axes_geoms = []
    sphere_geoms = []
    for i in range(rb_states.shape[0]):
        # Create helper geometry used for visualization
        # Create a wireframe axis
        axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

        axes_geoms.append(axes_geom)
        sphere_geoms.append(sphere_geom)

    motion_length = motion.local_rotation.shape[0]
    i = 0
    while not gym.query_viewer_has_closed(viewer):
        gym.clear_lines(viewer)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)

        motion_rb_states_pos = motion.global_translation[i]
        motion_rb_states_rot = motion.global_rotation[i]

        # rb_states[left_shoulder_roll_id, :3] = motion_rb_states_pos[left_shoulder_roll_id]
        # rb_states[left_shoulder_roll_id, 3:7] = motion_rb_states_rot[left_shoulder_roll_id]
        # rb_states[right_shoulder_roll_id, :3] = motion_rb_states_pos[right_shoulder_roll_id]
        # rb_states[right_shoulder_roll_id, 3:7] = motion_rb_states_rot[right_shoulder_roll_id]
        rb_states[:, :3] = motion_rb_states_pos[:]
        rb_states[:, 3:7] = motion_rb_states_rot[:]
        _rb_states = gymtorch.unwrap_tensor(rb_states)

        gym.set_rigid_body_state_tensor(sim, _rb_states)

        if verbose:
            # Draw axes and sphere at attractor location
            for j in range(rb_states.shape[0]):
                pose = gymapi.Transform()
                pose.p.x = motion_rb_states_pos[j, 0] + 1
                pose.p.y = motion_rb_states_pos[j, 1]
                pose.p.z = motion_rb_states_pos[j, 2]
                pose.r.w = motion_rb_states_rot[j, 3]
                pose.r.x = motion_rb_states_rot[j, 0]
                pose.r.y = motion_rb_states_rot[j, 1]
                pose.r.z = motion_rb_states_rot[j, 2]

                gymutil.draw_lines(axes_geoms[j], gym, viewer, env, pose)
                gymutil.draw_lines(sphere_geoms[j], gym, viewer, env, pose)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

        i += 1
        if i >= motion_length:
            i = 0

        # right_shoulder_start_pos = motion_rb_states_pos[right_shoulder_pitch_id, :3]
        # right_shoulder_start_rot = motion_rb_states_rot[right_shoulder_pitch_id, :4]
        # right_shoulder_end_pos = motion_rb_states_pos[right_shoulder_roll_id, :3]
        # right_shoulder_end_rot = motion_rb_states_rot[right_shoulder_roll_id, :4]

        # left_shoulder_start_pos = motion_rb_states_pos[left_shoulder_pitch_id, :3]
        # left_shoulder_start_rot = motion_rb_states_rot[left_shoulder_pitch_id, :4]
        # left_shoulder_end_pos = motion_rb_states_pos[left_shoulder_roll_id, :3]
        # left_shoulder_end_rot = motion_rb_states_rot[left_shoulder_roll_id, :4]

    # new_local_rotation = motion.local_rotation.clone()
    # new_local_rotation[..., right_shoulder_pitch_id, :] = right_shoulder_pitch_q
    # new_local_rotation[..., right_shoulder_yaw_id, :] = right_shoulder_yaw_q
    # new_local_rotation[..., right_shoulder_roll_id, :] = right_shoulder_roll_q
    # new_local_rotation[..., left_shoulder_pitch_id, :] = left_shoulder_pitch_q
    # new_local_rotation[..., left_shoulder_yaw_id, :] = left_shoulder_yaw_q
    # new_local_rotation[..., left_shoulder_roll_id, :] = left_shoulder_roll_q

    # new_local_rotation[..., right_hip_yaw_id, :] = right_hip_yaw_q
    # new_local_rotation[..., right_hip_pitch_id, :] = right_hip_pitch_q
    # new_local_rotation[..., right_hip_roll_id, :] = right_hip_roll_q
    # new_local_rotation[..., left_hip_yaw_id, :] = left_hip_yaw_q
    # new_local_rotation[..., left_hip_pitch_id, :] = left_hip_pitch_q
    # new_local_rotation[..., left_hip_roll_id, :] = left_hip_roll_q

    # new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation,
    #                                                                 motion.root_translation, is_local=True)
    # new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    # return new_motion


def _run_sim(motion):
    pelvis_id = motion.skeleton_tree._node_indices["pelvis"]
    right_shoulder_yaw_id = motion.skeleton_tree._node_indices["shoulder_pitch_link_r"]
    right_elbow_id = motion.skeleton_tree._node_indices["elbow_pitch_link_r"]
    left_shoulder_yaw_id = motion.skeleton_tree._node_indices["shoulder_pitch_link_l"]
    left_elbow_id = motion.skeleton_tree._node_indices["elbow_pitch_link_l"]
    right_hip_pitch_id = motion.skeleton_tree._node_indices["hip_pitch_link_r"]
    right_knee_id = motion.skeleton_tree._node_indices["knee_pitch_link_r"]
    right_ankle_id = motion.skeleton_tree._node_indices["ankle_pitch_link_r"]
    left_hip_pitch_id = motion.skeleton_tree._node_indices["hip_pitch_link_l"]
    left_knee_id = motion.skeleton_tree._node_indices["knee_pitch_link_l"]
    left_ankle_id = motion.skeleton_tree._node_indices["ankle_pitch_link_l"]

    motion_rb_states_pos = motion.global_translation
    motion_rb_states_rot = motion.global_rotation

    # motion_rb_states_pos[:, :, 2] += 1.0
    motion_rb_states = torch.cat([motion_rb_states_pos, motion_rb_states_rot], dim=-1)

    motion_root_pos = motion_rb_states_pos[:, 0]
    motion_root_rot = motion_rb_states_rot[:, 0]
    motion_root_vel = motion.global_root_velocity
    motion_root_ang_vel = motion.global_root_angular_velocity
    motion_root_states = torch.cat([motion_root_pos, motion_root_rot, motion_root_vel, motion_root_ang_vel], dim=-1)

    args = rf.config.get_sim_config("Bruce")
    Brucesim = rf.sim.RobotSim(args)
    Brucesim.run_traj_multi_rigid_bodies(
        traj=[
            motion_rb_states[:, right_elbow_id], motion_rb_states[:, left_elbow_id],
            motion_rb_states[:, pelvis_id],
            motion_rb_states[:, left_hip_pitch_id], motion_rb_states[:, right_hip_pitch_id],
            motion_rb_states[:, left_shoulder_yaw_id], motion_rb_states[:, right_shoulder_yaw_id],
            motion_rb_states[:, left_knee_id], motion_rb_states[:, right_knee_id],
            motion_rb_states[:, left_ankle_id], motion_rb_states[:, right_ankle_id]
        ],
        attr_rbs=["elbow_pitch_link_r", "elbow_pitch_link_l",
                  "pelvis",
                  "hip_pitch_link_l", "hip_pitch_link_r",
                  "shoulder_pitch_link_l", "shoulder_pitch_link_r",
                  "knee_pitch_link_l", "knee_pitch_link_r",
                  "ankle_pitch_link_l", "ankle_pitch_link_r"
                  ],
        update_freq=0.001,
        root_state=motion_root_states,
        key_bodies=["elbow_pitch_link_r", "elbow_pitch_link_l", "ankle_pitch_link_l", "ankle_pitch_link_r"],
        verbose=False
    )


def motion_from_fbx(fbx_file_path, root_joint, fps=60, visualize=True):
    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file_path,
        root_joint=root_joint,
        fps=fps
    )
    # visualize motion
    if visualize:
        rf.logger.beauty_print("Plot Xsens skeleton motion", type="module")
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
        rf.logger.beauty_print("Plot HOTU T-pose", type="module")
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

    # gym = gymapi.acquire_gym()
    # sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())
    # rofunc_path = rf.oslab.get_rofunc_path()
    # asset_root = os.path.join(rofunc_path, "simulator/assets")

    # # Load humanoid asset
    # asset_file = args.humanoid_asset
    # asset_options = gymapi.AssetOptions()
    # asset_options.angular_damping = 0.01
    # asset_options.max_angular_velocity = 100.0
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    # asset_options.disable_gravity = False
    # # asset_options.fix_base_link = True
    # humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    # asset_dof_dict = gym.get_asset_dof_dict(humanoid_asset)
    # asset_rigid_body_dict = gym.get_asset_rigid_body_dict(humanoid_asset)
    # asset_joint_dict = gym.get_asset_joint_dict(humanoid_asset)
    # target_motion.skeleton_tree.asset_dof_dict = asset_dof_dict
    # target_motion.skeleton_tree.asset_rigid_body_dict = asset_rigid_body_dict
    # target_motion.skeleton_tree.asset_joint_dict = asset_joint_dict

    # save retargeted motion
    # target_motion.to_file(retarget_cfg["target_motion_path"])

    if visualize:
        # visualize retargeted motion
        rf.logger.beauty_print("Plot Bruce skeleton motion", type="module")
        plot_skeleton_motion_interactive(target_motion, verbose=False)

        # state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, target_motion.rotation[0],
        #                                                          target_motion.root_translation[0], is_local=True)
        # plot_skeleton_state(state, verbose=True)

    # target_motion = _project_joints(target_motion)
    # plot_skeleton_motion_interactive(target_motion, verbose=False)
    _run_sim(target_motion)



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
        "target_motion_path": fbx_file.replace('_optitrack.fbx', '_optitrack2bruce.npy'),
        "source_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/source_optitrack_w_gloves_tpose.npy"),
        # "target_tpose": os.path.join(rofunc_path, "utils/datalab/poselib/data/target_hotu_humanoid_w_qbhand_tpose.npy"),
        "target_tpose": os.path.join(rofunc_path, args.target_tpose),
        "joint_mapping": {  # Left: Optitrack, Right: MJCF
            # hotu_humanoid.xml
            "Skeleton_Hips": "pelvis",
            # "Skeleton_LeftUpLeg": "left_hip_yaw_link",
            "Skeleton_LeftUpLeg": "hip_yaw_link_l",
            "Skeleton_LeftLeg": "knee_pitch_link_l",
            "Skeleton_LeftFoot": "ankle_pitch_link_l",
            # "Skeleton_RightUpLeg": "right_hip_yaw_link",
            "Skeleton_RightUpLeg": "hip_yaw_link_r",
            "Skeleton_RightLeg": "knee_pitch_link_r",
            "Skeleton_RightFoot": "ankle_pitch_link_r",
            # "Skeleton_Spine1": "torso_link",
            # "Skeleton_Neck": "head",
            "Skeleton_LeftArm": "shoulder_pitch_link_l",
            # "Skeleton_LeftArm": "left_shoulder_yaw_link",
            "Skeleton_LeftForeArm": "elbow_pitch_link_l",
            # "Skeleton_LeftHand": "left_hand",
            "Skeleton_RightArm": "shoulder_pitch_link_r",
            # "Skeleton_RightArm": "right_shoulder_yaw_link",
            "Skeleton_RightForeArm": "elbow_pitch_link_r",
            # "Skeleton_RightHand": "right_hand",
            # # extra mapping for hotu_humanoid_w_qbhand.xml
            # "Skeleton_LeftHandThumb1": "left_qbhand_thumb_knuckle_link",
            # "Skeleton_LeftHandThumb2": "left_qbhand_thumb_proximal_link",
            # "Skeleton_LeftHandThumb3": "left_qbhand_thumb_distal_link",
            # "Skeleton_LeftHandIndex1": "left_qbhand_index_proximal_link",
            # "Skeleton_LeftHandIndex2": "left_qbhand_index_middle_link",
            # "Skeleton_LeftHandIndex3": "left_qbhand_index_distal_link",
            # "Skeleton_LeftHandMiddle1": "left_qbhand_middle_proximal_link",
            # "Skeleton_LeftHandMiddle2": "left_qbhand_middle_middle_link",
            # "Skeleton_LeftHandMiddle3": "left_qbhand_middle_distal_link",
            # "Skeleton_LeftHandRing1": "left_qbhand_ring_proximal_link",
            # "Skeleton_LeftHandRing2": "left_qbhand_ring_middle_link",
            # "Skeleton_LeftHandRing3": "left_qbhand_ring_distal_link",
            # "Skeleton_LeftHandPinky1": "left_qbhand_little_proximal_link",
            # "Skeleton_LeftHandPinky2": "left_qbhand_little_middle_link",
            # "Skeleton_LeftHandPinky3": "left_qbhand_little_distal_link",
            # "Skeleton_RightHandThumb1": "right_qbhand_thumb_knuckle_link",
            # "Skeleton_RightHandThumb2": "right_qbhand_thumb_proximal_link",
            # "Skeleton_RightHandThumb3": "right_qbhand_thumb_distal_link",
            # "Skeleton_RightHandIndex1": "right_qbhand_index_proximal_link",
            # "Skeleton_RightHandIndex2": "right_qbhand_index_middle_link",
            # "Skeleton_RightHandIndex3": "right_qbhand_index_distal_link",
            # "Skeleton_RightHandMiddle1": "right_qbhand_middle_proximal_link",
            # "Skeleton_RightHandMiddle2": "right_qbhand_middle_middle_link",
            # "Skeleton_RightHandMiddle3": "right_qbhand_middle_distal_link",
            # "Skeleton_RightHandRing1": "right_qbhand_ring_proximal_link",
            # "Skeleton_RightHandRing2": "right_qbhand_ring_middle_link",
            # "Skeleton_RightHandRing3": "right_qbhand_ring_distal_link",
            # "Skeleton_RightHandPinky1": "right_qbhand_little_proximal_link",
            # "Skeleton_RightHandPinky2": "right_qbhand_little_middle_link",
            # "Skeleton_RightHandPinky3": "right_qbhand_little_distal_link",
        },
        # "rotation": [0.707, 0, 0, 0.707], xyzw
        "rotation": [0.5, 0.5, 0.5, 0.5],
        "scale": 0.001,  # Export millimeter to meter
        "root_height_offset": 0.0,
        "trim_frame_beg": 0,
        "trim_frame_end": -1
    }

    source_motion = motion_from_fbx(fbx_file, root_joint="Skeleton_Hips", fps=120, visualize=False)
    # config["target_motion_path"] = fbx_file.replace('.fbx', '_amp.npy')
    motion_retargeting(config, source_motion, visualize=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--fbx_dir", type=str, default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2")
    parser.add_argument("--fbx_dir", type=str, default=None)
    # parser.add_argument("--fbx_file", type=str,
    #                     default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/test_data_04_optitrack.fbx")
    parser.add_argument("--fbx_file", type=str,
                        default="/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/examples/data/hotu2/20240509/Ramdom (good)_Take 2024-05-09 04.49.16 PM_optitrack.fbx")
    parser.add_argument("--parallel", action="store_true")
    # Available asset:
    #                   1. mjcf/amp_humanoid_spoon_pan_fixed.xml
    #                   2. mjcf/amp_humanoid_sword_shield.xml
    #                   3. mjcf/hotu_humanoid.xml
    #                   4. mjcf/hotu_humanoid_w_qbhand_no_virtual.xml
    #                   5. mjcf/hotu_humanoid_w_qbhand_full.xml
    parser.add_argument("--humanoid_asset", type=str, default="mjcf/bruce/bruce.xml")
    parser.add_argument("--target_tpose", type=str,
                        default="utils/datalab/poselib/data/target_bruce_tpose.npy")
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
            npy_from_fbx(fbx_file)
