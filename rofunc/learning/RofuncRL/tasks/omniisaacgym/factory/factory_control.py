# Copyright (c) 2018-2022, NVIDIA Corporation
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

"""Factory: control module.

Imported by base, environment, and task classes. Not directly executed.
"""

import math

import omni.isaac.core.utils.torch as torch_utils
import torch


def compute_dof_pos_target(cfg_ctrl,
                           arm_dof_pos,
                           fingertip_midpoint_pos,
                           fingertip_midpoint_quat,
                           jacobian,
                           ctrl_target_fingertip_midpoint_pos,
                           ctrl_target_fingertip_midpoint_quat,
                           ctrl_target_gripper_dof_pos,
                           device):
    """Compute Franka DOF position target to move fingertips towards target pose."""

    ctrl_target_dof_pos = torch.zeros((cfg_ctrl['num_envs'], 9), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type=cfg_ctrl['jacobian_type'],
        rot_error_type='axis_angle')

    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)
    delta_arm_dof_pos = _get_delta_dof_pos(delta_pose=delta_fingertip_pose,
                                           ik_method=cfg_ctrl['ik_method'],
                                           jacobian=jacobian,
                                           device=device)

    ctrl_target_dof_pos[:, 0:7] = arm_dof_pos + delta_arm_dof_pos
    ctrl_target_dof_pos[:, 7:9] = ctrl_target_gripper_dof_pos  # gripper finger joints

    return ctrl_target_dof_pos


def compute_dof_torque(cfg_ctrl,
                       dof_pos,
                       dof_vel,
                       fingertip_midpoint_pos,
                       fingertip_midpoint_quat,
                       fingertip_midpoint_linvel,
                       fingertip_midpoint_angvel,
                       left_finger_force,
                       right_finger_force,
                       jacobian,
                       arm_mass_matrix,
                       ctrl_target_gripper_dof_pos,
                       ctrl_target_fingertip_midpoint_pos,
                       ctrl_target_fingertip_midpoint_quat,
                       ctrl_target_fingertip_contact_wrench,
                       device):
    """Compute Franka DOF torque to move fingertips towards target pose."""
    # References:
    # 1) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # 2) Modern Robotics

    dof_torque = torch.zeros((cfg_ctrl['num_envs'], 9), device=device)

    if cfg_ctrl['gain_space'] == 'joint':
        pos_error, axis_angle_error = get_pose_error(
            fingertip_midpoint_pos=fingertip_midpoint_pos,
            fingertip_midpoint_quat=fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            jacobian_type=cfg_ctrl['jacobian_type'],
            rot_error_type='axis_angle')
        delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

        # Set tau = k_p * joint_pos_error - k_d * joint_vel_error (ETH eq. 3.72)
        delta_arm_dof_pos = _get_delta_dof_pos(delta_pose=delta_fingertip_pose,
                                               ik_method=cfg_ctrl['ik_method'],
                                               jacobian=jacobian,
                                               device=device)
        dof_torque[:, 0:7] = cfg_ctrl['joint_prop_gains'] * delta_arm_dof_pos + \
                             cfg_ctrl['joint_deriv_gains'] * (0.0 - dof_vel[:, 0:7])

        if cfg_ctrl['do_inertial_comp']:
            # Set tau = M * tau, where M is the joint-space mass matrix
            arm_mass_matrix_joint = arm_mass_matrix
            dof_torque[:, 0:7] = (arm_mass_matrix_joint @ dof_torque[:, 0:7].unsqueeze(-1)).squeeze(-1)

    elif cfg_ctrl['gain_space'] == 'task':
        task_wrench = torch.zeros((cfg_ctrl['num_envs'], 6), device=device)

        if cfg_ctrl['do_motion_ctrl']:
            pos_error, axis_angle_error = get_pose_error(
                fingertip_midpoint_pos=fingertip_midpoint_pos,
                fingertip_midpoint_quat=fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                jacobian_type=cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')
            delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

            # Set tau = k_p * task_pos_error - k_d * task_vel_error (building towards eq. 3.96-3.98)
            task_wrench_motion = _apply_task_space_gains(delta_fingertip_pose=delta_fingertip_pose,
                                                         fingertip_midpoint_linvel=fingertip_midpoint_linvel,
                                                         fingertip_midpoint_angvel=fingertip_midpoint_angvel,
                                                         task_prop_gains=cfg_ctrl['task_prop_gains'],
                                                         task_deriv_gains=cfg_ctrl['task_deriv_gains'])

            if cfg_ctrl['do_inertial_comp']:
                # Set tau = Lambda * tau, where Lambda is the task-space mass matrix
                jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
                arm_mass_matrix_task = torch.inverse(jacobian @ torch.inverse(
                    arm_mass_matrix) @ jacobian_T)  # ETH eq. 3.86; geometric Jacobian is assumed
                task_wrench_motion = (arm_mass_matrix_task @ task_wrench_motion.unsqueeze(-1)).squeeze(-1)

            task_wrench = task_wrench + torch.tensor(cfg_ctrl['motion_ctrl_axes'], device=device).unsqueeze(
                0) * task_wrench_motion

        if cfg_ctrl['do_force_ctrl']:
            # Set tau = tau + F_t, where F_t is the target contact wrench
            task_wrench_force = torch.zeros((cfg_ctrl['num_envs'], 6), device=device)
            task_wrench_force = task_wrench_force + ctrl_target_fingertip_contact_wrench  # open-loop force control (building towards ETH eq. 3.96-3.98)

            if cfg_ctrl['force_ctrl_method'] == 'closed':
                force_error, torque_error = _get_wrench_error(
                    left_finger_force=left_finger_force,
                    right_finger_force=right_finger_force,
                    ctrl_target_fingertip_contact_wrench=ctrl_target_fingertip_contact_wrench,
                    num_envs=cfg_ctrl['num_envs'],
                    device=device)

                # Set tau = tau + k_p * contact_wrench_error
                task_wrench_force = task_wrench_force + cfg_ctrl['wrench_prop_gains'] * torch.cat(
                    (force_error, torque_error), dim=1)  # part of Modern Robotics eq. 11.61

            task_wrench = task_wrench + torch.tensor(cfg_ctrl['force_ctrl_axes'], device=device).unsqueeze(
                0) * task_wrench_force

        # Set tau = J^T * tau, i.e., map tau into joint space as desired
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    dof_torque[:, 7:9] = cfg_ctrl['gripper_prop_gains'] * (ctrl_target_gripper_dof_pos - dof_pos[:, 7:9]) + \
                         cfg_ctrl['gripper_deriv_gains'] * (0.0 - dof_vel[:, 7:9])  # gripper finger joints
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)

    return dof_torque


def get_pose_error(fingertip_midpoint_pos,
                   fingertip_midpoint_quat,
                   ctrl_target_fingertip_midpoint_pos,
                   ctrl_target_fingertip_midpoint_quat,
                   jacobian_type,
                   rot_error_type):
    """Compute task-space error between target Franka fingertip pose and current pose."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    if jacobian_type == 'geometric':  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        # Compute quat error (i.e., difference quat)
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html
        fingertip_midpoint_quat_norm = torch_utils.quat_mul(fingertip_midpoint_quat,
                                                            torch_utils.quat_conjugate(fingertip_midpoint_quat))[:,
                                       0]  # scalar component
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    elif jacobian_type == 'analytic':  # See example 2.9.7; note use of J_a and difference of rotation vectors
        # Compute axis-angle error
        axis_angle_error = axis_angle_from_quat(ctrl_target_fingertip_midpoint_quat) \
                           - axis_angle_from_quat(fingertip_midpoint_quat)

    if rot_error_type == 'quat':
        return pos_error, quat_error
    elif rot_error_type == 'axis_angle':
        return pos_error, axis_angle_error


def _get_wrench_error(left_finger_force,
                      right_finger_force,
                      ctrl_target_fingertip_contact_wrench,
                      num_envs,
                      device):
    """Compute task-space error between target Franka fingertip contact wrench and current wrench."""

    fingertip_contact_wrench = torch.zeros((num_envs, 6), device=device)

    fingertip_contact_wrench[:, 0:3] = left_finger_force + right_finger_force  # net contact force on fingers
    # Cols 3 to 6 are all zeros, as we do not have enough information

    force_error = ctrl_target_fingertip_contact_wrench[:, 0:3] - (-fingertip_contact_wrench[:, 0:3])
    torque_error = ctrl_target_fingertip_contact_wrench[:, 3:6] - (-fingertip_contact_wrench[:, 3:6])

    return force_error, torque_error


def _get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Get delta Franka DOF position from delta pose using specified IK method."""
    # References:
    # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    # 2) https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf (p. 47)

    if ik_method == 'pinv':  # Jacobian pseudoinverse
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == 'trans':  # Jacobian transpose
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
        lambda_val = 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val ** 2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == 'svd':  # adaptive SVD
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1. / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U,
                                                                                                                  dim0=1,
                                                                                                                  dim1=2)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(delta_fingertip_pose,
                            fingertip_midpoint_linvel,
                            fingertip_midpoint_angvel,
                            task_prop_gains,
                            task_deriv_gains):
    """Interpret PD gains as task-space gains. Apply to task-space error."""

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to lin error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + \
                          task_deriv_gains[:, 0:3] * (0.0 - fingertip_midpoint_linvel)

    # Apply gains to rot error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + \
                          task_deriv_gains[:, 3:6] * (0.0 - fingertip_midpoint_angvel)

    return task_wrench


def get_analytic_jacobian(fingertip_quat, fingertip_jacobian, num_envs, device):
    """Convert geometric Jacobian to analytic Jacobian."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # NOTE: Gym returns world-space geometric Jacobians by default

    batch = num_envs

    # Overview:
    # x = [x_p; x_r]
    # From eq. 2.189 and 2.192, x_dot = J_a @ q_dot = (E_inv @ J_g) @ q_dot
    # From eq. 2.191, E = block(E_p, E_r); thus, E_inv = block(E_p_inv, E_r_inv)
    # Eq. 2.12 gives an expression for E_p_inv
    # Eq. 2.107 gives an expression for E_r_inv

    # Compute E_inv_top (i.e., [E_p_inv, 0])
    I = torch.eye(3, device=device)
    E_p_inv = I.repeat((batch, 1)).reshape(batch, 3, 3)
    E_inv_top = torch.cat((E_p_inv, torch.zeros((batch, 3, 3), device=device)), dim=2)

    # Compute E_inv_bottom (i.e., [0, E_r_inv])
    fingertip_axis_angle = axis_angle_from_quat(fingertip_quat)
    fingertip_axis_angle_cross = get_skew_symm_matrix(fingertip_axis_angle, device=device)
    fingertip_angle = torch.linalg.vector_norm(fingertip_axis_angle, dim=1)
    factor_1 = 1 / (fingertip_angle ** 2)
    factor_2 = 1 - fingertip_angle * 0.5 * torch.sin(fingertip_angle) / (1 - torch.cos(fingertip_angle))
    factor_3 = factor_1 * factor_2
    E_r_inv = I \
              - 1 * 0.5 * fingertip_axis_angle_cross \
              + (fingertip_axis_angle_cross @ fingertip_axis_angle_cross) * factor_3.unsqueeze(-1).repeat(
        (1, 3 * 3)).reshape((batch, 3, 3))
    E_inv_bottom = torch.cat((torch.zeros((batch, 3, 3), device=device), E_r_inv), dim=2)

    E_inv = torch.cat((E_inv_top.reshape((batch, 3 * 6)), E_inv_bottom.reshape((batch, 3 * 6))), dim=1).reshape(
        (batch, 6, 6))

    J_a = E_inv @ fingertip_jacobian

    return J_a


def get_skew_symm_matrix(vec, device):
    """Convert vector to skew-symmetric matrix."""
    # Reference: https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication

    batch = vec.shape[0]
    I = torch.eye(3, device=device)
    skew_symm = torch.transpose(torch.cross(vec.repeat((1, 3)).reshape((batch * 3, 3)),
                                            I.repeat((batch, 1)))
                                .reshape(batch, 3, 3),
                                dim0=1,
                                dim1=2)

    return skew_symm


def translate_along_local_z(pos, quat, offset, device):
    """Translate global body position along local Z-axis and express in global coordinates."""

    num_vecs = pos.shape[0]
    offset_vec = offset * torch.tensor([0.0, 0.0, 1.0], device=device).repeat((num_vecs, 1))
    _, translated_pos = torch_utils.tf_combine(q1=quat,
                                               t1=pos,
                                               q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
                                                   (num_vecs, 1)),
                                               t2=offset_vec)

    return translated_pos


def axis_angle_from_euler(euler):
    """Convert tensor of Euler angles to tensor of axis-angles."""

    quat = torch_utils.quat_from_euler_xyz(roll=euler[:, 0], pitch=euler[:, 1], yaw=euler[:, 2])
    quat = quat * torch.sign(quat[:, 0]).unsqueeze(-1)  # smaller rotation 
    axis_angle = axis_angle_from_quat(quat)

    return axis_angle


def axis_angle_from_quat(quat, eps=1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference: https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    mag = torch.linalg.norm(quat[:, 1:4], dim=1)
    half_angle = torch.atan2(mag, quat[:, 0])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(torch.abs(angle) > eps,
                                            torch.sin(half_angle) / angle,
                                            1 / 2 - angle ** 2.0 / 48)
    axis_angle = quat[:, 1:4] / sin_half_angle_over_angle.unsqueeze(-1)

    return axis_angle


def axis_angle_from_quat_naive(quat):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference: https://en.wikipedia.org/wiki/quats_and_spatial_rotation#Recovering_the_axis-angle_representation
    # NOTE: Susceptible to undesirable behavior due to divide-by-zero

    mag = torch.linalg.vector_norm(quat[:, 1:4], dim=1)  # zero when quat = [1, 0, 0, 0]
    axis = quat[:, 1:4] / mag.unsqueeze(-1)
    angle = 2.0 * torch.atan2(mag, quat[:, 0])
    axis_angle = axis * angle.unsqueeze(-1)

    return axis_angle


def get_rand_quat(num_quats, device):
    """Generate tensor of random quaternions."""
    # Reference: http://planning.cs.uiuc.edu/node198.html

    u = torch.rand((num_quats, 3), device=device)
    quat = torch.zeros((num_quats, 4), device=device)
    quat[:, 0] = torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2])
    quat[:, 1] = torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1])
    quat[:, 2] = torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1])
    quat[:, 3] = torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2])

    return quat


def get_nonrand_quat(num_quats, rot_perturbation, device):
    """Generate tensor of non-random quaternions by composing random Euler rotations."""

    quat = torch_utils.quat_from_euler_xyz(
        torch.rand((num_quats, 1), device=device).squeeze() * rot_perturbation * 2.0 - rot_perturbation,
        torch.rand((num_quats, 1), device=device).squeeze() * rot_perturbation * 2.0 - rot_perturbation,
        torch.rand((num_quats, 1), device=device).squeeze() * rot_perturbation * 2.0 - rot_perturbation)

    return quat
