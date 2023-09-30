"""
CURI controllers
================

This example shows how to use basic controllers of the CURI robot.
"""

# TODO: Reformat

import math
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *

import rofunc as rf


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 2
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 2], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    z = torch.zeros_like(w)
    y = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(CURIsim.num_envs, 7)
    return u


damping = 0.05
args = gymutil.parse_arguments()
args.use_gpu_pipeline = True
controller = 'ik'
# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# CURI
CURIsim = rf.sim.CURISim(args, device=device)
DOF = CURIsim.robot_dof

# create table asset
table_dims = gymapi.Vec3(0.6, 0.5, 1.5)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = CURIsim.gym.create_box(CURIsim.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = CURIsim.gym.create_box(CURIsim.sim, box_size, box_size, box_size, asset_options)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(1, 0.5 * table_dims.y, 0.0)

box_poses = []
for i in range(CURIsim.num_envs):
    box_pose = gymapi.Transform()
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.z = table_pose.p.z + np.random.uniform(-0.3, -0.3)
    box_pose.p.y = table_dims.y + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.random.uniform(-math.pi, math.pi))
    box_poses.append(box_pose)

# add table and box to scene
table_handles, table_idxs = CURIsim.add_object(table_asset, table_pose, "table")
box_handles, box_idxs = CURIsim.add_object(box_asset, box_poses, "box")

# CURIsim.show()

# get link index of panda hand, which we will use as end effector
curi_link_dict = CURIsim.gym.get_asset_rigid_body_dict(CURIsim.robot_asset)
curi_hand_index = curi_link_dict["panda_left_hand"]

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
CURIsim.gym.prepare_sim(CURIsim.sim)

init_pos_list = []
init_rot_list = []
hand_idxs = []

for i in range(CURIsim.num_envs):
    # get inital hand pose
    hand_handle = CURIsim.gym.find_actor_rigid_body_handle(CURIsim.envs[i], CURIsim.robot_handles[i], "panda_left_hand")
    hand_pose = CURIsim.gym.get_rigid_transform(CURIsim.envs[i], hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = CURIsim.gym.find_actor_rigid_body_index(CURIsim.envs[i], CURIsim.robot_handles[i], "panda_left_hand",
                                                       gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(CURIsim.num_envs, 3).to(CURIsim.device)
init_rot = torch.Tensor(init_rot_list).view(CURIsim.num_envs, 4).to(CURIsim.device)

# hand orientation for grasping
down_q = torch.stack(CURIsim.num_envs * [torch.tensor([0.707, 0, 0, 0.707])]).to(CURIsim.device).view(
    (CURIsim.num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(CURIsim.num_envs * [corner_coord]).to(CURIsim.device)

# downard axis
down_dir = torch.Tensor([0, -1, 0]).to(CURIsim.device).view(1, 3)

_jacobian = CURIsim.gym.acquire_jacobian_tensor(CURIsim.sim, "robot")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to curi hand
j_eef = jacobian[:, curi_hand_index - 1, :, 7:14]

# get mass matrix tensor
_massmatrix = CURIsim.gym.acquire_mass_matrix_tensor(CURIsim.sim, "robot")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, 7:14, 7:14]  # only need elements corresponding to the curi arm

# get rigid body state tensor
_rb_states = CURIsim.gym.acquire_rigid_body_state_tensor(CURIsim.sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = CURIsim.gym.acquire_dof_state_tensor(CURIsim.sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(CURIsim.num_envs, DOF, 1)
dof_vel = dof_states[:, 1].view(CURIsim.num_envs, DOF, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([CURIsim.num_envs], False, dtype=torch.bool).to(CURIsim.device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# simulation loop
while not CURIsim.gym.query_viewer_has_closed(CURIsim.viewer):

    # step the physics
    CURIsim.gym.simulate(CURIsim.sim)
    CURIsim.gym.fetch_results(CURIsim.sim, True)

    # refresh tensors
    CURIsim.gym.refresh_rigid_body_state_tensor(CURIsim.sim)
    CURIsim.gym.refresh_dof_state_tensor(CURIsim.sim)
    CURIsim.gym.refresh_jacobian_tensors(CURIsim.sim)
    CURIsim.gym.refresh_mass_matrix_tensors(CURIsim.sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.11 if controller == "ik" else 0.10

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 14] + dof_pos[:, 15]
    gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(CURIsim.num_envs, 1, 3), hand_yaw_dir.view(CURIsim.num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 1] = torch.where(above_box, box_pos[:, 1] + grasp_offset, box_pos[:, 1] + grasp_offset * 2.5)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    import rofunc as rf

    rf.logger.beauty_print("pos_err: {}".format(pos_err), type="info")

    # Deploy control based on type
    if controller == "ik":
        pos_action[:, 7:14] = dof_pos.squeeze(-1)[:, 7:14] + control_ik(dpose)
    else:  # osc
        effort_action[:, 7:14] = control_osc(dpose)

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset + 0.02) | gripped
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 1] > 0.6)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * CURIsim.num_envs).to(CURIsim.device),
                            torch.Tensor([[0.04, 0.04]] * CURIsim.num_envs).to(CURIsim.device))
    pos_action[:, 14:16] = grip_acts

    # Deploy actions
    CURIsim.gym.set_dof_position_target_tensor(CURIsim.sim, gymtorch.unwrap_tensor(pos_action))
    CURIsim.gym.set_dof_actuation_force_tensor(CURIsim.sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    CURIsim.gym.step_graphics(CURIsim.sim)
    CURIsim.gym.draw_viewer(CURIsim.viewer, CURIsim.sim, False)
    CURIsim.gym.sync_frame_time(CURIsim.sim)

# cleanup
CURIsim.gym.destroy_viewer(CURIsim.viewer)
CURIsim.gym.destroy_sim(CURIsim.sim)
