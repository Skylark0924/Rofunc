"""
CURI screw nut
=================

This example shows how to use the gym interface to control the CURI robot to screw a nut onto a bolt.
"""
import os.path
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import rofunc as rf

DOF = 18
# Gripper_index = [14, 15, 23, 24]
Gripper_index = [7, 8, 16, 17]
# Arm_joint_index = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
Arm_joint_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[0:3] * torch.sign(q_r[3])


def control_ik(dpose, damping, j_eef, num_envs):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


class ScrewFSM:

    def __init__(self, sim_dt, nut_height, bolt_height, screw_speed, screw_limit_angle, device, env_idx):
        self._sim_dt = sim_dt
        self._nut_height = nut_height
        self._bolt_height = bolt_height
        self._screw_speed = screw_speed
        self._screw_limit_angle = screw_limit_angle
        self.device = device
        self.env_idx = env_idx

        # states:
        self._state = "go_above_nut"

        # control / position constants:
        self._above_offset = torch.tensor([0, 0, 0.08 + self._bolt_height], dtype=torch.float32, device=self.device)
        self._grip_offset = torch.tensor([0, 0, 0.12 + self._nut_height], dtype=torch.float32, device=self.device)
        self._lift_offset = torch.tensor([0, 0, 0.25 + self._bolt_height], dtype=torch.float32, device=self.device)
        self._above_bolt_offset = torch.tensor([0, 0, self._bolt_height], dtype=torch.float32,
                                               device=self.device) + self._grip_offset
        self._on_bolt_offset = torch.tensor([0, 0, 0.8 * self._bolt_height], dtype=torch.float32,
                                            device=self.device) + self._grip_offset
        self._hand_down_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        grab_angle = torch.tensor([np.pi / 6.0], dtype=torch.float32, device=self.device)
        grab_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        grab_quat = quat_from_angle_axis(grab_angle, grab_axis).squeeze()
        self._nut_grab_q = quat_mul(grab_quat, self._hand_down_quat)
        self._screw_angle = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        self._screw_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        self._dpose = torch.zeros(6, dtype=torch.float32, device=self.device)
        self._gripper_separation = 0.0

    def get_dp_from_target(self, target_pos, target_quat, hand_pose) -> float:
        self._dpose[:3] = target_pos - hand_pose[:3]
        self._dpose[3:] = orientation_error(target_quat, hand_pose[3:])
        return torch.norm(self._dpose, p=2)

    # returns
    def update(self, nut_pose, bolt_pose, hand_pose, current_gripper_sep):
        newState = self._state
        if self._state == "go_above_nut":
            self._gripper_separation = 0.08
            target_pos = nut_pose[:3] + self._above_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "prep_grip"
        elif self._state == "prep_grip":
            pass
            self._gripper_separation = 0.08
            target_pos = nut_pose[:3] + self._grip_offset
            targetQ = quat_mul(nut_pose[3:], self._nut_grab_q)
            error = self.get_dp_from_target(target_pos, targetQ, hand_pose)
            if error < 2e-3:
                newState = "grip"
        elif self._state == "grip":
            self._gripper_separation = 0.0
            target_pos = nut_pose[:3] + self._grip_offset
            targetQ = quat_mul(nut_pose[3:], self._nut_grab_q)
            error = self.get_dp_from_target(target_pos, targetQ, hand_pose)
            gripped = (current_gripper_sep < 0.035)
            if error < 1e-2 and gripped:
                newState = "lift"
        elif self._state == "lift":
            self._gripper_separation = 0.0
            target_pos = nut_pose[:3]
            target_pos[2] = bolt_pose[2] + 0.004
            target_pos = target_pos + self._lift_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "go_above_bolt"
        elif self._state == "go_above_bolt":
            self._gripper_separation = 0.0
            target_pos = bolt_pose[:3]
            target_pos = target_pos + self._above_bolt_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "go_on_bolt"
        elif self._state == "go_on_bolt":
            self._gripper_separation = 0.0
            target_pos = bolt_pose[:3]
            target_pos[2] = bolt_pose[2]
            target_pos = target_pos + self._on_bolt_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            if error < 2e-3:
                newState = "loosen_grip"
        elif self._state == "loosen_grip":
            target_sep = 0.037
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos = target_pos + self._on_bolt_offset
            error = self.get_dp_from_target(target_pos, self._hand_down_quat, hand_pose)
            un_gripped = current_gripper_sep > target_sep * 0.98
            if error < 2e-3 and un_gripped:
                self._screw_angle[0] = 0.0
                newState = "screw_motion"
        elif self._state == "screw_motion":
            target_sep = 0.037
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            self._screw_angle[0] = self._screw_angle[0] - self._sim_dt * self._screw_speed
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            if self._screw_angle[0] < -self._screw_limit_angle:
                newState = "ungrip_screw"
        elif self._state == "ungrip_screw":
            target_sep = 0.06
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            un_gripped = current_gripper_sep > target_sep * 0.98
            if un_gripped:
                newState = "rotate_back"
        elif self._state == "rotate_back":
            target_sep = 0.06
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            self._screw_angle[0] = self._screw_angle[0] + self._sim_dt * 2.0 * self._screw_speed
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            if self._screw_angle[0] > 0.99 * self._screw_limit_angle:
                newState = "back_to_screw_grip"
        elif self._state == "back_to_screw_grip":
            target_sep = 0.04
            self._gripper_separation = target_sep
            target_pos = bolt_pose[:3]
            target_pos[2] = nut_pose[2]
            target_pos = target_pos + self._grip_offset
            screw_quat = quat_from_angle_axis(self._screw_angle, self._screw_axis).squeeze()
            error = self.get_dp_from_target(target_pos, quat_mul(screw_quat, self._hand_down_quat), hand_pose)
            gripped = (current_gripper_sep < target_sep * 1.01)
            if error < 2e-3 and gripped:
                self._screw_angle[0] = self._screw_limit_angle
                newState = "screw_motion"

        if newState != self._state:
            self._state = newState
            print(f"Env {self.env_idx} going to state {newState}")

    @property
    def d_pose(self):
        return self._dpose

    @property
    def gripper_separation(self):
        return self._gripper_separation


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="curi Jacobian Inverse Kinematics (IK) Nut-Bolt Screwing",
    custom_parameters=custom_parameters,
)

# Force GPU:
if not args.use_gpu or args.use_gpu_pipeline:
    print("Forcing GPU sim - CPU sim not supported by SDF")
    args.use_gpu = True
    args.use_gpu_pipeline = True

# set torch device
device = args.sim_device

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 32
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.1

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

rofunc_path = rf.oslab.get_rofunc_path()
asset_root = os.path.join(rofunc_path, "simulator/assets")

# create table asset
table_dims = gymapi.Vec3(0.6, 2.5, 0.5)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create bolt asset
bolt_file = "urdf/nut_bolt/bolt_m4_tight_SI_5x.urdf"
bolt_options = gymapi.AssetOptions()
bolt_options.flip_visual_attachments = False  # default = False
bolt_options.fix_base_link = True
bolt_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
bolt_options.density = 800.0  # 7850.0
bolt_options.armature = 0.0  # default = 0.0
bolt_options.linear_damping = 0.0  # default = 0.0
bolt_options.max_linear_velocity = 1000.0  # default = 1000.0
bolt_options.angular_damping = 0.0  # default = 0.5
bolt_options.max_angular_velocity = 1000.0  # default = 64.0
bolt_options.disable_gravity = False  # default = False
bolt_options.enable_gyroscopic_forces = True  # default = True
bolt_asset = gym.load_asset(sim, asset_root, bolt_file, bolt_options)

# create nut asset
nut_file = "urdf/nut_bolt/nut_m4_tight_SI_5x.urdf"
nut_options = gymapi.AssetOptions()
nut_options.flip_visual_attachments = False  # default = False
nut_options.fix_base_link = False
nut_options.thickness = 0.0  # default = 0.02 (not overridden in .cpp)
nut_options.density = 800  # 7850.0  # default = 1000
nut_options.armature = 0.0  # default = 0.0
nut_options.linear_damping = 0.0  # default = 0.0
nut_options.max_linear_velocity = 1000.0  # default = 1000.0
nut_options.angular_damping = 0.0  # default = 0.5
nut_options.max_angular_velocity = 1000.0  # default = 64.0
nut_options.disable_gravity = False  # default = False
nut_options.enable_gyroscopic_forces = True  # default = True
nut_asset = gym.load_asset(sim, asset_root, nut_file, nut_options)

# create box asset

# asset_options = gymapi.AssetOptions()
# box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load curi asset
curi_asset_file = "urdf/curi/urdf/curi_isaacgym_dual_arm.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
curi_asset = gym.load_asset(sim, asset_root, curi_asset_file, asset_options)

# configure curi dofs
curi_dof_props = gym.get_asset_dof_properties(curi_asset)
curi_lower_limits = curi_dof_props["lower"]
curi_upper_limits = curi_dof_props["upper"]
curi_ranges = curi_upper_limits - curi_lower_limits
curi_mids = 0.3 * (curi_upper_limits + curi_lower_limits)

# use position drive for all dofs
curi_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
curi_dof_props["stiffness"][:].fill(400.0)
curi_dof_props["damping"][:].fill(40.0)
# grippers
curi_dof_props["driveMode"][7:9].fill(gymapi.DOF_MODE_POS)
curi_dof_props["stiffness"][7:9].fill(800.0)
curi_dof_props["damping"][7:9].fill(40.0)

# default dof states and position targets
curi_num_dofs = gym.get_asset_dof_count(curi_asset)
default_dof_pos = np.zeros(curi_num_dofs, dtype=np.float32)
default_dof_pos[:] = curi_mids[:]
# grippers open
default_dof_pos[7:9] = curi_upper_limits[7:9]

default_dof_state = np.zeros(curi_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
curi_link_dict = gym.get_asset_rigid_body_dict(curi_asset)
curi_hand_index = curi_link_dict["panda_left_hand"]

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

curi_pose = gymapi.Transform()
curi_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(1, 0.0, 0.5 * table_dims.z)
bolt_pose = gymapi.Transform()
nut_pose = gymapi.Transform()

# fsm parameters:
fsm_device = 'cpu'

envs = []
nut_idxs = []
bolt_idxs = []
hand_idxs = []
fsms = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add bolt
    bolt_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.1)
    bolt_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.0)
    bolt_pose.p.z = table_dims.z
    bolt_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    bolt_handle = gym.create_actor(env, bolt_asset, bolt_pose, "bolt", i, 0)
    bolt_props = gym.get_actor_rigid_shape_properties(env, bolt_handle)
    # bolt_props[0].filter = imesh
    bolt_props[0].friction = 0.0  # default = ?
    bolt_props[0].rolling_friction = 0.0  # default = 0.0
    bolt_props[0].torsion_friction = 0.0  # default = 0.0
    bolt_props[0].restitution = 0.0  # default = ?
    bolt_props[0].compliance = 0.0  # default = 0.0
    bolt_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, bolt_handle, bolt_props)

    # get global index of box in rigid body state tensor
    bolt_idx = gym.get_actor_rigid_body_index(env, bolt_handle, 0, gymapi.DOMAIN_SIM)
    bolt_idxs.append(bolt_idx)

    # add nut
    nut_pose.p.x = bolt_pose.p.x + np.random.uniform(-0.04, 0.04)
    nut_pose.p.y = bolt_pose.p.y + 0.2 + np.random.uniform(-0.04, 0.04)
    nut_pose.p.z = table_dims.z + 0.02
    nut_handle = gym.create_actor(env, nut_asset, nut_pose, "nut", i, 0)
    nut_props = gym.get_actor_rigid_shape_properties(env, nut_handle)
    # nut_props[0].filter = i
    nut_props[0].friction = 0.2  # default = ?
    nut_props[0].rolling_friction = 0.0  # default = 0.0
    nut_props[0].torsion_friction = 0.0  # default = 0.0
    nut_props[0].restitution = 0.0  # default = ?
    nut_props[0].compliance = 0.0  # default = 0.0
    nut_props[0].thickness = 0.0  # default = 0.0
    gym.set_actor_rigid_shape_properties(env, nut_handle, nut_props)

    # get global index of box in rigid body state tensor
    nut_idx = gym.get_actor_rigid_body_index(env, nut_handle, 0, gymapi.DOMAIN_SIM)
    nut_idxs.append(nut_idx)

    # add curi
    curi_handle = gym.create_actor(env, curi_asset, curi_pose, "curi", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, curi_handle, curi_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, curi_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, curi_handle, default_dof_pos)

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, curi_handle, "panda_left_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # create env's fsm - run them on CPU
    fsms.append(ScrewFSM(sim_params.dt, 0.016, 0.1, 30.0 / 180.0 * np.pi, 60.0 / 180.0 * np.pi, fsm_device, i))

# point camera at middle env
cam_pos = gymapi.Vec3(1, 0, 0.6)
cam_target = gymapi.Vec3(-1, 0, 0.5)
middle_env = envs[0]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# get jacobian tensor
# for fixed-base curi, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "curi")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to curi hand
j_eef = jacobian[:, curi_hand_index - 1, :, :7]

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, DOF, 1)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)

# dp and gripper sep tensors:
d_pose = torch.zeros((num_envs, 6), dtype=torch.float32, device=fsm_device)
grip_sep = torch.zeros((num_envs, 1), dtype=torch.float32, device=fsm_device)

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    rb_states_fsm = rb_states.to(fsm_device)
    nut_poses = rb_states_fsm[nut_idxs, :7]
    bolt_poses = rb_states_fsm[bolt_idxs, :7]
    hand_poses = rb_states_fsm[hand_idxs, :7]
    dof_pos_fsm = dof_pos.to(fsm_device)
    cur_grip_sep_fsm = dof_pos_fsm[:, 7] + dof_pos_fsm[:, 8]
    for env_idx in range(num_envs):
        fsms[env_idx].update(nut_poses[env_idx, :], bolt_poses[env_idx, :], hand_poses[env_idx, :],
                             cur_grip_sep_fsm[env_idx])
        d_pose[env_idx, :] = fsms[env_idx].d_pose
        grip_sep[env_idx] = fsms[env_idx].gripper_separation

    pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(d_pose.unsqueeze(-1).to(device), damping, j_eef,
                                                                num_envs)
    # gripper actions depend on distance between hand and box

    grip_acts = torch.cat((0.5 * grip_sep, 0.5 * grip_sep), 1).to(device)
    pos_action[:, 7:9] = grip_acts

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
