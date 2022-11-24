"""
Tracking the trajectory with interference by CURI
=================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""
import os
import numpy as np
import rofunc as rf
from isaacgym import gymutil
import torch

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj_l = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1l.npy'))
traj_r = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1r.npy'))
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

num_envs = 1
num_bodies = 39
num_dofs = 25
force_intf = 10000
torque_intf = 100000

forces = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float)
torques = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float)
# efforts = torch.zeros((num_envs, num_dofs, 1), dtype=torch.float)
efforts = np.zeros(num_dofs, dtype=np.float32)
# forces[:, 25, 2] = force_intf
torques[:, 9, 1] = torque_intf
# forces[:, 9, 2] = force_intf
# torques[:, 36, 2] = torque_intf
# efforts[:, 9, 0] = 100000
efforts[4] = -100000
rf.curi.run_traj_multi_joints_with_interference(args, traj=[traj_l, traj_r], intf_mode="body_forces",
                                                intf_forces=forces, intf_torques=torques,
                                                intf_index=[50],
                                                num_envs=num_envs, update_freq=0.001, save_name='state7')

# rf.curi.run_traj_multi_joints_with_interference(args, traj=[traj_l, traj_r], intf_index=[i for i in range(50, 100)],
#                                                 intf_mode="actor_dof_efforts",
#                                                 intf_efforts=efforts,
#                                                 num_envs=num_envs, update_freq=0.001, save_name='state5')
