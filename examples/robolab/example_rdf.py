"""
Robot distance field (RDF)
========================

This example demonstrates how to use the RDF_BP class to train a Bernstein Polynomial model for the robot distance field
from URDF/MJCF files and visualize the reconstructed whole body.
"""

import argparse
import os

import numpy as np
import torch

import rofunc as rf
from rofunc.utils.robolab.rdf.rdf import RDF_BP

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--domain_max', default=1.0, type=float)
parser.add_argument('--domain_min', default=-1.0, type=float)
parser.add_argument('--n_func', default=8, type=int)
parser.add_argument('--train_epoch', default=200, type=int)
parser.add_argument('--train', action='store_true')
parser.add_argument('--save_mesh_dict', action='store_false')
parser.add_argument('--load_sampled_points', action='store_false')
parser.add_argument('--sampled_points_dir',
                    default='/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/urdf/franka_description/rdf/sdf_points',
                    type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--robot_model',
                    default="/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/urdf/franka_description",
                    type=str)
args = parser.parse_args()

rdf_bp = RDF_BP(args)

#  train Bernstein Polynomial model
if args.train:
    rdf_bp.train()

# load trained model
model_path = os.path.join(args.robot_model, 'rdf/BP', f'BP_{args.n_func}.pt')
model = torch.load(model_path)

# visualize the Bernstein Polynomial model for each robot link
# rdf_bp.create_surface_mesh(model, nbData=128, vis=True, save_mesh_name=f'BP_{args.n_func}')

# # visualize the Bernstein Polynomial model for the whole body

joint_value = {"panda_joint1": 0.0,
               "panda_joint2": -0.3,
               "panda_joint3": 0.0,
               "panda_joint4": -2.2,
               "panda_joint5": 0.0,
               "panda_joint6": 2.0,
               "panda_joint7": np.pi / 4,
               "panda_finger_joint1": 0.03,
               "panda_finger_joint2": 0.03}
# pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(joint_value), 4, 4).float()

robot = rf.robolab.RobotModel(os.path.join(args.robot_model, "robots/franka_panda.urdf"), solve_engine="kinpy",
                              verbose=False)
pos, rot, ret = robot.get_fk(joint_value, "panda_leftfinger")

trans_list = {}
link_list = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6',
             'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
for link in link_list:
    val = ret[link]
    rot = rf.robolab.convert_quat_order(val.rot, "wxyz", "xyzw")
    homo_matrix = rf.robolab.homo_matrix_from_quat_tensor(rot, val.pos)
    trans_list[link.split("_")[-1]] = homo_matrix

rdf_bp.visualize_reconstructed_whole_body(model, trans_list, tag=f'BP_{args.n_func}')

# run RDF
x = torch.rand(128, 3).to(args.device) * 2.0 - 1.0
theta = torch.rand(2, 7).to(args.device).float()
pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(len(theta), 4, 4).float()
sdf, gradient = rdf_bp.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=True)
print('sdf:', sdf.shape, 'gradient:', gradient.shape)
sdf, joint_grad = rdf_bp.get_whole_body_sdf_with_joints_grad_batch(x, pose, theta, model)
print('sdf:', sdf.shape, 'joint gradient:', joint_grad.shape)
