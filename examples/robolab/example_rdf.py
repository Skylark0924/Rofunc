"""
Robot distance field (RDF)
========================

This example demonstrates how to use the RDF class to train a Bernstein Polynomial model for the robot distance field
from URDF/MJCF files and visualize the reconstructed whole body.
"""

import argparse
import os
import time

import numpy as np
import torch

import rofunc as rf


def rdf_from_robot_model(args):
    rdf_bp = rf.robolab.rdf.RDF(args)

    #  train Bernstein Polynomial model
    if args.train:
        rdf_bp.train()

    # load trained model
    rdf_model_path = os.path.join(args.robot_asset_root, 'rdf/BP', f'BP_{args.n_func}.pt')
    rdf_model = torch.load(rdf_model_path)

    # visualize the Bernstein Polynomial model for each robot link
    rdf_bp.create_surface_mesh(rdf_model, nbData=128, vis=False, save_mesh_name=f'BP_{args.n_func}')

    joint_max = rdf_bp.robot.joint_limit_max
    joint_min = rdf_bp.robot.joint_limit_min
    num_joint = rdf_bp.robot.num_joint
    # joint_value = torch.rand(num_joint).to(args.device) * (joint_max - joint_min) + joint_min
    joint_value = torch.zeros(num_joint).to(args.device)

    trans_dict = rdf_bp.robot.get_trans_dict(joint_value)
    # visualize the Bernstein Polynomial model for the whole body
    rdf_bp.visualize_reconstructed_whole_body(rdf_model, trans_dict, tag=f'BP_{args.n_func}')

    # run RDF
    x = torch.rand(10, 3).to(args.device) * 2.0 - 1.0
    joint_value = torch.rand(100, rdf_bp.robot.num_joint).to(args.device).float()
    base_trans = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(joint_value), 4,
                                                                                           4).float().to(args.device)

    start_time = time.time()
    sdf, gradient = rdf_bp.get_whole_body_sdf_batch(x, joint_value, rdf_model, base_trans=base_trans,
                                                    use_derivative=True)
    print('Time cost:', time.time() - start_time)
    print('sdf:', sdf.shape, 'gradient:', gradient.shape)

    start_time = time.time()
    sdf, joint_grad = rdf_bp.get_whole_body_sdf_with_joints_grad_batch(x, joint_value, rdf_model, base_trans=base_trans)
    print('Time cost:', time.time() - start_time)
    print('sdf:', sdf.shape, 'joint gradient:', joint_grad.shape)

    # visualize the 2D & 3D SDF with gradient
    # joint_value = torch.zeros(num_joint).to(args.device).reshape((-1, num_joint))

    joint_value = (torch.rand(num_joint).to(args.device).reshape((-1, num_joint))*0.5 * (joint_max - joint_min) + joint_min)
    rf.robolab.rdf.plot_2D_panda_sdf(joint_value, rdf_bp, nbData=80, model=rdf_model, device=args.device)
    rf.robolab.rdf.plot_3D_panda_with_gradient(joint_value, rdf_bp, model=rdf_model, device=args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=8, type=int)
    parser.add_argument('--train_epoch', default=200, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--save_mesh_dict', action='store_false')
    parser.add_argument('--sampled_points', action='store_false')
    parser.add_argument('--parallel', action='store_false')
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/urdf/alicia", type=str)
    # parser.add_argument('--robot_asset_name', default="Alicia_0624.xml", type=str)
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/urdf/franka_description", type=str)
    # parser.add_argument('--robot_asset_name', default="robots/franka_panda.urdf", type=str)
    parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/mjcf/bruce", type=str)
    parser.add_argument('--robot_asset_name', default="bruce.xml", type=str)
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/mjcf/hotu", type=str)
    # parser.add_argument('--robot_asset_name', default="hotu_humanoid.xml", type=str)
    parser.add_argument('--rdf_model_path', default=None)
    parser.add_argument('--joint_conf_path', default=None)
    parser.add_argument('--sampled_points_dir', default=None, type=str)
    args = parser.parse_args()
    args.rdf_model_path = f"{args.robot_asset_root}/rdf/BP/BP_{args.n_func}.pt"
    args.joint_conf_path = f"{args.robot_asset_root}/rdf/BP/joint_conf.pt"

    rdf_from_robot_model(args)
