"""
Bimanual box carrying using Robot distance field (RDF)
========================

This example plans the contacts of a bimanual box carrying task via optimization based on RDF.
"""
import argparse

import numpy as np
import torch
import trimesh

import rofunc as rf


def box_carrying_contact_rdf(args):
    box_size = np.array([0.18, 0.1, 0.16])
    box_pos = np.array([0.7934301890820722, 0.0, 0.3646743147850761])
    box_rotation = np.array([[0, 1.57, 0],
                             [-1.57, 0, 0],
                             [0, 0, 1]])

    rdf_model = torch.load(args.rdf_model_path)
    bbo_planner = rf.robolab.rdf.BBOPlanner(args, rdf_model, box_size, box_pos, box_rotation)
    num_joint = bbo_planner.rdf_bp.robot.num_joint

    # contact points
    contact_points = bbo_planner.contact_points
    p_l, p_r, n_l, n_r = contact_points[0], contact_points[1], contact_points[2], contact_points[3]

    # initial joint value
    joint_max = bbo_planner.rdf_bp.robot.joint_limit_max
    joint_min = bbo_planner.rdf_bp.robot.joint_limit_min
    mid_l = torch.rand(num_joint).to(args.device) * (joint_max - joint_min) + joint_min
    mid_r = torch.rand(num_joint).to(args.device) * (joint_max - joint_min) + joint_min

    # planning for both arm
    base_pose_l = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).float()
    base_pose_r = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).float()
    base_pose_l[0] = rf.robolab.homo_matrix_from_quat_tensor([-0.436865, 0.49775, 0.054428, 0.747283],
                                                             [0.396519, 0.07, 0.644388])[0].to(args.device)
    base_pose_r[0] = rf.robolab.homo_matrix_from_quat_tensor([0.436865, 0.49775, -0.054428, 0.747283],
                                                             [0.396519, -0.07, 0.644388])[0].to(args.device)
    # base_pose_l[0, :3, 3] = torch.tensor([0.4, 0.3, 0]).to(args.device)
    # base_pose_r[0, :3, 3] = torch.tensor([0.4, -0.3, 0]).to(args.device)

    joint_value_left = bbo_planner.optimizer(p_l, n_l, mid_l, base_trans=base_pose_l, batch=64)
    joint_value_right = bbo_planner.optimizer(p_r, n_r, mid_r, base_trans=base_pose_r, batch=64)
    joint_conf = {
        'joint_value_left': joint_value_left,
        'joint_value_right': joint_value_right
    }
    torch.save(joint_conf, args.joint_conf_path)

    # load planned joint conf
    data = torch.load(args.joint_conf_path)
    joint_value_left = data['joint_value_left']
    joint_value_right = data['joint_value_right']
    print('joint_value_left', joint_value_left.shape, 'joint_value_right', joint_value_right.shape)

    # visualize planning results
    scene = trimesh.Scene()
    pc1 = trimesh.PointCloud(bbo_planner.object_internal_points.detach().cpu().numpy(), colors=[0, 255, 0])
    pc2 = trimesh.PointCloud(p_l.detach().cpu().numpy(), colors=[255, 0, 0])
    pc3 = trimesh.PointCloud(p_r.detach().cpu().numpy(), colors=[255, 0, 0])
    scene.add_geometry([pc1, pc2, pc3])
    scene.add_geometry(bbo_planner.object_mesh)

    # visualize the final joint configuration
    for t_l, t_r in zip(joint_value_left, joint_value_right):
        print('t left:', t_l)
        robot_l = bbo_planner.rdf_bp.robot.get_forward_robot_mesh(t_l.reshape(-1, num_joint), base_pose_l)[0]
        robot_l = np.sum(robot_l)
        robot_l.visual.face_colors = [150, 150, 200, 200]
        scene.add_geometry(robot_l, node_name='robot_l')

        print('t right:', t_r)
        robot_r = bbo_planner.rdf_bp.robot.get_forward_robot_mesh(t_r.reshape(-1, num_joint), base_pose_r)[0]
        robot_r = np.sum(robot_r)
        robot_r.visual.face_colors = [150, 200, 150, 200]
        scene.add_geometry(robot_r, node_name='robot_r')
        scene.show()

        scene.delete_geometry('robot_l')
        scene.delete_geometry('robot_r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=8, type=int)
    parser.add_argument('--train_epoch', default=200, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--save_mesh_dict', action='store_false')
    parser.add_argument('--load_sampled_points', action='store_false')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/urdf/alicia", type=str)
    parser.add_argument('--robot_asset_name', default="Alicia_0624.xml", type=str)
    # parser.add_argument('--robot_asset_root', default="../../rofunc/simulator/assets/urdf/franka_description", type=str)
    # parser.add_argument('--robot_asset_name', default="robots/franka_panda.urdf", type=str)
    parser.add_argument('--rdf_model_path', default=None)
    parser.add_argument('--joint_conf_path', default=None)
    parser.add_argument('--sampled_points_dir', default=None, type=str)
    args = parser.parse_args()
    args.rdf_model_path = f"{args.robot_asset_root}/rdf/BP/BP_{args.n_func}.pt"
    args.joint_conf_path = f"{args.robot_asset_root}/rdf/BP/joint_conf.pt"

    box_carrying_contact_rdf(args)
