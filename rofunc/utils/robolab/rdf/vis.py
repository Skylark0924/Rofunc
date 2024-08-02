import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from rofunc.utils.robolab.rdf import utils


def plot_2D_panda_sdf(joint_value, rdf_bp, nbData, model, device):
    domain_0 = torch.linspace(-2.0, 2.0, nbData).to(device)
    domain_1 = torch.linspace(-2.0, 2.0, nbData).to(device)
    grid_x, grid_y = torch.meshgrid(domain_0, domain_1)
    p1 = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros_like(grid_x.reshape(-1))], dim=1)
    p2 = torch.stack([torch.zeros_like(grid_x.reshape(-1)), grid_x.reshape(-1) * 0.4, grid_y.reshape(-1) * 0.4 + 0.375],
                     dim=1)
    p3 = torch.stack(
        [grid_x.reshape(-1) * 0.4 + 0.2, torch.zeros_like(grid_x.reshape(-1)), grid_y.reshape(-1) * 0.4 + 0.375], dim=1)
    grid_x, grid_y = grid_x.detach().cpu().numpy(), grid_y.detach().cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.rc('font', size=25)
    p2_split = torch.split(p2, 1000, dim=0)
    sdf, ana_grad = [], []
    for p_2 in p2_split:
        sdf_split, ana_grad_split = rdf_bp.get_whole_body_sdf_batch(p_2, joint_value, model, use_derivative=True)
        sdf_split, ana_grad_split = sdf_split.squeeze(), ana_grad_split.squeeze()
        sdf.append(sdf_split)
        ana_grad.append(ana_grad_split)
    sdf = torch.cat(sdf, dim=0)
    ana_grad = torch.cat(ana_grad, dim=0)
    p2 = p2.detach().cpu().numpy()
    sdf = sdf.squeeze().reshape(nbData, nbData).detach().cpu().numpy()
    ct1 = plt.contour(grid_x * 0.4, grid_y * 0.4 + 0.375, sdf, levels=12)
    plt.clabel(ct1, inline=False, fontsize=10)
    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:, [1, 2]], dim=-1) * 0.01
    p2_3d = p2.reshape(nbData, nbData, 3)
    ana_grad_3d = ana_grad_2d.reshape(nbData, nbData, 2)
    plt.quiver(p2_3d[0:-1:4, 0:-1:4, 1], p2_3d[0:-1:4, 0:-1:4, 2],
               ana_grad_3d[0:-1:4, 0:-1:4, 0].detach().cpu().numpy(),
               ana_grad_3d[0:-1:4, 0:-1:4, 1].detach().cpu().numpy(), scale=0.5, color=[0.1, 0.1, 0.1])
    plt.title('YoZ')
    plt.show()

    # plt.subplot(1,3,3)
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=25)
    p3_split = torch.split(p3, 1000, dim=0)
    sdf, ana_grad = [], []
    for p_3 in p3_split:
        sdf_split, ana_grad_split = rdf_bp.get_whole_body_sdf_batch(p_3, joint_value, model, use_derivative=True)
        sdf_split, ana_grad_split = sdf_split.squeeze(), ana_grad_split.squeeze()
        sdf.append(sdf_split)
        ana_grad.append(ana_grad_split)
    sdf = torch.cat(sdf, dim=0)
    ana_grad = torch.cat(ana_grad, dim=0)
    p3 = p3.detach().cpu().numpy()
    sdf = sdf.squeeze().reshape(nbData, nbData).detach().cpu().numpy()
    ct1 = plt.contour(grid_x * 0.4 + 0.2, grid_y * 0.4 + 0.375, sdf, levels=12)
    plt.clabel(ct1, inline=False, fontsize=10)
    ana_grad_2d = -torch.nn.functional.normalize(ana_grad[:, [0, 2]], dim=-1) * 0.01
    p3_3d = p3.reshape(nbData, nbData, 3)
    ana_grad_3d = ana_grad_2d.reshape(nbData, nbData, 2)
    plt.quiver(p3_3d[0:-1:4, 0:-1:4, 0], p3_3d[0:-1:4, 0:-1:4, 2],
               ana_grad_3d[0:-1:4, 0:-1:4, 0].detach().cpu().numpy(),
               ana_grad_3d[0:-1:4, 0:-1:4, 1].detach().cpu().numpy(), scale=0.5, color=[0.1, 0.1, 0.1])
    plt.title('XoZ')
    plt.show()


def plot_3D_panda_with_gradient(joint_value, rdf_bp, model, device):
    robot_mesh = rdf_bp.robot.get_forward_robot_mesh(joint_value)[0]
    robot_mesh = np.sum(robot_mesh)

    surface_points = robot_mesh.vertices
    scene = trimesh.Scene()
    # robot mesh
    scene.add_geometry(robot_mesh)
    choice = np.random.choice(len(surface_points), 1024, replace=False)
    surface_points = surface_points[choice]
    p = torch.from_numpy(surface_points).float().to(device)
    ball_query = trimesh.creation.uv_sphere(1).vertices
    choice_ball = np.random.choice(len(ball_query), 1024, replace=False)
    ball_query = ball_query[choice_ball]
    p = p + torch.from_numpy(ball_query).float().to(device) * 0.5
    sdf, ana_grad = rdf_bp.get_whole_body_sdf_batch(p, joint_value, model, use_derivative=True)
    sdf, ana_grad = sdf.squeeze().detach().cpu().numpy(), ana_grad.squeeze().detach().cpu().numpy()
    # points
    pts = p.detach().cpu().numpy()
    colors = np.zeros_like(pts, dtype=object)
    colors[:, 0] = np.abs(sdf) * 400
    # pc =trimesh.PointCloud(pts,colors)
    # scene.add_geometry(pc)

    # gradients
    for i in range(len(pts)):
        dg = ana_grad[i]
        if dg.sum() == 0:
            continue
        c = colors[i]
        if np.any(c > 255):
            c = [255, 0, 0]
        # print(c)
        m = utils.create_arrow(-dg, pts[i], vec_length=0.05, color=c)
        scene.add_geometry(m)
    scene.show()


def generate_panda_mesh_sdf_points(max_dist=0.10):
    # represent SDF using basis functions
    import glob
    import mesh_to_sdf
    mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/panda_layer/meshes/voxel_128/*"
    mesh_files = glob.glob(mesh_path)
    mesh_files = sorted(mesh_files)[1:]  # except finger
    mesh_dict = {}

    for i, mf in enumerate(mesh_files):
        mesh_name = mf.split('/')[-1].split('.')[0]
        print(mesh_name)
        mesh = trimesh.load(mf)
        mesh_dict[i] = {}
        mesh_dict[i]['mesh_name'] = mesh_name

        vert = mesh.vertices
        points = vert + np.random.uniform(-max_dist, max_dist, size=vert.shape)
        sdf = random_sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                                   points,
                                                   surface_point_method='scan',
                                                   sign_method='normal',
                                                   bounding_radius=None,
                                                   scan_count=100,
                                                   scan_resolution=400,
                                                   sample_point_count=10000000,
                                                   normal_sample_count=100)
        mesh_dict[i]['points'] = points
        mesh_dict[i]['sdf'] = sdf
    np.save('data/panda_mesh_sdf.npy', mesh_dict)


def vis_panda_sdf(pose, joint_value, device):
    data = np.load('data/panda_mesh_sdf.npy', allow_pickle=True).item()
    trans = panda.get_transformations_each_link(pose, joint_value)
    pts = []
    for i, k in enumerate(data.keys()):
        points = data[k]['points']
        sdf = data[k]['sdf']
        print(points.shape, sdf.shape)
        choice = (sdf < 0.05) * (sdf > 0.045)
        points = points[choice]
        sdf = sdf[choice]

        sample = np.random.choice(len(points), 128, replace=True)
        points, sdf = points[sample], sdf[sample]

        points = torch.from_numpy(points).float().to(device)
        ones = torch.ones([len(points), 1], device=device).float()
        points = torch.cat([points, ones], dim=-1)
        t = trans[i].squeeze()
        print(points.shape, t.shape)

        trans_points = torch.matmul(t, points.t()).t()[:, :3]
        pts.append(trans_points)
    pts = torch.cat(pts, dim=0).detach().cpu().numpy()
    print(pts.shape)
    scene = trimesh.Scene()
    robot_mesh = panda.get_forward_robot_mesh(pose, joint_value)[0]
    robot_mesh = np.sum(robot_mesh)
    scene.add_geometry(robot_mesh)
    pc = trimesh.PointCloud(pts, colors=[255, 0, 0])
    scene.add_geometry(pc)
    scene.show()
