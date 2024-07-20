import copy
import trimesh
import numpy as np
import torch
import os
import glob
import mesh_to_sdf

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def transform_points(points, trans, device):
    # transfrom points in SE(3)        points:(N,3)       trans:(B,4,4)
    B, N = trans.shape[0], points.shape[0]
    ones = torch.ones([B, N, 1], device=device).float()
    points_ = torch.cat([points.unsqueeze(0).expand(B, N, 3), ones], dim=-1)
    points_ = torch.matmul(trans, points_.permute(0, 2, 1)).permute(0, 2, 1)
    return points_[:, :, :3].float()


def mse(yhat, y):
    return torch.nn.MSELoss(reduction='mean')(yhat, y)


def rmse(yhat, y):
    return torch.sqrt(mse(yhat, y))


def print_eval(yhat, y, string='default'):
    yhat, y = yhat.view(-1), y.view(-1)
    y_near = (y.abs() < 0.03)
    y_far = (y.abs() > 0.03)
    MAE = (yhat - y).abs().mean()
    MSE = mse(yhat, y)
    RMSE = rmse(yhat, y)
    MAE_near = (yhat[y_near] - y[y_near]).abs().mean()
    MSE_near = mse(yhat[y_near], y[y_near])
    RMSE_near = rmse(yhat[y_near], y[y_near])
    MAE_far = (yhat[y_far] - y[y_far]).abs().mean()
    MSE_far = mse(yhat[y_far], y[y_far])
    RMSE_far = rmse(yhat[y_far], y[y_far])
    # print(f'{string}\t'
    #       f'abs:{MAE:.6f}\t'
    #       f'mse:{MSE:.6f}\t'
    #       f'rmse:{RMSE:.6f}\t'
    #       f'abs_near:{MAE_near:.6f}\t'
    #       f'mse_near:{MSE_near:.6f}\t'
    #       f'rmse_near:{RMSE_near:.6f}\t'
    #       f'abs_far:{MAE_far:.6f}\t'
    #       f'mse_far:{MSE_far:.6f}\t'
    #       f'rmse_far:{RMSE_far:.6f}\t')
    res = [MAE, MSE, RMSE, MAE_near, MSE_near, RMSE_near, MAE_far, MSE_far, RMSE_far]
    return [r.item() for r in res]


def eval_chamfer_distance(tag):
    from chamfer_distance import ChamferDistance as chamfer_dist
    mesh_path = os.path.join(CUR_DIR, "panda_layer/meshes/voxel_128/*")
    mesh_files = glob.glob(mesh_path)
    mesh_files = sorted(mesh_files)[1:]  # except finger
    res = []
    for i, mf in enumerate(mesh_files):
        scene = trimesh.Scene()
        mesh_name = mf.split('/')[-1].split('.')[0]
        mesh = trimesh.load(mf)
        mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
        # scene.add_geometry(mesh)
        rec_mesh = trimesh.load(CUR_DIR + f'/output_meshes/{tag}_{mesh_name}.stl')
        # rec_mesh.vertices = rec_mesh.vertices + [2.0,0,0]
        # rec_mesh.visual.face_colors= [255,0,0,100]
        # scene.add_geometry(rec_mesh)

        mesh_points = trimesh.sample.sample_surface_even(mesh, 30000)[0]
        rec_mesh_points = trimesh.sample.sample_surface_even(rec_mesh, 30000)[0]

        chamfer = chamfer_dist()
        x_near, y_near, xidx_near, yidx_near = chamfer(torch.from_numpy(mesh_points).float().unsqueeze(0).to('cuda'),
                                                       torch.from_numpy(rec_mesh_points).float().unsqueeze(0).to(
                                                           'cuda'))
        cd_mean = (torch.mean(x_near) + torch.mean(y_near)).item() * 1000.0
        cd_max = (torch.max(x_near) + torch.max(y_near)).item() * 1000.0
        res.append(np.asarray([cd_mean, cd_max]))
    cd_mean, cd_max = np.mean(res, axis=0)
    return cd_mean, cd_max


def visualize_reconstructed_whole_body(model, trans_list, tag):
    mesh_path = os.path.join(CUR_DIR, f"src/output_meshes/{tag}_*.stl")
    mesh_files = glob.glob(mesh_path)
    mesh_files.sort()
    view_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    scene = trimesh.Scene()
    for i, mf in enumerate(mesh_files):
        mesh = trimesh.load(mf)
        mesh_dict = model[i]
        offset = mesh_dict['offset'].cpu().numpy()
        scale = mesh_dict['scale']
        mesh.vertices = mesh.vertices * scale + offset
        mesh.apply_transform(trans_list[i].squeeze().cpu().numpy())
        mesh.apply_transform(view_mat)
        scene.add_geometry(mesh)
    scene.show()


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    elif np.linalg.norm(a - b) < 1e-6:
        return np.eye(3)  # cross of all zeros only occurs on identical directions
    else:
        return -np.eye(3)


def create_arrow(vector, point, vec_length=0.05, color=[255, 0, 0]):
    v_norm = np.linalg.norm(vector)
    r = vec_length / 12.0
    h = vec_length / 2.0
    cy = trimesh.creation.cylinder(r / 2.0, h)
    cy.vertices[:, 2] = cy.vertices[:, 2] + h / 2.0
    cc = trimesh.creation.cone(r, h)
    cc.vertices[:, 2] = cc.vertices[:, 2] + h
    arrow = np.sum([cy, cc])

    transformation = np.eye(4)

    rot = rotation_matrix_from_vectors(np.array([0, 0, 1]), vector / v_norm)

    transformation[:3, :3] = rot
    transformation[:3, 3] = point
    arrow.apply_transform(transformation)
    arrow.visual.face_colors = np.array(color, dtype=object)
    return arrow
