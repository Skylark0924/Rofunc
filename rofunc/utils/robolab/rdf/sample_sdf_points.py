import trimesh
import glob
import os
import numpy as np
import mesh_to_sdf
import skimage
import pyrender
import torch

mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/panda_layer/meshes/voxel_128/*.stl"
mesh_files = glob.glob(mesh_path)
mesh_files = sorted(mesh_files)[1:]  # except finger

for mf in mesh_files:
    mesh_name = mf.split('/')[-1].split('.')[0]
    print(mesh_name)
    mesh = trimesh.load(mf)
    mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)

    center = mesh.bounding_box.centroid
    scale = np.max(np.linalg.norm(mesh.vertices - center, axis=1))

    # sample points near surface (as same as deepSDF)
    near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,
                                                                number_of_points=500000,
                                                                surface_point_method='scan',
                                                                sign_method='normal',
                                                                scan_count=100,
                                                                scan_resolution=400,
                                                                sample_point_count=10000000,
                                                                normal_sample_count=100,
                                                                min_size=0.015,
                                                                return_gradients=False)
    # # sample points randomly within the bounding box [-1,1]
    random_points = np.random.rand(500000, 3) * 2.0 - 1.0
    random_sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                         random_points,
                                         surface_point_method='scan',
                                         sign_method='normal',
                                         bounding_radius=None,
                                         scan_count=100,
                                         scan_resolution=400,
                                         sample_point_count=10000000,
                                         normal_sample_count=100)

    # save data
    data = {
        'near_points': near_points,
        'near_sdf': near_sdf,
        'random_points': random_points,
        'random_sdf': random_sdf,
        'center': center,
        'scale': scale
    }
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'data/sdf_points')
    if os.path.exists(save_path) is not True:
        os.mkdir(save_path)
    np.save(os.path.join(save_path, f'voxel_128_{mesh_name}.npy'), data)

    # # # for visualization
    # data = np.load(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),f'data/sdf_points/voxel_128_{mesh_name}.npy')), allow_pickle=True).item()
    # random_points = data['random_points']
    # random_sdf = data['random_sdf']
    # near_points = data['near_points']
    # near_sdf = data['near_sdf']
    # colors = np.zeros(random_points.shape)
    # colors[random_sdf < 0, 2] = 1
    # colors[random_sdf > 0, 0] = 1
    # cloud = pyrender.Mesh.from_points(random_points, colors=colors)
    # scene = pyrender.Scene()
    # scene.add(cloud)
    # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
