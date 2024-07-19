import trimesh
import glob
import os
import numpy as np
from mesh_to_sdf import mesh_to_voxels
import skimage
from mesh_to_sdf import surface_point_cloud
from mesh_to_sdf.utils import scale_to_unit_cube, scale_to_unit_sphere, get_raster_points, check_voxels

mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"
mesh_files = glob.glob(mesh_path)
print(mesh_files)
mesh_files = sorted(mesh_files)[1:]  # except finger
voxel_resolution = 128
for mf in mesh_files:
    mesh_name = mf.split('/')[-1].split('_')[0]
    print(mesh_name)
    scene = trimesh.Scene()
    mesh_origin = trimesh.load(mf)
    # mesh_origin.visual.face_colors = [255,0,0,150]
    center = mesh_origin.bounding_box.centroid
    scale = 2 / np.max(mesh_origin.bounding_box.extents)
    voxels = mesh_to_voxels(mesh_origin,
                            voxel_resolution=voxel_resolution,
                            surface_point_method='scan',
                            sign_method='normal',
                            scan_count=100,
                            scan_resolution=400,
                            sample_point_count=10000000,
                            normal_sample_count=100,
                            pad=True,
                            check_result=False)
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.0, spacing=(
    2 / voxel_resolution, 2 / voxel_resolution, 2 / voxel_resolution))
    mesh_voxelized = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh_voxelized.visual.face_colors = [0, 0, 255, 150]
    mesh_voxelized.vertices = mesh_voxelized.vertices / scale
    mesh_voxelized.vertices = mesh_voxelized.vertices - mesh_voxelized.bounding_box.centroid + center
    print(mesh_voxelized.vertices.shape)
    scene.add_geometry(mesh_voxelized)
    scene.show()
    # save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'meshes/voxel_{str(voxel_resolution)}')
    # if os.path.exists(save_path) is not True:
    #     os.mkdir(save_path)
    # trimesh.exchange.export.export_mesh(mesh_voxelized, os.path.join(save_path,f'{mesh_name}.stl'))
