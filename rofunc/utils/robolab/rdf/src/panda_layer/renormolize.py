import trimesh
import glob
import os
import numpy as np
import torch

mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/link8_vis.stl"
mesh_files = glob.glob(mesh_path)
save_mesh = True

for mf in mesh_files:
    scene = trimesh.Scene()
    mesh = trimesh.load(mf)
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.05, max_iter=10,
                                                       return_index=False)
    mesh = trimesh.Trimesh(vertices, faces)
    print(mesh.vertices.shape)
    center = np.mean(mesh.vertices, axis=0)
    verts = torch.from_numpy(mesh.vertices - center)
    normals = torch.from_numpy(mesh.vertex_normals)
    cosine = torch.cosine_similarity(verts, normals)
    print(cosine)
    normals[cosine < 0] = -normals[cosine < 0]
    normals = normals.numpy()
    ray_visualize = trimesh.load_path(np.hstack((mesh.vertices, mesh.vertices + normals / 100)).reshape(-1, 2, 3))
    scene.add_geometry(mesh)
    scene.add_geometry(ray_visualize)
    scene.show()
    # new_verts,new_faces = trimesh.remesh.subdivide_to_size(verts, faces, max_edge=0.1, max_iter=10, return_index=False)
    # new_mesh = trimesh.Trimesh(new_verts,new_faces)
    # # new_mesh = new_mesh.simplify_quadratic_decimation(500)
    # print(len(new_mesh.vertices))
    # # new_mesh.show()
    # if save_mesh:
    #     save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'meshes/smooth/')
    #     if os.path.exists(save_path) is not True:
    #         os.mkdir(save_path)
    #     trimesh.exchange.export.export_mesh(new_mesh, os.path.join(os.path.dirname(os.path.realpath(__file__)),f'meshes/smooth/{name}.stl'))
