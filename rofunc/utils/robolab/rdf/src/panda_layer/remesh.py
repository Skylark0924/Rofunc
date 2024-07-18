import trimesh
import glob
import os
import numpy as np

# mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/*"
# mesh_files = glob.glob(mesh_path)
# save_mesh = False
#
# for mf in mesh_files:
#     name = os.path.basename(mf)[:-4]
#     mesh = trimesh.load(mf)
#     new_mesh = mesh.simplify_quadratic_decimation(100)
#     if save_mesh:
#         trimesh.exchange.export.export_mesh(new_mesh, os.path.join(os.path.dirname(os.path.realpath(__file__)),f'meshes/{name}_face_100.stl'))


# mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/face_100/*.stl"
# mesh_files = glob.glob(mesh_path)
# save_mesh = True
#
# for mf in mesh_files:
#     name = os.path.basename(mf)[:-13]
#     print(name)
#     mesh = trimesh.load(mf)
#     verts = mesh.vertices
#     faces = mesh.faces
#     new_verts,new_faces = trimesh.remesh.subdivide_to_size(verts, faces, max_edge=0.1, max_iter=10, return_index=False)
#     new_mesh = trimesh.Trimesh(new_verts,new_faces)
#     # new_mesh = new_mesh.simplify_quadratic_decimation(500)
#     print(len(new_mesh.vertices))
#     # new_mesh.show()
#     if save_mesh:
#         save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),f'meshes/smooth/')
#         if os.path.exists(save_path) is not True:
#             os.mkdir(save_path)
#         trimesh.exchange.export.export_mesh(new_mesh, os.path.join(os.path.dirname(os.path.realpath(__file__)),f'meshes/smooth/{name}.stl'))
mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"
mesh_files = glob.glob(mesh_path)
for mf in mesh_files:
    mesh_name = mf.split('/')[-1].split('_')[0]
    print(mesh_name)
    if mesh_name == 'link0':
        continue
    scene = trimesh.Scene()
    mesh = trimesh.load(mf)
    mesh.show()
    print(mesh.vertices.shape)
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.005, max_iter=10,
                                                       return_index=False)
    print(vertices.shape)
    mesh = trimesh.Trimesh(vertices, faces)
    # ray_visualize = trimesh.load_path(np.hstack((mesh.vertices, mesh.vertices + mesh.vertex_normals / 100)).reshape(-1, 2, 3))
    # scene.add_geometry([mesh,ray_visualize])
    # scene.show()
    # trimesh.exchange.export.export_mesh(mesh, os.path.dirname(os.path.realpath(__file__)) + f"/meshes/high_quality/{mesh_name}.stl")
