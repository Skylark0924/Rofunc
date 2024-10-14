import os

import trimesh

from rofunc.utils.oslab.dir_process import list_absl_path, create_dir
from rofunc.utils.robolab.formatter.mjcf_parser.parser import from_path
# from rofunc.utils.visualab.geometry.override_trimesh_creation import cylinder, box, capsule, icosphere


class MJCF:
    def __init__(self, file_path):
        self.link_mesh_map = {}

        self.robot = from_path(file_path)
        self.get_link_mesh_map()

    def get_link_mesh_map(self):
        """
        Get the map of link and its corresponding geometries from the MJCF file.

        :return: {link_body_name: {geom_name: mesh_path}}
        """
        bodies = self.robot.find_all("body")
        self.robot.compiler.meshdir = self.robot.compiler.meshdir or "meshes"
        mesh_dir = os.path.join(self.robot.namescope.model_dir, self.robot.compiler.meshdir)
        create_dir(mesh_dir)
        all_mesh_file_stl = list_absl_path(mesh_dir, recursive=True, suffix=".stl")
        all_mesh_file_STL = list_absl_path(mesh_dir, recursive=True, suffix=".STL")
        all_mesh_files = all_mesh_file_stl + all_mesh_file_STL

        mesh_map = self.robot.get_assets_map()
        mesh_name_path_map = {}
        for mesh_name, mesh_file in mesh_map.items():
            mesh_path = None
            for mesh_file_exist in all_mesh_files:
                if mesh_file in mesh_file_exist:
                    mesh_path = mesh_file_exist
            if mesh_path is not None:
                mesh_name_path_map[mesh_name] = mesh_path
            else:
                raise FileNotFoundError(f"Mesh file {mesh_file} not found in the mesh directory.")

        # 遍历所有 bodies，处理几何体
        for body in bodies:
            geoms_this_body = body.geom
            self.link_mesh_map[body.name] = {}

            for geom in geoms_this_body:
                geom_type = geom.type or "capsule"  # 默认类型为胶囊
                geom_pos = geom.pos if geom.pos is not None else [0, 0, 0]

                # 处理不同的几何体类型
                if geom_type == "mesh":
                    geom_mesh_name = geom.mesh.name
                    geom_mesh_path = mesh_name_path_map[geom_mesh_name]
                    self.link_mesh_map[body.name][geom_mesh_name] = {
                        'type': 'mesh',
                        'params': {'mesh_path': geom_mesh_path, 'name': geom_mesh_name, 'position': geom_pos}
                    }

                elif geom_type == "sphere":
                    geom_mesh_size = geom.size[0]  # 球体的大小是半径
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'sphere',
                        'params': {'radius': geom_mesh_size, 'position': geom_pos}
                    }

                elif geom_type == "cylinder":
                    geom_mesh_size = geom.size  # 圆柱体的大小是 [半径, 高度]
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'cylinder',
                        'params': {'radius': geom_mesh_size[0], 'height': geom_mesh_size[1], 'position': geom_pos}
                    }

                elif geom_type == "box":
                    geom_mesh_size = geom.size  # 盒子的大小是 [x, y, z] 维度
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'box',
                        'params': {'extents': geom_mesh_size, 'position': geom_pos}
                    }

                elif geom_type == "capsule":
                    geom_mesh_size = geom.size  # 胶囊的半径储存在 size[0]
                    geom_fromto = geom.fromto  # 从fromto属性获取胶囊两端的坐标
                    from_point = geom_fromto[:3]  # 胶囊起点
                    to_point = geom_fromto[3:]  # 胶囊终点
                    # 计算胶囊的高度（两点之间的距离）
                    height = ((to_point[0] - from_point[0]) ** 2 +

                              (to_point[1] - from_point[1]) ** 2 +

                              (to_point[2] - from_point[2]) ** 2) ** 0.5
                    # 胶囊的参数化描述
                    self.link_mesh_map[body.name][geom.name] = {
                        'type': 'capsule',
                        'params': {
                            'radius': geom_mesh_size[0],  # 胶囊的半径
                            'height': height,  # 胶囊的高度
                            'from': from_point,  # 起点坐标
                            'to': to_point  # 终点坐标
                        }
                    }

                else:
                    raise ValueError(f"Unsupported geometry type {geom_type}.")
        # for body in bodies:
        #     geoms_this_body = body.geom
        #     self.link_mesh_map[body.name] = {}
        #     for geom in geoms_this_body:
        #         geom_type = geom.type
        #         geom_pos = geom.pos
        #
        #         geom_type = "capsule" if geom_type is None else geom_type
        #         geom_pos = [0, 0, 0] or geom_pos
        #
        #         if geom_type == "mesh":
        #             geom_mesh_name = geom.mesh.name
        #             geom_mesh_path = mesh_name_path_map[geom_mesh_name]
        #             self.link_mesh_map[body.name][geom_mesh_name] = geom_mesh_path
        #         elif geom_type == "sphere":
        #             geom_mesh_size = geom.size
        #             sphere_mesh = icosphere(radius=geom_mesh_size, transform=trimesh.transformations.translation_matrix(
        #                 [geom_pos[0], geom_pos[1], geom_pos[2]]))
        #             sphere_mesh.export(f"{mesh_dir}/{body.name}_{geom.name}_sphere.stl")
        #             self.link_mesh_map[body.name][geom.name] = f"{mesh_dir}/{body.name}_{geom.name}_sphere.stl"
        #         elif geom_type == "cylinder":
        #             geom_mesh_size = geom.size
        #             cylinder_mesh = cylinder(radius=geom_mesh_size[0], height=geom_mesh_size[1])
        #             cylinder_mesh.export(f"{mesh_dir}/{body.name}_{geom.name}_cylinder.stl")
        #             self.link_mesh_map[body.name][geom.name] = f"{mesh_dir}/{body.name}_{geom.name}_cylinder.stl"
        #         elif geom_type == "box":
        #             geom_mesh_size = geom.size
        #             box_mesh = box(extents=geom_mesh_size)
        #             box_mesh.export(f"{mesh_dir}/{body.name}_{geom.name}_box.stl")
        #             self.link_mesh_map[body.name][geom.name] = f"{mesh_dir}/{body.name}_{geom.name}_box.stl"
        #         elif geom_type == "capsule":
        #             geom_mesh_size = geom.size
        #             geom_fromto = geom.fromto
        #             geom_fromto_1 = geom_fromto[3:]
        #             geom_fromto_2 = geom_fromto[:3]
        #             height = ((geom_fromto_1[0] - geom_fromto_2[0]) ** 2 + (
        #                     geom_fromto_1[1] - geom_fromto_2[1]) ** 2 + (
        #                               geom_fromto_1[2] - geom_fromto_2[2]) ** 2) ** 0.5
        #             angle = trimesh.transformations.angle_between_vectors(geom_fromto_1, geom_fromto_2)
        #             direction = trimesh.transformations.unit_vector(geom_fromto_2 - geom_fromto_1)
        #             transform = trimesh.transformations.rotation_matrix(angle, direction, point=geom_fromto_1)
        #             capsule_mesh = capsule(radius=geom_mesh_size[0], height=height, transform=transform)
        #             capsule_mesh.export(f"{mesh_dir}/{body.name}_{geom.name}_capsule.stl")
        #             self.link_mesh_map[body.name][geom.name] = f"{mesh_dir}/{body.name}_{geom.name}_capsule.stl"
        #         else:
        #             raise ValueError(f"Unsupported geometry type {geom_type}.")
        return self.link_mesh_map
