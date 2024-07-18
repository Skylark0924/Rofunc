import torch
import trimesh
import glob
import os
import numpy as np
import math


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces + 1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class PandaLayer(torch.nn.Module):
    def __init__(self, device='cpu', mesh_path=os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"):
        # The forward kinematics equations implemented here are from
        super().__init__()
        dir_path = os.path.split(os.path.abspath(__file__))[0]
        self.device = device
        self.mesh_path = mesh_path
        self.meshes = self.load_meshes()

        # self.theta_min = [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895]
        # self.theta_max = [ 2.3093,  1.5133,  2.4937, -0.4461,  2.4800, 4.2094,  2.6895]

        self.theta_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]).to(self.device)
        self.theta_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]).to(self.device)
        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min - self.theta_mid) * 0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max - self.theta_mid) * 0.8 + self.theta_mid
        self.dof = len(self.theta_min)

        # meshes
        self.link0 = self.meshes["link0"][0]
        self.link0_normals = self.meshes["link0"][-1]

        self.link1 = self.meshes["link1"][0]
        self.link1_normals = self.meshes["link1"][-1]

        self.link2 = self.meshes["link2"][0]
        self.link2_normals = self.meshes["link2"][-1]

        self.link3 = self.meshes["link3"][0]
        self.link3_normals = self.meshes["link3"][-1]

        self.link4 = self.meshes["link4"][0]
        self.link4_normals = self.meshes["link4"][-1]

        self.link5 = self.meshes["link5"][0]
        self.link5_normals = self.meshes["link5"][-1]

        self.link6 = self.meshes["link6"][0]
        self.link6_normals = self.meshes["link6"][-1]
        self.link7 = self.meshes["link7"][0]
        self.link7_normals = self.meshes["link7"][-1]
        self.link8 = self.meshes["link8"][0]
        self.link8_normals = self.meshes["link8"][-1]
        self.finger = self.meshes["finger"][0]
        self.finger_normals = self.meshes["finger"][-1]

        # mesh faces
        self.robot_faces = [
            self.meshes["link0"][1], self.meshes["link1"][1], self.meshes["link2"][1],
            self.meshes["link3"][1], self.meshes["link4"][1], self.meshes["link5"][1],
            self.meshes["link6"][1], self.meshes["link7"][1], self.meshes["link8"][1],
            self.meshes["finger"][1]
        ]

        # self.vertices_face_areas = [
        #     self.meshes["link0"][2], self.meshes["link1"][2], self.meshes["link2"][2],
        #     self.meshes["link3"][2], self.meshes["link4"][2], self.meshes["link5"][2],
        #     self.meshes["link6"][2], self.meshes["link7"][2], self.meshes["link8"][2],
        #     self.meshes["finger"][2]
        # ]

        self.num_vertices_per_part = [
            self.meshes["link0"][0].shape[0], self.meshes["link1"][0].shape[0], self.meshes["link2"][0].shape[0],
            self.meshes["link3"][0].shape[0], self.meshes["link4"][0].shape[0], self.meshes["link5"][0].shape[0],
            self.meshes["link6"][0].shape[0], self.meshes["link7"][0].shape[0], self.meshes["link8"][0].shape[0],
            self.meshes["finger"][0].shape[0]
        ]

        self.A0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A3 = torch.tensor(0.0825, dtype=torch.float32, device=device)
        self.A4 = torch.tensor(-0.0825, dtype=torch.float32, device=device)
        self.A5 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.A6 = torch.tensor(0.088, dtype=torch.float32, device=device)
        self.A7 = torch.tensor(0.0, dtype=torch.float32, device=device)

    # def check_normal(self,verterices, normals):
    #     center = np.mean(verterices,axis=0)
    #     verts = torch.from_numpy(verterices-center).float()
    #     normals = torch.from_numpy(normals).float()
    #     cosine = torch.cosine_similarity(verts,normals).float()
    #     normals[cosine<0] = -normals[cosine<0]
    #     return normals

    def load_meshes(self):
        check_normal = False
        # mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/meshes/visual/*.stl"
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}

        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4].split('_')[0]
            mesh = trimesh.load(mesh_file)
            triangle_areas = trimesh.triangles.area(mesh.triangles)
            vert_area_weight = []
            # for i in range(mesh.vertices.shape[0]):
            #     vert_neighour_face = np.where(mesh.faces == i)[0]
            #     vert_area_weight.append(1000000*triangle_areas[vert_neighour_face].mean())
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            meshes[name] = [
                torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                # torch.LongTensor(np.asarray(mesh.faces)).to(self.device),
                mesh.faces,
                # torch.FloatTensor(np.asarray(vert_area_weight)).to(self.device),
                # vert_area_weight,
                # torch.FloatTensor(mesh.vertex_normals)
                torch.cat((torch.FloatTensor(np.array(mesh.vertex_normals)), temp), dim=-1).to(self.device).to(
                    torch.float),
            ]
        return meshes

    def forward(self, pose, theta):
        batch_size = pose.shape[0]
        link0_vertices = self.link0.repeat(batch_size, 1, 1)
        link0_vertices = torch.matmul(pose,
                                      link0_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link0_normals = self.link0_normals.repeat(batch_size, 1, 1)
        link0_normals = torch.matmul(pose,
                                     link0_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        link1_vertices = self.link1.repeat(batch_size, 1, 1)
        T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.333, theta[:, 0], batch_size).float()

        link2_vertices = self.link2.repeat(batch_size, 1, 1)
        T12 = self.forward_kinematics(self.A1, torch.tensor(-np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 1], batch_size).float()
        link3_vertices = self.link3.repeat(batch_size, 1, 1)
        T23 = self.forward_kinematics(self.A2, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0.316, theta[:, 2], batch_size).float()
        link4_vertices = self.link4.repeat(batch_size, 1, 1)
        T34 = self.forward_kinematics(self.A3, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 3], batch_size).float()
        link5_vertices = self.link5.repeat(batch_size, 1, 1)
        T45 = self.forward_kinematics(self.A4, torch.tensor(-np.pi / 2., dtype=torch.float32, device=self.device),
                                      0.384, theta[:, 4], batch_size).float()
        link6_vertices = self.link6.repeat(batch_size, 1, 1)
        T56 = self.forward_kinematics(self.A5, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 5], batch_size).float()
        link7_vertices = self.link7.repeat(batch_size, 1, 1)
        T67 = self.forward_kinematics(self.A6, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 6], batch_size).float()
        link8_vertices = self.link8.repeat(batch_size, 1, 1)
        T78 = self.forward_kinematics(self.A7, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.107, torch.tensor(-np.pi / 4, dtype=torch.float32, device=self.device),
                                      batch_size).float()
        # finger_vertices = self.finger.repeat(batch_size, 1, 1)

        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        link1_vertices = torch.matmul(
            pose_to_T01,
            link1_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link1_normals = self.link1_normals.repeat(batch_size, 1, 1)
        link1_normals = torch.matmul(pose_to_T01,
                                     link1_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        link2_vertices = torch.matmul(
            pose_to_T12,
            link2_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link2_normals = self.link2_normals.repeat(batch_size, 1, 1)
        link2_normals = torch.matmul(pose_to_T12,
                                     link2_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        link3_vertices = torch.matmul(
            pose_to_T23,
            link3_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link3_normals = self.link3_normals.repeat(batch_size, 1, 1)
        link3_normals = torch.matmul(pose_to_T23,
                                     link3_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        link4_vertices = torch.matmul(
            pose_to_T34,
            link4_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link4_normals = self.link4_normals.repeat(batch_size, 1, 1)
        link4_normals = torch.matmul(pose_to_T34,
                                     link4_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        link5_vertices = torch.matmul(
            pose_to_T45,
            link5_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link5_normals = self.link5_normals.repeat(batch_size, 1, 1)
        link5_normals = torch.matmul(pose_to_T45,
                                     link5_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T56 = torch.matmul(pose_to_T45, T56)
        link6_vertices = torch.matmul(
            pose_to_T56,
            link6_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link6_normals = self.link6_normals.repeat(batch_size, 1, 1)
        link6_normals = torch.matmul(pose_to_T56,
                                     link6_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T67 = torch.matmul(pose_to_T56, T67)
        link7_vertices = torch.matmul(
            pose_to_T67,
            link7_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link7_normals = self.link7_normals.repeat(batch_size, 1, 1)
        link7_normals = torch.matmul(pose_to_T67,
                                     link7_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        pose_to_T78 = torch.matmul(pose_to_T67, T78)
        link8_vertices = torch.matmul(
            pose_to_T78,
            link8_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        link8_normals = self.link8_normals.repeat(batch_size, 1, 1)
        link8_normals = torch.matmul(pose_to_T78,
                                     link8_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return [link0_vertices, link1_vertices, link2_vertices, \
                link3_vertices, link4_vertices, link5_vertices, \
                link6_vertices, link7_vertices, link8_vertices, \
                link0_normals, link1_normals, link2_normals, \
                link3_normals, link4_normals, link5_normals, \
                link6_normals, link7_normals, link8_normals]

    def get_transformations_each_link(self, pose, theta):
        batch_size = pose.shape[0]
        T01 = self.forward_kinematics(self.A0, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.333, theta[:, 0], batch_size).float()

        # link2_vertices = self.link2.repeat(batch_size, 1, 1)
        T12 = self.forward_kinematics(self.A1, torch.tensor(-np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 1], batch_size).float()
        # link3_vertices = self.link3.repeat(batch_size, 1, 1)
        T23 = self.forward_kinematics(self.A2, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0.316, theta[:, 2], batch_size).float()
        # link4_vertices = self.link4.repeat(batch_size, 1, 1)
        T34 = self.forward_kinematics(self.A3, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 3], batch_size).float()
        # link5_vertices = self.link5.repeat(batch_size, 1, 1)
        T45 = self.forward_kinematics(self.A4, torch.tensor(-np.pi / 2., dtype=torch.float32, device=self.device),
                                      0.384, theta[:, 4], batch_size).float()
        # link6_vertices = self.link6.repeat(batch_size, 1, 1)
        T56 = self.forward_kinematics(self.A5, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 5], batch_size).float()
        # link7_vertices = self.link7.repeat(batch_size, 1, 1)
        T67 = self.forward_kinematics(self.A6, torch.tensor(np.pi / 2., dtype=torch.float32, device=self.device),
                                      0, theta[:, 6], batch_size).float()
        # link8_vertices = self.link8.repeat(batch_size, 1, 1)
        T78 = self.forward_kinematics(self.A7, torch.tensor(0, dtype=torch.float32, device=self.device),
                                      0.107, torch.tensor(-np.pi / 4, dtype=torch.float32, device=self.device),
                                      batch_size).float()
        # finger_vertices = self.finger.repeat(batch_size, 1, 1)
        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        pose_to_T56 = torch.matmul(pose_to_T45, T56)
        pose_to_T67 = torch.matmul(pose_to_T56, T67)
        pose_to_T78 = torch.matmul(pose_to_T67, T78)

        return [pose_to_Tw0, pose_to_T01, pose_to_T12, pose_to_T23, pose_to_T34, pose_to_T45, pose_to_T56, pose_to_T67,
                pose_to_T78]

    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, 4, 4), device=self.device)
        l_1_to_l[:, 0, 0] = c_theta
        l_1_to_l[:, 0, 1] = -s_theta
        l_1_to_l[:, 0, 3] = A
        l_1_to_l[:, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, 1, 2] = -s_alpha
        l_1_to_l[:, 1, 3] = -s_alpha * D
        l_1_to_l[:, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, 2, 2] = c_alpha
        l_1_to_l[:, 2, 3] = c_alpha * D
        l_1_to_l[:, 3, 3] = 1
        return l_1_to_l

    def get_robot_mesh(self, vertices_list, faces):
        link0_verts = vertices_list[0]
        link0_faces = faces[0]

        link1_verts = vertices_list[1]
        link1_faces = faces[1]

        link2_verts = vertices_list[2]
        link2_faces = faces[2]

        link3_verts = vertices_list[3]
        link3_faces = faces[3]

        link4_verts = vertices_list[4]
        link4_faces = faces[4]

        link5_verts = vertices_list[5]
        link5_faces = faces[5]

        link6_verts = vertices_list[6]
        link6_faces = faces[6]

        link7_verts = vertices_list[7]
        link7_faces = faces[7]

        link8_verts = vertices_list[8]
        link8_faces = faces[8]

        link0_mesh = trimesh.Trimesh(link0_verts, link0_faces)
        # link0_mesh.visual.face_colors = [150,150,150]
        link1_mesh = trimesh.Trimesh(link1_verts, link1_faces)
        # link1_mesh.visual.face_colors = [150,150,150]
        link2_mesh = trimesh.Trimesh(link2_verts, link2_faces)
        # link2_mesh.visual.face_colors = [150,150,150]
        link3_mesh = trimesh.Trimesh(link3_verts, link3_faces)
        # link3_mesh.visual.face_colors = [150,150,150]
        link4_mesh = trimesh.Trimesh(link4_verts, link4_faces)
        # link4_mesh.visual.face_colors = [150,150,150]
        link5_mesh = trimesh.Trimesh(link5_verts, link5_faces)
        # link5_mesh.visual.face_colors = [250,150,150]
        link6_mesh = trimesh.Trimesh(link6_verts, link6_faces)
        # link6_mesh.visual.face_colors = [250,150,150]
        link7_mesh = trimesh.Trimesh(link7_verts, link7_faces)
        # link7_mesh.visual.face_colors = [250,150,150]
        link8_mesh = trimesh.Trimesh(link8_verts, link8_faces)
        # link8_mesh.visual.face_colors = [250,150,150]

        robot_mesh = [
            link0_mesh,
            link1_mesh,
            link2_mesh,
            link3_mesh,
            link4_mesh,
            link5_mesh,
            link6_mesh,
            link7_mesh,
            link8_mesh
        ]
        # robot_mesh = np.sum(robot_mesh)
        return robot_mesh

    def get_forward_robot_mesh(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        vertices_list = [[
            outputs[0][i].detach().cpu().numpy(),
            outputs[1][i].detach().cpu().numpy(),
            outputs[2][i].detach().cpu().numpy(),
            outputs[3][i].detach().cpu().numpy(),
            outputs[4][i].detach().cpu().numpy(),
            outputs[5][i].detach().cpu().numpy(),
            outputs[6][i].detach().cpu().numpy(),
            outputs[7][i].detach().cpu().numpy(),
            outputs[8][i].detach().cpu().numpy()] for i in range(batch_size)]
        mesh = [self.get_robot_mesh(vertices, self.robot_faces) for vertices in vertices_list]

        return mesh

    def get_forward_vertices(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        robot_vertices = torch.cat((
            outputs[0].view(batch_size, -1, 3),
            outputs[1].view(batch_size, -1, 3),
            outputs[2].view(batch_size, -1, 3),
            outputs[3].view(batch_size, -1, 3),
            outputs[4].view(batch_size, -1, 3),
            outputs[5].view(batch_size, -1, 3),
            outputs[6].view(batch_size, -1, 3),
            outputs[7].view(batch_size, -1, 3),
            outputs[8].view(batch_size, -1, 3)), 1)  # .squeeze()

        robot_vertices_normal = torch.cat((
            outputs[9].view(batch_size, -1, 3),
            outputs[10].view(batch_size, -1, 3),
            outputs[11].view(batch_size, -1, 3),
            outputs[12].view(batch_size, -1, 3),
            outputs[13].view(batch_size, -1, 3),
            outputs[14].view(batch_size, -1, 3),
            outputs[15].view(batch_size, -1, 3),
            outputs[16].view(batch_size, -1, 3),
            outputs[17].view(batch_size, -1, 3)), 1)  # .squeeze()

        return robot_vertices, robot_vertices_normal


if __name__ == "__main__":
    device = 'cuda'
    panda = PandaLayer(device).to(device)
    scene = trimesh.Scene()

    # show robot
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = torch.tensor([0.0, -0.3, 0, -0.2, 0, 2.0, np.pi / 4]).float().to(device).reshape(-1, 7)
    robot_mesh = panda.get_forward_robot_mesh(pose, theta)
    robot_mesh = np.sum(robot_mesh)
    robot_mesh.show()
