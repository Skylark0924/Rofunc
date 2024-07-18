import torch
import trimesh
import numpy as np
import math
from panda_layer.panda_layer import PandaLayer
import bf_sdf

import rofunc as rf


class BBOPlanner():
    def __init__(self, n_func, domain_min, domain_max, robot, device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        self.robot = robot
        self.theta_max = self.robot.theta_max_soft
        self.theta_min = self.robot.theta_min_soft
        self.bp_sdf = bf_sdf.BPSDF(n_func, domain_min, domain_max, robot, device)
        self.model = torch.load(f'models/BP_8.pt')
        self.object_mesh = self.load_box_object()
        self.object_internal_points = self.compute_internal_points(num=5, use_surface_points=False)
        self.contact_points = self.compute_contact_points()
        self.pose_l = torch.from_numpy(np.identity(4)).to(self.device).reshape(-1, 4, 4).float()
        self.pose_r = torch.from_numpy(np.identity(4)).to(self.device).reshape(-1, 4, 4).float()

        self.pose_l[0, :3, :3] = torch.tensor(
            rf.robolab.homo_matrix_from_quaternion([-0.436865, 0.49775, 0.054428, 0.747283])[:3, :3]).to(self.device)
        self.pose_r[0, :3, :3] = torch.tensor(
            rf.robolab.homo_matrix_from_quaternion([0.436865, 0.49775, -0.054428, 0.747283])[:3, :3]).to(self.device)

        # self.pose_l[0, :3, :3] = torch.tensor(
        #     rf.robolab.homo_matrix_from_quaternion([-0.437, 0.747, -0.054, -0.498, ])[:3, :3]).to(self.device)
        # self.pose_r[0, :3, :3] = torch.tensor(
        #     rf.robolab.homo_matrix_from_quaternion([0.437, 0.747, -0.054, -0.498, ])[:3, :3]).to(self.device)

        # self.pose_r[0, :3, 3] = torch.tensor([0.6198, -0.7636, 0]).to(self.device)

        self.pose_l[0, :3, 3] = torch.tensor([0.396519, 0.07, 0.644388]).to(self.device)
        self.pose_r[0, :3, 3] = torch.tensor([0.396519, -0.07, 0.644388]).to(self.device)
        # self.pose_l[0, :3, 3] = torch.tensor([0.398, -0.07, 1.2]).to(self.device)
        # self.pose_r[0, :3, 3] = torch.tensor([0.398, 0.07, 1.2]).to(self.device)
        # self.mid_l = torch.tensor([ 0.42704887,  0.17838557,  0.10469598, -1.74670609, -0.05181788 , 2.16040988,-2.29006758]).reshape(-1,7).to(self.device)
        # self.mid_r = torch.tensor( [ 0.79520689 , 0.37705809, -0.01953359, -1.50133787,  0.14086509,  1.87535585, 1.05259796]).reshape(-1,7).to(self.device)

    def load_box_object(self):
        """load a box with size .3*.3*.3 based on urdf path"""
        # self.box_pos = np.array([0.7934301890820722, 0.05027181819137394, 0.2246743147850761])

        self.box_pos = np.array([0.7934301890820722, 0.05027181819137394, 0.2246743147850761])
        self.box_size = np.array([0.38, 0.24, 0.26])
        # self.box_rotation = np.array([[0.707, 0.707, 0],
        #                               [-0.707, 0.707, 0],
        #                               [0, 0, 1]])

        # self.box_rotation = np.array([[1, 0, 0],
        #                               [0, 1, 0],
        #                               [0, 0, 1]])

        self.box_rotation = np.array([[0,  1.57, 0],
                                      [-1.57, 0, 0],
                                      [0, 0, 1]])
        #  internal points
        mesh = trimesh.creation.box(self.box_size)
        mat = np.eye(4)
        mat[:3, 3] = self.box_pos
        mat[:3, :3] = self.box_rotation
        mesh.apply_transform(mat)
        return mesh

    def compute_internal_points(self, num=10, use_surface_points=False):
        if use_surface_points:
            grid_x, grid_z = torch.meshgrid(
                torch.linspace(-self.box_size[0] / 2.0, self.box_size[0] / 2.0, num).to(self.device),
                torch.linspace(-self.box_size[2] / 2.0, self.box_size[2] / 2.0, num).to(self.device))
            grid_y = torch.zeros_like(grid_x) + self.box_size[1] / 2.0
            iternal_points_l = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)], dim=1)
            iternal_points_r = torch.cat([grid_x.reshape(-1, 1), -grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)], dim=1)
            iternal_points = torch.cat([iternal_points_l, iternal_points_r], dim=0).float()

        else:
            grid_x, grid_y, grid_z = torch.meshgrid(
                torch.linspace(-self.box_size[0] / 2.0, self.box_size[0] / 2.0, num).to(self.device),
                torch.linspace(-self.box_size[1] / 2.0, self.box_size[1] / 2.0, num).to(self.device),
                torch.linspace(-self.box_size[2] / 2.0, self.box_size[2] / 2.0, num).to(self.device))

            iternal_points = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)],
                                       dim=1).float()
        iternal_points = torch.mm(iternal_points,
                                  torch.from_numpy(self.box_rotation).float().to(self.device).T) + torch.from_numpy(
            self.box_pos).float().to(self.device)
        return iternal_points

    def compute_contact_points(self):
        # # surface contact
        p_y = torch.linspace(-0.03, 0.03, 5, device=self.device)
        p_x = torch.ones_like(p_y).float().to(self.device) * (self.box_size[0] / 2)
        p_z_h = torch.ones_like(p_y).float().to(self.device) * 0.005
        p_z_l = -p_z_h
        p_l_h = torch.cat([-p_x.reshape(-1, 1), p_y.reshape(-1, 1), p_z_h.reshape(-1, 1)], dim=1).float()
        p_l_l = torch.cat([-p_x.reshape(-1, 1), p_y.reshape(-1, 1), p_z_l.reshape(-1, 1)], dim=1).float()
        p_l = torch.cat([p_l_h, p_l_l], dim=0)
        p_l = torch.mm(p_l, torch.from_numpy(self.box_rotation).float().to(self.device).T) + torch.from_numpy(
            self.box_pos).float().to(self.device)
        p_r_h = torch.cat([p_x.reshape(-1, 1), p_y.reshape(-1, 1), p_z_h.reshape(-1, 1)], dim=1).float()
        p_r_l = torch.cat([p_x.reshape(-1, 1), p_y.reshape(-1, 1), p_z_l.reshape(-1, 1)], dim=1).float()
        p_r = torch.cat([p_r_h, p_r_l], dim=0)
        p_r = torch.mm(p_r, torch.from_numpy(self.box_rotation).float().to(self.device).T) + torch.from_numpy(
            self.box_pos).float().to(self.device)
        n_r = torch.tensor([[-0.707, 0.707, 0]]).expand(len(p_r), 3).float().to(self.device)
        n_l = torch.tensor([[0.707, -0.707, 0]]).expand(len(p_l), 3).float().to(self.device)
        return p_l, p_r, n_l, n_r

    def reaching_cost(self, pose, theta, p):
        B = theta.shape[0]
        sdf, joint_grad = self.bp_sdf.get_whole_body_sdf_with_joints_grad_batch(p, pose, theta, self.model,
                                                                                used_links=[5, 6, 7, 8])
        sdf, joint_grad = sdf.squeeze(0), joint_grad.squeeze(0)
        # reaching multiple points
        dist = sdf.mean(dim=1)
        cost = (sdf ** 2).mean(dim=1)
        grad = (2 * sdf.unsqueeze(-1).expand_as(joint_grad) * joint_grad).mean(dim=1)
        return cost.reshape(B, 1), grad.reshape(B, 1, 7), dist.reshape(B, 1)

    def collision_cost(self, pose, theta, p):
        B = theta.shape[0]
        sdf, joint_grad = self.bp_sdf.get_whole_body_sdf_with_joints_grad_batch(p, pose, theta, self.model,
                                                                                used_links=[5, 6, 7, 8])
        sdf, joint_grad = sdf.squeeze(), joint_grad.squeeze()
        coll_mask = sdf < 0
        sdf[~coll_mask] = 0
        joint_grad[~coll_mask] = 0
        cost = (sdf ** 2).mean(dim=1)
        grad = (2 * sdf.unsqueeze(-1).expand_as(joint_grad) * joint_grad).mean(dim=1)
        penetration = -sdf.sum(dim=1)
        return cost.reshape(B, 1), grad.reshape(B, 1, 7), penetration.reshape(B, 1)

    def normal_cost(self, pose, theta, p, tgt_normal):
        B = theta.shape[0]
        delta = 0.001
        normal = self.bp_sdf.get_whole_body_normal_with_joints_grad_batch(p, pose, theta, self.model,
                                                                          used_links=[5, 6, 7, 8])
        tgt_normal = tgt_normal.unsqueeze(1).unsqueeze(0).expand_as(normal)
        cosine_similarities = 1 - torch.sum(normal * tgt_normal, dim=-1)
        cost = cosine_similarities[:, :, 0].mean(dim=1)
        grad = ((cosine_similarities[:, :, 1:] - cosine_similarities[:, :, :1]) / delta).mean(dim=1)
        return cost.reshape(B, 1), grad.reshape(B, 1, 7)

    def limit_angles(self, theta):
        theta = theta % (2 * math.pi)  # Wrap angles between 0 and 2*pi
        theta[theta > math.pi] -= 2 * math.pi  # Shift angles to -pi to pi range
        theta_5_mask = (theta[:, 5] > -math.pi) * (theta[:, 5] < self.theta_max[5] - 2 * math.pi)
        theta[:, 5][theta_5_mask] = theta[:, 5][theta_5_mask] + 2 * math.pi
        return theta

    def joint_limits_cost(self, theta, theta_max, theta_min):
        B = theta.shape[0]
        theta_max = self.theta_max.unsqueeze(0).expand(B, -1)
        theta_min = self.theta_min.unsqueeze(0).expand(B, -1)
        # print(theta)
        cost = torch.sum((theta - theta_max).clamp(min=0) ** 2, dim=1) + torch.sum(
            (theta_min - theta).clamp(min=0) ** 2, dim=1)
        grad = 2 * ((theta - theta_max).clamp(min=0) - (theta_min - theta).clamp(min=0))
        return cost.reshape(B, 1), grad.reshape(B, 1, 7)

    def middle_joint_cost(self, theta, theta_mid):
        B = theta.shape[0]
        theta_mid = theta_mid.expand(B, -1)
        cost = torch.sum((theta - theta_mid) ** 2, dim=1)
        grad = 2 * (theta - theta_mid)
        # grad = torch.nn.functional.normalize(grad,dim=1)
        return cost.reshape(B, 1), grad.reshape(B, 1, 7)

    def optimizer(self, pose, p, n, theta_mid, batch=64):
        theta = self.theta_min + torch.rand(batch, 7).to(self.device) * (self.theta_max - self.theta_min)
        valid_theta_list = []
        num_accept = 0
        while 1:
            c_reaching, J_reaching, dist = self.reaching_cost(pose, theta, p)
            c_collision, J_collision, penetration = self.collision_cost(pose, theta, self.object_internal_points)
            c_normal, J_normal = self.normal_cost(pose, theta, p, n)
            c_joints_limits, J_joints_limits = self.joint_limits_cost(theta, self.theta_max, self.theta_min)
            c_joints_middle, J_joints_middle = self.middle_joint_cost(theta, theta_mid)
            c = torch.cat([c_reaching * 10.0, c_collision * 1.0, c_joints_limits * 10.0, c_joints_middle * 0.1], dim=1)
            J = torch.cat([J_reaching * 1.0, J_collision * 1.0, J_joints_limits * 1.0, J_joints_middle], dim=1)
            # c = torch.cat([c_normal],dim=1)
            # J = torch.cat([J_normal],dim=1)
            # print('c',c)
            joint_accept = ((theta < self.theta_max).all(dim=1) * (theta > self.theta_min).all(dim=1)).unsqueeze(1)
            accept = (dist < 0.005) * (penetration < 0.01) * (c_normal < 0.2) * joint_accept
            # accept = (c_reaching<0.01) * (c_normal<0.1)
            if accept.sum() > 0:
                num_accept += accept.sum()
                print('num_accept:', num_accept)
                accept = accept.squeeze()
                theta_accept = theta[accept]
                # print('theta_accept',theta_accept)
                cost_accept = c[accept]
                # print('cost_accept',cost_accept)
                theta = theta[~accept]
                for (i, th_a) in enumerate(theta_accept):
                    valid_theta_list.append(th_a)

                d_theta = (torch.matmul(torch.linalg.pinv(-J), c.unsqueeze(-1)) * 10. * 0.01)[:, :, 0]  # Gauss-Newton
                # d_theta = torch.clamp(d_theta_l, -0.05, 0.05)
                theta += d_theta[~accept]  # Update state
                theta = self.limit_angles(theta)
                if len(valid_theta_list) >= 3:
                    valid_theta_list = valid_theta_list[:10]
                    break
            else:
                d_theta = (torch.matmul(torch.linalg.pinv(-J), c.unsqueeze(-1)) * 10. * 0.01)[:, :, 0]  # Gauss-Newton
                # d_theta = torch.clamp(d_theta_l, -0.05, 0.05)
                theta += d_theta  # Update state
                theta = self.limit_angles(theta)
        valid_theta_list = torch.cat(valid_theta_list, dim=0).reshape(-1, 7)
        return valid_theta_list


if __name__ == '__main__':
    device = 'cuda'
    panda = PandaLayer(device, mesh_path="panda_layer/meshes/visual/*.stl")
    bbo_planner = BBOPlanner(n_func=8, domain_min=-1.0, domain_max=1.0, robot=panda, device=device)

    contact_points = bbo_planner.contact_points
    p_l, p_r, n_l, n_r = contact_points[0], contact_points[1], contact_points[2], contact_points[3]
    pose_l, pose_r = bbo_planner.pose_l, bbo_planner.pose_r
    mid_l = torch.tensor(
        [0.42704887, 0.17838557, 0.10469598, -1.74670609, -0.05181788, 2.16040988, -2.29006758]).reshape(-1, 7).to(
        device)
    mid_r = torch.tensor(
        [0.79520689, 0.37705809, -0.01953359, -1.50133787, 0.14086509, 1.87535585, 1.05259796]).reshape(-1, 7).to(
        device)

    # planning for both arm
    theta_left = bbo_planner.optimizer(bbo_planner.pose_l, p_l, n_l, mid_l, batch=64)
    theta_right = bbo_planner.optimizer(bbo_planner.pose_r, p_r, n_r, mid_r, batch=64)
    joint_conf = {
        'theta_left': theta_left,
        'theta_right': theta_right
    }
    torch.save(joint_conf, 'joint_conf.pt')

    # load planned joint conf
    data = torch.load('joint_conf.pt')
    theta_left = data['theta_left']
    theta_right = data['theta_right']
    print('theta_left', theta_left.shape, 'theta_right', theta_right.shape)

    # visualize planning results
    scene = trimesh.Scene()
    pc1 = trimesh.PointCloud(bbo_planner.object_internal_points.detach().cpu().numpy(), colors=[0, 255, 0])
    pc2 = trimesh.PointCloud(p_l.detach().cpu().numpy(), colors=[255, 0, 0])
    pc3 = trimesh.PointCloud(p_r.detach().cpu().numpy(), colors=[255, 0, 0])
    scene.add_geometry([pc1, pc2, pc3])
    # mesh.visual.face_colors = [200,200,200,255]
    scene.add_geometry(bbo_planner.object_mesh)

    # # visualize the final joint configuration
    for t_l, t_r in zip(theta_left, theta_right):
        print('t left:', t_l)
        robot_l = panda.get_forward_robot_mesh(pose_l, t_l.reshape(-1, 7))[0]
        robot_l = np.sum(robot_l)
        robot_l.visual.face_colors = [150, 150, 200, 200]
        scene.add_geometry(robot_l, node_name='robot_l')

        print('t right:', t_r)
        robot_r = panda.get_forward_robot_mesh(pose_r, t_r.reshape(-1, 7))[0]
        robot_r = np.sum(robot_r)
        robot_r.visual.face_colors = [150, 200, 150, 200]
        scene.add_geometry(robot_r, node_name='robot_r')
        scene.show()

        scene.delete_geometry('robot_l')
        scene.delete_geometry('robot_r')
