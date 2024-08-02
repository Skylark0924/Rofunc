import math
import numpy as np
import torch
import trimesh

from rofunc.utils.robolab.rdf import rdf


class BBOPlanner:
    def __init__(self, args, rdf_model, box_size, box_pos, box_rotation):
        """
        Bimanual box carrying using Robot distance field (RDF)

        :param args: the arguments
        :param rdf_model: the RDF model
        :param box_size: the size of the box
        :param box_pos: the position of the box
        :param box_rotation: the rotation of the box
        """
        self.n_func = args.n_func
        self.domain_min = args.domain_min
        self.domain_max = args.domain_max
        self.device = args.device
        self.box_size = box_size
        self.box_pos = box_pos
        self.box_rotation = box_rotation
        self.rdf_model = rdf_model

        self.rdf_bp = rdf.RDF(args)
        self.robot = self.rdf_bp.robot
        self.theta_max = self.rdf_bp.robot.joint_limit_max.to(self.device)
        self.theta_min = self.rdf_bp.robot.joint_limit_min.to(self.device)

        self.object_mesh = self.load_box_object()
        self.object_internal_points = self.compute_internal_points(num=5, use_surface_points=False)
        self.contact_points = self.compute_contact_points()

    def load_box_object(self):
        """
        load a box with size .3*.3*.3 based on urdf path
        """
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
        # surface contact
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

    def reaching_cost(self, joint_value, p, base_trans):
        batch_size = joint_value.shape[0]
        sdf, joint_grad = self.rdf_bp.get_whole_body_sdf_with_joints_grad_batch(p, joint_value, self.rdf_model,
                                                                                base_trans=base_trans)
        sdf, joint_grad = sdf.squeeze(0), joint_grad.squeeze(0)
        # reaching multiple points
        dist = sdf.mean(dim=1)
        cost = (sdf ** 2).mean(dim=1)
        grad = (2 * sdf.unsqueeze(-1).expand_as(joint_grad) * joint_grad).mean(dim=1)
        return cost.reshape(batch_size, 1), grad.reshape(batch_size, 1, self.rdf_bp.robot.num_joint), dist.reshape(
            batch_size, 1)

    def collision_cost(self, joint_value, p, base_trans):
        batch_size = joint_value.shape[0]
        sdf, joint_grad = self.rdf_bp.get_whole_body_sdf_with_joints_grad_batch(p, joint_value, self.rdf_model,
                                                                                base_trans=base_trans)
        sdf, joint_grad = sdf.squeeze(), joint_grad.squeeze()
        coll_mask = sdf < 0
        sdf[~coll_mask] = 0
        joint_grad[~coll_mask] = 0
        cost = (sdf ** 2).mean(dim=1)
        grad = (2 * sdf.unsqueeze(-1).expand_as(joint_grad) * joint_grad).mean(dim=1)
        penetration = -sdf.sum(dim=1)
        return cost.reshape(batch_size, 1), grad.reshape(batch_size, 1,
                                                         self.rdf_bp.robot.num_joint), penetration.reshape(batch_size,
                                                                                                           1)

    def normal_cost(self, joint_value, p, tgt_normal, base_trans):
        batch_size = joint_value.shape[0]
        delta = 0.001
        normal = self.rdf_bp.get_whole_body_normal_with_joints_grad_batch(p, joint_value, self.rdf_model,
                                                                          base_trans=base_trans)
        tgt_normal = tgt_normal.unsqueeze(1).unsqueeze(0).expand_as(normal)
        cosine_similarities = 1 - torch.sum(normal * tgt_normal, dim=-1)
        cost = cosine_similarities[:, :, 0].mean(dim=1)
        grad = ((cosine_similarities[:, :, 1:] - cosine_similarities[:, :, :1]) / delta).mean(dim=1)
        return cost.reshape(batch_size, 1), grad.reshape(batch_size, 1, self.rdf_bp.robot.num_joint)

    def limit_angles(self, joint_value):
        joint_value = joint_value % (2 * math.pi)  # Wrap angles between 0 and 2*pi
        joint_value[joint_value > math.pi] -= 2 * math.pi  # Shift angles to -pi to pi range
        theta_5_mask = (joint_value[:, 5] > -math.pi) * (joint_value[:, 5] < self.theta_max[5] - 2 * math.pi)
        joint_value[:, 5][theta_5_mask] = joint_value[:, 5][theta_5_mask] + 2 * math.pi
        return joint_value

    def joint_limits_cost(self, joint_value, theta_max, theta_min):
        batch_size = joint_value.shape[0]
        theta_max = self.theta_max.unsqueeze(0).expand(batch_size, -1)
        theta_min = self.theta_min.unsqueeze(0).expand(batch_size, -1)
        # print(joint_value)
        cost = torch.sum((joint_value - theta_max).clamp(min=0) ** 2, dim=1) + torch.sum(
            (theta_min - joint_value).clamp(min=0) ** 2, dim=1)
        grad = 2 * ((joint_value - theta_max).clamp(min=0) - (theta_min - joint_value).clamp(min=0))
        return cost.reshape(batch_size, 1), grad.reshape(batch_size, 1, self.rdf_bp.robot.num_joint)

    def middle_joint_cost(self, joint_value, theta_mid):
        batch_size = joint_value.shape[0]
        theta_mid = theta_mid.expand(batch_size, -1)
        cost = torch.sum((joint_value - theta_mid) ** 2, dim=1)
        grad = 2 * (joint_value - theta_mid)
        # grad = torch.nn.functional.normalize(grad,dim=1)
        return cost.reshape(batch_size, 1), grad.reshape(batch_size, 1, self.rdf_bp.robot.num_joint)

    def optimizer(self, p, n, theta_mid, base_trans=None, batch=64):
        joint_value = self.theta_min + torch.rand(batch, self.rdf_bp.robot.num_joint).to(self.device) * (
                self.theta_max - self.theta_min)
        valid_theta_list = []
        num_accept = 0
        while True:
            c_reaching, J_reaching, dist = self.reaching_cost(joint_value, p, base_trans=base_trans)
            c_collision, J_collision, penetration = self.collision_cost(joint_value, self.object_internal_points,
                                                                        base_trans=base_trans)
            c_normal, J_normal = self.normal_cost(joint_value, p, n, base_trans=base_trans)
            c_joints_limits, J_joints_limits = self.joint_limits_cost(joint_value, self.theta_max, self.theta_min)
            c_joints_middle, J_joints_middle = self.middle_joint_cost(joint_value, theta_mid)
            c = torch.cat([c_reaching * 10.0, c_collision * 1.0, c_joints_limits * 10.0, c_joints_middle * 0.1], dim=1)
            J = torch.cat([J_reaching * 1.0, J_collision * 1.0, J_joints_limits * 1.0, J_joints_middle], dim=1)
            # c = torch.cat([c_normal],dim=1)
            # J = torch.cat([J_normal],dim=1)
            # print('c',c)
            joint_accept = ((joint_value < self.theta_max).all(dim=1) * (joint_value > self.theta_min).all(
                dim=1)).unsqueeze(1)
            # accept = (dist < 0.005) * (penetration < 0.01) * (c_normal < 0.2) * joint_accept
            accept = (dist < 0.005) * (penetration < 0.01) * joint_accept
            # accept = (c_reaching<0.01) * (c_normal<0.1)
            if accept.sum() > 0:
                num_accept += accept.sum()
                print('num_accept:', num_accept)
                accept = accept.squeeze()
                theta_accept = joint_value[accept]
                # print('theta_accept',theta_accept)
                cost_accept = c[accept]
                # print('cost_accept',cost_accept)
                joint_value = joint_value[~accept]
                for (i, th_a) in enumerate(theta_accept):
                    valid_theta_list.append(th_a)

                d_theta = (torch.matmul(torch.linalg.pinv(-J), c.unsqueeze(-1)) * 10. * 0.01)[:, :, 0]  # Gauss-Newton
                # d_theta = torch.clamp(d_theta_l, -0.05, 0.05)
                joint_value += d_theta[~accept]  # Update state
                joint_value = self.limit_angles(joint_value)
                if len(valid_theta_list) >= 3:
                    valid_theta_list = valid_theta_list[:10]
                    break
            else:
                d_theta = (torch.matmul(torch.linalg.pinv(-J), c.unsqueeze(-1)) * 10. * 0.01)[:, :, 0]  # Gauss-Newton
                # d_theta = torch.clamp(d_theta_l, -0.05, 0.05)
                joint_value += d_theta  # Update state
                joint_value = self.limit_angles(joint_value)
        valid_theta_list = torch.cat(valid_theta_list, dim=0).reshape(-1, self.rdf_bp.robot.num_joint)
        return valid_theta_list
