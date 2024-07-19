import torch
import os
import numpy as np
import yaml
from panda_layer.panda_layer import PandaLayer
from rofunc.utils.robolab.rdf import utils

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class SphereSDF():
    def __init__(self, device):
        self.device = device
        self.panda = PandaLayer(self.device)
        self.conf = self.load_conf()
        self.r, self.c = self.get_sphere_param(self.conf)

    def load_conf(self):
        with open(os.path.join(CUR_DIR, 'panda_layer/franka_sphere.yaml'), 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)['collision_spheres']
        return conf

    def get_sphere_param(self, sphere_conf):
        rs, cs = [], []
        for i, k in enumerate(sphere_conf.keys()):
            sphere_each_link = sphere_conf[k]
            r_list, c_list = [], []
            for sphere in sphere_each_link:
                r_list.append(torch.tensor(sphere['radius']).unsqueeze(0).to(self.device))
                c_list.append(torch.tensor(sphere['center']).unsqueeze(0).to(self.device))
            radius = torch.cat(r_list)
            center = torch.cat(c_list)
            rs.append(radius)
            cs.append(center)
        rs = torch.cat(rs, dim=0)
        # cs = torch.cat(cs, dim=0)
        return rs, cs

    def get_sdf(self, x, pose, theta, rs, cs):
        B, N = theta.shape[0], x.shape[0]
        pose = pose.reshape(-1, 4, 4).expand(len(theta), 4, 4).float()
        trans = self.panda.get_transformations_each_link(pose, theta)
        c_list = []
        for c, t in zip(cs, trans):
            trans_c = utils.transform_points(c, t, self.device)
            c_list.append(trans_c)
        c = torch.cat(c_list, dim=1)
        N_s = c.shape[1]
        x = x.unsqueeze(0).expand(B, N, 3)
        x_ = x.unsqueeze(2).expand(B, N, N_s, 3)
        c_ = c.unsqueeze(1).expand(B, N, N_s, 3)

        dist = torch.norm(x_ - c_, dim=-1) - rs.unsqueeze(0).unsqueeze(0).expand(B, N, N_s)
        sdf, idx = dist.min(dim=-1)
        c_idx = c_.gather(2, idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, N_s, 3))[:, :, 0, :]
        grad = torch.nn.functional.normalize(c_idx - x, dim=-1)
        return sdf, grad

    def get_sdf_with_joint_grad(self, x, pose, theta, rs, cs, delta=0.001):
        # t0 = time.time()
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7,
                                                                                                7) * delta).reshape(B,
                                                                                                                    -1,
                                                                                                                    7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.expand(B * 8, 4, 4)
        # t1 = time.time()
        sdf, _ = self.get_sdf(x, pose, theta, rs, cs)
        sdf = sdf.reshape(B, 8, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        # t2 = time.time()
        # print(t2-t1,t1-t0)
        return sdf[:, 0, :], d_sdf.transpose(1, 2)

    def get_sdf_normal_with_joint_grad(self, x, pose, theta, rs, cs, delta=0.001):
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7,
                                                                                                7) * delta).reshape(B,
                                                                                                                    -1,
                                                                                                                    7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.expand(B * 8, 4, 4)
        sdf, normal = self.get_sdf(x, pose, theta, rs, cs)
        normal = normal.reshape(B, 8, -1, 3).transpose(1, 2)
        return normal  # normal size: (B,N,8,3) normal[:,:,0,:] origin normal vector normakl[:,:,1:,:] derivatives with respect to joints


if __name__ == "__main__":
    device = "cuda"
    sphere_sdf = SphereSDF(device)

    with open(os.path.join(CUR_DIR, 'panda_layer/franka_sphere.yaml'), 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)['collision_spheres']

    x = torch.rand(128, 3).to(device) * 2.0 - 1.0
    theta = torch.rand(1, 7).to(device).float()
    pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).expand(len(theta), 4, 4).float()
    rs, cs = sphere_sdf.get_sphere_param(conf)
    sdf, grad = sphere_sdf.get_sdf(x, pose, theta, rs, cs)
    sdf, grad = sphere_sdf.get_sdf_with_joint_grad(x, pose, theta, rs, cs)
    print(sdf.shape, grad.shape)
