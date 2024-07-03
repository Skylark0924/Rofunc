import os
import numpy as np
import glob
import torch
import trimesh
from mlp import MLPRegression
import torch.nn.functional as F
import mesh_to_sdf
import skimage
from panda_layer.panda_layer import PandaLayer
import utils

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class NNSDF():
    def __init__(self, robot, lr=0.002, device='cuda'):
        self.device = device
        self.robot = robot
        self.lr = lr
        self.model_path = os.path.join(CUR_DIR, 'models')

    def train_nn(self, epoches=2000):
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/panda_layer/meshes/voxel_128/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = sorted(mesh_files)[1:]  # except finger
        mesh_dict = {}
        for i, mf in enumerate(mesh_files):
            mesh_name = mf.split('/')[-1].split('.')[0]
            print('mesh_name: ', mesh_name)
            mesh = trimesh.load(mf)
            mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices - offset, axis=1))

            mesh_dict[i] = {}
            mesh_dict[i]['mesh_name'] = mesh_name

            # load data
            data = np.load(f'./data/sdf_points/voxel_128_{mesh_name}.npy', allow_pickle=True).item()
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]
            model = MLPRegression(input_dims=3, output_dims=1, mlp_layers=[128, 128, 128, 128, 128], skips=[],
                                  act_fn=torch.nn.ReLU, nerf=True)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                                   threshold=0.01, threshold_mode='rel',
                                                                   cooldown=0, min_lr=0, eps=1e-04, verbose=True)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            for iter in range(epoches):
                model.train()
                with torch.cuda.amp.autocast():
                    choice_near = np.random.choice(len(point_near_data), 1024, replace=False)
                    p_near, sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(
                        device), torch.from_numpy(sdf_near_data[choice_near]).float().to(device)
                    choice_random = np.random.choice(len(point_random_data), 256, replace=False)
                    p_random, sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(
                        device), torch.from_numpy(sdf_random_data[choice_random]).float().to(device)
                    p = torch.cat([p_near, p_random], dim=0)
                    sdf = torch.cat([sdf_near, sdf_random], dim=0)
                    sdf_pred = model.forward(p)
                    loss = F.mse_loss(sdf_pred[:, 0], sdf, reduction='mean')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                if iter % 100 == 0:
                    print(loss.item())

            mesh_dict[i]['model'] = model
            mesh_dict[i]['offset'] = torch.from_numpy(offset)
            mesh_dict[i]['scale'] = scale

        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)
        torch.save(mesh_dict, f'{self.model_path}/NN_{epoches}.pt')  # save nn sdf model
        print(f'{self.model_path}/NN_{epoches}.pt model saved!')

    def sdf_to_mesh(self, model, nbData):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            mesh_name_list.append(mesh_name)
            model_k = mesh_dict['model'].to(self.device)
            model_k.eval()
            domain = torch.linspace(-1.0, 1.0, nbData).to(self.device)
            grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
            p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)
            # split data to deal with memory issues
            p_split = torch.split(p, 100000, dim=0)
            d = []
            with torch.no_grad():
                for p_s in p_split:
                    d_s = model_k.forward(p_s)
                    d.append(d_s)
            d = torch.cat(d, dim=0)
            # scene.add_geometry(mesh)
            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0, spacing=np.array([(2.0) / nbData] * 3)
            )
            verts = verts - [1, 1, 1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list, mesh_name_list

    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        verts_list, faces_list, mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces, mesh_name in zip(verts_list, faces_list, mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts, faces)
            if vis:
                rec_mesh.show()
            if save_mesh_name != None:
                save_path = os.path.join(CUR_DIR, "output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh,
                                                    os.path.join(save_path, f"{save_mesh_name}_{mesh_name}.stl"))

    def whole_body_nn_sdf(self, x, pose, theta, model, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        B = len(theta)
        N = len(x)
        K = len(used_links)
        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()
        trans_list = self.robot.get_transformations_each_link(pose, theta)

        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(-1, 4,
                                                                                                       4)  # B,K,4,4
        x_robot_frame_batch = utils.transform_points(x.float(), torch.linalg.inv(fk_trans).float(),
                                                     device=self.device)  # B*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)  # B*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled.reshape(B, K, N, 3).transpose(0, 1)  # K,B,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        # sdf
        sdf = []
        for i in model.keys():
            model_i = model[i]['model'].eval().to(self.device)
            sdf.append(model_i.forward(x_bounded[i]))
        sdf = torch.cat(sdf, dim=0).reshape(K, B, N)

        sdf = sdf + res_x.norm(dim=-1)
        sdf = sdf.transpose(0, 1)

        sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
        sdf_value, idx = sdf.min(dim=1)
        return sdf_value

    def whole_body_nn_sdf_with_joints_grad_batch(self, x, pose, theta, model, used_links=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B, 7, 7) + torch.eye(7, device=self.device).unsqueeze(0).expand(B, 7,
                                                                                                7) * delta).reshape(B,
                                                                                                                    -1,
                                                                                                                    7)
        theta = torch.cat([theta, d_theta], dim=1).reshape(B * 8, 7)
        pose = pose.expand(B * 8, 4, 4)
        sdf = self.whole_body_nn_sdf(x, pose, theta, model, used_links=used_links).reshape(B, 8, -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)


if __name__ == '__main__':
    device = 'cuda'
    lr = 0.002
    panda = PandaLayer(device)
    nn_sdf = NNSDF(panda, lr, device)

    # # train neural network model   
    # nn_sdf.train_nn(epoches=200)

    # visualize the Bernstein Polynomial model for each robot link
    model_path = f'models/NN_AD.pt'
    model = torch.load(model_path)
    nn_sdf.create_surface_mesh(model, nbData=128, vis=True)

    # run nn sdf model
    x = torch.rand(128, 3).to(device) * 2.0 - 1.0
    theta = torch.rand(1, 7).to(device).float()
    pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).expand(len(theta), 4, 4).float()
    sdf_value = nn_sdf.whole_body_nn_sdf(x, pose, theta, model)
    sdf, joint_grad = nn_sdf.whole_body_nn_sdf_with_joints_grad_batch(x, pose, theta, model)
    print(sdf_value.shape)
    print(sdf_value.shape, joint_grad.shape)
