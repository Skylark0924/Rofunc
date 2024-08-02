import sys

sys.setrecursionlimit(100000)
import os

import numpy as np
import torch

np.set_printoptions(threshold=np.inf)
import trimesh
import mesh_to_sdf
import skimage
import rofunc as rf
from tqdm import tqdm
from rofunc.utils.robolab.rdf import utils


class RDF:
    def __init__(self, args):
        """
        Use Bernstein Polynomial to represent the SDF of the robot
        """
        self.args = args
        self.n_func = args.n_func
        self.domain_min = args.domain_min
        self.domain_max = args.domain_max
        self.device = args.device
        self.robot_asset_root = args.robot_asset_root
        self.robot_model_path = os.path.join(self.robot_asset_root, args.robot_asset_name)
        self.save_mesh_dict = args.save_mesh_dict

        # Build the robot from the URDF/MJCF file
        self.robot = rf.robolab.RobotModel(self.robot_model_path, solve_engine="pytorch_kinematics", device=self.device,
                                           verbose=False)
        assert os.path.exists(self.robot.mesh_dir), "Please organize the robot meshes in the 'meshes' folder!"

        self.link_list = self.robot.get_link_list()
        self.link_mesh_map = self.robot.get_link_mesh_map()
        self.link_mesh_name_map = {k: os.path.basename(v).split(".")[0] for k, v in self.link_mesh_map.items()}

    def _binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def _build_bernstein_t(self, t, use_derivative=False):
        # t is normalized to [0,1]
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self._binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (
                    1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def _build_basis_function_from_points(self, points, use_derivative=False):
        N = len(points)
        points = ((points - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self._build_bernstein_t(points, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func ** 2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func ** 3)
        if not use_derivative:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]
            d_phi_x = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_y = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1D).view(-1, self.n_func ** 3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
            return phi_xyz, d_phi_xyz

    def _sample_sdf_points(self, mesh, mesh_name):
        print(f'Sampling points for mesh {mesh_name}...')
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
                                                                    min_size=0.0,
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
        save_path = os.path.join(self.robot_asset_root, 'rdf/sdf_points')
        rf.oslab.create_dir(save_path)
        np.save(os.path.join(save_path, f'voxel_128_{mesh_name}.npy'), data)
        print(f'Sampling points for mesh {mesh_name} finished!')
        return data

    def train(self):
        mesh_files = rf.oslab.list_absl_path(self.robot.mesh_dir, recursive=True, suffix='.stl')
        mesh_files2 = rf.oslab.list_absl_path(self.robot.mesh_dir, recursive=True, suffix='.STL')
        mesh_files = mesh_files + mesh_files2
        mesh_dict = {}

        # sample points for each mesh
        if self.args.sampled_points:
            save_path = os.path.join(self.robot_asset_root, 'rdf/sdf_points')
            rf.oslab.create_dir(save_path)
            if self.args.parallel:
                import multiprocessing
                pool = multiprocessing.Pool(processes=12)
                data_list = pool.map(job, [(mf, mf.split('/')[-1].split('.')[0], save_path) for i, mf in
                                           enumerate(mesh_files) if not os.path.exists(
                        os.path.join(save_path, f'voxel_128_{mf.split("/")[-1].split(".")[0]}.npy'))])
                for data in data_list:
                    mesh_name = data['mesh_name']
                    np.save(os.path.join(save_path, f'voxel_128_{mesh_name}.npy'), data)
            else:
                for i, mf in enumerate(tqdm(mesh_files)):
                    mesh_name = mf.split('/')[-1].split('.')[0]
                    if os.path.exists(os.path.join(save_path, f'voxel_128_{mesh_name}.npy')):
                        continue
                    data = sample_sdf_points(mf, mesh_name, save_path)
                    np.save(os.path.join(save_path, f'voxel_128_{mesh_name}.npy'), data)

        def train_single_mesh(mf, i, data):
            mesh_name = mf.split('/')[-1].split('.')[0]
            print(f'Mesh {mesh_name} start processing...')
            mesh = trimesh.load(mf)
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices - offset, axis=1))

            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]
            wb = torch.zeros(self.n_func ** 3).float().to(self.device)
            batch_size = (torch.eye(self.n_func ** 3) / 1e-4).float().to(self.device)
            # loss_list = []
            for iter in range(self.args.train_epoch):
                choice_near = np.random.choice(len(point_near_data), 1024, replace=False)
                p_near, sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(
                    self.device), torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)
                choice_random = np.random.choice(len(point_random_data), 256, replace=False)
                p_random, sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(
                    self.device), torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)
                p = torch.cat([p_near, p_random], dim=0)
                sdf = torch.cat([sdf_near, sdf_random], dim=0)
                phi_xyz, _ = self._build_basis_function_from_points(p.float().to(self.device),
                                                                    use_derivative=False)

                K = torch.matmul(batch_size, phi_xyz.T).matmul(torch.linalg.inv(
                    (torch.eye(len(p)).float().to(self.device) + torch.matmul(torch.matmul(phi_xyz, batch_size),
                                                                              phi_xyz.T))))
                batch_size -= torch.matmul(K, phi_xyz).matmul(batch_size)
                delta_wb = torch.matmul(K, (sdf - torch.matmul(phi_xyz, wb)).squeeze())
                # loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz,wb).squeeze(), sdf, reduction='mean').item()
                # loss_list.append(loss)
                wb += delta_wb

            print(f'Mesh {mesh_name} finished!')
            mesh_dict_single = {
                'i': i,
                'mesh_name': mesh_name,
                'weights': wb,
                'offset': torch.from_numpy(offset),
                'scale': scale,
            }
            return mesh_dict_single

        # train the model for each mesh
        for i, mf in enumerate(tqdm(mesh_files)):
            mesh_name = mf.split('/')[-1].split('.')[0]
            sampled_point_data = np.load(f'{self.robot_asset_root}/rdf/sdf_points/voxel_128_{mesh_name}.npy',
                                         allow_pickle=True).item()
            res = train_single_mesh(mf, i, sampled_point_data)
            mesh_dict[res['mesh_name']] = res

        self.mesh_dict = mesh_dict
        if self.save_mesh_dict:
            rdf_model_path = os.path.join(self.robot_asset_root, 'rdf', 'BP')
            rf.oslab.create_dir(rdf_model_path)
            torch.save(mesh_dict, f'{rdf_model_path}/BP_{self.n_func}.pt')  # save the robot sdf model
            print(f'{rdf_model_path}/BP_{self.n_func}.pt model saved!')

    def sdf_to_mesh(self, model, nbData, use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].to(self.device)

            domain = torch.linspace(self.domain_min, self.domain_max, nbData).to(self.device)
            grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
            p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d = []
            for p_s in p_split:
                phi_p, d_phi_p = self._build_basis_function_from_points(p_s, use_derivative)
                d_s = torch.matmul(phi_p, weights)
                d.append(d_s)
            d = torch.cat(d, dim=0)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0,
                spacing=np.array([(self.domain_max - self.domain_min) / nbData] * 3)
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
            if save_mesh_name is not None:
                save_path = os.path.join(self.robot_asset_root, 'rdf', "output_meshes")
                rf.oslab.create_dir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh,
                                                    os.path.join(save_path, f"{save_mesh_name}_{mesh_name}.stl"))

    def get_whole_body_sdf_batch(self, points, joint_value, model, base_trans=None, use_derivative=True,
                                 used_links=None):

        batch_size = len(joint_value)
        N = len(points)

        if used_links is None:
            used_links = self.robot.get_link_list()
            used_links = [link for link in used_links if link in self.link_mesh_name_map]

        K = len(used_links)
        offset = torch.cat([model[self.link_mesh_name_map[i]]['offset'].unsqueeze(0) for i in used_links if
                            i in self.link_mesh_name_map], dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(batch_size, K, 3).reshape(batch_size * K, 3).float()
        scale = torch.tensor([model[self.link_mesh_name_map[i]]['scale'] for i in used_links if
                              i in self.link_mesh_name_map], device=self.device)
        scale = scale.unsqueeze(0).expand(batch_size, K).reshape(batch_size * K).float()

        trans_dict = self.robot.get_trans_dict(joint_value, base_trans)
        trans_lists = np.array([val.detach().cpu().numpy() for key, val in trans_dict.items() if key in used_links])
        trans_lists = torch.tensor(trans_lists).reshape((K, batch_size, 4, 4)).to(self.device)

        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_lists], dim=1)[:, :, :, :].reshape(-1, 4,
                                                                                               4)  # batch_size,K,4,4
        x_robot_frame_batch = utils.transform_points(points.float(), torch.linalg.inv(fk_trans).float(),
                                                     device=self.device)  # batch_size*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)  # batch_size*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self._build_basis_function_from_points(x_bounded.reshape(batch_size * K * N, 3),
                                                            use_derivative=False)
            phi = phi.reshape(batch_size, K, N, -1).transpose(0, 1).reshape(K, batch_size * N, -1)  # K,batch_size*N,-1
            weights_near = torch.cat([model[self.link_mesh_name_map[i]]['weights'].unsqueeze(0) for i in used_links if
                                      i in self.link_mesh_name_map], dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, batch_size, N).transpose(0, 1).reshape(
                batch_size * K,
                N)  # batch_size,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(batch_size, K, N)
            sdf = sdf * scale.reshape(batch_size, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            return sdf_value, None
        else:
            phi, dphi = self._build_basis_function_from_points(x_bounded.reshape(batch_size * K * N, 3),
                                                               use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(batch_size, K, N, -1, 4).transpose(0, 1).reshape(K, batch_size * N, -1,
                                                                                       4)  # K,batch_size*N,-1,4

            weights_near = torch.cat([model[self.link_mesh_name_map[i]]['weights'].unsqueeze(0) for i in used_links],
                                     dim=0).to(self.device)

            output = torch.einsum('ijkl,ik->ijl', phi_cat, weights_near).reshape(K, batch_size, N, 4).transpose(0,
                                                                                                                1).reshape(
                batch_size * K, N, 4)
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(batch_size, K, N)
            sdf = sdf * (scale.reshape(batch_size, K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
            # gradient = gradient.reshape(batch_size,K,N,3)
            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl', fk_rotation, gradient.transpose(1, 2)).transpose(1,
                                                                                                                2).reshape(
                batch_size, K, N, 3)
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx = idx.unsqueeze(1).unsqueeze(-1).expand(batch_size, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx)[:, 0, :, :]
            # gradient_value = None
            return sdf_value, gradient_value

    def get_whole_body_sdf_with_joints_grad_batch(self, points, joint_value, model, base_trans=None, used_links=None):
        """
        Get the SDF value and gradient of the whole body with respect to the joints

        :param points: (batch_size, 3)
        :param joint_value: (batch_size, joint_num)
        :param model: the trained RDF model
        :param base_trans: the transformation matrix of base pose, (1, 4, 4)
        :param used_links: the links to be used, list of link names
        :return:
        """
        delta = 0.001
        batch_size = joint_value.shape[0]
        joint_num = joint_value.shape[1]
        link_num = len(self.robot.get_link_list())
        joint_value = joint_value.unsqueeze(1)

        d_joint_value = (joint_value.expand(batch_size, joint_num, joint_num) + torch.eye(joint_num,
                                                                                          device=self.device).unsqueeze(
            0).expand(batch_size, joint_num, joint_num) * delta).reshape(batch_size, -1, joint_num)
        joint_value = torch.cat([joint_value, d_joint_value], dim=1).reshape(batch_size * (joint_num + 1), joint_num)

        if base_trans is not None:
            base_trans = base_trans.unsqueeze(1).expand(batch_size, (joint_num + 1), 4, 4).reshape(
                batch_size * (joint_num + 1), 4, 4)
        sdf, _ = self.get_whole_body_sdf_batch(points, joint_value, model, base_trans=base_trans, use_derivative=False,
                                               used_links=used_links)
        sdf = sdf.reshape(batch_size, (joint_num + 1), -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)

    def get_whole_body_normal_with_joints_grad_batch(self, points, joint_value, model, base_trans=None,
                                                     used_links=None):
        """
        Get the normal vector of the whole body with respect to the joints

        :param points: (batch_size, 3)
        :param joint_value: (batch_size, joint_num)
        :param model: the trained RDF model
        :param base_trans: the transformation matrix of base pose, (1, 4, 4)
        :param used_links: the links to be used, list of link names
        :return:
        """
        delta = 0.001
        batch_size = joint_value.shape[0]
        joint_num = joint_value.shape[1]
        link_num = len(self.robot.get_link_list())
        joint_value = joint_value.unsqueeze(1)

        d_joint_value = (joint_value.expand(batch_size, joint_num, joint_num) +
                         torch.eye(joint_num, device=self.device).unsqueeze(0).expand(batch_size, joint_num,
                                                                                      joint_num) * delta).reshape(
            batch_size, -1, joint_num)
        joint_value = torch.cat([joint_value, d_joint_value], dim=1).reshape(batch_size * (joint_num + 1), joint_num)

        if base_trans is not None:
            base_trans = base_trans.unsqueeze(1).expand(batch_size, (joint_num + 1), 4, 4).reshape(
                batch_size * (joint_num + 1), 4, 4)
        sdf, normal = self.get_whole_body_sdf_batch(points, joint_value, model, base_trans=base_trans,
                                                    use_derivative=True, used_links=used_links)
        normal = normal.reshape(batch_size, (joint_num + 1), -1, 3).transpose(1, 2)
        return normal  # normal size: (batch_size,N,8,3) normal[:,:,0,:] origin normal vector normal[:,:,1:,:] derivatives with respect to joints

    def visualize_reconstructed_whole_body(self, model, trans_list, tag):
        """
        Visualize the reconstructed whole body

        :param model: the trained RDF model
        :param trans_list: the transformation matrices of all links
        :param tag: the tag of the mesh, e.g., 'BP_8'
        :return:
        """
        view_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        scene = trimesh.Scene()

        for link_name, origin_mf in self.link_mesh_map.items():
            if origin_mf is not None:
                mesh_name = origin_mf.split('/')[-1].split('.')[0]
                mf = os.path.join(self.robot_asset_root, f"rdf/output_meshes/{tag}_{mesh_name}.stl")
                mesh = trimesh.load(mf)
                mesh_dict = model[mesh_name]
                offset = mesh_dict['offset'].cpu().numpy()
                scale = mesh_dict['scale']
                mesh.vertices = mesh.vertices * scale + offset

                all_related_link = [key for key in trans_list.keys() if link_name in key]
                related_link = all_related_link[-1]
                mesh.apply_transform(trans_list[related_link].squeeze().cpu().numpy())
                mesh.apply_transform(view_mat)
                scene.add_geometry(mesh)
        scene.show()


def job(args):
    return sample_sdf_points(args[0], args[1], args[2])


def sample_sdf_points(mf, mesh_name, save_path):
    print(f'Sampling points for mesh {mesh_name}...')
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
                                                                min_size=0.0,
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
        'mesh_name': mesh_name,
        'near_points': near_points,
        'near_sdf': near_sdf,
        'random_points': random_points,
        'random_sdf': random_sdf,
        'center': center,
        'scale': scale
    }
    np.save(os.path.join(save_path, f'voxel_128_{mesh_name}.npy'), data)
    print(f'Sampling points for mesh {mesh_name} finished!')
    return data
