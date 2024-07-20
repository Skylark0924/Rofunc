import os
import numpy as np
import torch
import time
import bf_sdf, nn_sdf, sphere
from panda_layer.panda_layer import PandaLayer
import argparse
from rofunc.utils.robolab.rdf import utils
import yaml

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--method', default='BP_8', type=str)
    parser.add_argument('--type', default='RDF', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_rec_robot_surface', action='store_true')
    args = parser.parse_args()

    data = np.load(f'./data/sdf_points/test_data.npy', allow_pickle=True).item()
    panda = PandaLayer(args.device)

    if args.method == 'BP_8':
        bpSdf = bf_sdf.BPSDF(8, -1, 1, panda, args.device)
        model = torch.load(f'models/{args.method}.pt')
    elif args.method == 'BP_24':
        bpSdf = bf_sdf.BPSDF(24, -1, 1, panda, args.device)
        model = torch.load(f'models/{args.method}.pt')
    elif args.method == 'NN_LD' or args.method == 'NN_AD':
        nnSdf = nn_sdf.NNSDF(panda, device=args.device)
        model = torch.load(f'models/{args.method}.pt')
    elif args.method == 'Sphere':
        sphere_sdf = sphere.SphereSDF(args.device)
        with open(os.path.join(CUR_DIR, 'panda_layer/franka_sphere.yaml'), 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)['collision_spheres']
        rs, cs = sphere_sdf.get_sphere_param(conf)

    # eval EACH LINK
    if args.type == 'LINK':
        # save reconstructed mesh for each robo link
        print('save mesh from sdfs...')
        if args.method == 'BP_8' or args.method == 'BP_24':
            bpSdf.create_surface_mesh(model, nbData=256, vis=args.vis, save_mesh_name=args.method)
        elif args.method == 'NN_LD' or args.method == 'NN_AD':
            nnSdf.create_surface_mesh(model, nbData=256, vis=args.vis, save_mesh_name=args.method)

        # eval chamfer distance
        cd_mean, cd_max = utils.eval_chamfer_distance(tag=args.method)
        print(f'eval chamfer distance:\tmethod:{args.method} cd mean:{cd_mean}\tcd max:{cd_max}')

    # eval RDF
    if args.type == 'RDF':
        # eval quality
        res = []
        for k in data.keys():
            x = data[k]['pts']
            sdf = data[k]['sdf']
            pose = data[k]['pose']
            theta = data[k]['theta']
            if args.method == 'BP_8' or args.method == 'BP_24':
                pred_sdf, _ = bpSdf.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=False)
            elif args.method == 'NN_LD' or args.method == 'NN_AD':
                pred_sdf = nnSdf.whole_body_nn_sdf(x, pose, theta, model)
            elif args.method == 'Sphere':
                pred_sdf, _ = sphere_sdf.get_sdf(x, pose, theta, rs, cs)
            res.append(utils.print_eval(pred_sdf, sdf))
        res = np.mean(res, axis=0) * 1000.
        print(f'Evaluate produced robot distance field:\t'
              f'Method:{args.method}\t'
              f'MAE:{res[0]:.3f}\t'
              f'RMSE:{res[2]:.3f}\t'
              f'MAE_NEAR:{res[3]:.3f}\t'
              f'RMSE_NEAR:{res[5]:.3f}\t'
              f'MAE_FAR:{res[6]:.3f}\t'
              f'RMSE_FAR:{res[8]:.3f}\t'
              )

        # eval time
        t = []
        x = torch.rand(1024, 3).to(args.device) * 2.0 - 1.0
        panda = PandaLayer(args.device)
        theta = torch.rand(1, 7).to(args.device).float()
        pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(len(theta), 4, 4).float()
        for _ in range(100):
            time0 = time.time()
            if args.method == 'BP_8' or args.method == 'BP_24':
                pred_sdf, _ = bpSdf.get_whole_body_sdf_batch(x, pose, theta, model, use_derivative=False)
            elif args.method == 'NN_LD' or args.method == 'NN_AD':
                pred_sdf = nnSdf.whole_body_nn_sdf(x, pose, theta, model)
            elif args.method == 'Sphere':
                pred_sdf, _ = sphere_sdf.get_sdf(x, pose, theta, rs, cs)
            t.append(time.time() - time0)
        print(f'Method:{args.method}\t Time Cost:{np.mean(t[1:]) * 1000.}ms\t')

    # vis reconstructed robot surface (please make sure you have saved the reconstructed mesh before)
    if args.vis_rec_robot_surface:
        theta = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]).float().to(args.device).reshape(-1, 7)
        pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(theta), 4, 4).float()
        trans_list = panda.get_transformations_each_link(pose, theta)
        utils.visualize_reconstructed_whole_body(model, trans_list, tag=args.method)
