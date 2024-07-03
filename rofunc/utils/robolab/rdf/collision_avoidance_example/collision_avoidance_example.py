import pybullet as p
import pybullet_data as pd
import numpy as np
import sys
import time
from pybullet_panda_sim import PandaSim, SphereManager
import torch
import os
import math

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_PATH, '../'))
import bf_sdf
from panda_layer.panda_layer import PandaLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
robot = PandaLayer(device)
bp_sdf = bf_sdf.BPSDF(8, -1.0, 1.0, robot, device)
bp_sdf_model = torch.load(os.path.join(CUR_PATH, '../models/BP_8.pt'))


def main_loop():
    # p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
    #                          ' --background_color_blue=0.5 --width=1600 --height=1000')
    p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                             ' --background_color_blue=1 --width=1000 --height=1000')

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(lightPosition=[5, 5, 5])
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=145, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.6])

    p.setAdditionalSearchPath(pd.getDataPath())
    timeStep = 0.01
    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    ## spawn franka robot
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    panda = PandaSim(p, base_pos, base_rot)
    q_target = np.array([1.34643478, -0.06273081, 0.16822745, -2.54468149, -0.32664142, 3.57070459, 1.14682636])
    q_init = np.array([-0.23643478, -0.06273081, 0.16822745, -2.54468149, -0.32664142, 3.57070459, 1.14682636])
    panda.set_joint_positions(q_init)
    time.sleep(1)

    # obstacle
    sphere_manager = SphereManager(p)
    sphere_center = [0.3, 0.4, 0.5]
    pose = torch.eye(4).unsqueeze(0).to(device).float()
    while True:
        sphere_manager.create_sphere(sphere_center, 0.05, [0.8500, 0.3250, 0.0980, 1])
        pts = torch.tensor(sphere_center).unsqueeze(0).to(device).float()
        q_current = panda.get_joint_positions()
        theta = torch.tensor(q_current).unsqueeze(0).to(device).float()
        sdf, joint_grad = bp_sdf.get_whole_body_sdf_with_joints_grad_batch(pts, pose, theta, bp_sdf_model)
        # print(f'sdf: {sdf.shape}, joint_grad: {joint_grad.shape}')
        sdf, joint_grad = sdf.squeeze(0), joint_grad.squeeze(0)
        sdf_min, sdf_min_idx = torch.min(sdf, dim=0)
        sdf_min_grad = joint_grad[sdf_min_idx]
        # print(f'sdf_min: {sdf_min.shape}, sdf_min_grad: {sdf_min_grad.shape}')

        goal_reaching_vec = q_target - q_current
        goal_reaching_vec = goal_reaching_vec / np.linalg.norm(goal_reaching_vec)

        collision_avoidance_vec = torch.nn.functional.normalize(sdf_min_grad, dim=-1).detach().cpu().numpy()
        print(sdf_min)
        if sdf_min > 0.2:
            vec = goal_reaching_vec
        else:
            vec = collision_avoidance_vec

        q_current = q_current + 0.05 * vec
        print(f'q_current: {q_current}')
        panda.set_joint_positions(q_current)
        sphere_center = sphere_center + np.array([0.0, -0.01, 0.0])
        time.sleep(0.01)
        sphere_manager.delete_spheres()


if __name__ == '__main__':
    main_loop()
