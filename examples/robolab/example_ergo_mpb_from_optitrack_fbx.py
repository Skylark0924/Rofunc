"""
Ergo manipulation
===========================

Visualize ergonomics and manipulability of a humanoid robot using Optitrack data and Jacobian matrix.
"""

import isaacgym

from rofunc.utils.datalab.poselib.robot_utils.HOTU.optitrack_fbx_to_hotu_npy import *
from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers
from tqdm import tqdm

import argparse
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import os


def inference(custom_args):
    task_name = "HumanoidASEViewMotion"
    args_overrides = [
        f"task={task_name}",
        "train=HumanoidASEViewMotionASERofuncRL",
        f"device_id=0",
        f"rl_device=cuda:{gpu_id}",
        "headless={}".format(True),
        "num_envs={}".format(1),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file
    cfg.task.env.asset.assetFileName = custom_args.humanoid_asset

    cfg_dict = omegaconf_to_dict(cfg.task)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[task_name](cfg=cfg_dict,
                                            rl_device=cfg.rl_device,
                                            sim_device=f'cuda:{cfg.device_id}',
                                            graphics_device_id=cfg.device_id,
                                            headless=cfg.headless,
                                            virtual_screen_capture=cfg.capture_video,
                                            force_render=cfg.force_render,
                                            csv_path=rf.oslab.get_rofunc_path(
                                                f'../examples/data/hotu2/{input_file_name}.csv'))

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map["ase"](cfg=cfg,
                                            env=infer_env,
                                            device=cfg.rl_device,
                                            env_name=task_name,
                                            hrl=False,
                                            inference=True)

    # Start inference
    trainer.inference()


def update(frame):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    joint_rotations = skeleton_joint[frame]
    global_positions, global_rotations = rf.maniplab.forward_kinematics(skeleton_joint_local_translation,
                                                                        joint_rotations, skeleton_parent_indices)
    rf.visualab.plot_skeleton(ax, global_positions, skeleton_parent_indices)
    right_hand_index = 5
    left_hand_index = 8
    jacobian_right = np.squeeze(jacobians_right[frame])
    eigenvalues_right, eigenvectors_right = rf.maniplab.calculate_manipulability(jacobian_right)
    rf.visualab.plot_ellipsoid(ax, eigenvalues_right, eigenvectors_right, global_positions[right_hand_index], 'b')
    jacobian_left = np.squeeze(jacobians_left[frame])
    eigenvalues_left, eigenvectors_left = rf.maniplab.calculate_manipulability(jacobian_left)
    rf.visualab.plot_ellipsoid(ax, eigenvalues_left, eigenvectors_left, global_positions[left_hand_index], 'r')

    rf.visualab.plot_skeleton(ax, global_positions, skeleton_parent_indices)

    # Define a structure to map joint types to their corresponding functions and indices
    joint_analysis = {
        'upper_arm': {
            'indices': [skeleton_joint_name.index('right_upper_arm'), skeleton_joint_name.index('left_upper_arm')],
            'degree_func': rf.robolab.ergonomics.UpperArmDegree(global_positions).upper_arm_degrees,
            'reba_func': rf.robolab.ergonomics.UAREBA,
            'score_func': lambda reba: reba.upper_arm_reba_score(),
            'color_func': rf.visualab.ua_get_color
        },
        'lower_arm': {
            'indices': [skeleton_joint_name.index('right_lower_arm'), skeleton_joint_name.index('left_lower_arm')],
            'degree_func': rf.robolab.ergonomics.LADegrees(global_positions).lower_arm_degree,
            'reba_func': rf.robolab.ergonomics.LAREBA,
            'score_func': lambda reba: reba.lower_arm_score(),
            'color_func': rf.visualab.la_get_color
        },
        'trunk': {
            'indices': [skeleton_joint_name.index('pelvis')],
            'degree_func': rf.robolab.ergonomics.TrunkDegree(global_positions, global_rotations).trunk_degrees,
            'reba_func': rf.robolab.ergonomics.TrunkREBA,
            'score_func': lambda reba: reba.trunk_reba_score(),
            'color_func': rf.visualab.trunk_get_color
        }
    }

    # Process each joint type with its corresponding REBA analysis and coloring
    for joint_type, settings in joint_analysis.items():
        degrees = settings['degree_func']()
        reba = settings['reba_func'](degrees)
        scores = settings['score_func'](reba)
        for i, idx in enumerate(settings['indices']):
            score = scores[i]
            color = settings['color_func'](score)
            ax.scatter(*global_positions[idx], color=color, s=100)  # Use scatter for single points


if __name__ == "__main__":
    gpu_id = 0
    input_file_name = 'demo_3_andrew_only'

    # # Calculate from optitrack data to HOTU joint information
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx_dir", type=str, default=None)
    parser.add_argument("--fbx_file", type=str,
                        default=f"{rf.oslab.get_rofunc_path()}/../examples/data/hotu2/{input_file_name}_optitrack.fbx")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--humanoid_asset", type=str, default="mjcf/hotu/hotu_humanoid.xml")
    parser.add_argument("--target_tpose", type=str,
                        default="utils/datalab/poselib/data/target_hotu_humanoid_tpose.npy")
    args = parser.parse_args()

    rofunc_path = rf.oslab.get_rofunc_path()

    if args.fbx_dir is not None:
        fbx_dir = args.fbx_dir
        fbx_files = rf.oslab.list_absl_path(fbx_dir, suffix='.fbx')
    elif args.fbx_file is not None:
        fbx_files = [args.fbx_file]
    else:
        raise ValueError("Please provide a valid fbx_dir or fbx_file.")

    if args.parallel:
        pool = multiprocessing.Pool()
        pool.map(npy_from_fbx, fbx_files)
    else:
        with tqdm(total=len(fbx_files)) as pbar:
            for fbx_file in fbx_files:
                npy_from_fbx(args, fbx_file)
                pbar.update(1)

    # # # Calculate joint angles
    parser.add_argument("--motion_file", type=str, default=rf.oslab.get_rofunc_path(
        f"../examples/data/hotu2/{input_file_name}_optitrack2hotu.npy"))
    custom_args = parser.parse_args()

    inference(custom_args)

    # # Calculate Jacobian matrix
    rf.logger.beauty_print("########## Jacobian from URDF or MuJoCo XML files with RobotModel class ##########")
    model_path = "../../rofunc/simulator/assets/mjcf/hotu/hotu_humanoid.xml"
    joint_value = [0.1 for _ in range(34)]
    export_links = ["right_hand_2", "left_hand_2"]

    robot = rf.robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)
    input_csv_path = rf.oslab.get_rofunc_path(
        f'../examples/data/hotu2/{input_file_name}.csv')
    save_directory = rf.oslab.get_rofunc_path('../examples/data/jacobian_data')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    jacobians = {link: [] for link in export_links}

    with open(input_csv_path, mode='r') as file:
        reader = csv.reader(file)
        joint_data = list(reader)

    for joint_values in joint_data:
        joint_values = list(map(float, joint_values))
        for link in export_links:
            J = robot.get_jacobian(joint_values, link)
            jacobians[link].append(J)

    for link in export_links:
        filename = os.path.join(save_directory, f'jacobian_{input_file_name}_{link}.npy')
        np.save(filename, np.array(jacobians[link]))
        print(f"Saved all Jacobians for {link} to {filename}")

    # # Calculate and draw skeleton model, ergonomics and manipulability
    skeleton_joint_name, skeleton_joint, skeleton_parent_indices, skeleton_joint_local_translation = rf.maniplab.read_skeleton_motion(
        rf.oslab.get_rofunc_path(f'../examples/data/hotu2/{input_file_name}_optitrack2hotu.npy'))
    skeleton_joint = skeleton_joint[::40, :, :]
    jacobians_left = np.array(jacobians["left_hand_2"])[
                     ::10]
    jacobians_right = np.array(jacobians["right_hand_2"])[::10]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(skeleton_joint), repeat=True)
    # ani.save('/home/ubuntu/Ergo-Manip/data/gif/demo_2_andrew.gif', writer='imagemagick', fps=3)
    plt.show()
