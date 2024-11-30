"""
BiRP
=================================================

The code for the paper "BiRP: BiRP: Learning Robot Generalized Bimanual Coordination using Relative Parameterization
Method on Human Demonstration" by Junjia LIU, et al. (https://arxiv.org/abs/2307.05933)
"""

from isaacgym import gymutil

import os
import pandas as pd
import numpy as np
import rofunc as rf
import matplotlib.pyplot as plt

save_params = {'save_dir': '/home/ubuntu/Pictures/BIRP6', 'format': ['png']}


def traj_rot(demos, pos_rot_matrix, pos_rot_quat=None, ori_rot_quat=None):
    roted_demos = []
    for demo in demos:
        demo[:, :3] = np.matmul(pos_rot_matrix, demo[:, :3].transpose()).transpose()
        if pos_rot_quat is not None:
            for j in range(len(demo[:, 3:])):
                demo[j, 3:] = rf.robolab.quaternion_multiply(pos_rot_quat, demo[j, 3:])
        if ori_rot_quat is not None:
            for j in range(len(demo[:, 3:])):
                demo[j, 3:] = rf.robolab.quaternion_multiply(ori_rot_quat, demo[j, 3:])
        roted_demos.append(demo)
    return roted_demos


def traj_offset(demos, offset):
    """

    :param demos:
    :param offset: [off_x, off_y, off_z]
    :return:
    """
    offset_demos = []
    for demo in demos:
        for i in range(len(offset)):
            demo[:, i] += offset[i]
        offset_demos.append(demo)
    return offset_demos


def get_traj():
    path = '/home/ubuntu/Data/optitrack_record/2023_03_29'

    objs, meta = rf.optitrack.get_objects(path)
    objs = objs[0]
    meta = meta[0]

    data, labels = rf.optitrack.data_clean(path, legacy=False, objs=objs)[0]

    label_idx = labels.index('table.pose.x')
    table_pos_x = data[label_idx, :]

    data_lst = []
    for i in os.listdir(path):
        # if i.endswith('.csv') and i in ['demo_exp_12.csv', 'demo_exp_15.csv', 'demo_exp_16.csv']:  # Pouring water
        if i.endswith('.csv') and i in ['demo_exp_7.csv', 'demo_exp_2.csv', 'demo_exp_3.csv']:  # Box carrying
            data_lst.append(np.array(pd.read_csv(os.path.join(path, i), skiprows=1)))

    demos_left_x = [data[::5, 3:10] / 1000. for data in data_lst]
    demos_right_x = [data[::5, 10:17] / 1000. for data in data_lst]
    demos_left_x = [data[: 200] for data in demos_left_x]
    demos_right_x = [data[: 200] for data in demos_right_x]

    pos_rot_quat = rf.robolab.quaternion_about_axis(-np.pi / 2, [0, 1, 0])
    pos_rot_matrix = rf.robolab.homo_matrix_from_quaternion(pos_rot_quat)
    pos_rot_matrix = pos_rot_matrix[:3, :3]
    ori_rot_quat = rf.robolab.quaternion_about_axis(np.pi, [1, 0, 0])
    traj_rot(demos_left_x, pos_rot_matrix)
    traj_rot(demos_right_x, pos_rot_matrix)
    traj_offset(demos_left_x, [0.95, 0., -0.2])
    traj_offset(demos_right_x, [0.95, 0., -0.2])

    demos_first_box = [data[::5, 17:24] / 1000. for data in data_lst]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', fc='white')
    rf.visualab.traj_plot(demos_left_x, legend='left', g_ax=ax, ori=False)
    rf.visualab.traj_plot(demos_right_x, legend='right', g_ax=ax, ori=False)
    # rf.visualab.traj_plot([demos_left_cup[0]], legend='left_cup', g_ax=ax)
    # rf.visualab.traj_plot([demos_right_cup[0]], legend='right_cup', g_ax=ax)

    rf.visualab.save_img(fig, save_params['save_dir'])
    plt.show()
    return demos_left_x, demos_right_x


def learn(demos_left_x, demos_right_x, demo_idx=0):
    # Define observe frames
    start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]
    via1_xdx_l = [demos_left_x[i][50] for i in range(len(demos_left_x))]
    via2_xdx_l = [demos_left_x[i][100] for i in range(len(demos_left_x))]
    end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]

    start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
    via1_xdx_r = [demos_right_x[i][50] for i in range(len(demos_right_x))]
    via2_xdx_r = [demos_right_x[i][100] for i in range(len(demos_right_x))]
    end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]

    task_params = {'left': {'frame_origins': [start_xdx_l, via1_xdx_l, via2_xdx_l, end_xdx_l],
                            'frame_names': ['start', 'via1', 'via2', 'end']},
                   'right': {'frame_origins': [start_xdx_r, via1_xdx_r, via2_xdx_r, end_xdx_r],
                             'frame_names': ['start', 'via1', 'via2', 'end']}}

    # Create representation model
    representation = rf.RofuncML.TPGMM_RPCtrl(demos_left_x, demos_right_x, nb_states=4, plot=True, save=False,
                                          save_params=save_params, task_params=task_params)

    # Define observe frames for new situation
    start_xdx_l = representation.repr_l.demos_xdx[demo_idx][0]
    via1_xdx_l = np.hstack((np.array([1, 0.8, 0.5]), representation.repr_l.demos_xdx[demo_idx][100][3: 7]))
    via2_xdx_l = np.hstack((np.array([1, 0.8, -0.5]), representation.repr_l.demos_xdx[demo_idx][150][3: 7]))
    end_xdx_l = np.hstack((np.array([1, 0.8, -0.5]), representation.repr_l.demos_xdx[demo_idx][-1][3: 7]))

    start_xdx_r = representation.repr_r.demos_xdx[demo_idx][0]
    via1_xdx_r = np.hstack((np.array([1, 0.8, 0.5]), representation.repr_r.demos_xdx[demo_idx][100][3: 7]))
    via2_xdx_r = np.hstack((np.array([1, 0.8, -0.5]), representation.repr_r.demos_xdx[demo_idx][150][3: 7]))
    end_xdx_r = np.hstack((np.array([1, 0.8, 0.5]), representation.repr_r.demos_xdx[demo_idx][-1][3: 7]))

    task_params = {'left': {'frame_origins': [start_xdx_l, via1_xdx_l, via2_xdx_l, end_xdx_l],
                            'frame_names': ['start', 'via1', 'via2', 'end']},
                   'right': {'frame_origins': [start_xdx_r, via1_xdx_r, via2_xdx_r, end_xdx_r],
                             'frame_names': ['start', 'via1', 'via2', 'end']}}
    # 'traj': representation.repr_r.demos_x[demo_idx]}}
    if isinstance(representation, rf.RofuncML.TPGMM_RPCtrl) or isinstance(representation, rf.RofuncML.TPGMM_RPAll):
        model_l, model_r, model_c = representation.fit()
        representation.reproduce(model_l, model_r, model_c, show_demo_idx=demo_idx)
        traj_l, traj_r, _, _ = representation.generate(model_l, model_r, model_c, ref_demo_idx=demo_idx,
                                                       task_params=task_params)
    else:
        model_l, model_r = representation.fit()
        representation.reproduce(model_l, model_r, show_demo_idx=demo_idx)
        leader = None
        traj_leader, traj_follower, _, _ = representation.generate(model_l, model_r, ref_demo_idx=demo_idx,
                                                                   task_params=task_params, leader=leader)

        traj_l, traj_r = (traj_leader, traj_follower) if leader in ['left', None] else (traj_follower, traj_leader)

    nb_dim = len(demos_left_x[0][0])
    data_lst = [traj_l[:, :nb_dim], traj_r[:, :nb_dim]]
    fig = rf.visualab.traj_plot(data_lst, title='Generated Trajectories', ori=True)
    # rf.visualab.save_img(fig, save_params['save_dir'])
    # plt.show()
    return traj_l, traj_r


def sim(left_x, right_x):
    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False
    CURIsim = rf.sim.CURISim(args, fix_base_link=True)
    CURIsim.init()
    CURIsim.run_traj([left_x, right_x])


if __name__ == '__main__':
    demos_left_x, demos_right_x = get_traj()
    traj_l, traj_r = learn(demos_left_x, demos_right_x)
    sim(traj_l, traj_r)
