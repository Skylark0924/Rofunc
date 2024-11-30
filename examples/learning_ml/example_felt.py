"""
FeLT
=================

The coder for the paper "FeLT: Fully Tactile-driven Robot Plate Cleaning Skill Learning from Human Demonstration
 with Tactile Sensor" by Junjia LIU, et al.
"""
import numpy as np
import os
import rofunc as rf
import pandas as pd


# --- Data processing ---
def data_process(data_dir):
    all_files = rf.oslab.list_absl_path(data_dir, recursive=False, prefix='trial')

    demos_x = []
    demos_taxels_pressure = []

    for file in all_files:
        hand_rigid = pd.read_csv(os.path.join(file, 'mocap_hand_rigid.csv'))
        object_rigid = pd.read_csv(os.path.join(file, 'mocap_object_rigid.csv'))
        hand_marker_positions = pd.read_csv(os.path.join(file, 'mocap_hand.csv'))
        object_marker_positions = pd.read_csv(os.path.join(file, 'mocap_object.csv'))
        sensor_data = pd.read_csv(os.path.join(file, 'sensor_comb.csv'))['taxels_pressure'].to_numpy()

        def get_center_position(df):
            data = df.to_numpy().reshape((len(df.to_numpy()), -1, 3))
            return np.mean(data, axis=1)

        def get_orientation(df):
            data = df.to_numpy().reshape((len(df.to_numpy()), 3, 3))
            data = np.array(
                [rf.robolab.quaternion_from_homo_matrix(rf.robolab.homo_matrix_from_rot_matrix(i)) for i in data])
            return data

        hand_position = get_center_position(hand_marker_positions)  # p_WH
        object_position = get_center_position(object_marker_positions)  # p_WO
        hand_ori = get_orientation(hand_rigid)  # q_H
        object_ori = get_orientation(object_rigid)  # q_O

        # hand_pose = np.concatenate((hand_position, hand_ori), axis=1)  # p_WH
        # object_pose = np.concatenate((object_position, object_ori), axis=1)  # p_WO

        T_WO = np.array(
            [rf.robolab.homo_matrix_from_quaternion(object_ori[i], object_position[i]) for i in range(len(object_ori))])

        T_WO_inv = np.array([np.linalg.inv(i) for i in T_WO])

        p_OH = np.array(
            [T_WO_inv[i].dot(np.hstack((hand_position[i], np.ones(1)))) for i in range(len(T_WO_inv))])[:, :3]

        demos_x.append(np.hstack((p_OH, sensor_data.reshape((-1, 1)) / 1000)))
        demos_taxels_pressure.append(sensor_data.reshape((-1, 1)) / 1000)

    return demos_x, demos_taxels_pressure


demos_x, demos_taxels_pressure = data_process('../data/felt/wipe_raster')

# --- TP-GMM ---
demos_x = [demo_x[:500, :3] for demo_x in demos_x]
# demos_x = [demos_x[0]]
# Define the task parameters
start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
# Fit the model
Repr = rf.RofuncML.TPGMM(demos_x, task_params, plot=True, nb_states=100)
model = Repr.fit()

# Reproductions for the same situations
traj, _ = Repr.reproduce(model, show_demo_idx=0)

# Reproductions for new situations: set the endpoint as the start point to make a cycled motion
ref_demo_idx = 0
start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
end_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
traj, _ = Repr.generate(model, ref_demo_idx)
