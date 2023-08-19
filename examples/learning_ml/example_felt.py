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
    all_files = rf.rfos.list_absl_path(data_dir, recursive=False, prefix='trial')
    for file in all_files:
        hand_rigid = pd.read_csv(os.path.join(file, 'mocap_hand_rigid.csv'))
        object_rigid = pd.read_csv(os.path.join(file, 'mocap_object_rigid.csv'))
        hand_marker_positions = pd.read_csv(os.path.join(file, 'mocap_hand.csv'))
        object_marker_positions = pd.read_csv(os.path.join(file, 'mocap_object.csv'))

        def get_center_position(df):
            data = df.to_numpy().reshape((len(df.to_numpy()), -1, 3))
            return np.mean(data, axis=1)

        def get_orientation(df):
            data = df.to_numpy().reshape((len(df.to_numpy()), 3, 3))
            data = np.array([rf.robolab.quaternion_from_matrix(rf.robolab.homo_matrix_from_rot_matrix(i)) for i in data])
            return data

        hand_position = get_center_position(hand_marker_positions)
        object_position = get_center_position(object_marker_positions)
        hand_ori = get_orientation(hand_rigid)
        object_ori = get_orientation(object_rigid)

    return demos_x


demos_x = data_process('../data/felt/wipe_circle')

# --- TP-GMM ---
# Define the task parameters
start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
# Fit the model
Repr = rf.ml.TPGMM(demos_x, task_params, plot=True)
model = Repr.fit()

# Reproductions for the same situations
traj, _ = Repr.reproduce(model, show_demo_idx=2)

# Reproductions for new situations: set the endpoint as the start point to make a cycled motion
ref_demo_idx = 2
start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
end_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
traj, _ = Repr.generate(model, ref_demo_idx)
