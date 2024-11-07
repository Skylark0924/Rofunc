import numpy as np

def read_skeleton_motion(file_path):
    skeleton = np.load(file_path, allow_pickle=True).item()
    return skeleton['skeleton_tree']['node_names'], skeleton['rotation']['arr'], skeleton['skeleton_tree']['parent_indices']['arr'], skeleton['skeleton_tree']['local_translation']['arr']