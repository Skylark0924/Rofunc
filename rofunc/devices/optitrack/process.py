import os

import pandas as pd
from tqdm import tqdm
import numpy as np

def delete_lines(in_path, out_path, head, tail=0):
    with open(in_path, 'r') as fin:
        a = fin.readlines()
    with open(out_path, 'w') as fout:
        b = ''.join(a[head:])
        fout.write(b)


def data_clean(input_path):
    """
    Args:
        input_path: csv file path

    Returns: csv file without the first 6 lines

    """
    demo_csvs = os.listdir(input_path)
    demo_csvs = sorted(demo_csvs)
    for demo_csv in demo_csvs:
        if 'Take' in demo_csv:
            if 'Manus' in demo_csv:
                out_path = os.path.join(input_path, 'process')
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                demo_path = os.path.join(input_path, demo_csv)
                out_file_path = os.path.join(out_path, demo_csv)
                delete_lines(demo_path, out_file_path, 14)
                csv_data = pd.read_csv(out_file_path)
                # csv_data = pd.read_csv(demo_path, skiprows=6)
                if '3f6ec26f' in demo_csv:
                    csv_data.to_csv(os.path.join(input_path, "left_manus.csv"))
                elif '7b28f20b' in demo_csv:
                    csv_data.to_csv(os.path.join(input_path, "right_manus.csv"))
            else:
                demo_path = os.path.join(input_path, demo_csv)
                # The first 6 rows are headers: https://v22.wiki.optitrack.com/index.php?title=Data_Export:_CSV
                csv_data = pd.read_csv(demo_path, skiprows=6)
                csv_data.to_csv(os.path.join(input_path, "opti_hands.csv"))
    print('{} finished'.format(input_path.split('/')[-1]))


def data_clean_batch(input_dir):
    demos = os.listdir(input_dir)
    demos = sorted(demos)
    for demo in tqdm(demos):
        input_path = os.path.join(input_dir, demo)
        data_clean(input_path)


def export(input_dir):
    csv_data = pd.read_csv(input_dir, skiprows=7, header=None)

    type_data = pd.read_csv(input_dir, skiprows=2, nrows=0)
    type_list = list(type_data.columns)

    name_data = pd.read_csv(input_dir, skiprows=3, nrows=0)
    name_list = list(name_data.columns)

    time_data = pd.read_csv(input_dir, skiprows=6, usecols=['Frame'])
    time_data_list = list(time_data.index)

    rigid_body_index_list = []
    for i in range(len(type_list)):
        if (":Marker" not in name_list[i]) and ("Rigid Body" in type_list[i]):
            rigid_body_index_list.append(i)

    num_rigid_body = int(len(rigid_body_index_list) / 7)
    optitrack_data_list = []

    for i in time_data_list:
        frame_data = []
        for j in range(num_rigid_body):
            rigid_body_index_start = 7 * j
            rigid_body_index_end = 7 + 7 * j
            frame_data.append(
                list(csv_data.values[i, rigid_body_index_list[rigid_body_index_start:rigid_body_index_end]]))
        optitrack_data_list.append(frame_data)
    return np.array(optitrack_data_list)



