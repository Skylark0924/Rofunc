import os
import csv

import pandas as pd
from tqdm import tqdm
import numpy as np
import rofunc as rf

def get_objects(input_path):
    objs = {}
    meta = {}
    demo_csvs = os.listdir(input_path)
    demo_csvs = sorted(demo_csvs)
    for demo_csv in demo_csvs:
        if 'Take' in demo_csv:
            out_path = os.path.join(input_path, 'process')
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            demo_path = os.path.join(input_path, demo_csv)
            with open(demo_path) as f:
                data = csv.reader(f)
                row = next(data)
                meta = dict(zip(row[::2], row[1::2]))
                next(data)
                t = next(data)
                n = next(data)
                id = next(data)
                tr = next(data)
                ax = next(data)

                for i, o in enumerate(n):
                    o = o.lower()
                    if o and o != 'name':
                        if o[-7:-1] == 'marker':
                            obj = o[:-8]
                            m = o[-1]
                            if obj not in objs:
                                objs[obj] = {'markers': {}}
                            if str(m) not in objs[obj]['markers']:
                                objs[obj]['markers'][str(m)] = {'pose': {tr[i]: {ax[i]: i}}}
                            else:
                                if tr[i] not in objs[obj]['markers'][str(m)]['pose']:
                                    objs[obj]['markers'][str(m)]['pose'][tr[i]] = {}
                                objs[obj]['markers'][str(m)]['pose'][tr[i]][ax[i]] = i
                        elif not o in objs:
                            objs[o] = {
                                'type': t[i],
                                'pose': {tr[i]: {ax[i]: i}},
                                'markers': {},
                                'id': {id[i]}
                            }
                        else:
                            if tr[i] not in objs[o]['pose']:
                                objs[o]['pose'][tr[i]] = {}
                            objs[o]['pose'][tr[i]][ax[i]] = i

    return objs, meta


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
                rf.utils.delete_lines(demo_path, out_file_path, 14)
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


def get_time_series(input_dir, meta):
    print('[get_time_series] Loading data...')
    print('[get_time_series] data path: ', os.path.join(input_dir, f"{meta['Take Name']}.csv"))
    data = pd.read_csv(os.path.join(input_dir, f"{meta['Take Name']}.csv"), skiprows=6)

    return data


def export(input_dir):
    """
    Export rigid body motion data.
    Args:
        input_dir: csv file path

    Returns: [number of frames, number of rigid bodies, pose dimension = 7]
    """
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
