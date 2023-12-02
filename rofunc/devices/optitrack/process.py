"""Process functions for Optitrack data.

Usage Example 1:
from rofunc.devices.optitrack import get_objects, data_clean

input_path = "/path/to/optitrack/data"
objs, meta = get_objects(input_path)
data = get_time_series(input_path, meta[0])

table_pos_x = data.iloc[:, objs[0]['table']['pose']['Position']['X']]

Usage Example 2:
from rofunc.devices.optitrack import get_objects, get_time_series
del_objects = ['cup', 'hand_right']
objs, meta = get_objects(input_path)

# Remove unused objects from the data
for obj in del_objects:
    del objs[obj]

data, labels = data_clean(input_path, legacy=False, objs=objs)[0]

label_idx = labels.index('table.pose.x')
table_pos_x = data[label_idx, :]

"""
import os
import csv
import glob

import pickle as pkl
import pandas as pd
from tqdm import tqdm
import numpy as np
import rofunc as rf


def get_objects(input_path: str):
    """Returns a dictionary of objects from the Optitrack data.
    The Optitack csv must have the original name format (e.g. "Take 2020-06-03 15-00-00.csv").
    The returned list does not necessarily have the same order as your file explorer, but the meta ond objects list do.\
    Check the meta to make sure you work on the correct file.

    Args:
        input_path (str): path to the Optitrack data.\
                          If the path is to a folder, all the file with names like "Take[...].csv" are read.
    Returns:
        tuple: (objects, meta)
    """
    objs_list = list()
    meta_list = list()
    if input_path.endswith('.csv'):
        glob_path = input_path
    else:
        glob_path = os.path.join(input_path, 'Take*.csv')
    demo_csvs = glob.glob(glob_path)
    demo_csvs = sorted(demo_csvs)
    for demo_csv in demo_csvs:
        objs = {}
        demo_path = demo_csv
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
        objs_list.append(objs)
        meta_list.append(meta)

    return objs_list, meta_list


def data_clean(input_path: str, legacy: bool = True, objs: dict = None, no_unlabeled=True, save: bool = False):
    """
    Cleans the Optitrack data.

    :param input_path: path to the Optitrack data.
    :param legacy: if True, it will use the legacy version of the function. Defaults to True.
    :param objs: dictionary of objects to keep. If set to None, export all data. Defaults to None.
    :param no_unlabeled: if True, it will remove the unlabeled data. Defaults to True.
    :param save: if True, it will save the cleaned data to disk. Defaults to False.
    :return: list of cleaned data for all csv in folder. Type of elements in list depend on args.
    """
    if input_path.endswith('.csv'):
        parent_dir = os.path.dirname(input_path)
        demo_csvs = [os.path.basename(input_path)]
        input_path = parent_dir
        out_path = os.path.join(input_path, 'process')
    else:  # A folder
        out_path = os.path.join(input_path, 'process')
        if save:
            rf.oslab.create_dir(out_path)
        demo_csvs = os.listdir(input_path)
        demo_csvs = sorted(demo_csvs)
    out_list = list()
    for i in range(len(demo_csvs)):
        demo_csv = demo_csvs[i]
        if legacy:
            out_list.append(data_clean_legacy(input_path, demo_csv, out_path))
        else:
            if objs is None:
                out_data = pd.read_csv(os.path.join(input_path, demo_csv), skiprows=6)
                if save:
                    out_data.to_csv(os.path.join(out_path, demo_csv))
                out_list.append(out_data)
            else:
                labels = ['frame', 'time']
                out_data = []
                data_raw = pd.read_csv(os.path.join(input_path, demo_csv), skiprows=6)
                out_data.append(data_raw.iloc[:, 0])
                out_data.append(data_raw.iloc[:, 1])
                for obj in objs:
                    if no_unlabeled and 'unlabeled' in obj:
                        continue
                    labels.extend([f"{obj}.pose.x", f"{obj}.pose.y", f"{obj}.pose.z"]),
                    out_data.append(data_raw.iloc[:, objs[obj]['pose']["Position"]['X']])
                    out_data.append(data_raw.iloc[:, objs[obj]['pose']["Position"]['Y']])
                    out_data.append(data_raw.iloc[:, objs[obj]['pose']["Position"]['Z']])
                    if objs[obj]['type'] == 'Rigid Body':
                        labels.extend([f"{obj}.pose.qx", f"{obj}.pose.qy", f"{obj}.pose.qz", f"{obj}.pose.qw"])
                        out_data.append(data_raw.iloc[:, objs[obj]['pose']["Rotation"]['X']])
                        out_data.append(data_raw.iloc[:, objs[obj]['pose']["Rotation"]['Y']])
                        out_data.append(data_raw.iloc[:, objs[obj]['pose']["Rotation"]['Z']])
                        out_data.append(data_raw.iloc[:, objs[obj]['pose']["Rotation"]['W']])
                    for marker in objs[obj]['markers']:
                        labels.extend(
                            [f"{obj}.marker.{marker}.x", f"{obj}.marker.{marker}.y", f"{obj}.marker.{marker}.z"])
                        out_data.append(data_raw.iloc[:, objs[obj]['markers'][marker]['pose']["Position"]['X']])
                        out_data.append(data_raw.iloc[:, objs[obj]['markers'][marker]['pose']["Position"]['Y']])
                        out_data.append(data_raw.iloc[:, objs[obj]['markers'][marker]['pose']["Position"]['Z']])
                out_data = np.array(out_data).T
                out_list.append((out_data, labels))
                if save:
                    with open(os.path.join(out_path, demo_csv.replace('.csv', '_labels.pkl')), 'wb') as f:
                        pkl.dump(labels, f)
                    np.save(os.path.join(out_path, demo_csv.replace('csv', 'npy')), out_data)

    print('{} finished'.format(input_path.split('/')[-1]))
    return out_list


def data_clean_legacy(input_path: str, demo_csv: str, out_path: str):
    """
    Cleans the Optitrack data. legacy version
    Args:
        input_path (str): path to the Optitrack data.
        demo_csv (str): name of the csv file
        out_path (str): path to save the cleaned data for Manus

    Returns:
        csv_data (:pandas:`DataFrame`): cleaned data as a pandas dataframe
    """
    if 'Manus' in demo_csv:
        demo_path = os.path.join(input_path, demo_csv)
        out_file_path = os.path.join(out_path, demo_csv)
        rf.oslab.delete_lines(demo_path, out_file_path, 14)
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

    return csv_data


def data_clean_batch(input_dir: str):
    demos = os.listdir(input_dir)
    demos = sorted(demos)
    for demo in tqdm(demos):
        input_path = os.path.join(input_dir, demo)
        data_clean(input_path)


def export(input_dir: str):
    """
    Export rigid body motion data.
    :param input_dir: csv file path
    :return: [number of frames, number of rigid bodies, pose dimension = 7]
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
