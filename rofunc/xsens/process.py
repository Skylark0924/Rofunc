import ast
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from rofunc.xsens.src.load_mvnx import load_mvnx


def data_clean(input_path, output_dir):
    """
    Example:
        from rofunc.xsens.process import data_clean

        input_path = '/home/ubuntu/Data/06_24/Xsens_postprocess/dough_01.csv'
        output_json_dir = '../xsens_data'
        data_clean(input_path, output_json_dir)
    """
    file_name = input_path.split('/')[-1]
    if file_name.split('.')[-1] == 'csv':
        csv_data = pd.read_csv(input_path, sep=",")
        csv_data.to_json("tmp.json")
        with open("tmp.json", 'r') as f:
            data = json.load(f)
            data = list(data)[0]
            data = ast.literal_eval(data)
    else:
        raise Exception('Wrong file type, only support .csv')
    json_data = json.dumps(data, indent=4)
    with open(os.path.join(output_dir, "{}.json".format(file_name.split('.')[0])), 'w') as f:
        f.write(json_data)
    print('{} finished!'.format(file_name))


def data_clean_batch(input_dir, output_dir):
    """
    Example:
        from rofunc.xsens.process import data_clean_batch

        input_dir = '/home/ubuntu/Data/06_24/Xsens_postprocess'
        output_json_dir = '../xsens_data'
        data_clean_batch(input_dir, output_json_dir)
    """
    demos = os.listdir(input_dir)
    for demo in tqdm(demos):
        demo_path = os.path.join(input_dir, demo)
        data_clean(demo_path, output_dir)


def get_skeleton_from_json(json_path):
    """
    Example:
        from rofunc.xsens.process import get_skeleton_from_json

        json_path = '../xsens_data/dough_01.json'
        get_skeleton_from_json(json_path)
    """
    json_name = json_path.split('/')[-1].split('.')[0]
    json_root_path = json_path.split('.json')[0]
    if not os.path.exists(json_root_path):
        os.mkdir(json_root_path)

    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        raw_seg_data = raw_data['segmentData']
        raw_left_finger_data = raw_data['fingerDataLeft']
        raw_right_finger_data = raw_data['fingerDataRight']

        dim = len(raw_data['frame'])
        for key in raw_seg_data:
            label = key['label']
            pose = np.hstack((np.array(key['position']), np.array(key['orientation'])))
            assert dim == pose.shape[0]
            np.save(os.path.join(json_root_path, "{}.npy".format(label)), pose)

        for key in raw_left_finger_data:
            label = key['label']
            pose = np.hstack((np.array(key['positionFingersLeft']), np.array(key['orientationFingersLeft'])))
            assert dim == pose.shape[0]
            np.save(os.path.join(json_root_path, "left_finger_{}.npy".format(label)), pose)

        for key in raw_right_finger_data:
            label = key['label']
            pose = np.hstack((np.array(key['positionFingersRight']), np.array(key['orientationFingersRight'])))
            assert dim == pose.shape[0]
            np.save(os.path.join(json_root_path, "right_finger_{}.npy".format(label)), pose)

        print('{} data got!'.format(json_name))


def get_skeleton_from_json_batch(json_dir):
    """
    Example:
        from rofunc.xsens.process import get_skeleton_from_json_batch

        json_dir = '../xsens_data'
        get_skeleton_from_json_batch(json_dir)
    """
    jsons = os.listdir(json_dir)
    for json in tqdm(jsons):
        json_path = os.path.join(json_dir, json)
        get_skeleton_from_json(json_path)


def get_skeleton(mvnx_path, output_dir=None):
    if mvnx_path.split('.')[-1] == 'mvnx':
        mvnx_file = load_mvnx(mvnx_path)
    else:
        raise Exception('Wrong file type, only support .mvnx')

    if output_dir is None:
        output_dir = mvnx_path.split('.mvnx')[0]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            raise Exception('There are already some files in {}, please delete this directory.'.format(output_dir))
        print('Save .npys in {}'.format(output_dir))
    else:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            print('{} not exist, created.'.format(output_dir))

    segment_count = mvnx_file.segment_count
    dim = mvnx_file.frame_count

    for idx in range(segment_count):
        segment_name = mvnx_file.segment_name_from_index(idx)
        segment_pos = mvnx_file.get_segment_pos(idx)
        segment_ori = mvnx_file.get_segment_ori(idx)

        label = segment_name
        pose = np.hstack((np.array(segment_pos), np.array(segment_ori)))
        assert dim == pose.shape[0]
        np.save(os.path.join(output_dir, "{}.npy".format(label)), pose)

    for finger_segment_name in mvnx_file.file_data['finger_segments']['names']['left']:
        segment_pos = mvnx_file.get_finger_segment_pos('left', finger_segment_name)
        segment_ori = mvnx_file.get_finger_segment_ori('left', finger_segment_name)

        label = finger_segment_name
        pose = np.hstack((np.array(segment_pos), np.array(segment_ori)))
        assert dim == pose.shape[0]
        np.save(os.path.join(output_dir, "left_finger_{}.npy".format(label)), pose)

    for finger_segment_name in mvnx_file.file_data['finger_segments']['names']['right']:
        segment_pos = mvnx_file.get_finger_segment_pos('right', finger_segment_name)
        segment_ori = mvnx_file.get_finger_segment_ori('right', finger_segment_name)

        label = finger_segment_name
        pose = np.hstack((np.array(segment_pos), np.array(segment_ori)))
        assert dim == pose.shape[0]
        np.save(os.path.join(output_dir, "right_finger_{}.npy".format(label)), pose)

