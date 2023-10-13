import ast
import json
import os
from tqdm import tqdm
import pathlib

import numpy as np
import pandas as pd
import rofunc as rf
from .src.load_mvnx import load_mvnx
from rofunc.utils.logger.beauty_logger import beauty_print


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
    rf.oslab.create_dir(json_root_path)

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


def export(mvnx_path, output_type='segment', output_dir=None):
    """
    Export data from .mvnx file to skeleton data
    :param mvnx_path:
    :param output_type: type of output data, support segment or joint
    :param output_dir: specific output directory
    :return:
    """
    if mvnx_path.endswith('.mvnx'):
        mvnx_file = load_mvnx(mvnx_path)
    else:
        raise Exception('Wrong file type, only support .mvnx')

    if output_dir is None:
        output_dir = mvnx_path.split('.mvnx')[0]
        output_dir = os.path.join(output_dir, output_type)
    rf.oslab.create_dir(output_dir)
    rf.logger.beauty_print('Save .npys in {}'.format(output_dir), type="info")

    if output_type == 'segment':
        segment_count = mvnx_file.segment_count
        dim = mvnx_file.frame_count

        for idx in range(segment_count):
            segment_name = mvnx_file.segment_name_from_index(idx)
            segment_pos = mvnx_file.get_segment_pos(idx)
            segment_ori = mvnx_file.get_segment_ori(idx)

            label = segment_name
            pose = np.hstack((np.array(segment_pos), np.array(segment_ori)))
            assert dim == pose.shape[0]
            np.save(os.path.join(output_dir, "{}_{}.npy".format(idx, label)), pose)

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
    elif output_type == 'joint':
        joint_count = mvnx_file.joint_count
        dim = mvnx_file.frame_count

        for idx in range(joint_count):
            joint_name = mvnx_file.joint_name_from_index(idx)
            joint_angle = mvnx_file.get_joint_angle(idx)

            label = joint_name
            assert dim == np.array(joint_angle).shape[0]
            np.save(os.path.join(output_dir, "{}_{}.npy".format(idx, label)), joint_angle)

        ergo_joint_count = mvnx_file.ergo_joint_count
        for idx in range(ergo_joint_count):
            ergo_joint_name = mvnx_file.ergo_joint_name_from_index(idx)
            ergo_joint_angle = mvnx_file.get_ergo_joint_angle(idx)

            label = ergo_joint_name
            assert dim == np.array(ergo_joint_angle).shape[0]
            np.save(os.path.join(output_dir, "ergo_{}_{}.npy".format(idx, label)), ergo_joint_angle)
    else:
        raise Exception('Wrong output type, only support segment or joint')


def export_time(mvnx_path, output_dir=None):
    if mvnx_path.endswith('mvnx'):
        mvnx_file = load_mvnx(mvnx_path)
    else:
        raise Exception('Wrong file type, only support .mvnx')

    if output_dir is None:
        output_dir = mvnx_path.split('.mvnx')[0]
    rf.oslab.create_dir(output_dir)

    time = [int(i) for i in mvnx_file.file_data['frames']['ms']]
    np.save(os.path.join(output_dir, "ms.npy"), np.array(time))


def export_batch(mvnx_dir, output_type='segment'):
    mvnxs = os.listdir(mvnx_dir)
    for mvnx in tqdm(mvnxs):
        if mvnx.split('.')[-1] == 'mvnx':
            mvnx_path = os.path.join(mvnx_dir, mvnx)
            export(mvnx_path, output_type)
            export_time(mvnx_path)
