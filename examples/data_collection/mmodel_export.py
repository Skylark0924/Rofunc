"""
Multimodal data fusion
==========================

This example shows how to synchronise and use multimodal data.
"""

import sys
import time
import argparse
import numpy as np
import rofunc as rf
import open3d as o3d
import os.path as osp

from rofunc.utils.datalab.data_sampler.utils import pcd_concat
from rofunc.utils.datalab.data_sampler import XsensDataHolder, OptitrackDataHolder, MultimodalDataHandler

# Bounds of the demonstrations in the data.
# (start_frame, end_frame)
# The frames number are from original optitrack data, can be got from plot_objects
EXPS = [
    [500, 900],
    [6300, 7200]
]


def clean_exp(exp_raw: np.array, labels: list, data_labels: list):
    new_name = ['t7', 't1', 't4', 'rh', 'rl', 'lh', 'll']
    new_data = [exp_raw[:, :2]]
    new_labels = ['frame', 'time']
    for i, l in enumerate(labels):
        if l.startswith('left'):
            new_labels.append(l)
            new_data.append(exp_raw[:, i][:, None])
    for i, l in enumerate(data_labels):
        new_labels.extend([f'{new_name[i]}.pose.{c}' for c in ['x', 'y', 'z']])
        ptr = labels.index(f'unlabeled {l}.pose.x')
        new_data.append(exp_raw[:, ptr:ptr + 3])
    new_data = np.concatenate(new_data, axis=1)

    return [new_data, new_labels]


def get_one_frame(frame_number, data_handler):
    out_pcds = data_handler.to_pcd(frame_number, show_holder=[True, True])
    out_pcds = pcd_concat(out_pcds)

    return out_pcds


def main(args):
    anim_pcd = []
    data_holders = []

    # Optitrack
    print("Loading Optitrack data...")
    objs, meta = rf.optitrack.get_objects(f"{args.data_path}/optitrack_data")
    if len(objs) > 1:
        print('More than one optitrack file found. Using the first one.')
    objs = objs[0]
    meta = meta[0]

    del_objs = []
    for obj in objs:
        if obj.startswith('unlabeled'):
            del_objs.append(obj)

    for o in del_objs:
        del objs[o]

    # To see the data, uncomment the following line
    # print(f'Optitrack objects: {objs}')
    # rf.optitrack.plot_objects(args.data_path, objs, meta)

    ot_full, ot_labels = rf.optitrack.data_clean(f"{args.data_path}/optitrack_data", legacy=False, objs=objs)[0]

    ot_dh = OptitrackDataHolder(meta, ot_labels, ot_full)
    data_holders.append(ot_dh)

    # Xsens
    print("Loading Xsens data...")
    xsens_dh = XsensDataHolder(f"{args.data_path}/xsens_data")
    data_holders.append(xsens_dh)

    tstep = 1 / 100

    # tstep in ms
    mmh = MultimodalDataHandler(tstep=tstep * 1000, data_holders=data_holders)
    sp_start = time.time()
    ts = mmh.sample()
    print(f"Sampling took {time.time() - sp_start:.2f}s")
    # Sampling can be long for long demonstrations

    # Data from XSens is subject to sensor drift.
    # sampled_match updates the linear transformation of the XSens data holder so that it
    # matches the Optitrack data
    ot_idx = mmh.data_holders[0].labels.index('left.pose.x')
    ot_obj = mmh.sampled_data[0][:, ot_idx:ot_idx + 7]
    ot_obj = mmh.data_holders[0].lin_trans(ot_obj)
    mmh.data_holders[1].sampled_match('LeftHand', ot_obj[args.frame_number, :3][None, :])
    pcd = get_one_frame(args.frame_number, mmh)
    o3d.visualization.draw_geometries([pcd])

    # Matching over just one frame is not enough to have the data match correctly for the
    # whole length of demonstrations usually.
    # See below

    exp_data = []
    exp_params = []
    sampled_bounds = []
    tracked_pts = ['left_finger_LeftThirdDP', 'right_finger_RightThirdDP']
    colors = []
    for i, e in enumerate(EXPS):
        print(f"EXP - {i}")
        # Find the index of bounds as the re-sampling changes the indices
        sampled_bounds.append(
            (mmh.data_holders[0].orig_fn_to_sampled(e[0]),
             mmh.data_holders[0].orig_fn_to_sampled(e[1]))
        )
        data = mmh.sampled_data[0][sampled_bounds[-1][0]:sampled_bounds[-1][1], :]
        labels = mmh.data_holders[0].labels

        mmh.data_holders[1].sampled_match('LeftHand',
                                          ot_obj[:, :3],
                                          fnb=sampled_bounds[-1][0],
                                          bnds=sampled_bounds[-1])
        # Matching in this way provide better results than just matching on one frame
        # Over the lenght of the demonstration

        xs_data = []
        for l, (imin, imax) in mmh.data_holders[1].labels.items():
            if imax - imin == 7:
                # Optitrack is the reference frame
                # Xsens data needs to be linear transformed to match it
                _data = mmh.data_holders[1].lin_trans(
                    mmh.sampled_data[1][sampled_bounds[-1][0]:sampled_bounds[-1][1],
                    imin:imax])
            else:
                _data = mmh.sampled_data[1][sampled_bounds[-1][0]:sampled_bounds[-1][1],
                        imin:imax]
            xs_data.append(_data)
        xs_data = np.concatenate(xs_data, axis=1)
        data = np.concatenate([data, xs_data], axis=1)
        print(data.shape)
        # Play with your data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, nargs='?',
                        default='../../data/MULTIMODAL',
                        help='Path to data folder')
    parser.add_argument('--frame_number', type=int, nargs='?',
                        default=0,
                        help='Frame to visualize')
    args = parser.parse_args()
    
    if not osp.exists(args.data_path):
        print(f"Argument data_path ({args.data_path}) points to invalid folder. Please make sure to point to Rofunc/examples/data/MULTIMODAL.")
        print("Exiting...")
        sys.exit(1)

    main(args)
