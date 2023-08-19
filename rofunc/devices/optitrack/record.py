import socket
import re
import os
import threading
import numpy as np
from typing import Tuple, List
import rofunc as rf


def data_process(data: str):
    """
    Args:
        data: string received from optitrack win server

    Returns:
        position, orientation: position and orientation of the rigidbody
    """
    data = data.split('ID')
    print(len(data[1:]))
    position_list = []
    orientation_list = []
    for i in data[1:]:
        position = re.findall(r"Position\s*:\s*(.*)", i)[0]
        orientation = re.findall(r"Orientation\s*:\s*(.*)", i)[0]
        position_list.append(eval(position))
        orientation_list.append(eval(orientation))
    return position_list, orientation_list


def opti_run(root_dir: str, exp_name: str, ip: str, port: int) -> None:
    """
    Args:
        root_dir: root dictionary
        exp_name: dictionary saving the npy file, named according to time
        ip: ip address
        port: port

    Returns:
        None
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))
    print("Connected to socket: {}:{}".format(ip, port))
    opti_data = np.array([])
    while True:
        utf_data = client.recv(1024).decode("utf-8")
        raw_position, raw_orientation = data_process(utf_data)
        client.send("ok".encode("utf-8"))
        raw_pose = np.append(raw_position, raw_orientation)
        opti_data = np.append(opti_data, raw_pose, axis=0)
        np.save(root_dir + '/' + exp_name + '/' + 'opti_data.npy', opti_data)


def record(root_dir: str, exp_name: str, ip: str, port: int) -> None:
    """
    Args:
        root_dir: root directory
        exp_name: npy file location
        ip: ip address of server computer
        port: port number of optitrack server

    Returns: None
    """
    if os.path.exists('{}/{}'.format(root_dir, exp_name)):
        raise Exception('There are already some files in {}, please rename the exp_name.'.format(
            '{}/{}'.format(root_dir, exp_name)))
    else:
        rf.oslab.create_dir('{}/{}'.format(root_dir, exp_name))
        print('Recording folder: {}/{}'.format(root_dir, exp_name))
        opti_thread = threading.Thread(target=opti_run, args=(root_dir, exp_name, ip, port))
        opti_thread.start()
        opti_thread.join()
        print('Optitrack record finished')
