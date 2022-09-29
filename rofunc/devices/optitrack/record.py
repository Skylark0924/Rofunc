import socket
import re
import os
import threading
import time
import numpy as np
import keyboard


def data_process(data):
    position = re.findall(r"Position\s*:\s*(.*)", data)[0]
    orientation = re.findall(r"Orientation\s*:\s*(.*)", data)[0]
    return eval(position), eval(orientation)


def opti_run(root_dir, exp_name, ip, port):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.settimeout(20.0)
    client.connect((ip, port))
    print("Connected to socket: {}:{}".format(ip, port))
    # receive optitrack data
    opti_data = np.array([])
    while True:
        utf_data = client.recv(1024).decode("utf-8")
        print(utf_data)
        raw_position, raw_orientation = data_process(utf_data)
        client.send("ok".encode("utf-8"))
        raw_pose = np.append(raw_position, raw_orientation)
        opti_data = np.append(opti_data, raw_pose, axis=0)

        # save optitrack data to npy file
        # if keyboard.read_key() == 's':
        np.save(root_dir + '/' + exp_name + '/' + 'opti_data.npy', opti_data)
        # break


def record(root_dir, exp_name, ip, port):
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
        os.mkdir('{}/{}'.format(root_dir, exp_name))
        print('Recording folder: {}/{}'.format(root_dir, exp_name))
        opti_thread = threading.Thread(target=opti_run, args=(root_dir, exp_name, ip, port))
        opti_thread.start()
        opti_thread.join()
        print('Optitrack record finished')


if __name__ == "__main__":
    root_dir = '/home/skylark/Data/optitrack_record'
    ip = '192.168.13.118'
    port = 6688

    from datetime import datetime
    exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    # import rofunc as rf
    record(root_dir, exp_name, ip, port)
    # filename = '/home/skylark/Data/optitrack_record/20220929_134531/opti_data.npy'
    # data = np.load(filename).reshape((-1, 7))
    # print(data[7:])

