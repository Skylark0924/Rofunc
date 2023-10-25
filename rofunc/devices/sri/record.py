import time
import os
import socket
import struct
import threading
import numpy as np
from sys import getsizeof

import argparse
from datetime import datetime
import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print


def connect_and_set_sensor(ip: str, port: int, sample_rate: int, buff_size: int):
    """
    Args:
        ip: ip address
        port: port
        sample_rate:
        buff_size:

    Returns:
        None
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))
    beauty_print("Connected to socket: {}:{}".format(ip, port), type='info')

    set_sam_rate = 'AT+SMPF=' + str(sample_rate) + '\r\n'
    client.send(set_sam_rate.encode('utf-8'))  # 设置采样率
    data_raw = client.recv(buff_size)
    print(data_raw)

    client.send(b'AT+DCKMD=SUM\r\n')  # 设置数据校验方法（sum）
    data_raw = client.recv(buff_size)
    print(data_raw)

    client.send(b'AT+ADJZF=1;1;1;1;1;1\r\n')  # 传感器信号调零
    data_raw = client.recv(buff_size)
    print(data_raw)

    client.send(b'AT+GSD=\r\n')  # 连续上传数据包
    # client.send(b'AT+GOD=\r\n')  # 只上传一个包数据
    return client


def length_check(data_raw):
    head_index = 0
    if getsizeof(data_raw)/2 > 30:
        for i in range(30):
            if data_raw[i] == 0xAA and data_raw[i + 1] == 0x55:
                head_index = i
        frame_len = data_raw[head_index + 2] * 256 + data_raw[head_index + 3]
        return frame_len
    else:
        return 0


def sum_check(data_raw):
    check = data_raw[30]
    check_new = 0x00
    for i in range(6, 30):
        check_new = check_new + int(data_raw[i])
    if hex(check)[-2:] == hex(check_new)[-2:]:
        return True
    else:
        return False


def read_batch(client, buff_size, sample_rate, label):
    force_data = []
    data_number = 0
    time_and_force = np.zeros((6 + 1))
    time_start = time.time()
    while data_number < sample_rate:
        if time.time()-time_start > 10:
            # print('Read Data Timeout')
            beauty_print('Read Data Timeout', type='warning')
        data_raw = client.recv(buff_size)
        frame_len = length_check(data_raw)  # 长度校验（27）
        if frame_len == 27:
            force_sensor_data = struct.unpack('ffffff', data_raw[6:30])  # 依次为六轴力传感器数据
            if sum_check(data_raw) is True:  # 数据校验（sum）
                data_number = data_number + 1
                time_and_force[0] = time.time() - time_start
                time_and_force[1:] = force_sensor_data
                force_data.append(time_and_force.copy())
                if label is not None:
                    print(label + ':', force_sensor_data)
                    # if label == 'right':
                    #     print(label + ':', force_sensor_data)
                else:
                    print(force_sensor_data)
    return force_data


def save_file_thread1(root_path: str, exp_name: str, ip: str, port: int, sample_rate: int, buff_size: int, t: int,
                      label=None):
    """
    Args:
        root_path: root dictionary
        exp_name: dictionary saving the npy file, named according to time
        ip: ip address
        port: port
        sample_rate:
        buff_size:
        t:
        label:

    Returns:
        None
    """
    client = connect_and_set_sensor(ip, port, sample_rate, buff_size)
    force_data = read_batch(client, buff_size, sample_rate * t, label)
    force_sensor_data = np.transpose(force_data)
    client.close()
    if label is not None:
        exp_name_label = exp_name + '_' + label
        np.save(os.path.join(root_path, exp_name_label), force_sensor_data)
    else:
        np.save(os.path.join(root_path, exp_name), force_sensor_data)


def save_file_thread2(root_path: str, exp_name: str, ip: str, port: int, sample_rate: int, buff_size: int, t: int,
                      label: str):
    """
    Args:
        root_path: root dictionary
        exp_name: dictionary saving the npy file, named according to time
        ip: ip address
        port: port
        sample_rate:
        buff_size:
        t:
        label:

    Returns:
        None
    """
    client = connect_and_set_sensor(ip, port, sample_rate, buff_size)
    force_data = read_batch(client, buff_size, sample_rate * t, label)
    force_sensor_data = np.transpose(force_data)
    client.close()
    if label is not None:
        exp_name_label = exp_name + '_' + label
        np.save(os.path.join(root_path, exp_name_label), force_sensor_data)
    else:
        np.save(os.path.join(root_path, exp_name), force_sensor_data)


def record(ip1, ip2, port, sample_rate, t, root_path, exp_name, buff_size=38):
    """
    Args:
        ip1: IP address of an SRI force sensor.
        ip2: IP address of an SRI force sensor.
        port:
        sample_rate:
        t: number of batches (running time * sample_rate / samples_per_read)
        root_path: root dictionary
        exp_name: dictionary saving the npy file, named according to time
        buff_size:

    Returns: None
    """

    beauty_print('Recording folder: {}/{}'.format(root_path, exp_name), type='info')
    if ip2 is not None:
        label1 = 'left'
        label2 = 'right'
        sri_thread1 = threading.Thread(target=save_file_thread1,
                                       args=(root_path, exp_name, ip1, port, sample_rate, buff_size, t, label1))
        sri_thread2 = threading.Thread(target=save_file_thread2,
                                       args=(root_path, exp_name, ip2, port, sample_rate, buff_size, t, label2))
        sri_thread1.start()
        sri_thread2.start()
        sri_thread1.join()
        sri_thread2.join()
    else:
        sri_thread = threading.Thread(target=save_file_thread1,
                                      args=(root_path, exp_name, ip1, port, sample_rate, buff_size, t))
        sri_thread.start()
        sri_thread.join()
    beauty_print('SRI force sensor record finished', type='info')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--addr',
        dest='host1',
        default='192.168.0.110',
        help="IP address of the force sensor 1.")
    parser.add_argument(
        '-a2', '--addr2',
        dest='host2',
        default='192.168.0.111',
        # default=None,
        help="IP address of the force sensor 2.")
    args = parser.parse_args()

    root_dir = 'D:/'
    exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    rf.oslab.create_dir(root_dir)

    record(ip1=args.host1, ip2=args.host2, port=4008, sample_rate=200, t=5,
           root_path=root_dir, exp_name=exp_name)
