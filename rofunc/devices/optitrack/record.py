import socket
import re
import os
import threading
import time

def data_process(data):
    position = re.findall(r"Position\s*:\s*(.*)", data)[0]
    orientation = re.findall(r"Orientation\s*:\s*(.*)", data)[0]
    return eval(position), eval(orientation)


def opti_run(root_dir, exp_name, ip, port):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(20.0)
    client.connect((ip, port))
    print("Connected to socket: {}:{}".format(ip, port))
    # receive optitrack data
    while True:
        utf_data = client.recv(1024).decode("utf-8")
        raw_position, raw_orientation = data_process(utf_data)
        client.send("ok".encode("utf-8"))

        # save optitrack data to npy file


def record(root_dir, exp_name, ip, port):
    if os.path.exists('{}/{}'.format(root_dir, exp_name)):
        raise Exception('There are already some files in {}, please rename the exp_name.'.format(
            '{}/{}'.format(root_dir, exp_name)))
    else:
        os.mkdir('{}/{}'.format(root_dir, exp_name))
        print('Recording folder: {}/{}'.format(root_dir, exp_name))
        opti_thread = threading.Thread(target=opti_run, args=(root_dir, exp_name, ip, port))
        opti_thread.start()


if __name__ == "__main__":
    root_dir = './'
    ip = ''
    port = ''

    from datetime import datetime
    exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    import rofunc as rf
    rf.optitrack.record(root_dir, exp_name, ip, port)
