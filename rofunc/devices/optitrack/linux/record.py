import socket

import rospy

import nav_msgs
import geometry_msgs

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import re
import os

from rotools.utility.common import to_ros_pose, sd_pose, get_transform_same_target


def data_process(data):
    position = re.findall(r"Position\s*:\s*(.*)", data)[0]
    orientation = re.findall(r"Orientation\s*:\s*(.*)", data)[0]
    return eval(position), eval(orientation)


class OptiTrackClient(object):
    """Class for receiving rigid body tracking information from OptiTrack device."""

    def __init__(self, ip, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(20.0)  # in seconds
        self.client.connect(ip, port)
        print("Connected to socket: {}:{}".format(ip, port))

    def __del__(self):
        if isinstance(self._client, socket.socket):
            self._client.close()



def record(root_dir, exp_name, ip, port):

    if os.path.exists('{}/{}'.format(root_dir, exp_name)):
        raise Exception('There are already some files in {}, please rename the exp_name.'.format(
            '{}/{}'.format(root_dir, exp_name)))
    else:
        os.mkdir('{}/{}'.format(root_dir, exp_name))
        print('Recording folder: {}/{}'.format(root_dir, exp_name))

    optitrackclient = OptiTrackClient(ip, port)
    utf_data = optitrackclient.client.recv(1024).decode("utf-8")
    raw_position, raw_orientation = data_process(utf_data)
    raw_pose = sd_pose(raw_position + raw_orientation, check=True)
    optitrackclient._client.send("ok".encode("utf-8"))


    