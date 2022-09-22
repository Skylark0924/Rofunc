import socket

import rospy

import nav_msgs
import geometry_msgs

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import re

from rotools.utility.common import to_ros_pose, sd_pose, get_transform_same_target


def data_process(data):
    position = re.findall(r"Position\s*:\s*(.*)", data)[0]
    orientation = re.findall(r"Orientation\s*:\s*(.*)", data)[0]
    return eval(position), eval(orientation)


class OptiTrackClient(object):
    """Class for receiving rigid body tracking information from OptiTrack device."""

    def __init__(self, kwargs):
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.settimeout(20.0)  # in seconds
        self._client.connect((kwargs["ip"], kwargs["port"]))
        rospy.loginfo("Connected to socket: {}:{}".format(kwargs["ip"], kwargs["port"]))

        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / kwargs["rate"]), self._socket_cb
        )

        self._advertise_dict = {}

        if kwargs["pose_topic"] is not None:
            topics = kwargs["pose_topic"]
            self.register_topic(topics, geometry_msgs.msg.Pose)

        if kwargs["odom_topic"] is not None:
            topics = kwargs["odom_topic"]
            self.register_topic(topics, nav_msgs.msg.Odometry)

        if kwargs["transform"] is not None:
            self._transform = sd_pose(kwargs["transform"], check=True)
        else:
            self._transform = sd_pose([0, 0, 0, 0, 0, 0, 1], check=True)

    def __del__(self):
        if isinstance(self._client, socket.socket):
            self._client.close()

    def register_topic(self, topics, msg_type):
        if isinstance(topics, str):
            publisher = self.create_publisher(topics, msg_type)
            self._advertise_dict[topics] = [msg_type, publisher]
        elif isinstance(topics, list) or isinstance(topics, tuple):
            for topic in topics:
                publisher = self.create_publisher(topic, msg_type)
                self._advertise_dict[topic] = [msg_type, publisher]
        else:
            raise NotImplementedError

    @staticmethod
    def create_publisher(topic_id, msg_type):
        return rospy.Publisher(topic_id, msg_type, queue_size=1)

    def _socket_cb(self, _):
        utf_data = self._client.recv(1024).decode("utf-8")
        raw_position, raw_orientation = data_process(utf_data)
        raw_pose = sd_pose(raw_position + raw_orientation, check=True)
        transformed_pose = to_ros_pose(
            get_transform_same_target(raw_pose, self._transform)
        )
        for _, entity in self._advertise_dict.items():
            msg_type, publisher = entity
            if msg_type is geometry_msgs.msg.Pose:
                msg = Pose()
                msg.position = transformed_pose.position
                msg.orientation = transformed_pose.orientation
            elif msg_type is nav_msgs.msg.Odometry:
                msg = Odometry()
                msg.pose.pose.position = transformed_pose.position
                msg.pose.pose.orientation = transformed_pose.orientation
            else:
                raise NotImplementedError
            publisher.publish(msg)
        self._client.send("ok".encode("utf-8"))
