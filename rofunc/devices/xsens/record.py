import numpy as np
import socket
import struct
import os
import threading
from typing import List
import rofunc as rf

from rofunc.utils.logger.beauty_logger import beauty_print

event = threading.Event()


def byte_to_str(data, n):
    fmt = "!{}c".format(n)
    char_array = struct.unpack(fmt, data)
    str_out = ""
    for c in char_array:
        s = c.decode("utf-8")
        str_out += s
    return str_out


def byte_to_float(data):
    return struct.unpack("!f", data)[0]


def byte_to_uint32(data):
    return struct.unpack("!I", data)[0]


def byte_to_uint16(data):
    return struct.unpack("!H", data)[0]


def byte_to_uint8(data):
    return struct.unpack("!B", data)[0]


class Header:
    def __init__(self, header):
        assert isinstance(header, list) and len(header) == 10
        self.ID_str = header[0]
        self.sample_counter = header[1]
        self.datagram_counter = header[2]
        self.item_counter = header[3]  # Amount of items (point/segments)
        self.time_code = header[4]  # Time since start measurement
        self.character_ID = header[5]  # Amount of people recorded at the same time
        self.body_segments_num = header[6]  # number of body segments measured
        self.props_num = header[7]  # Amount of property sensors
        self.finger_segments_num = header[8]  # Number of finger data segments
        self.payload_size = header[9]  # Size of the measurement excluding the header

    def __repr__(self):
        s = (
            "Header {}: \nsample_counter {}, datagram_counter {},\n"
            "item #{}, body segment #{}, prop #{}, finger segment #{}\n".format(
                self.ID_str,
                self.sample_counter,
                self.datagram_counter,
                self.item_counter,
                self.body_segments_num,
                self.props_num,
                self.finger_segments_num,
            )
        )
        return s

    @property
    def is_valid(self):
        if self.ID_str != "MXTP02":
            print(
                "XSensInterface: Current only support MXTP02, but got {}".format(
                    self.ID_str
                )
            )
            return False
        if (
                self.item_counter
                != self.body_segments_num + self.props_num + self.finger_segments_num
        ):
            print(
                "XSensInterface: Segments number in total does not match item counter"
            )
            return False
        if self.payload_size % self.item_counter != 0:
            print(
                "XSensInterface: Payload size {} is not dividable by item number {}".format(
                    self.payload_size, self.item_num
                )
            )
            return False
        return True

    @property
    def is_object(self):
        """External tracked object's datagram have less segments than body datagram."""
        return self.item_counter < 23

    @property
    def item_num(self):
        return self.item_counter

    @property
    def item_size(self):
        """Define how many bytes in a item"""
        return self.payload_size // self.item_num


class Datagram(object):
    def __init__(self, header, payload):
        self.header = header
        self.payload = payload

    @property
    def is_object(self):
        return self.header.is_object

    def decode_to_pose_array_msg(
            self, ref_frame, ref_frame_id=None, scaling_factor=1.0
    ):
        """
        Decode the bytes in the streaming data to pose array message.
        Args:
            ref_frame: str Reference frame name of the generated pose array message.
            ref_frame_id: None/int If not None, all poses will be shifted subject to
                             the frame with this ID. This frame should belong to the human.
            scaling_factor: float Scale the position of the pose if src_frame_id is not None.
                               Its value equals to the robot/human body dimension ratio

        Returns:
            pose_array_msg:
        """
        pose_array_msg = []

        for i in range(self.header.item_num):
            item = self.payload[
                   i * self.header.item_size: (i + 1) * self.header.item_size
                   ]
            pose_msg = self._decode_to_pose_msg(item)
            if pose_msg is None:
                return None
            pose_array_msg.append(pose_msg)

        return pose_array_msg

    @staticmethod
    def _decode_to_pose_msg(item: str) -> List:
        """
        Decode a type 02 stream to pose.
        Args:
            item: stream item from xsens

        Returns:
            pose: human pose
        """
        if len(item) != 32:
            print(
                "XSensInterface: Payload pose data size is not 32: {}".format(len(item))
            )
            return None
        x = byte_to_float(item[4:8])
        y = byte_to_float(item[8:12])
        z = byte_to_float(item[12:16])
        qw = byte_to_float(item[16:20])
        qx = byte_to_float(item[20:24])
        qy = byte_to_float(item[24:28])
        qz = byte_to_float(item[28:32])
        pose = [x, y, z, qx, qy, qz, qw]
        return pose


class XsensInterface(object):
    def __init__(
            self,
            udp_ip,
            udp_port,
            ref_frame,
            scaling=1.0,
            buffer_size=4096,
            **kwargs  # DO NOT REMOVE
    ):
        super(XsensInterface, self).__init__()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self._sock.bind((udp_ip, udp_port))

        self._buffer_size = buffer_size

        self._body_frames = {
            "pelvis": 0,
            "l5": 1,
            "l3": 2,
            "t12": 3,
            "t8": 4,
            "neck": 5,
            "head": 6,
            "right_shoulder": 7,
            "right_upper_arm": 8,
            "right_forearm": 9,
            "right_hand": 10,
            "left_shoulder": 11,
            "left_upper_arm": 12,
            "left_forearm": 13,
            "left_hand": 14,
            "right_upper_leg": 15,
            "right_lower_leg": 16,
            "right_foot": 17,
            "right_toe": 18,
            "left_upper_leg": 19,
            "left_lower_leg": 20,
            "left_foot": 21,
            "left_toe": 22,
        }

        if ref_frame is not None:
            ref_frame_lowercase = ref_frame.lower()
            if ref_frame_lowercase in self._body_frames:
                self.ref_frame = ref_frame_lowercase
                self.ref_frame_id = self._body_frames[ref_frame_lowercase]
            elif ref_frame_lowercase == "" or ref_frame_lowercase == "world":
                print("XSensInterface: Reference frame is the world frame")
                self.ref_frame = "world"
                self.ref_frame_id = None
            else:
                print("XSensInterface: Using customized reference frame {}".format(ref_frame_lowercase))
                self.ref_frame = ref_frame_lowercase
                self.ref_frame_id = None
        else:
            print("XSensInterface: Reference frame is the world frame")
            self.ref_frame = "world"
            self.ref_frame_id = None

        self.scaling_factor = scaling
        self.header = None
        self.object_poses = None
        self.body_segments_poses = None

    def get_datagram(self):
        """[Main entrance function] Get poses from the datagram."""
        data, _ = self._sock.recvfrom(self._buffer_size)
        datagram = self._get_datagram(data)
        if datagram is not None:
            pose_array_msg = datagram.decode_to_pose_array_msg(
                self.ref_frame, self.ref_frame_id
            )
            return pose_array_msg
        else:
            return None

    def save_file_thread(self, root_dir: str, exp_name: str) -> None:
        """
        save xsens motion data to the file
        Args:
            root_dir: root dictionary
            exp_name: dictionary saving the npy file, named according to time
        Returns:
            None
        """
        xsens_data = []
        while True:
            data = self.get_datagram()
            print(type(data))
            if type(data) == list:
                print(data)
                xsens_data.append(data)
            if event.isSet():
                np.save(root_dir + '/' + exp_name + '/' + 'xsens_data.npy', np.array(xsens_data))
                break

    @staticmethod
    def _get_header(data):
        """
        Get the header data from the received MVN Awinda datagram.
        :param data: Tuple From self._sock.recvfrom(self._buffer_size)
        :return: header
        """
        if len(data) < 24:
            print(
                "XSensInterface: Data length {} is less than 24".format(len(data))
            )
            return None
        id_str = byte_to_str(data[0:6], 6)
        sample_counter = byte_to_uint32(data[6:10])
        datagram_counter = struct.unpack("!B", data[10].to_bytes(1, 'big'))
        item_number = byte_to_uint8(data[11:12])
        time_code = byte_to_uint32(data[12:16])
        character_id = byte_to_uint8(data[16:17])
        body_segments_num = byte_to_uint8(data[17:18])
        props_num = byte_to_uint8(data[18:19])
        finger_segments_num = byte_to_uint8(data[19:20])
        # 20 21 are reserved for future use
        payload_size = byte_to_uint16(data[22:24])
        header = Header(
            [
                id_str,
                sample_counter,
                datagram_counter,
                item_number,
                time_code,
                character_id,
                body_segments_num,
                props_num,
                finger_segments_num,
                payload_size,
            ]
        )
        # rospy.logdebug(header.__repr__())
        return header

    def _get_datagram(self, data):
        header = self._get_header(data)
        if header is not None and header.is_valid:
            self.header = header
            return Datagram(header, data[24:])
        else:
            return None


def on_press(key):
    from pynput.keyboard import Key

    # When pressing the Esc key, stop recording
    if key == Key.esc:
        event.set()
        beauty_print("Esc button is pressed, stop recording", type="info")
        return False


def record(root_dir: str, exp_name: str, ip: str, port: int, ref_frame: str = None) -> None:
    """
    record xsens motion data
    :param root_dir: root directory
    :param exp_name: npy file location
    :param ip: ip address of server computer
    :param port: port number of optitrack server
    :param ref_frame: reference frame
    :return:
    """
    from pynput.keyboard import Listener

    if os.path.exists('{}/{}'.format(root_dir, exp_name)):
        raise Exception('There are already some files in {}, please rename the exp_name.'.format(
            '{}/{}'.format(root_dir, exp_name)))
    else:
        rf.oslab.create_dir('{}/{}'.format(root_dir, exp_name))
        beauty_print('Recording folder: {}/{}'.format(root_dir, exp_name), type='info')
        interface = XsensInterface(ip, port, ref_frame=ref_frame)
        listener = Listener(on_press=on_press)
        listener.start()
        xsens_thread = threading.Thread(target=interface.save_file_thread, args=(root_dir, exp_name))
        xsens_thread.start()
        xsens_thread.join()
        beauty_print('Xsens record finished', type='info')
