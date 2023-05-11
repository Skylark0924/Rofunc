from ..src import mvn
import warnings


class MvnxFileAccessor:

    @property
    def original_file_name(self):
        return self.file_data['meta_data']['original_filename']

    @property
    def actor_name(self):
        return self.file_data['meta_data']['name']

    @property
    def actor_color(self):
        return self.file_data['meta_data']['color']

    @property
    def profile(self):
        return self.file_data['meta_data']['scenario']

    @property
    def configuration(self):
        return self.file_data['meta_data']['configuration']

    @property
    def comments(self):
        return self.file_data['meta_data']['comments']

    @property
    def quality(self):
        return self.file_data['meta_data']['quality']

    @property
    def frame_count(self):
        return self._last_frame - self._first_frame

    @property
    def segment_count(self):
        return self.file_data['frames']['segment_count']

    @property
    def joint_count(self):
        return self.file_data['frames']['joint_count']

    @property
    def finger_joint_count(self):
        return self.file_data['frames']['finger_joint_count']

    @property
    def ergo_joint_count(self):
        ergo_joint_count = len(self.file_data['ergo_joints']['names']) if self.file_data[
                                                                              'ergo_joints'] is not None else None
        return ergo_joint_count  # 'ergoJointCount' non-existing in mvnx

    @property
    def sensor_count(self):
        return self.file_data['frames']['sensor_count']

    @property
    def frame_rate(self) -> int:
        if 'sample_rate' in self.file_data['meta_data']:
            return int(self.file_data['meta_data']['sample_rate'])
        else:
            warnings.warn('Using default sample rate of 240')
            return 240

    @property
    def recording_date(self) -> str:
        return self.file_data['meta_data']['rec_date']

    @property
    def version(self):
        return self.file_data['meta_data']['version']

    def __init__(self):
        self.file_data = {}
        self._all_frames_slice = None
        self._first_frame = 0
        self._last_frame = 0
        self._index_to_segment = {}
        self._index_to_joint = {}
        self._index_to_sensor = {}
        self._index_to_ergo_joint = {}

    def create_index_to_segment_dict(self):
        segment_index = 0
        for _, segment in self.file_data['segments']['elements'].items():
            if 'label' in segment:
                self._index_to_segment[segment_index] = segment['label']
            segment_index += 1

    def create_index_to_joint_dict(self):
        joint_index = 0
        for joint in self.file_data['joints']['elements']:
            if 'label' in joint:
                self._index_to_joint[joint_index] = joint['label']
            joint_index += 1

    def create_index_to_sensor_dict(self):
        sensor_index = 0
        for sensor in self.file_data['sensors']['elements']:
            if 'label' in sensor:
                self._index_to_sensor[sensor_index] = sensor['label']
            sensor_index += 1

    def create_index_to_ergo_joint_dict(self):
        ergo_joint_index = 0
        for ergo_joint in self.file_data['ergo_joints']['elements']:
            if 'label' in ergo_joint:
                self._index_to_ergo_joint[ergo_joint_index] = ergo_joint['label']
            ergo_joint_index += 1

    def set_frame_window(self, first_frame, last_frame):
        self._first_frame = first_frame
        self._last_frame = last_frame
        self._all_frames_slice = slice(self._first_frame, self._last_frame)

    def reset_frame_window(self):
        self._first_frame = 0
        self._last_frame = len(self.file_data['frames']['segment_data'])
        self._all_frames_slice = slice(self._first_frame, self._last_frame)

    def window_profile(self):
        """
        This method determines the filter profile (aka scenario) for the currently selected window on the
        data.
        :return: The profile name (singleLevel, multiLevel, ...) or 'mixed' if different profiles
        ere applied to the window.
        """
        if 'profiles' not in self.file_data['meta_data']:
            return self.profile
        else:
            profiles = set()
            for profile_spec in self.file_data['meta_data']['profiles']:
                if profile_spec[1] > self._first_frame and profile_spec[0] <= self._last_frame:
                    profiles.add(profile_spec[2])

            if len(profiles) == 1:
                return profiles.pop()
            else:
                return 'mixed'

    def frame_to_mapped_slice(self, frame) -> (slice, bool):
        # convert the frame parameter to a slice that is mapped to the current view on the file
        single_frame = False
        if not isinstance(frame, slice):
            if frame == mvn.FRAMES_ALL:
                # all frames slice already is mapped to the current view
                self.reset_frame_window()
                frame = self._all_frames_slice  # use the prepared 'all frames' slice
            else:
                # for a single frame, map the frame to the current view on all data by adding the virtual first frame
                frame = slice(self._first_frame + frame, self._first_frame + frame + 1)  # create a single frame slice
                single_frame = True
        else:
            # shift the slice to the current virtual 'view'
            start = frame.start + self._first_frame
            if frame.stop:
                stop = frame.stop + self._first_frame
            else:  # a slice with end given as None implies end at last frame
                stop = self._last_frame
            step = frame.step
            frame = slice(start, stop, step)

        return frame, single_frame

    def segment_name_from_index(self, segment_index):
        return self._index_to_segment[segment_index]

    def joint_name_from_index(self, joint_index):
        return self._index_to_joint[joint_index]

    def sensor_name_from_index(self, sensor_index):
        return self._index_to_sensor[sensor_index]

    def ergo_joint_name_from_index(self, ergo_joint_index):
        return self._index_to_ergo_joint[ergo_joint_index]

    def point_name_from_indices(self, segment_index, point_index):
        segment_name = self.segment_name_from_index(segment_index)
        segment = self.file_data['segments']['elements'][segment_name]
        return segment['info']['point_label_from_index'][point_index]

    """ Pose methods """

    def identity_pose_is_valid(self):
        return ('identity' in self.file_data) and \
            (self.identity_pose()['segments_counts'] > 0)

    def identity_pose_segment_pos(self, segment):
        return self.identity_pose()['segments'][segment]['pos_g']

    def identity_pose_segment_ori(self, segment):
        return self.identity_pose()['segments'][segment]['q_gb']

    def t_pose_is_valid(self):
        return ('tpose' in self.file_data) and \
            (self.t_pose()['segments_counts'] > 0)

    def t_pose_segment_pos(self, segment):
        return self.t_pose()['segments'][segment]['pos_g']

    def t_pose_segment_ori(self, segment):
        return self.t_pose()['segments'][segment]['q_gb']

    def identity_pose(self):
        return self.file_data['identity']

    def t_pose(self):
        return self.file_data['tpose']

    """ Segment methods """

    def get_segment_pos(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the position information for a segment

        :param segment: The index of the segment to return the data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with position values
        """

        return self.get_segment_data('pos', segment, frame, axis)

    def get_segment_ori(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the orientation information for a segment

        :param segment: The index of the segment to return the data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with orientation values
        """

        # For orientation data, if all axes requested, return all. If specific axis requested, shift index so that w,
        # x,y,z becomes 0,1,2,3
        axis = (axis + 1) % 4 if axis != mvn.AXIS_ALL else axis
        return self.get_segment_data('ori', segment, frame, axis)

    def get_segment_point_pos(self, segment, point):
        segment_name = mvn.SEGMENTS[segment]
        segment_info = self.file_data['segments'][segment_name]
        points = segment_info['info']['point_label_from_index']
        point_name = points[point]
        return segment_info['points_mvn'][point_name]

    def get_point_pos(self, segment, point):
        segment_name = mvn.SEGMENTS[segment]

        if segment_name == 'LeftFoot':
            point_name = mvn.POINTS_LEFT_FOOT[point]
        elif segment_name == 'RightFoot':
            point_name = mvn.POINTS_RIGHT_FOOT[point]
        elif segment_name == 'LeftToe':
            point_name = mvn.POINTS_LEFT_TOE[point]
        elif segment_name == 'RightToe':
            point_name = mvn.POINTS_RIGHT_TOE[point]

        return self.file_data['segments'][segment_name]['points_mvn'][point_name]

    def get_segment_vel(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the local velocity information for a segment

        :param segment: The index of the segment to return the data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with velocity values
        """

        return self.get_segment_data('vel', segment, frame, axis)

    def get_segment_acc(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the acceleration information for a segment

        :param segment: The index of the segment to return the data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with acceleration values
        """

        return self.get_segment_data('acc', segment, frame, axis)

    def get_segment_angular_vel(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the acceleration information for a segment

        :param segment: The index of the segment to return the data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with angular velocity values
        """

        return self.get_segment_data('ang_vel', segment, frame, axis)

    def get_segment_angular_acc(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the acceleration information for a segment

        :param segment: The index of the segment to return the data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with acceleration values
        """

        return self.get_segment_data('ang_acc', segment, frame, axis)

    def get_segment_data(self, data_field, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        return self.get_data('segment_data', data_field, segment, frame, axis)

    def get_finger_segment_pos(self, hand_idx, segment_name, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        return self.get_finger_segment_data('pos', hand_idx, segment_name, frame, axis)

    def get_finger_segment_ori(self, hand_idx, segment_name, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        return self.get_finger_segment_data('ori', hand_idx, segment_name, frame, axis)

    def get_finger_segment_data(self, data_field, hand_idx, segment_name, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        frame, is_single_frame = self.frame_to_mapped_slice(frame)
        if axis == mvn.AXIS_ALL:
            return_values = [value[hand_idx][segment_name][data_field] for value in
                             self.file_data['frames']['finger_segment_data'][frame]]
        else:
            return_values = [value[hand_idx][segment_name][data_field][axis] for value in
                             self.file_data['frames']['finger_segment_data'][frame]]

        return return_values[0] if is_single_frame else return_values

    """ Joint methods """

    def get_joint_angle(self, joint, frame=mvn.FRAMES_ALL, angle=mvn.ANGLE_ALL):
        joint_name = mvn.JOINTS[joint]
        data_set = 'joint_data'

        frame, is_single_frame = self.frame_to_mapped_slice(frame)

        if angle == mvn.ANGLE_ALL:
            return_values = [value[joint_name] for value in self.file_data['frames'][data_set][frame]]
        else:
            return_values = [value[joint_name][angle] for value in self.file_data['frames'][data_set][frame]]

        return return_values[0] if is_single_frame else return_values

    def get_joint_angle_xzy(self, joint, frame=mvn.FRAMES_ALL, angle=mvn.ANGLE_ALL):
        joint_name = mvn.JOINTS[joint]
        data_set = 'joint_data_xzy'

        frame, is_single_frame = self.frame_to_mapped_slice(frame)

        if angle == mvn.ANGLE_ALL:
            return_values = [value[joint_name] for value in self.file_data['frames'][data_set][frame]]
        else:
            return_values = [value[joint_name][angle] for value in self.file_data['frames'][data_set][frame]]

        return return_values[0] if is_single_frame else return_values

    def get_ergo_joint_angle(self, joint, frame=mvn.FRAMES_ALL, angle=mvn.ANGLE_ALL):
        joint_name = mvn.ERGO_JOINTS[joint]
        data_set = 'ergo_joint_data'

        frame, is_single_frame = self.frame_to_mapped_slice(frame)

        if angle == mvn.ANGLE_ALL:
            return_values = [value[joint_name] for value in self.file_data['frames'][data_set][frame]]
        else:
            return_values = [value[joint_name][angle] for value in self.file_data['frames'][data_set][frame]]

        return return_values[0] if is_single_frame else return_values

    """ Center of Mass methods """

    def get_center_of_mass_pos(self, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the position information for center of mass

        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The axis to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with position values
        """

        return self.get_segment_data('pos', mvn.SEGMENT_CENTER_OF_MASS, frame, axis)

    def get_center_of_mass_vel(self, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the velocity information for center of mass

        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The axis to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with position values
        """

        return self.get_segment_data('vel', mvn.SEGMENT_CENTER_OF_MASS, frame, axis)

    def get_center_of_mass_acc(self, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the acceleration information for center of mass

        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The axis to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with position values
        """

        return self.get_segment_data('acc', mvn.SEGMENT_CENTER_OF_MASS, frame, axis)

    """ Sensor methods """

    def get_sensor_ori(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the orientation information for a sensor

        :param segment: The index of the segment to return the sensor data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with orientation values
        """

        # if all axes requested, return all. If specific axis requested, shift index so that w,x,y,z becomes 0,1,2,3
        axis = (axis + 1) % 4 if axis != mvn.AXIS_ALL else axis
        return self.get_sensor_data('ori', segment, frame, axis)

    def get_sensor_free_acc(self, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        """
        Get the free acceleration information for a sensor

        :param segment: The index of the segment to return the sensor data for (Mvn.SEGMENT_.... )
        :param frame: Can be the index of the frame to return, a slice to return a range of frames
                      or Mvn.FRAMES_ALL (default) to return all frames
        :param axis: The frame number to return the data for (Mvn.AXIS_... ALL for all axes)
        :return: A single value, list, or list of lists with acceleration values
        """

        return self.get_sensor_data('acc', segment, frame, axis)

    def get_sensor_data(self, data_field, sensor_segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        return self.get_data('sensor_data', data_field, sensor_segment, frame, axis)

    """ Contact methods """

    def get_foot_contacts(self, frame):
        """
        Get the contacts for a frame
        :param frame: The frame, or a range of frames to retrieve the contacts for.
        :return: The contacts
        """
        frame, is_single_frame = self.frame_to_mapped_slice(frame)
        return_values = self.file_data['frames']['footContacts'][frame]
        return return_values[0] if is_single_frame else return_values

    def has_foot_contact(self, frame, foot_contact_flags=0):
        """
        Find out if the frame has a contact, optionally for a specific segment/point combo

        :param frame: The frame (or range) to retrieve the contacts for.
        :param foot_contact_flags: The specific contact to check for:
                                    mvn.FOOT_CONTACT_LEFT_HEEL,
                                    mvn.FOOT_CONTACT_LEFT_TOE,
                                    mvn.FOOT_CONTACT_RIGHT_HEEL or
                                    mvn.FOOT_CONTACT_RIGHT_TOE

                                 It is possible to combine contacts by summing the values, eg:
                                    mvn.FOOT_CONTACT_LEFT_HEEL + mvn.FOOT_CONTACT_LEFT_TOE
                                 This will return a contact if either of the flags has a contact

                                 Passing 0 (or nothing) will return True if there is any contact

        :return: Per frame True if a contact was found, False otherwise.
        """
        frame_contacts = self.get_foot_contacts(frame)  # frame will be shifted in the called method

        if isinstance(frame_contacts, int):
            if foot_contact_flags == 0:
                return frame_contacts > 0
            else:
                return True if (frame_contacts & foot_contact_flags) > 0 else False
        else:
            has_contacts = []
            for contacts in frame_contacts:
                if foot_contact_flags == 0:
                    has_contacts.append(frame_contacts > 0)
                else:
                    has_contacts.append(True if (contacts & foot_contact_flags) > 0 else False)
            return has_contacts

    """ Generic methods """

    def get_data(self, data_set, data_field, segment, frame=mvn.FRAMES_ALL, axis=mvn.AXIS_ALL):
        if segment == -1:
            segment_name = 'com'
        else:
            segment_name = mvn.SEGMENTS[segment]

        frame, is_single_frame = self.frame_to_mapped_slice(frame)
        if data_set == 'joint_data':
            return_values = [value[data_field] for value in self.file_data['frames'][data_set][frame]]
        elif axis == mvn.AXIS_ALL:
            return_values = [value[segment_name][data_field] for value in self.file_data['frames'][data_set][frame]]
        else:
            return_values = [value[segment_name][data_field][axis] for value in
                             self.file_data['frames'][data_set][frame]]

        return return_values[0] if is_single_frame else return_values
