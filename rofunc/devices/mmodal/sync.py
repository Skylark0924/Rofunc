import sys
import datetime

import numpy as np
import pandas as pd

from rofunc.devices.xsens.process import load_mvnx
from rofunc.devices.zed.export import progress_bar
from pytz import timezone


# 1. get the time of each device
def get_optitrack_time(optitrack_input_path, time_reference='PM', time_zone='Etc/GMT-8'):
    '''
    Args:
        optitrack_time_path: the path of optitrack csv file
        time_reference: the time reference of optitrack csv file, 'PM' or 'AM'
        time_zone: the time zone of optitrack csv file, check your timezone name by pytz.all_timezones

    Returns: the time list of optitrack (np.array)
    '''
    # obtain the time of the first frame of optitrack
    first_columns = pd.read_csv(optitrack_input_path, nrows=0)
    optitrack_start_time = first_columns.columns[11][:23]
    year = int(optitrack_start_time[:4])
    month = int(optitrack_start_time[5:7])
    day = int(optitrack_start_time[8:10])
    if time_reference == 'PM':
        hour = int(optitrack_start_time[11:13]) + 12
    elif time_reference == 'AM':
        hour = int(optitrack_start_time[11:13])
    else:
        raise ValueError('time_reference should be PM or AM')
    minute = int(optitrack_start_time[14:16])
    second = int(optitrack_start_time[17:19])
    millisecond = int(optitrack_start_time[20:23])
    tz = timezone(time_zone)

    init_unix_time = datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)
    init_unix_time = init_unix_time.replace(tzinfo=tz).astimezone(timezone('UTC'))
    init_unix_time = int(init_unix_time.timestamp() * 1000)

    # obtain the time of each frame of optitrack
    timestamp_dataframe = pd.read_csv(optitrack_input_path, skiprows=6, usecols=['Time (Seconds)'])
    optitrack_timestamp = timestamp_dataframe.to_numpy()
    optitrack_timestamp = np.squeeze(optitrack_timestamp).tolist()
    optitrack_time = [int(init_unix_time + i * 1000) for i in optitrack_timestamp]

    return np.array(optitrack_time)


def get_xsens_time(mvnx_input_path):
    '''
    Args:
        mvnx_path: the path of xsens mvnx file
        init_unix_time: the time of the first frame of xsens (millisecond)

    Returns: the time list of xsens (np.array)
    '''
    mvnx_file = load_mvnx(mvnx_input_path)
    xsens_time = [int(i) for i in mvnx_file.file_data['frames']['time']]
    init_unix_time = mvnx_file.file_data['meta_data']['start_time']
    init_unix_time = int(init_unix_time)
    xsens_time = [init_unix_time + i for i in xsens_time]
    return np.array(xsens_time)


def get_zed_time(svo_input_path):
    import pyzed.sl as sl

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    nb_frames = zed.get_svo_number_of_frames()

    zed_timelist = []
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            zed_time = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            zed_time = zed_time.get_milliseconds()
            zed_timelist.append(zed_time)
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        elif zed.grab(rt_param) == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    return np.array(zed_timelist)


# time sync: get synced index table of each device
def data_sync(optitrack_input_path, mvnx_input_path, svo_input_path):
    zed_time_array = get_zed_time(svo_input_path)
    xsens_time_array = get_xsens_time(mvnx_input_path)
    optitrack_time_array = get_optitrack_time(optitrack_input_path)
    xsens_index_list = []
    optitrack_index_list = []

    for zed_time in zed_time_array:
        xsens_index_list.append((np.abs(zed_time - xsens_time_array)).argmin())
        optitrack_index_list.append((np.abs(zed_time - optitrack_time_array)).argmin())
    return xsens_index_list, optitrack_index_list
