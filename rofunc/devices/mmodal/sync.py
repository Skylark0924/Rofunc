import numpy as np
import pandas as pd
from rofunc.devices.xsens.process import load_mvnx
import datetime
# 1. get the time of each device
def get_optitrack_time(optitrack_time_path, init_unix_time):
    '''
    Args:
        optitrack_time_path: the path of optitrack csv file
        init_unix_time: the unix time of the first frame of optitrack (millisecond)

    Returns: the time list of optitrack (np.array)
    '''
    # obtain the time of the first frame of optitrack
    first_columns = pd.read_csv(optitrack_time_path, nrows = 0)
    optitrack_start_time = first_columns.columns[11][:23]
    year = int(optitrack_start_time[:4])
    month = int(optitrack_start_time[5:7])
    day = int(optitrack_start_time[8:10])
    hour = int(optitrack_start_time[11:13])
    minute = int(optitrack_start_time[14:16])
    second = int(optitrack_start_time[17:19])
    millisecond = int(optitrack_start_time[20:23])
    optitrack_start_time = datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)

    # obtain the time of each frame of optitrack
    optitrack_time = pd.read_csv(optitrack_time_path, skiprows = 7, header = None)
    optitrack_time = optitrack_time['MocapTime']
    optitrack_time = np.array(optitrack_time)


    optitrack_time = optitrack_time + optitrack_start_time
    return optitrack_time


def get_xsens_time(mvnx_path):
    '''
    Args:
        mvnx_path: the path of xsens mvnx file
        init_unix_time: the time of the first frame of xsens (millisecond)

    Returns: the time list of xsens (np.array)
    '''
    mvnx_file = load_mvnx(mvnx_path)
    xsens_time = [int(i) for i in mvnx_file.file_data['frames']['time']]
    init_unix_time = mvnx_file.file_data['meta_data']['start_time']
    xsens_time = init_unix_time + xsens_time
    return xsens_time



def get_zed_time():
    pass


# time sync: get synced index table of each device
def data_sync():
    zed_timelist = get_zed_time()
    xsens_timelist = get_xsens_time()
    optitrack_timelist = get_optitrack_time()
    xsens_index_list = []
    optitrack_index_list = []
    for zed_time in zed_timelist:
        xsens_index_list.append((np.abs(zed_time - xsens_timelist)).argmin())
        optitrack_index_list.append((np.abs(zed_time - optitrack_timelist)).argmin())
    return xsens_index_list, optitrack_index_list

