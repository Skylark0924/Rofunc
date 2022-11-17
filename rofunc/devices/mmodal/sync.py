import numpy as np


# 1. get the time of each device
def get_optitrack_time():
    pass


def get_xsens_time():
    pass


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

