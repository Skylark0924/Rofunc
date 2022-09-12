import rofunc as rf
import os


def export(filedir):
    xsens_filedir = os.path.join(filedir, 'xsens_mvnx')
    optitrack_filedir = os.path.join(filedir, 'optitrack_csv')
    zed_filedir = os.path.join(filedir, 'zed')

    if not os.path.exists(xsens_filedir):
        raise Exception('Please rename the xsens folder as \'xsens_mvnx\'')
    if not os.path.exists(optitrack_filedir):
        raise Exception('Please rename the xsens folder as \'optitrack_csv\'')
    if not os.path.exists(zed_filedir):
        raise Exception('Please rename the xsens folder as \'zed\'')

    rf.xsens.get_skeleton_batch(xsens_filedir)
    # rf.xsens.plot_skeleton(xsens_filedir)
    print('Xsens export finished!')

    rf.optitrack.process.data_clean_batch(optitrack_filedir)
    print('Optitrack export finished!')

    rf.zed.export_batch(zed_filedir, mode_lst=[1])
    print('Zed export finished!')


if __name__ == '__main__':
    import rofunc as rf

    rf.mmodal.export('/home/ubuntu/Data/2022_09_09_Taichi')
