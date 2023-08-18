import os
import rofunc as rf


def list_absl_path(dir, recursive=False, prefix=None, suffix=None):
    """
    Get the absolute path of each file in the directory.

    Example::

        >>> list_absl_path('/home/ubuntu/Github/Rofunc/examples/data/felt', recursive=True)
        ['/home/ubuntu/Github/Rofunc/examples/data/felt/trial_1/mocap_hand_rigid.npy',
            '/home/ubuntu/Github/Rofunc/examples/data/felt/trial_1/mocap_object_rigid.npy',
            ...]

    :param dir: directory path
    :param recursive: if True, list the files in the subdirectories as well
    :param prefix: if not None, only list the files with the appointed prefix
    :param suffix: if not None, only list the files with the appointed suffix
    :return: list
    """
    if recursive:
        return [os.path.join(root, file) for root, dirs, files in os.walk(dir) for file in files if
                suffix is None or file.endswith(suffix) and prefix is None or file.startswith(prefix)]
    else:
        return [os.path.join(dir, file) for file in os.listdir(dir) if
                suffix is None or file.endswith(suffix) and prefix is None or file.startswith(prefix)]


def delete_files(dir, file_list_to_delete, recursive=False):
    """
    Delete the appointed files in the directory.

    Example::

        >>> delete_files('/home/ubuntu/Github/Rofunc/examples/data/felt', ['desktop.ini'], recursive=True)

    :param dir: directory path
    :param file_list_to_delete: the list of file names need to be deleted, need suffix
    :param recursive: if True, delete the files in the subdirectories as well
    :return: None
    """
    all_files = list_absl_path(dir, recursive=recursive, )
    for file in all_files:
        if os.path.basename(file) in file_list_to_delete:
            os.remove(file)
            rf.logger.beauty_print('File {} deleted.'.format(file), type='info')


def rename_files(dir, source_file_list=None, target_file_list=None, recursive=False):
    """
    Rename the appointed files from source_file_list to target_file_list.

    Example::

        >>> rename_files('/home/ubuntu/Github/Rofunc/examples/data/felt', \
                    source_file_list=['wiping_spiral_mocap_hand.csv', 'wiping_spiral_mocap_hand_rigid.csv'], \
                    target_file_list=['mocap_hand.csv', 'mocap_hand_rigid.csv', 'mocap_object.csv'], \
                    recursive=True)

    :param dir: directory path
    :param source_file_list: the list of file names need to be renamed, need suffix
    :param target_file_list: the list of file names need to be renamed to, need suffix
    :param recursive: if True, rename the files in the subdirectories as well
    :return: None
    """
    all_files = list_absl_path(dir, recursive=recursive)
    for file in all_files:
        if os.path.basename(file) in source_file_list:
            target_name = target_file_list[source_file_list.index(os.path.basename(file))]
            os.rename(file, os.path.join(os.path.dirname(file), target_name))
            rf.logger.beauty_print('File {} renamed from {} to {}.'.format(file, os.path.basename(file), target_name),
                                   type='info')
