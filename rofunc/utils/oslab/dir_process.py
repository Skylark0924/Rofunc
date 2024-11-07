#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com


import os
import pathlib
import shutil

import rofunc as rf


def create_dir(path, local_verbose=False):
    """
    Create the directory if it does not exist, can create the parent directories as well.

    Example::

        >>> import rofunc as rf
        >>> rf.oslab.create_dir('/home/ubuntu/Github/Rofunc/examples/data/felt/trial_1', local_verbose=True)

    :param path: the path of the directory
    :param local_verbose: if True, print the message
    :return:
    """
    if not pathlib.Path(path).exists():
        if local_verbose:
            rf.logger.beauty_print('{} not exist, created.'.format(path), type='info')
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


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
                (suffix is None or file.endswith(suffix)) and (prefix is None or file.startswith(prefix))]
    else:
        return [os.path.join(dir, file) for file in os.listdir(dir) if
                (suffix is None or file.endswith(suffix)) and (prefix is None or file.startswith(prefix))]


def delete_files(dir, file_list_to_delete, recursive=False):
    """
    Delete the appointed files in the directory.

    Example::

        >>> import rofunc as rf
        >>> rf.utils.delete_files('/home/ubuntu/Github/Rofunc/examples/data/felt', ['desktop.ini'], recursive=True)

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


def rename_files(dir, src_file_list=None, dst_file_list=None, recursive=False):
    """
    Rename the appointed files from source_file_list to target_file_list.

    Example::

        >>> import rofunc as rf
        >>> rf.utils.rename_files('/home/ubuntu/Github/Rofunc/examples/data/felt',
        ...                 source_file_list=['wiping_spiral_mocap_hand.csv', 'wiping_spiral_mocap_hand_rigid.csv'],
        ...                 target_file_list=['mocap_hand.csv', 'mocap_hand_rigid.csv', 'mocap_object.csv'],
        ...                 recursive=True)

    :param dir: directory path
    :param src_file_list: the list of file names need to be renamed, need suffix
    :param dst_file_list: the list of file names need to be renamed to, need suffix
    :param recursive: if True, rename the files in the subdirectories as well
    :return: None
    """
    all_files = list_absl_path(dir, recursive=recursive)
    for file in all_files:
        if os.path.basename(file) in src_file_list:
            target_name = dst_file_list[src_file_list.index(os.path.basename(file))]
            os.rename(file, os.path.join(os.path.dirname(file), target_name))
            rf.logger.beauty_print('File {} renamed from {} to {}.'.format(file, os.path.basename(file), target_name),
                                   type='info')


def shutil_files(files, src_dir, dst_dir):
    """
    Copy the appointed files from src_dir to dst_dir.

    Example::

        >>> import rofunc as rf
        >>> rf.utils.shutil_files(['mocap_hand.csv', 'mocap_hand_rigid.csv', 'mocap_object.csv'],
        ...                 src_dir='/home/ubuntu/Github/Rofunc/examples/data/felt/trial_1',
        ...                 dst_dir='/home/ubuntu/Github/Rofunc/examples/data/felt/trial_2')

    :param files: the list of file names need to be copied, need suffix
    :param src_dir: source directory path
    :param dst_dir: destination directory path
    :return:
    """
    rf.oslab.create_dir(dst_dir)

    for file in files:
        src = os.path.join(src_dir, file)
        file = file.split("/")[-1]
        dst = os.path.join(dst_dir, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            raise FileNotFoundError("File {} not found".format(src))
