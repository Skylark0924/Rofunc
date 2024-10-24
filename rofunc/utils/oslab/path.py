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
import subprocess
import sys

import rofunc as rf


def get_rofunc_path(extra_path=None):
    """
    Get the path of the rofunc package.

    :return: the absolute path of the rofunc package
    """
    if not hasattr(rf, "__path__"):
        raise RuntimeError("rofunc package is not installed")
    rofunc_path = list(rf.__path__)[0]

    if extra_path is not None:
        rofunc_path = os.path.join(rofunc_path, extra_path)
    return rofunc_path


def get_elegantrl_path():
    """
    Get the path of the elegantrl package.

    :return: the absolute path of the elegantrl package
    """
    import elegantrl as erl
    if not hasattr(erl, "__path__"):
        raise RuntimeError("elegantrl package is not installed")
    elegantrl_path = list(erl.__path__)[0]
    return elegantrl_path


def check_ckpt_exist(ckpt_name):
    """
    Check if the checkpoint file exists.

    :param ckpt_name: the name of the checkpoint file
    :return: True if the checkpoint file exists, False otherwise
    """
    rofunc_path = get_rofunc_path()
    if os.path.exists(os.path.join(rofunc_path, "learning/pre_trained_models/{}".format(ckpt_name))):
        return True
    else:
        return False


def is_absl_path(path):
    """
    Check if the path is an absolute path.

    :param path: the path to be checked
    :return: True if the path is an absolute path, False otherwise
    """
    return os.path.isabs(path)


def check_package_exist(package_name: str):
    """
    Check if the package is installed, if not, install the package.

    :param package_name: the name of the package
    """
    try:
        __import__(package_name)
    except ImportError:
        package_name = package_name.replace("_", "-")

        print(f"{package_name} is not installed, installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} is installed successfully!")
