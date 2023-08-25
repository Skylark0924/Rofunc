# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import rofunc as rf


def get_rofunc_path():
    """
    Get the path of the rofunc package.

    :return: the absolute path of the rofunc package
    """
    if not hasattr(rf, "__path__"):
        raise RuntimeError("rofunc package is not installed")
    rofunc_path = list(rf.__path__)[0]
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
