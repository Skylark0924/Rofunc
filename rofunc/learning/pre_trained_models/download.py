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

import gdown
import json
import os
from rofunc.utils.oslab.path import get_rofunc_path, check_ckpt_exist
from rofunc.utils.logger.beauty_logger import beauty_print


def download_ckpt(url, name, md5=None):
    beauty_print("Downloading pretrained model {}".format(name), type="info")
    ckpt_path = os.path.join(get_rofunc_path(), 'learning/pre_trained_models/{}'.format(name))
    gdown.cached_download(url, ckpt_path)
    return ckpt_path


def model_zoo(name, zoo_path="config/learning/model_zoo.json"):
    zoo = json.load(open(os.path.join(get_rofunc_path(), zoo_path)))
    if name in zoo:
        if check_ckpt_exist(zoo[name]['name']):
            beauty_print("Pretrained model {} already exists".format(name), type="info")
            ckpt_path = os.path.join(get_rofunc_path(), "learning/pre_trained_models/{}".format(zoo[name]['name']))
        else:
            ckpt_path = download_ckpt(**zoo[name])
    else:
        raise ValueError(f"Model name {name} not found in model zoo")
    return ckpt_path
