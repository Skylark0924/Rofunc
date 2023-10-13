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

import gdown
import json
import os
from rofunc.utils.oslab.path import get_rofunc_path, check_ckpt_exist
from rofunc.utils.logger.beauty_logger import beauty_print


def download_ckpt(url, name, md5):
    beauty_print("Downloading pretrained model {}".format(name), type="info")
    ckpt_path = os.path.join(get_rofunc_path(), 'learning/pre_trained_models/{}'.format(name))
    gdown.cached_download(url, ckpt_path, md5=md5)
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


