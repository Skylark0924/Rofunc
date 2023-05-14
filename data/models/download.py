import gdown
import json
import os
from rofunc.utils.file.path import get_rofunc_path, check_ckpt_exist
from rofunc.utils.logger.beauty_logger import beauty_print


def download_ckpt(url, name, md5):
    beauty_print("Downloading pretrained model {}".format(name), level=2)
    ckpt_path = os.path.join(get_rofunc_path(), 'data/models/{}'.format(name))
    gdown.cached_download(url, ckpt_path, md5=md5)
    return ckpt_path


def model_zoo(name, zoo_path="config/learning/model_zoo.json"):
    zoo = json.load(open(os.path.join(get_rofunc_path(), zoo_path)))
    if name in zoo:
        if check_ckpt_exist(zoo[name]['name']):
            beauty_print("Pretrained model {} already exists".format(name), level=2)
            ckpt_path = os.path.join(get_rofunc_path(), "data/models/{}".format(zoo[name]['name']))
        else:
            ckpt_path = download_ckpt(**zoo[name])
    else:
        raise ValueError(f"Model name {name} not found in model zoo")
    return ckpt_path


