import gdown
import json
import os
from rofunc.utils.file.path import get_rofunc_path, check_ckpt_exist


def download_ckpt(url, name, md5):
    print("\033[1;35mDownloading pretrained model: {}...\033[0m".format(name))
    ckpt_path = os.path.join(get_rofunc_path(), 'data/models/{}'.format(name))
    gdown.cached_download(url, ckpt_path)
    return ckpt_path


def model_zoo(name, zoo_path="config/learning/model_zoo.json"):
    zoo = json.load(open(os.path.join(get_rofunc_path(), zoo_path)))
    if name in zoo:
        if check_ckpt_exist(zoo[name]['name']):
            print("\033[1;35mPretrained model {} already exists.\033[0m".format(name))
            ckpt_path = os.path.join(get_rofunc_path(), "data/models/{}".format(zoo[name]['name']))
        else:
            ckpt_path = download_ckpt(**zoo[name])
    else:
        raise ValueError(f"Model name {name} not found in model zoo")
    return ckpt_path


