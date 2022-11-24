import rofunc as rf
import os


def get_rofunc_path():
    if not hasattr(rf, "__path__"):
        raise RuntimeError("rofunc package is not installed")
    rofunc_path = list(rf.__path__)[0]
    return rofunc_path


def check_ckpt_exist(ckpt_name):
    rofunc_path = get_rofunc_path()
    if os.path.exists(os.path.join(rofunc_path, "data/models/{}".format(ckpt_name))):
        return True
    else:
        return False
