import rofunc as rf
import elegantrl as erl
import os
import shutil


def get_rofunc_path():
    if not hasattr(rf, "__path__"):
        raise RuntimeError("rofunc package is not installed")
    rofunc_path = list(rf.__path__)[0]
    return rofunc_path


def get_elegantrl_path():
    if not hasattr(erl, "__path__"):
        raise RuntimeError("elegantrl package is not installed")
    elegantrl_path = list(erl.__path__)[0]
    return elegantrl_path


def check_ckpt_exist(ckpt_name):
    rofunc_path = get_rofunc_path()
    if os.path.exists(os.path.join(rofunc_path, "data/models/{}".format(ckpt_name))):
        return True
    else:
        return False


def shutil_exp_files(files, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file in files:
        src = os.path.join(src_dir, file)
        file = file.split("/")[-1]
        dst = os.path.join(dst_dir, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            raise FileNotFoundError("File {} not found".format(src))
