import os

import h5py
import json
import torch
from robomimic.envs.env_base import EnvType
from typing import *

def write_to_hdf5(cfg: Dict[str, Any], actions: torch.Tensor,
                  rewards: Dict[str, torch.Tensor], observations: Dict[str, torch.Tensor],
                  dones: torch.Tensor) -> None:
    hdf5_dir = cfg["demo"]["collection"]["dir"]
    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)
    hdf5_path = hdf5_dir + f"/{cfg['task_name']}.hdf5"
    f = h5py.File(hdf5_path, "a")
    if len(f.keys()) == 0:
        grp = f.create_group("data")
        grp.attrs["total"] = 0
        env_meta = dict(
            type=EnvType.GYMGRASP_TYPE,
            env_name=cfg["task_name"],
            env_kwargs={"task": cfg["task"],
                        "task_name": cfg["task_name"],
                        "sim_device": cfg["sim_device"],
                        "rl_device": cfg["rl_device"],
                        "graphics_device_id": cfg["graphics_device_id"],
                        "headless": cfg["headless"],
                        "multi_gpu": cfg["multi_gpu"]}
        )
        grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
        demo_group = grp.create_group("demo_0")
    else:
        grp = f["data"]
        num_demos = len(grp.keys())
        demo_group = grp.create_group(f"demo_{num_demos}")
    demo_group.attrs["seed"] = cfg["seed"]
    demo_group.attrs["num_samples"] = actions.shape[0]
    demo_group.create_dataset("actions", data=actions)
    demo_group.create_dataset("sparse_rewards", data=rewards["sparse"].flatten().cpu())
    demo_group.create_dataset("dense_rewards", data=rewards["dense"].flatten().cpu())
    demo_group.create_dataset("dones", data=dones.flatten().cpu())
    obs_group = demo_group.create_group("obs")
    next_obs_group = demo_group.create_group("next_obs")
    for k, v in observations.items():
        obs_group.create_dataset(k, data=v[:-1].cpu())
        next_obs_group.create_dataset(k, data=v[1:].cpu())
    grp.attrs["total"] += demo_group.attrs["num_samples"]
    print(f"Collected {demo_group.name}.")
    f.close()

def dict_key_equal_rec(key: str, old_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Recursively checks whether values of key are equal in two dictionaries."""
    differences = {}
    try:
        old_val = old_dict[key]
    except KeyError:
        differences[key] = "KeyError in old configuration"
        return False, differences
    try:
        new_val = new_dict[key]
    except KeyError:
        differences[key] = "KeyError in new configuration"
        return False, differences
    if type(old_val) == dict:
        differences[key] = {}
        for k in old_val.keys():
            equal, differences_k = dict_key_equal_rec(k, old_val, new_val)
            if not equal:
                differences[key][k] = differences_k[k]
        if differences[key] == {}: # no differences
            return True, differences
        else:
            return False, differences
    else:  # not dict
        equal = (old_val == new_val)
        if not equal:
            differences[key] = f"{old_val} -> {new_val}"
        return equal, differences

def check_config_equality(cfg: Dict[str, Any]) -> bool:
    hdf5_dir = cfg["demo"]["collection"]["dir"]
    if not os.path.exists(hdf5_dir):  # did not collect demos before
        return True
    hdf5_path = hdf5_dir + f"/{cfg['task_name']}.hdf5"
    f = h5py.File(hdf5_path, "a")
    if len(f.keys()) == 0:  # did not collect demos for this task before
        f.close()
        return True
    grp = f["data"]
    env_meta_old = json.loads(grp.attrs["env_args"])
    env_meta_new = dict(
        type=EnvType.GYMGRASP_TYPE,
        env_name=cfg["task_name"],
        env_kwargs={"task": cfg["task"],
                    "task_name": cfg["task_name"],
                    "sim_device": cfg["sim_device"],
                    "rl_device": cfg["rl_device"],
                    "graphics_device_id": cfg["graphics_device_id"],
                    "headless": cfg["headless"],
                    "multi_gpu": cfg["multi_gpu"]}
    )
    if env_meta_old == env_meta_new:
        return True
    else:
        for k in env_meta_old.keys():
            _, differences = dict_key_equal_rec(k, env_meta_old, env_meta_new)
        print(f"Detected differences between environment configurations:\n{json.dumps(differences, indent=4)}")
        return False