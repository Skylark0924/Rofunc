from typing import Dict

from omegaconf import DictConfig


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """
    Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation.
    """
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret
