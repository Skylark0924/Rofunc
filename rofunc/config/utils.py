from typing import Dict

from omegaconf import DictConfig


def omegaconf_to_dict(config: DictConfig) -> Dict:
    """
    Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation.
    """
    d = {}
    for k, v in config.items():
        d[k] = omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
    return d
