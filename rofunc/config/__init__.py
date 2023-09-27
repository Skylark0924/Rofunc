from __future__ import absolute_import

from omegaconf import OmegaConf
from functools import reduce
import operator

OmegaConf.register_new_resolver("add", lambda *x: int(sum(x)))
OmegaConf.register_new_resolver("plus", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("multi", lambda *x: int(reduce(operator.mul, x, 1)))

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

from .utils import get_config, print_config, omegaconf_to_dict, dict_to_omegaconf, get_sim_config
