from omegaconf import DictConfig, OmegaConf
from functools import reduce
import operator

OmegaConf.register_new_resolver("add", lambda *x: int(sum(x)))
OmegaConf.register_new_resolver("plus", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("multi", lambda *x: int(reduce(operator.mul, x, 1)))
