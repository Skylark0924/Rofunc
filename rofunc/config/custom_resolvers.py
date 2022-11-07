from omegaconf import DictConfig, OmegaConf
from functools import reduce
import operator

try:
    OmegaConf.register_new_resolver("add", lambda *x: int(sum(x)))
except Exception as e:
    pass
try:
    OmegaConf.register_new_resolver("plus", lambda x, y: int(x) + int(y))
except Exception as e:
    pass
try:
    OmegaConf.register_new_resolver("multi", lambda *x: int(reduce(operator.mul, x, 1)))
except Exception as e:
    pass
try:
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
except Exception as e:
    pass
try:
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
except Exception as e:
    pass
try:
    OmegaConf.register_new_resolver('if', lambda condition, a, b: a if condition else b)
except Exception as e:
    pass
try:
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
except Exception as e:
    pass