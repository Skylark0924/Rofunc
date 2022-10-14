from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
import hydra
from rofunc.config.custom_resolvers import *


def get_config(config_path=None, config_name=None) -> DictConfig:
    # reset current hydra config if already parsed (but not passed in here)
    if HydraConfig.initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    if config_path is not None and config_name is not None:
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(config_name=config_name)
    else:
        with initialize(config_path="./", version_base=None):
            cfg = compose(config_name="lqt")
    print(OmegaConf.to_yaml(cfg))
    return cfg
