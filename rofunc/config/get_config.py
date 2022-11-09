import hydra
import os
from hydra import compose, initialize
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from rofunc.config.custom_resolvers import *
from rofunc.utils.file.path import get_rofunc_path


def get_config(config_path=None, config_name=None, args=None, debug=False) -> DictConfig:
    # reset current hydra config if already parsed (but not passed in here)
    if HydraConfig.initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    if config_path is not None and config_name is not None:
        if args is None:
            with initialize(config_path=config_path, version_base=None):
                cfg = compose(config_name=config_name)
        else:
            rofunc_path = get_rofunc_path()
            absl_config_path = os.path.join(rofunc_path, "config/{}".format(config_path))
            search_path = create_automatic_config_search_path(config_name, None, absl_config_path)
            hydra_object = Hydra.create_main_hydra2(task_name='load_isaacgymenv', config_search_path=search_path)
            cfg = hydra_object.compose_config(config_name, args.overrides, run_mode=RunMode.RUN)
    else:
        with initialize(config_path="./", version_base=None):
            cfg = compose(config_name="lqt")
    if debug:
        print(OmegaConf.to_yaml(cfg))
    return cfg
