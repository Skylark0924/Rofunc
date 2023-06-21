import os
from typing import Dict

import hydra
from hydra import compose, initialize
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

import rofunc as rf
from rofunc.config import *
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

            # Find the available task and train config files
            try:
                cfg = hydra_object.compose_config(config_name, args, run_mode=RunMode.RUN)
                rf.logger.beauty_print('Use task config: {}.yaml'.format(args[0].split('=')[1]), type='info')
                rf.logger.beauty_print('Use train config: {}.yaml'.format(args[1].split('=')[1]), type='info')
            except Exception as e:
                rf.logger.beauty_print(e, type='warning')
                original_task = args[0].split('=')[1]
                if args[0].split('=')[1].split('_')[0] == 'Gym':
                    task = 'GymBaseTask'
                    args[0] = 'task={}'.format(task)
                    rf.logger.beauty_print('Use task config: {}.yaml'.format(task), type='warning')
                try:
                    cfg = hydra_object.compose_config(config_name, args, run_mode=RunMode.RUN)
                    rf.logger.beauty_print('Use train config: {}.yaml'.format(args[1].split('=')[1]),
                                           type='info')
                except Exception as e:
                    rf.logger.beauty_print(e, type='warning')
                    train = 'BaseTask' + args[1].split('=')[1].split(original_task)[1]
                    args[1] = 'train={}'.format(train)
                    cfg = hydra_object.compose_config(config_name, args, run_mode=RunMode.RUN)
                    rf.logger.beauty_print('Use train config: {}.yaml'.format(train), type='warning')
    else:
        with initialize(config_path="./", version_base=None):
            cfg = compose(config_name="lqt")
    if debug:
        print_config(cfg)
    return cfg


def print_config(config: DictConfig):
    print("-----------------------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------------------")


def omegaconf_to_dict(config: DictConfig) -> Dict:
    """
    Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation.
    """
    d = {}
    for k, v in config.items():
        d[k] = omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
    return d


def dict_to_omegaconf(d: Dict, save_path: str = None) -> DictConfig:
    """
    Converts a python Dict to an omegaconf DictConfig, respecting variable interpolation.
    """
    conf = OmegaConf.create(d)
    if save_path is not None:
        with open(save_path, 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name)
            loaded = OmegaConf.load(fp.name)
            assert conf == loaded
    else:
        return conf


if __name__ == '__main__':
    TD3_DEFAULT_CONFIG = {
        "gradient_steps": 1,  # gradient steps
        "batch_size": 64,  # training batch size

        "discount_factor": 0.99,  # discount factor (gamma)
        "polyak": 0.005,  # soft update hyperparameter (tau)

        "actor_learning_rate": 1e-3,  # actor learning rate
        "critic_learning_rate": 1e-3,  # critic learning rate
        "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
        "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

        "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
        "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})

        "random_timesteps": 0,  # random exploration steps
        "learning_starts": 0,  # learning starts after this many steps

        "exploration": {
            "noise": None,  # exploration noise
            "initial_scale": 1.0,  # initial scale for noise
            "final_scale": 1e-3,  # final scale for noise
            "timesteps": None,  # timesteps for noise decay
        },

        "policy_delay": 2,  # policy delay update with respect to critic update
        "smooth_regularization_noise": None,  # smooth noise for regularization
        "smooth_regularization_clip": 0.5,  # clip for smooth regularization

        "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

        "experiment": {
            "directory": "",  # experiment's parent directory
            "experiment_name": "",  # experiment name
            "write_interval": 250,  # TensorBoard writing interval (timesteps)

            "checkpoint_interval": 1000,  # interval for checkpoints (timesteps)
            "store_separately": False,  # whether to store checkpoints separately
        }
    }

    dict_to_omegaconf(TD3_DEFAULT_CONFIG,
                      save_path="/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/config/learning/rl/agent/td3_default_config_skrl.yaml")
