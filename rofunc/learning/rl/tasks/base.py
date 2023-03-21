""" Define all environments and provide helper functions to load environments. """

# OpenAI gym interface
import gym

from ..util.subproc_vec_env import SubprocVecEnv

REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def get_env(name):
    """
    Gets the environment class given @name.
    """
    if name not in REGISTERED_ENVS:
        raise Exception(
            "Unknown environment name: {}\nAvailable environments: {}".format(
                name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[name]


def make_env(name, config=None):
    """
    Creates a new environment instance with @name and @config.
    """
    env = get_env(name)

    # get default config if not provided
    if config is None:
        import argparse
        import config.furniture as furniture_config
        from util import str2bool

        parser = argparse.ArgumentParser()
        furniture_config.add_argument(parser)
        parser.add_argument("--seed", type=int, default=123)
        parser.add_argument("--debug", type=str2bool, default=False)

        config, unparsed = parser.parse_known_args()

    return env(config)


def get_gym_env(env_id, env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    return env


def make_vec_env(env_id, num_env, config=None, env_kwargs=None):
    """
    Creates a wrapped SubprocVecEnv using OpenAI gym interface.
    Unity app will use the port number from @config.port to (@config.port + @num_env - 1).

    Code modified based on
    https://github.com/openai/baselines/blob/master/baselines/common/cmd_util.py

    Args:
        env_id: environment id registered in in `env/__init__.py`.
        num_env: number of environments to launch.
        config: general configuration for the environment.
    """
    env_kwargs = env_kwargs or {}

    if config is not None:
        for key, value in config.__dict__.items():
            env_kwargs[key] = value

    def make_thunk(rank):
        new_env_kwargs = env_kwargs.copy()
        new_env_kwargs["port"] = env_kwargs["port"] + rank
        new_env_kwargs["seed"] = env_kwargs["seed"] + rank
        return lambda: get_gym_env(env_id, new_env_kwargs)

    return SubprocVecEnv([make_thunk(i) for i in range(num_env)])


class EnvMeta(type):
    """ Meta class for registering environments. """

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["FurnitureEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls
