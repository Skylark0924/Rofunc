import os
import sys

import isaacgymenvs
from rofunc.config.utils import omegaconf_to_dict
from rofunc.utils.file.path import get_rofunc_path
from rofunc.config.custom_resolvers import *
from rofunc.lfd.rl.tasks import task_map
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
from hydra.types import RunMode
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import torch
import torch.nn as nn

# set the seed for reproducibility
set_seed(42)


# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, states, taken_actions, role):
        if role == "policy":
            return GaussianMixin.act(self, states, taken_actions, role)
        elif role == "value":
            return DeterministicMixin.act(self, states, taken_actions, role)

    def compute(self, states, taken_actions, role):
        if role == "policy":
            return self.mean_layer(self.net(states)), self.log_std_parameter
        elif role == "value":
            return self.value_layer(self.net(states))


#  get rofunc path from rofunc package metadata
config_file = "config"
rofunc_path = get_rofunc_path()
config_path = os.path.join(rofunc_path, "config/rl")

# get hydra config without use @hydra.main
sys.argv.append("task={}".format("FrankaCabinet"))
args = get_args_parser().parse_args()
search_path = create_automatic_config_search_path(config_file, None, config_path)
hydra_object = Hydra.create_main_hydra2(task_name='load_isaacgymenv', config_search_path=search_path)
config = hydra_object.compose_config(config_file, args.overrides, run_mode=RunMode.RUN)

cfg = omegaconf_to_dict(config.task)

env = task_map["FrankaCabinet"](cfg=cfg,
                                rl_device=config.rl_device,
                                sim_device=config.sim_device,
                                graphics_device_id=config.graphics_device_id,
                                headless=config.headless,
                                virtual_screen_capture=config.capture_video,  # TODO: check
                                force_render=config.force_render)
env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
models_ppo["value"] = models_ppo["policy"]  # same instance: shared model

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 16  # memory_size
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 8  # 16 * 4096 / 8192
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 5e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 120
cfg_ppo["experiment"]["checkpoint_interval"] = 1200

agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 24000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
