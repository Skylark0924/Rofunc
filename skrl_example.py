import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
# from skrl.envs.torch import load_isaacgym_env_preview4
from rofunc.examples.learning.tasks import task_map


# Define the models (stochastic and deterministic models) for the agents using mixins.
# - StochasticActor: takes as input the environment's observation/state and returns an action
# - DeterministicActor: takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        return self.net(states), self.log_std_parameter


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, states, taken_actions, role):
        return self.net(states)


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions, role):
        return self.net(torch.cat([states, taken_actions], dim=1))


# Load and wrap the Isaac Gym environment
import sys
from hydra._internal.utils import get_args_parser
from rofunc.config.utils import get_config, omegaconf_to_dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, default="sac")
parser.add_argument("--sim_device", type=str, default="cuda:0")
parser.add_argument("--rl_device", type=str, default="cuda:0")
parser.add_argument("--graphics_device_id", type=int, default=0)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--train", action="store_false", help="turn to train mode while adding this argument")
custom_args = parser.parse_args()

sys.argv.append("task={}".format("FrankaCabinet"))
sys.argv.append("sim_device={}".format(custom_args.sim_device))
sys.argv.append("rl_device={}".format(custom_args.rl_device))
sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
sys.argv.append("headless={}".format(custom_args.headless))
args = get_args_parser().parse_args()
cfg = get_config('./learning/rl', 'config', args=args)
cfg_dict = omegaconf_to_dict(cfg.task)
env = task_map["FrankaCabinet"](cfg=cfg_dict,
                                rl_device=cfg.rl_device,
                                sim_device=cfg.sim_device,
                                graphics_device_id=cfg.graphics_device_id,
                                headless=cfg.headless,
                                virtual_screen_capture=cfg.capture_video,  # TODO: check
                                force_render=cfg.force_render)
# env = load_isaacgym_env_preview4(task_name="Cartpole")  # preview 3 and 4 use the same loader
env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory (without replacement) as shared experience replay memory
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device, replacement=True)

# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_ddpg["critic"] = Critic(env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = Critic(env.observation_space, env.action_space, device)
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models
models_td3 = {}
models_td3["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_td3["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_td3["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_td3["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
for model in models_td3.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg_ddpg["gradient_steps"] = 1
cfg_ddpg["batch_size"] = 512
cfg_ddpg["random_timesteps"] = 0
cfg_ddpg["learning_starts"] = 0
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 25
cfg_ddpg["experiment"]["checkpoint_interval"] = 1000
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#configuration-and-hyperparameters
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

cfg_td3 = TD3_DEFAULT_CONFIG.copy()
cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.03, device=device)
cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.03, device=device)
cfg_td3["smooth_regularization_clip"] = 0.1
cfg_td3["gradient_steps"] = 1
cfg_td3["batch_size"] = 256
cfg_td3["random_timesteps"] = 0
cfg_td3["learning_starts"] = 0
cfg_td3["actor_learning_rate"] = 5e-4
cfg_td3["critic_learning_rate"] = 5e-4
cfg_td3["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg_td3["learning_rate_scheduler"] = KLAdaptiveRL
cfg_td3["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_td3["state_preprocessor"] = RunningStandardScaler
cfg_td3["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_td3["experiment"]["write_interval"] = 25
cfg_td3["experiment"]["checkpoint_interval"] = 1000
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 512
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 0
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 25
cfg_sac["experiment"]["checkpoint_interval"] = 1000

agent_ddpg = DDPG(models=models_ddpg,
                  memory=memory,
                  cfg=cfg_ddpg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)

agent_td3 = TD3(models=models_td3,
                memory=memory,
                cfg=cfg_td3,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

agent_sac = SAC(models=models_sac,
                memory=memory,
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# Configure and instantiate the RL trainer
cfg = {"timesteps": 40000, "headless": True}
trainer = SequentialTrainer(cfg=cfg,
                            env=env,
                            agents=agent_sac)
                            # agents=[agent_ddpg, agent_td3, agent_sac],
                            # agents_scope=[])

# start training
trainer.train()
