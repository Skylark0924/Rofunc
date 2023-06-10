import datetime
import os
import sys
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from hydra._internal.utils import get_args_parser
from skrl.agents.torch.ppo import PPO as PPOAgent
from skrl.agents.torch.sac import SAC as SACAgent
from skrl.agents.torch.td3 import TD3 as TD3Agent
from skrl.agents.torch.ddpg import DDPG as DDPGAgent
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed
from tensorboard import program

import rofunc as rf
from rofunc.config.utils import get_config, omegaconf_to_dict
from rofunc.utils.file.internet import reserve_sock_addr
from rofunc.utils.logger.beauty_logger import BeautyLogger


# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}


# Define the pre_trained_models (stochastic and deterministic pre_trained_models) for the agents using mixins.
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

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"])), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


def set_models_ddpg(env, device):
    """
    DDPG requires 4 pre_trained_models, visit its documentation for more details
    https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
    """
    models_ddpg = {}
    models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
    models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device,
                                                      clip_actions=True)
    models_ddpg["critic"] = Critic(env.observation_space, env.action_space, device)
    models_ddpg["target_critic"] = Critic(env.observation_space, env.action_space, device)
    # Initialize the pre_trained_models' parameters (weights and biases) using a Gaussian distribution
    for model in models_ddpg.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return models_ddpg


def set_cfg_ppo(cfg, env, device, eval_mode=False):
    """
    # Configure and instantiate the agent.
    # Only modify some default configuration, visit its documentation to see all the options
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
    """
    cfg_ppo = omegaconf_to_dict(cfg.train)
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
    cfg_ppo["experiment"]["directory"] = os.path.join(os.getcwd(), "runs")
    if eval_mode:
        cfg_ppo["experiment"]["experiment_name"] = "Eval_{}{}_{}".format(cfg.task.name, "SKRLPPO",
                                                                         datetime.datetime.now().strftime(
                                                                             "%y-%m-%d_%H-%M-%S-%f"))
    else:
        cfg_ppo["experiment"]["experiment_name"] = "{}{}_{}".format(cfg.task.name, "SKRLPPO",
                                                                    datetime.datetime.now().strftime(
                                                                        "%y-%m-%d_%H-%M-%S-%f"))
    return cfg_ppo


def set_cfg_td3(cfg, env, device, eval_mode=False):
    """
    https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#configuration-and-hyperparameters
    """
    cfg_td3 = TD3_DEFAULT_CONFIG.copy()
    cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.2, device=device)
    cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
    cfg_td3["smooth_regularization_clip"] = 0.1
    cfg_td3["gradient_steps"] = 1
    cfg_td3["batch_size"] = 512
    cfg_td3["random_timesteps"] = 0
    cfg_td3["learning_starts"] = 0
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_td3["experiment"]["write_interval"] = 100
    cfg_td3["experiment"]["checkpoint_interval"] = 1000
    if eval_mode:
        cfg_td3["experiment"]["experiment_name"] = "Eval_{}{}_{}".format(cfg.task.name, "TD3",
                                                                         datetime.datetime.now().strftime(
                                                                             "%y-%m-%d_%H-%M-%S-%f"))
    else:
        cfg_td3["experiment"]["experiment_name"] = "{}{}_{}".format(cfg.task.name, "TD3",
                                                                    datetime.datetime.now().strftime(
                                                                        "%y-%m-%d_%H-%M-%S-%f"))
    return cfg_td3


def set_cfg_ddpg(cfg, env, device, eval_mode=False):
    """
    https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
    """
    cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
    cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
    cfg_ddpg["gradient_steps"] = 1
    cfg_ddpg["batch_size"] = 512
    cfg_ddpg["random_timesteps"] = 0
    cfg_ddpg["learning_starts"] = 0
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_ddpg["experiment"]["write_interval"] = 100
    cfg_ddpg["experiment"]["checkpoint_interval"] = 1000
    if eval_mode:
        cfg_ddpg["experiment"]["experiment_name"] = "Eval_{}{}_{}".format(cfg.task.name, "DDPG",
                                                                          datetime.datetime.now().strftime(
                                                                              "%y-%m-%d_%H-%M-%S-%f"))
    else:
        cfg_ddpg["experiment"]["experiment_name"] = "{}{}_{}".format(cfg.task.name, "DDPG",
                                                                     datetime.datetime.now().strftime(
                                                                         "%y-%m-%d_%H-%M-%S-%f"))
    return cfg_ddpg


def set_cfg_sac(cfg, env, device, eval_mode=False):
    """
    https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
    """
    cfg_sac = SAC_DEFAULT_CONFIG.copy()
    cfg_sac["gradient_steps"] = 1
    cfg_sac["batch_size"] = 256
    cfg_sac["random_timesteps"] = 1000
    cfg_sac["learning_starts"] = 1000
    cfg_sac["actor_learning_rate"] = 1e-3
    cfg_sac["critic_learning_rate"] = 1e-3
    cfg_sac["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_sac["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_sac["state_preprocessor"] = RunningStandardScaler
    cfg_sac["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    #
    cfg_sac["learn_entropy"] = True
    # cfg_sac["entropy_learning_rate"] = 5e-3
    cfg_sac["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_sac["experiment"]["write_interval"] = 100
    cfg_sac["experiment"]["checkpoint_interval"] = 1000
    if eval_mode:
        cfg_sac["experiment"]["experiment_name"] = "Eval_{}{}_{}".format(cfg.task.name, "SAC",
                                                                         datetime.datetime.now().strftime(
                                                                             "%y-%m-%d_%H-%M-%S-%f"))
    else:
        cfg_sac["experiment"]["experiment_name"] = "{}{}_{}".format(cfg.task.name, "SAC",
                                                                    datetime.datetime.now().strftime(
                                                                        "%y-%m-%d_%H-%M-%S-%f"))
    return cfg_sac


def setup(custom_args, eval_mode=False):
    # set the seed for reproducibility
    set_seed(42)

    # get config
    sys.argv.append("task={}".format(custom_args.task))
    sys.argv.append("train={}{}SKRL".format(custom_args.task, custom_args.agent.upper()))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("headless={}".format(custom_args.headless))
    if custom_args.agent.lower() == "sac":
        sys.argv.append("num_envs={}".format(64))

    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    cfg_dict = omegaconf_to_dict(cfg.task)

    if eval_mode:
        cfg_dict['env']['numEnvs'] = 16

    from rofunc.learning.rl.tasks import task_map
    env = task_map[custom_args.task](cfg=cfg_dict,
                                     rl_device=cfg.rl_device,
                                     sim_device=cfg.sim_device,
                                     graphics_device_id=cfg.graphics_device_id,
                                     headless=cfg.headless,
                                     virtual_screen_capture=cfg.capture_video,  # TODO: check
                                     force_render=cfg.force_render)
    env = wrap_env(env)

    device = env.device

    if custom_args.agent.lower() == "ppo":
        memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
        models_ppo = {"policy": Shared(env.observation_space, env.action_space, device)}
        models_ppo["value"] = models_ppo["policy"]  # same instance: shared model
        cfg_ppo = set_cfg_ppo(cfg, env, device, eval_mode)
        agent = PPOAgent(models=models_ppo,
                         memory=memory,
                         cfg=cfg_ppo,
                         observation_space=env.observation_space,
                         action_space=env.action_space,
                         device=device)
    elif custom_args.agent.lower() == "sac":
        memory = RandomMemory(memory_size=1000000, num_envs=env.num_envs, device=device, replacement=True)
        models_sac = {}
        models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
        models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
        models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
        models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
        models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
        for model in models_sac.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
        cfg_sac = set_cfg_sac(cfg.train, env, device, eval_mode)
        agent = SACAgent(models=models_sac,
                         memory=memory,
                         cfg=cfg_sac,
                         observation_space=env.observation_space,
                         action_space=env.action_space,
                         device=device)
    elif custom_args.agent.lower() == "td3":
        memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device, replacement=True)
        models_td3 = {}
        models_td3["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
        models_td3["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device,
                                                         clip_actions=True)
        models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
        models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
        models_td3["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
        models_td3["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
        for model in models_td3.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
        cfg_td3 = set_cfg_td3(cfg.train, env, device, eval_mode)
        agent = TD3Agent(models=models_td3,
                         memory=memory,
                         cfg=cfg_td3,
                         observation_space=env.observation_space,
                         action_space=env.action_space,
                         device=device)
    elif custom_args.agent.lower() == "ddpg":
        memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device, replacement=True)
        models_ddpg = {}
        models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
        models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device,
                                                          clip_actions=True)
        models_ddpg["critic"] = Critic(env.observation_space, env.action_space, device)
        models_ddpg["target_critic"] = Critic(env.observation_space, env.action_space, device)
        for model in models_ddpg.values():
            model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
        cfg_ddpg = set_cfg_ddpg(cfg.train, env, device, eval_mode)
        agent = DDPGAgent(models=models_ddpg,
                          memory=memory,
                          cfg=cfg_ddpg,
                          observation_space=env.observation_space,
                          action_space=env.action_space,
                          device=device)
    else:
        raise ValueError("Agent not supported")

    tb = program.TensorBoard()
    # Find a free port
    with reserve_sock_addr() as (h, p):
        argv = ['tensorboard', f"--logdir={agent.experiment_dir}",
                f"--port={p}"]
        tb_extra_args = os.getenv('TB_EXTRA_ARGS', "")
        if tb_extra_args:
            argv += tb_extra_args.split(' ')
        tb.configure(argv)
    url = tb.launch()
    rf.logger.beauty_print(f"Tensorflow listening on {url}", type='info')
    rf.utils.create_dir(agent.experiment_dir)

    rofunc_logger = BeautyLogger(agent.experiment_dir, 'rofunc.log')
    rofunc_logger.info("Configurations: \n{}".format(agent.cfg))

    return env, agent
