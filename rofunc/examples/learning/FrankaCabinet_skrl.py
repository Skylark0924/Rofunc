import os
import sys

from rofunc.config.utils import omegaconf_to_dict
from rofunc.examples.learning.base_skrl import set_cfg_ppo, set_models_ppo
from rofunc.lfd.rl.tasks import task_map
from rofunc.utils.file.path import get_rofunc_path

from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
from hydra.types import RunMode
from skrl.agents.torch.ppo import PPO
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
# Import the skrl components to build the RL system
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# set the seed for reproducibility
set_seed(42)

#  get rofunc path from rofunc package metadata
config_file = "config"
rofunc_path = get_rofunc_path()
config_path = os.path.join(rofunc_path, "config/learning/rl")

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

models_ppo = set_models_ppo(env, device)
cfg_ppo = set_cfg_ppo(env, device)

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
