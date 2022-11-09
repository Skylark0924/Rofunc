import sys

from rofunc.config.utils import omegaconf_to_dict
from rofunc.examples.learning.tasks import task_map
from rofunc.examples.learning.base_skrl import set_cfg_ppo, set_models_ppo
from rofunc.config.get_config import get_config

from hydra._internal.utils import get_args_parser
from skrl.agents.torch.ppo import PPO
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
# Import the skrl components to build the RL system
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# set the seed for reproducibility
set_seed(42)

# get config
sys.argv.append("task={}".format("CURICabinet"))
sys.argv.append("sim_device={}".format("cuda:1"))
sys.argv.append("rl_device={}".format("cuda:1"))
sys.argv.append("graphics_device_id={}".format(1))
args = get_args_parser().parse_args()
cfg = get_config('./learning/rl', 'config', args=args)
cfg_dict = omegaconf_to_dict(cfg.task)

env = task_map["CURICabinet"](cfg=cfg_dict,
                              rl_device=cfg.rl_device,
                              sim_device=cfg.sim_device,
                              graphics_device_id=cfg.graphics_device_id,
                              headless=cfg.headless,
                              virtual_screen_capture=cfg.capture_video,  # TODO: check
                              force_render=cfg.force_render)
env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

models_ppo = set_models_ppo(cfg, env, device)
cfg_ppo = set_cfg_ppo(cfg, env, device)

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
