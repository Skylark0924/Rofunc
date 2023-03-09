"""
Humanoid
===========================

Temp
"""

import argparse
import sys

import isaacgym
from rofunc.config.utils import get_config
from rofunc.config.utils import omegaconf_to_dict
from rofunc.lfd.rl.utils.skrl_utils import set_cfg_ppo, set_models_ppo
from rofunc.lfd.rl.tasks import task_map
from rofunc.data.models import model_zoo
from rofunc.utils.logger.beauty_logger import beauty_print

from hydra._internal.utils import get_args_parser
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.sac import SAC
from skrl.agents.torch.td3 import TD3
from skrl.agents.torch.amp import AMP
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
# Import the skrl components to build the RL system
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


def setup(custom_args, task_name, eval_mode=False):
    # set the seed for reproducibility
    set_seed(42)

    # get config
    sys.argv.append("task={}".format(task_name))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    cfg_dict = omegaconf_to_dict(cfg.task)

    if eval_mode:
        cfg_dict['env']['numEnvs'] = 16

    env = task_map[task_name](cfg=cfg_dict,
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

    if custom_args.agent == "ppo":
        models_ppo = set_models_ppo(cfg, env, device)
        cfg_ppo = set_cfg_ppo(cfg, env, device, eval_mode)
        agent = PPO(models=models_ppo,
                    memory=memory,
                    cfg=cfg_ppo,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    elif custom_args.agent == "sac":
        models_sac = set_models_sac(cfg, env, device)
        cfg_sac = set_cfg_sac(cfg, env, device, eval_mode)
        agent = SAC(models=models_sac,
                    memory=memory,
                    cfg=cfg_sac,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    elif custom_args.agent == "td3":
        models_td3 = set_models_td3(cfg, env, device)
        cfg_td3 = set_cfg_td3(cfg, env, device, eval_mode)
        agent = TD3(models=models_td3,
                    memory=memory,
                    cfg=cfg_td3,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    elif custom_args.agent == "amp":
        models_amp = set_models_amp(cfg, env, device)
        cfg_amp = set_cfg_amp(cfg, env, device, eval_mode)
        agent = AMP(models=models_amp,
                    memory=memory,
                    cfg=cfg_amp,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    else:
        raise ValueError("Agent not supported")

    return env, agent


def train(custom_args, task_name):
    beauty_print("Start training")

    env, agent = setup(custom_args, task_name)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 40000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


def eval(custom_args, task_name, ckpt_path=None):
    beauty_print("Start evaluating")

    env, agent = setup(custom_args, task_name, eval_mode=True)

    # load checkpoint (agent)
    if ckpt_path is None:
        ckpt_path = model_zoo(name="CURICabinetPPO_right_arm.pt")
    agent.load(ckpt_path)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1600, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # evaluate the agent
    trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--sim_device", type=str, default="cuda:1")
    parser.add_argument("--rl_device", type=str, default="cuda:1")
    parser.add_argument("--graphics_device_id", type=int, default=1)
    parser.add_argument("--train", action="store_false", help="turn to train mode while adding this argument")
    custom_args = parser.parse_args()

    task_name = "Humanoid"

    if custom_args.train:
        train(custom_args, task_name)
    else:
        ckpt_path = "/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/examples/learning/runs/CURICabinetBimanualPPO_22-11-16_17-46-07-677096/checkpoints/best_agent.pt"
        eval(custom_args, task_name, ckpt_path=ckpt_path)