"""
AntOmni (RofuncRL)
===========================

Ant RL using RofuncRL
"""

import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.pre_trained_models.download import model_zoo
from rofunc.learning.utils.utils import set_seed
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    custom_args.num_envs = 64 if custom_args.agent.upper() in ["SAC", "TD3"] else custom_args.num_envs

    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.device_id),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    # cfg_dict = omegaconf_to_dict(cfg.task)
    cfg_dict = omegaconf_to_dict(cfg)

    set_seed(cfg.train.Trainer.seed)

    # Startup IsaacSim simulator
    env_omni = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.device_id)

    # Instantiate the Isaac Gym environment
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(cfg_dict)
    env = Tasks("omniisaacgym").task_map[custom_args.task](name=cfg.task_name,
                                                           sim_config=sim_config,
                                                           env=env_omni)
    init_sim = True
    env_omni.set_task(task=env, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg.train,
                                                        env=env_omni,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task)

    # Start training
    trainer.train()


def inference(custom_args):
    pass


if __name__ == '__main__':
    gpu_id = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="AntOmni")
    parser.add_argument("--agent", type=str, default="ppo")  # Available agents: ppo, sac, td3, a2c
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--device_id", type=str, default=gpu_id)
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)
