"""
HumanoidOmni (RofuncRL)
===========================

Humanoid RL using RofuncRL
"""

import argparse

from rofunc.config.utils import get_config, process_omni_config
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.pre_trained_models.download import model_zoo
from rofunc.learning.utils.utils import set_seed


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    custom_args.num_envs = 64 if custom_args.agent.upper() in ["SAC", "TD3"] else custom_args.num_envs

    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.device_id),
                      "rl_device={}".format(custom_args.rl_device),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    set_seed(cfg.train.Trainer.seed)

    # Startup IsaacSim simulator
    from omni.isaac.gym.vec_env import VecEnvBase
    omni_env = VecEnvBase(headless=cfg.headless, sim_device=cfg.device_id)

    # Instantiate the OmniIsaacGym environment
    omni_cfg = process_omni_config(cfg)
    env = Tasks("omniisaacgym").task_map[custom_args.task](name=cfg.task_name, sim_config=omni_cfg, env=omni_env)
    omni_env.set_task(task=env, sim_params=omni_cfg.get_physics_params(), backend="torch", init_sim=True)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=omni_env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task)

    # Start training
    trainer.train()


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.device_id),
                      "rl_device={}".format(custom_args.rl_device),
                      "headless={}".format(False),
                      "num_envs={}".format(16)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    set_seed(cfg.train.Trainer.seed)

    # Startup IsaacSim simulator
    from omni.isaac.gym.vec_env import VecEnvBase
    omni_env = VecEnvBase(headless=cfg.headless, sim_device=cfg.device_id)

    # Instantiate the OmniIsaacGym environment
    omni_cfg = process_omni_config(cfg)
    env = Tasks("omniisaacgym").task_map[custom_args.task](name=cfg.task_name, sim_config=omni_cfg, env=omni_env)
    omni_env.set_task(task=env, sim_params=omni_cfg.get_physics_params(), backend="torch", init_sim=True)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=omni_env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task,
                                                        inference=True)
    # load checkpoint
    if custom_args.ckpt_path is None:
        raise ValueError("Please specify the checkpoint path for inference.")
    trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start inference
    trainer.inference()


if __name__ == '__main__':
    gpu_id = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="HumanoidOmni")
    parser.add_argument("--agent", type=str, default="ppo")  # Available agents: ppo, sac, td3, a2c
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--device_id", type=int, default=gpu_id)
    parser.add_argument("--rl_device", type=str, default=f"cuda:{gpu_id}")
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)
