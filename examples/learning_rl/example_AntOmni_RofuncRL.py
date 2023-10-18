"""
Ant (RofuncRL)
===========================

Ant RL using RofuncRL
"""

import argparse
from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.trainers import trainer_map
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
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env_omni = VecEnvRLGames(headless=cfg.headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)


    # Instantiate the Isaac Gym environment
    from rofunc.learning.RofuncRL.tasks import task_map
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    env = task_map[custom_args.task](name=cfg.task_name,
                                     sim_config=SimConfig(cfg_dict),
                                     env=env_omni)

    # Instantiate the RL trainer
    trainer = trainer_map[custom_args.agent](cfg=cfg.train,
                                             env=env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task)

    # Start training
    trainer.train()


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "sim_device={}".format(custom_args.sim_device),
                      "rl_device={}".format(custom_args.rl_device),
                      "graphics_device_id={}".format(custom_args.graphics_device_id),
                      "headless={}".format(False),
                      "num_envs={}".format(16)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    infer_env = task_map[custom_args.task](cfg=cfg_dict,
                                           rl_device=cfg.rl_device,
                                           sim_device=cfg.sim_device,
                                           graphics_device_id=cfg.graphics_device_id,
                                           headless=cfg.headless,
                                           virtual_screen_capture=cfg.capture_video,  # TODO: check
                                           force_render=cfg.force_render)

    # Instantiate the RL trainer
    trainer = trainer_map[custom_args.agent](cfg=cfg.train,
                                             env=infer_env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task)
    # load checkpoint
    if custom_args.ckpt_path is None:
        custom_args.ckpt_path = model_zoo(name="AntRofuncRLPPO.pth")
    trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start inference
    trainer.inference()


if __name__ == '__main__':
    # # import the environment loader
    # from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
    #
    # # load environment
    # env = load_omniverse_isaacgym_env(task_name="Cartpole")

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
