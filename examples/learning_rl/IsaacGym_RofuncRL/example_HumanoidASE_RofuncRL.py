"""
HumanoidASE (RofuncRL)
===========================

Humanoid soldier Getup/Perturb/Heading/Location/Reach/Strike, trained by RofuncRL
"""

import isaacgym
import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.pre_trained_models.download import model_zoo
from rofunc.learning.utils.utils import set_seed


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file
    cfg.task.env.asset.assetFileName = custom_args.asset
    cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    env = Tasks().task_map[custom_args.task](cfg=cfg_dict,
                                             rl_device=cfg.rl_device,
                                             sim_device=f'cuda:{cfg.device_id}',
                                             graphics_device_id=cfg.device_id,
                                             headless=cfg.headless,
                                             virtual_screen_capture=cfg.capture_video,  # TODO: check
                                             force_render=cfg.force_render)

    # Instantiate the RL trainer
    hrl = False if custom_args.task in ['HumanoidASEGetupSwordShield', 'HumanoidASEPerturbSwordShield',
                                        'HumanoidViewMotion'] else True
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task,
                                                        hrl=hrl)

    # Start training
    trainer.train()


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device),
                      "headless={}".format(False),
                      "num_envs={}".format(16)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file
    cfg.task.env.asset.assetFileName = custom_args.asset
    cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[custom_args.task](cfg=cfg_dict,
                                                   rl_device=cfg.rl_device,
                                                   sim_device=f'cuda:{cfg.device_id}',
                                                   graphics_device_id=cfg.device_id,
                                                   headless=cfg.headless,
                                                   virtual_screen_capture=cfg.capture_video,  # TODO: check
                                                   force_render=cfg.force_render)

    # Instantiate the RL trainer
    hrl = False if custom_args.task in ['HumanoidASEGetupSwordShield', 'HumanoidASEPerturbSwordShield',
                                        'HumanoidViewMotion'] else True
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=infer_env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task,
                                                        hrl=hrl,
                                                        inference=True)
    # load checkpoint
    if custom_args.task not in ['HumanoidViewMotion']:
        if custom_args.ckpt_path is None:
            custom_args.ckpt_path = model_zoo(name=f"{custom_args.task}.pth")
        trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start inference
    trainer.inference()


if __name__ == '__main__':
    gpu_id = 1

    parser = argparse.ArgumentParser()
    # Available tasks and motion files:
    # HumanoidASEGetupSwordShield -> reallusion_sword_shield/dataset_reallusion_sword_shield.yaml
    # HumanoidASEPerturbSwordShield -> reallusion_sword_shield/dataset_reallusion_sword_shield.yaml
    # HumanoidASEHeadingSwordShield -> reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy
    # HumanoidASEReachSwordShield -> reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy
    # HumanoidASELocationSwordShield -> reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy
    # HumanoidASEStrikeSwordShield -> reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy
    parser.add_argument("--task", type=str, default="HumanoidASEReachSwordShield")
    parser.add_argument("--motion_file", type=str,
                        default="reallusion_sword_shield/dataset_reallusion_sword_shield.yaml")
    parser.add_argument("--asset", type=str, default="mjcf/amp_humanoid_sword_shield.xml")
    parser.add_argument("--agent", type=str, default="ase")  # Available agent: ase
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--sim_device", type=int, default=0)
    parser.add_argument("--rl_device", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)
