"""
Dexterous Hands (RofuncRL)
===========================

Examples of learning hand (Shadow Hand, Allegro Hand, qbSofthand) tasks by RofuncRL
"""

import isaacgym
import argparse
import datetime

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.pre_trained_models.download import model_zoo
from rofunc.learning.utils.utils import set_seed


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    custom_args.num_envs = 64 if custom_args.agent.upper() in ["SAC", "TD3"] else custom_args.num_envs

    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    if custom_args.task == "CURIQbSoftHandSynergyGrasp":
        cfg.task.env.objectType = custom_args.object.lower()
        cfg.task.env.useSynergy = True if custom_args.use_synergy.lower() == "true" else False
        cfg.train.Trainer.experiment_name = "{}_{}_{}".format(custom_args.object,
                                                              "Synergy" if custom_args.use_synergy.lower() == "true" else "NoSynergy",
                                                              datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
    cfg.train.Trainer.maximum_steps = 100000
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
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task)
    if custom_args.ckpt_path is not None:
        trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start training
    trainer.train()


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device),
                      "headless={}".format(True),
                      "num_envs={}".format(1)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    if custom_args.task == "CURIQbSoftHandSynergyGrasp":
        cfg.task.env.objectType = custom_args.object.lower()
        cfg.task.env.useSynergy = True if custom_args.use_synergy.lower() == "true" else False
        cfg.train.Trainer.experiment_name = custom_args.object
    cfg.train.Trainer.inference_steps = 10000

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
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=infer_env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task,
                                                        inference=True)
    # load checkpoint
    if custom_args.ckpt_path is None:
        custom_args.ckpt_path = model_zoo(name=f"{custom_args.task}{custom_args.object}RofuncRLPPO.pth")
    trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start inference
    trainer.inference()


if __name__ == '__main__':
    gpu_id = 0

    parser = argparse.ArgumentParser()
    # Available tasks: BiShadowHandOver, BiShadowHandBlockStack, BiShadowHandBottleCap, BiShadowHandCatchAbreast,
    #                  BiShadowHandCatchOver2Underarm, BiShadowHandCatchUnderarm, BiShadowHandDoorOpenInward,
    #                  BiShadowHandDoorOpenOutward, BiShadowHandDoorCloseInward, BiShadowHandDoorCloseOutward,
    #                  BiShadowHandGraspAndPlace, BiShadowHandLiftUnderarm, BiShadowHandPen, BiShadowHandPointCloud,
    #                  BiShadowHandPushBlock, BiShadowHandReOrientation, BiShadowHandScissors, BiShadowHandSwingCup,
    #                  BiShadowHandSwitch, BiShadowHandTwoCatchUnderarm
    #                  QbSoftHandGrasp, BiQbSoftHandGraspAndPlace, BiQbSoftHandSynergyGrasp, QbSoftHandSynergyGrasp
    #                  ShadowHandGrasp, CURIQbSoftHandSynergyGrasp
    parser.add_argument("--task", type=str, default="CURIQbSoftHandSynergyGrasp")
    # Available objects: Hammer, Spatula, Large_Clamp, Mug, Power_Drill, Knife, Scissors, Large_Marker, Phillips_Screw_Driver
    # Only for CURIQbSoftHandSynergyGrasp
    parser.add_argument("--object", type=str, default="Mug")
    # Only for CURIQbSoftHandSynergyGrasp
    parser.add_argument("--use_synergy", type=str, default="True")
    parser.add_argument("--agent", type=str, default="ppo")  # Available agents: ppo, sac, td3, a2c
    parser.add_argument("--num_envs", type=int, default=1024)
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
