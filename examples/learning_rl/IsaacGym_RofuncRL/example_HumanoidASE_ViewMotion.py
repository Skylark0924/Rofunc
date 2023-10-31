"""
Humanoid Motion View (RofuncRL)
===========================

Preview the motion of the digital humanoid
"""

import click

from rofunc.config.utils import omegaconf_to_dict, get_config, load_view_motion_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers


def inference(config_name, motion_file):
    view_motion_config = load_view_motion_config(config_name)
    task_name = "HumanoidViewMotion"
    args_overrides = [
        f"task={task_name}",
        "train=HumanoidViewMotionASERofuncRL",
        "device_id=0",
        "rl_device=cuda:0",
        "headless={}".format(False),
        "num_envs={}".format(16),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = motion_file

    # Overwrite
    cfg.task.env.asset.assetFileName = view_motion_config["asset_name"]
    cfg.task.env.asset.assetBodyNum = view_motion_config["asset_body_num"]
    cfg.task.env.asset.assetJointNum = view_motion_config["asset_joint_num"]

    cfg_dict = omegaconf_to_dict(cfg.task)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[task_name](cfg=cfg_dict,
                                            rl_device=cfg.rl_device,
                                            sim_device=f'cuda:{cfg.device_id}',
                                            headless=cfg.headless,
                                            virtual_screen_capture=cfg.capture_video,  # TODO: check
                                            force_render=cfg.force_render)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map["ase"](cfg=cfg.train,
                                            env=infer_env,
                                            device=cfg.rl_device,
                                            env_name=task_name,
                                            hrl=False,
                                            inference=True)

    # Start inference
    trainer.inference()


@click.command()
@click.argument("motion_file")
@click.option("--config_name", default="HumanoidSpoonPanSimple")
def main(config_name, motion_file):
    inference(config_name, motion_file)


if __name__ == "__main__":
    main()
