"""
HumanoidPhysHOI Motion View (RofuncRL)
===========================

Preview the motion of the digital humanoid
"""

import isaacgym
import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = [
        f"task=HumanoidPhysHOIViewMotion",
        "train=BaseTaskPHYSHOIRofuncRL",
        f"device_id=0",
        f"rl_device=cuda:{gpu_id}",
        "headless={}".format(False),
        "num_envs={}".format(1),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file
    cfg.task.env.playdataset = True

    cfg_dict = omegaconf_to_dict(cfg.task)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[custom_args.task](cfg=cfg_dict,
                                                   rl_device=cfg.rl_device,
                                                   sim_device=f'cuda:{cfg.device_id}',
                                                   graphics_device_id=cfg.device_id,
                                                   headless=cfg.headless,
                                                   virtual_screen_capture=cfg.capture_video,  # TODO: check
                                                   force_render=cfg.force_render)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map["physhoi"](cfg=cfg,
                                                env=infer_env,
                                                device=cfg.rl_device,
                                                env_name=custom_args.task,
                                                inference=True)

    # Start inference
    trainer.inference()


if __name__ == "__main__":
    gpu_id = 0

    parser = argparse.ArgumentParser()
    # Available tasks: HumanoidPhyshoi
    parser.add_argument("--task", type=str, default="HumanoidPhysHOI")
    # Available motion files: backdribble, backspin, changeleg, fingerspin, pass, rebound, toss, walkpick
    parser.add_argument("--motion_file", type=str, default="examples/data/ballplay/backdribble.pt")
    custom_args = parser.parse_args()

    inference(custom_args)
