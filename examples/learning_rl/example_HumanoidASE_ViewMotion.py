import click

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import task_map
from rofunc.learning.RofuncRL.trainers import trainer_map
from rofunc.learning.utils.utils import set_seed


def inference(motion_file):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = [
        "task=HumanoidViewMotion",
        "train=HumanoidViewMotionASERofuncRL",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
        "graphics_device_id=0",
        "headless={}".format(False),
        "num_envs={}".format(16),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = motion_file
    cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    infer_env = task_map["HumanoidViewMotion"](
        cfg=cfg_dict,
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=cfg.capture_video,  # TODO: check
        force_render=cfg.force_render,
    )

    # Instantiate the RL trainer
    trainer = trainer_map["ase"](
        cfg=cfg.train,
        env=infer_env,
        device=cfg.rl_device,
        env_name="HumanoidViewMotion",
        hrl=False,
        inference=True,
    )

    # Start inference
    trainer.inference()


@click.command()
@click.argument("motion_file")
def main(motion_file):
    inference(motion_file)


if __name__ == "__main__":
    main()
