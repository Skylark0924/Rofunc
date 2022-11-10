import argparse
import sys

from rofunc.config.get_config import get_config
from rofunc.config.utils import omegaconf_to_dict
from rofunc.examples.learning.base_skrl import set_cfg_ppo, set_models_ppo
from rofunc.examples.learning.tasks import task_map
from rofunc.data.models import model_zoo
from rofunc.utils.logger.beauty_logger import beauty_print

from hydra._internal.utils import get_args_parser
from skrl.agents.torch.ppo import PPO
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
# Import the skrl components to build the RL system
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


def setup(custom_args, eval_mode=False):
    # set the seed for reproducibility
    set_seed(42)

    # get config
    sys.argv.append("task={}".format("FrankaCabinet"))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    cfg_dict = omegaconf_to_dict(cfg.task)

    if eval_mode:
        cfg_dict['env']['numEnvs'] = 16

    env = task_map["FrankaCabinet"](cfg=cfg_dict,
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

    return env, agent


def train(custom_args):
    beauty_print("Start training")

    env, agent = setup(custom_args)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 24000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


def eval(custom_args, ckpt_path=None):
    beauty_print("Start evaluating")

    env, agent = setup(custom_args, eval_mode=True)

    # load checkpoint (agent)
    if ckpt_path is None:
        ckpt_path = model_zoo(name="FrankaCabinet.pt")
    agent.load(ckpt_path)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1600, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # evaluate the agent
    trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--train", action="store_true", help="turn to train mode while adding this argument")
    custom_args = parser.parse_args()

    if custom_args.train:
        train(custom_args)
    else:
        eval(custom_args)
