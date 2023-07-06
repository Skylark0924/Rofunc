"""
CURICabinet (ElegantRL)
===========================

Open drawers with a humanoid CURI robot, trained by ElegantRL
"""

import argparse
import isaacgym

from elegantrl.train.run import train_and_evaluate

from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.learning.pre_trained_models import model_zoo
from rofunc.learning.RofuncRL.utils.elegantrl_utils import setup


def train(custom_args):
    beauty_print("Start training")

    env, args = setup(custom_args)

    # start training
    train_and_evaluate(args)


def eval(custom_args, ckpt_path=None):
    # TODO: add support for eval mode
    beauty_print("Start evaluating")

    env, agent = setup(custom_args, eval_mode=True)

    # load checkpoint
    if ckpt_path is None:
        ckpt_path = model_zoo(name="CURICabinetPPO_right_arm.pt")
    agent.save_or_load_agent(cwd=ckpt_path, if_save=False)

    # evaluate the agent
    state = env.reset()
    episode_reward = 0
    for i in range(2 ** 10):
        action = agent.act.get_action(state).detach()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward.mean()
        # if done:
        #     print(f'Step {i:>6}, Episode return {episode_reward:8.3f}')
        #     break
        # else:
        state = next_state


if __name__ == '__main__':
    gpu_id = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CURICabinet")
    parser.add_argument("--agent", type=str, default="sac")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        train(custom_args)
    else:
        ckpt_path = "/examples/learning/result/CURICabinet_SAC_42/actor_53608448_00007.742.pth"
        eval(custom_args, ckpt_path=ckpt_path)
