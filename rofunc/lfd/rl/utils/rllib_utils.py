import gym.spaces
import isaacgym
import numpy as np
import torch
from ray.rllib.env import VectorEnv

from elegantrl.envs.IsaacGym import IsaacVecEnv
from rofunc.lfd.rl.tasks import task_map
from rofunc.lfd.rl.utils.elegantrl_utils import ElegantRLIsaacGymEnvWrapper
from rofunc.config.utils import get_config, omegaconf_to_dict
from hydra._internal.utils import get_args_parser
from tqdm.auto import tqdm


class RLlibEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.env = ElegantRLIsaacGymEnvWrapper(env=env, cfg=env_config)
        self.action_space = self.env.env.action_space
        self.observation_space = self.env.env.observation_space

    def reset(self):
        return np.array(self.env.reset().cpu())[0]

    def step(self, action):
        action = torch.tensor(np.array([action])).to(self.env.device)
        observations, rewards, dones, info_dict = self.env.step(action)
        return np.array(observations.cpu())[0], np.array(rewards.cpu())[0], np.array(dones.cpu())[0], info_dict


class RLlibIsaacGymEnvWrapper(VectorEnv):
    def __init__(self, env_config):
        # self.env = IsaacVecEnv(env_name=env_config["task_name"], env_num=1024, sim_device_id=env_config["gpu_id"],
        #                        rl_device_id=env_config["gpu_id"], should_print=True)

        env = task_map[env_config["task_name"]](cfg=env_config["task_cfg_dict"],
                                                rl_device=env_config["cfg"].rl_device,
                                                sim_device=env_config["cfg"].sim_device,
                                                graphics_device_id=env_config["cfg"].graphics_device_id,
                                                headless=env_config["cfg"].headless,
                                                virtual_screen_capture=env_config["cfg"].capture_video,  # TODO: check
                                                force_render=env_config["cfg"].force_render)
        self.env = ElegantRLIsaacGymEnvWrapper(env=env, cfg=env_config["cfg"])
        # self.sub_env = IsaacOneEnv(env_name=env_config["env_name"])
        self.action_space = self.env.env.action_space
        self.observation_space = self.env.env.observation_space
        self.num_envs = self.env.env_num
        super().__init__(self.observation_space, self.action_space, self.num_envs)

        self._prv_obs = [None for _ in range(self.num_envs)]

    def reset_at(self, index=None):
        return self._prv_obs[index]

    def vector_reset(self):
        self._prv_obs = np.array(self.env.reset().cpu()).reshape((self.num_envs, -1))
        return self._prv_obs

    # @override(VectorEnv)
    def vector_step(self, actions):
        actions = torch.tensor(np.array(actions)).to(self.env.device)
        observations, rewards, dones, info_dict_raw = self.env.step(actions)
        info_dict_raw["time_outs"] = np.array(info_dict_raw["time_outs"].cpu())
        # info_dict = [{"agent_0": {"training_enabled": False}} for i in range(self.num_envs)]
        # info_dict = {i: {} for i in range(self.num_envs)}
        info_dict = [{} for i in range(self.num_envs)]
        obs = np.array(observations.cpu()).reshape((self.num_envs, -1))
        self._prv_obs = obs

        return obs, np.array(rewards.cpu()), np.array(dones.cpu()), info_dict

    # @override(VectorEnv)
    # def get_sub_environments(self):
    #     return self.sub_env


def ray_test(custom_args):
    import ray, sys
    from ray.rllib.agents import ppo

    gpu_id = 0

    ray.init()

    sys.argv.append("task={}".format(custom_args.task))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("headless={}".format(custom_args.headless))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    task_cfg_dict = omegaconf_to_dict(cfg.task)

    trainer = ppo.PPOTrainer(env=RLlibIsaacGymEnvWrapper, config={
        "env_config": {"gpu_id": gpu_id,
                       "task_name": custom_args.task,
                       "task_cfg_dict": task_cfg_dict,
                       "cfg": cfg},  # config to pass to env class
        # "framework": "torch",
        "num_workers": 0,
        # 'explore': True,
        # 'exploration_config': {
        #     'type': 'StochasticSampling'
        #     # 'type': 'Curiosity',
        #     # 'eta': 1.0,
        #     # 'lr': 0.001,
        #     # 'feature_dim': 288,
        #     # "feature_net_config": {
        #     #    "fcnet_hiddens": [],
        #     #    "fcnet_activation": "relu",
        #     # },
        #     # "inverse_net_hiddens": [256],
        #     # "inverse_net_activation": "relu",
        #     # "forward_net_hiddens": [256],
        #     # "forward_net_activation": "relu",
        #     # "beta": 0.2,
        #     # "sub_exploration": {
        #     #    "type": "StochasticSampling",
        #     # }
        # },
        # "num_envs_per_worker": 1,
        'gamma': 0.998,

        'train_batch_size': 2048,
        'sgd_minibatch_size': 2048,
        'rollout_fragment_length': 64,
        'num_sgd_iter': 3,
        'lr': 5e-5,

        'vf_loss_coeff': 0.5,
        'vf_share_layers': True,
        'kl_coeff': 0.0,
        'kl_target': 0.1,
        'clip_param': 0.1,
        'entropy_coeff': 0.005,

        'grad_clip': 1.0,
        'lambda': 0.8,
    })

    # config = ppo.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 0
    # config["num_workers"] = 1
    # config["eager"] = False
    # trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

    try:
        with tqdm(range(32768)) as pbar:
            for i in pbar:
                results = trainer.train()
                if i % 64 == 0:
                    avg_reward = results['episode_reward_mean']
                    pbar.set_description(
                        F'Iter: {i}; avg.rew={avg_reward:02f}')
                if i % 1024 == 0:
                    ckpt = trainer.save()
                    print(F'saved ckpt = {ckpt}')
    finally:
        ckpt = trainer.save()
        print(F'saved ckpt = {ckpt}')


if __name__ == '__main__':
    import argparse

    gpu_id = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CURICabinet")
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        ray_test(custom_args)
