# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym.spaces
import isaacgym
import numpy as np
import torch
from ray.rllib.env import VectorEnv

from rofunc.learning.RofuncRL.utils.elegantrl_utils import ElegantRLIsaacGymEnvWrapper


class RLlibIsaacGymEnvWrapper(gym.Env):
    def __init__(self, env_config):
        from rofunc.learning.RofuncRL.tasks import task_map
        env = task_map[env_config["task_name"]](cfg=env_config["task_cfg_dict"],
                                                rl_device=env_config["cfg"].rl_device,
                                                sim_device=env_config["cfg"].sim_device,
                                                graphics_device_id=env_config["cfg"].graphics_device_id,
                                                headless=env_config["cfg"].headless,
                                                virtual_screen_capture=env_config["cfg"].capture_video,
                                                force_render=env_config["cfg"].force_render)
        self.env = ElegantRLIsaacGymEnvWrapper(env=env, cfg=env_config["cfg"])
        self.action_space = self.env.env.action_space
        self.observation_space = self.env.env.observation_space

    def reset(self):
        return np.array(self.env.reset().cpu())[0]

    def step(self, action):
        action = torch.tensor(np.array([action])).to(self.env.device)
        observations, rewards, dones, info_dict = self.env.step(action)
        return np.array(observations.cpu())[0], np.array(rewards.cpu())[0], np.array(dones.cpu())[0], info_dict


class RLlibIsaacGymVecEnvWrapper(VectorEnv):
    def __init__(self, env_config):
        # self.env = IsaacVecEnv(env_name=env_config["task_name"], env_num=1024, sim_device_id=env_config["gpu_id"],
        #                        rl_device_id=env_config["gpu_id"], should_print=True)
        from rofunc.learning.RofuncRL.tasks import task_map
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

    def seed(self, seed):
        pass

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
        info_dict = [{"time_outs": info_dict_raw["time_outs"][i]} for i in range(self.num_envs)]
        obs = np.array(observations.cpu()).reshape((self.num_envs, -1))
        self._prv_obs = obs

        return obs, np.array(rewards.cpu()), np.array(dones.cpu()), info_dict
