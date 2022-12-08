import gym
import numpy as np
from ray.rllib import BaseEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn, VecEnvObs
from typing import Optional, List, Union, Sequence, Type, Any

import RayEnvWrapper


class WrapperRayVecEnv(VecEnv):

    def __init__(self, make_env, num_workers, per_worker_env):
        self.one_env = make_env(0)
        self.remote: BaseEnv = RayEnvWrapper.CustomRayRemoteVectorEnv(make_env, num_workers, per_worker_env, False)
        super(WrapperRayVecEnv, self).__init__(num_workers * per_worker_env, self.one_env.observation_space, self.one_env.action_space)

    def reset(self) -> VecEnvObs:
        return self.remote.poll()[0]

    def step_async(self, actions: np.ndarray) -> None:
        self.remote.send_actions(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.remote.poll()

    def close(self) -> None:
        self.remote.stop()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        pass

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass