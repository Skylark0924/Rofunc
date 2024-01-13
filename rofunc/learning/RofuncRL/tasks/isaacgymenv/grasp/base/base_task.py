# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from abc import ABC, abstractmethod
from gym import spaces, Space
from isaacgym import gymapi
import numpy as np
import torch
from typing import *
import warnings


class Task(ABC):
    def __init__(
            self,
            cfg: Dict[str, Any],
            sim_device: str,
            graphics_device_id: int,
            headless: bool
    ) -> None:
        """Initializes the basic reinforcement learning task.

        Args:
            cfg: Configuration dictionary that contains parameters for
                how the simulation should run and defines properties of the
                task.
            sim_device: Device on which the physics simulation is run
                (e.g. "cuda:0" or "cpu").
            graphics_device_id: ID of the device to run the rendering.
            headless:  Whether to run the simulation without a viewer.
        """
        self.cfg = cfg
        self.headless = headless
        self._init_device(sim_device, cfg)
        self._init_graphics_device(graphics_device_id, cfg)
        self._init_environment_specs(cfg)
        self.sim_params = self._parse_sim_params(cfg["physics_engine"],
                                                 cfg["sim"])
        self._set_optimization_flags()

        self.clip_obs = cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = cfg["env"].get("clipActions", np.Inf)
        self.step_simulation = step_simulation

    def __len__(self) -> int:
        return self.num_envs

    def __repr__(self) -> str:
        return (f"{self.cfg['name']} "
                f"(num_envs={len(self)}, "
                f"observation_space={self.observation_space},"
                f"action_space={self.action_space})")

    def _init_device(self, sim_device: str, cfg: Dict[str, Any]) -> None:
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if cfg["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() in ["gpu", "cuda"]:
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                warnings.warn("Tried to use GPU pipeline with simulation "
                              "running on CPU.")
                cfg["sim"]["use_gpu_pipeline"] = False
        self.rl_device = cfg.get("rl_device", "cuda:0")

    def _init_graphics_device(self, graphics_device_id: int,
                              cfg: Dict[str, Any]) -> None:
        enable_camera_sensors = cfg["sim"].get("enableCameraSensors", False)
        if "cameras" in cfg.keys():
            enable_camera_sensors = True

        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors is False and self.headless is True:
            self.graphics_device_id = -1

    def _init_environment_specs(self, cfg: Dict[str, Any]) -> None:
        self.num_environments = cfg["env"]["numEnvs"]
        self.num_agents = cfg["env"].get("numAgents", 1)
        self.num_observations = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["control"].get("controlFreqInv", 1)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf,
                                    np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf,
                                      np.ones(self.num_states) * np.Inf)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1.,
                                    np.ones(self.num_actions) * 1.)

    def _parse_sim_params(self, physics_engine: str,
                          sim_cfg: Dict[str, Any]) -> gymapi.SimParams:
        sim_params = gymapi.SimParams()

        sim_params.dt = sim_cfg["dt"]
        sim_params.num_client_threads = sim_cfg.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = sim_cfg["use_gpu_pipeline"]
        sim_params.substeps = sim_cfg.get("substeps", 2)
        sim_params.gravity = gymapi.Vec3(*sim_cfg["gravity"])

        if sim_cfg["up_axis"] == "z":
            self.up_axis_idx = 2
            sim_params.up_axis = gymapi.UP_AXIS_Z
        elif sim_cfg["up_axis"] == "y":
            self.up_axis_idx = 1
            sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid value for up-axis given.")

        if physics_engine == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            if "physx" in sim_cfg:
                for opt in sim_cfg["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt,
                                gymapi.ContactCollection(sim_cfg["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, sim_cfg["physx"][opt])
        else:
            raise ValueError("Physics engine should always be PhysX.")
        return sim_params

    @staticmethod
    def _set_optimization_flags() -> None:
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

    @abstractmethod
    def allocate_buffers(self):
        """Allocate PyTorch buffers for obs, rew, etc."""
        raise NotImplementedError

    @abstractmethod
    def step(
            self,
            actions: torch.Tensor
    ) -> Tuple[
            Dict[str, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Dict[str, Any]]:
        """Steps the environment.

        Args:
            actions: Actions to apply to the environments
        Returns:
            obs, rew, done, info
        """
        raise NotImplementedError

    @abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Applies actions to the environment.

        Args:
            actions: Actions to apply.
        """
        raise NotImplementedError

    @abstractmethod
    def post_physics_step(self):
        """Steps the environment. Computes observation, rewards and resets any
        envs that require it."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments.

        Returns:
            obs
        """
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environments specified by env_ids.
        """
        raise NotImplementedError

    @property
    def observation_space(self) -> Space:
        return self.obs_space

    @property
    def action_space(self) -> Space:
        return self.act_space

    @property
    def num_envs(self) -> int:
        return self.num_environments

    @property
    def num_acts(self) -> int:
        return self.num_actions

    @property
    def num_obs(self) -> int:
        return self.num_observations


def step_simulation(gym, sim) -> None:
    gym.simulate(sim)
