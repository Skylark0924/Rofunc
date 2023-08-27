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
import torch
import torch.nn as nn
from omegaconf import DictConfig

import rofunc as rf
from rofunc.config.utils import omegaconf_to_dict
from rofunc.learning.RofuncRL.models.base_models import BaseMLP


class EmptyEncoder(nn.Module):
    def __init__(self):
        super(EmptyEncoder, self).__init__()

    def forward(self, x):
        return x


class BaseEncoder(nn.Module):
    def __init__(self, cfg: DictConfig, cfg_name: str = 'state_encoder'):
        super(BaseEncoder, self).__init__()

        self.cfg = cfg
        self.cfg_dict = omegaconf_to_dict(self.cfg)
        self.cfg_name = cfg_name

        self.input_dim = self.cfg_dict[cfg_name]['inp_channels']
        self.output_dim = self.cfg_dict[cfg_name]['out_channels']

        self.use_pretrained = self.cfg_dict[cfg_name]['use_pretrained']
        self.freeze = self.cfg_dict[cfg_name]['freeze']
        self.model_ckpt = self.cfg_dict[cfg_name]['model_ckpt']
        self.model_module_name = self.cfg_dict[cfg_name]['model_module_name'] if 'model_module_name' in self.cfg_dict[
            cfg_name] else None

    def set_up(self):
        if self.freeze:
            self.freeze_network()
        if self.use_pretrained:
            self.pre_trained_mode()

    def freeze_network(self):
        for net in self.freeze_net_list:
            for param in net.parameters():
                param.requires_grad = False
        rf.logger.beauty_print(f"Freeze state encoder", type="info")

    def pre_trained_mode(self):
        if self.use_pretrained is True and self.model_ckpt is None:
            raise ValueError("Cannot freeze the encoder without a checkpoint")
        if self.use_pretrained:
            self.load_ckpt(self.model_ckpt)

    def _get_internal_value(self, module):
        return module.state_dict() if hasattr(module, "state_dict") else module

    def save_ckpt(self, path: str):
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = self._get_internal_value(module)
        torch.save(modules, path)

    def load_ckpt(self, path: str):
        modules = torch.load(path)
        if self.model_module_name is not None:
            modules = modules[self.model_module_name]
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
        rf.logger.beauty_print(f"Loaded pretrained state encoder model from {self.model_ckpt}", type="info")


class MLPEncoder(BaseMLP):
    def __init__(self, cfg, cfg_name):
        super().__init__(cfg, cfg_name)
