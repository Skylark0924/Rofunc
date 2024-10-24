#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

from typing import Optional, Union, Tuple

import torch
from torch.distributions import Normal


class Noise:
    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Base class representing a noise
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

    def sample_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Sample a noise with the same size (shape) as the input tensor
        :param tensor: Input tensor used to determine output tensor size (shape)
        :return: Sampled noise
        """
        return self.sample(tensor.size())

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """
        Noise sampling method to be implemented by the inheriting classes
        :param size: Shape of the sampled tensor
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Class representing a Gaussian noise
        :param mean: Mean of the normal distribution
        :param std: Standard deviation of the normal distribution
        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        """
        super().__init__(device)
        self.distribution = Normal(loc=torch.tensor(mean, device=self.device, dtype=torch.float32),
                                   scale=torch.tensor(std, device=self.device, dtype=torch.float32))

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """
        Sample a Gaussian noise
        :param size: Shape of the sampled tensor
        """
        return self.distribution.sample(size)
