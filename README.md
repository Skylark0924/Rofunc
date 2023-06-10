![](doc/img/logo7.png)

# Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation

[![Release](https://img.shields.io/github/v/release/Skylark0924/Rofunc)](https://pypi.org/project/rofunc/)
![License](https://img.shields.io/github/license/Skylark0924/Rofunc?color=blue)
![](https://img.shields.io/github/downloads/skylark0924/Rofunc/total)
[![](https://img.shields.io/github/issues-closed-raw/Skylark0924/Rofunc?color=brightgreen)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aissue+is%3Aclosed)
[![](https://img.shields.io/github/issues-raw/Skylark0924/Rofunc?color=orange)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aopen+is%3Aissue)
[![Documentation Status](https://readthedocs.org/projects/rofunc/badge/?version=latest)](https://rofunc.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FSkylark0924%2FRofunc%2Fbadge%3Fref%3Dmain&style=flat)](https://actions-badge.atrox.dev/Skylark0924/Rofunc/goto?ref=main)

> **Repository address: https://github.com/Skylark0924/Rofunc**

Rofunc package focuses on the **Imitation Learning (IL), Reinforcement Learning (RL) and Learning from Demonstration (LfD)** for 
**(Humanoid) Robot Manipulation**. It provides valuable and convenient python functions, including _demonstration collection, data
pre-processing, LfD algorithms, planning, and control methods_. We also provide an Isaac Gym-based robot simulator for
evaluation. This package aims to advance the field by building a full-process toolkit and validation platform that
simplifies and standardizes the process of demonstration data collection, processing, learning, and its deployment on
robots.

![](doc/img/pipeline.png)

- [Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation](#rofunc-the-full-process-python-package-for-robot-learning-from-demonstration-and-robot-manipulation)
  - [Installation](#installation)
    - [Install from PyPI (stable version)](#install-from-pypi-stable-version)
    - [Install from Source (nightly version, recommended)](#install-from-source-nightly-version-recommended)
  - [Documentation](#documentation)
  - [Star History](#star-history)
  - [Cite](#cite)
  - [Related Papers](#related-papers)
  - [The Team](#the-team)
  - [Acknowledge](#acknowledge)
    - [Learning from Demonstration](#learning-from-demonstration)
    - [Planning and Control](#planning-and-control)


## Installation

### Install from PyPI (stable version)

The installation is very easy,

```
pip install rofunc
```

and as you'll find later, it's easy to use as well!

```python
import rofunc as rf
```

Thus, have fun in the robotics world!
> **Note**
> Several requirements need to be installed before using the package. Please refer to
> the [installation guide](https://rofunc.readthedocs.io/en/latest/overview.html#installation) for more details.

### Install from Source (nightly version, recommended)

```python
git clone https://github.com/Skylark0924/Rofunc.git
cd Rofunc

# Create a conda environment
# Python 3.8 is strongly recommended
conda create -n rofunc python=3.8

# For Linux user
sh ./scripts/install.sh
# [Option] Install with baseline RL frameworks (SKRL, RLlib, Stable Baselines3)
sh ./scripts/install_w_baselines.sh
# [Option] For MacOS user (brew is required, Isaac Gym based simulator is not supported on MacOS)
sh ./scripts/mac_install.sh
```

> **Note**
> If you want to use functions related to ZED camera, you need to
> install [ZED SDK](https://www.stereolabs.com/developers/release/#downloads) manually. (We have tried to package it as
> a `.whl` file to add it to `requirements.txt`, unfortunately, the ZED SDK is not very friendly and doesn't support
> direct installation.)

## Documentation

[![Documentation](https://img.shields.io/badge/Documentation-Access-brightgreen?style=for-the-badge)](https://rofunc.readthedocs.io/en/latest/)
[![Example Gallery](https://img.shields.io/badge/Example%20Gallery-Access-brightgreen?style=for-the-badge)](https://rofunc.readthedocs.io/en/latest/auto_examples/index.html)

> **Note**
> Currently, we provide a simple document; please refer to [here](./rofunc/).
> A comprehensive one with both English and Chinese versions is built via
> the [readthedoc](https://rofunc.readthedocs.io/en/latest/).
> We provide a simple but interesting example: learning to play
> Taichi by learning from human demonstration.

To give you a quick overview of the pipeline of `rofunc`, we provide an interesting example of learning to play Taichi
from human demonstration. You can find it in the [Quick start](https://rofunc.readthedocs.io/en/latest/quickstart.html)
section of the documentation.

The available functions and plans can be found as follows.

> **Note**
> ✅: Achieved 🔃: Reformatting ⛔: TODO

|                                      Data                                       |   |      Learning       |    |                                   P&C                                   |     |      Tools       |    |                                 Simulator                                 |    |
|:-------------------------------------------------------------------------------:|---|:-------------------:|----|:-----------------------------------------------------------------------:|-----|:----------------:|----|:-------------------------------------------------------------------------:|----|
|  [`xsens.record`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)   | ✅ |        `DMP`        | ⛔  |   [`LQT`](https://rofunc.readthedocs.io/en/latest/planning/lqt.html)    | ✅   |     `Config`     | ✅  | [`Franka`](https://rofunc.readthedocs.io/en/latest/simulator/franka.html) | ✅  |
|  [`xsens.export`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)   | ✅ |        `GMR`        | ✅  |                                 `LQTBi`                                 | ✅   | `robolab.coord`  | ✅  |   [`CURI`](https://rofunc.readthedocs.io/en/latest/simulator/curi.html)   | ✅  |
|  [`xsens.visual`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)   | ✅ |       `TPGMM`       | ✅  | [`LQTFb`](https://rofunc.readthedocs.io/en/latest/planning/lqt_fb.html) | ✅   |   `robolab.fk`   | ✅  |                                `CURIMini`                                 | 🔃 |
| [`opti.record`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html) | ✅ |      `TPGMMBi`      | ✅  | [`LQTCP`](https://rofunc.readthedocs.io/en/latest/planning/lqt_cp.html) | ✅   |   `robolab.ik`   | ✅  |                              `CURISoftHand`                               | ✅  |
| [`opti.export`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html) | ✅ |    `TPGMM_RPCtl`    | ✅  |                               `LQTCPDMP`                                | ✅   |   `robolab.fd`   | ⛔  |                                 `Walker`                                  | ✅  |
| [`opti.visual`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html) | ✅ |   `TPGMM_RPRepr`    | ✅  |                                  `LQR`                                  | ✅   |   `robolab.id`   | ⛔  |                                  `Gluon`                                  | 🔃 |
|    [`zed.record`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)     | ✅ |       `TPGMR`       | ✅  |                               `PoGLQRBi`                                | ✅   | `visualab.dist`  | ✅  |                                 `Baxter`                                  | 🔃 |
|    [`zed.export`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)     | ✅ |      `TPGMRBi`      | ✅  |  [`iLQR`](https://rofunc.readthedocs.io/en/latest/planning/ilqr.html)   | 🔃  | `visualab.ellip` | ✅  |                                 `Sawyer`                                  | 🔃 |
|    [`zed.visual`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)     | ✅ |      `TPHSMM`       | ✅  |                                `iLQRBi`                                 | 🔃  | `visualab.traj`  | ✅  |                               `Multi-Robot`                               | ✅  |
|                                  `emg.record`                                   | ✅ |        `BCO`        | 🔃 |                                `iLQRFb`                                 | 🔃  |                  |    |                                                                           |    |
|                                  `emg.export`                                   | ✅ |      `STrans`       | ⛔  |                                `iLQRCP`                                 | 🔃  |                  |    |                                                                           |    |
|                                  `emg.visual`                                   | ✅ |     `PPO(SKRL)`     | ✅  |                               `iLQRDyna`                                | 🔃  |                  |    |                                                                           |    |
|                                 `mmodal.record`                                 | ⛔ |     `SAC(SKRL)`     | ✅  |                                `iLQRObs`                                | 🔃  |                  |    |                                                                           |    |
|                                 `mmodal.export`                                 | ✅ |     `TD3(SKRL)`     | ✅  |                                  `MPC`                                  | ⛔   |                  |    |                                                                           |    |
|                                                                                 |   |     `PPO(SB3)`      | ⛔  |                                  `RMP`                                  | ⛔   |                  |    |                                                                           |    |
|                                                                                 |   |     `SAC(SB3)`      | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |     `TD3(SB3)`      | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |    `PPO(RLlib)`     | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |    `SAC(RLlib)`     | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |    `TD3(RLlib)`     | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |    `PPO(ElegRL)`    | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |    `SAC(ElegRL)`    | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |    `TD3(ElegRL)`    | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |   `PPO(RofuncRL)`   | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |   `SAC(RofuncRL)`   | 🔃 |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |   `TD3(RofuncRL)`   | 🔃 |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   | `ODTrans(RofuncRL)` | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |  `RT-1(RofuncRL)`   | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |   `CQL(RofuncRL)`   | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |  `TD3BC(RofuncRL)`  | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   | `DTrans(RofuncRL)`  | 🔃 |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |  `EDAC(RofuncRL)`   | ⛔  |                                                                         |     |                  |    |                                                                           |    |
                                                                                  


  


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skylark0924/Rofunc&type=Date)](https://star-history.com/#Skylark0924/Rofunc&Date)

## Citation

If you use rofunc in a scientific publication, we would appreciate citations to the following paper:

```
@misc{Rofunc2022,
      author = {Liu, Junjia and Li, Chenzui and Delehelle, Donatien and Li, Zhihao and Chen, Fei},
      title = {Rofunc: The full process python package for robot learning from demonstration and robot manipulation},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Skylark0924/Rofunc}},
}
```

## Related Papers

1. Robot cooking with stir-fry: Bimanual non-prehensile manipulation of semi-fluid objects ([IEEE RA-L 2022](https://arxiv.org/abs/2205.05960) | [Code](./rofunc/learning/dl/structured_transformer/strans.py))
```
@article{liu2022robot,
         title={Robot cooking with stir-fry: Bimanual non-prehensile manipulation of semi-fluid objects},
         author={Liu, Junjia and Chen, Yiting and Dong, Zhipeng and Wang, Shixiong and Calinon, Sylvain and Li, Miao and Chen, Fei},
         journal={IEEE Robotics and Automation Letters},
         volume={7},
         number={2},
         pages={5159--5166},
         year={2022},
         publisher={IEEE}
}
```
2. SoftGPT: Learn Goal-oriented Soft Object Manipulation Skills by Generative Pre-trained Heterogeneous Graph Transformer (IROS 2023)
3. Learning Robot Generalized Bimanual Coordination using Relative Parameterization Method on Human Demonstration (IEEE CDC 2023 | [Code](./rofunc/learning/ml/tpgmm.py))


## The Team

Rofunc is developed and maintained by the [CLOVER Lab (Collaborative and Versatile Robots Laboratory)](https://feichenlab.com/), CUHK.

## Acknowledge

We would like to acknowledge the following projects:

### Learning from Demonstration

1. [pbdlib](https://gitlab.idiap.ch/rli/pbdlib-python)
2. [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
3. [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
4. [SKRL](https://github.com/Toni-SM/skrl)

### Planning and Control

1. [Robotics codes from scratch (RCFS)](https://gitlab.idiap.ch/rli/robotics-codes-from-scratch)
