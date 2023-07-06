![](../img/logo7.png)

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

![](../img/pipeline.png)

## Installation

Please refer to [Installation](https://rofunc.readthedocs.io/en/latest/installation.html) for installation.


## Available functions and future plans

The available functions and plans can be found as follows.

> **Note**
> ✅: Achieved 🔃: Reformatting ⛔: TODO

|                                      Data                                       |   |                                           Learning                                           |    |                                   P&C                                   |     |      Tools       |    |                                 Simulator                                 |    |
|:-------------------------------------------------------------------------------:|---|:--------------------------------------------------------------------------------------------:|----|:-----------------------------------------------------------------------:|-----|:----------------:|----|:-------------------------------------------------------------------------:|----|
|  [`xsens.record`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)   | ✅ |                                            `DMP`                                             | ⛔  |   [`LQT`](https://rofunc.readthedocs.io/en/latest/planning/lqt.html)    | ✅   |     `Config`     | ✅  | [`Franka`](https://rofunc.readthedocs.io/en/latest/simulator/franka.html) | ✅  |
|  [`xsens.export`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)   | ✅ |                                            `GMR`                                             | ✅  |                                 `LQTBi`                                 | ✅   | `robolab.coord`  | ✅  |   [`CURI`](https://rofunc.readthedocs.io/en/latest/simulator/curi.html)   | ✅  |
|  [`xsens.visual`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)   | ✅ |                                           `TPGMM`                                            | ✅  | [`LQTFb`](https://rofunc.readthedocs.io/en/latest/planning/lqt_fb.html) | ✅   |   `robolab.fk`   | ✅  |                                `CURIMini`                                 | 🔃 |
| [`opti.record`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html) | ✅ |                                          `TPGMMBi`                                           | ✅  | [`LQTCP`](https://rofunc.readthedocs.io/en/latest/planning/lqt_cp.html) | ✅   |   `robolab.ik`   | ✅  |                              `CURISoftHand`                               | ✅  |
| [`opti.export`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html) | ✅ |                                        `TPGMM_RPCtl`                                         | ✅  |                               `LQTCPDMP`                                | ✅   |   `robolab.fd`   | ⛔  |                                 `Walker`                                  | ✅  |
| [`opti.visual`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html) | ✅ |                                        `TPGMM_RPRepr`                                        | ✅  |                                  `LQR`                                  | ✅   |   `robolab.id`   | ⛔  |                                  `Gluon`                                  | 🔃 |
|    [`zed.record`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)     | ✅ |                                           `TPGMR`                                            | ✅  |                               `PoGLQRBi`                                | ✅   | `visualab.dist`  | ✅  |                                 `Baxter`                                  | 🔃 |
|    [`zed.export`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)     | ✅ |                                          `TPGMRBi`                                           | ✅  |  [`iLQR`](https://rofunc.readthedocs.io/en/latest/planning/ilqr.html)   | 🔃  | `visualab.ellip` | ✅  |                                 `Sawyer`                                  | 🔃 |
|    [`zed.visual`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)     | ✅ |                                           `TPHSMM`                                           | ✅  |                                `iLQRBi`                                 | 🔃  | `visualab.traj`  | ✅  |                               `Multi-Robot`                               | ✅  |
|                                  `emg.record`                                   | ✅ |       [`RLBaseLine(SKRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RLBaseLine/SKRL.html) | ✅  |                                `iLQRFb`                                 | 🔃  |                  |    |                                                                           |    |
|                                  `emg.export`                                   | ✅ |                                     `RLBaseLine(RLlib)`                                      | ✅  |                                `iLQRCP`                                 | 🔃  |                  |    |                                                                           |    |
|                                  `emg.visual`                                   | ✅ |                                     `RLBaseLine(ElegRL)`                                     | ✅  |                               `iLQRDyna`                                | 🔃  |                  |    |                                                                           |    |
|                                 `mmodal.record`                                 | ⛔ |                                       `BCO(RofuncIL)`                                        | 🔃 |                                `iLQRObs`                                | 🔃  |                  |    |                                                                           |    |
|                                 `mmodal.export`                                 | ✅ |                                       `BC-Z(RofuncIL)`                                       | ⛔  |                                  `MPC`                                  | ⛔   |                  |    |                                                                           |    |
|                                                                                 |   |                                      `STrans(RofuncIL)`                                      | ⛔  |                                  `RMP`                                  | ⛔   |                  |    |                                                                           |    |
|                                                                                 |   |                                       `RT-1(RofuncIL)`                                       | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |       [`A2C(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/A2C.html)       | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |       [`PPO(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/PPO.html)       | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |       [`SAC(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/SAC.html)       | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |       [`TD3(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/TD3.html)       | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |                                       `CQL(RofuncRL)`                                        | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |                                      `TD3BC(RofuncRL)`                                       | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |                                      `DTrans(RofuncRL)`                                      | 🔃 |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |                                       `EDAC(RofuncRL)`                                       | ⛔  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |       [`AMP(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/AMP.html)       | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |       [`ASE(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/ASE.html)       | ✅  |                                                                         |     |                  |    |                                                                           |    |
|                                                                                 |   |                                     `ODTrans(RofuncRL)`                                      | ⛔  |                                                                         |     |                  |    |                                                                           |    |

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


```{toctree}
:maxdepth: 3
:caption: Get Started
:hidden:
:glob:

installation
quickstart
auto_examples/index
```

```{toctree}
:maxdepth: 3
:caption: Core Modules
:hidden:
:glob:

devices/index
lfd/index
planning/index
tools/index
simulator/index
```

```{toctree}
:maxdepth: 3
:caption: API Reference
:hidden:
:glob:

apidocs/index
```