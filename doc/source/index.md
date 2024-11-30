![](../img/logo/logo7.png)

# Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation

[![Release](https://img.shields.io/github/v/release/Skylark0924/Rofunc)](https://pypi.org/project/rofunc/)
![License](https://img.shields.io/github/license/Skylark0924/Rofunc?color=blue)
![](https://img.shields.io/github/downloads/skylark0924/Rofunc/total)
[![](https://img.shields.io/github/issues-closed-raw/Skylark0924/Rofunc?color=brightgreen)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aissue+is%3Aclosed)
[![](https://img.shields.io/github/issues-raw/Skylark0924/Rofunc?color=orange)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aopen+is%3Aissue)
[![Documentation Status](https://readthedocs.org/projects/rofunc/badge/?version=latest)](https://rofunc.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FSkylark0924%2FRofunc%2Fbadge%3Fref%3Dmain&style=flat)](https://actions-badge.atrox.dev/Skylark0924/Rofunc/goto?ref=main)

> **Repository address: https://github.com/Skylark0924/Rofunc** <br>
> **Documentation: https://rofunc.readthedocs.io/**

Rofunc package focuses on the **Imitation Learning (IL), Reinforcement Learning (RL) and Learning from Demonstration (
LfD)** for
**(Humanoid) Robot Manipulation**. It provides valuable and convenient python functions, including _demonstration
collection, data pre-processing, LfD algorithms, planning, and control methods_. We also provide an Isaac Gym-based
robot simulator for
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

|                                                  Data                                                   |   |                                                Learning                                                |    |                                                        P&C                                                         |    |                                                        Tools                                                        |   |                                                  Simulator                                                   |    |
|:-------------------------------------------------------------------------------------------------------:|---|:------------------------------------------------------------------------------------------------------:|----|:------------------------------------------------------------------------------------------------------------------:|----|:-------------------------------------------------------------------------------------------------------------------:|---|:------------------------------------------------------------------------------------------------------------:|----|
|              [`xsens.record`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)               | ✅ |                                                 `DMP`                                                  | ⛔  |                         [`LQT`](https://rofunc.readthedocs.io/en/latest/planning/lqt.html)                         | ✅  |                                                      `config`                                                       | ✅ |                  [`Franka`](https://rofunc.readthedocs.io/en/latest/simulator/franka.html)                   | ✅  |
|              [`xsens.export`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)               | ✅ |      [`GMR`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.gmr.html)       | ✅  |       [`LQTBi`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqt.lqt.html)       | ✅  |      [`logger`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.logger.beauty_logger.html)      | ✅ |                    [`CURI`](https://rofunc.readthedocs.io/en/latest/simulator/curi.html)                     | ✅  |
|              [`xsens.visual`](https://rofunc.readthedocs.io/en/latest/devices/xsens.html)               | ✅ |    [`TPGMM`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tpgmm.html)     | ✅  |                      [`LQTFb`](https://rofunc.readthedocs.io/en/latest/planning/lqt_fb.html)                       | ✅  |            [`datalab`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.datalab.html)            | ✅ |                                                  `CURIMini`                                                  | 🔃 |
|             [`opti.record`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html)             | ✅ |   [`TPGMMBi`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tpgmm.html)    | ✅  |                      [`LQTCP`](https://rofunc.readthedocs.io/en/latest/planning/lqt_cp.html)                       | ✅  | [`robolab.coord`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.robolab.coord.transform.html) | ✅ |   [`CURISoftHand`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.simulator.curi_sim.html)    | ✅  |
|             [`opti.export`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html)             | ✅ | [`TPGMM_RPCtl`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tpgmm.html)  | ✅  |  [`LQTCPDMP`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqt.lqt_cp_dmp.html)  | ✅  |   [`robolab.fk`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.robolab.kinematics.fk.html)    | ✅ |     [`Walker`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.simulator.walker_sim.html)      | ✅  |
|             [`opti.visual`](https://rofunc.readthedocs.io/en/latest/devices/optitrack.html)             | ✅ | [`TPGMM_RPRepr`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tpgmm.html) | ✅  |        [`LQR`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqr.lqr.html)        | ✅  |   [`robolab.ik`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.robolab.kinematics.ik.html)    | ✅ |                                                   `Gluon`                                                    | 🔃 |
|                [`zed.record`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)                 | ✅ |    [`TPGMR`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tpgmr.html)     | ✅  |     [`PoGLQRBi`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqr.lqr.html)      | ✅  |                                                    `robolab.fd`                                                     | ⛔ |                                                   `Baxter`                                                   | 🔃 |
|                [`zed.export`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)                 | ✅ |   [`TPGMRBi`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tpgmr.html)    | ✅  |                        [`iLQR`](https://rofunc.readthedocs.io/en/latest/planning/ilqr.html)                        | 🔃 |                                                    `robolab.id`                                                     | ⛔ |                                                   `Sawyer`                                                   | 🔃 |
|                [`zed.visual`](https://rofunc.readthedocs.io/en/latest/devices/zed.html)                 | ✅ |   [`TPHSMM`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.learning.ml.tphsmm.html)    | ✅  |    [`iLQRBi`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqr.ilqr_bi.html)     | 🔃 |  [`visualab.dist`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.visualab.distribution.html)  | ✅ |   [`Humanoid`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.simulator.humanoid_sim.html)    | ✅  |
|  [`emg.record`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.devices.emg.record.html)  | ✅ |         [`RLBaseLine(SKRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RLBaseLine/SKRL.html)         | ✅  |                                                      `iLQRFb`                                                      | 🔃 |   [`visualab.ellip`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.visualab.ellipsoid.html)   | ✅ | [`Multi-Robot`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.simulator.multirobot_sim.html) | ✅  |
|  [`emg.export`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.devices.emg.export.html)  | ✅ |                                          `RLBaseLine(RLlib)`                                           | ✅  |    [`iLQRCP`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqr.ilqr_cp.html)     | 🔃 |   [`visualab.traj`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.visualab.trajectory.html)   | ✅ |                                                                                                              |    |
|                                             `mmodal.record`                                             | ⛔ |                                          `RLBaseLine(ElegRL)`                                          | ✅  |  [`iLQRDyna`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqr.ilqr_dyna.html)   | 🔃 |   [`oslab.dir_proc`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.oslab.dir_process.html)    | ✅ |                                                                                                              |    |
| [`mmodal.sync`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.devices.mmodal.sync.html) | ✅ |                                            `BCO(RofuncIL)`                                             | 🔃 | [`iLQRObs`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.planning_control.lqr.ilqr_obstacle.html) | 🔃 |  [`oslab.file_proc`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.oslab.file_process.html)   | ✅ |                                                                                                              |    |
|                                                                                                         |   |                                            `BC-Z(RofuncIL)`                                            | ⛔  |                                                       `MPC`                                                        | ⛔  |     [`oslab.internet`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.oslab.internet.html)     | ✅ |                                                                                                              |    |
|                                                                                                         |   |                                           `STrans(RofuncIL)`                                           | ⛔  |                                                       `RMP`                                                        | ⛔  |         [`oslab.path`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.utils.oslab.path.html)         | ✅ |                                                                                                              |    |
|                                                                                                         |   |                                            `RT-1(RofuncIL)`                                            | ⛔  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |            [`A2C(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/A2C.html)            | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |            [`PPO(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/PPO.html)            | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |            [`SAC(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/SAC.html)            | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |            [`TD3(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/TD3.html)            | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |                                            `CQL(RofuncRL)`                                             | ⛔  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |                                           `TD3BC(RofuncRL)`                                            | ⛔  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |         [`DTrans(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/DTrans.html)         | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |                                            `EDAC(RofuncRL)`                                            | ⛔  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |            [`AMP(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/AMP.html)            | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |            [`ASE(RofuncRL)`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/ASE.html)            | ✅  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |
|                                                                                                         |   |                                          `ODTrans(RofuncRL)`                                           | ⛔  |                                                                                                                    |    |                                                                                                                     |   |                                                                                                              |    |

## Citation

If you use rofunc in a scientific publication, we would appreciate citations to the following paper:

```
@software{liu2023rofunc, 
          title = {Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation},
          author = {Liu, Junjia and Dong, Zhipeng and Li, Chenzui and Li, Zhihao and Yu, Minghao and Delehelle, Donatien and Chen, Fei},
          year = {2023},
          publisher = {Zenodo},
          doi = {10.5281/zenodo.10016946},
          url = {https://doi.org/10.5281/zenodo.10016946},
          dimensions = {true},
          google_scholar_id = {0EnyYjriUFMC},
}
```

> **Warning** <br>
> **If our code is found to be used in a published paper without proper citation, we reserve the right to address this
> issue formally by contacting the editor to report potential academic misconduct!**
>
> **如果我们的代码被发现用于已发表的论文而没有被恰当引用，我们保留通过正式联系编辑报告潜在学术不端行为的权利。**

## Related Papers

1. Robot cooking with stir-fry: Bimanual non-prehensile manipulation of semi-fluid
   objects ([IEEE RA-L 2022](https://arxiv.org/abs/2205.05960) | [Code](../../rofunc/learning/RofuncIL/structured_transformer/strans.py))

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

2. SoftGPT: Learn Goal-oriented Soft Object Manipulation Skills by Generative Pre-trained Heterogeneous Graph
   Transformer ([IROS 2023](https://arxiv.org/abs/2306.12677)｜Code coming soon)

```
@inproceedings{liu2023softgpt,
               title={Softgpt: Learn goal-oriented soft object manipulation skills by generative pre-trained heterogeneous graph transformer},
               author={Liu, Junjia and Li, Zhihao and Lin, Wanyu and Calinon, Sylvain and Tan, Kay Chen and Chen, Fei},
               booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
               pages={4920--4925},
               year={2023},
               organization={IEEE}
}
```

3. BiRP: Learning Robot Generalized Bimanual Coordination using Relative Parameterization Method on Human
   Demonstration ([IEEE CDC 2023](https://arxiv.org/abs/2307.05933) | [Code](../../rofunc/learning/RofuncML/tpgmm.py))

```
@article{liu2023birp,
        title={BiRP: Learning Robot Generalized Bimanual Coordination using Relative Parameterization Method on Human Demonstration},
        author={Liu, Junjia and Sim, Hengyi and Li, Chenzui and Chen, Fei},
        journal={arXiv preprint arXiv:2307.05933},
        year={2023}
}
```

## The Team

Rofunc is developed and maintained by
the [CLOVER Lab (Collaborative and Versatile Robots Laboratory)](https://feichenlab.com/), CUHK.

```{toctree}
:maxdepth: 3
:caption: Get Started
:hidden:
:glob:

installation
quickstart
examples/index
```

```{toctree}
:maxdepth: 3
:caption: Tutorial
:hidden:
:glob:

tutorial/config_system
tutorial/customizeRL
tutorial/customizeIL
tutorial/customizeML
tutorial/customizePC
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