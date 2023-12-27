![](doc/img/logo8.png)

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

Rofunc package focuses on the **Imitation Learning (IL), Reinforcement Learning (RL) and Learning from Demonstration (LfD)** for **(Humanoid) Robot Manipulation**. It provides valuable and convenient python functions, including
_demonstration collection, data pre-processing, LfD algorithms, planning, and control methods_. We also provide an
`IsaacGym` and `OmniIsaacGym` based robot simulator for evaluation. This package aims to advance the field by building a full-process
toolkit and validation platform that simplifies and standardizes the process of demonstration data collection,
processing, learning, and its deployment on robots.

![](doc/img/pipeline.png)

- [Rofunc: The Full Process Python Package for Robot Learning from Demonstration and Robot Manipulation](#rofunc-the-full-process-python-package-for-robot-learning-from-demonstration-and-robot-manipulation)
  - [Update News 🎉🎉🎉](#update-news-)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [RofuncRL](#rofuncrl)
  - [Star History](#star-history)
  - [Citation](#citation)
  - [Related Papers](#related-papers)
  - [The Team](#the-team)
  - [Acknowledge](#acknowledge)
    - [Learning from Demonstration](#learning-from-demonstration)
    - [Planning and Control](#planning-and-control)


## Update News 🎉🎉🎉
- [2023-12-24] 🚀 [Dexterous hand (Shadow Hand, Allegro Hand, qbSofthand) tasks](https://github.com/Skylark0924/Rofunc/blob/main/examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py) are supported to be trained by `RofuncRL`.
- [2023-12-03] 🖼️ [Segment-Anything (SAM)](https://segment-anything.com/) is supported in an interactive mode, check the examples in Visualab ([segment anything](https://github.com/Skylark0924/Rofunc/blob/main/examples/visualab/example_sam_seg.py), [segment with prompt](https://github.com/Skylark0924/Rofunc/blob/main/examples/visualab/example_sam_seg_w_prompt.py)).
- **[2023-10-31] 🚀 [`RofuncRL`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/index.html): A modular easy-to-use Reinforcement Learning sub-package designed for Robot Learning tasks is released. It has been tested with simulators like `OpenAIGym`, `IsaacGym`, `OmniIsaacGym` (see [example gallery](https://rofunc.readthedocs.io/en/latest/examples/learning_rl/index.html)), and also differentiable simulators like `PlasticineLab` and `DiffCloth`.**
- ...
- If you want to know more about the update news, please refer to the [changelog](https://github.com/Skylark0924/Rofunc/blob/main/changelog.md).


## Installation

Please refer to the [installation guide](https://rofunc.readthedocs.io/en/latest/installation.html).

## Documentation

[![Documentation](https://img.shields.io/badge/Documentation-Access-brightgreen?style=for-the-badge)](https://rofunc.readthedocs.io/en/latest/)
[![Example Gallery](https://img.shields.io/badge/Example%20Gallery-Access-brightgreen?style=for-the-badge)](https://rofunc.readthedocs.io/en/latest/examples/index.html)

To give you a quick overview of the pipeline of `rofunc`, we provide an interesting example of learning to play Taichi
from human demonstration. You can find it in the [Quick start](https://rofunc.readthedocs.io/en/latest/quickstart.html)
section of the documentation.

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

## RofuncRL

`RofuncRL` is one of the most important sub-packages of `Rofunc`. It is a modular easy-to-use Reinforcement Learning sub-package designed for Robot Learning tasks. It has been tested with simulators like `OpenAIGym`, `IsaacGym`, `OmniIsaacGym` (see [example gallery](https://rofunc.readthedocs.io/en/latest/examples/learning_rl/index.html)), and also differentiable simulators like `PlasticineLab` and `DiffCloth`. Here is a list of robot tasks trained by `RofuncRL`:

> For more details, please check [**the documentation for RofuncRL**](https://rofunc.readthedocs.io/en/latest/examples/learning_rl/index.html).

| Tasks                          | Animation                                                  | Performance | [ModelZoo](https://github.com/Skylark0924/Rofunc/blob/main/rofunc/config/learning/model_zoo.json) |
| ------------------------------ |------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------|
| Ant                            | ![](doc/img/AntRofuncRLPPO.gif)                            |             | ✅                                                                                                 |
| Cartpole                       |                                                            |             |                                                                                                   |
| FrankaCabinet                  | ![](doc/img/FrankaCabinetRofuncRLPPO.gif)                  |             | ✅                                                                                                 |
| FrankaCubeStack                |                                                            |             |                                                                                                   |
| CURICabinet                    | ![](doc/img/CURICabinet.gif)                               |             | ✅                                                                                                 |
| CURICabinetImage               | ![](doc/img/CURICabinet.gif)                               |             |                                                                                                   |
| CURICabinetBimanual            |                                                            |             |                                                                                                   |
| Humanoid                       | ![](doc/img/HumanoidRofuncRLPPO.gif)                       |             | ✅                                                                                                 |
| HumanoidAMP_backflip           | ![](doc/img/RofuncAMP_HumanoidFlip.gif)                    |             | ✅                                                                                                 |
| HumanoidAMP_walk               |                                                            |             | ✅                                                                                                 |
| HumanoidAMP_run                | ![](doc/img/RofuncAMP_HumanoidRun.gif)                     |             | ✅                                                                                                 |
| HumanoidAMP_dance              | ![](doc/img/RofuncAMP_HumanoidDance.gif)                   |             | ✅                                                                                                 |
| HumanoidAMP_hop                | ![](doc/img/RofuncAMP_HumanoidHop.gif)                     |             | ✅                                                                                                 |
| HumanoidASEGetupSwordShield    | ![](doc/img/ASE3.gif)                                      |             | ✅                                                                                                 |
| HumanoidASEPerturbSwordShield  | ![](doc/img/ASE1.gif)                                      |             | ✅                                                                                                 |
| HumanoidASEHeadingSwordShield  | ![](doc/img/ASE5.gif)                                      |             | ✅                                                                                                 |
| HumanoidASELocationSwordShield | ![](doc/img/HumanoidASELocationSwordShieldRofuncRLPPO.gif) |             | ✅                                                                                                 |
| HumanoidASEReachSwordShield    |                                                            |             | ✅                                                                                                 |
| HumanoidASEStrikeSwordShield   | ![](doc/img/ASE4.gif)                                      |             | ✅                                                                                                 |
| BiShadowHandBlockStack         | ![](doc/img/BiShadowHandBlockStackRofuncRLPPO.gif)                                                            |            | ✅                                                                                                 |
| BiShadowHandBottleCap          | ![](doc/img/BiShadowHandBottleCapRofuncRLPPO.gif)          |             | ✅                                                                                                 |
| BiShadowHandCatchAbreast       | ![](doc/img/BiShadowHandCatchAbreastRofuncRLPPO.gif)       |             | ✅                                                                                                 |
| BiShadowHandCatchOver2Underarm | ![](doc/img/BiShadowHandCatchOver2UnderarmRofuncRLPPO.gif) |             | ✅                                                                                                 |
| BiShadowHandCatchUnderarm      | ![](doc/img/BiShadowHandCatchUnderarmRofuncRLPPO.gif)      |             | ✅                                                                                                 |
| BiShadowHandDoorOpenInward     |                                                            |             | ✅                                                                                                 |
| BiShadowHandDoorOpenOutward    | ![](doc/img/BiShadowHandDoorOpenOutwardRofuncRLPPO.gif)    |             | ✅                                                                                                 |
| BiShadowHandDoorCloseInward    | ![](doc/img/BiShadowHandDoorCloseInwardRofuncRLPPO.gif)    |            | ✅                                                                                                 |
| BiShadowHandDoorCloseOutward   |                                                            |             |                                                                                                   |
| BiShadowHandGraspAndPlace      | ![](doc/img/BiShadowHandGraspAndPlaceRofuncRLPPO.gif)      |             | ✅                                                                                                 |
| BiShadowHandLiftUnderarm       | ![](doc/img/BiShadowHandLiftUnderarmRofuncRLPPO.gif)                                                           |             | ✅                                                                                                  |
| BiShadowHandOver               | ![](doc/img/BiShadowHandOverRofuncRLPPO.gif)               |             | ✅                                                                                                 |
| BiShadowHandPen                | ![](doc/img/BiShadowHandPenRofuncRLPPO.gif)                |             | ✅                                                                                                 |
| BiShadowHandPointCloud         |                                                            |             |                                                                                                   |
| BiShadowHandPushBlock          | ![](doc/img/BiShadowHandPushBlockRofuncRLPPO.gif)          |            | ✅                                                                                                 |
| BiShadowHandReOrientation      | ![](doc/img/BiShadowHandReOrientationRofuncRLPPO.gif)      |            | ✅                                                                                                 |
| BiShadowHandScissors           | ![](doc/img/BiShadowHandScissorsRofuncRLPPO.gif)           |             | ✅                                                                                                 |
| BiShadowHandSwingCup           | ![](doc/img/BiShadowHandSwingCupRofuncRLPPO.gif)           |             | ✅                                                                                                 |
| BiShadowHandSwitch             | ![](doc/img/BiShadowHandSwitchRofuncRLPPO.gif)             |             | ✅                                                                                                 |
| BiShadowHandTwoCatchUnderarm   |                                                            |             |                                                                                                   |



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skylark0924/Rofunc&type=Date)](https://star-history.com/#Skylark0924/Rofunc&Date)

## Citation

If you use rofunc in a scientific publication, we would appreciate citations to the following paper:

```
@software{liu2023rofunc,
          title={Rofunc: The full process python package for robot learning from demonstration and robot manipulation},
          author={Liu, Junjia and Li, Chenzui and Delehelle, Donatien and Li, Zhihao and Chen, Fei},
          month=jun,
          year= 2023,
          publisher={Zenodo},
          doi={10.5281/zenodo.8084510},
          url={https://doi.org/10.5281/zenodo.8084510}
}
```

## Related Papers

1. Robot cooking with stir-fry: Bimanual non-prehensile manipulation of semi-fluid
   objects ([IEEE RA-L 2022](https://arxiv.org/abs/2205.05960) | [Code](rofunc/learning/RofuncIL/structured_transformer/strans.py))

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
   Demonstration ([IEEE CDC 2023](https://arxiv.org/abs/2307.05933) | [Code](./rofunc/learning/ml/tpgmm.py))

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

## Acknowledge

We would like to acknowledge the following projects:

### Learning from Demonstration

1. [pbdlib](https://gitlab.idiap.ch/rli/pbdlib-python)
2. [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
3. [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
4. [SKRL](https://github.com/Toni-SM/skrl)
5. [DexterousHands](https://github.com/PKU-MARL/DexterousHands)

### Planning and Control

1. [Robotics codes from scratch (RCFS)](https://gitlab.idiap.ch/rli/robotics-codes-from-scratch)
