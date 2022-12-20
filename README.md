![](./img/logo8.png)
# Rofunc: The Full Process Python Package for Robot Learning from Demonstration

[![Release](https://img.shields.io/github/v/release/Skylark0924/Rofunc)](https://pypi.org/project/rofunc/)
[![Documentation Status](https://readthedocs.org/projects/rofunc/badge/?version=latest)](https://rofunc.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/Skylark0924/Rofunc)
![](https://img.shields.io/github/downloads/skylark0924/Rofunc/total)
[![](https://img.shields.io/github/issues-closed-raw/Skylark0924/Rofunc)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aissue+is%3Aclosed)
[![](https://img.shields.io/github/issues-raw/Skylark0924/Rofunc)](https://github.com/Skylark0924/Rofunc/issues?q=is%3Aopen+is%3Aissue)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FSkylark0924%2FRofunc%2Fbadge%3Fref%3Dmain&style=flat)](https://actions-badge.atrox.dev/Skylark0924/Rofunc/goto?ref=main)

> **Repository address: https://github.com/Skylark0924/Rofunc**

Rofunc package focuses on the **robotic Imitation Learning (IL) and Learning from Demonstration (LfD)** fields and provides valuable and convenient python functions for robotics, including _demonstration collection, data pre-processing, LfD algorithms, planning, and control methods_. We also provide an Isaac Gym-based robot simulator for evaluation. This package aims to advance the field by building a full-process toolkit and validation platform that simplifies and standardizes the process of demonstration data collection, processing, learning, and its deployment on robots.

![](./img/pipeline.png)

## Installation
### Install from PyPI
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
> Several requirements need to be installed before using the package. Please refer to the [installation guide](https://rofunc.readthedocs.io/en/latest/overview.html#installation) for more details.

### Install from Source (Recommended)
```python
git clone https://github.com/Skylark0924/Rofunc.git
cd Rofunc

# Create a conda environment
conda create -n rofunc python=3.8
conda activate rofunc

# Install the requirements and rofunc
pip install -r requirements.txt
pip install .
```
> **Note**
> If you want to use functions related to ZED camera, you need to install [ZED SDK](https://www.stereolabs.com/developers/release/#downloads) manually. (We have tried to package it as a `.whl` file to add it to `requirements.txt`, unfortunately, the ZED SDK is not very friendly and doesn't support direct installation.)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Skylark0924/Rofunc&type=Date)](https://star-history.com/#Skylark0924/Rofunc&Date)


## Documentation
[![Documentation](https://img.shields.io/badge/Documentation-Access-brightgreen?style=for-the-badge)](https://rofunc.readthedocs.io/en/latest/)
[![Example Gallery](https://img.shields.io/badge/Example%20Gallery-Access-brightgreen?style=for-the-badge)](https://rofunc.readthedocs.io/en/latest/auto_examples/index.html)

> **Note**
> Currently, we provide a simple document; please refer to [here](./rofunc/).
A comprehensive one with both English and Chinese versions is built via the [readthedoc](https://rofunc.readthedocs.io/en/latest/).
We provide a simple but interesting example: learning to play
Taichi by learning from human demonstration.

To give you a quick overview of the pipeline of `rofunc`, we provide an interesting example of learning to play Taichi from human demonstration. You can find it in the [Quick start](https://rofunc.readthedocs.io/en/latest/quickstart.html) section of the documentation.

The available functions and plans can be found as follows. 


| Classes                                         | Types                  | Functions                                                                     | Description                                                                              | Status |
|-------------------------------------------------|------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|--------|
| **Demonstration collection and pre-processing** | Xsens                  | `xsens.record`                                                                | Record the human motion via network streaming                                            | ✅      |
|                                                 |                        | `xsens.process`                                                               | Decode the `.mvnx` file                                                                  | ✅      |
|                                                 |                        | `xsens.visualize`                                                             | Show or save gif about the motion                                                        | ✅      |
|                                                 | Optitrack              | `optitrack.record`                                                            | Record the motion of markers via network streaming                                       | ✅      |
|                                                 |                        | `optitrack.process`                                                           | Process the output `.csv` data                                                           | ✅      |
|                                                 |                        | `optitrack.visualize`                                                         | Show or save gif about the motion                                                        |        |
|                                                 | ZED                    | `zed.record`                                                                  | Record with multiple (1~n) cameras                                                       | ✅      |
|                                                 |                        | `zed.playback`                                                                | Playback the recording and save snapshots                                                | ✅      |
|                                                 |                        | `zed.export`                                                                  | Export the recording to mp4 or image sequences                                           | ✅      |
|                                                 | Delsys EMG             | `emg.record`                                                                  | Record real-time EMG data via network streaming                                          | ✅      |
|                                                 |                        | `emg.process`                                                                 | Filtering the EMG data                                                                   | ✅      |
|                                                 |                        | `emg.visualize`                                                               | Some visualization functions for EMG data                                                | ✅      |
|                                                 | Multimodal             | `mmodal.record`                                                               | Record multi-modal demonstration data simultaneously                                     |        |
|                                                 |                        | `mmodal.export`                                                               | Export multi-modal demonstration data in one line                                        | ✅      |
| **Learning from Demonstration**                 | Machine learning       | `dmp.uni`                                                                     | DMP for uni-manual robot with several (or one) demonstrated trajectories                 |        |
|                                                 |                        | `gmr.uni`                                                                     | GMR for uni-manual robot with several (or one) demonstrated trajectories                 | ✅      |
|                                                 |                        | `gmm.uni`                                                                     | GMM for uni-manual robot with several (or one) demonstrated trajectories                 |        |
|                                                 |                        | `tpgmm.uni`                                                                   | TP-GMM for uni-manual robot with several (or one) demonstrated trajectories              | ✅      |
|                                                 |                        | `tpgmm.bi`                                                                    | TP-GMM for bimanual robot with coordination learned from demonstration                   | ✅      |
|                                                 |                        | `tpgmr.uni`                                                                   | TP-GMR for uni-manual robot with several (or one) demonstrated trajectories              | ✅      |
|                                                 |                        | `tpgmr.bi`                                                                    | TP-GMR for bimanual robot with coordination learned from demonstration                   | ✅      |
|                                                 | Deep learning          | `bco`                                                                         | Behavior cloning from observation                                                        | ✅      |
|                                                 |                        | `strans`                                                                      | Structured-Transformer method proposed in [IEEE RAL](https://arxiv.org/abs/2205.05960)   |        |
|                                                 | Reinforcement learning | SKRL (`ppo`, `sac`, `tq3`)                                                    | Provide API for SKRL framework and robot examples in Isaac Gym                           | ✅      |
|                                                 |                        | StableBaseline3 (`ppo`, `sac`, `tq3`)                                         | Provide API for StableBaseline3 framework and robot examples in Isaac Gym                |        |
|                                                 |                        | RLlib (`ppo`, `sac`, `tq3`)                                                   | Provide API for Ray RLlib framework and robot examples in Isaac Gym                      | ✅      |
|                                                 |                        | ElegantRL (`ppo`, `sac`, `tq3`)                                               | Provide API for ElegantRL framework and robot examples in Isaac Gym                      | ✅      |
|                                                 |                        | `cql`                                                                         | Conservative Q-learning for fully offline learning                                       |        |
| **Planning**                                    | LQT                    | [`lqt.uni`](https://rofunc.readthedocs.io/en/latest/planning/lqt.html)        | Linear Quadratic Tracking (LQT) for uni-manual robot with several via-points             | ✅      |
|                                                 |                        | `lqt.bi`                                                                      | LQT for bimanual robot with coordination constraints                                     | ✅      |
|                                                 |                        | [`lqt.uni_fb`](https://rofunc.readthedocs.io/en/latest/planning/lqt_fb.html)  | Generate smooth trajectories with feedback                                               | ✅      |
|                                                 |                        | [`lqt.uni_cp`](https://rofunc.readthedocs.io/en/latest/planning/lqt_cp.html)  | LQT with control primitive                                                               | ✅      |
|                                                 | iLQR                   | [`ilqr.uni`](https://rofunc.readthedocs.io/en/latest/planning/ilqr.html)      | Iterative Linear Quadratic Regulator (iLQR) for uni-manual robot with several via-points | ✅      |
|                                                 |                        | `ilqr.bi`                                                                     | iLQR for bimanual robots with several via-points                                         | ✅      |
|                                                 |                        | `ilqr.uni_fb`                                                                 | iLQR with feedback                                                                       |        |
|                                                 |                        | `ilqr.uni_cp`                                                                 | iLQR with control primitive                                                              | ✅      |
|                                                 |                        | `ilqr.uni_obstacle`                                                           | iLQR with obstacle avoidance                                                             | ✅      |
|                                                 |                        | `ilqr.uni_dyna`                                                               | iLQR with dynamics and force control                                                     | ✅      |
|                                                 | MPC                    | `mpc.uni`                                                                     | Model Predictive Control (MPC)                                                           |        |
|                                                 | CIO                    | `cio.whole`                                                                   | Contact-invariant Optimization (CIO)                                                     |        |
| **Tools**                                       | Logger                 | `logger.write`                                                                | General logger based on `tensorboard`                                                    |        |
|                                                 | Config                 | `config.get_config`                                                           | General config API based on `hydra`                                                      | ✅      |
|                                                 | VisuaLab               | `visualab.trajectory`                                                         | 2-dim/3-dim/with ori trajectory visualization                                            | ✅      |
|                                                 |                        | `visualab.distribution`                                                       | 2-dim/3-dim distribution visualization                                                   | ✅      |
|                                                 |                        | `visualab.ellipsoid`                                                          | 2-dim/3-dim ellipsoid visualization                                                      | ✅      |
|                                                 | RoboLab                | `robolab.transform`                                                           | Useful functions about coordinate transformation                                         | ✅      |
|                                                 |                        | `robolab.fk`                                                                  | Forward kinematics w.r.t URDF file                                                       | ✅      |
|                                                 |                        | `robolab.ik`                                                                  | Inverse kinematics w.r.t URDF file                                                       | ✅      |
|                                                 |                        | `robolab.fd`                                                                  | Forward dynamics w.r.t URDF file                                                         |        |
|                                                 |                        | `robolab.id`                                                                  | Inverse dynamics w.r.t URDF file                                                         |        |
| **Simulator**                                   | Franka                 | [`franka.sim`](https://rofunc.readthedocs.io/en/latest/simulator/franka.html) | Execute specific trajectory via single Franka Panda arm in Isaac Gym                     | ✅      |
|                                                 | CURI mini              | `curi_mini.sim`                                                               | Execute specific trajectory via dual Franka Panda arm in Isaac Gym                       |        |
|                                                 | CURI                   | [`curi.sim`](https://rofunc.readthedocs.io/en/latest/simulator/curi.html)     | Execute specific trajectory via human-like CURI robot in Isaac Gym                       | ✅      |
|                                                 | Walker                 | `walker.sim`                                                                  | Execute specific trajectory via UBTECH Walker robot  in Isaac Gym                        | ✅      |


## Cite

If you use rofunc in a scientific publication, we would appreciate citations to the following paper:

```
@misc{Rofunc2022,
      author = {Liu, Junjia and Li, Zhihao and Li, Chenzui},
      title = {Rofunc: The full process python package for robot learning from demonstration},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Skylark0924/Rofunc}},
}
```

## The Team
Rofunc is developed and maintained by the [CLOVER Lab (Collaborative and Versatile Robot Laboratory)](https://feichenlab.com), CUHK.
