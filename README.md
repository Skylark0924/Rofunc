# Rofunc: The Full Process Python Package for Robot Learning from Demonstration

## Rofunc

Rofunc package focuses on the **robotic Imitation Learning (IL) and Learning from Demonstration (LfD)** fields and provides useful and 
convenient python functions for robotics, including _demonstration collection, data pre-processing, LfD algorithms, planning
and control methods_. We also plan to provide Isaac Gym based robot simulator for evaluation. The purpose of this package is
attempting to advance the field by building a full-process toolkit and validation platform that simplifies and standardizes 
the process of demonstration data collection, processing, and learning.

### Installation
The installation is very easy,

```
pip install rofunc
```

and as you'll find later, it's easy to use as well!

```python
import rofunc as rf
```

Thus, have fun in the robotics world!

### [Documentation](./rofunc/)
Currently, we provide a simple document; please refer to [here](./rofunc/). A comprehensive one with both English and 
Chinese versions is built via the [readthedoc](https://rofunc.readthedocs.io/en/stable/). 
The available functions and plans can be found as follows.


| Classes                         | Types        | Functions               | Description                                                          | Status |
|---------------------------------|--------------|-------------------------|----------------------------------------------------------------------|--------|
| **Devices**                     | Xsens        | `xsens.record`          | Record the human motion via network streaming                        |        |
|                                 |              | `xsens.process`         | Decode the .mvnx file                                                | ✅      |
|                                 |              | `xsens.visualize`       | Show or save gif about the motion                                    | ✅      |
|                                 | Optitrack    | `optitrack.record`      | Record the motion of markers via network streaming                   |        |
|                                 |              | `optitrack.process`     | Process the output .csv data                                         | ✅      |
|                                 |              | `optitrack.visualize`   | Show or save gif about the motion                                    |        |
|                                 | ZED          | `zed.record`            | Record with multiple cameras                                         | ✅      |
|                                 |              | `zed.playback`          | Playback the recording and save snapshots                            | ✅      |
|                                 |              | `zed.export`            | Export the recording to mp4                                          | ✅      |
|                                 | Multi-modal  | `mmodal.record`         | Record multi-modal demonstration data simultaneously                 |        |
|                                 |              | `mmodal.export`         | Export multi-modal demonstration data in one line                    | ✅      |
| **Learning from Demonstration** | DMP          | `dmp.uni`               | DMP for one agent with several (or one) demonstrated trajectories    |        |
|                                 | GMR          | `gmr.uni`               | GMR for one agent with several (or one) demonstrated trajectories    | ✅      |
|                                 | TP-GMM       | `tpgmm.uni`             | TP-GMM for one agent with several (or one) demonstrated trajectories | ✅      |
|                                 |              | `tpgmm.bi`              | TP-GMM for two agent with coordination learned from demonstration    | ✅      |
|                                 | TP-GMR       | `tpgmr.uni`             | TP-GMR for one agent with several (or one) demonstrated trajectories | ✅      |
|                                 |              | `tpgmr.bi`              | TP-GMR for two agent with coordination learned from demonstration    | ✅      |
| **Planning**                    | LQT          | `lqt.uni`               | LQT for one agent with several via-points                            | ✅      |
|                                 |              | `lqt.bi`                | LQT for two agent with coordination constraints                      | ✅      |
|                                 |              | `lqt.recursive`         | Generate smooth trajectories for robot execution recursively         | ✅      |
| **Logger**                      |              | `logger.write`          | Custom tensorboard-based logger                                      |        |
| **Coordinate**                  |              | `coord.custom_class`    | Define the custom class of `Pose`                                    |        |
|                                 |              | `coord.transform`       | Useful functions about coordinate transformation                     | ✅      |
| **VisuaLab**                    | Trajectory   | `visualab.trajectory`   | 2-dim/3-dim/with ori trajectory visualization                        | ✅      |
|                                 | Distribution | `visualab.distribution` | 2-dim/3-dim distribution visualization                               | ✅      |
|                                 | Ellipsoid    | `visualab.ellipsoid`    | 2-dim/3-dim ellipsoid visualization                                  | ✅      |

## Roadmap

Roadmap is a personal learning experience and also simple guidance about robotics and Learning from Demonstration (LfD) fields.

