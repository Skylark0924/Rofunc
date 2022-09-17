# Roadmap-for-robot-science

## Rofunc

Rofunc is a package for real-world robotics experiments, including useful functions for devices (Xsens, Optitrack, Zed) and planning.

```python
import rofunc as rf
```

### Installation

```
pip install rofunc
```

### [Documentation](./rofunc/)
Currently, we provide a simple document; please refer to [here](./rofunc/).
## Available functions

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

Roadmap is a personal learning experience and also simple guidance about Robotics.

