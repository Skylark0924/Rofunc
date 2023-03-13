# Robotics Functions (RoFunc)


## Hello, robot world!

```
pip install rofunc
```

```python
import rofunc as rf
```

- [Robotics Functions (RoFunc)](#robotics-functions-rofunc)
  - [Hello, robot world!](#hello-robot-world)
  - [Devices](#devices)
    - [Xsens](#xsens)
      - [Convert mvnx file to npys](#convert-mvnx-file-to-npys)
      - [Visualize the motion data](#visualize-the-motion-data)
    - [Optitrack](#optitrack)
      - [Get useful data](#get-useful-data)
      - [Visualize the motion data](#visualize-the-motion-data-1)
    - [Zed](#zed)
      - [Record](#record)
      - [Playback](#playback)
      - [Export](#export)
    - [Multi-modal](#multi-modal)
      - [Export](#export-1)
  - [Planning](#planning)
    - [LQT](#lqt)
      - [Unimanual](#unimanual)
      - [Bimanual](#bimanual)
  - [Learning from Demonstration](#learning-from-demonstration)
    - [TP-GMM](#tp-gmm)
      - [Unimanual](#unimanual-1)
      - [Bimanual](#bimanual-1)



The available functions and plans can be found as follows. 


| Classes                                         | Types                  | Functions                                                    | Description                                                  | Status |
| ----------------------------------------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------ |
| **Demonstration collection and pre-processing** | Xsens                  | `xsens.record`                                               | Record the human motion via network streaming                | ✅      |
|                                                 |                        | `xsens.process`                                              | Decode the `.mvnx` file                                      | ✅      |
|                                                 |                        | `xsens.visualize`                                            | Show or save gif about the motion                            | ✅      |
|                                                 | Optitrack              | `optitrack.record`                                           | Record the motion of markers via network streaming           | ✅      |
|                                                 |                        | `optitrack.process`                                          | Process the output `.csv` data                               | ✅      |
|                                                 |                        | `optitrack.visualize`                                        | Show or save gif about the motion                            |        |
|                                                 | ZED                    | `zed.record`                                                 | Record with multiple (1~n) cameras                           | ✅      |
|                                                 |                        | `zed.playback`                                               | Playback the recording and save snapshots                    | ✅      |
|                                                 |                        | `zed.export`                                                 | Export the recording to mp4 or image sequences               | ✅      |
|                                                 | Delsys EMG             | `emg.record`                                                 | Record real-time EMG data via network streaming              | ✅      |
|                                                 |                        | `emg.process`                                                | Filtering the EMG data                                       | ✅      |
|                                                 |                        | `emg.visualize`                                              | Some visualization functions for EMG data                    | ✅      |
|                                                 | Multimodal             | `mmodal.record`                                              | Record multi-modal demonstration data simultaneously         |        |
|                                                 |                        | `mmodal.export`                                              | Export multi-modal demonstration data in one line            | ✅      |
| **Learning from Demonstration**                 | Machine learning       | `dmp.uni`                                                    | DMP for uni-manual robot with several (or one) demonstrated trajectories |        |
|                                                 |                        | `gmr.uni`                                                    | GMR for uni-manual robot with several (or one) demonstrated trajectories | ✅      |
|                                                 |                        | `gmm.uni`                                                    | GMM for uni-manual robot with several (or one) demonstrated trajectories |        |
|                                                 |                        | `tpgmm.uni`                                                  | TP-GMM for uni-manual robot with several (or one) demonstrated trajectories | ✅      |
|                                                 |                        | `tpgmm.bi`                                                   | TP-GMM for bimanual robot with coordination learned from demonstration | ✅      |
|                                                 |                        | `tpgmr.uni`                                                  | TP-GMR for uni-manual robot with several (or one) demonstrated trajectories | ✅      |
|                                                 |                        | `tpgmr.bi`                                                   | TP-GMR for bimanual robot with coordination learned from demonstration | ✅      |
|                                                 | Deep learning          | `bco`                                                        | Behavior cloning from observation                            | ✅      |
|                                                 |                        | `strans`                                                     | Structured-Transformer method proposed in [IEEE RAL](https://arxiv.org/abs/2205.05960) |        |
|                                                 | Reinforcement learning | SKRL (`ppo`, `sac`, `tq3`)                                   | Provide API for SKRL framework and robot examples in Isaac Gym | ✅      |
|                                                 |                        | StableBaseline3 (`ppo`, `sac`, `tq3`)                        | Provide API for StableBaseline3 framework and robot examples in Isaac Gym |        |
|                                                 |                        | RLlib (`ppo`, `sac`, `tq3`)                                  | Provide API for Ray RLlib framework and robot examples in Isaac Gym | ✅      |
|                                                 |                        | ElegantRL (`ppo`, `sac`, `tq3`)                              | Provide API for ElegantRL framework and robot examples in Isaac Gym | ✅      |
|                                                 |                        | `cql`                                                        | Conservative Q-learning for fully offline learning           |        |
| **Planning**                                    | LQT                    | [`lqt.uni`](https://rofunc.readthedocs.io/en/latest/planning/lqt.html) | Linear Quadratic Tracking (LQT) for uni-manual robot with several via-points | ✅      |
|                                                 |                        | `lqt.bi`                                                     | LQT for bimanual robot with coordination constraints         | ✅      |
|                                                 |                        | [`lqt.uni_fb`](https://rofunc.readthedocs.io/en/latest/planning/lqt_fb.html) | Generate smooth trajectories with feedback                   | ✅      |
|                                                 |                        | [`lqt.uni_cp`](https://rofunc.readthedocs.io/en/latest/planning/lqt_cp.html) | LQT with control primitive                                   | ✅      |
|                                                 | iLQR                   | [`ilqr.uni`](https://rofunc.readthedocs.io/en/latest/planning/ilqr.html) | Iterative Linear Quadratic Regulator (iLQR) for uni-manual robot with several via-points | ✅      |
|                                                 |                        | `ilqr.bi`                                                    | iLQR for bimanual robots with several via-points             | ✅      |
|                                                 |                        | `ilqr.uni_fb`                                                | iLQR with feedback                                           |        |
|                                                 |                        | `ilqr.uni_cp`                                                | iLQR with control primitive                                  | ✅      |
|                                                 |                        | `ilqr.uni_obstacle`                                          | iLQR with obstacle avoidance                                 | ✅      |
|                                                 |                        | `ilqr.uni_dyna`                                              | iLQR with dynamics and force control                         | ✅      |
|                                                 | MPC                    | `mpc.uni`                                                    | Model Predictive Control (MPC)                               |        |
|                                                 | CIO                    | `cio.whole`                                                  | Contact-invariant Optimization (CIO)                         |        |
| **Tools**                                       | Logger                 | `logger.write`                                               | General logger based on `tensorboard`                        |        |
|                                                 | Config                 | `config.get_config`                                          | General config API based on `hydra`                          | ✅      |
|                                                 | VisuaLab               | `visualab.trajectory`                                        | 2-dim/3-dim/with ori trajectory visualization                | ✅      |
|                                                 |                        | `visualab.distribution`                                      | 2-dim/3-dim distribution visualization                       | ✅      |
|                                                 |                        | `visualab.ellipsoid`                                         | 2-dim/3-dim ellipsoid visualization                          | ✅      |
|                                                 | RoboLab                | `robolab.transform`                                          | Useful functions about coordinate transformation             | ✅      |
|                                                 |                        | `robolab.fk`                                                 | Forward kinematics w.r.t URDF file                           | ✅      |
|                                                 |                        | `robolab.ik`                                                 | Inverse kinematics w.r.t URDF file                           | ✅      |
|                                                 |                        | `robolab.fd`                                                 | Forward dynamics w.r.t URDF file                             |        |
|                                                 |                        | `robolab.id`                                                 | Inverse dynamics w.r.t URDF file                             |        |
| **Simulator**                                   | Franka                 | [`franka.sim`](https://rofunc.readthedocs.io/en/latest/simulator/franka.html) | Execute specific trajectory via single Franka Panda arm in Isaac Gym | ✅      |
|                                                 | CURI mini              | `curi_mini.sim`                                              | Execute specific trajectory via dual Franka Panda arm in Isaac Gym |        |
|                                                 | CURI                   | [`curi.sim`](https://rofunc.readthedocs.io/en/latest/simulator/curi.html) | Execute specific trajectory via human-like CURI robot in Isaac Gym | ✅      |
|                                                 | Walker                 | `walker.sim`                                                 | Execute specific trajectory via UBTECH Walker robot  in Isaac Gym | ✅      |

## Devices

### Xsens

#### Convert mvnx file to npys

`get_skeleton(mvnx_path, output_dir=None)`

```python
import rofunc as rf

mvnx_file = '/home/ubuntu/Data/06_24/Xsens/dough_01.mvnx'
rf.xsens.get_skeleton(mvnx_file) 
```

Then you will get a folder with multiple .npy files, each one refers to one segment.

> We also provide a batch form for converting mvnx files in parallel.

```python
import rofunc as rf

mvnx_dir = '../xsens_data'
rf.xsens.get_skeleton_batch(mvnx_dir)
```

#### Visualize the motion data

After obtains data of each segment, we can get a whole-body visualization
by `plot_skeleton(skeleton_data_path: str, save_gif=False)`

```python
import rofunc as rf

# dough_01 must be a folder with multiple .npy files about the skeleton which can be generated by `get_skeleton`
skeleton_data_path = './xsens_data/dough_01'
rf.xsens.plot_skeleton(skeleton_data_path)
```

![](../img/dough_01.gif)

> We also provide a batch form for saving gifs of several skeleton data in parallel.

```python
import rofunc as rf

# There must contain a folder with multiple .npy files about the skeleton which can be generated by `get_skeleton`
skeleton_dir = './xsens_data/'
rf.xsens.plot_skeleton_batch(skeleton_dir)
```

### Optitrack

#### Get useful data

You need first prepare your raw data in the following structure.

```
├── dough_01
│   ├── Take 2022-06-24 07.40.52 PM.csv
│   ├── Take 2022-06-24 07.40.52 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv (if applicable)
│   └── Take 2022-06-24 07.40.52 PM_ManusVRGlove_7b28f20b_7b28f20b.csv (if applicable)
├── dough_02
│   ├── Take 2022-06-24 07.44.15 PM.csv
│   ├── Take 2022-06-24 07.44.15 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv (if applicable)
│   └── Take 2022-06-24 07.44.15 PM_ManusVRGlove_7b28f20b_7b28f20b.csv (if applicable)
├── dough_03
...
```

You can get the useful data by `data_clean(input_path)`

```python
import rofunc as rf

root_path = '/home/ubuntu/Github/DGform/data/opti_data/dough_01'
rf.optitrack.data_clean(root_path)
```

Then you will obtain new csv files in the same directory.

```
├── dough_01
│   ├── left_manus.csv
│   ├── opti_hands.csv
│   ├── process
│   │   ├── Take 2022-06-24 07.40.52 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv
│   │   └── Take 2022-06-24 07.40.52 PM_ManusVRGlove_7b28f20b_7b28f20b.csv
│   ├── right_manus.csv
│   ├── Take 2022-06-24 07.40.52 PM.csv
│   ├── Take 2022-06-24 07.40.52 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv
│   └── Take 2022-06-24 07.40.52 PM_ManusVRGlove_7b28f20b_7b28f20b.csv
├── dough_02
│   ├── left_manus.csv
│   ├── opti_hands.csv
│   ├── process
│   │   ├── Take 2022-06-24 07.44.15 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv
│   │   └── Take 2022-06-24 07.44.15 PM_ManusVRGlove_7b28f20b_7b28f20b.csv
│   ├── right_manus.csv
│   ├── Take 2022-06-24 07.44.15 PM.csv
│   ├── Take 2022-06-24 07.44.15 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv
│   └── Take 2022-06-24 07.44.15 PM_ManusVRGlove_7b28f20b_7b28f20b.csv
├── dough_03
...
```

> We also provide a batch form cleaning several data in parallel.

```python 
import rofunc as rf

input_dir = '/home/ubuntu/Github/DGform/data/opti_data/'
rf.optitrack.process.data_clean_batch(input_dir)
```

#### Visualize the motion data

### Zed

#### Record

It is capable to check the camera devices connected to the computer autonomously and record multiple cameras in
parallel.

```python 
import rofunc as rf

root_dir = '/home/ubuntu/Data/zed_record'
exp_name = '20220909'
rf.zed.record(root_dir, exp_name)
```

#### Playback

It is capable to check the camera devices connected to the computer autonomously and record multiple cameras in
parallel.

```python 
import rofunc as rf

recording_path = '/home/ubuntu/Data/06_24/Video/20220624_1649/38709363.svo'
rf.zed.playback(recording_path)
```

You can save the snapshots as prompted

```
Reading SVO file: /home/ubuntu/Data/06_24/Video/20220624_1649/38709363.svo
  Save the current image:     s
  Quit the video reading:     q

Saving image 0.png : SUCCESS
Saving image 1.png : SUCCESS
Saving image 2.png : SUCCESS
Saving image 3.png : SUCCESS
...
```

#### Export

```python
def export(filepath, mode=1):
    """
    Export the svo file with specific mode.
    Args:
        filepath: SVO file path (input) : path/to/file.svo
        mode: Export mode:  0=Export LEFT+RIGHT AVI.
                            1=Export LEFT+DEPTH_VIEW AVI.
                            2=Export LEFT+RIGHT image sequence.
                            3=Export LEFT+DEPTH_VIEW image sequence.
                            4=Export LEFT+DEPTH_16Bit image sequence.

    Returns:

    """
```

```python
import rofunc as rf

rf.zed.export('/home/ubuntu/Data/06_24/Video/20220624_1649/38709363.svo', 2)
```

```python
import rofunc as rf

rf.zed.export_batch('/home/ubuntu/Data/06_24/Video/20220624_1649', core_num=20)
```

### Multi-modal

#### Export

The raw data folder should have these following contents:

```
.
├── optitrack_csv (necessary)
│   ├── exp_01
│   │   ├── Take 2022-09-09 06.32.26 PM.csv
│   │   ├── Take 2022-09-09 06.32.26 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv
│   │   └── Take 2022-09-09 06.32.26 PM_ManusVRGlove_7b28f20b_7b28f20b.csv
│   └── exp_02
│       ├── Take 2022-09-09 06.34.11 PM.csv
│       ├── Take 2022-09-09 06.34.11 PM_ManusVRGlove_3f6ec26f_3f6ec26f.csv
│       └── Take 2022-09-09 06.34.11 PM_ManusVRGlove_7b28f20b_7b28f20b.csv
├── xsens_mvnx (necessary)
│   ├── exp_01.mvnx
│   ├── ...
└── zed (necessary)
    ├── HD1080_SN38709363_18-24-20.svo
    ├── ...
 
```


```python
import rofunc as rf

rf.mmodal.export('/home/ubuntu/Data/2022_09_09_Taichi')
```

## Planning

### LQT

#### Unimanual

```python
import rofunc as rf
import numpy as np

param = {
    "nbData": 500,   # Number of data points
    "nbVarPos": 7,   # Dimension of position data
    "nbDeriv": 2,    # Number of static and dynamic features (2 -> [x,dx])
    "dt": 1e-2,      # Time step duration
    "rfactor": 1e-8  # Control cost
}
cfg.nbVar = cfg.nbVarPos * cfg.nbDeriv  # Dimension of state vector

data = np.load('data/z_manipulator_poses.npy')
filter_indices = [0, 1, 5, 10, 22, 36]
data = data[filter_indices]

u_hat, x_hat, muQ, idx_slices = rf.lqt.uni(param, data)
rf.lqt.plot_3d_uni(x_hat, muQ, idx_slices, ori=False, save=False)
```

![](../img/lqt_uni.png)

#### Bimanual

> Currently, there is no coordination between the generated bimanual trajectories.

```python
import rofunc as rf
import numpy as np

param = {
    "nbData": 500,  # Number of data points
    "nbVarPos": 7,  # Dimension of position data
    "nbDeriv": 2,  # Number of static and dynamic features (2 -> [x,dx])
    "dt": 1e-2,  # Time step duration
    "rfactor": 1e-8  # Control cost
}
cfg.nbVar = cfg.nbVarPos * cfg.nbDeriv  # Dimension of state vector

data = np.loadtxt('data/link7_loc_ori.txt', delimiter=', ')
l_data = data[0:len(data):2]
r_data = data[1:len(data):2]
u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices = rf.lqt.bi(param, l_data, r_data)
rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices, ori=False, save=False)
```

![](../img/lqt_bi.png)
