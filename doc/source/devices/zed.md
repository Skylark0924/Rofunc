# Zed

In our Rofunc project, we will provide some demos and examples to show what Zed cameras can do with robots.

The website of Zed camera:
> https://www.stereolabs.com/

- [Zed](#zed)
    - [Installation](#installation)
        - [System Requirements](#system-requirements)
        - [Run](#run)
        - [Show Helps](#show-helps)
    - [Usage](#usage)
        - [ZED Explorer and Depth Viewer](#zed-explorer-and-depth-viewer)

## Installation

### System Requirements

The script can be run on:

Ubuntu 22.04, 20.04 and 18.04. We provide a one-step setup tool to help you simple the installation process of Zed
camera's SDK.

### Run

Under the main source folder, the default installation method contains zed's python API installation. Optional
installation is available.

```shell
cd scripts
bash zed_setup.sh
```

### Optional installation

#### help

```shell
cd scripts
bash zed_setup.sh -h
```

#### python api

```shell
cd scripts
bash zed_setup.sh -api
```

#### python dependence

```shell
cd scripts
bash zed_setup.sh -d
```

## Usage

### ZED Explorer and Depth Viewer

The ZED Explorer is an application for ZED live preview and recording. It lets you change video resolution, aspect
ratio, camera parameters, and capture high resolution snapshots and 3D video. The ZED Depth Viewer uses the SDK to
capture and display the depth map and 3D point cloud. Run the ZED Depth Viewer to check that the depth map is displayed
correctly.

You just need to type the below command in your terminal under any path.

```shell
# Open the ZED_Explorer
ZED_Explorer
```

```shell
# Open the ZED_Depth_viewer
ZED_Depth_Viewer
```