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
    - [ZED Explorer](#zed-explorer)

##  Installation

### System Requirements
The script can be run on:

Ubuntu 22.04, 20.04 and 18.04.
We provide a one-step setup tool to help you simple the installation process of Zed camera's SDK.

### Run
Under the main source folder 
```shell
cd scripts
bash zed_setup.sh
```

### Show Helps
```shell
cd scripts
bash zed_setup.sh -h
```
## Usage
### ZED Explorer
The ZED Explorer is an application for ZED live preview and recording. It lets you change video resolution, aspect ratio, camera parameters, and capture high resolution snapshots and 3D video.

If the ZED is recognized by your computer, you’ll see the 3D video from your camera.

You just need to type the below command in your terminal under any path.
```shell
ZED_Explorer
```