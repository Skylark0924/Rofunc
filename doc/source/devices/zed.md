# Zed

- [Zed](#zed)
  - [Setup](#setup)
    - [System Requirements](#system-requirements)
    - [Run](#run)
    - [Show Helps](#show-helps)
  - [Record](#record)
  - [Playback](#playback)
  - [Export](#export)


In our Rofunc project, we will provide some demos and examples to show what Zed cameras can do with robots.

The website of Zed camera: 
> https://www.stereolabs.com/

## Setup

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

## Record

It is capable to check the camera devices connected to the computer autonomously and record multiple cameras in
parallel.

```python 
import rofunc as rf

root_dir = '/home/ubuntu/Data/zed_record'
exp_name = '20220909'
rf.zed.record(root_dir, exp_name)
```

## Playback

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

## Export

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
    """
```

```python
import rofunc as rf

rf.zed.export('[your_path]/38709363.svo', 2)
```

```python
import rofunc as rf

rf.zed.export_batch('[your_path]/20220624_1649', core_num=20)
```