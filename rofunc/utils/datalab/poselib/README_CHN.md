# 使用方法

## 复现颠勺视频

### 使用dzp-fbx-37-new分支

首先使用脚本generate_amp_humanoid_tpose_with_tools，以amp_humanoid_spoon_pan.xml文件全路径为参数，生成AMP需要的amp_humanoid_generated_new_tpose.npy文件。

然后使用generate_samp_humanoid_tpose脚本，以024为参数（即是使用024号fbx文件），生成Xsens需要的tpose文件。

> 024的起始姿态符合标准NPOSE，个人认为这对后续生成正确的动作至关重要，如果手没有贴在裤线两侧并与身体平行，所得效果不理想（参见028动作序列）

最后使用generate_amp_humanoid_tpose_with_tools脚本，以amp_humanoid_spoon_pan.xml文件全路径为参数，生成npy格式的动作序列。

### 使用dzp-dev-w_shoulder分支

直接运行example_HumanoidASE_ViewMotion.py脚本，参数为上一步生成的npy全路径及所用配置文件名称：

```
/home/dzp/Rofunc/data/024_amp.npy --config_name HumanoidSpoonPan
```

## 将人体关键点数据连同物体位姿数据转换为npy格式

使用脚本：
https://github.com/clover-cuhk/Rofunc/blob/dzp-fbx-37/rofunc/utils/datalab/poselib/xsens_fbx_to_amp_npy.py

运行该脚本会将Rofunc/data下面的所有fbx文件转换为npy文件，其有两个可选参数，为--start和--end，用于给出与物体交互的起始帧与结束帧序号。

使用时，需要先在不给定以上参数情况下重放一遍，确定起始帧和结束帧，再次运行并给定参数。

建议每次仅在data文件夹中放一个fbx，因为每个fbx对应的start/end是不同的。
