# 使用方法

## 将人体关键点数据连同物体位姿数据转换为npy格式

使用脚本：
https://github.com/clover-cuhk/Rofunc/blob/dzp-fbx-37/rofunc/utils/datalab/poselib/xsens_fbx_to_amp_npy.py

运行该脚本会将Rofunc/data下面的所有fbx文件转换为npy文件，其有两个可选参数，为--start和--end，用于给出与物体交互的起始帧与结束帧序号。

使用时，需要先在不给定以上参数情况下重放一遍，确定起始帧和结束帧，再次运行并给定参数。

建议每次仅在data文件夹中放一个fbx，因为每个fbx对应的start/end是不同的。
