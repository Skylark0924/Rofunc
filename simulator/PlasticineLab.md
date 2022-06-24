# PlasticineLab 安装与使用

- [PlasticineLab 安装与使用](#plasticinelab-安装与使用)
  - [Installation](#installation)
    - [官方安装](#官方安装)
    - [补充](#补充)
  - [Usage](#usage)
    - [Bugs](#bugs)

Github库：https://github.com/hzaskywalker/PlasticineLab


## Installation 
### 官方安装

```
git clone https://github.com/hzaskywalker/PlasticineLab.git
python3 -m pip install -e .
```

### 补充
1. ERROR: Failed building wheel for mpi4py
   
   ```
   sudo apt-get install libopenmpi-dev
   ```
   
1. No module named 'baselines.common.vec_env.shmem_vec_env'
   
   ```
   pip install git+https://github.com/openai/baselines@ea25b9e8
   ```
   
1.  module 'taichi' has no attribute 'complex_kernel'

      ```
      pip install taichi==0.7.26  # 实测0.8.5及以上会出现问题
      ```

1. CUDA Error CUDA_ERROR_ASSERT: device-side assert triggered while calling mem_free (cuMemFree_v2)
   
   ```
   gedit ~/.bashrc
   
   # 在文件中添加以下内容
   export TI_DEVICE_MEMORY_FRACTION=0.9
   export TI_DEVICE_MEMORY_GB=4
   ```
   
   ---
   
1. 无法找到 `mujoco210`
   
   ```
   mkdir .mujoco # 在home路徑下下創建
   # Mujoco官网下载 mujoco210 folder，并放进上述 folder 中
   ```
   
2. 配置`.bashrc`
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   ```
   
3. 安装 GLEW 库
   ```
   sudo apt-get install libglew-dev glew-utils
   ```


## Usage

```
python3 -m plb.algorithms.solve --algo action --env_name Move-v1 --path output
```
- `plb.algorithms.solve`: 
- `--algo`: `RL_ALGOS = ['sac', 'td3', 'ppo']` 以及 `DIFF_ALGOS = ['action', 'nn']`；
- `--env_name`: 自带环境可以在`plb/envs/assets`中找到，有`Move`, `Torus`, `Rope`, `Writer`, `Pinch`, `Rollingpin`, `Chopsticks`, `Table`, `TripleMove`, `Assembly`等，每一个都需要在后面指定`v1`~`v5`。例如，`Rollingpin-v3`, `TripleMove-v4`；
- `--path`: PlasticineLab的结果是以逐帧图片形式输出的，所以这里是一个会自动创建的输出文件夹名。



### Bugs

1. taichi.lang.exception.TaichiSyntaxError: cannot assign scalar expr to taichi class <class 'taichi.lang.matrix.Matrix'>, maybe you want to use `a.fill(b)` instead?

   降低 taichi 版本

   ```
   pip install taichi==0.7.26
   ```

   > Refs:
   >
   > https://blog.csdn.net/weixin_43940314/article/details/121725979
   >
   > https://forum.taichi.graphics/t/do-we-had-any-method-put-a-color-with-single-float32-format/1202/6

   