# Isaac Gym 安装与使用

- [Isaac Gym 安装与使用](#isaac-gym-安装与使用)
  - [Installation](#installation)
    - [官网安装](#官网安装)
- [补充](#补充)


## Installation
### 官网安装
Github库：https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

1. 官网下载 https://developer.nvidia.com/isaac-gym/download
2. 解压 `IsaacGym_Preview_3_Package`
3. 打开 `IsaacGym_Preview_3_Package/isaacgym/docs/index.html`，即可找到官方安装指南
4. 安装
   - **创建新conda的方式**
        在 `IsaacGym_Preview_3_Package/isaacgym` 文件夹下
        ```
        ./create_conda_env_rlgpu.sh
        conda activate rlgpu
        ```
    - **在已有的环境下**
        在 `IsaacGym_Preview_3_Package/isaacgym/python` 文件夹下
        ```
        pip install -e .
        ```
5. 安装 RL examples
    ```
    git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
    pip install -e .
    ```
6. 测试
   ```
   cd IsaacGym_Preview_3_Package/isaacgym/python/examples
   python joint_monkey.py
   ```

# 补充
1. libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
   ```
   sudo apt-get install libpython3.7
   ```