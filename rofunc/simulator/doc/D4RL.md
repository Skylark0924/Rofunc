# D4RL

- [D4RL](#d4rl)
  - [Installation](#installation)
    - [官方安装](#官方安装)
    - [补充](#补充)
  - [Usage](#usage)

## Installation

### 官方安装

```
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```

另外一种简单的安装方法

```
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

### 补充
1. 无法找到 `mujoco210`
   ```
   mkdir .mujoco
   # Mujoco官网下载 mujoco210 folder，并放进上述 folder 中
2. 配置`.bashrc`
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   ```
3. 安装指定版本 `gym`
   ```
   pip install gym==0.18.0
   ```
4. 安装其他依赖包
   ```
   pip install flow
   pip install carla
   pip install agents
   sudo apt-get -y install patchelf
   sudo apt install libosmesa6-dev
   ```
5. 降低 `tensorflow` 版本至 1.x
   ```
   pip install tensorflow-gpu==1.14.0
   ```
6. 安装 GLEW 库
   ```
   sudo apt-get install libglew-dev glew-utils
   ```

### 测试
编写 `test.py` 脚本
```python
import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
```
运行测试
```
python test.py
```
结果
```
load datafile: 100%|██████████████████████████████| 8/8 [00:00<00:00, 32.36it/s]
[[ 1.0856489   1.9745734   0.00981035  0.02174424]
 [ 1.0843927   1.97413    -0.12562364 -0.04433781]
 [ 1.0807577   1.9752754  -0.3634883   0.11453988]
 ...
 [ 1.1328583   2.8062387  -4.484303    0.09555068]
 [ 1.0883482   2.8068895  -4.4510083   0.06509537]
 [ 1.0463258   2.8074222  -4.202244    0.05324839]]
load datafile: 100%|██████████████████████████████| 8/8 [00:00<00:00, 33.65it/s]
```
> 注：如果遇到类似 Unable to open file (truncated file: eof = 26322430, sblock->base_addr = 0, stored_eof = 34720209) 的报错，原因则是在自动下载 dataset 的时候因为网络等问题爬虫断开。需要先删除位于 `\home` 下的 `.d4rl/dataset` 文件夹中的受损数据集，然后再重新运行测试进行下载。

## Usage