# RLBaseLine (SKRL)

## Demos

### Gym tasks

The arguments in `example_GymTasks_SKRL.py`:

```python
gym_task_name = 'Pendulum-v1'
# Available tasks:
# Classic: ['Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1']
# Box2D: ['BipedalWalker-v3', 'CarRacing-v1', 'LunarLander-v2']  `pip install gymnasium[box2d]`
# MuJoCo: ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedDoublePendulum-v2',
#          'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']  `pip install -U mujoco-py`

parser.add_argument("--task", type=str, default="Gym_{}".format(gym_task_name))  # Start with 'Gym_'
parser.add_argument("--agent", type=str, default="ppo")  # Available agents: ppo, sac, td3, a2c
parser.add_argument("--render_mode", type=str, default=None)  # Available render_mode: None, "human", "rgb_array"
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default=None)
```

Train `gym` or `gymnasium` tasks with `SKRL` by the following command:

```shell
python examples/learning_rl/example_GymTasks_SKRL.py --task Gym_[gym_task_name]
```

### Ant

The arguments in `example_Ant_SKRL.py`:

```python
gpu_id = 0
parser.add_argument("--task", type=str, default="Ant")
parser.add_argument("--agent", type=str, default="td3")  # Available agents: ppo, sac, td3
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default=None)
```

Train the `IsaacGym Ant` task with `SKRL` by the following command:

```shell
python examples/learning_rl/example_Ant_SKRL.py
```

### CURICabinet

The arguments in `example_CURICabinet_SKRL.py`:

```python
gpu_id = 0
parser.add_argument("--task", type=str, default="CURICabinet")
parser.add_argument("--agent", type=str, default="ppo")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default=None)
```

Train the `IsaacGym CURICabinet` task with `SKRL` by the following command:

```shell
python examples/learning_rl/example_CURICabinet_SKRL.py
```

### FrankaCabinet

The arguments in `example_FrankaCabinet_SKRL.py`:

```python
gpu_id = 0
parser.add_argument("--task", type=str, default="FrankaCabinet")
parser.add_argument("--agent", type=str, default="ppo")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default=None)
```

Train the `IsaacGym FrankaCabinet` task with `SKRL` by the following command:

```shell
python examples/learning_rl/example_FrankaCabinet_SKRL.py
```

### Humanoid

The arguments in `example_Humanoid_SKRL.py`:

```python
parser.add_argument("--task", type=str, default="Humanoid")
parser.add_argument("--agent", type=str, default="PPO")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default=None)
```

Train the `IsaacGym Humanoid` task with `SKRL` by the following command:

```shell
python examples/learning_rl/example_Humanoid_SKRL.py
```

### HumanoidAMP

The arguments in `example_HumanoidAMP_SKRL.py`:

```python
gpu_id = 1
parser.add_argument("--task", type=str, default="HumanoidAMP")
parser.add_argument("--agent", type=str, default="AMP")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default=None)
``` 

Train the `IsaacGym HumanoidAMP` task with `SKRL` by the following command:

```shell
python examples/learning_rl/example_HumanoidAMP_SKRL.py
```

