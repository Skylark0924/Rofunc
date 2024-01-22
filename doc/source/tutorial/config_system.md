# Configuration system

We adopt `hydra` as our configuration system. It is a powerful tool to manage AI projects, which use `yaml` files to define configurations. You can find more details in [hydra](https://hydra.cc/docs/intro/).

## Configuration for RofuncRL

The pre-defined configurations for `RofuncRL` can be found in [`rofunc/config/learning/rl`](https://github.com/Skylark0924/Rofunc/tree/main/rofunc/config/learning/rl). Configurations for learning each task contains three files:

- `task/TaskName.yaml`: the task-related configurations for the task `TaskName` (e.g. DoughRolling)
- `train/TaskNameAgent.yaml`: the agent-related configurations for training. (e.g. DoughRollingPPORofuncRL)
- `config`: the template for generating configurations for both task and training algorithm.

Therefore, in each `main.py` script, you need to specify the task and the agent you want to train. For example, if you want to train the `DoughRolling` task with `PPO` algorithm, you need to specify `task=DoughRolling` and `train=DoughRollingPPORofuncRL` in the command line arguments. Then, the configuration system will automatically load the corresponding configurations for the task and the agent.

```python
import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import task_map
from rofunc.learning.RofuncRL.trainers import trainer_map
from rofunc.learning.pre_trained_models.download import model_zoo
from rofunc.learning.utils.utils import set_seed


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "sim_device={}".format(custom_args.sim_device),
                      "rl_device={}".format(custom_args.rl_device),
                      "graphics_device_id={}".format(custom_args.graphics_device_id),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    env = task_map[custom_args.task](...)

    # Instantiate the RL trainer
    trainer = trainer_map[custom_args.agent](...)
 
    # Start training
    trainer.train()

if __name__ == '__main__':
    gpu_id = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant")
    parser.add_argument("--agent", type=str, default="ppo")  # Available agents: ppo, a2c, sac, td3
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    train(custom_args)
```

:::{tip}
More examples can be found in [Example/RofuncRL](https://rofunc.readthedocs.io/en/latest/examples/learning_rl/index.html).
:::

## Customize configurations

You can customize the configurations for your own task and agent by directly pass the absolute path of the configuration file to [`get_config`](https://rofunc.readthedocs.io/en/latest/apidocs/rofunc/rofunc.config.utils.html) function. For example, if you want to customize the configurations for `DoughRolling` task, you need to

1. create a `DoughRolling.yaml` file in `[path]/task`
2. customize the configurations for `PPO` algorithm by creating a `DoughRollingPPO.yaml` file in `[path]/train`
3. remember to copy the `config` file to your own `[path]`

```python
import argparse

from rofunc.config.utils import get_config
from rofunc.learning.RofuncRL.trainers import trainer_map
from myenv import MyEnv


def train(custom_args):
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}".format(custom_args.task, custom_args.agent)]
    cfg = get_config(absl_config_path=[path], config_name='config', args=args_overrides)

    env = MyEnv(cfg.task)
    trainer = trainer_map[custom_args.agent](cfg=cfg, env=env, 
                                             device=custom_args.device, env_name=custom_args.task)

    # Start training
    trainer.train()


if __name__ == '__main__':
    gpu_id = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DoughRolling")
    parser.add_argument("--agent", type=str, default="PPO")
    parser.add_argument("--device", type=str, default=f"cuda:{gpu_id}")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)
```