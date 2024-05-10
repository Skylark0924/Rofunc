# Isaac Sim Installation
\* *Note that this part refers to [Rofunc installation documentation](https://rofunc.readthedocs.io/en/latest/installation.html) and [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)*.

**Isaac Sim** has to be installed firstly by following this [documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html). Note that the **Isaac Sim** version should be **2022.2.1**. Other versions are not supported yet since their default python version is not compatible with **Rofunc**. Once installed, locate the [python executable in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html). By default, this should be `python.sh`. We will refer to this path as `pysim2`.

```
# Alias the Isaac Sim python in .bashrc
gedit ~/.bashrc
# Add the following line to the end of the file
alias pysim2="/home/[user_name]/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh"
# Save and close the file
source ~/.bashrc

git clone https://github.com/Skylark0924/Rofunc.git --branch jysun-dev --single-branch
cd Rofunc

# Install rofunc
pysim2 -m pip install .
```
If you use Pycharm, its interpreter should be also set as `/home/[user_name]/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh`.

# Training
To train the Elfin robot to pick bags from baskets, run:
```
pysim2 examples/learning_rl/OmniIsaacGym_RofuncRL/example_ElfinBagOmni_RofuncRL.py --task ElfinBagBasketOmni
```
The task is defined in `rofunc/learning/RofuncRL/tasks/omniisaacgymenv/elfin_bag.py`. The corresponding task config file is `rofunc/config/learning/rl/task/ElfinBagBasketOmni.yaml`. The training results are saved in `examples/learning_rl/OmniIsaacGym_RofuncRL/runs` and the folder name should start with `RofuncRL_PPOTrainer_ElfinBagBasketOmni`.

Note that by default, we run the script without GUI to avoid slow training. If you need that, just run:
```
pysim2 examples/learning_rl/OmniIsaacGym_RofuncRL/example_ElfinBagOmni_RofuncRL.py --task ElfinBagBasketOmni --headless False
```

Similarly, to train the Elfin robot to pick bags from washers, run:
```
pysim2 examples/learning_rl/OmniIsaacGym_RofuncRL/example_ElfinBagOmni_RofuncRL.py --task ElfinBagWasherOmni
```

# Inference
\* *Note that we use `ElfinBagBasketOmni` task as an example in this section.*

To load a trained checkpoint and perform inference, run:
```
pysim2 examples/learning_rl/OmniIsaacGym_RofuncRL/example_ElfinBagOmni_RofuncRL.py --task ElfinBagBasketOmni --inference --ckpt_path examples/learning_rl/OmniIsaacGym_RofuncRL/runs/[ckpt_name]/checkpoints/best_ckpt.pth
```
To load our pre-trained model, run:
```
pysim2 examples/learning_rl/OmniIsaacGym_RofuncRL/example_ElfinBagOmni_RofuncRL.py --task ElfinBagBasketOmni --inference
```