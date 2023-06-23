# RofuncRL ASE (Adversarial Skill Embeddings)

## Algorithm 

![](../../../img/ASE2.png)

## Demos

### Pre-trained latent space model

![](../../../img/ASE3.gif)

```shell
python examples/learning_rl/example_HumanoidASE_RofuncRL.py --task HumanoidASEGetupSwordShield --motion_file reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --inference
```

### Pre-trained latent space model with perturbation

You can test the robustness of the latent space model by changing to `HumanoidASEPerturbSwordShield` task (throwing boxes to the humanoid robot). It will use **the same pre-trained latent space model as previous demo**, but set the `reset` function to reset by the maximum length of the episode, rather than resetting immediately when robots fall on the ground.

![](../../../img/ASE1.gif)

```shell
python examples/learning_rl/example_HumanoidASE_RofuncRL.py --task HumanoidASEPerturbSwordShield --motion_file reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --inference
```


## Baseline comparison

## Tricks

## Network update function

```{literalinclude} ../../../../rofunc/learning/RofuncRL/agents/mixline/ase_agent.py
:pyobject: ASEAgent.update_net
```