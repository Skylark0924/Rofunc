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

![](../../../img/ASE1.gif)


## Baseline comparison

## Tricks

## Network update function

```{literalinclude} ../../../../rofunc/learning/RofuncRL/agents/mixline/ase_agent.py
:pyobject: ASEAgent.update_net
```