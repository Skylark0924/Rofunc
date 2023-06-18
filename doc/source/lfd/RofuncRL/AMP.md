# RofuncRL AMP (Adversarial Motion Priors)


## Algorithm 

```{literalinclude} ../../../../rofunc/learning/RofuncRL/agents/online/amp_agent.py
:pyobject: AMPAgent.update_net
```

## Performance comparison

We compare the performance of the AMP algorithm with different tricks and an open source baseline 
([SKRL](https://github.com/Toni-SM/skrl/tree/main)). These experiments were conducted on the `Humanoid` environment.
The results are shown below:

### Humanoid Run
![HumanoidAMPRun](../../../img/RofuncAMP_HumanoidRun_perf.png)
- `Pink`: SKRL AMP
- `Green`: Rofunc AMP 

![HumanoidAMPRun Inference](../../../img/RofuncAMP_HumanoidRun.gif)


### Humanoid BackFlip
![HumanoidAMPFlip](../../../img/RofuncAMP_HumanoidFlip_perf.png)
- `Pink`: Rofunc AMP

![HumanoidAMPFlip Inference](../../../img/RofuncAMP_HumanoidFlip.gif)


### Humanoid Dance
![HumanoidAMPFlip Inference](../../../img/RofuncAMP_HumanoidDance.gif)


### Humanoid Hop
![HumanoidAMPFlip Inference](../../../img/RofuncAMP_HumanoidHop.gif)
