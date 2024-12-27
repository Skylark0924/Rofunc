# Change log

> - 🚀 for RofuncRL updates
> - 🚄 for planning and control updates
> - 🖼️ for Visualab updates
> - 📈 for Datalab updates
> - 🦾 for Robolab updates
> - 📚 for documentation updates
> - 🎮 for simulator updates
> - 🐛 for bug fixes
> - 🏄‍♂️ for system updates
> - 🎉 for event celebrations

## Update News 🎉🎉🎉

- [2024-12-24] 🎮 Start trying to support [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator.

### v0.0.2.6 Support dexterous grasping and human-humanoid robot skill transfer
- [2024-12-20] 🎉🚀 **Human-level skill transfer from human to heterogeneous humanoid robots have been completed and are awaiting release.**
- [2024-01-24] 🚀 [CURI Synergy-based Softhand grasping tasks](https://github.com/Skylark0924/Rofunc/blob/main/examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py) are supported to be trained by `RofuncRL`.
- [2023-12-24] 🚀 [Dexterous hand (Shadow Hand, Allegro Hand, qbSofthand) tasks](https://github.com/Skylark0924/Rofunc/blob/main/examples/learning_rl/IsaacGym_RofuncRL/example_DexterousHands_RofuncRL.py) are supported to be trained by `RofuncRL`.
- [2023-12-07] 🖼️ [EfficientSAM](https://yformer.github.io/efficient-sam/) is supported for high-speed segmentation on edge devices like Nvidia Jetson, check the [example](https://github.com/Skylark0924/Rofunc/blob/main/examples/visualab/example_efficient_sam_seg_w_prompt.py) in Visualab.
- [2023-12-04] 🖼️ [VLPart-SAM](https://github.com/Cheems-Seminar/grounded-segment-any-parts) is supported for part-level segmentation with text prompt, check the [example](https://github.com/Skylark0924/Rofunc/blob/main/examples/visualab/example_vlpart_sam_seg_w_prompt.py).
- [2023-12-03] 🖼️ [Segment-Anything (SAM)](https://segment-anything.com/) is supported in an interactive mode, check the examples in Visualab ([segment anything](https://github.com/Skylark0924/Rofunc/blob/main/examples/visualab/example_sam_seg.py), [segment with prompt](https://github.com/Skylark0924/Rofunc/blob/main/examples/visualab/example_sam_seg_w_prompt.py)).
- **[2023-10-31] 🚀 [`RofuncRL`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/index.html): A modular easy-to-use Reinforcement Learning sub-package designed for Robot Learning tasks is released. It has been tested with simulators like `OpenAIGym`, `IsaacGym`, `OmniIsaacGym` (see [example gallery](https://rofunc.readthedocs.io/en/latest/examples/learning_rl/index.html)), and also differentiable simulators like `PlasticineLab` and `DiffCloth`.**
- [2023-10-31] 🚀 [`RofuncRL`](https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/index.html) supports [OmniIsaacGym](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs) tasks.

### v0.0.2.5 More examples
- [2023-10-16] 🎉 More [examples](https://rofunc.readthedocs.io/en/latest/examples/learning_rl/index.html) are added for [RofuncRL](ttps://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/index.html) and `planning and control` modules.
- **[2023-07-12] 🎉 BiRP paper is accepted by IEEE CDC 2023. Check the arxiv version [here](https://arxiv.org/abs/2307.05933).**
- **[2023-06-24] 🎉 SoftGPT paper is accepted by IROS2023. Check the arxiv version [here](https://arxiv.org/abs/2306.12677).**

### v0.0.2
- [2023-06-13] 🎉 This is the second official release version of `Rofunc` package. We provide a reinforcement learning baseline framework (`RofuncRL`) that performs well in robot tasks 🦾, and several state-of-the-art online reinforcement learning algorithms (PPO, SAC, and TD3) have been completed.🥳
- [2023-05-21] 🚄 LQT (Linear Quadratic Tracking), iLQR (iterative Linear Quadratic Regulator) and their variants are supported in `planning and control` module. Check the [example gallery](https://rofunc.readthedocs.io/en/latest/examples/index.html#planning-and-control-methods).
  
### v0.0.1
- [2022-12-17] 🎉 This is the first offlical release version `Rofunc` package. The four core parts (Multimodal sensors, RL baselines, Control and Isaac simulator) are initially supported. 🎉 

  - **Devices**: support Xsens, optitrack, Zed 2i (multi), Delsys EMG, Manus gloves and multimodal
  - **Learning algorithms**: TP-GMM, TP-GMR, RL baselines (RLlib, ElegantRL, SKRL with IsaacGym)
  - **Planning and control**: LQT variants, iLQR variants
  - **Simulator (IsaacGym-based)**: CURI, Ubtech walker, Franka, Baxter, Sawyer, CURI-mini
 