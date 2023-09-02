# DataLab

## Parameterized human model from Xsens

For this part, we can extract parameterized URDF human model from Xsens based on a template URDF model.

1. The template URDF model can be found in [template urdf model](https://github.com/Skylark0924/Rofunc/blob/main/rofunc/simulator/assets/urdf/human/human_xsenstemplate_48dof.urdf). In this urdf, we decouple the ball joints of human into 3 rotation joints in the order of Z-X-Y. 
2. By following [this example](https://github.com/Skylark0924/Rofunc/blob/main/examples/simulator/example_xsens_to_human_model.py), we can extract the parameters that the template model needs from Xsens mvnx file and generate a new URDF model.
3. I also have tried to use the joint value in Xsens mvnx file to control this urdf. The trail can be found in [this script](https://github.com/Skylark0924/Rofunc/blob/main/rofunc/simulator/human_sim.py). However, the result is not good, the motion is quite weird. I analyzed that the reason should be the joint values in Xsens mvnx file rotate based on a different axis with our urdf. In order to leverage the experience in AMP and ASE, I finally chose to use the motion retargeting method provided by them.

## Motion retargeting from Xsens

The motion retargeting is not based on URDF model now, it is built on [poselib](https://github.com/Skylark0924/Rofunc/tree/main/rofunc/utils/datalab/poselib) and [MJCF humanoid model](https://github.com/Skylark0924/Rofunc/blob/main/rofunc/simulator/assets/mjcf/amp_humanoid.xml).

1. The poselib can convert the Xsens `.fbx` file to a `.npy` file, please check [this](https://github.com/Skylark0924/Rofunc/blob/main/rofunc/utils/datalab/poselib/fbx_to_amp_npy.py).
2. The `.npy` motion can be visualized in Isaac Gym by following command
   ```bash
   python examples/learning_rl/example_HumanoidASE_RofuncRL.py --task HumanoidViewMotion --motion_file /path/to/motion.npy --headless=False --inference
   ```
   Some example data is provided in [`examples/data/ase`](https://github.com/Skylark0924/Rofunc/tree/main/examples/data/ase) and [`examples/data/amp`](https://github.com/Skylark0924/Rofunc/tree/main/examples/data/amp).