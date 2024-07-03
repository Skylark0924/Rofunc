<!-- This code is submitted as supplemental materials for paper: "Learning Robot Geometry as Distance Fields: Applications to Whole-body Manipulation" -->

### Code for paper "Learning Robot Geometry as Distance Fields: Applications to Whole-body Manipulation"

[[Paper]](https://arxiv.org/pdf/2307.00533.pdf)[[Project]](https://sites.google.com/view/lrdf)

<img src='robot_sdf.gif'/>

#### Dependencies

- Python version: 3.8 (Tested)
- Pytorch version:1.13.0 (Tested)
- Install necessary packages

```sh
pip install -r requirements.txt
```

- Install [Chamfer Distance](https://github.com/otaheri/chamfer_distance) (optional, only used for evaluating the
  chamfer distance)

#### Usage

##### Run RDF represented with basis functions

```sh
python bf_sdf.py --n_func 24 --device cuda
```

Given points with size (N,3) and joint configurations (B,7), it will output SDF values (B,N) and gradients w.r.t. both
points (analytical, with shape(B,N,3)) and joints (numerical, with shape(B,N,7)).

You can also uncomment the code

``` python
    # # visualize the Bernstein Polynomial model for each robot link
    # bp_sdf.create_surface_mesh(model,nbData=128,vis=True)

    # # visualize the Bernstein Polynomial model for the whole body
    # theta = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(args.device).reshape(-1,7)
    # pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(theta),4,4).float()
    # trans_list = panda.get_transformations_each_link(pose,theta)
    # utils.visualize_reconstructed_whole_body(model, trans_list, tag=f'BP_{args.n_func}')
```

to visualize the reconstructed meshes for each robot link and whole body, respectively.

We provide some pretrained models in ```models```.

##### Visualize the SDF

We provide 2D and 3D visualization for produced SDF and gradient, just run

```sh
python vis.py
```

to see the results.

##### Train RDF model with basis functions

Generate SDF data for each robot link:

```sh
python sample_sdf_points.py 
```

It computes the SDF value based on [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf). The sampled points and sdf
values are saved in ```data/sdf_points```. You can also download the
data [here](https://drive.google.com/file/d/1lsdJzxECFOILhYiCJydOcruKoqB6QiJR/view?usp=sharing). After this, please
put '*.npy' files in ```data/sdf_points/```

Then just run

```sh
python bf_sdf.py --train --n_func 8 --device cuda
```

to learn weights of basis functions. Normally it will take 1~2 minutes to train a model when using 8 basis functions.

##### Evaluation

For instance, you can run

```sh
# method:[BP_8,BP_24,NN_LD,NN_AD,Sphere]
# type: [RDF,LINK] 
python eval.py --device cuda --method BP_8 --type RDF 
```

to evaluate the quality of RDF. BP_8 and BP_24 donate numbers of basis functios we use, while NN_LD and NN_AD donate nn
models trained with limited data and argumented data.

##### Dual arm grasp planning

You can run

```sh
python bbo_planning.py
```

to see how our RDF model can be used for whole arm lifting task with Gauss-Newton algorithm. It will plan 3 valid joint
configurations for both arms.

##### Train RDF for your own robot

- Build a differentiable robot layer for forward kinematics (see ```panda_layer/panda_layer.py``` for details)

- Train RDF model using basis functions (We use .stl file for SDF computation and reconstruction, which can be found in
  the URDF file)

- Use it!

Note: Another option is to use the pytorch kinematics library to parse the urdf file automatically to build RDF for your
own robot: https://github.com/UM-ARM-Lab/pytorch_kinematics

RDF is maintained by Yiming LI and licensed under the MIT License.

Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
