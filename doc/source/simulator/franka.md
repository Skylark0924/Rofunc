# Franka simulator

### Interactive mode 
```python
import rofunc as rf
from isaacgym import gymutil

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

frankasim = rf.sim.FrankaSim(args)
frankasim.show()
```

![](../../img/simulator_gif/franka_interative.gif)

### Run the trajectory in the Cartesian space

```python
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj = np.load(os.path.join(rf.oslab.get_rofunc_path(), 'data/taichi_1l.npy'))
rf.lqt.plot_3d_uni(traj, ori=False)

frankasim = rf.sim.FrankaSim(args)
frankasim.run_traj(traj)
```

![](../../img/simulator_gif/FrankaTaichi.gif)