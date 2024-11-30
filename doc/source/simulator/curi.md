# CURI simulator

### Interactive mode 
```python
from isaacgym import gymutil
import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

CURIsim = rf.sim.CURISim(args)
CURIsim.show(visual_obs_flag=False)
```

![](../../img/simulator_gif/curi_interactive.gif)

### Run the bimanual trajectory in the Cartesian space

```python
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj_l = np.load(os.path.join(rf.oslab.get_rofunc_path(), 'data/taichi_1l.npy'))
traj_r = np.load(os.path.join(rf.oslab.get_rofunc_path(), 'data/taichi_1r.npy'))
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

CURIsim = rf.sim.CURISim(args)
CURIsim.run_traj(traj=[traj_l, traj_r], update_freq=0.001)
```

![](../../img/simulator_gif/CURITaichiFlat.gif)

