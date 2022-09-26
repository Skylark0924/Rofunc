# CURI simulator

### Interactive mode 
```python
import rofunc as rf
from isaacgym import gymutil

args = gymutil.parse_arguments()
rf.curi.show(args)
```


### Run the bimanual trajectory in the Cartesian space

```python
import numpy as np
import rofunc as rf
from importlib_resources import files
from isaacgym import gymutil

args = gymutil.parse_arguments(description="CURI Attractor Example")
traj_l = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))  # [traj_len, 7]
traj_r = np.load(files('rofunc.data').joinpath('taichi_1r.npy'))  # [traj_len, 7]

rf.curi.run_traj_bi(args, traj_l, traj_r)
```


