# CURI simulator

### Interactive mode 
```python
import rofunc as rf
from isaacgym import gymutil

args = gymutil.parse_arguments()
rf.curi.show(args)
```

![curi_interactive](../img/curi_interactive.gif)

### Run the bimanual trajectory in the Cartesian space

```python
import numpy as np
import rofunc as rf
from importlib_resources import files
from isaacgym import gymutil

args = gymutil.parse_arguments()
traj_l = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))  # [traj_len, 7]
traj_r = np.load(files('rofunc.data').joinpath('taichi_1r.npy'))  # [traj_len, 7]

rf.curi.run_traj_bi(args, traj_l, traj_r, update_freq=0.001)
```

![](../img/FormatFactoryPart1.gif)

