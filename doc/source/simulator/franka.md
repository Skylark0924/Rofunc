# Franka simulator

### Interactive mode 
```python
import rofunc as rf
from isaacgym import gymutil

args = gymutil.parse_arguments()
rf.franka.show(args)
```


### Run the trajectory in the Cartesian space

```python
import numpy as np
import rofunc as rf
from importlib_resources import files
from isaacgym import gymutil

args = gymutil.parse_arguments(description="Franka Attractor Example")
traj = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))

rf.franka.run_traj(args, traj)
```