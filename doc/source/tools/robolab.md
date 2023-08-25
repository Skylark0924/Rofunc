# RoboLab

RoboLab is a subpackage of the `rofunc` package. It contains useful functions for robotics.

## Coordinate transformations and rigid body motions

We consider the common coordinate and motion representation in robotics, and implement the functions to convert between them.

For rotations, we consider the following representations:

- Euler angle: `euler`
- Rotation matrix: `rot_matrix`
- Quaternion: `quaternion`

For translations, we consider the following representations:

- Cartesian coordinates: `translation`

For rigid body motions, we consider the following representations:

- Homogeneous transformation matrix: `homo_matrix`

### Convert between rotation representations

To convert from one rotation representation `[src_repr]` to another `[dst_repr]`, use the following functions:

```python
import rofunc as rf

[src_repr]_R = ...
[dst_repr]_R = rf.robolab.[dst_repr]_from_[src_repr]([src_repr]_R)
```

Detailed usage examples for each functions can be found in the [rofunc.utils.robolab.coord.transform](#rofunc.utils.robolab.coord.transform).

### Convert from rotation and translation to homogeneous transformation

To convert from rotation `[src_repr]_R` and translation `[translation]_p` to homogeneous transformation `T`, use the following functions:

```python
import rofunc as rf

[src_repr]_R = ...
[translation]_p = ...
T = rf.robolab.homo_matrix_from_[src_repr]([src_repr]_R, [translation]_p)
```

## Robot kinematics

## Robot dynamics
