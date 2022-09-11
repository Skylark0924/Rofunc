# Robotics codes from scratch (RCFS)

RCFS is a collection of source codes to study robotics through simple 2D examples. Most examples are coded in Python and Matlab/Octave (full compatibility with GNU Octave). Some are also coded in C++ and Julia. The code examples have .m, .py, .cpp and .jl extensions that can be found in their respective folders matlab, python, cpp and julia. 


### List of examples

| Filename | Description | .m | .py | .cpp | .jl |
|----------|-------------|----|-----|------|-----|
| MP | Movement primitives with various basis functions | ✅ | ✅ |  |  |
| IK | Inverse kinematics for a planar manipulator | ✅ | ✅ | ✅ | ✅ |
| IK_num | Inverse kinematics with numerical computation for a planar manipulator | ✅  | ✅  |  |  |
| FD | Forward Dynamics computed in matrix form for a planar manipulator | ✅ | ✅ |  |  |
| FD_recursive | Forward Dynamics with recursive computation for a planar manipulator | ✅ | ✅ |  |  |
| LQT | Linear quadratic tracking (LQT) applied to a viapoint task (batch formulation) | ✅ | ✅ |  |  |
| LQT_tennisServe | LQT in a ballistic task mimicking a bimanual tennis serve problem (batch formulation) | ✅ |  |  |  |
| LQT_recursive | LQT applied to a viapoint task with a recursive formulation based on augmented state space to find a controller) | ✅ | ✅ |  |  |
| LQT_recursive_LS | LQT applied to a viapoint task with a recursive formulation based on least squares and an augmented state space to find a controller | ✅ |  |  |  |
| LQT_recursive_LS_multiAgents | LQT applied to a multi-agent system with recursive formulation based on least squares and augmented state, by using a precision matrix with nonzero offdiagonal elements to find a controller in which the two agents coordinate their movements to find an optimal meeting point | ✅ |  |  |  |
| LQT_CP | LQT with control primitives applied to a viapoint task (batch formulation) | ✅ | ✅ |  |  |
| LQT_CP_DMP | LQT with control primitives applied to trajectory tracking with a formulation similar to dynamical movement primitives (DMP), by using the least squares formulation of recursive LQR on an augmented state space | ✅ |  |  |  |
| iLQR_obstacle | Iterative linear quadratic regulator (iLQR) applied to a viapoint task with obstacles avoidance (batch formulation) | ✅ | ✅ |  |  |
| iLQR_manipulator | iLQR applied to a planar manipulator for a viapoints task (batch formulation) | ✅ | ✅ | ✅ |  |
| iLQR_manipulator_recursive | iLQR applied to a planar manipulator for a viapoints task (recursive formulation to find a controller) | ✅ | ✅ |  |  |
| iLQR_manipulator_CoM | iLQR applied to a planar manipulator for a tracking problem involving the center of mass (CoM) and the end-effector (batch formulation) | ✅ |  |  |  |
| iLQR_manipulator_obstacle | iLQR applied to a planar manipulator for a viapoints task with obstacles avoidance (batch formulation) | ✅ |  |  |  |
| iLQR_manipulator_CP | iLQR with control primitives applied to a viapoint task with a manipulator (batch formulation) | ✅ | ✅ |  |  |
| iLQR_manipulator_object_affordance | iLQR applied to an object affordance planning problem with a planar manipulator, by considering object boundaries (batch formulation) | ✅ | ✅ |  |  |
| iLQR_manipulator_dynamics | iLQR applied to a reaching task by considering the dynamics of the manipulator | ✅ | ✅ |  |  |
| iLQR_bimanual | iLQR applied to a planar bimanual robot for a tracking problem involving the center of mass (CoM) and the end-effector (batch formulation) | ✅ |  |  |  |
| iLQR_bimanual_manipulability | iLQR applied to a planar bimanual robot problem with a cost on tracking a desired manipulability ellipsoid at the center of mass (batch formulation) | ✅ |  |  |  |
| iLQR_bicopter | iLQR applied to a bicopter problem (batch formulation) | ✅ | ✅ | ✅ |  |
| iLQR_car | iLQR applied to a car parking problem (batch formulation) | ✅ | ✅ | ✅ |  |

Check also the [PDF with the corresponding mathematical descriptions](doc/rcfs.pdf).

Additional reading material can be be found as [video lectures](https://tube.switch.ch/channels/e5e11e14?order=oldest-first) with [corresponding slides in PDF format](https://drive.switch.ch/index.php/s/PAn7xsA5CNQKzyo).


### Work in progress

| Filename | Main responsible |
|----------|------------------|
| Example similar to demo_OC_LQT_nullspace_minimal01.m | Hakan |
| Example similar to demo_OC_LQT_Lagrangian01.m (inclusion of constraints) | ??? |


**TODO:**

- fix glut display issues in C++ (ellipsoids, resizing windows)
- add missing equations related to S¹ manifold in RCFS.tex

### License

RCFS is maintained by Sylvain Calinon, https://calinon.ch/.

Contributors: Jérémy Maceiras, Tobias Löw, Amirreza Razmjoo, Julius Jankowski, Boyang Ti, Teng Xue

Copyright (c) 2022 Idiap Research Institute, https://idiap.ch/

RCFS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.

You should have received a copy of the GNU General Public License along with RCFS. If not, see <https://www.gnu.org/licenses/>.
