"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Terrain examples
-------------------------
Demonstrates the use terrain meshes.
Press 'R' to reset the  simulation
"""

import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymutil, gymapi
from isaacgym.terrain_utils import *
from math import sqrt


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# load ball asset
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# set up the env grid
num_envs = 800
num_per_row = 80
env_spacing = 0.56
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)
pose.p.z = 1.
pose.p.x = 3.

envs = []
# set random seed
np.random.seed(17)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# create all available terrain types
num_terains = 8
terrain_width = 12.
terrain_length = 12.
horizontal_scale = 0.25  # [m]
vertical_scale = 0.005  # [m]
num_rows = int(terrain_width/horizontal_scale)
num_cols = int(terrain_length/horizontal_scale)
heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)


def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)


heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw
heightfield[num_rows:2*num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
heightfield[2*num_rows:3*num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
heightfield[3*num_rows:4*num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.5, min_size=1., max_size=5., num_rects=20).height_field_raw
heightfield[4*num_rows:5*num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2., amplitude=1.).height_field_raw
heightfield[5*num_rows:6*num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
heightfield[6*num_rows:7*num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
heightfield[7*num_rows:8*num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
                                                                stone_distance=1., max_height=0.5, platform_size=0.).height_field_raw

# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-5, -5, 15)
cam_target = gymapi.Vec3(0, 0, 10)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
