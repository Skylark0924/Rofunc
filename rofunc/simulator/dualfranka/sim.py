"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Attractor
----------------
Positional control of franka panda robot with a target attractor that the robot tries to reach
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Franka Attractor Example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Load franka asset
asset_root = "../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset_l = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)
franka_asset_r = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# Set up the env grid
num_envs = 1
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
franka_handles = []
franka_hand = "panda_hand"

# Attractor setup
attractor_handles = []
attractor_properties_l = gymapi.AttractorProperties()
attractor_properties_l.stiffness = 5e5
attractor_properties_l.damping = 5e3

attractor_properties_r = gymapi.AttractorProperties()
attractor_properties_r.stiffness = 5e5
attractor_properties_r.damping = 5e3

# Make attractor in all axes
attractor_properties_l.axes = gymapi.AXIS_ALL
attractor_properties_r.axes = gymapi.AXIS_ALL

pose_l = gymapi.Transform()
pose_l.p = gymapi.Vec3(0, 0.0, 0.4)
pose_l.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

pose_r = gymapi.Transform()
pose_r.p = gymapi.Vec3(0, 0.0, -0.4)
pose_r.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add franka
    franka_actor_l = gym.create_actor(env, franka_asset_l, pose_l, "franka_l", i, 2)
    body_dict_l = gym.get_actor_rigid_body_dict(env, franka_actor_l)
    props_l = gym.get_actor_rigid_body_states(env, franka_actor_l, gymapi.STATE_POS)
    hand_handle_l = body_l = gym.find_actor_rigid_body_handle(env, franka_actor_l, franka_hand)

    franka_actor_r = gym.create_actor(env, franka_asset_r, pose_r, "franka_r", i, 2)
    body_dict_r = gym.get_actor_rigid_body_dict(env, franka_actor_r)
    props_r = gym.get_actor_rigid_body_states(env, franka_actor_r, gymapi.STATE_POS)
    hand_handle_r = body_r = gym.find_actor_rigid_body_handle(env, franka_actor_r, franka_hand)

    # Initialize the attractor
    attractor_properties_l.target = props_l['pose'][:][body_dict_l[franka_hand]]
    attractor_properties_l.target.p.y -= 0.1
    attractor_properties_l.target.p.z = 0.5
    attractor_properties_l.rigid_handle = hand_handle_l

    attractor_properties_r.target = props_r['pose'][:][body_dict_r[franka_hand]]
    attractor_properties_r.target.p.y -= 0.1
    attractor_properties_r.target.p.z = -0.3
    attractor_properties_r.rigid_handle = hand_handle_r

    # Draw axes and sphere at attractor location
    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties_l.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties_l.target)

    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties_r.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties_r.target)

    franka_handles.append([franka_actor_l, franka_actor_r])
    attractor_handle_l = gym.create_rigid_body_attractor(env, attractor_properties_l)
    attractor_handle_r = gym.create_rigid_body_attractor(env, attractor_properties_r)
    attractor_handles.append([attractor_handle_l, attractor_handle_r])

# get joint limits and ranges for Franka
franka_dof_props_l = gym.get_actor_dof_properties(envs[0], franka_handles[0][0])
franka_lower_limits = franka_dof_props_l['lower']
franka_upper_limits = franka_dof_props_l['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props_l)

# override default stiffness and damping values
franka_dof_props_l['stiffness'].fill(1000.0)
franka_dof_props_l['damping'].fill(1000.0)

# Give a desired pose for first 2 robot joints to improve stability
franka_dof_props_l["driveMode"][0:2] = gymapi.DOF_MODE_POS

franka_dof_props_l["driveMode"][7:] = gymapi.DOF_MODE_POS
franka_dof_props_l['stiffness'][7:] = 1e10
franka_dof_props_l['damping'][7:] = 1.0

franka_dof_props_r = gym.get_actor_dof_properties(envs[0], franka_handles[0][1])
franka_lower_limits = franka_dof_props_r['lower']
franka_upper_limits = franka_dof_props_r['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props_r)

# override default stiffness and damping values
franka_dof_props_r['stiffness'].fill(1000.0)
franka_dof_props_r['damping'].fill(1000.0)

# Give a desired pose for first 2 robot joints to improve stability
franka_dof_props_r["driveMode"][0:2] = gymapi.DOF_MODE_POS

franka_dof_props_r["driveMode"][7:] = gymapi.DOF_MODE_POS
franka_dof_props_r['stiffness'][7:] = 1e10
franka_dof_props_r['damping'][7:] = 1.0

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], franka_handles[i][0], franka_dof_props_l)
    gym.set_actor_dof_properties(envs[i], franka_handles[i][1], franka_dof_props_r)


def update_franka(t):
    gym.clear_lines(viewer)
    for i in range(num_envs):
        for j in range(len(attractor_handles[i])):
            # Update attractor target from current franka state
            attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i][j])
            pose = attractor_properties.target
            pose.p.x = 0.2 * math.sin(1.5 * t - math.pi * float(i) / num_envs)
            pose.p.y = 0.7 + 0.1 * math.cos(2.5 * t - math.pi * float(i) / num_envs)
            if j == 0:
                pose.p.z = 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs) + 0.4
            else:
                pose.p.z = 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs) - 0.4

            gym.set_attractor_target(envs[i], attractor_handles[i][j], pose)

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
            gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)


for i in range(num_envs):
    for j in range(len(franka_handles[i])):
        # Set updated stiffness and damping properties
        if j == 0:
            gym.set_actor_dof_properties(envs[i], franka_handles[i][j], franka_dof_props_l)
        else:
            gym.set_actor_dof_properties(envs[i], franka_handles[i][j], franka_dof_props_r)

        # Set ranka pose so that each joint is in the middle of its actuation range
        franka_dof_states = gym.get_actor_dof_states(envs[i], franka_handles[i][j], gymapi.STATE_NONE)
        for k in range(franka_num_dofs):
            franka_dof_states['pos'][k] = franka_mids[k]
        gym.set_actor_dof_states(envs[i], franka_handles[i][j], franka_dof_states, gymapi.STATE_POS)

# Point camera at environments
cam_pos = gymapi.Vec3(-4.0, 4.0, -1.0)
cam_target = gymapi.Vec3(0.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Time to wait in seconds before moving robot
next_franka_update_time = 1.5

while not gym.query_viewer_has_closed(viewer):
    # Every 0.01 seconds the pose of the attactor is updated
    t = gym.get_sim_time(sim)
    if t >= next_franka_update_time:
        update_franka(t)
        next_franka_update_time += 0.01

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
