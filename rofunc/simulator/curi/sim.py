"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


curi Attractor
----------------
Positional control of curi panda robot with a target attractor that the robot tries to reach
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil


def create_attractor():
    # Attractor setup
    attractor_properties = gymapi.AttractorProperties()
    attractor_properties.stiffness = 5e5
    attractor_properties.damping = 5e3

    # Make attractor in all axes
    attractor_properties.axes = gymapi.AXIS_ALL
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0., 0.0, 0.)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # Create helper geometry used for visualization
    # Create a wireframe axis
    axes_geom = gymutil.AxesGeometry(0.1)
    # Create a wireframe sphere
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

    return attractor_properties, axes_geom, sphere_geom, pose


def set_curi(gym, envs, curi_handles):
    # get joint limits and ranges for curi
    curi_dof_props = gym.get_actor_dof_properties(envs[0], curi_handles[0])
    curi_lower_limits = curi_dof_props['lower']
    curi_upper_limits = curi_dof_props['upper']
    curi_ranges = curi_upper_limits - curi_lower_limits
    curi_mids = 0.5 * (curi_upper_limits + curi_lower_limits)
    curi_num_dofs = len(curi_dof_props)

    # override default stiffness and damping values
    curi_dof_props['stiffness'].fill(1000.0)
    curi_dof_props['damping'].fill(1000.0)

    # Give a desired pose for first 2 robot joints to improve stability
    curi_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

    curi_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
    curi_dof_props['stiffness'][7:] = 1e10
    curi_dof_props['damping'][7:] = 1.0

    for i in range(len(envs)):
        gym.set_actor_dof_properties(envs[i], curi_handles[i], curi_dof_props)

    return [curi_handles, curi_dof_props, curi_num_dofs, curi_mids]


def update_curi(gym, envs, viewer, curi_infos, attractor_infos, t):
    curi_handles, curi_dof_props, curi_num_dofs, curi_mids = curi_infos
    attractor_handles, axes_geom, sphere_geom = attractor_infos

    gym.clear_lines(viewer)
    num_envs = len(envs)
    for i in range(num_envs):
        # Update attractor target from current curi state
        attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        pose = attractor_properties.target
        pose.p.y = -0.2 + 0.2 * math.sin(1.5 * t - math.pi * float(i) / num_envs) + 1
        # pose.p.y = 0.7 + 0.1 * math.cos(2.5 * t - math.pi * float(i) / num_envs)
        pose.p.x = 0.7 + 0.2 * math.cos(1.5 * t - math.pi * float(i) / num_envs) + 0.1
        pose.p.z = 0.5

        gym.set_attractor_target(envs[i], attractor_handles[i], pose)

        # Draw axes and sphere at attractor location
        gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)

    for i in range(num_envs):
        # Set updated stiffness and damping properties
        gym.set_actor_dof_properties(envs[i], curi_handles[i], curi_dof_props)

        # Set ranka pose so that each joint is in the middle of its actuation range
        curi_dof_states = gym.get_actor_dof_states(envs[i], curi_handles[i], gymapi.STATE_NONE)
        for j in range(curi_num_dofs):
            curi_dof_states['pos'][j] = curi_mids[j]
        gym.set_actor_dof_states(envs[i], curi_handles[i], curi_dof_states, gymapi.STATE_POS)


def create_sim(gym, args):
    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
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

    # Point camera at environments
    cam_pos = gymapi.Vec3(3.0, 2.0, 0.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # Load curi asset
    asset_root = "../assets"
    curi_asset_file = "urdf/curi/urdf/curi_pinocchio.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    print("Loading asset '%s' from '%s'" % (curi_asset_file, asset_root))
    curi_asset = gym.load_asset(sim, asset_root, curi_asset_file, asset_options)

    # Set up the env grid
    num_envs = 1
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Some common handles for later use
    envs = []
    curi_handles = []
    curi_hand = "panda_right_hand"

    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))

    attractor_properties, axes_geom, sphere_geom, pose = create_attractor()
    attractor_handles = []
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add curi
        curi_handle = gym.create_actor(env, curi_asset, pose, "curi", i, 2)
        body_dict = gym.get_actor_rigid_body_dict(env, curi_handle)
        props = gym.get_actor_rigid_body_states(env, curi_handle, gymapi.STATE_POS)
        hand_handle = body = gym.find_actor_rigid_body_handle(env, curi_handle, curi_hand)

        # Initialize the attractor
        attractor_properties.target = props['pose'][:][body_dict[curi_hand]]
        attractor_properties.target.p.y -= 0.1
        attractor_properties.target.p.z = 0.1
        attractor_properties.rigid_handle = hand_handle

        # Draw axes and sphere at attractor location
        gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

        curi_handles.append(curi_handle)
        attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
        attractor_handles.append(attractor_handle)

    curi_infos = set_curi(gym, envs, curi_handles)
    attractor_infos = [attractor_handles, axes_geom, sphere_geom]

    return sim, envs, viewer, curi_infos, attractor_infos


def main():
    # Initialize gym
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(description="curi Attractor Example")

    # Create the sim
    sim, envs, viewer, curi_infos, attractor_infos = create_sim(gym, args)

    # Time to wait in seconds before moving robot
    next_curi_update_time = 3

    while not gym.query_viewer_has_closed(viewer):
        # Every 0.01 seconds the pose of the attactor is updated
        t = gym.get_sim_time(sim)
        if t >= next_curi_update_time:
            update_curi(gym, envs, viewer, curi_infos, attractor_infos, t)
            next_curi_update_time += 0.01

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


if __name__ == '__main__':
    main()
