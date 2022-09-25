import math

import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil


def init_sim(args):
    # Initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.gravity.y = -9.80
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
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 75.0
    camera_props.width = 1920
    camera_props.height = 1080
    # camera_props.use_collision_geometry = True
    viewer = gym.create_viewer(sim, camera_props)
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    return gym, sim_params, sim, viewer


def init_env(gym, sim, viewer, asset_root, asset_file, num_envs=1, spacing=1.0, fix_base_link=True,
             cam_pos=gymapi.Vec3(3.0, 2.0, 0.0), cam_target=gymapi.Vec3(0.0, 0.0, 0.0)):
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fix_base_link
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # Set up the env grid
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    envs = []
    handles = []

    # Point camera at environments
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0.0, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add robot
        handle = gym.create_actor(env, asset, pose, "robot", i, 2)
        handles.append(handle)

    dof_props = gym.get_actor_dof_properties(envs[0], handles[0])

    # override default stiffness and damping values
    dof_props['stiffness'].fill(1000.0)
    dof_props['damping'].fill(1000.0)

    # Give a desired pose for first 2 robot joints to improve stability
    dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

    dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][7:] = 1e10
    dof_props['damping'][7:] = 1.0

    for i in range(num_envs):
        gym.set_actor_dof_properties(envs[i], handles[i], dof_props)

    return envs, handles


def init_attractor(gym, envs, viewer, handles, attracted_joint):
    # Attractor setup
    attractor_handles = []
    attractor_properties = gymapi.AttractorProperties()
    attractor_properties.stiffness = 5e5
    attractor_properties.damping = 5e3

    # Make attractor in all axes
    attractor_properties.axes = gymapi.AXIS_ALL

    # Create helper geometry used for visualization
    # Create a wireframe axis
    axes_geom = gymutil.AxesGeometry(0.1)
    # Create a wireframe sphere
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

    for i in range(len(envs)):
        env = envs[i]
        handle = handles[i]

        body_dict = gym.get_actor_rigid_body_dict(env, handle)
        props = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
        attracted_joint_handle = body = gym.find_actor_rigid_body_handle(env, handle, attracted_joint)

        # Initialize the attractor
        attractor_properties.target = props['pose'][:][body_dict[attracted_joint]]
        attractor_properties.target.p.y -= 0.1
        attractor_properties.target.p.z = 0.1
        attractor_properties.rigid_handle = attracted_joint_handle

        # Draw axes and sphere at attractor location
        gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

        attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
        attractor_handles.append(attractor_handle)
    return attractor_handles, axes_geom, sphere_geom