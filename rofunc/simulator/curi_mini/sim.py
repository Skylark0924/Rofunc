import os


def update_robot(traj, gym, envs, attractor_handles, axes_geom, sphere_geom, viewer, num_envs, index, t):
    from isaacgym import gymutil

    for i in range(num_envs):
        # Update attractor target from current franka state
        attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        pose = attractor_properties.target
        # pose.p: (x, y, z), pose.r: (w, x, y, z)
        pose.p.x = traj[index, 0]
        pose.p.y = traj[index, 2]
        pose.p.z = traj[index, 1]
        pose.r.w = traj[index, 6]
        pose.r.x = traj[index, 3]
        pose.r.y = traj[index, 5]
        pose.r.z = traj[index, 4]
        gym.set_attractor_target(envs[i], attractor_handles[i], pose)

        # Draw axes and sphere at attractor location
        gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)


def setup_curi(args, asset_root, num_envs, for_test):
    from isaacgym import gymapi
    from rofunc.simulator.base.base_sim import init_sim, init_env, get_num_bodies

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args, for_test=for_test)

    # Load CURI asset and set the env
    if asset_root is None:
        from rofunc.utils.file import get_rofunc_path
        asset_root = os.path.join(get_rofunc_path(), "simulator/assets")
    asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
    envs, curi_handles = init_env(gym, sim, asset_root, asset_file, num_envs=num_envs, fix_base_link=False)

    num_bodies = get_num_bodies(gym, sim, asset_root, asset_file)

    # get joint limits and ranges for CURI
    curi_dof_props = gym.get_actor_dof_properties(envs[0], curi_handles[0])
    curi_lower_limits = curi_dof_props['lower']
    curi_upper_limits = curi_dof_props['upper']
    curi_mids = 0.5 * (curi_upper_limits + curi_lower_limits)
    curi_num_dofs = len(curi_dof_props)

    for i in range(len(envs)):
        # Set updated stiffness and damping properties
        gym.set_actor_dof_properties(envs[i], curi_handles[i], curi_dof_props)

        # Set ranka pose so that each joint is in the middle of its actuation range
        curi_dof_states = gym.get_actor_dof_states(envs[i], curi_handles[i], gymapi.STATE_NONE)
        for j in range(curi_num_dofs):
            curi_dof_states['pos'][j] = curi_mids[j]
        gym.set_actor_dof_states(envs[i], curi_handles[i], curi_dof_states, gymapi.STATE_POS)
    return gym, sim_params, sim, viewer, envs, curi_handles, num_bodies


def setup_attractor(gym, envs, viewer, curi_handles, traj, attracted_joints, for_test):
    from rofunc.simulator.base.base_sim import init_attractor

    if attracted_joints is None:
        attracted_joints = ["panda_left_hand", "panda_right_hand"]
    else:
        assert isinstance(attracted_joints, list) and len(attracted_joints) > 0, "The attracted joints should be a list"
    assert len(attracted_joints) == len(traj), "The number of trajectories should be the same as the number of joints"

    attractor_handles, axes_geoms, sphere_geoms = [], [], []
    for i in range(len(attracted_joints)):
        attractor_handle, axes_geom, sphere_geom = init_attractor(gym, envs, viewer, curi_handles, attracted_joints[i],
                                                                  for_test=for_test)
        attractor_handles.append(attractor_handle)
        axes_geoms.append(axes_geom)
        sphere_geoms.append(sphere_geom)
    return attracted_joints, attractor_handles, axes_geoms, sphere_geoms


def show(args, asset_root=None):
    """

    Args:
        args:
        asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets

    Returns:

    """

    from rofunc.simulator.base.base_sim import init_sim, init_env
    from rofunc.utils.logger.beauty_logger import beauty_print

    beauty_print("Show the CURI mini simulator in the interactive mode", 1)

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args)

    # Load CURI asset and set the env
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/curi_mini/urdf/diablo_simulation.urdf"
    init_env(gym, sim, asset_root, asset_file, num_envs=5, spacing=3.0, fix_base_link=False,
             flip_visual_attachments=False)

    while not gym.query_viewer_has_closed(viewer):
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)


def run_traj(args, traj, attracted_joint="panda_right_hand", asset_root=None, update_freq=0.001):
    """

    Args:
        args:
        traj:
        attracted_joint:
        asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets
        update_freq:

    Returns:

    """
    from isaacgym import gymapi
    from rofunc.simulator.base.base_sim import init_sim, init_env, init_attractor

    print('\033[1;32m--------{}--------\033[0m'.format('Execute trajectory with the CURI simulator'))

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args)

    # Load CURI asset and set the env
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
    envs, curi_handles = init_env(gym, sim, asset_root, asset_file, num_envs=1, fix_base_link=False)

    # Create the attractor
    attractor_handles, axes_geom, sphere_geom = init_attractor(gym, envs, viewer, curi_handles, attracted_joint)

    # get joint limits and ranges for Franka
    curi_dof_props = gym.get_actor_dof_properties(envs[0], curi_handles[0])
    curi_lower_limits = curi_dof_props['lower']
    curi_upper_limits = curi_dof_props['upper']
    curi_mids = 0.5 * (curi_upper_limits + curi_lower_limits)
    curi_num_dofs = len(curi_dof_props)

    for i in range(len(envs)):
        # Set updated stiffness and damping properties
        gym.set_actor_dof_properties(envs[i], curi_handles[i], curi_dof_props)

        # Set ranka pose so that each joint is in the middle of its actuation range
        curi_dof_states = gym.get_actor_dof_states(envs[i], curi_handles[i], gymapi.STATE_NONE)
        for j in range(curi_num_dofs):
            curi_dof_states['pos'][j] = curi_mids[j]
        gym.set_actor_dof_states(envs[i], curi_handles[i], curi_dof_states, gymapi.STATE_POS)

    # Time to wait in seconds before moving robot
    next_curi_update_time = 1

    index = 0
    while not gym.query_viewer_has_closed(viewer):
        # Every 0.01 seconds the pose of the attractor is updated
        t = gym.get_sim_time(sim)
        if t >= next_curi_update_time:
            gym.clear_lines(viewer)
            update_robot(traj, gym, envs, attractor_handles, axes_geom, sphere_geom, viewer, len(envs), index, t)
            next_curi_update_time += update_freq
            index += 1
            if index >= len(traj):
                index = 0

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
