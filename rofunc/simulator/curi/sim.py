import os.path

from isaacgym import gymapi
from isaacgym import gymutil

from rofunc.simulator.base.base_sim import init_sim, init_env, init_attractor


def update_robot(traj, gym, envs, attractor_handles, axes_geom, sphere_geom, viewer, num_envs, index, t):
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


def show(args, asset_root=None):
    """

    Args:
        args:
        asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets

    Returns:

    """
    print('\033[1;32m--------{}--------\033[0m'.format('Show the CURI simulator in the interactive mode'))

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args)

    # Load CURI asset and set the env
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
    init_env(gym, sim, asset_root, asset_file, num_envs=5, spacing=3.0, fix_base_link=False)

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


def run_traj_bi(args, traj_l, traj_r, attracted_joints=None, asset_root=None, update_freq=0.001, for_test=False):
    """

    Args:
        args:
        traj_l:
        traj_r:
        attracted_joints: [list], e.g. ["panda_left_hand", "panda_right_hand"]
        asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets
        update_freq:
        for_test:

    Returns:

    """
    print('\033[1;32m--------{}--------\033[0m'.format('Execute bimanual trajectory with the CURI simulator'))

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args, for_test=for_test)

    # Load CURI asset and set the env
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
    envs, curi_handles = init_env(gym, sim, asset_root, asset_file, num_envs=1, fix_base_link=False)

    # Create the attractor
    if attracted_joints is None:
        attracted_joint_l = "panda_left_hand"
        attracted_joint_r = "panda_right_hand"
    else:
        assert isinstance(attracted_joints, list)
        attracted_joint_l = attracted_joints[0]
        attracted_joint_r = attracted_joints[1]
    attractor_handles_l, axes_geom_l, sphere_geom_l = init_attractor(gym, envs, viewer, curi_handles, attracted_joint_l,
                                                                     for_test=for_test)
    attractor_handles_r, axes_geom_r, sphere_geom_r = init_attractor(gym, envs, viewer, curi_handles, attracted_joint_r,
                                                                     for_test=for_test)

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
            update_robot(traj_l, gym, envs, attractor_handles_l, axes_geom_l, sphere_geom_l, viewer, len(envs), index,
                         t)
            update_robot(traj_r, gym, envs, attractor_handles_r, axes_geom_r, sphere_geom_r, viewer, len(envs), index,
                         t)
            next_curi_update_time += update_freq
            index += 1
            if index >= len(traj_l):
                index = 0

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        if not for_test:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    import numpy as np
    from importlib_resources import files
    import rofunc as rf
    from isaacgym import gymutil

    args = gymutil.parse_arguments()

    traj_l = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))
    traj_r = np.load(files('rofunc.data').joinpath('taichi_1r.npy'))
    rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)
    run_traj_bi(args, traj_l, traj_r, update_freq=0.001)
