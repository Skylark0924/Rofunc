import os

from isaacgym import gymapi
from isaacgym import gymutil

from rofunc.simulator.base.base_sim import init_sim, init_env, init_attractor


def update_robot(traj, gym, envs, attractor_handles, axes_geom, sphere_geom, viewer, num_envs, index):
    gym.clear_lines(viewer)
    for i in range(num_envs):
        # Update attractor target from current franka state
        attractor_properties = gym.get_attractor_properties(envs[i], attractor_handles[i])
        pose = attractor_properties.target
        # pose.p: (x, y, z), pose.r: (w, x, y, z)
        pose.p.x = traj[index, 0] * 0.5
        pose.p.y = traj[index, 2] * 0.5
        pose.p.z = traj[index, 1] * 0.5
        pose.r.w = traj[index, 6]
        pose.r.x = traj[index, 3]
        pose.r.y = traj[index, 5]
        pose.r.z = traj[index, 4]
        gym.set_attractor_target(envs[i], attractor_handles[i], pose)

        # Draw axes and sphere at attractor location
        gymutil.draw_lines(axes_geom, gym, viewer, envs[i], pose)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], pose)


def show(args, asset_root=None):
    print('\033[1;32m--------{}--------\033[0m'.format('Show the Franka simulator in the interactive mode'))

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args)
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    init_env(gym, sim, asset_root, asset_file, num_envs=1)

    while not gym.query_viewer_has_closed(viewer):
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)


def run_traj(args, traj, attracted_joint="panda_hand", asset_root=None, for_test=False):
    print('\033[1;32m--------{}--------\033[0m'.format('Execute trajectory with the Franka simulator'))

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args, for_test=for_test)

    # Load franka asset and set the env
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    envs, franka_handles = init_env(gym, sim, asset_root, asset_file, num_envs=1)

    # Create the attractor
    attractor_handles, axes_geom, sphere_geom = init_attractor(gym, envs, viewer, franka_handles, attracted_joint, for_test=for_test)

    # get joint limits and ranges for Franka
    franka_dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
    franka_lower_limits = franka_dof_props['lower']
    franka_upper_limits = franka_dof_props['upper']
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
    franka_num_dofs = len(franka_dof_props)

    for i in range(len(envs)):
        # Set updated stiffness and damping properties
        gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

        # Set ranka pose so that each joint is in the middle of its actuation range
        franka_dof_states = gym.get_actor_dof_states(envs[i], franka_handles[i], gymapi.STATE_NONE)
        for j in range(franka_num_dofs):
            franka_dof_states['pos'][j] = franka_mids[j]
        gym.set_actor_dof_states(envs[i], franka_handles[i], franka_dof_states, gymapi.STATE_POS)

    # Time to wait in seconds before moving robot
    next_franka_update_time = 1

    index = 0
    while not gym.query_viewer_has_closed(viewer):
        # Every 0.01 seconds the pose of the attactor is updated
        t = gym.get_sim_time(sim)
        if t >= next_franka_update_time:
            update_robot(traj, gym, envs, attractor_handles, axes_geom, sphere_geom, viewer, len(envs), index)
            next_franka_update_time += 0.01
            index += 1
            if index >= len(traj):
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
