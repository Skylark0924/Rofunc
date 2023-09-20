"""Transferring Optitrack Data of Recorded Object Pose into Isaac gym simulation.

Usage Example 1:

Usage Example 2:


"""

from isaacgym import gymutil
from isaacgym import gymapi
from rofunc.devices.optitrack.process import export


def env_setup():
    """
   set up Isaac gym environment for object pose transfer.
   """
    # initialize gym
    gym = gymapi.acquire_gym()
    # parse arguments
    args = gymutil.parse_arguments(description="Object Pose Transferring")
    # configure sim
    sim_params = gymapi.SimParams()
    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.relaxation = 0.9
        sim_params.flex.dynamic_friction = 0.0
        sim_params.flex.static_friction = 0.0
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
    # create viewer using the default camera properties
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.static_friction = 0.0
    plane_params.dynamic_friction = 0.0
    gym.add_ground(sim, plane_params)
    # set up the env grid
    # num_envs = 3
    spacing = 1.8
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    # create list to mantain environment and asset handles = [] = [] = []
    # create capsule asset
    asset_options = gymapi.AssetOptions()
    asset_options.density = 100.
    asset_capsule = gym.create_capsule(sim, 0.2, 0.2, asset_options)
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)
    return gym, sim, viewer, env, asset_capsule


def data_process(input_dir: str):
    """
   process raw optitrack .csv data and generate rigid body motion data.
   :param input_dir: csv file path
   :return: [number of frames, number of rigid bodies, pose dimension = 7]
   """
    data_array = export(input_dir)

    return data_array


def update_object_pose(capsule_handle, pose):
    gym.set_actor_dof_states(env, capsule_handle, pose)


def object_pose_transfer(csv_path, gym, sim, viewer, env, asset_capsule):
    # add capsule actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    capsule_handle = gym.create_actor(env, asset_capsule, pose, "actor2", 1, 0)

    # process.csv file and generate pose sequence
    pose_sequence = data_process(csv_path)

    next_franka_update_time = 0
    step = 0

    # step the env
    while not gym.query_viewer_has_closed(viewer):
        # Every 0.01 seconds the pose of the attactor is updated
        t = gym.get_sim_time(sim)

        if t >= next_franka_update_time:
            # update object pose
            update_object_pose(capsule_handle, pose_sequence[step])
            next_franka_update_time += 0.01
            step += 1

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    print('Done')
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    # .csv path of optitrack data of recorded object pose
    csv_path = '/home/roboy/roboy3/src/roboy3/rofunc/utils/datalab/opti_data/2021-08-11-16-11-10/opti_data.npy'

    # obtain the setup env variables
    gym, sim, viewer, env, asset_capsule = env_setup()

    # transfer the recorded object pose into Isaac gym simulation
    object_pose_transfer(csv_path, gym, sim, viewer, env, asset_capsule)
