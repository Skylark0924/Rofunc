from typing import List

import numpy as np

from rofunc.simulator.base.base_sim import RobotSim


class CURISim(RobotSim):
    def __init__(self, args, **kwargs):
        super().__init__(args, robot_name="CURI", **kwargs)
        self._setup_curi()

    def _setup_curi(self):
        from isaacgym import gymapi

        self.curi_handles = self.handles
        # get joint limits and ranges for CURI
        curi_dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.curi_handles[0])
        curi_lower_limits = curi_dof_props['lower']
        curi_upper_limits = curi_dof_props['upper']
        curi_mids = 0.5 * (curi_upper_limits + curi_lower_limits)
        curi_num_dofs = len(curi_dof_props)

        for i in range(len(self.envs)):
            # Set updated stiffness and damping properties
            self.gym.set_actor_dof_properties(self.envs[i], self.curi_handles[i], curi_dof_props)

            # Set ranka pose so that each joint is in the middle of its actuation range
            curi_dof_states = self.gym.get_actor_dof_states(self.envs[i], self.curi_handles[i], gymapi.STATE_NONE)
            for j in range(curi_num_dofs):
                curi_dof_states['pos'][j] = curi_mids[j]
            self.gym.set_actor_dof_states(self.envs[i], self.curi_handles[i], curi_dof_states, gymapi.STATE_POS)

    def setup_attractors(self, traj, attracted_joints):
        if attracted_joints is None:
            attracted_joints = ["panda_left_hand", "panda_right_hand"]
        else:
            assert isinstance(attracted_joints, list), "The attracted joints should be a list"
            assert len(attracted_joints) > 0, "The length of the attracted joints should be greater than 0"
        assert len(attracted_joints) == len(traj), "The first dimension of trajectory should equal to attracted_joints"

        attractor_handles, axes_geoms, sphere_geoms = [], [], []
        for i in range(len(attracted_joints)):
            attractor_handle, axes_geom, sphere_geom = self._init_attractor(attracted_joints[i])
            attractor_handles.append(attractor_handle)
            axes_geoms.append(axes_geom)
            sphere_geoms.append(sphere_geom)
        return attracted_joints, attractor_handles, axes_geoms, sphere_geoms

    def show(self, visual_obs_flag=False):
        """
        Visualize the CURI robot
        :param visual_obs_flag: if True, show visual observation
        """
        from isaacgym import gymapi

        # Camera Sensor
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 1280

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.12, 0, 0.18)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(90.0)) * \
                            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90.0))
        super(CURISim, self).show(visual_obs_flag, camera_props, "head_link2", local_transform)

    # def run_traj(self, traj, attracted_joint="panda_right_hand", asset_root=None, update_freq=0.001):
    #     from isaacgym import gymapi
    #
    #     print('\033[1;32m--------{}--------\033[0m'.format('Execute trajectory with the CURI simulator'))
    #
    #     # Initial gym and sim
    #     gym, sim_params, sim, viewer = init_sim(args)
    #
    #     # Load CURI asset and set the env
    #     if asset_root is None:
    #         from rofunc.utils.file import get_rofunc_path
    #         asset_root = os.path.join(get_rofunc_path(), "simulator/assets")
    #     asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
    #     envs, curi_handles = init_env(gym, sim, asset_root, asset_file, num_envs=1, fix_base_link=False)
    #
    #     # Create the attractor
    #     attractor_handles, axes_geom, sphere_geom = init_attractor(gym, envs, viewer, curi_handles, attracted_joint)
    #
    #     # get joint limits and ranges for Franka
    #     curi_dof_props = gym.get_actor_dof_properties(envs[0], curi_handles[0])
    #     curi_lower_limits = curi_dof_props['lower']
    #     curi_upper_limits = curi_dof_props['upper']
    #     curi_mids = 0.5 * (curi_upper_limits + curi_lower_limits)
    #     curi_num_dofs = len(curi_dof_props)
    #
    #     for i in range(len(envs)):
    #         # Set updated stiffness and damping properties
    #         gym.set_actor_dof_properties(envs[i], curi_handles[i], curi_dof_props)
    #
    #         # Set ranka pose so that each joint is in the middle of its actuation range
    #         curi_dof_states = gym.get_actor_dof_states(envs[i], curi_handles[i], gymapi.STATE_NONE)
    #         for j in range(curi_num_dofs):
    #             curi_dof_states['pos'][j] = curi_mids[j]
    #         gym.set_actor_dof_states(envs[i], curi_handles[i], curi_dof_states, gymapi.STATE_POS)
    #
    #     # Time to wait in seconds before moving robot
    #     next_curi_update_time = 1
    #
    #     index = 0
    #     while not gym.query_viewer_has_closed(viewer):
    #         # Every 0.01 seconds the pose of the attractor is updated
    #         t = gym.get_sim_time(sim)
    #         if t >= next_curi_update_time:
    #             gym.clear_lines(viewer)
    #             update_robot(traj, gym, envs, attractor_handles, axes_geom, sphere_geom, viewer, len(envs), index, t)
    #             next_curi_update_time += update_freq
    #             index += 1
    #             if index >= len(traj):
    #                 index = 0
    #
    #         # Step the physics
    #         gym.simulate(sim)
    #         gym.fetch_results(sim, True)
    #
    #         # Step rendering
    #         gym.step_graphics(sim)
    #         gym.draw_viewer(viewer, sim, False)
    #         gym.sync_frame_time(sim)
    #
    #     print("Done")
    #
    #     gym.destroy_viewer(viewer)
    #     gym.destroy_sim(sim)

    def run_traj_multi_joints_with_interference(self, traj: List, intf_index: List, intf_mode: str,
                                                intf_forces=None, intf_torques=None, intf_joints: List = None,
                                                intf_efforts: np.ndarray = None, attracted_joints: List = None,
                                                update_freq=0.001, save_name=None):
        """
        Run the trajectory with multiple joints with interference, the default is to run the trajectory with the left and
        right hand of the CURI robot.
        Args:
            traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
            intf_index: a list of the timing indices of the interference occurs
            intf_mode: the mode of the interference, ["actor_dof_efforts", "body_forces", "body_force_at_pos"]
            intf_forces: a tensor of shape (num_envs, num_bodies, 3), the interference forces applied to the bodies
            intf_torques: a tensor of shape (num_envs, num_bodies, 3), the interference torques applied to the bodies
            intf_joints: [list], e.g. ["panda_left_hand"]
            intf_efforts: array containing the efforts for all degrees of freedom of the actor.
            attracted_joints: [list], e.g. ["panda_left_hand", "panda_right_hand"]
            asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets
            update_freq: the frequency of updating the robot pose
            num_envs: the number of environments
        """
        from isaacgym import gymapi
        from isaacgym import gymtorch
        import torch
        from rofunc.utils.logger.beauty_logger import beauty_print

        assert isinstance(traj, list) and len(traj) > 0, "The trajectory should be a list of numpy arrays"
        assert intf_mode in ["actor_dof_efforts", "body_forces", "body_force_at_pos"], \
            "The interference mode should be one of ['actor_dof_efforts', 'body_forces', 'body_force_at_pos']"

        beauty_print('Execute multi-joint trajectory with interference with the CURI simulator')

        device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'
        num_bodies = self.get_num_bodies()
        if intf_forces is not None:
            assert intf_forces.shape == torch.Size(
                [self.num_envs, num_bodies, 3]), "The shape of forces should be (num_envs, num_bodies, 3)"
            intf_forces = intf_forces.to(device)
        if intf_torques is not None:
            assert intf_torques.shape == torch.Size(
                [self.num_envs, num_bodies, 3]), "The shape of torques should be (num_envs, num_bodies, 3)"
            intf_torques = intf_torques.to(device)

        # Create the attractor
        attracted_joints, attractor_handles, axes_geoms, sphere_geoms = self.setup_attractors(traj, attracted_joints)

        # Time to wait in seconds before moving robot
        next_curi_update_time = 1
        index = 0
        states = []
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)
            if t >= next_curi_update_time:
                self.gym.clear_lines(self.viewer)
                for i in range(len(attracted_joints)):
                    self.update_robot(traj[i], attractor_handles[i], axes_geoms[i], sphere_geoms[i], index)
                next_curi_update_time += update_freq
                index += 1
                if index >= len(traj[0]):
                    index = 0

                # Create the interference
                if index in intf_index:
                    if intf_mode == "actor_dof_efforts":
                        # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(intf_efforts))
                        for i in range(len(self.envs)):
                            self.gym.apply_actor_dof_efforts(self.envs[i], self.curi_handles[i], intf_efforts)
                    elif intf_mode == "body_forces":
                        # set intf_forces and intf_torques for the specific bodies
                        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(intf_forces),
                                                                gymtorch.unwrap_tensor(intf_torques), gymapi.ENV_SPACE)

                # Get current robot state
                state = self.get_robot_state(mode='dof_state')
                states.append(np.array(state))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        print("Done")

        with open('{}.npy'.format(save_name), 'wb') as f:
            np.save(f, np.array(states))
        beauty_print('{}.npy saved'.format(save_name), 2)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
