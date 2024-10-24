#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

from typing import List

import numpy as np

from rofunc.simulator.base_sim import RobotSim
from rofunc.utils.logger.beauty_logger import beauty_print


class GluonSim(RobotSim):
    def __init__(self, args):
        super().__init__(args)

    def show(self, visual_obs_flag=False, camera_props=None, attached_body=None, local_transform=None):
        """
        Visualize the CURI robot
        :param visual_obs_flag: if True, show visual observation
        :param camera_props: If visual_obs_flag is True, use this camera_props to config the camera
        :param attached_body: If visual_obs_flag is True, use this to refer the body the camera attached to
        :param local_transform: If visual_obs_flag is True, use this local transform to adjust the camera pose
        """
        from isaacgym import gymapi

        if visual_obs_flag:
            # Setup a first-person camera embedded in CURI's head
            if camera_props is None:
                # Camera Sensor
                camera_props = gymapi.CameraProperties()
                camera_props.width = 1280
                camera_props.height = 1280

            if attached_body is None:
                attached_body = "head_link2"

            if local_transform is None:
                local_transform = gymapi.Transform()
                local_transform.p = gymapi.Vec3(0.12, 0, 0.18)
                local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.radians(90.0)) * \
                                    gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90.0))

        super(GluonSim, self).show(visual_obs_flag, camera_props, attached_body, local_transform)

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index):
        from isaacgym import gymutil

        for i in range(self.num_envs):
            # Update attractor target from current franka state
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], attractor_handles[i])
            pose = attractor_properties.target
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            pose.p.x = traj[index, 0]
            pose.p.y = traj[index, 2]
            pose.p.z = traj[index, 1]
            pose.r.w = traj[index, 6]
            pose.r.x = traj[index, 3]
            pose.r.y = traj[index, 5]
            pose.r.z = traj[index, 4]
            self.gym.set_attractor_target(self.envs[i], attractor_handles[i], pose)

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def run_traj(self, traj, attracted_rigid_bodies="6_Link", update_freq=0.001):
        self.run_traj_multi_rigid_bodies([traj], [attracted_rigid_bodies], update_freq)

    def run_traj_multi_rigid_bodies_with_interference(self, traj: List, intf_index: List, intf_mode: str,
                                                intf_forces=None, intf_torques=None, intf_joints: List = None,
                                                intf_efforts: np.ndarray = None, attracted_rigid_bodies: List = None,
                                                update_freq=0.001, save_name=None):
        """
        Run the trajectory with multiple rigid bodies with interference, the default is to run the trajectory with the left and
        right hand of the CURI robot.
        Args:
            traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
            intf_index: a list of the timing indices of the interference occurs
            intf_mode: the mode of the interference, ["actor_dof_efforts", "body_forces", "body_force_at_pos"]
            intf_forces: a tensor of shape (num_envs, num_bodies, 3), the interference forces applied to the bodies
            intf_torques: a tensor of shape (num_envs, num_bodies, 3), the interference torques applied to the bodies
            intf_joints: [list], e.g. ["panda_left_hand"]
            intf_efforts: array containing the efforts for all degrees of freedom of the actor.
            attracted_rigid_bodies: [list], e.g. ["panda_left_hand", "panda_right_hand"]
            update_freq: the frequency of updating the robot pose
        """
        from isaacgym import gymapi
        from isaacgym import gymtorch
        import torch

        assert isinstance(traj, list) and len(traj) > 0, "The trajectory should be a list of numpy arrays"
        assert intf_mode in ["actor_dof_efforts", "body_forces", "body_force_at_pos"], \
            "The interference mode should be one of ['actor_dof_efforts', 'body_forces', 'body_force_at_pos']"

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
        attracted_rigid_bodies, attractor_handles, axes_geoms, sphere_geoms = self.setup_attractors(traj, attracted_rigid_bodies)

        # Time to wait in seconds before moving robot
        next_gluon_update_time = 1
        index = 0
        states = []
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)
            if t >= next_gluon_update_time:
                self.gym.clear_lines(self.viewer)
                for i in range(len(attracted_rigid_bodies)):
                    self.update_robot(traj[i], attractor_handles[i], axes_geoms[i], sphere_geoms[i], index)
                next_gluon_update_time += update_freq
                index += 1
                if index >= len(traj[0]):
                    index = 0

                # Create the interference
                if index in intf_index:
                    if intf_mode == "actor_dof_efforts":
                        # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(intf_efforts))
                        for i in range(len(self.envs)):
                            self.gym.apply_actor_dof_efforts(self.envs[i], self.robot_handles[i], intf_efforts)
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
        beauty_print('{}.npy saved'.format(save_name), type="info")

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
