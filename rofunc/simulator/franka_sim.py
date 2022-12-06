from rofunc.simulator.base.base_sim import RobotSim


class FrankaSim(RobotSim):
    def __init__(self, args, **kwargs):
        super().__init__(args, robot_name="franka", **kwargs)

    def update_robot(self, traj, attractor_handles, axes_geom, sphere_geom, index):
        # TODO: the traj is a little weird, need to be fixed
        from isaacgym import gymutil

        self.gym.clear_lines(self.viewer)
        for i in range(self.num_envs):
            # Update attractor target from current franka state
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], attractor_handles[i])
            pose = attractor_properties.target
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            pose.p.x = traj[index, 0] * 0.5
            pose.p.y = traj[index, 2] * 0.5
            pose.p.z = traj[index, 1] * 0.5
            pose.r.w = traj[index, 6]
            pose.r.x = traj[index, 3]
            pose.r.y = traj[index, 5]
            pose.r.z = traj[index, 4]
            self.gym.set_attractor_target(self.envs[i], attractor_handles[i], pose)

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def run_traj(self, traj, attracted_joint="panda_hand"):
        self.run_traj_multi_joints([traj], [attracted_joint])
