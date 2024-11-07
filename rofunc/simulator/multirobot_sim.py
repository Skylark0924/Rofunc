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

import math
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
from PIL import Image as Im

from rofunc.simulator.base_sim import PlaygroundSim
from rofunc.utils.logger.beauty_logger import beauty_print


class MultiRobotSim:
    def __init__(self, args, robot_sims: Dict, num_envs=1, **kwargs):
        self.args = args
        self.num_envs = num_envs
        self.robot_sims = robot_sims

    def init(self):
        # Initial gym, sim, viewer and env
        self.PlaygroundSim = PlaygroundSim(self.args)
        self.gym = self.PlaygroundSim.gym
        self.sim = self.PlaygroundSim.sim
        self.viewer = self.PlaygroundSim.viewer
        self.init_env()

        for robot_name, robot_sim in self.robot_sims.items():
            robot_sim.setup_robot_dof_prop(self.gym, self.envs, self.multi_robot_assets[robot_name],
                                           self.multi_robot_handles[robot_name])
            # self.robot_dof = self.gym.get_actor_dof_count(self.envs[0], self.robot_handles[0])

    def init_env(self, spacing=3.0):
        from isaacgym import gymapi

        # Set up the env grid
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # load robot asset
        self.multi_robot_assets = {}
        for robot_name, robot_sim in self.robot_sims.items():
            s = robot_sim
            beauty_print("Loading robot asset {} from {}".format(s.asset_file, s.asset_root), type="info")
            robot_asset = self.gym.load_asset(self.sim, s.asset_root, s.asset_file, s.asset_options)
            self.multi_robot_assets[robot_name] = robot_asset

        envs = []
        multi_robot_handles = defaultdict(list)

        # configure env grid
        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)

            # add robots
            for robot_name, robot_sim in self.robot_sims.items():
                s = robot_sim
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(s.init_pose_vec[0], s.init_pose_vec[1], s.init_pose_vec[2])
                pose.r = gymapi.Quat(s.init_pose_vec[3], s.init_pose_vec[4], s.init_pose_vec[5], s.init_pose_vec[6])
                robot_handle = self.gym.create_actor(env, self.multi_robot_assets[robot_name], pose, robot_name, i, 2)
                self.gym.enable_actor_dof_force_sensors(env, robot_handle)
                multi_robot_handles[robot_name].append(robot_handle)

        self.envs = envs
        self.multi_robot_handles = multi_robot_handles

    def show(self, visual_obs_flag=False, camera_props=None, attached_body=None, local_transform=None):
        """
        Visualize the robot in an interactive viewer
        :param visual_obs_flag: If True, show the visual observation
        :param camera_props: If visual_obs_flag is True, use this camera_props to config the camera
        :param attached_body: If visual_obs_flag is True, use this to refer the body the camera attached to
        :param local_transform: If visual_obs_flag is True, use this local transform to adjust the camera pose
        """
        from isaacgym import gymapi

        # beauty_print("Show the {} simulator in the interactive mode".format(self.robot_name), 1)

        if visual_obs_flag:
            fig = plt.figure("Visual observation", figsize=(8, 8))
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
            body_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], attached_body)
            self.gym.attach_camera_to_body(camera_handle, self.envs[0], body_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

        while not self.gym.query_viewer_has_closed(self.viewer):
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)

            if visual_obs_flag:
                # digest image
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                cam_img = self.gym.get_camera_image(self.sim, self.envs[0], camera_handle, gymapi.IMAGE_COLOR).reshape(
                    1280, 1280, 4)
                cam_img = Im.fromarray(cam_img)
                plt.imshow(cam_img)
                plt.axis('off')
                plt.pause(1e-9)
                fig.clf()

                self.gym.end_access_image_tensors(self.sim)

            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
