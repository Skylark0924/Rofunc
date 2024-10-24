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

# Self-implemented human simulation with urdf built from xsens model
# Deprecated since the xsens model is not accurate enough
# Try use humanoid robot simulation instead (see humanoid_sim.py)

from rofunc.simulator.base_sim import RobotSim
import numpy as np


class HumanSim(RobotSim):
    def __init__(self, args, robot_name, asset_root=None, asset_file=None, fix_base_link=None,
                 flip_visual_attachments=True, init_pose_vec=None, num_envs=1, device="cpu"):
        super().__init__(args, robot_name, asset_root, asset_file, fix_base_link, flip_visual_attachments,
                         init_pose_vec, num_envs, device)
        self.asset_file = asset_file
        self.flip_visual_attachments = False
        self.fix_base_link = False if fix_base_link is None else fix_base_link
        pos_y, pos_z = 0.8, 0.

    def setup_robot_dof_prop(self, gym=None, envs=None, robot_asset=None, robot_handles=None):
        from isaacgym import gymapi

        gym = self.gym if gym is None else gym
        envs = self.envs if envs is None else envs
        robot_asset = self.robot_asset if robot_asset is None else robot_asset
        robot_handles = self.robot_handles if robot_handles is None else robot_handles

        # configure robot dofs
        robot_dof_props = gym.get_asset_dof_properties(robot_asset)
        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.3 * (robot_upper_limits + robot_lower_limits)

        robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][:].fill(1000000.0)
        robot_dof_props["damping"][:].fill(1000000.0)

        # default dof states and position targets
        robot_num_dofs = gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        # default_dof_pos[:] = robot_mids[:]

        default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # # send to torch
        # default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

        for i in range(len(envs)):
            # set dof properties
            gym.set_actor_dof_properties(envs[i], robot_handles[i], robot_dof_props)

            # set initial dof states
            gym.set_actor_dof_states(envs[i], robot_handles[i], default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            gym.set_actor_dof_position_targets(envs[i], robot_handles[i], default_dof_pos)

    def run_demo(self, xsens_data):
        from isaacgym import gymapi

        frame = 0

        # Simulate
        while not self.gym.query_viewer_has_closed(self.viewer):

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
            default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
            default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"] = default_dof_pos

            # dof_names = ['jL5S1_rotx', 'jL5S1_roty', 'jL4L3_rotx', 'jL4L3_roty', 'jL1T12_rotx', 'jL1T12_roty',
            #              'jT9T8_rotz', 'jT9T8_rotx', 'jT9T8_roty', 'jLeftC7Shoulder_rotx', 'jLeftShoulder_rotz',
            #              'jLeftShoulder_rotx', 'jLeftShoulder_roty', 'jLeftElbow_rotz', 'jLeftElbow_roty',
            #              'jLeftWrist_rotz', 'jLeftWrist_rotx', 'jRightC7Shoulder_rotx', 'jRightShoulder_rotz',
            #              'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightElbow_rotz', 'jRightElbow_roty',
            #              'jRightWrist_rotz', 'jRightWrist_rotx', 'jT1C7_rotz', 'jT1C7_rotx', 'jT1C7_roty',
            #              'jC1Head_rotx', 'jC1Head_roty', 'jLeftHip_rotz', 'jLeftHip_rotx', 'jLeftHip_roty',
            #              'jLeftKnee_rotz', 'jLeftKnee_roty', 'jLeftAnkle_rotz', 'jLeftAnkle_rotx', 'jLeftAnkle_roty',
            #              'jLeftBallFoot_roty', 'jRightHip_rotz', 'jRightHip_rotx', 'jRightHip_roty', 'jRightKnee_rotz',
            #              'jRightKnee_roty', 'jRightAnkle_rotz', 'jRightAnkle_rotx', 'jRightAnkle_roty',
            #              'jRightBallFoot_roty']
            for i in range(self.num_envs):
                dof_info = self.get_dof_info()
                dof_pos = []
                for dof_name in dof_info['dof_names']:
                    # dof_handle = self.gym.find_actor_dof_handle(self.envs[i], self.robot_handles[i], dof_name)
                    if dof_name[-1] == 'x':
                        index = 1
                    elif dof_name[-1] == 'y':
                        index = 2
                    elif dof_name[-1] == 'z':
                        index = 0
                    else:
                        raise ValueError('Invalid dof name')

                    # set initial dof states
                    joint_name = dof_name.split('_')[0]
                    if joint_name == 'jLeftC7Shoulder':
                        joint_name = 'jLeftT4Shoulder'
                    if joint_name == 'jRightC7Shoulder':
                        joint_name = 'jRightT4Shoulder'
                    joint_value = xsens_data.file_data['frames']['joint_data'][frame][joint_name]

                    # if 'LeftShoulder' in joint_name or 'RightShoulder' in joint_name:
                    #     # if dof_name[-1] == 'x':
                    #     #     index = 0
                    #     # elif dof_name[-1] == 'y':
                    #     #     index = 2
                    #     # elif dof_name[-1] == 'z':
                    #     #     index = 1
                    #
                    #     joint_value = xsens_data.file_data['frames']['joint_data_xzy'][frame][joint_name]

                    # if joint_name in ['jLeftWrist']:
                    joint_value = np.pi * joint_value / 180.
                    # self.gym.set_dof_target_position(self.envs[i], dof_handle, joint_value[index])
                    dof_pos.append(joint_value[index])
                self.gym.set_actor_dof_states(self.envs[i], self.robot_handles[i], dof_pos, gymapi.STATE_ALL)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
            frame += 1
            if frame >= len(xsens_data.file_data['frames']['joint_data']):
                break

        print('Done')
