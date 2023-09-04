# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rofunc.simulator.base.base_sim import RobotSim
from rofunc.learning.RofuncRL.tasks.amp.motion_lib import MotionLib
from isaacgym.torch_utils import *
import torch


class HumanoidSim(RobotSim):
    def __init__(self, args, **kwargs):
        # args.up_axis = 'Z'
        super().__init__(args, robot_name="humanoid", **kwargs)

    def setup_robot_dof_prop(self, **kwargs):
        from isaacgym import gymtorch

        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        key_bodies = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)

        self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        self._dof_obs_size = 72
        self._num_actions = 28
        self._num_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._humanoid_root_states = self._root_states.view(1, 1, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

    def run_traj_npy(self, motion_file):
        from isaacgym import gymapi
        from isaacgym import gymtorch

        self._load_motion(motion_file)
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

            for i in range(self.num_envs):
                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
                    self._motion_lib.motion_ids, [frame])
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
                self._humanoid_actor_ids = 1 * torch.arange(self.num_envs, device=self.device,
                                                                     dtype=torch.int32)

                self._set_env_state(env_ids=env_ids,
                                    root_pos=root_pos,
                                    root_rot=root_rot,
                                    dof_pos=dof_pos,
                                    root_vel=root_vel,
                                    root_ang_vel=root_ang_vel,
                                    dof_vel=dof_vel)

                env_ids_int32 = self._humanoid_actor_ids[env_ids]

                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                             gymtorch.unwrap_tensor(self._root_states[0]),
                                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                # self.gym.set_dof_state_tensor_indexed(self.sim,
                #                                       gymtorch.unwrap_tensor(self._dof_state),
                #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
                self._dof_state = [(i, j) for i, j in zip(self._dof_pos[0], self._dof_vel[0])]
                self.gym.set_actor_dof_states(self.envs[i], self.robot_handles[i], self._dof_state, gymapi.STATE_ALL)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
            frame += 1
            # if frame >= 29:
            #     frame = 0

        print('Done')

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.robot_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(),
                                     device=self.device)

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel


if __name__ == '__main__':
    from isaacgym import gymutil

    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False
    sim = HumanoidSim(args)
    sim.init()
    sim.show()
    # import os
    #
    # motion_file = "amp_humanoid_backflip.npy"
    # motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                                 "../../examples/data/amp/" + motion_file)
    # sim.run_traj_npy(motion_file_path)
