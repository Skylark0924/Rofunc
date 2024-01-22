import os
from torch import Tensor
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch
from gym_grasp.tasks.base.grasping_task import GraspingTask
from typing import *


class OpenDrawer(GraspingTask):
    def __init__(self,
                 cfg: Dict[str, Any],
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool) -> None:
        """Creates UR5e-SIH robot on the task of opening a drawer.

        Arguments:
            cfg: (Dict) Configuration dictionary that contains parameters for
                how the simulation should run and defines properties of the
                task.
            sim_device: (str) Device on which the physics simulation is run
                (e.g. "cuda:0" or "cpu").
            graphics_device_id: (int) ID of the device to run the rendering.
            headless: (bool) Whether to run the simulation without a viewer.
            """
        super().__init__(cfg, sim_device, graphics_device_id, headless)

    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        super()._parse_cfg(cfg)
        self.obs_types = self._parse_observation_type(
            cfg["task"]["observationType"], cfg["task"]["num_observations"])
        self.cfg["env"]["numObservations"] = self.num_observations

    def _parse_observation_type(self, observation_type,
                                num_observations) -> List:
        if isinstance(observation_type, str):
            observation_type = [observation_type]

        self.num_observations = 0
        observations = []
        for obs in observation_type:
            if not (obs in num_observations.keys()):
                raise ValueError(f"Unknown observation type '{obs}' given. "
                                 f"Should be in {num_observations.keys()}.")
            observations.append(obs)
            self.num_observations += num_observations[obs]
        self.num_obs_dict = num_observations

        if "fingertipContactForces" in observation_type:
            self.cfg["sim"]["useContactForces"] = True
        return observations

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.drawer_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, self.num_robot_dofs:]
        self.drawer_dof_pos = self.drawer_dof_state[..., 0]
        self.drawer_dof_vel = self.drawer_dof_state[..., 1]

    def _load_task_assets(self) -> Dict[str, Any]:
        self.drawers, self.drawer_indices = [], []
        drawer_asset, drawer_start_pose = self._load_drawer_asset()
        self._set_drawer_dof_limits(drawer_asset)
        assets_dict = {"drawer": {}}
        assets_dict["drawer"]["asset"] = drawer_asset
        assets_dict["drawer"]["start_pose"] = drawer_start_pose
        return assets_dict

    def _set_drawer_dof_limits(self, drawer_asset):
        self.num_drawer_dofs = self.gym.get_asset_dof_count(drawer_asset)
        self.drawer_dof_props = self.gym.get_asset_dof_properties(drawer_asset)
        # Set DOF limits and default positions and velocities
        self.drawer_dof_lower_limits = []
        self.drawer_dof_upper_limits = []
        self.drawer_dof_default_vel = []
        for i in range(self.num_drawer_dofs):
            self.drawer_dof_lower_limits.append(self.drawer_dof_props["lower"][i])
            self.drawer_dof_upper_limits.append(self.drawer_dof_props["upper"][i])
            self.drawer_dof_default_vel.append(0.0)

        self.drawer_dof_lower_limits = to_torch(self.drawer_dof_lower_limits,
                                               device=self.device)
        self.drawer_dof_upper_limits = to_torch(self.drawer_dof_upper_limits,
                                               device=self.device)
        self.drawer_dof_default_pos = to_torch([0.],
                                              device=self.device)
        self.drawer_dof_default_vel = to_torch(self.drawer_dof_default_vel,
                                              device=self.device)

    def _load_drawer_asset(self):
        """Loads the drawer to be opened."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True

        # Enable convex decomposition for handle geometry
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000
        
        drawer_asset = self.gym.load_asset(
            self.sim, self.asset_root, self.drawer_asset_file,
            asset_options)

        x, y, z = self.cfg["asset"]["drawerPosition"]
        q_w, q_x, q_y, q_z = self.cfg["asset"]["drawerRotation"]
        drawer_start_pose = gymapi.Transform(
            p=gymapi.Vec3(x, y, z), r=gymapi.Quat(q_x, q_y, q_z, q_w))

        self.num_drawer_bodies = self.gym.get_asset_rigid_body_count(
            drawer_asset)
        self.num_drawer_shapes = self.gym.get_asset_rigid_shape_count(
            drawer_asset)
        return drawer_asset, drawer_start_pose

    @property
    def drawer_asset_file(self) -> os.path:
        drawer_asset_file = "urdf/drawer.urdf"
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "drawerAssetFile", drawer_asset_file))
        return drawer_asset_file

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any],
                  agg_info) -> None:
        self._add_drawer(env_ptr, i, task_asset_dict["drawer"]["asset"],
                         task_asset_dict["drawer"]["start_pose"])

    def _process_task_handles(self) -> None:
        self.drawer_indices = to_torch(
            self.drawer_indices, dtype=torch.long, device=self.device)

    def _begin_aggregate(self, env_ptr, env_idx: int) -> None:
        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies + \
                         self.num_drawer_bodies
        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes + \
                         self.num_drawer_shapes
        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)

    def _add_drawer(self, env_ptr, i, drawer_asset, drawer_start_pose) -> None:
        # create drawer actor
        drawer_actor = self.gym.create_actor(
            env_ptr, drawer_asset, drawer_start_pose, 'drawer', i, 0, 0)

        if "drawerColor" in self.cfg["asset"].keys():
            self.gym.set_rigid_body_color(
                env_ptr, drawer_actor, 2, gymapi.MeshType.MESH_VISUAL,
                gymapi.Vec3(*self.cfg["asset"]["drawerColor"]))

        self.handle_handle = [self.gym.find_asset_rigid_body_index(
            drawer_asset, 'handle') + self.num_robot_bodies +
                              self.num_table_bodies]
        self.handle_handle = to_torch(
            self.handle_handle, dtype=torch.long, device=self.device)

        self.drawer_indices.append(self.gym.get_actor_index(
            env_ptr, drawer_actor, gymapi.DOMAIN_SIM))
        self.drawers.append(drawer_actor)

    def compute_reward(self) -> None:
        if self.cfg["control"]["useRelativeControl"]:
            actions = self.actions
        else:
            actions = self.actions - self.prev_actions
        self.prev_actions = self.actions

        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], \
        self.successes[:], self.success_rate[:], \
        action_reward, \
        fingertips_to_handle_distance_reward, \
        drawer_opening_reward, \
        target_reward, \
        self.rewards_dict, \
        = compute_drawer_opening_reward(
            self.rew_buf, self.reset_buf, self.progress_buf,
            self.successes, self.success_rate,
            self.handle_pos,
            to_torch(self.cfg["reward"]["handleDefaultPos"],
                     device=self.device),
            self.fingertip_pos, actions,
            self.cfg["reward"]["scale"]["actionPenalty"],
            self.cfg["reward"]["scale"]["fingertipsToHandleDistanceReward"],
            self.max_episode_length,
            self.cfg["reward"]["scale"]["drawerOpeningReward"],
            self.cfg["reward"]["scale"]["targetReward"],
            self.cfg["reward"]["targetOpening"],
            self.cfg["reward"]["epsFingertips"],
            self.cfg["reward"]["sparse"],
            self.cfg["reset"]["drawerOpened"],
            self.cfg["reward"]["returnRewardsDict"]
        )

        if self.cfg["debug"]["verbose"]:
            self.writer.add_scalar(
                "action_reward",
                torch.mean(action_reward), self.env_steps)
            self.writer.add_scalar(
                "fingertips_to_handle_distance_reward",
                torch.mean(fingertips_to_handle_distance_reward), self.env_steps)
            self.writer.add_scalar(
                "drawer_opening_reward",
                torch.mean(drawer_opening_reward), self.env_steps)
            self.writer.add_scalar(
                "target_reward",
                torch.mean(target_reward), self.env_steps)

        self.writer.add_scalar("successes", torch.sum(self.successes),
                               self.env_steps)
        self.writer.add_scalar("success_rate", self.success_rate,
                               self.env_steps)

    def compute_observations(self):
        """Updates the state tensors and adds the desired information to the
        observation buffer."""
        self._refresh_state_tensors()
        self._process_state_tensors()

        obs_idx = 0
        for obs in self.obs_types:
            if obs == "fingertipContactForces":
                self.add_fingertip_contact_forces(obs_idx)
            elif obs == "fingertipPos":
                self.add_fingertip_pos(obs_idx)
            elif obs == "handPose":
                self.add_hand_pose(obs_idx)
            elif obs == "handDofPos":
                self.add_hand_dof_pos(obs_idx)
            elif obs == "jointPos":
                self.add_joint_pos(obs_idx)
            elif obs == "jointVel":
                self.add_joint_vel(obs_idx)
            elif obs == "previousAction":
                self.add_previous_action(obs_idx)
            elif obs == "handlePos":
                self.add_handle_pos(obs_idx)
            elif obs == "drawerOpening":
                self.add_drawer_opening(obs_idx)

            obs_idx += self.num_obs_dict[obs]

    def _process_state_tensors(self) -> None:
        super()._process_state_tensors()

        self.handle_state = self.rigid_body_states[:, self.handle_handle][
                            :, :, 0:13]
        self.handle_rigid_body_pos = self.handle_state[:, 0, 0:3]
        self.handle_pos = self.calculate_handle_pos()

    def calculate_handle_pos(self) -> Tensor:
        handle_pos = self.handle_rigid_body_pos.clone()
        handle_pos += to_torch([[0.175, 0, 0.9]], device=self.device
                               ).repeat(self.num_envs, 1)
        return handle_pos

    def add_handle_pos(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx + 3] = self.handle_pos
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("handlePos", self.handle_pos)

    def add_drawer_opening(self, start_idx: int) -> None:
        # 1-dimensional version of handle pos observation

        self.obs_buf[:, start_idx:start_idx + 1] = self.handle_pos[:, 1:2]
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("drawerOpening", self.handle_pos[:, 1:2])

    def reset_idx(self, env_ids):
        # Domain randomization would be added here ...

        if not self.initial_reset and not self.cfg["control"]["teleoperated"]:
            self._reset_robot(env_ids, apply_dof_resets=False)
            self._reset_drawer(env_ids)
            # Reset DOF state
            robot_indices = self.robot_indices[env_ids].to(torch.int32)
            drawer_indices = self.drawer_indices[env_ids].to(torch.int32)
            dof_indices = torch.cat([robot_indices, drawer_indices])
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))
        else:
            self._reset_robot(env_ids, apply_dof_resets=False,
                              initial_sampling=True)
            self._reset_drawer(env_ids)
            # Reset DOF state
            robot_indices = self.robot_indices[env_ids].to(torch.int32)
            drawer_indices = self.drawer_indices[env_ids].to(torch.int32)
            dof_indices = torch.cat([robot_indices, drawer_indices])
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(dof_indices), len(dof_indices))

            # simulate for one time-step to update the tracker position
            self.render()
            self.gym.simulate(self.sim)
            self._refresh_state_tensors()
            self.initial_tracker_pose = self.tracker_pose
            self.initial_finger_angles = self.robot_dof_pos[:,
                                         self.hand_actuated_dof_indices]

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_drawer(self, env_ids):
        rand_floats = torch_rand_float(
            0.0, 1.0, (len(env_ids), self.num_drawer_dofs),
            device=self.device)

        delta_max = self.drawer_dof_upper_limits - self.drawer_dof_default_pos
        delta_min = self.drawer_dof_lower_limits - self.drawer_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats
        pos = self.drawer_dof_default_pos + \
              self.cfg["initState"]["noise"]["drawerDofPos"] * rand_delta

        self.drawer_dof_pos[env_ids, :] = pos
        self.drawer_dof_vel[env_ids, :] = self.drawer_dof_default_vel

        drawer_indices = self.drawer_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(drawer_indices), len(drawer_indices))

    def _draw_debug_visualization(self, i: int) -> None:
        super()._draw_debug_visualization(i)
        if self.cfg["debug"]["drawHandlePos"]:
            self._draw_handle_pos(i)

    def _draw_handle_pos(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.handle_pos[i],
                                     to_torch([0, 0, 0, 1], device=self.device), axis_length=0.5)


@torch.jit.script
def compute_drawer_opening_reward(
    rew_buf, reset_buf, progress_buf, successes, success_rate,
    drawer_pos, handle_default_pos,
    fingertip_pos,  actions,
    action_penalty_scale: float,
    fingertips_to_handle_distance_reward_scale: float,
    max_episode_length: int,
    drawer_opening_reward_scale: float,
    target_reward_scale: float,
    target_opening: float,
    eps_fingertips: float,
    sparse: bool,
    reset_when_drawer_is_opened: bool,
    return_rewards_dict: bool,
):

    # ============ computing reward ... ============
    # penalize large actions: r ~ -|| a_t ||^2
    squared_action_norm = torch.sum(actions.pow(2), dim=-1)
    action_reward = squared_action_norm * action_penalty_scale

    # penalize fingertips-handle distance: r ~ (1 / (d(ft, handle)^2 + ϵ_ft))^2
    handle_center = drawer_pos.unsqueeze(1).repeat(1, fingertip_pos.shape[1], 1)
    fingertips_to_handle_distance = torch.norm(fingertip_pos - handle_center, dim=-1)
    thumb_scaling = 2
    fingertips_to_handle_distance[:, 0] *= thumb_scaling
    fingertips_to_handle_distance = torch.sum(fingertips_to_handle_distance, dim=-1)
    fingertips_to_handle_distance_reward = \
        1.0 / (eps_fingertips + 5 * fingertips_to_handle_distance.pow(2))
    fingertips_to_handle_distance_reward *= fingertips_to_handle_distance_reward
    fingertips_to_handle_distance_reward *= fingertips_to_handle_distance_reward_scale

    # continuous drawer-opening reward: r ~ Δd
    drawer_opened_by = -(drawer_pos[:, 1] - handle_default_pos[1])
    drawer_opening_reward = drawer_opening_reward_scale * drawer_opened_by

    # reward reaching the target drawer opening: r ~ 1(drawer_opened == True)
    drawer_opened = drawer_opened_by >= target_opening
    target_reward = target_reward_scale * drawer_opened.float()

    # compute final reward
    rewards_dict = {}
    if sparse:
        reward = drawer_opened.float() - 1
        if return_rewards_dict:
            dense_reward = action_reward + \
                fingertips_to_handle_distance_reward + \
                drawer_opening_reward + \
                target_reward
            rewards_dict["sparse"] = reward
            rewards_dict["dense"] = dense_reward
    else:
        reward = action_reward + \
            fingertips_to_handle_distance_reward + \
            drawer_opening_reward + \
            target_reward
        if return_rewards_dict:
            sparse_reward = drawer_opened.float() - 1
            rewards_dict["sparse"] = sparse_reward
            rewards_dict["dense"] = reward

    # ============ determining resets ... ============
    # reset environments that have reached the maximum number of steps
    resets = torch.where(progress_buf >= max_episode_length,
                         torch.ones_like(reset_buf), reset_buf)

    # reset environments that opened the drawer successfully
    if reset_when_drawer_is_opened:
        resets = torch.where(drawer_opened,
                             torch.ones_like(resets), resets)

    # determine total number of successful episodes and success rate
    total_resets = torch.sum(resets)
    avg_factor = 0.01
    successful_resets = torch.sum(drawer_opened.float())
    successes += drawer_opened.float()
    success_rate = torch.where(total_resets > 0,
                               avg_factor * (successful_resets/total_resets) +
                               (1. - avg_factor) * success_rate,
                               success_rate)

    return reward, resets, progress_buf, \
           successes, success_rate, \
           action_reward, \
           fingertips_to_handle_distance_reward, \
           drawer_opening_reward, \
           target_reward, \
           rewards_dict
