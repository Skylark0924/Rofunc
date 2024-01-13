import os
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch
from gym_grasp.tasks.base.grasping_task import GraspingTask
from typing import *
import math


class PourCup(GraspingTask):
    def __init__(self,
                 cfg: Dict[str, Any],
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool) -> None:
        """Creates UR5e-SIH robot on the task of pouring from a cup.

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

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.balls_poured = torch.zeros(self.num_envs,
                                        self.cfg["asset"]["numBalls"],
                                        dtype=torch.bool, device=self.device)
        self.num_balls_poured = torch.zeros(self.num_envs, dtype=torch.int,
                                            device=self.device)
        self.full_cup_in_empty_cup = torch.zeros(self.num_envs,
                                                 dtype=torch.bool,
                                                 device=self.device)

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

    def _load_task_assets(self) -> Dict[str, Any]:
        self.full_cups, self.empty_cups = [], []
        self.full_cup_indices, self.empty_cup_indices = [], []
        self.full_cup_init_state = []
        self.balls = [[] for _ in range(self.num_envs)]
        self.ball_init_state = []
        self.ball_indices = torch.zeros(
            [self.num_envs, self.cfg["asset"]["numBalls"]],
            dtype=torch.long, device=self.device)

        full_cup_asset, empty_cup_asset, full_cup_start_pose, \
        empty_cup_start_pose = self._load_cup_assets()
        ball_asset, ball_start_poses = self._load_ball_assets()
        assets_dict = {"full_cup": {}, "empty_cup": {}, "ball": {}}
        assets_dict["full_cup"]["asset"] = full_cup_asset
        assets_dict["full_cup"]["start_pose"] = full_cup_start_pose
        assets_dict["empty_cup"]["asset"] = empty_cup_asset
        assets_dict["empty_cup"]["start_pose"] = empty_cup_start_pose
        assets_dict["ball"]["asset"] = ball_asset
        assets_dict["ball"]["start_pose"] = ball_start_poses
        return assets_dict

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any],
                  agg_info) -> None:
        self._add_cups(env_ptr, i, task_asset_dict["full_cup"]["asset"],
                       task_asset_dict["empty_cup"]["asset"],
                       task_asset_dict["full_cup"]["start_pose"],
                       task_asset_dict["empty_cup"]["start_pose"])
        self._add_balls(env_ptr, i, task_asset_dict["ball"]["asset"],
                        task_asset_dict["ball"]["start_pose"])

    def _process_task_handles(self) -> None:
        self.ball_init_state = to_torch(
            self.ball_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, self.cfg["asset"]["numBalls"], 13)
        self.full_cup_init_state = to_torch(
            self.full_cup_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.full_cup_indices = to_torch(
            self.full_cup_indices, dtype=torch.long, device=self.device)
        self.empty_cup_indices = to_torch(
            self.empty_cup_indices, dtype=torch.long, device=self.device)

    def _load_cup_assets(self):
        """Loads the cups."""
        empty_cup_asset_options = gymapi.AssetOptions()
        empty_cup_asset_options.fix_base_link = True
        empty_cup_asset_options.use_mesh_materials = True
        empty_cup_asset_options.vhacd_enabled = True
        empty_cup_asset_options.vhacd_params = gymapi.VhacdParams()
        empty_cup_asset_options.vhacd_params.resolution = 1000000
        empty_cup_asset = self.gym.load_asset(
            self.sim, self.asset_root, self.empty_cup_asset_file,
            empty_cup_asset_options)
        full_cup_asset_options = gymapi.AssetOptions()
        full_cup_asset_options.fix_base_link = False
        full_cup_asset_options.use_mesh_materials = True
        full_cup_asset_options.vhacd_enabled = True
        full_cup_asset_options.vhacd_params = gymapi.VhacdParams()
        full_cup_asset_options.vhacd_params.resolution = 1000000
        full_cup_asset = self.gym.load_asset(
            self.sim, self.asset_root, self.full_cup_asset_file,
            full_cup_asset_options)

        x, y, z = self.cfg["asset"]["emptyCupPosition"]
        q_w, q_x, q_y, q_z = self.cfg["asset"]["emptyCupRotation"]
        empty_cup_start_pose = gymapi.Transform(
            p=gymapi.Vec3(x, y, z), r=gymapi.Quat(q_x, q_y, q_z, q_w))

        x, y, z = self.cfg["asset"]["fullCupPosition"]
        q_w, q_x, q_y, q_z = self.cfg["asset"]["fullCupRotation"]
        full_cup_start_pose = gymapi.Transform(
            p=gymapi.Vec3(x, y, z), r=gymapi.Quat(q_x, q_y, q_z, q_w))

        self.num_cups_bodies = self.gym.get_asset_rigid_body_count(
            empty_cup_asset) + self.gym.get_asset_rigid_body_count(
            full_cup_asset)
        self.num_cups_shapes = self.gym.get_asset_rigid_shape_count(
            empty_cup_asset) + self.gym.get_asset_rigid_shape_count(
            full_cup_asset)

        self.empty_cup_pos = to_torch(
            self.cfg["asset"]["emptyCupPosition"], device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return full_cup_asset, empty_cup_asset, full_cup_start_pose, \
               empty_cup_start_pose

    def _load_ball_assets(self):
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.fix_base_link = False

        ball_asset = self.gym.create_sphere(
            self.sim, self.cfg["asset"]["ballRadius"], ball_asset_options)

        ball_start_poses = []
        ball_init_room = 0.015
        ls_extent = int(math.ceil(self.cfg["asset"]["numBalls"] ** (1 / 3)))
        linspace = torch.linspace(-ball_init_room, ball_init_room, ls_extent)

        x_mid, y_mid, z_mid = self.cfg["asset"]["fullCupPosition"]
        for i, x in enumerate(linspace):
            for j, y in enumerate(linspace):
                for k, z in enumerate(linspace):
                    ball_idx = k + ls_extent * j + (ls_extent ** 2) * i
                    if ball_idx < self.cfg["asset"]["numBalls"]:
                        ball_start_poses.append(
                            gymapi.Transform(
                                p=gymapi.Vec3(x_mid + x, y_mid + y, z_mid + z))
                        )

        self.num_ball_bodies = self.cfg["asset"]["numBalls"] * \
                               self.gym.get_asset_rigid_body_count(ball_asset)
        self.num_ball_shapes = self.cfg["asset"]["numBalls"] * \
                               self.gym.get_asset_rigid_shape_count(ball_asset)

        return ball_asset, ball_start_poses

    @property
    def full_cup_asset_file(self) -> os.path:
        full_cup_asset_file = "urdf/ycb/025_mug.urdf"
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "fullCupAssetFile", full_cup_asset_file))
        return full_cup_asset_file

    @property
    def empty_cup_asset_file(self) -> os.path:
        empty_cup_asset_file = "urdf/ycb/024_bowl.urdf"
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "emptyCupAssetFile", empty_cup_asset_file))
        return empty_cup_asset_file

    def _begin_aggregate(self, env_ptr, env_idx: int) -> None:
        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies + \
                         self.num_cups_bodies + self.num_ball_bodies
        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes + \
                         self.num_cups_shapes + self.num_ball_shapes
        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)

    def _add_cups(self, env_ptr, i, full_cup_asset, empty_cup_asset,
                  full_cup_start_pose, empty_cup_start_pose) -> None:
        # create cup actors
        full_cup_actor = self.gym.create_actor(
            env_ptr, full_cup_asset, full_cup_start_pose, 'full_cup', i, 0, 0)
        empty_cup_actor = self.gym.create_actor(
            env_ptr, empty_cup_asset, empty_cup_start_pose, 'empty_cup', i, 0,
            0)

        self.full_cup_indices.append(self.gym.get_actor_index(
            env_ptr, full_cup_actor, gymapi.DOMAIN_SIM))
        self.empty_cup_indices.append(self.gym.get_actor_index(
            env_ptr, empty_cup_actor, gymapi.DOMAIN_SIM))

        self.full_cup_init_state.append(
            [full_cup_start_pose.p.x,
             full_cup_start_pose.p.y,
             full_cup_start_pose.p.z,
             full_cup_start_pose.r.x,
             full_cup_start_pose.r.y,
             full_cup_start_pose.r.z,
             full_cup_start_pose.r.w,
             0, 0, 0, 0, 0, 0])

        self.full_cups.append(full_cup_actor)
        self.empty_cups.append(empty_cup_actor)

    def _add_balls(self, env_ptr, i, ball_asset, ball_start_poses) -> None:
        # create ball actors
        for j in range(self.cfg["asset"]["numBalls"]):
            ball_actor = self.gym.create_actor(
                env_ptr, ball_asset, ball_start_poses[j], "ball_" + str(j))

            self.ball_init_state.append(
                [ball_start_poses[j].p.x,
                 ball_start_poses[j].p.y,
                 ball_start_poses[j].p.z,
                 ball_start_poses[j].r.x,
                 ball_start_poses[j].r.y,
                 ball_start_poses[j].r.z,
                 ball_start_poses[j].r.w,
                 0, 0, 0, 0, 0, 0])

            ball_idx = self.gym.get_actor_index(env_ptr, ball_actor,
                                                gymapi.DOMAIN_SIM)
            self.ball_indices[i, j] = ball_idx
            self.balls[i].append(ball_actor)

    def compute_reward(self) -> None:
        if self.cfg["control"]["useRelativeControl"]:
            actions = self.actions
        else:
            actions = self.actions - self.prev_actions
        self.prev_actions = self.actions

        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], \
        self.successes[:], self.success_rate[:], \
        self.balls_poured[:], self.num_balls_poured[:], \
        self.full_cup_in_empty_cup[:], \
        action_reward, \
        fingertips_to_full_cup_distance_reward, \
        pouring_reward, \
        self.rewards_dict, \
        = compute_cup_pouring_reward(
            self.rew_buf, self.reset_buf, self.progress_buf,
            self.successes, self.success_rate,
            self.full_cup_pos, self.empty_cup_pos, self.ball_pos,
            self.fingertip_pos, actions,
            self.cfg["reward"]["scale"]["actionPenalty"],
            self.cfg["reward"]["scale"]["fingertipsToFullCupDistanceReward"],
            self.max_episode_length,
            self.cfg["reward"]["scale"]["pouringReward"] / self.cfg["asset"]["numBalls"],
            self.cfg["reward"]["scale"]["pouredSuccessfully"],
            self.cfg["reward"]["epsFingertips"],
            self.cfg["reward"]["sparse"],
            self.cfg["reset"]["cupPoured"],
            self.cfg["reward"]["returnRewardsDict"]
        )

        if self.cfg["debug"]["verbose"]:
            self.writer.add_scalar(
                "action_reward",
                torch.mean(action_reward), self.env_steps)
            self.writer.add_scalar(
                "fingertips_to_full_cup_distance_reward",
                torch.mean(fingertips_to_full_cup_distance_reward),
                self.env_steps)
            self.writer.add_scalar(
                "pouring_reward",
                torch.mean(pouring_reward), self.env_steps)

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
            elif obs == "handDofPos":
                self.add_hand_dof_pos(obs_idx)
            elif obs == "handPose":
                self.add_hand_pose(obs_idx)
            elif obs == "jointPos":
                self.add_joint_pos(obs_idx)
            elif obs == "jointVel":
                self.add_joint_vel(obs_idx)
            elif obs == "previousAction":
                self.add_previous_action(obs_idx)
            elif obs == "fullCupPose":
                self.add_full_cup_pose(obs_idx)
            elif obs == "emptyCupPose":
                self.add_empty_cup_pose(obs_idx)
            obs_idx += self.num_obs_dict[obs]

    def _process_state_tensors(self) -> None:
        self.fingertip_state = self.rigid_body_states[
                               :, self.fingertip_handles][:, :, 0:13]
        self.fingertip_rigid_body_pos = self.rigid_body_states[
                                        :, self.fingertip_handles][:, :, 0:3]
        self.fingertip_rot = self.rigid_body_states[
                             :, self.fingertip_handles][:, :, 3:7]
        self.fingertip_pos = self.calculate_fingertip_pos()

        self.full_cup_pos = self.root_state_tensor[self.full_cup_indices, 0:3]
        self.full_cup_rot = self.root_state_tensor[self.full_cup_indices, 3:7]
        self.empty_cup_pos = self.root_state_tensor[self.empty_cup_indices, 0:3]
        self.empty_cup_rot = self.root_state_tensor[self.empty_cup_indices, 3:7]

        self.ball_pos = self.root_state_tensor[self.ball_indices, 0:3].view(
            self.num_envs, self.cfg["asset"]["numBalls"], 3)
        self.ball_rot = self.root_state_tensor[self.ball_indices, 3:7].view(
            self.num_envs, self.cfg["asset"]["numBalls"], 4)

    def add_full_cup_pose(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx + 3] = self.full_cup_pos
        self.obs_buf[:, start_idx + 3:start_idx + 7] = self.full_cup_rot
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("fullCupPose", torch.cat([self.full_cup_pos, self.full_cup_rot], dim=1))

    def add_empty_cup_pose(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx:start_idx + 3] = self.empty_cup_pos
        self.obs_buf[:, start_idx + 3:start_idx + 7] = self.empty_cup_rot
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("emptyCupPose", torch.cat([self.empty_cup_pos, self.empty_cup_rot], dim=1))    

    def reset_idx(self, env_ids):
        # Domain randomization would be added here ...
        if not self.initial_reset and not self.cfg["control"]["teleoperated"]:
            self._reset_robot(env_ids)
        else:
            self._reset_robot(env_ids, initial_sampling=True)
            # simulate for one time-step to update the tracker position
            self.render()
            self.gym.simulate(self.sim)
            self._refresh_state_tensors()

            self.initial_tracker_pose = self.tracker_pose
            self.initial_finger_angles = self.robot_dof_pos[:,
                                         self.hand_actuated_dof_indices]
        self._reset_full_cup(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_full_cup(self, env_ids):
        # Reset cup itself
        self.root_state_tensor[self.full_cup_indices[env_ids]] = \
            self.full_cup_init_state[env_ids].clone()

        # Reset balls
        self.root_state_tensor[self.ball_indices[env_ids].flatten()] = \
            self.ball_init_state[env_ids].view(-1, 13)

        # Set root state tensor of the simulation
        indices = torch.unique(
            torch.cat([self.full_cup_indices[env_ids],
                       self.ball_indices[env_ids].flatten()]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(indices), len(indices))

    def _draw_debug_visualization(self, i: int) -> None:
        super()._draw_debug_visualization(i)
        if self.cfg["debug"]["drawFullCupPose"]:
            self._draw_full_cup_pose(i)
        if self.cfg["debug"]["drawEmptyCupPose"]:
            self._draw_empty_cup_pose(i)
        if self.cfg["debug"]["drawBallPose"]:
            self._draw_ball_pose(i)
        if self.cfg["debug"]["colorBallsOnSuccess"]:
            self._color_balls_on_success(i)
        if self.cfg["debug"]["colorFullCupInEmptyCup"]:
            self._color_full_cup_in_empty_cup(i)

    def _draw_full_cup_pose(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.full_cup_pos[i],
                                     self.full_cup_rot[i])

    def _draw_empty_cup_pose(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.empty_cup_pos[i],
                                     self.empty_cup_rot[i])

    def _draw_ball_pose(self, i: int) -> None:
        for j in range(self.cfg["asset"]["numBalls"]):
            self._draw_coordinate_system(self.envs[i], self.ball_pos[i, j],
                                         self.ball_rot[i, j], axis_length=0.05)

    def _color_balls_on_success(self, i: int) -> None:
        for j in range(self.cfg["asset"]["numBalls"]):
            if self.balls_poured[i, j]:
                self.gym.set_rigid_body_color(self.envs[i], self.balls[i][j], 0,
                                              gymapi.MeshType.MESH_VISUAL,
                                              gymapi.Vec3(0.1, 0.7, 0.1))
            else:
                self.gym.set_rigid_body_color(self.envs[i], self.balls[i][j], 0,
                                              gymapi.MeshType.MESH_VISUAL,
                                              gymapi.Vec3(0.9, 0.1, 0.1))

    def _color_full_cup_in_empty_cup(self, i: int) -> None:
        if self.full_cup_in_empty_cup[i]:
            self.gym.set_rigid_body_color(self.envs[i], self.full_cups[i],
                                          0, gymapi.MeshType.MESH_VISUAL,
                                          gymapi.Vec3(0.9, 0.1, 0.1))
        else:
            self.gym.set_rigid_body_color(self.envs[i], self.full_cups[i],
                                          0, gymapi.MeshType.MESH_VISUAL,
                                          gymapi.Vec3(0.1, 0.6, 0.1))


@torch.jit.script
def compute_cup_pouring_reward(
    rew_buf, reset_buf, progress_buf, successes, success_rate,
    full_cup_pos, empty_cup_pos, ball_pos,
    fingertip_pos,  actions,
    action_penalty_scale: float,
    fingertips_to_full_cup_distance_reward_scale: float,
    max_episode_length: int,
    pouring_reward_scale: float,
    poured_successfully_reward_scale: float,
    eps_fingertips: float,
    sparse: bool,
    reset_when_cup_is_poured: bool,
    return_rewards_dict: bool,
):

    # ============ computing reward ... ============
    # penalize large actions: r ~ -|| a_t ||^2
    squared_action_norm = torch.sum(actions.pow(2), dim=-1)
    action_reward = squared_action_norm * action_penalty_scale

    # penalize fingertips-cup distance: r ~ (1 / (d(ft, handle)^2 + ϵ_ft))^2
    cup_center = full_cup_pos.unsqueeze(1).repeat(1, fingertip_pos.shape[1], 1)
    fingertips_to_cup_distance = torch.norm(fingertip_pos - cup_center, dim=-1)
    thumb_scaling = 2
    fingertips_to_cup_distance[:, 0] *= thumb_scaling
    fingertips_to_cup_distance = torch.sum(fingertips_to_cup_distance, dim=-1)
    fingertips_to_full_cup_distance_reward = \
        1.0 / (eps_fingertips + 5 * fingertips_to_cup_distance.pow(2))
    fingertips_to_full_cup_distance_reward *= fingertips_to_full_cup_distance_reward
    fingertips_to_full_cup_distance_reward *= fingertips_to_full_cup_distance_reward_scale

    # count balls poured into empty cup
    target_pos = empty_cup_pos.unsqueeze(1).repeat(1, ball_pos.shape[1], 1)
    balls_to_target_xy_dist = torch.norm(
        ball_pos[:, :, 0:2] - target_pos[:, :, 0:2], dim=2)
    balls_to_target_z_dist = torch.norm(
        ball_pos[:, :, 2:] - target_pos[:, :, 2:], dim=2)
    balls_poured = torch.logical_and(balls_to_target_xy_dist < 0.05,
                                     balls_to_target_z_dist < 0.02)
    num_balls_poured = torch.sum(balls_poured, dim=1)

    # check whether full cup is placed inside empty cup
    full_cup_to_empty_cup_xy_dist = torch.norm(
        full_cup_pos[:, 0:2] - empty_cup_pos[:, 0:2], dim=1)
    full_cup_to_empty_cup_z_dist = torch.norm(
        full_cup_pos[:, 2:] - empty_cup_pos[:, 2:], dim=1)

    full_cup_in_empty_cup = torch.logical_and(
        full_cup_to_empty_cup_xy_dist < 0.05,
        full_cup_to_empty_cup_z_dist < 0.03)

    # do not count balls poured when full cup is placed in empty cup
    num_balls_poured *= 1 - full_cup_in_empty_cup.long()

    min_success_rate = 0.9
    poured_successfully = (num_balls_poured / ball_pos.shape[
        1]) > min_success_rate

    # pouring reward: r ~ num_balls_in_target_cup
    pouring_reward = num_balls_poured * pouring_reward_scale

    # task completion reward: r ~ 1(poured_successfully == True)
    poured_successfully_reward = poured_successfully_reward_scale * poured_successfully.float()

    # compute final reward
    rewards_dict = {}
    if sparse:
        reward = poured_successfully.float() - 1
        if return_rewards_dict:
            dense_reward = action_reward + \
                fingertips_to_full_cup_distance_reward + \
                pouring_reward + poured_successfully_reward
            rewards_dict["sparse"] = reward
            rewards_dict["dense"] = dense_reward
    else:
        reward = action_reward + \
            fingertips_to_full_cup_distance_reward + \
            pouring_reward + poured_successfully_reward
        if return_rewards_dict:
            sparse_reward = poured_successfully.float() - 1
            rewards_dict["sparse"] = sparse_reward
            rewards_dict["dense"] = reward

    # ============ determining resets ... ============
    # reset environments that have reached the maximum number of steps
    resets = torch.where(progress_buf >= max_episode_length,
                         torch.ones_like(reset_buf), reset_buf)

    # reset environments that emptied the cup
    if reset_when_cup_is_poured:
        resets = torch.where(poured_successfully,
                             torch.ones_like(resets), resets)

    # determine total number of successful episodes and success rate
    total_resets = torch.sum(resets)
    avg_factor = 0.01
    successful_resets = torch.sum(poured_successfully.float())
    successes += poured_successfully.float()
    success_rate = torch.where(total_resets > 0,
                               avg_factor * (successful_resets/total_resets) +
                               (1. - avg_factor) * success_rate,
                               success_rate)

    return reward, resets, progress_buf, \
           successes, success_rate, \
           balls_poured, num_balls_poured, \
           full_cup_in_empty_cup, \
           action_reward, \
           fingertips_to_full_cup_distance_reward, \
           pouring_reward, \
           rewards_dict
