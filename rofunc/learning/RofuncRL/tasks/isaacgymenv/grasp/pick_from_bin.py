#import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from typing import *
import random
from .empty_bin import EmptyBin
from .lift_object import compute_object_lifting_reward


class PickFromBin(EmptyBin):
    def __init__(self,
                 cfg: Dict[str, Any],
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool) -> None:
        """Creates UR5e-SIH robot on the task of picking a specific object from
        a bin filled with various objects. The agent is only rewarded for
        lifting the goal object, which is shown above each environment.

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

    def _load_task_assets(self) -> Dict[str, Any]:
        self.objects_in_each_env = []
        self.goals = [[] for _ in range(self.num_envs)]
        self.goal_indices = torch.zeros(
            [self.num_envs, self.cfg["task"]["numObjectsPerBin"]],
            dtype=torch.long, device=self.device)
        self.goal_object_numbers = torch.zeros(self.num_envs,
                                               device=self.device,
                                               dtype=torch.long)

        assets_dict = super()._load_task_assets()
        assets_dict["goal"] = {}
        goal_assets, goal_start_poses = self._load_goal_assets()
        assets_dict["goal"]["asset"] = goal_assets
        assets_dict["goal"]["start_pose"] = goal_start_poses
        return assets_dict

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any],
                  agg_info) -> None:
        self.objects_in_each_env.append(agg_info)
        super()._add_task(env_ptr, i, task_asset_dict, agg_info)
        self._add_goals(env_ptr, i, task_asset_dict["goal"]["asset"],
                        agg_info)

    def _process_task_handles(self) -> None:
        super()._process_task_handles()
        self.objects_in_each_env = to_torch(
            self.objects_in_each_env, dtype=torch.long, device=self.device)

        x, y, z = self.cfg["asset"]["goalObjectPosition"]
        self.active_goal_state = to_torch(
            [[x, y, z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], device=self.device)

        self.passive_goal_state = to_torch(
            [[x, y, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], device=self.device)
        self.object_identifiers = torch.arange(
            self.cfg["task"]["numObjectsPerBin"]).unsqueeze(0).repeat(
            self.num_envs, 1).to(self.device).unsqueeze(2)

    def _load_goal_assets(self):
        """Loads visualizations for the target objects to be picked."""
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True

        goal_assets, goal_start_poses = [], []

        for ycb_object in self.ycb_objects:
            goal_asset_file = os.path.join(
                self.ycb_object_asset_root + "/visual",
                os.path.normpath(ycb_object + ".urdf"))
            goal_start_poses.append(gymapi.Transform(
                p=gymapi.Vec3(0, 0, -1)))
            goal_assets.append(self.gym.load_asset(
                self.sim, self.asset_root, goal_asset_file,
                object_asset_options))
        for egad_object in self.egad_objects:
            goal_asset_file = os.path.join(
                self.egad_object_asset_root + "/visual",
                os.path.normpath(egad_object + ".urdf"))
            goal_start_poses.append(gymapi.Transform(
                p=gymapi.Vec3(0, 0, -1)))
            goal_assets.append(self.gym.load_asset(
                self.sim, self.asset_root, goal_asset_file,
                object_asset_options))
        return goal_assets, goal_start_poses

    def _add_goals(self, env_ptr, i, goal_assets, object_indices) -> None:
        for j, object_idx in enumerate(object_indices):
            goal_start_pose = gymapi.Transform(p=gymapi.Vec3(0, 0, -1))
            goal_actor = self.gym.create_actor(env_ptr, goal_assets[object_idx],
                                               goal_start_pose,
                                               self.objects_list[object_idx] + "_goal",
                                               -1, 0, 0)

            # Set color of EGAD goals
            if self.objects_list[object_idx] in self.egad_objects:
                self.gym.set_rigid_body_color(
                    env_ptr, goal_actor, 0, gymapi.MeshType.MESH_VISUAL,
                    gymapi.Vec3(*self.egad_colors[
                        self.objects_list[object_idx]]))

            goal_idx = self.gym.get_actor_index(env_ptr, goal_actor,
                                                gymapi.DOMAIN_SIM)
            self.goal_indices[i, j] = goal_idx
            self.goals[i].append(goal_actor)

    def _begin_aggregate(self, env_ptr, env_idx: int) -> List:
        possible_objects = list(range(self.num_different_objects))
        object_indices = random.sample(possible_objects,
                                       self.cfg["task"]["numObjectsPerBin"])

        # Add numObjectsPerBin bodies for the goals in each env
        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies + \
                         + self.num_bin_bodies + \
                         self.cfg["task"]["numObjectsPerBin"]

        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes + \
                         self.num_bin_shapes

        for object_idx in object_indices:
            max_agg_bodies += self.num_object_bodies[object_idx]
            max_agg_shapes += self.num_object_shapes[object_idx]

        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)
        return object_indices

    def _reset_objects(self, env_ids) -> None:
        self._reset_goals(env_ids, set_root_state_tensor=False)
        self.root_state_tensor[self.object_indices[env_ids].flatten()] = self.object_init_state[env_ids].view(-1, 13)
        indices = torch.unique(
            torch.cat([self.object_indices[env_ids].flatten(), self.goal_indices[env_ids].flatten()]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(indices), len(indices))

    def _end_of_drop_objects(self, env_ids) -> None:
        self._reset_goals(env_ids)

    def _drop_objects_into_done_bin(self, env_ids, skip_steps) -> None:
        pass

    def _reset_goals(self, env_ids, set_root_state_tensor=True) -> None:
        object_numbers = torch.randint(self.cfg["task"]["numObjectsPerBin"],
                                       (len(env_ids),)).to(self.device)
        self.goal_object_numbers[env_ids] = object_numbers
        goal_mask = torch.nn.functional.one_hot(
            object_numbers, num_classes=self.cfg["task"]["numObjectsPerBin"]
        ).to(self.device).bool()

        goal_object_indices = torch.masked_select(self.objects_in_each_env[env_ids], goal_mask)
        active_goals = torch.masked_select(self.goal_indices[env_ids], goal_mask)
        passive_goals = torch.masked_select(self.goal_indices[env_ids], ~goal_mask)

        self.root_state_tensor[active_goals] = self.active_goal_state.repeat(len(active_goals), 1)
        self.root_state_tensor[passive_goals] = self.passive_goal_state.repeat(len(passive_goals), 1)

        if self.cfg["debug"]["verbose"] and not self.headless:
            for i, num in enumerate(goal_object_indices):
                print(f"Pick {self.objects_list[int(num)]} from Env {i}.")

        if set_root_state_tensor:
            indices = torch.unique(
                torch.cat([self.goal_indices[env_ids].flatten()]).to(
                    torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(indices), len(indices))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_observations(self):
        """Updates the state tensors and adds the desired information to the
        observation buffer."""
        self._refresh_state_tensors()
        self._process_state_tensors()

        obs_idx = self.add_object_obs(self.object_obs)

        for obs in self.robot_obs:
            if obs == "fingertipContactForces":
                self.add_fingertip_contact_forces(obs_idx)
            elif obs == "fingertipPos":
                self.add_fingertip_pos(obs_idx)
            elif obs == "goalIdentifier":
                self.add_goal_identifier(obs_idx)
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
            obs_idx += self.num_obs_dict[obs]

    def add_goal_identifier(self, start_idx: int) -> None:
        self.obs_buf[:, start_idx] = self.goal_object_numbers
        if self.cfg["task"]["returnObsDict"]:
            self.add_separate_obs("goalIdentifier", self.goal_object_numbers)

    def add_object_obs(self, object_obs: List) -> int:
        obs_idx = 0
        obj_obs = []

        def add_to_obs(obs_list, key, obs) -> List:
            obs_list.append(obs)
            if self.cfg["task"]["returnObsDict"]:
                self.add_separate_obs(key, obs)
            return obs_list

        for obs_key in object_obs:
            if obs_key == "objectPos":
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_pos)
            elif obs_key == "objectLinVel":
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_linvel)
            elif obs_key == "objectRot":
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_rot)
            elif obs_key == "objectAngVel":
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_angvel)
            elif obs_key == "objectGoalOneHot":
                goal_one_hot = torch.nn.functional.one_hot(
                    self.goal_object_numbers,
                    num_classes=self.cfg["task"]["numObjectsPerBin"]
                ).unsqueeze(2).to(self.device)
                obj_obs = add_to_obs(obj_obs, obs_key, goal_one_hot)
            elif obs_key == "objectIdentifier":
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_identifiers)
            elif obs_key == "objectBboxBounds":  # min, max corners
                obj_obs = add_to_obs(obj_obs, obs_key,
                                     self.object_bbox_bounds.flatten(2, 3))
            elif obs_key == "objectBboxCorners":  # 8 3D corner points
                obj_obs = add_to_obs(obj_obs, obs_key,
                                     self.object_bbox_corners.flatten(2, 3))
            elif obs_key == "objectBboxPose":  # Center pose of bounding box
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_bbox_pose)
            elif obs_key == "objectSurfaceSamples":  # Samples on mesh surface
                obj_obs = add_to_obs(obj_obs, obs_key,
                                     self.object_surface_samples.flatten(2, 3))

            obs_idx += self.num_obs_dict[obs_key]
        obj_obs = torch.cat(obj_obs, dim=2)
        self.obs_buf[:, 0:obs_idx] = obj_obs.view(self.num_envs, -1)
        return obs_idx

    def compute_reward(self) -> None:
        self.goal_object_pos = self.object_pos.gather(
            1, self.goal_object_numbers.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.goal_object_linvel = self.object_linvel.gather(
            1, self.goal_object_numbers.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.goal_object_init_pos = self.object_init_pos.gather(
            1, self.goal_object_numbers.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)

        if self.cfg["control"]["useRelativeControl"]:
            actions = self.actions
        else:
            actions = self.actions - self.prev_actions
        self.prev_actions = self.actions

        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], \
        self.successes[:], self.success_rate[:], \
        action_reward, \
        fingertips_to_object_distance_reward, \
        object_height_reward, \
        object_velocity_reward, \
        target_height_reward, \
        self.rewards_dict, \
            = compute_object_lifting_reward(
            self.rew_buf, self.reset_buf, self.progress_buf,
            self.successes, self.success_rate,
            self.goal_object_pos, self.goal_object_linvel,
            self.goal_object_init_pos,
            self.fingertip_pos, actions,
            self.cfg["reward"]["scale"]["actionPenalty"],
            self.cfg["reward"]["scale"]["fingertipsToObjectDistanceReward"],
            self.max_episode_length,
            self.cfg["reward"]["scale"]["objectHeightReward"],
            self.cfg["reward"]["scale"]["objectVelocityReward"],
            self.cfg["reward"]["scale"]["targetHeightReward"],
            self.cfg["reward"]["scale"]["objectFallsOffTablePenalty"],
            self.cfg["reward"]["liftOffHeight"],
            self.cfg["reward"]["maxXyDrift"],
            self.cfg["reward"]["targetHeight"],
            self.cfg["reward"]["epsFingertips"],
            self.cfg["reward"]["epsHeight"],
            torch.Tensor(self.cfg["reward"]["xyCenter"]).to(
                self.device).unsqueeze(0).repeat(self.num_envs, 1),
            self.cfg["reward"]["sparse"],
            self.cfg["reset"]["objectFallsOffTable"],
            self.cfg["reset"]["objectLifted"],
            self.cfg["reward"]["returnRewardsDict"]
        )

        if self.cfg["debug"]["verbose"]:
            self.writer.add_scalar(
                "action_reward",
                torch.mean(action_reward), self.env_steps)
            self.writer.add_scalar(
                "fingertips_to_object_distance_reward",
                torch.mean(fingertips_to_object_distance_reward),
                self.env_steps)
            self.writer.add_scalar(
                "object_height_reward",
                torch.mean(object_height_reward), self.env_steps)
            self.writer.add_scalar(
                "object_velocity_reward",
                torch.mean(object_velocity_reward), self.env_steps)
            self.writer.add_scalar(
                "target_height_reward",
                torch.mean(target_height_reward), self.env_steps)

        self.writer.add_scalar("successes", torch.sum(self.successes),
                               self.env_steps)
        self.writer.add_scalar("success_rate", self.success_rate,
                               self.env_steps)

    def _draw_hand_object_distance(self, i: int) -> None:
        hand_pos = self.hand_pos[i].cpu().numpy()
        object_pos = self.goal_object_pos[i].cpu().numpy()
        self.gym.add_lines(self.viewer, self.envs[i], 1,
                           [object_pos[0], object_pos[1], object_pos[2],
                            hand_pos[0], hand_pos[1], hand_pos[2]],
                           [0.85, 0.1, 0.85])
