import os
from torch import Tensor
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
from typing import *
import random
from gym_grasp.tasks.base.grasping_task import GraspingTask
from .lift_object import LiftObject
import math


class EmptyBin(LiftObject):
    def __init__(self,
                 cfg: Dict[str, Any],
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool) -> None:
        """Creates UR5e-SIH robot on the task of emptying a bin filled with
        various objects. The agent is rewarded for every object removed from the
        bin and objects removed successfully are stored in a done bin, until all
        objects are cleared. This is similar to the way the lifting policy is
        trained in the tasks in QT-Opt on the scenario where objects are not put
        back.

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

    def _parse_observation_type(self, observation_type,
                                num_observations) -> Tuple[List, List]:
        # Adjust dimensionality to the number of objects per bin
        for obs in num_observations.keys():
            if obs.startswith("object"):
                num_observations[obs] *= self.cfg["task"]["numObjectsPerBin"]
        return super()._parse_observation_type(observation_type,
                                               num_observations)

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.remaining_objects = torch.ones(
            self.num_envs, self.cfg["task"]["numObjectsPerBin"],
            device=self.device, dtype=torch.long)
        self.num_remaining_objects = self.cfg["task"]["numObjectsPerBin"] * \
                                     torch.ones(self.num_envs,
                                                device=self.device,
                                                dtype=torch.long)

        self.prev_remaining_objects = self.remaining_objects.clone()

    def _load_task_assets(self) -> Dict[str, Any]:
        assets_dict = {}
        assets_dict["object"] = super()._load_task_assets()
        self.object_indices = torch.zeros(
            [self.num_envs, self.cfg["task"]["numObjectsPerBin"]],
            dtype=torch.long, device=self.device)
        self.object_cleared_state = []
        self.objects = [[] for _ in range(self.num_envs)]
        self.object_ids = []

        active_bin_asset, done_bin_asset, active_bin_start_pose, \
        done_bin_start_pose = self._load_bin_assets()
        assets_dict["active_bin"] = {}
        assets_dict["done_bin"] = {}
        assets_dict["active_bin"]["asset"] = active_bin_asset
        assets_dict["active_bin"]["start_pose"] = active_bin_start_pose
        assets_dict["done_bin"]["asset"] = done_bin_asset
        assets_dict["done_bin"]["start_pose"] = done_bin_start_pose
        return assets_dict

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any], agg_info) -> None:
        self._add_objects(env_ptr, i, task_asset_dict["object"]["asset"],
                          task_asset_dict["object"]["start_pose"], agg_info)
        self._add_bins(env_ptr, i, task_asset_dict["active_bin"]["asset"],
                       task_asset_dict["done_bin"]["asset"],
                       task_asset_dict["active_bin"]["start_pose"],
                       task_asset_dict["done_bin"]["start_pose"])

    def _process_task_handles(self) -> None:
        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, self.cfg["task"]["numObjectsPerBin"], 13)
        self.object_cleared_state = to_torch(
            self.object_cleared_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, self.cfg["task"]["numObjectsPerBin"], 13)

    def _load_bin_assets(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        # Enable convex decomposition
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000

        active_bin_asset = self.gym.load_asset(
            self.sim, self.asset_root, self.active_bin_asset_file,
            asset_options)

        done_bin_asset = self.gym.load_asset(
            self.sim, self.asset_root, self.done_bin_asset_file,
            asset_options)

        active_x, active_y, active_z = self.cfg["asset"]["activeBinPosition"]
        done_x, done_y, done_z = self.cfg["asset"]["doneBinPosition"]
        active_bin_start_pose = gymapi.Transform(p=gymapi.Vec3(
            active_x, active_y, active_z))
        done_bin_start_pose = gymapi.Transform(p=gymapi.Vec3(
            done_x, done_y, done_z), r=gymapi.Quat(0, 0, 0, 1))

        self.num_bin_bodies = self.gym.get_asset_rigid_body_count(
            active_bin_asset) + self.gym.get_asset_rigid_body_count(
            done_bin_asset)
        self.num_bin_shapes = self.gym.get_asset_rigid_shape_count(
            active_bin_asset) + self.gym.get_asset_rigid_shape_count(
            done_bin_asset)
        return active_bin_asset, done_bin_asset, active_bin_start_pose, \
               done_bin_start_pose

    @property
    def active_bin_asset_file(self) -> os.path:
        bin_asset_file = "urdf/bin.urdf"
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "activeBinAssetFile", bin_asset_file))
        return bin_asset_file

    @property
    def done_bin_asset_file(self) -> os.path:
        bin_asset_file = "urdf/bin.urdf"
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "doneBinAssetFile", bin_asset_file))
        return bin_asset_file

    def _begin_aggregate(self, env_ptr, env_idx: int) -> List:
        possible_objects = list(range(self.num_different_objects))
        object_indices = random.sample(possible_objects,
                                       self.cfg["task"]["numObjectsPerBin"])

        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies + \
                         + self.num_bin_bodies

        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes + \
                         self.num_bin_shapes

        for object_idx in object_indices:
            max_agg_bodies += self.num_object_bodies[object_idx]
            max_agg_shapes += self.num_object_shapes[object_idx]

        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)
        return object_indices

    def _add_objects(self, env_ptr, i, object_assets,
                     object_start_poses, object_indices,
                     min_height: float = 0.9,
                     height_diff: float = 0.1) -> None:
        self.object_ids.append(object_indices)
        for j, object_idx in enumerate(object_indices):
            x, y, _ = self.cfg["asset"]["activeBinPosition"]
            z = min_height + height_diff * j
            object_start_pose = gymapi.Transform(p=gymapi.Vec3(x, y, z))
            object_actor = self.gym.create_actor(env_ptr,
                                                 object_assets[object_idx],
                                                 object_start_pose,
                                                 self.objects_list[object_idx],
                                                 i, 0, 0)

            # Set color of EGAD objects
            if self.objects_list[object_idx] in self.egad_objects:
                self.gym.set_rigid_body_color(
                    env_ptr, object_actor, 0, gymapi.MeshType.MESH_VISUAL,
                    gymapi.Vec3(*self.egad_colors[
                        self.objects_list[object_idx]]))

            self.object_init_state.append(
                [object_start_pose.p.x,
                 object_start_pose.p.y,
                 object_start_pose.p.z,
                 object_start_pose.r.x,
                 object_start_pose.r.y,
                 object_start_pose.r.z,
                 object_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.object_cleared_state.append(
                [self.cfg["asset"]["doneBinPosition"][0],
                 self.cfg["asset"]["doneBinPosition"][1],
                 object_start_pose.p.z,
                 object_start_pose.r.x,
                 object_start_pose.r.y,
                 object_start_pose.r.z,
                 object_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            object_idx = self.gym.get_actor_index(env_ptr, object_actor,
                                                  gymapi.DOMAIN_SIM)
            self.object_indices[i, j] = object_idx
            self.objects[i].append(object_actor)

    def _add_bins(self, env_ptr, i, active_bin_asset, done_bin_asset,
                  active_bin_start_pose, done_bin_start_pose) -> None:
        if self.cfg["asset"]["useActiveBin"]:
            active_bin_actor = self.gym.create_actor(env_ptr, active_bin_asset,
                                                     active_bin_start_pose,
                                                     "active_bin", i, -1, 1)

        done_bin_actor = self.gym.create_actor(env_ptr, done_bin_asset,
                                               done_bin_start_pose, "done_bin",
                                               i, -1, 1)

        # change color of done bin to green
        self.gym.set_rigid_body_color(env_ptr, done_bin_actor, 0,
                                      gymapi.MeshType.MESH_VISUAL,
                                      gymapi.Vec3(0.05, 0.65, 0.05))

    def compute_reward(self) -> None:
        if self.cfg["control"]["useRelativeControl"]:
            actions = self.actions
        else:
            actions = self.actions - self.prev_actions
        self.prev_actions = self.actions

        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], \
        self.successes[:], self.success_rate[:], \
        self.remaining_objects[:], self.num_remaining_objects[:], \
        action_reward, \
        fingertips_to_object_distance_reward, \
        object_height_reward, \
        target_height_reward, \
        self.rewards_dict, \
            = compute_bin_picking_reward(
            self.rew_buf, self.reset_buf, self.progress_buf,
            self.successes, self.success_rate, self.remaining_objects,
            self.object_pos, self.object_linvel, self.object_init_pos,
            self.fingertip_pos, actions,
            self.cfg["reward"]["scale"]["actionPenalty"],
            self.cfg["reward"]["scale"]["fingertipsToObjectDistanceReward"],
            self.max_episode_length,
            self.cfg["reward"]["scale"]["objectHeightReward"],
            self.cfg["reward"]["scale"]["targetHeightReward"],
            self.cfg["reward"]["scale"]["objectFallsOffTablePenalty"],
            self.cfg["reward"]["liftOffHeight"],
            self.cfg["reward"]["maxXyDrift"],
            self.cfg["reward"]["targetHeight"],
            self.cfg["reward"]["epsFingertips"],
            self.cfg["reward"]["epsHeight"],
            torch.Tensor(self.cfg["reward"]["xyCenter"]).to(
                self.device).unsqueeze(0).repeat(self.num_envs, 1).unsqueeze(
                1).repeat(1, self.cfg["task"]["numObjectsPerBin"], 1),
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
                "target_height_reward",
                torch.mean(target_height_reward), self.env_steps)

        self.writer.add_scalar("successes", torch.sum(self.successes),
                               self.env_steps)
        self.writer.add_scalar("success_rate", self.success_rate,
                               self.env_steps)

    def _process_state_tensors(self) -> None:
        GraspingTask._process_state_tensors(self)
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3].view(
            self.num_envs, self.cfg["task"]["numObjectsPerBin"], 3)
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7].view(
            self.num_envs, self.cfg["task"]["numObjectsPerBin"], 4)
        self.object_linvel = self.root_state_tensor[self.object_indices,
                             7:10].view(self.num_envs,
                                        self.cfg["task"]["numObjectsPerBin"], 3)
        self.object_angvel = self.root_state_tensor[self.object_indices,
                             10:13].view(self.num_envs,
                                         self.cfg["task"]["numObjectsPerBin"],
                                         3)
        if any([o.startswith("objectBbox") for o in self.object_obs]):
            self.calculate_object_bbox()
        if "objectSurfaceSamples" in self.object_obs:
            self.calculate_object_surface_samples()

    def calculate_object_bbox(self) -> None:
        if not hasattr(self, "object_bbox_extent"):
            self.init_object_bbox()

        bbox_pos = self.object_pos + quat_apply(
            self.object_rot, self.object_bbox_pos_offset)
        bbox_rot = quat_mul(self.object_rot, self.object_bbox_rot_offset)
        self.object_bbox_pose = torch.cat([bbox_pos, bbox_rot], dim=2)
        self.object_bbox_corners = bbox_pos.unsqueeze(2).repeat(1, 1, 8, 1) + \
            quat_apply(
                bbox_rot.unsqueeze(2).repeat(1, 1, 8, 1),
                self.bbox_corner_coords * self.object_bbox_extent.unsqueeze(
                    2).repeat(1, 1, 8, 1))
        self.object_bbox_bounds = torch.stack(
            [self.object_bbox_corners[:, :, 0],
             self.object_bbox_corners[:, :, -1]],
            dim=2)

    def init_object_bbox(self) -> None:
        from scipy.spatial.transform import Rotation as R
        self.object_bbox_extent = []
        self.object_bbox_pos_offset = []
        self.object_bbox_rot_offset = []
        for env_id in range(self.num_envs):
            self.object_bbox_extent.append([])
            self.object_bbox_pos_offset.append([])
            self.object_bbox_rot_offset.append([])
            for obj_id in range(self.cfg["task"]["numObjectsPerBin"]):
                self.object_bbox_extent[env_id].append(
                    self.object_assets_dict["bbox"][self.object_ids[env_id][obj_id]][1])
                to_origin = self.object_assets_dict["bbox"][
                    self.object_ids[env_id][obj_id]][0]
                from_origin = np.linalg.inv(to_origin)
                r = R.from_matrix(from_origin[0:3, 0:3])
                r_quat = r.as_quat()
                translation = np.array(
                    [from_origin[0, 3], from_origin[1, 3], from_origin[2, 3]])
                self.object_bbox_pos_offset[env_id].append(translation)
                self.object_bbox_rot_offset[env_id].append(r_quat)

        self.object_bbox_extent = torch.from_numpy(
            np.array(self.object_bbox_extent)).to(self.device).float()
        self.object_bbox_pos_offset = torch.from_numpy(
            np.array(self.object_bbox_pos_offset)).to(self.device).float()
        self.object_bbox_rot_offset = torch.from_numpy(
            np.array(self.object_bbox_rot_offset)).to(self.device).float()

        self.bbox_corner_coords = torch.tensor(
            [[[[-0.5, -0.5, -0.5],
               [-0.5, -0.5, 0.5],
               [-0.5, 0.5, -0.5],
               [-0.5, 0.5, 0.5],
               [0.5, -0.5, -0.5],
               [0.5, -0.5, 0.5],
               [0.5, 0.5, -0.5],
               [0.5, 0.5, 0.5]]]],
            device=self.device).repeat(
            self.num_envs, self.cfg["task"]["numObjectsPerBin"], 1, 1)

    def calculate_object_surface_samples(self) -> None:
        if not hasattr(self, "object_surface_sample_offset_pos"):
            self.init_object_surface_samples()
        object_pos = self.object_pos.unsqueeze(2).repeat(
            1, 1, self.object_surface_sample_offset_pos.shape[2], 1)
        object_rot = self.object_rot.unsqueeze(2).repeat(
            1, 1, self.object_surface_sample_offset_pos.shape[2], 1)
        self.object_surface_samples = object_pos + quat_apply(
            object_rot, self.object_surface_sample_offset_pos)

    def init_object_surface_samples(self) -> None:
        self.object_surface_sample_offset_pos = []
        for env_id in range(self.num_envs):
            self.object_surface_sample_offset_pos.append([])
            for obj_id in range(self.cfg["task"]["numObjectsPerBin"]):
                self.object_surface_sample_offset_pos[env_id].append(
                    self.object_assets_dict["samples"][
                        self.object_ids[env_id][obj_id]])
        self.object_surface_sample_offset_pos = torch.from_numpy(
            np.array(self.object_surface_sample_offset_pos)).to(
            self.device).float()

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

    def _drop_objects(self, env_ids, skip_steps: int = 250) -> None:
        assert torch.equal(env_ids,
                           to_torch(list(range(self.num_envs)),
                                    device=self.device, dtype=torch.long)), \
            "Objects must be dropped in all environments simultaneously, as " \
            "this is a distinct phase where the physics simulation is " \
            "stepped before the RL agent can actually interact with the task."

        self._drop_objects_into_done_bin(env_ids, skip_steps)
        self._drop_objects_into_active_bin(env_ids, skip_steps)
        self._end_of_drop_objects(env_ids)
        self._objects_dropped = True

    def _drop_objects_into_active_bin(self, env_ids, skip_steps) -> None:
        self._drop_objects_into_bin(
            env_ids, self.cfg["asset"]["activeObjectDropPos"], skip_steps)
        self._set_object_init_state(env_ids, active=True)

    def _drop_objects_into_done_bin(self, env_ids, skip_steps) -> None:
        self._drop_objects_into_bin(
            env_ids, self.cfg["asset"]["doneObjectDropPos"], skip_steps)
        self._set_object_init_state(env_ids, active=False)

    def _drop_objects_into_bin(self, env_ids, drop_pos, skip_steps) -> None:
        for obj_idx in range(self.cfg["task"]["numObjectsPerBin"]):
            # Select which objects to drop
            active_object_indices = self.object_indices[:, obj_idx].flatten()
            passive_object_indices = self.object_indices[:, obj_idx+1:].flatten()
            object_indices = torch.cat([active_object_indices,
                                        passive_object_indices])

            # Sample states for object being dropped
            active_object_state = self._sample_active_object_state(
                active_object_indices, drop_pos)

            num_passive_objects = len(passive_object_indices) / self.num_envs
            linspace = torch.linspace(-self.cfg["env"]["envSpacing"],
                                      self.cfg["env"]["envSpacing"],
                                      int(math.ceil(math.sqrt(
                                          num_passive_objects))) + 2)[1:-1]

            passive_object_state = to_torch(
                [[1, 0, 0.1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], device=self.device
            ).repeat(int(len(passive_object_indices) / self.num_envs), 1)

            for i, x in enumerate(linspace):
                for j, y in enumerate(linspace):
                    obj_idx = int(math.ceil(math.sqrt(num_passive_objects))) * i + j
                    if obj_idx < passive_object_state.shape[0]:
                        passive_object_state[obj_idx][0] = x
                        passive_object_state[obj_idx][1] = y

            passive_object_state = passive_object_state.repeat(self.num_envs, 1)

            self._position_objects_for_drop(
                active_object_indices, passive_object_indices,
                active_object_state, passive_object_state)

            # Set root state tensor
            object_indices = torch.unique(
                torch.cat([object_indices]).to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(object_indices), len(object_indices))

            # Run simulation
            for _ in range(skip_steps):
                self.render()
                self.gym.simulate(self.sim)

    def _sample_active_object_state(self, active_object_indices, drop_pos) -> Tensor:
        active_rand_floats = torch_rand_float(
            -1.0, 1.0, (len(active_object_indices), 5), device=self.device)
        active_object_state = torch.zeros(len(active_object_indices), 13,
                                          device=self.device)
        active_object_state[:, 0] = drop_pos[0] + \
                                    self.cfg["initState"]["noise"][
                                        "objectPos"] * active_rand_floats[:, 0]
        active_object_state[:, 1] = drop_pos[1] + \
                                    self.cfg["initState"]["noise"][
                                        "objectPos"] * active_rand_floats[:, 1]
        active_object_state[:, 2] = drop_pos[2] + \
                                    self.cfg["initState"]["noise"][
                                        "objectPos"] * active_rand_floats[:, 2]
        active_object_rot = randomize_rotation(
            active_rand_floats[:, 3], active_rand_floats[:, 4],
            self.x_unit_tensor[0].unsqueeze(0).repeat(
                len(active_object_indices), 1),
            self.y_unit_tensor[0].unsqueeze(0).repeat(
                len(active_object_indices), 1))
        active_object_state[:, 3:7] = active_object_rot
        return active_object_state

    def _end_of_drop_objects(self, env_ids) -> None:
        pass

    def _not_in_active_bin(self, object_init_state) -> Tensor:
        fallen_objects = object_init_state[:, :, 0] < self.cfg["asset"]["activeBinPosition"][0] - 0.3
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 0] > self.cfg["asset"]["activeBinPosition"][0] + 0.3)
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 1] < self.cfg["asset"]["activeBinPosition"][1] - 0.175)
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 1] > self.cfg["asset"]["activeBinPosition"][1] + 0.175)
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 2] < 0.8)
        return fallen_objects

    def _not_in_done_bin(self, object_init_state) -> Tensor:
        fallen_objects = object_init_state[:, :, 0] < self.cfg["asset"]["doneBinPosition"][0] - 0.3
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 0] > self.cfg["asset"]["doneBinPosition"][0] + 0.3)
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 1] < self.cfg["asset"]["doneBinPosition"][1] - 0.175)
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 1] > self.cfg["asset"]["doneBinPosition"][1] + 0.175)
        fallen_objects = torch.logical_or(fallen_objects,
                                          object_init_state[:, :, 2] < 0.8)
        return fallen_objects

    def _set_object_init_state(self, env_ids, active: bool) -> None:
        self.gym.refresh_actor_root_state_tensor(self.sim)
        object_init_state = self.root_state_tensor[self.object_indices[env_ids]]
        if active:
            fallen_objects = self._not_in_active_bin(object_init_state)
        else:
            fallen_objects = self._not_in_done_bin(object_init_state)

        num_redropping_attempts = 5
        for attempt in range(num_redropping_attempts):
            if torch.any(fallen_objects):
                print(f"Dropping fallen objects again (Attempt {attempt}).")
                if active:
                    fallen_objects, object_init_state = self._redrop_objects(
                        fallen_objects,
                        self.cfg["asset"]["activeObjectDropPos"], active=True)
                else:
                    fallen_objects, object_init_state = self._redrop_objects(
                        fallen_objects, self.cfg["asset"]["doneObjectDropPos"],
                        active=False)

        if torch.any(fallen_objects):
            assert False, "Not all objects were positioned correctly."

        if active:
            self.init_root_state_tensor = self.root_state_tensor.clone()
            self.object_init_state = object_init_state
            self.object_init_pos = self.object_init_state[:, :, 0:3]
        else:
            self.object_cleared_state = object_init_state

    def _redrop_objects(self, fallen_objects, drop_pos, skip_steps=250,
                        active=True):
        num_fallen_objects = torch.sum(fallen_objects.int(), dim=1)
        env_nums = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        fallen_object_indices = -torch.ones(self.num_envs, torch.max(num_fallen_objects) + 1, dtype=torch.long, device=self.device)
        curr_fallen_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        indices = torch.stack([env_nums, curr_fallen_idx])

        for obj_idx in range(self.cfg["task"]["numObjectsPerBin"]):
            if torch.any(fallen_objects[:, obj_idx]):
                fallen_object_indices[tuple(indices)] = torch.where(
                    fallen_objects[:, obj_idx],
                    torch.ones_like(fallen_object_indices[tuple(indices)]) * obj_idx,
                    torch.ones_like(fallen_object_indices[tuple(indices)]) * -1)
                indices[1] += fallen_objects[:, obj_idx].int()

        for fallen_idx in range(fallen_object_indices.shape[1] - 1):
            active_object_indices = []
            for env_id, idx in enumerate(fallen_object_indices[:, fallen_idx]):
                if idx >= 0:
                    active_object_indices.append(self.object_indices[env_id, idx])

            active_object_indices = torch.Tensor(active_object_indices).long().to(self.device)

            # Sample states for object being dropped
            active_object_state = self._sample_active_object_state(
                active_object_indices, drop_pos)

            self._position_objects_for_drop(
                active_object_indices, torch.Tensor([]).long(),
                active_object_state, torch.Tensor([]))

            # Set root state tensor
            object_indices = torch.unique(
                torch.cat([active_object_indices]).to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(object_indices), len(object_indices))

            # Run simulation
            for _ in range(skip_steps):
                self.render()
                self.gym.simulate(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        object_init_state = self.root_state_tensor[self.object_indices]
        if active:
            fallen_objects = self._not_in_active_bin(object_init_state)
        else:
            fallen_objects = self._not_in_done_bin(object_init_state)
        object_init_state[..., 7:] = 0.
        return fallen_objects, object_init_state

    def _position_objects_for_drop(self, active_object_indices: Tensor,
                                   passive_object_indices: Tensor,
                                   active_object_state: Tensor,
                                   passive_object_state: Tensor) -> None:
        self.root_state_tensor[active_object_indices] = \
            active_object_state

        if len(passive_object_indices) > 0:
            self.root_state_tensor[passive_object_indices] = \
                passive_object_state

    def _reset_objects(self, env_ids) -> None:
        # Find cleared and remaining objects and their init states
        remaining_object_indices = torch.masked_select(
            self.object_indices, self.remaining_objects == 1)
        remaining_object_init_state = torch.masked_select(
            self.object_init_state,
            (self.remaining_objects == 1).unsqueeze(2).repeat(1, 1, 13)).view(-1, 13)
        cleared_object_indices = torch.masked_select(
            self.object_indices, self.remaining_objects == 0)
        cleared_object_init_state = torch.masked_select(
            self.object_cleared_state,
            (self.remaining_objects == 0).unsqueeze(2).repeat(1, 1, 13)).view(-1, 13)

        # Reset remaining and cleared objects
        if len(remaining_object_indices) > 0:
            self.root_state_tensor[remaining_object_indices] = \
                remaining_object_init_state
        if len(cleared_object_indices) > 0:
            self.root_state_tensor[cleared_object_indices] = \
                cleared_object_init_state

        self.prev_remaining_objects = self.remaining_objects

        # Set root state tensor of the simulation
        object_indices = torch.unique(
            torch.cat([self.object_indices[env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # Refill empty bins
        empty_bin = (self.num_remaining_objects == 0).bool()
        if torch.any(empty_bin):
            selection_mask = empty_bin.unsqueeze(1).repeat(1, self.cfg["task"]["numObjectsPerBin"])
            refill_indices = torch.masked_select(self.object_indices, selection_mask)
            self.root_state_tensor[refill_indices] = self.init_root_state_tensor[refill_indices]
            object_indices = torch.unique(
                torch.cat([refill_indices]).to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def get_object_set(self):
        obj_set = self.obs_buf[:, 0:self.object_state_size * self.cfg["task"]["numObjectsPerBin"]]
        obj_set = obj_set.view(self.num_envs, self.cfg["task"]["numObjectsPerBin"],
                               self.object_state_size)
        return obj_set

    def _draw_object_pose(self, i: int) -> None:
        for j in range(self.cfg["task"]["numObjectsPerBin"]):
            self._draw_coordinate_system(self.envs[i], self.object_pos[i, j],
                                         self.object_rot[i, j])

    def _draw_hand_object_distance(self, i: int) -> None:
        hand_pos = self.hand_pos[i].cpu().numpy()
        for j in range(self.cfg["task"]["numObjectsPerBin"]):
            object_pos = self.object_pos[i, j].cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 1,
                               [object_pos[0], object_pos[1], object_pos[2],
                                hand_pos[0], hand_pos[1], hand_pos[2]],
                               [0.85, 0.1, 0.85])

    def _draw_object_target_distance(self, i: int) -> None:
        for j in range(self.cfg["task"]["numObjectsPerBin"]):
            object_pos = self.object_pos[i, j].cpu().numpy()
            target_pos = object_pos.copy()
            target_pos[2] = self.cfg["reward"]["targetHeight"]
            self.gym.add_lines(self.viewer, self.envs[i], 1,
                               [object_pos[0], object_pos[1], object_pos[2],
                                target_pos[0], target_pos[1], target_pos[2]],
                               [0.85, 0.85, 0.1])

    def _draw_adaptive_relation_graph(self, i: int, num_relations: int = 2) -> None:
        object_pos = self.object_pos[i].cpu()
        object_pos_i = object_pos.unsqueeze(0).repeat(
            self.cfg["task"]["numObjectsPerBin"], 1, 1)
        object_pos_j = object_pos.unsqueeze(1).repeat(
            1, self.cfg["task"]["numObjectsPerBin"], 1)
        o_pair = torch.stack([object_pos_i, object_pos_j], dim=-1)

        object_distances = torch.norm((o_pair[..., 0] - o_pair[..., 1]), dim=2)

        smallest_distances, indices = torch.topk(object_distances, num_relations + 1, dim=1, largest=False)
        smallest_distances = smallest_distances[:, 1:]
        indices = indices[:, 1:]

        for o in range(self.cfg["task"]["numObjectsPerBin"]):
            start_pos = object_pos[o].numpy()
            for idx in indices[o]:
                target_pos = object_pos[idx].numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [start_pos[0], start_pos[1], start_pos[2],
                                    target_pos[0], target_pos[1],
                                    target_pos[2]],
                                   [0.85, 0.1, 0.85])

    def _color_objects_on_success(self, i: int) -> None:
        for j in range(self.cfg["task"]["numObjectsPerBin"]):
            if torch.norm(self.object_pos[i] - self.goal_pos) <= \
                    self.cfg["reward"]["goalProximityThreshold"]:
                self.gym.set_rigid_body_color(self.envs[i], self.objects[i][j],
                                              0, gymapi.MeshType.MESH_VISUAL,
                                              gymapi.Vec3(0.1, 0.95, 0.1))

    def _draw_debug_visualization(self, i: int) -> None:
        super()._draw_debug_visualization(i)
        if self.cfg["debug"]["drawAdaptiveRelationGraph"]:
            self._draw_adaptive_relation_graph(i)

    def _draw_object_bbox(
            self,
            i: int,
            draw_bounds: bool = False,
            draw_corners: bool = True,
            draw_pose: bool = False
    ) -> None:
        assert any([o.startswith("objectBbox") for o in self.object_obs]), \
            "Cannot draw bounding boxes of objects if it is not included " \
            "in the observations."
        for j in range(self.cfg["task"]["numObjectsPerBin"]):
            # draw bounding box
            relative_bbox = torch.stack([-0.5 * self.object_bbox_extent[i, j],
                                         0.5 * self.object_bbox_extent[i, j]],
                                        dim=0)
            bbox_pos = self.object_bbox_pose[i, j, 0:3].cpu().numpy()
            bbox_rot = self.object_bbox_pose[i, j, 3:7].cpu().numpy()
            bbox_pose = gymapi.Transform(
                p=gymapi.Vec3(
                    x=bbox_pos[0], y=bbox_pos[1], z=bbox_pos[2]),
                r=gymapi.Quat(
                    x=bbox_rot[0], y=bbox_rot[1], z=bbox_rot[2], w=bbox_rot[3]))
            bbox = gymutil.WireframeBBoxGeometry(relative_bbox, pose=bbox_pose)
            gymutil.draw_lines(bbox, self.gym, self.viewer, self.envs[i],
                               pose=gymapi.Transform())

            if draw_bounds:
                bbox_bounds = self.object_bbox_bounds[i, j].cpu().numpy()
                front_lower_left_pose = gymapi.Transform(
                    p=gymapi.Vec3(x=bbox_bounds[0, 0],
                                  y=bbox_bounds[0, 1],
                                  z=bbox_bounds[0, 2]))
                back_upper_right_pose = gymapi.Transform(
                    p=gymapi.Vec3(x=bbox_bounds[1, 0],
                                  y=bbox_bounds[1, 1],
                                  z=bbox_bounds[1, 2]))
                front_lower_left = gymutil.WireframeSphereGeometry(
                    0.005, 12, 12, front_lower_left_pose, color=(1, 1, 0))
                back_upper_right = gymutil.WireframeSphereGeometry(
                    0.005, 12, 12, back_upper_right_pose, color=(1, 1, 0))
                gymutil.draw_lines(front_lower_left, self.gym, self.viewer,
                                   self.envs[i],
                                   pose=gymapi.Transform())
                gymutil.draw_lines(back_upper_right, self.gym, self.viewer,
                                   self.envs[i],
                                   pose=gymapi.Transform())
            if draw_corners:
                bbox_corners = self.object_bbox_corners[i, j].cpu().numpy()
                for k in range(bbox_corners.shape[0]):
                    corner_pose = gymapi.Transform(
                        p=gymapi.Vec3(x=bbox_corners[k, 0],
                                      y=bbox_corners[k, 1],
                                      z=bbox_corners[k, 2]))
                    corner = gymutil.WireframeSphereGeometry(
                        0.005, 12, 12, corner_pose, color=(1, 0, 1))
                    gymutil.draw_lines(corner, self.gym, self.viewer,
                                       self.envs[i],
                                       pose=gymapi.Transform())

            if draw_pose:
                self._draw_coordinate_system(self.envs[i],
                                             self.object_bbox_pose[i, j, 0:3],
                                             self.object_bbox_pose[i, j, 3:7])

    def _draw_object_surface_samples(self, i: int) -> None:
        assert "objectSurfaceSamples" in self.object_obs, \
            "Cannot draw surface samples of objects if they are not included " \
            "in the observations."
        for j in range(self.cfg["task"]["numObjectsPerBin"]):
            surface_samples = self.object_surface_samples[i, j].cpu().numpy()
            for k in range(surface_samples.shape[0]):
                sample_pose = gymapi.Transform(
                    p=gymapi.Vec3(x=surface_samples[k, 0],
                                  y=surface_samples[k, 1],
                                  z=surface_samples[k, 2]))
                sample_sphere = gymutil.WireframeSphereGeometry(
                    0.0025, 12, 12, sample_pose, color=(1, 0, 1))
                gymutil.draw_lines(sample_sphere, self.gym, self.viewer,
                                   self.envs[i], pose=gymapi.Transform())


@torch.jit.script
def compute_bin_picking_reward(
        rew_buf, reset_buf, progress_buf, successes, success_rate, remaining_objects,
        object_pos, object_linvel, object_init_pos,
        fingertip_pos, actions,
        action_penalty_scale: float,
        fingertips_to_object_distance_reward_scale: float,
        max_episode_length: int,
        object_height_reward_scale: float,
        target_height_reward_scale: float,
        object_falls_off_table_penalty_scale: float,
        lift_off_height: float,
        max_xy_drift: float,
        target_height: float,
        eps_fingertips: float,
        eps_height: float,
        xy_center,
        sparse: bool,
        reset_when_object_falls_off_table: bool,
        reset_when_object_is_lifted: bool,
        return_rewards_dict: bool,
):
    # Rewards are calculated by first calculating usual object lifting rewards
    # for all objects, that multiplying the rewards by a mask of objects which
    # are still in the bin that the agent is picking from and then summing up
    # the result.

    # ============ computing reward ... ============
    # penalize large actions: r ~ -|| a_t ||^2
    squared_action_norm = torch.sum(actions.pow(2), dim=-1)
    action_reward = squared_action_norm * action_penalty_scale

    # penalize fingertips-object distance: r ~ (1 / (d(ft, obj)^2 + ϵ_ft))^2
    under_object_center = object_pos.clone()
    under_object_center[:, :, 2] -= 0.0
    under_object_center = under_object_center.unsqueeze(1).repeat(
        1, fingertip_pos.shape[1], 1, 1)
    fingertips_to_object_distance = torch.norm(
        fingertip_pos.unsqueeze(2).repeat(1, 1, object_pos.shape[1], 1)
        - under_object_center, dim=-1)
    thumb_scaling = 2
    fingertips_to_object_distance[:, 0] *= thumb_scaling
    fingertips_to_object_distance = torch.sum(fingertips_to_object_distance,
                                              dim=1)

    fingertips_to_object_distance_reward = \
        1.0 / (eps_fingertips + 5 * fingertips_to_object_distance.pow(2))
    fingertips_to_object_distance_reward *= fingertips_to_object_distance_reward
    fingertips_to_object_distance_reward *= fingertips_to_object_distance_reward_scale

    # continuous height reward: r ~ 1 / (Δh + eps_h)
    object_raised_by = object_pos[:, :, 2] - object_init_pos[:, :, 2]
    delta_height = torch.clamp(
        object_raised_by - lift_off_height, max=0)
    object_height_reward = (1.0 / (torch.abs(delta_height) + eps_height)) - (
                1.0 / (lift_off_height + eps_height))
    object_height_reward *= object_height_reward_scale

    # object height is not rewarded far away from center
    xy_drift = torch.norm(object_pos[:, :, 0:2] - xy_center, dim=2)
    close_to_center = xy_drift <= max_xy_drift
    object_height_reward *= close_to_center.float()

    # reward reaching the target height: r ~ 1(object_lifted == True)
    object_lifted = object_raised_by >= target_height
    target_height_reward = target_height_reward_scale * object_lifted.float()

    # compute final reward
    rewards_dict = {}
    if sparse:
        reward = object_lifted.float() - 1
        if return_rewards_dict:
            dense_reward = fingertips_to_object_distance_reward + \
                 object_height_reward + \
                 target_height_reward
            rewards_dict["sparse"] = reward
            rewards_dict["dense"] = dense_reward        
    else:
        reward = fingertips_to_object_distance_reward + \
                 object_height_reward + \
                 target_height_reward
        if return_rewards_dict:
            sparse_reward = object_lifted.float() - 1
            rewards_dict["sparse"] = sparse_reward
            rewards_dict["dense"] = reward                         

    # Mask reward to only include objects that are still in the bin
    reward *= remaining_objects
    # Sum up reward over all objects
    reward = torch.sum(reward, dim=1)
    reward += action_reward

    # ============ determining resets ... ============
    # reset environments that have reached the maximum number of steps
    resets = torch.where(progress_buf >= max_episode_length,
                         torch.ones_like(reset_buf), reset_buf)

    # reset environments that lifted the object successfully
    if reset_when_object_is_lifted:
        num_objects_lifted = torch.sum(object_lifted.int(), dim=1)
        resets = torch.where(num_objects_lifted > 0,
                             torch.ones_like(resets), resets)

    # reset environments if object falls of the table
    if reset_when_object_falls_off_table:
        table_height = 0.8
        num_objects_fallen = torch.sum((object_pos[:, :, 2] < table_height).int(), dim=1)
        resets = torch.where(num_objects_fallen > 0,
                             torch.ones_like(resets), resets)
        reward += num_objects_fallen * object_falls_off_table_penalty_scale

    # remove lifted objects from bin
    remaining_objects -= object_lifted.long()
    num_remaining_objects = torch.sum(remaining_objects, dim=1)

    empty_bin = (num_remaining_objects == 0).unsqueeze(1).repeat(
        1, object_pos.shape[1])

    # Put all objects back into the starting bin if none are remaining
    remaining_objects = torch.where(empty_bin,
                                    torch.ones_like(remaining_objects),
                                    remaining_objects)

    return reward, resets, progress_buf, \
           successes, success_rate, \
           remaining_objects, num_remaining_objects, \
           action_reward, \
           fingertips_to_object_distance_reward, \
           object_height_reward, \
           target_height_reward, \
           rewards_dict


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
