import os
import glob
from collections import defaultdict
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
from typing import *
from rofunc.learning.RofuncRL.tasks.isaacgymenv.grasp.base.grasping_task import GraspingTask


class LiftObjectTask(GraspingTask):
    def __init__(self,
                 cfg: Dict[str, Any],
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool) -> None:
        """Creates UR5e-SIH robot on the task of lifting an object off the
        table.

        Args:
            cfg: Configuration dictionary that contains parameters for
                how the simulation should run and defines properties of the
                task.
            sim_device: Device on which the physics simulation is run
                (e.g. "cuda:0" or "cpu").
            graphics_device_id: ID of the device to run the rendering.
            headless:  Whether to run the simulation without a viewer.
        """
        super().__init__(cfg, sim_device, graphics_device_id, headless)

    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        super()._parse_cfg(cfg)
        self.ycb_objects, self.egad_objects = self._parse_objects(cfg["task"])
        self.egad_colors = {name: tuple(np.random.rand(3))
                            for name in self.egad_objects}
        self.objects_list = self.ycb_objects + self.egad_objects
        self.object_obs, self.robot_obs = self._parse_observation_type(
            cfg["task"]["observationType"], cfg["task"]["num_observations"])
        self.cfg["env"]["numObservations"] = self.num_observations

    def _parse_objects(self, task_cfg: Dict[str, Any]) -> List[List[str]]:
        object_datasets = ["ycbObjects", "egadObjects"]
        used_objects = []
        for dataset in object_datasets:
            if dataset in task_cfg.keys():
                if isinstance(task_cfg[dataset], str):
                    requested_objects = [task_cfg[dataset]]
                else:
                    requested_objects = task_cfg[dataset]
                object_list = []
                for obj in requested_objects:
                    if "*" in obj:
                        object_list += self._solve_object_regex(obj, dataset)
                    else:
                        object_list.append(obj)
                used_objects.append(object_list)
            else:
                used_objects.append([])
        return used_objects

    def _solve_object_regex(self, regex: str, dataset: str) -> List[str]:
        if dataset == "ycbObjects":
            root = os.path.join(self.asset_root, self.ycb_object_asset_root)
        elif dataset == "egadObjects":
            root = os.path.join(self.asset_root, self.egad_object_asset_root)
        else:
            assert False
        object_list = []
        regex = os.path.join(root, regex)
        for path in glob.glob(regex):
            file_name = path.split("/")[-1]
            if "." in file_name:
                obj, extension = file_name.split(".")
            else:
                obj = file_name
                extension = ""
            if extension == "urdf":
                object_list.append(obj)
        return object_list

    def _parse_observation_type(self, observation_type,
                                num_observations) -> Tuple[List, List]:
        if isinstance(observation_type, str):
            observation_type = [observation_type]

        self.num_observations = 0
        object_observations, robot_observations = [], []
        for obs in observation_type:
            if not (obs in num_observations.keys()):
                raise ValueError(f"Unknown observation type '{obs}' given. "
                                 f"Should be in {num_observations.keys()}.")
            if obs.startswith("object"):
                object_observations.append(obs)
            else:
                robot_observations.append(obs)
            self.num_observations += num_observations[obs]
        self.num_obs_dict = num_observations

        if "fingertipContactForces" in observation_type:
            self.cfg["sim"]["useContactForces"] = True

        return object_observations, robot_observations

    def _load_task_assets(self) -> Dict[str, Any]:
        self.objects = []
        self.object_init_state = []
        self.object_indices, self.object_rigid_body_indices = [], []
        self.object_ids = []
        object_assets_dict = self._load_object_assets()
        self.object_assets_dict = object_assets_dict
        return object_assets_dict

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any],
                  agg_info) -> None:
        self._add_object(env_ptr, i, task_asset_dict["asset"],
                         task_asset_dict["start_pose"])

    def _process_task_handles(self) -> None:
        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.default_object_init_state = self.object_init_state.clone()
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device)
        self.object_rigid_body_indices = to_torch(
            self.object_rigid_body_indices, dtype=torch.long,
            device=self.device)

    def _load_object_assets(self):
        """Loads the object(s) to be picked up from the YCB or EGAD dataset."""
        object_assets_dict = defaultdict(list)

        self.num_object_bodies, self.num_object_shapes = [], []
        # load YCB objects
        for ycb_object in self.ycb_objects:
            object_asset_file = os.path.join(
                self.ycb_object_asset_root,
                os.path.normpath(ycb_object + "/" + ycb_object + ".urdf"))
            object_assets_dict = self._load_object(
                object_assets_dict, object_asset_file,
                start_pose=gymapi.Transform(p=gymapi.Vec3(0, 0, 1.0)))

        # load EGAD objects
        for egad_object in self.egad_objects:
            object_asset_file = os.path.join(
                self.egad_object_asset_root,
                os.path.normpath(egad_object + ".urdf"))
            object_assets_dict = self._load_object(
                object_assets_dict, object_asset_file,
                start_pose=gymapi.Transform(p=gymapi.Vec3(0, 0, 1.0)))
        return object_assets_dict

    def _load_object(self, object_assets_dict, asset_file, start_pose) -> Dict[str, Any]:
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        # Enable convex decomposition
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 1000000

        object_assets_dict["asset"].append(
            self.gym.load_asset(self.sim, self.asset_root, asset_file,
                                object_asset_options))
        object_assets_dict["start_pose"].append(start_pose)

        # retrieve mesh-specific information in case of shape-dependent obs
        if any([o.startswith("objectBbox") for o in self.object_obs]):
            from urdfpy import URDF
            import trimesh
            object_urdf = URDF.load(
                os.path.join(self.asset_root, asset_file))
            object_bounding_box = trimesh.bounds.oriented_bounds(
                object_urdf.base_link.collision_mesh)
            object_assets_dict["bbox"].append(object_bounding_box)

        if "objectSurfaceSamples" in self.object_obs:
            from urdfpy import URDF
            import trimesh
            object_urdf = URDF.load(
                os.path.join(self.asset_root, asset_file))
            object_mesh = object_urdf.base_link.collision_mesh
            surface_samples = np.array(trimesh.sample.sample_surface(
                object_mesh, count=20)[0]).astype(np.float)
            object_assets_dict["samples"].append(surface_samples)

        # update number of object bodies and shapes
        self.num_object_bodies.append(
            self.gym.get_asset_rigid_body_count(
                object_assets_dict["asset"][-1]))
        self.num_object_shapes.append(
            self.gym.get_asset_rigid_shape_count(
                object_assets_dict["asset"][-1]))
        return object_assets_dict

    @property
    def egad_object_asset_root(self) -> os.path:
        object_asset_root = os.path.normpath("urdf/egad/train")
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "egadObjectAssetRoot", object_asset_root))
        return object_asset_root

    @property
    def ycb_object_asset_root(self) -> os.path:
        object_asset_root = os.path.normpath("urdf/ycb_video_objects/centered")
        if "asset" in self.cfg:
            return os.path.normpath(self.cfg["asset"].get(
                "ycbObjectAssetRoot", object_asset_root))
        return object_asset_root

    @property
    def num_different_objects(self) -> int:
        return len(self.ycb_objects) + len(self.egad_objects)

    def _begin_aggregate(self, env_ptr, env_idx: int) -> None:
        object_idx = env_idx % self.num_different_objects
        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies + \
                         self.num_object_bodies[object_idx]
        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes + \
                         self.num_object_shapes[object_idx]
        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)

    def _add_object(self, env_ptr, i, object_assets,
                    object_start_poses) -> None:
        # choose object and create actor
        object_id = i % self.num_different_objects
        self.object_ids.append(object_id)
        object_actor = self.gym.create_actor(env_ptr, object_assets[object_id],
                                             object_start_poses[object_id],
                                             self.objects_list[object_id], i,
                                             0, 0)

        # infer rigid body index
        object_rigid_body_name = self.gym.get_actor_rigid_body_names(
            env_ptr, object_actor)[0]
        object_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, object_actor, object_rigid_body_name, gymapi.DOMAIN_SIM)

        self.object_init_state.append(
            [object_start_poses[object_id].p.x,
             object_start_poses[object_id].p.y,
             object_start_poses[object_id].p.z,
             object_start_poses[object_id].r.x,
             object_start_poses[object_id].r.y,
             object_start_poses[object_id].r.z,
             object_start_poses[object_id].r.w,
             0, 0, 0, 0, 0, 0])
        self.object_indices.append(self.gym.get_actor_index(
            env_ptr, object_actor, gymapi.DOMAIN_SIM))
        self.object_rigid_body_indices.append(object_rigid_body_index)
        self.objects.append(object_actor)

    def compute_reward(self) -> None:
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
            self.object_pos, self.object_linvel, self.object_init_pos,
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
            self.cfg["reward"]["returnRewardsDict"],
        )

        if self.cfg["debug"]["verbose"]:
            self.writer.add_scalar(
                "action_reward",
                torch.mean(action_reward), self.env_steps)
            self.writer.add_scalar(
                "fingertips_to_object_distance_reward",
                torch.mean(fingertips_to_object_distance_reward), self.env_steps)
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

    def _process_state_tensors(self) -> None:
        super()._process_state_tensors()
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

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
        self.object_bbox_pose = torch.cat([bbox_pos, bbox_rot], dim=1)
        self.object_bbox_corners = bbox_pos.unsqueeze(1).repeat(1, 8, 1) + \
            quat_apply(
                bbox_rot.unsqueeze(1).repeat(1, 8, 1),
                self.bbox_corner_coords * self.object_bbox_extent.unsqueeze(
                    1).repeat(1, 8, 1))
        self.object_bbox_bounds = torch.stack(
            [self.object_bbox_corners[:, 0], self.object_bbox_corners[:, -1]],
            dim=1)

    def init_object_bbox(self) -> None:
        from scipy.spatial.transform import Rotation as R
        self.object_bbox_extent = []
        self.object_bbox_pos_offset = []
        self.object_bbox_rot_offset = []
        for env_id in range(self.num_envs):
            self.object_bbox_extent.append(
                self.object_assets_dict["bbox"][self.object_ids[env_id]][1])
            to_origin = self.object_assets_dict["bbox"][
                self.object_ids[env_id]][0]
            from_origin = np.linalg.inv(to_origin)
            r = R.from_matrix(from_origin[0:3, 0:3])
            r_quat = r.as_quat()
            translation = np.array(
                [from_origin[0, 3], from_origin[1, 3], from_origin[2, 3]])
            self.object_bbox_pos_offset.append(translation)
            self.object_bbox_rot_offset.append(r_quat)

        self.object_bbox_extent = torch.from_numpy(
            np.stack(self.object_bbox_extent)).to(self.device).float()
        self.object_bbox_pos_offset = torch.from_numpy(
            np.stack(self.object_bbox_pos_offset)).to(self.device).float()
        self.object_bbox_rot_offset = torch.from_numpy(
            np.stack(self.object_bbox_rot_offset)).to(self.device).float()

        self.bbox_corner_coords = torch.tensor(
            [[[-0.5, -0.5, -0.5],
              [-0.5, -0.5, 0.5],
              [-0.5, 0.5, -0.5],
              [-0.5, 0.5, 0.5],
              [0.5, -0.5, -0.5],
              [0.5, -0.5, 0.5],
              [0.5, 0.5, -0.5],
              [0.5, 0.5, 0.5]]],
            device=self.device).repeat(self.num_envs, 1, 1)

    def calculate_object_surface_samples(self) -> None:
        if not hasattr(self, "object_surface_sample_offset_pos"):
            self.init_object_surface_samples()
        object_pos = self.object_pos.unsqueeze(1).repeat(
            1, self.object_surface_sample_offset_pos.shape[1], 1)
        object_rot = self.object_rot.unsqueeze(1).repeat(
            1, self.object_surface_sample_offset_pos.shape[1], 1)
        self.object_surface_samples = object_pos + quat_apply(
            object_rot, self.object_surface_sample_offset_pos)

    def init_object_surface_samples(self) -> None:
        self.object_surface_sample_offset_pos = []
        for env_id in range(self.num_envs):
            self.object_surface_sample_offset_pos.append(
                self.object_assets_dict["samples"][self.object_ids[env_id]])
        self.object_surface_sample_offset_pos = torch.from_numpy(
            np.stack(self.object_surface_sample_offset_pos)).to(
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
                                     self.object_bbox_bounds.flatten(1, 2))
            elif obs_key == "objectBboxCorners":  # 8 3D corner points
                obj_obs = add_to_obs(obj_obs, obs_key,
                                     self.object_bbox_corners.flatten(1, 2))
            elif obs_key == "objectBboxPose":  # Center pose of bounding box
                obj_obs = add_to_obs(obj_obs, obs_key, self.object_bbox_pose)
            elif obs_key == "objectSurfaceSamples":  # Samples on mesh surface
                obj_obs = add_to_obs(obj_obs, obs_key,
                                     self.object_surface_samples.flatten(1, 2))

            obs_idx += self.num_obs_dict[obs_key]
        obj_obs = torch.cat(obj_obs, dim=1)
        self.obs_buf[:, 0:obs_idx] = obj_obs
        return obs_idx

    def reset_idx(self, env_ids):
        # Domain randomization would be added here ...

        if self._objects_dropped and not self.cfg["control"]["teleoperated"]:
            self._reset_robot(env_ids)
            self._reset_objects(env_ids)

        else:
            self._reset_robot(env_ids, self.robot_dof_away_pos, initial_sampling=True)
            self._drop_objects(env_ids)
            self._reset_robot(env_ids, initial_sampling=True)

            # simulate for one time-step to update the tracker position
            self.render()
            self.gym.simulate(self.sim)
            self._refresh_state_tensors()

            self.initial_tracker_pose = self.tracker_pose
            self.initial_finger_angles = self.robot_dof_pos[:, self.hand_actuated_dof_indices]

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _drop_objects(self, env_ids, skip_steps: int = 250) -> None:
        assert torch.equal(env_ids,
                           to_torch(list(range(self.num_envs)),
                                    device=self.device, dtype=torch.long)), \
            "Objects must be dropped in all environments simultaneously, as " \
            "this is a distinct phase where the physics simulation is " \
            "stepped before the RL agent can actually interact with the task."

        # Position objects above table
        self._randomize_object_positions(env_ids)

        # Run simulation
        for _ in range(skip_steps):
            self.render()
            self.gym.simulate(self.sim)

        # Set object state after drop as object init state
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.object_init_state = self.root_state_tensor[self.object_indices[env_ids]]
        self.object_init_pos = self.object_init_state[:, 0:3]
        self.compute_observations()
        self._objects_dropped = True

    def _randomize_object_positions(self, env_ids) -> None:
        # Generate random values
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 5), device=self.device)

        # Reset object position and rotation
        self.root_state_tensor[self.object_indices[env_ids]] = \
        self.default_object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = \
            self.default_object_init_state[env_ids, 0:2] + \
            self.cfg["initState"]["noise"]["objectPos"] * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids],
                               self.up_axis_idx] = \
            self.default_object_init_state[env_ids, self.up_axis_idx] + \
            self.cfg["initState"]["noise"]["objectPos"] * \
            rand_floats[:, self.up_axis_idx]
        new_object_rot = randomize_rotation(rand_floats[:, 3],
                                            rand_floats[:, 4],
                                            self.x_unit_tensor[env_ids],
                                            self.y_unit_tensor[env_ids])
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = \
            new_object_rot

        # Set object velocities to zero
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = \
            torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        # Set root state tensor of the simulation
        object_indices = torch.unique(
            torch.cat([self.object_indices[env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def _reset_objects(self, env_ids) -> None:
        self.root_state_tensor[self.object_indices[env_ids]] = \
            self.object_init_state[env_ids].clone()

        # Set root state tensor of the simulation
        object_indices = torch.unique(
            torch.cat([self.object_indices[env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def _draw_debug_visualization(self, i: int) -> None:
        super()._draw_debug_visualization(i)
        if self.cfg["debug"]["drawObjectPose"]:
            self._draw_object_pose(i)
        if self.cfg["debug"]["drawObjectBbox"]:
            self._draw_object_bbox(i)
        if self.cfg["debug"]["drawObjectSurfaceSamples"]:
            self._draw_object_surface_samples(i)
        if self.cfg["debug"]["drawHandObjectDistance"]:
            self._draw_hand_object_distance(i)
        if self.cfg["debug"]["drawObjectTargetDistance"]:
            self._draw_object_target_distance(i)
        if self.cfg["debug"]["colorObjectsOnSuccess"]:
            self._color_objects_on_success(i)

    def _draw_object_pose(self, i: int) -> None:
        self._draw_coordinate_system(self.envs[i], self.object_pos[i],
                                     self.object_rot[i])

    def _draw_hand_object_distance(self, i: int) -> None:
        object_pos = self.object_pos[i].cpu().numpy()
        hand_pos = self.hand_pos[i].cpu().numpy()
        self.gym.add_lines(self.viewer, self.envs[i], 1,
                           [object_pos[0], object_pos[1], object_pos[2],
                            hand_pos[0], hand_pos[1], hand_pos[2]],
                           [0.85, 0.1, 0.85])

    def _draw_object_target_distance(self, i: int) -> None:
        object_pos = self.object_pos[i].cpu().numpy()
        init_object_pos = self.object_init_pos[i].cpu().numpy()
        target_pos = object_pos.copy()
        target_pos[2] = init_object_pos[2] + self.cfg["reward"]["objectLiftingThreshold"]
        self.gym.add_lines(self.viewer, self.envs[i], 1,
                           [object_pos[0], object_pos[1], object_pos[2],
                            target_pos[0], target_pos[1], target_pos[2]],
                           [0.85, 0.85, 0.1])

    def _color_objects_on_success(self, i: int) -> None:
        if self.object_pos[i][2] >= self.object_init_pos[i][2] + \
                self.cfg["reward"]["objectLiftingThreshold"]:
            self.gym.set_rigid_body_color(self.envs[i], self.objects[i], 0,
                                          gymapi.MeshType.MESH_VISUAL,
                                          gymapi.Vec3(0.1, 0.95, 0.1))

    def _draw_object_bbox(
            self,
            i: int,
            draw_bounds: bool = False,
            draw_corners: bool = True,
            draw_pose: bool = True
    ) -> None:
        assert any([o.startswith("objectBbox") for o in self.object_obs]), \
            "Cannot draw bounding boxes of objects if it is not included in " \
            "the observations."

        # draw bounding box
        relative_bbox = torch.stack([-0.5 * self.object_bbox_extent[i],
                                     0.5 * self.object_bbox_extent[i]], dim=0)
        bbox_pos = self.object_bbox_pose[i, 0:3].cpu().numpy()
        bbox_rot = self.object_bbox_pose[i, 3:7].cpu().numpy()
        bbox_pose = gymapi.Transform(
            p=gymapi.Vec3(
                x=bbox_pos[0], y=bbox_pos[1], z=bbox_pos[2]),
            r=gymapi.Quat(
                x=bbox_rot[0], y=bbox_rot[1], z=bbox_rot[2], w=bbox_rot[3]))
        bbox = gymutil.WireframeBBoxGeometry(relative_bbox, pose=bbox_pose)
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.envs[i],
                           pose=gymapi.Transform())

        if draw_bounds:
            bbox_bounds = self.object_bbox_bounds[i].cpu().numpy()
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
            bbox_corners = self.object_bbox_corners[i].cpu().numpy()
            for j in range(bbox_corners.shape[0]):
                corner_pose = gymapi.Transform(
                    p=gymapi.Vec3(x=bbox_corners[j, 0],
                                  y=bbox_corners[j, 1],
                                  z=bbox_corners[j, 2]))
                corner = gymutil.WireframeSphereGeometry(
                    0.005, 12, 12, corner_pose, color=(1, 0, 1))
                gymutil.draw_lines(corner, self.gym, self.viewer, self.envs[i],
                                   pose=gymapi.Transform())

        if draw_pose:
            self._draw_coordinate_system(self.envs[i],
                                         self.object_bbox_pose[i, 0:3],
                                         self.object_bbox_pose[i, 3:7])

    def _draw_object_surface_samples(self, i: int) -> None:
        assert "objectSurfaceSamples" in self.object_obs, \
            "Cannot draw surface samples of objects if they are not included " \
            "in the observations."
        surface_samples = self.object_surface_samples[i].cpu().numpy()
        for j in range(surface_samples.shape[0]):
            sample_pose = gymapi.Transform(
                p=gymapi.Vec3(x=surface_samples[j, 0],
                              y=surface_samples[j, 1],
                              z=surface_samples[j, 2]))
            sample_sphere = gymutil.WireframeSphereGeometry(
                0.0025, 12, 12, sample_pose, color=(1, 0, 1))
            gymutil.draw_lines(sample_sphere, self.gym, self.viewer,
                               self.envs[i], pose=gymapi.Transform())


@torch.jit.script
def compute_object_lifting_reward(
    rew_buf, reset_buf, progress_buf, successes, success_rate,
    object_pos, object_linvel, object_init_pos,
    fingertip_pos, actions,
    action_penalty_scale: float,
    fingertips_to_object_distance_reward_scale: float,
    max_episode_length: int,
    object_height_reward_scale: float,
    object_velocity_reward_scale: float,
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

    # ============ computing reward ... ============
    # penalize large actions: r ~ -|| a_t ||^2
    squared_action_norm = torch.sum(actions.pow(2), dim=-1)
    action_reward = squared_action_norm * action_penalty_scale

    # penalize fingertips-object distance: r ~ (1 / (d(ft, obj)^2 + ϵ_ft))^2
    under_object_center = object_pos.clone()
    under_object_center[:, 2] -= 0.0
    under_object_center = under_object_center.unsqueeze(1).repeat(1, fingertip_pos.shape[1], 1)
    fingertips_to_object_distance = torch.norm(fingertip_pos - under_object_center, dim=-1)
    thumb_scaling = 2
    fingertips_to_object_distance[:, 0] *= thumb_scaling
    fingertips_to_object_distance = torch.sum(fingertips_to_object_distance, dim=-1)
    fingertips_to_object_distance_reward = \
        1.0 / (eps_fingertips + 5 * fingertips_to_object_distance.pow(2))
    fingertips_to_object_distance_reward *= fingertips_to_object_distance_reward
    fingertips_to_object_distance_reward *= fingertips_to_object_distance_reward_scale

    # continuous height reward: r ~ 1 / (Δh + eps_h)
    object_raised_by = object_pos[:, 2] - object_init_pos[:, 2]
    delta_height = torch.clamp(
        object_raised_by - lift_off_height, max=0)
    object_height_reward = (1.0 / (torch.abs(delta_height) + eps_height)) - (1.0 / (lift_off_height + eps_height))
    object_height_reward *= object_height_reward_scale

    # object height is not rewarded far away from center
    xy_drift = torch.norm(object_pos[:, 0:2] - xy_center, dim=1)
    close_to_center = xy_drift <= max_xy_drift
    object_height_reward *= close_to_center.float()

    # reward reaching the target height: r ~ 1(object_lifted == True)
    object_lifted = object_raised_by >= target_height
    target_height_reward = target_height_reward_scale * object_lifted.float()

    # reward moving the object: r ~ tanh(10 * object_vel)
    object_velocity = torch.norm(object_linvel, dim=1)
    object_velocity_reward = torch.tanh(10. * object_velocity)
    object_velocity_reward *= object_velocity_reward_scale

    # compute final reward
    rewards_dict = {}
    if sparse:
        reward = object_lifted.float() - 1
        if return_rewards_dict:
            dense_reward = action_reward + \
                fingertips_to_object_distance_reward + \
                object_height_reward + \
                object_velocity_reward + \
                target_height_reward
            rewards_dict["sparse"] = reward
            rewards_dict["dense"] = dense_reward
    else:
        reward = action_reward + \
            fingertips_to_object_distance_reward + \
            object_height_reward + \
            object_velocity_reward + \
            target_height_reward
        if return_rewards_dict:
            sparse_reward = object_lifted.float() - 1
            rewards_dict["sparse"] = sparse_reward
            rewards_dict["dense"] = reward

    # ============ determining resets ... ============
    # reset environments that have reached the maximum number of steps
    resets = torch.where(progress_buf >= max_episode_length,
                         torch.ones_like(reset_buf), reset_buf)

    # reset environments that lifted the object successfully
    if reset_when_object_is_lifted:
        resets = torch.where(object_lifted,
                             torch.ones_like(resets), resets)

    # reset environments if object falls of the table
    if reset_when_object_falls_off_table:
        table_height = 0.8
        resets = torch.where(object_pos[:, 2] < table_height,
                             torch.ones_like(resets), resets)
        object_has_fallen_off_table = object_pos[:, 2] < table_height
        reward += object_has_fallen_off_table.float() * object_falls_off_table_penalty_scale

    # determine total number of successful episodes and success rate
    total_resets = torch.sum(resets)
    avg_factor = 0.01
    successful_resets = torch.sum(object_lifted.float())
    successes += object_lifted.float()
    success_rate = torch.where(total_resets > 0,
                               avg_factor * (successful_resets/total_resets) +
                               (1. - avg_factor) * success_rate,
                               success_rate)

    return reward, resets, progress_buf, \
           successes, success_rate, \
           action_reward, \
           fingertips_to_object_distance_reward, \
           object_height_reward, \
           object_velocity_reward, \
           target_height_reward, \
           rewards_dict


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
