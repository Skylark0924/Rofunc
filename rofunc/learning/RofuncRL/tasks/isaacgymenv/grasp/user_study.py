from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from gym_grasp.tasks.pour_cup import PourCup
from typing import *


class UserStudy(PourCup):
    def __init__(self,
                 cfg: Dict[str, Any],
                 sim_device: str,
                 graphics_device_id: int,
                 headless: bool) -> None:
        """Creates UR5e-SIH robot on a group of user study tasks.

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
        # Load pour cups assets
        assets_dict = super()._load_task_assets()

        self.red_cubes, self.green_cubes, self.blue_cubes = [], [], []
        self.red_cube_indices, self.green_cube_indices, self.blue_cube_indices = [], [], []
        self.red_cube_init_state, self.green_cube_init_state, self.blue_cube_init_state = [], [], []

        red_cube_asset, red_cube_start_pose = self._load_cube_asset("red")
        green_cube_asset, green_cube_start_pose = self._load_cube_asset("green")
        blue_cube_asset, blue_cube_start_pose = self._load_cube_asset("blue")
        corrugated_plate_asset, corrugated_plate_start_pose = self._load_corrugated_plate_asset()

        assets_dict.update({"red_cube": {}, "green_cube": {}, "blue_cube": {},
                            "corrugated_plate": {}})
        assets_dict["red_cube"]["asset"] = red_cube_asset
        assets_dict["red_cube"]["start_pose"] = red_cube_start_pose
        assets_dict["green_cube"]["asset"] = green_cube_asset
        assets_dict["green_cube"]["start_pose"] = green_cube_start_pose
        assets_dict["blue_cube"]["asset"] = blue_cube_asset
        assets_dict["blue_cube"]["start_pose"] = blue_cube_start_pose
        assets_dict["corrugated_plate"]["asset"] = corrugated_plate_asset
        assets_dict["corrugated_plate"]["start_pose"] = corrugated_plate_start_pose
        return assets_dict

    def _load_cube_asset(self, color: str):
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.fix_base_link = False
        cube_asset = self.gym.create_box(
            self.sim, self.cfg["asset"]["cubeSize"],
            self.cfg["asset"]["cubeSize"], self.cfg["asset"]["cubeSize"],
            cube_asset_options)

        if color == "red":
            x, y, z = self.cfg["asset"]["redCubePosition"]
        elif color == "green":
            x, y, z = self.cfg["asset"]["greenCubePosition"]
        elif color == "blue":
            x, y, z = self.cfg["asset"]["blueCubePosition"]
        else:
            assert False
        cube_start_pose = gymapi.Transform(p=gymapi.Vec3(x, y, z))
        return cube_asset, cube_start_pose

    def _load_corrugated_plate_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        corrugated_plate_asset = self.gym.load_asset(
            self.sim, self.asset_root,
            self.cfg["asset"]["corrugatedPlateAssetFile"], asset_options)

        x, y, z = self.cfg["asset"]["corrugatedPlatePosition"]
        q_w, q_x, q_y, q_z = self.cfg["asset"]["corrugatedPlateRotation"]
        corrugated_plate_start_pose = gymapi.Transform(
            p=gymapi.Vec3(x, y, z), r=gymapi.Quat(q_x, q_y, q_z, q_w))
        return corrugated_plate_asset, corrugated_plate_start_pose

    def _add_task(self, env_ptr, i, task_asset_dict: Dict[str, Any],
                  agg_info) -> None:
        super()._add_task(env_ptr, i, task_asset_dict, agg_info)

        self._add_cubes(env_ptr, i,
                        task_asset_dict["red_cube"]["asset"],
                        task_asset_dict["green_cube"]["asset"],
                        task_asset_dict["blue_cube"]["asset"],
                        task_asset_dict["red_cube"]["start_pose"],
                        task_asset_dict["green_cube"]["start_pose"],
                        task_asset_dict["blue_cube"]["start_pose"])

        corrugated_plate_actor = self.gym.create_actor(
            env_ptr, task_asset_dict["corrugated_plate"]["asset"],
            task_asset_dict["corrugated_plate"]["start_pose"],
            'corrugated_plate', i, 0, 0)

    def _add_cubes(self, env_ptr, i, red_cube_asset, green_cube_asset,
                   blue_cube_asset, red_cube_start_pose, green_cube_start_pose,
                   blue_cube_start_pose) -> None:
        # create cube actors
        red_cube_actor = self.gym.create_actor(
            env_ptr, red_cube_asset, red_cube_start_pose, 'red_cube', i, 0, 0)
        green_cube_actor = self.gym.create_actor(
            env_ptr, green_cube_asset, green_cube_start_pose, 'green_cube', i, 0, 0)
        blue_cube_actor = self.gym.create_actor(
            env_ptr, blue_cube_asset, blue_cube_start_pose, 'blue_cube', i, 0, 0)

        self.gym.set_rigid_body_color(env_ptr, red_cube_actor, 0,
                                      gymapi.MeshType.MESH_VISUAL,
                                      gymapi.Vec3(0.7, 0.1, 0.1))
        self.gym.set_rigid_body_color(env_ptr, green_cube_actor, 0,
                                      gymapi.MeshType.MESH_VISUAL,
                                      gymapi.Vec3(0.1, 0.7, 0.1))
        self.gym.set_rigid_body_color(env_ptr, blue_cube_actor, 0,
                                      gymapi.MeshType.MESH_VISUAL,
                                      gymapi.Vec3(0.1, 0.1, 0.7))

        self.red_cube_indices.append(self.gym.get_actor_index(
            env_ptr, red_cube_actor, gymapi.DOMAIN_SIM))
        self.green_cube_indices.append(self.gym.get_actor_index(
            env_ptr, green_cube_actor, gymapi.DOMAIN_SIM))
        self.blue_cube_indices.append(self.gym.get_actor_index(
            env_ptr, blue_cube_actor, gymapi.DOMAIN_SIM))

        self.red_cube_init_state.append(
            [red_cube_start_pose.p.x,
             red_cube_start_pose.p.y,
             red_cube_start_pose.p.z,
             red_cube_start_pose.r.x,
             red_cube_start_pose.r.y,
             red_cube_start_pose.r.z,
             red_cube_start_pose.r.w,
             0, 0, 0, 0, 0, 0])
        self.green_cube_init_state.append(
            [green_cube_start_pose.p.x,
             green_cube_start_pose.p.y,
             green_cube_start_pose.p.z,
             green_cube_start_pose.r.x,
             green_cube_start_pose.r.y,
             green_cube_start_pose.r.z,
             green_cube_start_pose.r.w,
             0, 0, 0, 0, 0, 0])
        self.blue_cube_init_state.append(
            [blue_cube_start_pose.p.x,
             blue_cube_start_pose.p.y,
             blue_cube_start_pose.p.z,
             blue_cube_start_pose.r.x,
             blue_cube_start_pose.r.y,
             blue_cube_start_pose.r.z,
             blue_cube_start_pose.r.w,
             0, 0, 0, 0, 0, 0])

        self.red_cubes.append(red_cube_actor)
        self.green_cubes.append(green_cube_actor)
        self.blue_cubes.append(blue_cube_actor)

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

        self.red_cube_indices = to_torch(
            self.red_cube_indices, dtype=torch.long, device=self.device)
        self.green_cube_indices = to_torch(
            self.green_cube_indices, dtype=torch.long, device=self.device)
        self.blue_cube_indices = to_torch(
            self.blue_cube_indices, dtype=torch.long, device=self.device)

        self.red_cube_init_state = to_torch(
            self.red_cube_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.green_cube_init_state = to_torch(
            self.green_cube_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.blue_cube_init_state = to_torch(
            self.blue_cube_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)

    def _begin_aggregate(self, env_ptr, env_idx: int) -> None:
        max_agg_bodies = self.num_robot_bodies + self.num_table_bodies + \
                         self.num_cups_bodies + self.num_ball_bodies
        max_agg_shapes = self.num_robot_shapes + self.num_table_shapes + \
                         self.num_cups_shapes + self.num_ball_shapes

        self.aggregate_mode = 0

        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies,
                                     max_agg_shapes, True)

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
        self._reset_cubes(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_cubes(self, env_ids) -> None:
        self.root_state_tensor[self.red_cube_indices[env_ids]] = \
            self.red_cube_init_state[env_ids].clone()
        self.root_state_tensor[self.green_cube_indices[env_ids]] = \
            self.green_cube_init_state[env_ids].clone()
        self.root_state_tensor[self.blue_cube_indices[env_ids]] = \
            self.blue_cube_init_state[env_ids].clone()

        # Set root state tensor of the simulation
        cube_indices = torch.unique(
            torch.cat([self.red_cube_indices[env_ids],
                       self.green_cube_indices[env_ids],
                       self.blue_cube_indices[env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(cube_indices), len(cube_indices))
