# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.core.prims import RigidPrim, XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from pxr import UsdGeom

from rofunc.learning.RofuncRL.tasks.omniisaacgym.articulations.elfin import Elfin
from rofunc.learning.RofuncRL.tasks.omniisaacgym.articulations.views.elfin_view import ElfinView
from rofunc.learning.RofuncRL.tasks.omniisaacgym.base.rl_task import RLTask


class ElfinBagOmniTask(RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config  # task configuration, yaml file

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.enable_washer_task = self._task_cfg["env"]["enable_washer_task"]

        self.dt = 1 / 60.
        self._num_observations = 18  # joint pos (6) and vel (6), dist to bag (3), dist to virtual (3)
        self._num_actions = 6  # joint vel (6)

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:

        self.get_elfin()
        if self.enable_washer_task:
            self.get_washer()
            # self.get_washer_table()
            self.get_washer_bag()
            self.get_virtual_bag()
        else:
            self.get_basket()
            self.get_basket_table()
            self.get_basket_bag()

        super().set_up_scene(scene)
        self._elfins = ElfinView(prim_paths_expr="/World/envs/.*/elfin_s20", name="elfin_view")
        scene.add(self._elfins)
        scene.add(self._elfins._grippers)

        if self.enable_washer_task:
            self._washers = RigidPrimView(prim_paths_expr="/World/envs/.*/washer", name="washer_view", reset_xform_properties=False)
            # self._washer_tables = RigidPrimView(prim_paths_expr="/World/envs/.*/washer_table", name="washer_table_view", reset_xform_properties=False)
            self._bags = RigidPrimView(prim_paths_expr="/World/envs/.*/bag", name="bag_view", reset_xform_properties=False)
            self._virtual_bags = RigidPrimView(prim_paths_expr="/World/envs/.*/virtual_bag", name="virtual_bag_view", reset_xform_properties=False)

            scene.add(self._washers)
            # scene.add(self._washer_tables)
            scene.add(self._bags)
            scene.add(self._virtual_bags)
        else:
            self._baskets = RigidPrimView(prim_paths_expr="/World/envs/.*/basket", name="basket_view", reset_xform_properties=False)
            self._basket_tables = RigidPrimView(prim_paths_expr="/World/envs/.*/basket_table", name="basket_table_view", reset_xform_properties=False)
            self._bags = RigidPrimView(prim_paths_expr="/World/envs/.*/bag", name="bag_view", reset_xform_properties=False)

            scene.add(self._baskets)
            scene.add(self._basket_tables)
            scene.add(self._bags)


        self.init_data()
        return

    def get_elfin(self):
        elfin = Elfin(
            prim_path=self.default_zero_env_path + "/elfin_s20",
            name="elfin_s20",
            usd_path="/home/clover/isaac/USD/S20_fake_gripper.usd",
            translation=torch.tensor([0.0, 0.0, 0.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        self._sim_config.apply_articulation_settings("elfin_s20", get_prim_at_path(elfin.prim_path),
                                                     self._sim_config.parse_actor_config("elfin_s20"))

    # def get_washer(self):
    #     prim_path = self.default_zero_env_path + "/washer"
    #     usd_path = "/home/clover/isaac/USD/washer(cover).usd"
    #     add_reference_to_stage(usd_path, prim_path)
    #     washer = RigidPrim(prim_path=self.default_zero_env_path + "/washer",
    #                        name="washer",
    #                        translation=torch.tensor([0, -1.6, 0.844]),
    #                        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0])
    #                        )
    #     self._sim_config.apply_articulation_settings("washer", get_prim_at_path(washer.prim_path),
    #                                                  self._sim_config.parse_actor_config("washer"))

    def get_washer(self):
        prim_path = self.default_zero_env_path + "/washer"
        usd_path = "/home/clover/isaac/USD/washer.usd"
        add_reference_to_stage(usd_path, prim_path)
        washer = RigidPrim(prim_path=self.default_zero_env_path + "/washer",
                           name="washer",
                           translation=torch.tensor([0, -1.6, 0]),
                           orientation=torch.tensor([0.0, 0.0, 0.0, 1.0])
                           )
        self._sim_config.apply_articulation_settings("washer", get_prim_at_path(washer.prim_path),
                                                     self._sim_config.parse_actor_config("washer"))


    def get_washer_table(self):
        washer_table = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/washer_table",
            name="washer_table",
            color=torch.tensor([0.2, 0.4, 0.6]),
            scale=torch.tensor([1.5, 1.5, 1.0]),
            translation=torch.tensor([0.0, -1.6, 0.5])
        )
        self._sim_config.apply_articulation_settings("washer_table", get_prim_at_path(washer_table.prim_path),
                                                     self._sim_config.parse_actor_config("washer_table"))

    def get_washer_bag(self):
        prim_path = self.default_zero_env_path + "/bag"
        usd_path = "/home/clover/isaac/USD/bag.usd"
        add_reference_to_stage(usd_path, prim_path)
        bag = RigidPrim(prim_path=prim_path,
                        name="bag",
                        translation=torch.tensor([0.35859, -1.03675, 1.1]),
                        orientation=torch.tensor([0.7071068, 0.7071068, 0, 0]))
        self._sim_config.apply_articulation_settings("bag", get_prim_at_path(bag.prim_path),
                                                     self._sim_config.parse_actor_config("bag"))

    def get_virtual_bag(self):
        prim_path = self.default_zero_env_path + "/virtual_bag"
        virtual_bag = XFormPrim(prim_path=prim_path, name="virtual_bag", translation=torch.tensor([1.32231, 0.04733, 0.8]))
        self._sim_config.apply_articulation_settings("virtual_bag", get_prim_at_path(virtual_bag.prim_path),
                                                     self._sim_config.parse_actor_config("bag"))


    def get_basket(self):
        prim_path = self.default_zero_env_path + "/basket"
        usd_path = "/home/clover/isaac/USD/basket.usd"
        add_reference_to_stage(usd_path, prim_path)
        basket = RigidPrim(prim_path=self.default_zero_env_path + "/basket",
                           name="basket",
                           translation=torch.tensor([1.3, 0.0, 0.66687]),
                           orientation=torch.tensor([0.7071068, 0.7071068, 0.0, 0.0]))
        self._sim_config.apply_articulation_settings("basket", get_prim_at_path(basket.prim_path),
                                                     self._sim_config.parse_actor_config("basket"))

    def get_basket_table(self):
        basket_table = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/basket_table",
            name="basket_table",
            color=torch.tensor([0.2, 0.4, 0.6]),
            scale=torch.tensor([1.0, 1.0, 0.5]),
            translation=torch.tensor([1.3, 0.0, 0.25])
        )
        self._sim_config.apply_articulation_settings("basket_table", get_prim_at_path(basket_table.prim_path),
                                                     self._sim_config.parse_actor_config("basket_table"))

    def get_basket_bag(self):
        prim_path = self.default_zero_env_path + "/bag"
        usd_path = "/home/clover/isaac/USD/bag.usd"
        add_reference_to_stage(usd_path, prim_path)
        bag = RigidPrim(prim_path=prim_path,
                        name="bag",
                        translation=torch.tensor([1.32231, 0.04733, 0.8]),
                        orientation=torch.tensor([0.7071068, 0.7071068, 0, 0]))
        self._sim_config.apply_articulation_settings("bag", get_prim_at_path(bag.prim_path),
                                                     self._sim_config.parse_actor_config("bag"))

    def get_props(self):
        prim_path = self.default_zero_env_path + "/prop_0"
        usd_path = "/home/clover/isaac/USD/bag.usd"
        add_reference_to_stage(usd_path, prim_path)
        prop = RigidPrim(prim_path=prim_path, name="prop")

        self._sim_config.apply_articulation_settings("prop", get_prim_at_path(prop.prim_path),
                                                     self._sim_config.parse_actor_config("bag"))

        # prop_pos = [[0.35859, -1.43675, 1.6],
        #             [-0.09787, -1.44181, 1.59397],
        #             [-0.04536, -1.67567, 1.60273],
        #             [0.24796, -1.71223, 1.61449],
        #             [1.62231, 0.04733, 1.06685],
        #             [1.6244, -0.0565, 1.06543],
        #             [1.62173, 0.14851, 1.0645],
        #             [1.61526, 0.24952, 1.05556]
        #             ]
        # prop_ori = [[0.6078713, 0.7887495, -0.0318794, 0.0857343],
        #             [0.6142525, 0.7842759, -0.0747694, 0.0448861],
        #             [0.5335826, 0.8416763, -0.0803849, 0.0202219],
        #             [0.5961767, 0.7823589, 0.1235763, -0.1312131],
        #             [0.9658403, -0.2516233, 0.0592054, -0.0182487],
        #             [0.9650754, -0.2546022, 0.0592631, -0.017178],
        #             [0.9658403, -0.2516233, 0.0592054, -0.0182487],
        #             [0.9658403, -0.2516233, 0.0592054, -0.0182487]
        #             ]

        prop_pos = [[0.35859, -1.43675, 1.6]]
        prop_ori = [[0.6078713, 0.7887495, -0.0318794, 0.0857343]]

        prop_paths = [f"{self.default_zero_env_path}/prop_{j}" for j in range(1)]
        Cloner().clone(
            source_prim_path=self.default_zero_env_path + "/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos),
            orientations=np.array(prop_ori),
            replicate_physics=False,
        )

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        if self.enable_washer_task:
            self.elfin_default_dof_pos = torch.tensor([1.57, 0.0, 0.0, -1.57, -1.57, 0.0],
                                                  device=self._device)  # for washer
        else:
            self.elfin_default_dof_pos = torch.tensor([3.14, 0.0, 0.0, -1.57, -1.57, 0.0], device=self._device) # for basket
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        self.gripper_forward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        if self.enable_washer_task:
            self.bag_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
                (self._num_envs, 1))
            self.virtual_bag_forward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
                (self._num_envs, 1))
            self.virtual_bag_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
                (self._num_envs, 1))
        else:
            self.bag_forward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
                (self._num_envs, 1))

        self.bag_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

    def get_observations(self) -> dict:
        gripper_pos, gripper_rot = self._elfins._grippers.get_world_poses(clone=False)
        bags_pos, bags_rot = self._bags.get_world_poses(clone=False)
        virtual_bags_pos, virtual_bags_rot = self._virtual_bags.get_world_poses(clone=False)
        elfin_dof_pos = self._elfins.get_joint_positions(clone=False)
        elfin_dof_vel = self._elfins.get_joint_velocities(clone=False)
        self.elfin_dof_pos = elfin_dof_pos
        self.elfin_grasp_pos = gripper_pos
        self.bag_grasp_pos = bags_pos
        self.elfin_grasp_rot = gripper_rot
        self.bag_grasp_rot = bags_rot
        self.virtual_place_pos = virtual_bags_pos
        self.virtual_place_rot = virtual_bags_rot

        dof_pos_scaled = (
                2.0
                * (elfin_dof_pos - self.elfin_dof_lower_limits)
                / (self.elfin_dof_upper_limits - self.elfin_dof_lower_limits)
                - 1.0
        )
        to_target = self.bag_grasp_pos - self.elfin_grasp_pos
        to_virtual = self.virtual_place_pos - self.elfin_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                elfin_dof_vel * self.dof_vel_scale,
                to_target,
                to_virtual
            ),
            dim=-1,
        )

        observations = {
            self._elfins.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.elfin_dof_targets + self.elfin_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.elfin_dof_targets[:] = tensor_clamp(targets, self.elfin_dof_lower_limits, self.elfin_dof_upper_limits)
        env_ids_int32 = torch.arange(self._elfins.count, dtype=torch.int32, device=self._device)

        self._elfins.set_joint_position_targets(self.elfin_dof_targets, indices=env_ids_int32)

        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

        # link the bag and end-effector
        gripper_pos, gripper_rot = self._elfins._grippers.get_world_poses(clone=False)
        bags_pos, bags_rot = self._bags.get_world_poses(clone=False)
        bag_grasping_buf = torch.where(torch.norm(gripper_pos - bags_pos, p=2, dim=-1) < 0.2, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        bag_grasping_ids = bag_grasping_buf.nonzero(as_tuple=False).squeeze(-1)
        self._bags.set_world_poses(gripper_pos[bag_grasping_ids], gripper_rot[bag_grasping_ids], bag_grasping_ids)

        # self._bags.set_world_poses(torch.where(torch.norm(self.elfin_grasp_pos - self.bag_grasp_pos, p=2, dim=-1) < 0.12,
        #                                        self._elfins._grippers.get_world_poses(clone=False),
        #                                        self._bags.get_world_poses(clone=False))
        #                                        )

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset elfin
        pos = tensor_clamp(
            self.elfin_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_elfin_dofs), device=self._device) - 0.5),
            self.elfin_dof_lower_limits,
            self.elfin_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._elfins.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._elfins.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.elfin_dof_targets[env_ids, :] = pos
        self.elfin_dof_pos[env_ids, :] = pos

        self._elfins.set_joint_position_targets(self.elfin_dof_targets[env_ids], indices=indices)
        self._elfins.set_joint_positions(dof_pos, indices=indices)
        self._elfins.set_joint_velocities(dof_vel, indices=indices)

        # reset bags
        self._bags.set_world_poses(
            self.default_bag_pos[self.bag_indices[env_ids]],
            self.default_bag_rot[self.bag_indices[env_ids]],
            self.bag_indices[env_ids].to(torch.int32),
        )

        # reset washers
        if self.enable_washer_task:
            self._washers.set_world_poses(
                self.default_washer_pos[self.washer_indices[env_ids].flatten()],
                self.default_washer_rot[self.washer_indices[env_ids].flatten()],
                self.washer_indices[env_ids].flatten().to(torch.int32),
            )
        else:
            self._baskets.set_world_poses(
                self.default_basket_pos[self.basket_indices[env_ids].flatten()],
                self.default_basket_rot[self.basket_indices[env_ids].flatten()],
                self.basket_indices[env_ids].flatten().to(torch.int32),
            )


        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_elfin_dofs = self._elfins.num_dof
        self.elfin_dof_pos = torch.zeros((self.num_envs, self.num_elfin_dofs), device=self._device)
        dof_limits = self._elfins.get_dof_limits()
        self.elfin_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.elfin_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.elfin_dof_speed_scales = torch.ones_like(self.elfin_dof_lower_limits)
        self.elfin_dof_targets = torch.zeros(
            (self._num_envs, self.num_elfin_dofs), dtype=torch.float, device=self._device
        )

        self.default_bag_pos, self.default_bag_rot = self._bags.get_world_poses()
        self.bag_indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        if self.enable_washer_task:
            self.default_washer_pos, self.default_washer_rot = self._washers.get_world_poses()
            self.washer_indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        else:
            self.default_basket_pos, self.default_basket_rot = self._baskets.get_world_poses()
            self.basket_indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_elfin_reward()

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(torch.norm(self.elfin_grasp_pos - self.virtual_place_pos, p=2, dim=-1) < 0.08,
        #                              torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

    # def compute_grasp_transforms(
    #     self,
    #     drawer_rot,
    #     drawer_pos,
    #     drawer_local_grasp_rot,
    #     drawer_local_grasp_pos,
    # ):

    #     global_drawer_rot, global_drawer_pos = tf_combine(
    #         drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
    #     )

    #     return global_drawer_rot, global_drawer_pos

    def compute_elfin_reward(self):
        # distance from hand to the drawer
        d = torch.norm(self.elfin_grasp_pos - self.bag_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.3, dist_reward * 2, dist_reward)

        axis1 = tf_vector(self.elfin_grasp_rot, self.gripper_forward_axis)
        axis2 = tf_vector(self.bag_grasp_rot, self.bag_forward_axis)
        axis3 = tf_vector(self.elfin_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.bag_grasp_rot, self.bag_up_axis)

        dot1 = torch.bmm(axis1.view(self._num_envs, 1, 3), axis2.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of forward axis for gripper
        dot2 = torch.bmm(axis3.view(self._num_envs, 1, 3), axis4.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)


        # distance to the virtual bag
        d_virtual = torch.norm(self.elfin_grasp_pos - self.virtual_place_pos, p=2, dim=-1)
        dist_reward_virtual = 1.0 / (1.0 + d_virtual ** 2)
        dist_reward_virtual *= dist_reward_virtual
        dist_reward_virtual = torch.where(d_virtual <= 0.12, dist_reward_virtual * 2, dist_reward_virtual)
        dist_reward_virtual = torch.where(d <= 0.02, dist_reward_virtual, torch.zeros_like(dist_reward))

        axis1_virtual = tf_vector(self.elfin_grasp_rot, self.gripper_forward_axis)
        axis2_virtual = tf_vector(self.virtual_place_rot, self.virtual_bag_forward_axis)
        axis3_virtual = tf_vector(self.elfin_grasp_rot, self.gripper_up_axis)
        axis4_virtual = tf_vector(self.virtual_place_rot, self.virtual_bag_up_axis)

        dot1_virtual = torch.bmm(axis1_virtual.view(self._num_envs, 1, 3), axis2_virtual.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of forward axis for gripper
        dot2_virtual = torch.bmm(axis3_virtual.view(self._num_envs, 1, 3), axis4_virtual.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward_virtual = 0.5 * (torch.sign(dot1_virtual) * dot1_virtual ** 2 + torch.sign(dot2_virtual) * dot2_virtual ** 2)











        # # bonus if left finger is above the drawer handle and right below
        # around_handle_reward = torch.zeros_like(rot_reward)
        # around_handle_reward = torch.where(elfin_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
        #                                    torch.where(elfin_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
        #                                                around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
        # # reward for distance of each finger from the drawer
        # finger_dist_reward = torch.zeros_like(rot_reward)
        # lfinger_dist = torch.abs(elfin_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        # rfinger_dist = torch.abs(elfin_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        # finger_dist_reward = torch.where(elfin_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
        #                                  torch.where(elfin_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
        #                                              (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

        # finger_close_reward = torch.zeros_like(rot_reward)
        # finger_close_reward = torch.where(d <=0.03, (0.04 - joint_positions[:, 6]) + (0.04 - joint_positions[:, 7]), finger_close_reward)

        # # regularization on the actions (summed for each environment)
        # action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # # how far the cabinet has been opened out
        # open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        # rewards = dist_reward_scale * dist_reward
        # + rot_reward_scale * rot_reward \
        # + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
        # + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty + finger_close_reward * finger_close_reward_scale

        # rewards = dist_reward_scale * dist_reward - self.action_penalty_scale * action_penalty
        rewards = self.dist_reward_scale * dist_reward + self.rot_reward_scale * rot_reward + self.open_reward_scale * dist_reward_virtual

        # # bonus for opening drawer properly
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        return rewards
