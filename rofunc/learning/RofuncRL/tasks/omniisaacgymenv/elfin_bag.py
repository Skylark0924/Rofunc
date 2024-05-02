# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *

from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.elfin import Elfin
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.views.elfin_view import ElfinView
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.base.rl_task import RLTask

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

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]

        self.enable_washer_task = self._task_cfg["env"]["enable_washer_task"]

        self.dt = 1 / 60.
        self._num_observations = 15  # joint pos (6) and vel (6), dist to bag (3)
        self._num_actions = 6  # joint vel (6)

        RLTask.__init__(self, name, env, offset)
        return

    def set_up_scene(self, scene) -> None:
        self.get_elfin()
        if self.enable_washer_task:
            self.get_rigid_body(
                name_in_scene="washer",
                usd_path="/home/clover/isaac/USD/washer_with_table.usd",
                actor_name="washer",
                translation=torch.tensor([0, -1.6, 0]),
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0])
            )
            self.get_rigid_body(
                name_in_scene="bag",
                usd_path="/home/clover/isaac/USD/bag.usd",
                actor_name="bag",
                translation=torch.tensor([0.35859, -1.03675, 1.1]),
                orientation=torch.tensor([0.7071068, 0.7071068, 0, 0])
            )
        else:
            self.get_rigid_body(
                name_in_scene="basket",
                usd_path="/home/clover/isaac/USD/basket_with_bags.usd",
                actor_name="basket",
                translation=torch.tensor([1.3, 0.0, 0.66687]),
                orientation=torch.tensor([0.7071068, 0.7071068, 0.0, 0.0])
            )
            self.get_rigid_body(
                name_in_scene="bag",
                usd_path="/home/clover/isaac/USD/bag.usd",
                actor_name="bag",
                translation=torch.tensor([1.32231, 0.04733, 0.8]),
                orientation=torch.tensor([0.7071068, 0.7071068, 0, 0])
            )
            self.get_basket_table()

        super().set_up_scene(scene)
        self._elfins = ElfinView(prim_paths_expr="/World/envs/.*/elfin_s20", name="elfin_view")
        scene.add(self._elfins)
        scene.add(self._elfins._grippers)

        if self.enable_washer_task:
            self._washers = RigidPrimView(prim_paths_expr="/World/envs/.*/washer", name="washer_view",
                                          reset_xform_properties=False)
            self._bags = RigidPrimView(prim_paths_expr="/World/envs/.*/bag", name="bag_view",
                                       reset_xform_properties=False)
            scene.add(self._washers)
            scene.add(self._bags)
        else:
            self._baskets = RigidPrimView(prim_paths_expr="/World/envs/.*/basket", name="basket_view",
                                          reset_xform_properties=False)
            self._basket_tables = RigidPrimView(prim_paths_expr="/World/envs/.*/basket_table", name="basket_table_view",
                                                reset_xform_properties=False)
            self._bags = RigidPrimView(prim_paths_expr="/World/envs/.*/bag", name="bag_view",
                                       reset_xform_properties=False)
            scene.add(self._baskets)
            scene.add(self._basket_tables)
            scene.add(self._bags)

        self.init_data()
        return

    def get_elfin(self) -> None:
        elfin = Elfin(
            prim_path=self.default_zero_env_path + "/elfin_s20",
            name="elfin_s20",
            usd_path="/home/clover/isaac/USD/S20_fake_gripper.usd",
            translation=torch.tensor([0.0, 0.0, 0.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        self._sim_config.apply_articulation_settings("elfin_s20", get_prim_at_path(elfin.prim_path),
                                                     self._sim_config.parse_actor_config("elfin_s20"))

    def get_basket_table(self) -> None:
        basket_table = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/basket_table",
            name="basket_table",
            color=torch.tensor([0.2, 0.4, 0.6]),
            scale=torch.tensor([1.0, 1.0, 0.5]),
            translation=torch.tensor([1.3, 0.0, 0.25])
        )
        self._sim_config.apply_articulation_settings("basket_table", get_prim_at_path(basket_table.prim_path),
                                                     self._sim_config.parse_actor_config("basket_table"))

    def get_rigid_body(self, name_in_scene, usd_path, actor_name, translation, orientation) -> None:
        prim_path = self.default_zero_env_path + "/" + name_in_scene
        add_reference_to_stage(usd_path, prim_path)
        rigid_body = RigidPrim(prim_path=prim_path, name=name_in_scene, translation=translation,
                               orientation=orientation)
        self._sim_config.apply_articulation_settings(actor_name, get_prim_at_path(rigid_body.prim_path),
                                                     self._sim_config.parse_actor_config(actor_name))

    def init_data(self) -> None:
        # Set the initial robot configuration
        if self.enable_washer_task:
            self.elfin_default_dof_pos = torch.tensor([1.57, 0.0, 0.0, -1.57, -1.57, 0.0],
                                                      device=self._device)
        else:
            self.elfin_default_dof_pos = torch.tensor([3.14, 0.0, 0.0, -1.57, -1.57, 0.0],
                                                      device=self._device)
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        # Set the local axis for the gripper and bag
        self.gripper_forward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        if self.enable_washer_task:
            self.bag_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
                (self._num_envs, 1))
        else:
            self.bag_forward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
                (self._num_envs, 1))
        self.bag_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

    def get_observations(self) -> dict:
        self.elfin_grasp_pos, self.elfin_grasp_rot = self._elfins._grippers.get_world_poses(clone=False)
        self.bag_grasp_pos, self.bag_grasp_rot = self._bags.get_world_poses(clone=False)
        self.elfin_dof_pos = self._elfins.get_joint_positions(clone=False)
        elfin_dof_vel = self._elfins.get_joint_velocities(clone=False)

        dof_pos_scaled = (
                2.0
                * (self.elfin_dof_pos - self.elfin_dof_lower_limits)
                / (self.elfin_dof_upper_limits - self.elfin_dof_lower_limits)
                - 1.0
        )
        to_target = self.bag_grasp_pos - self.elfin_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                elfin_dof_vel * self.dof_vel_scale,
                to_target
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

        # # Attach the bag to the gripper after they become close enough
        # gripper_pos, gripper_rot = self._elfins._grippers.get_world_poses(clone=False)
        # bag_pos, bag_rot = self._bags.get_world_poses(clone=False)
        # bag_grasping_buf = torch.where(torch.norm(gripper_pos - bag_pos, p=2, dim=-1) < 0.08,
        #                                torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        # bag_grasping_ids = bag_grasping_buf.nonzero(as_tuple=False).squeeze(-1)
        # self._bags.set_world_poses(gripper_pos[bag_grasping_ids], gripper_rot[bag_grasping_ids], bag_grasping_ids)

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

        self._bags.set_world_poses(
            self.default_bag_pos[self.bag_indices[env_ids]],
            self.default_bag_rot[self.bag_indices[env_ids]],
            self.bag_indices[env_ids].to(torch.int32),
        )
        if self.enable_washer_task:
            self._washers.set_world_poses(
                self.default_washer_pos[self.washer_indices[env_ids]],
                self.default_washer_rot[self.washer_indices[env_ids]],
                self.washer_indices[env_ids].to(torch.int32),
            )
        else:
            self._baskets.set_world_poses(
                self.default_basket_pos[self.basket_indices[env_ids]],
                self.default_basket_rot[self.basket_indices[env_ids]],
                self.basket_indices[env_ids].to(torch.int32),
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
        self.elfin_dof_targets = torch.zeros((self._num_envs, self.num_elfin_dofs), dtype=torch.float,
                                             device=self._device)
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
        # reset if the bag and gripper become close enough or max length reached
        self.reset_buf = torch.where(torch.norm(self.elfin_grasp_pos - self.bag_grasp_pos, p=2, dim=-1) < 0.03,
                                     torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

    def compute_elfin_reward(self):
        # distance from the gripper to bag
        d = torch.norm(self.elfin_grasp_pos - self.bag_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.1, dist_reward * 2, dist_reward)

        # calculate the global axis
        axis1 = tf_vector(self.elfin_grasp_rot, self.gripper_forward_axis)
        axis2 = tf_vector(self.bag_grasp_rot, self.bag_forward_axis)
        axis3 = tf_vector(self.elfin_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.bag_grasp_rot, self.bag_up_axis)

        dot1 = torch.bmm(axis1.view(self._num_envs, 1, 3), axis2.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of forward axis for gripper
        dot2 = torch.bmm(axis3.view(self._num_envs, 1, 3), axis4.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of up axis for gripper
        # reward for matching the orientation of the gripper to bag
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        rewards = self.dist_reward_scale * dist_reward \
                   + self.rot_reward_scale * rot_reward \
                   # - self.action_penalty_scale * action_penalty

        return rewards
