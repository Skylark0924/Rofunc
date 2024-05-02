# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from pxr import UsdGeom

from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.aubo import Aubo
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.basket import Basket
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.views.aubo_view import AuboView
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.views.basket_view import BasketView
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.bag import Bag
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.articulations.views.bag_view import BagView
from rofunc.learning.RofuncRL.tasks.omniisaacgymenv.base.rl_task import RLTask


class AuboCubeOmniTask(RLTask):
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

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        self._num_observations = 22  # joint pos (8) and vel (8), dist to cabinet (3), drawer pos and vel (2)
        self._num_actions = 8  # joint vel (8)

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:

        self.get_aubo()
        self.get_basket()
        self.get_bag()
        if self.num_props > 0:
            self.get_props()

        super().set_up_scene(scene)

        self._aubos = AuboView(prim_paths_expr="/World/envs/.*/aubo_i12", name="aubo_view")  # many envs running
        self._baskets = BasketView(prim_paths_expr="/World/envs/.*/basket", name="basket_view")
        self._bags = BagView(prim_paths_expr="/World/envs/.*/bag", name="bag_view")


        scene.add(self._aubos)
        scene.add(self._aubos._hands)
        scene.add(self._aubos._lfingers)
        scene.add(self._aubos._rfingers)
        scene.add(self._baskets)
        scene.add(self._bags)
        # scene.add(self._bags._grasp_points)
        # scene.add(self._bags._pillows)
        # scene.add(self._bags._pillows_01)
        # scene.add(self._bags._pillows_02)


        if self.num_props > 0:
            self._props = RigidPrimView(prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view",
                                        reset_xform_properties=False)
            scene.add(self._props)

        self.init_data()
        return

    def get_aubo(self):
        aubo = Aubo(prim_path=self.default_zero_env_path + "/aubo_i12", name="aubo_i12",
                    usd_path="/home/clover/isaac/aubo_instanceable.usd")
        self._sim_config.apply_articulation_settings("aubo_i12", get_prim_at_path(aubo.prim_path),
                                                     self._sim_config.parse_actor_config("aubo_i12"))

    def get_basket(self):
        basket = Basket(prim_path=self.default_zero_env_path + "/basket",
                        name="basket",
                        usd_path="/home/clover/isaac/basket.usd",
                        translation=torch.tensor([0.2, 0.0, 0.2]), # 0.1661
                        orientation=torch.tensor([0.7071, 0.7071, 0.0, 0.0])) # w, x, y, z
        self._sim_config.apply_articulation_settings("basket", get_prim_at_path(basket.prim_path),
                                                     self._sim_config.parse_actor_config("basket"))
    def get_bag(self):
        bag = Bag(prim_path=self.default_zero_env_path + "/bag",
                        name="bag",
                        usd_path="/home/clover/isaac/bag.usd",
                        translation=torch.tensor([0.2, 0.0, 0.2]),
                        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0])) # w, x, y, z
        self._sim_config.apply_articulation_settings("bag", get_prim_at_path(bag.prim_path),
                                                     self._sim_config.parse_actor_config("bag"))

    def get_props(self):
        prop_cloner = Cloner()
        tabletop_pos = torch.tensor([0.4, 0.0, 0.3])
        prop_color = torch.tensor([0.2, 0.4, 0.6])

        props_per_row = int(math.ceil(math.sqrt(self.num_props)))
        prop_size = 0.001
        prop_spacing = 0.18
        xmin = -0.5 * prop_spacing * (props_per_row - 1)  # the props are around the local origin
        zmin = -0.5 * prop_spacing * (props_per_row - 1)
        prop_count = 0

        prop_pos = []
        for j in range(props_per_row):
            prop_up = zmin + j * prop_spacing
            for k in range(props_per_row):
                if prop_count >= self.num_props:
                    break
                propx = xmin + k * prop_spacing
                prop_pos.append([propx, prop_up, 0.0])

                prop_count += 1

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            color=prop_color,
            size=prop_size,
            density=100.0
        )
        self._sim_config.apply_articulation_settings("prop", get_prim_at_path(prop.prim_path),
                                                     self._sim_config.parse_actor_config("prop"))

        prop_paths = [f"{self.default_zero_env_path}/prop/prop_{j}" for j in range(self.num_props)]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + "/prop/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos) + tabletop_pos.numpy(),
            replicate_physics=False
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

        stage = get_current_stage()
        hand_pose = get_env_local_pose(self._env_pos[0],
                                       UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/aubo_i12/wrist3_link")),
                                       self._device)
        lfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/aubo_i12/panda_leftfinger")),
            self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/aubo_i12/panda_rightfinger")),
            self._device
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        grasp_pose_axis = 1
        aubo_local_grasp_pose_rot, aubo_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos,
                                                                    finger_pose[3:7], finger_pose[0:3])
        # aubo_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.aubo_local_grasp_pos = aubo_local_pose_pos.repeat((self._num_envs, 1))
        self.aubo_local_grasp_rot = aubo_local_grasp_pose_rot.repeat((self._num_envs, 1))

        self.pillow_local_grasp_pos = torch.tensor([0.2061, 0.06162, 0.28934], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.pillow_local_grasp_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1))

        self.aubo_default_dof_pos = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.035, 0.035], device=self._device
        )  # modified

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._aubos._hands.get_world_poses(clone=False)
        prop_pos, prop_rot = self._props.get_world_poses(clone=False)
        pillow_pos, pillow_rot = self._bags.get_world_poses(clone=False)
        cubeB_pos, cubeB_rot = prop_pos[::2, :], prop_rot[::2, :]
        cubeA_pos, cubeA_rot = prop_pos[1::2, :], prop_rot[1::2, :]
        aubo_dof_pos = self._aubos.get_joint_positions(clone=False)
        aubo_dof_vel = self._aubos.get_joint_velocities(clone=False)
        self.aubo_dof_pos = aubo_dof_pos
        self.cubeA_pos, self.cubeA_rot = cubeA_pos, cubeA_rot
        self.cubeB_pos, self.cubeB_rot = cubeB_pos, cubeB_rot

        self.aubo_grasp_rot, self.aubo_grasp_pos, self.pillow_grasp_rot, self.pillow_grasp_pos = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.aubo_local_grasp_rot,
            self.aubo_local_grasp_pos,
            pillow_rot,
            pillow_pos,
            self.pillow_local_grasp_rot,
            self.pillow_local_grasp_pos,
        )

        self.aubo_lfinger_pos, self.aubo_lfinger_rot = self._aubos._lfingers.get_world_poses(clone=False)
        self.aubo_rfinger_pos, self.aubo_rfinger_rot = self._aubos._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
                2.0
                * (aubo_dof_pos - self.aubo_dof_lower_limits)
                / (self.aubo_dof_upper_limits - self.aubo_dof_lower_limits)
                - 1.0
        )
        cubeA_pos_relative = self.cubeA_pos - self.aubo_grasp_pos
        cubeA_to_cubeB_pos = self.cubeB_pos - self.cubeA_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                aubo_dof_vel * self.dof_vel_scale,
                cubeA_pos_relative,
                cubeA_to_cubeB_pos
            ),
            dim=-1,
        )

        observations = {
            self._aubos.name: {
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
        targets = self.aubo_dof_targets + self.aubo_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.aubo_dof_targets[:] = tensor_clamp(targets, self.aubo_dof_lower_limits, self.aubo_dof_upper_limits)
        env_ids_int32 = torch.arange(self._aubos.count, dtype=torch.int32, device=self._device)

        self._aubos.set_joint_position_targets(self.aubo_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset aubo
        pos = tensor_clamp(
            self.aubo_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_aubo_dofs), device=self._device) - 0.5),
            self.aubo_dof_lower_limits,
            self.aubo_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._aubos.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._aubos.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.aubo_dof_targets[env_ids, :] = pos
        self.aubo_dof_pos[env_ids, :] = pos

        # reset basket

        # reset props
        if self.num_props > 0:
            self._props.set_world_poses(
                self.default_prop_pos[self.prop_indices[env_ids].flatten()],
                self.default_prop_rot[self.prop_indices[env_ids].flatten()],
                self.prop_indices[env_ids].flatten().to(torch.int32)
            )

        self._aubos.set_joint_position_targets(self.aubo_dof_targets[env_ids], indices=indices)
        self._aubos.set_joint_positions(dof_pos, indices=indices)
        self._aubos.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.num_aubo_dofs = self._aubos.num_dof
        self.aubo_dof_pos = torch.zeros((self.num_envs, self.num_aubo_dofs), device=self._device)
        dof_limits = self._aubos.get_dof_limits()
        self.aubo_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.aubo_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.aubo_dof_speed_scales = torch.ones_like(self.aubo_dof_lower_limits)
        self.aubo_dof_speed_scales[self._aubos.gripper_indices] = 0.1
        self.aubo_dof_targets = torch.zeros(
            (self._num_envs, self.num_aubo_dofs), dtype=torch.float, device=self._device
        )

        if self.num_props > 0:
            self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
            self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
                self._num_envs, self.num_props
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_aubo_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.aubo_grasp_pos, self.aubo_grasp_rot, self.cubeA_pos, self.cubeA_rot, self.pillow_grasp_pos, self.pillow_local_grasp_rot,
            self.aubo_lfinger_pos, self.aubo_rfinger_pos,
        )

    def is_done(self) -> None:
        # refer to how to compute resets in compute_franka_reward
        # self.reset_buf = torch.where(self.cubeA_pos[:, 2] > 0.06, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

    def compute_grasp_transforms(
            self,
            hand_rot,
            hand_pos,
            aubo_local_grasp_rot,
            aubo_local_grasp_pos,
            pillow_rot,
            pillow_pos,
            pillow_local_grasp_rot,
            pillow_local_grasp_pos,
    ):

        global_aubo_rot, global_aubo_pos = tf_combine(
            hand_rot, hand_pos, aubo_local_grasp_rot, aubo_local_grasp_pos
        )
        global_pillow_rot, global_pillow_pos = tf_combine(
            pillow_rot, pillow_pos, pillow_local_grasp_rot, pillow_local_grasp_pos
        )

        return global_aubo_rot, global_aubo_pos, global_pillow_rot, global_pillow_pos

    def compute_aubo_reward(
            self, reset_buf, progress_buf, actions,
            aubo_grasp_pos, aubo_grasp_rot, cubeA_pos, cubeA_rot, pillow_grasp_pos, pillow_grasp_rot,
            aubo_lfinger_pos, aubo_rfinger_pos,
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # # Compute per-env physical parameters
        # prop_size = 0.08
        # cubeA_size = prop_size
        # cubeB_size = prop_size
        # target_height = cubeB_size + cubeA_size / 2.0

        # distance from hand to the cube
        d = torch.norm(aubo_grasp_pos - pillow_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.06, dist_reward * 2, dist_reward)

        gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        gripper_right_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        pillow_forward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        pillow_down_axis = torch.tensor([0, 0, -1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        axis1 = tf_vector(aubo_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(pillow_grasp_rot, pillow_down_axis)
        axis3 = tf_vector(aubo_grasp_rot, gripper_right_axis)
        axis4 = tf_vector(pillow_grasp_rot, pillow_forward_axis)

        dot1 = (
            torch.bmm(axis1.view(self._num_envs, 1, 3), axis2.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(self._num_envs, 1, 3), axis4.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of right axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # finger_close_reward = torch.zeros_like(rot_reward)
        # joint_positions = self._aubos.get_joint_positions(clone=False)
        # finger_close_reward = torch.where(d <= 0.04, (0.04 - joint_positions[:, 6]) + (0.04 - joint_positions[:, 7]),
        #                                   finger_close_reward)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions ** 2, dim=-1)

        # # distance from hand to the cubeA
        # d = torch.norm(cubeA_pos - aubo_grasp_pos, dim=-1)
        # d_lf = torch.norm(cubeA_pos - aubo_lfinger_pos, dim=-1)
        # d_rf = torch.norm(cubeA_pos - aubo_rfinger_pos, dim=-1)
        # # dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
        # dist_reward = - (d + d_lf + d_rf) / 3
        # self.dist_reward = dist_reward

        # # reward for lifting cubeA
        # table_height = 0.4053
        # cubeA_height = cubeA_pos[:, 2] - table_height
        # cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
        # lift_reward = cubeA_lifted

        # # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
        # cubeA_to_cubeB_pos = cubeB_pos - cubeA_pos
        # offset = torch.zeros_like(cubeA_to_cubeB_pos)
        # offset[:, 2] = (cubeA_size + cubeB_size) / 2
        # d_ab = torch.norm(cubeA_to_cubeB_pos + offset, dim=-1)
        # align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted
        #
        # # Dist reward is maximum of dist and align reward
        # dist_reward = torch.max(dist_reward, align_reward)
        #
        # # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
        # cubeA_align_cubeB = (torch.norm(cubeA_to_cubeB_pos[:, :2], dim=-1) < 0.02)
        # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
        # gripper_away_from_cubeA = (d > 0.04)
        # stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA
        # self.stack_reward = stack_reward
        # Compose rewards

        # We either provide the stack reward or the align + dist reward
        dist_reward_scale = 2.0
        rot_reward_scale = 0.5
        action_penalty_scale = 0.01
        finger_close_reward_scale = 0 # 10.0

        # rewards = torch.where(
        #     stack_reward,
        #     stack_reward_scale * stack_reward,
        #     dist_reward_scale * dist_reward + lift_reward_scale * lift_reward +
        #     align_reward_scale * align_reward,
        # )

        rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward - action_penalty_scale * action_penalty
        # rewards = dist_reward

        return rewards
