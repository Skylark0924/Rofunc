# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import math
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema


class Elfin(Robot):
    def __init__(
            self,
            prim_path: str,
            usd_path: str,
            name: Optional[str] = "elfin_s20",
            translation: Optional[torch.tensor] = None,
            orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            raise ValueError("Please provide the asset path.")

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        # prim = get_prim_at_path("/World/envs/env_0/task/elfin_s20/elfin_base/elfin_joint1")
        # if not prim.IsValid():
        #     print("Invalid prim")
        # else:
        #     print("Valid prim")
        dof_paths = [
            "elfin_base/elfin_joint1",
            "elfin_link1/elfin_joint2",
            "elfin_link2/elfin_joint3",
            "elfin_link3/elfin_joint4",
            "elfin_link4/elfin_joint5",
            "elfin_link5/elfin_joint6"
        ]

        drive_type = ["angular"] * 6
        default_dof_pos = [math.degrees(x) for x in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        stiffness = [400 * np.pi / 180] * 6
        damping = [80 * np.pi / 180] * 6
        max_force = [87, 87, 87, 12, 12, 12]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.61, 2.61, 2.61]]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )
            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i])
