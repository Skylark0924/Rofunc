
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class ElfinView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "ElfinView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        # self._grippers = RigidPrimView(prim_paths_expr="/World/envs/.*/elfin_s20/ee_link", name="grippers_view", reset_xform_properties=False)
        self._grippers = RigidPrimView(prim_paths_expr="/World/envs/.*/elfin_s20/simplified_gripper", name="grippers_view", reset_xform_properties=False)


    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)