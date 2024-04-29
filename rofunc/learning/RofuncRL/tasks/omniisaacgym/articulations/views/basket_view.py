
from typing import Optional

from omni.isaac.core.prims import RigidPrimView


class BasketView(RigidPrimView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "BasketView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

