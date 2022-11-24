import os.path
from typing import List

import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch

from rofunc.simulator.base.base_sim import init_sim, init_env, init_attractor, get_num_bodies, get_robot_state
from rofunc.utils.logger.beauty_logger import beauty_print


def show(args, asset_root=None):
    """

    Args:
        args:
        asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets

    Returns:

    """
    beauty_print("Show the CURI mini simulator in the interactive mode", 1)

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args)

    # Load CURI asset and set the env
    if asset_root is None:
        import site
        pip_root_path = site.getsitepackages()[0]
        asset_root = os.path.join(pip_root_path, "rofunc/simulator/assets")
    asset_file = "urdf/walker/urdf/walker.urdf"
    init_env(gym, sim, asset_root, asset_file, num_envs=5, spacing=3.0, fix_base_link=True,
             flip_visual_attachments=False, init_pose_vec=gymapi.Vec3(0, 2.0, 0.0))

    while not gym.query_viewer_has_closed(viewer):
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
