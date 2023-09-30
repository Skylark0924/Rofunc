"""
Apply Forces and Torques On CURI
=================================

This example shows how to apply rigid body forces and torques using the tensor API.
"""

# TODO: Reformat


import os

import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch

from rofunc.simulator.base_sim import init_sim, init_env
from rofunc.utils.oslab import get_rofunc_path

# parse arguments
args = gymutil.parse_arguments(
    description="Example of applying forces to bodies at given positions")

# Initial gym and sim
gym, sim_params, sim, viewer = init_sim(args)

asset_root = os.path.join(get_rofunc_path(), "simulator/assets")
asset_file = "urdf/curi/urdf/curi_isaacgym.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())
envs, curi_handles = init_env(gym, sim, asset_root, asset_file, num_envs=1, fix_base_link=True)

num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies)

# set random seed
np.random.seed(17)
num_envs = 1
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

gym.prepare_sim(sim)

rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_tensor)
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)

torque_amt = 100000

frame_count = 0
while not gym.query_viewer_has_closed(viewer):

    if (frame_count - 99) % 200 == 0:
        # gym.refresh_rigid_body_state_tensor(sim)

        # set forces and force positions for ant root bodies (first body in each env)
        forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        torques = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        # forces[:, 9, 1] = 3000
        torques[:, 9, 1] = torque_amt
        gym.apply_rigid_body_force_tensors(sim, None, gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        torque_amt = -torque_amt

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
