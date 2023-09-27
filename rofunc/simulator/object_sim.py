# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path


def show(args, object_name, asset_root=None):
    """

    Args:
        args:
        asset_root: the location of `assets` folder, e.g., /home/ubuntu/anaconda3/envs/plast/lib/python3.7/site-packages/rofunc/simulator/assets

    Returns:

    """
    from rofunc.simulator.base_sim import init_sim, init_env
    from rofunc.utils.logger.beauty_logger import beauty_print

    beauty_print("Show the {} in the interactive mode".format(object_name), 1)

    # Initial gym and sim
    gym, sim_params, sim, viewer = init_sim(args)

    # Load CURI asset and set the env
    if asset_root is None:
        from rofunc.utils.oslab import get_rofunc_path
        asset_root = os.path.join(get_rofunc_path(), "simulator/assets")
    if object_name == "Cabinet":
        asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
    elif object_name == "Cabinet2":
        asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"
    elif object_name == "tray":
        asset_file = "urdf/tray/tray.urdf"
    elif object_name == "banana":
        asset_file = "urdf/ycb/011_banana/011_banana.urdf"
    elif object_name == "meat_can":
        asset_file = "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf"
    elif object_name == "mug":
        asset_file = "urdf/ycb/025_mug/025_mug.urdf"
    elif object_name == "brick":
        asset_file = "urdf/ycb/061_foam_brick/061_foam_brick.urdf"

    init_env(gym, sim, asset_root, asset_file, num_envs=5, spacing=3.0, fix_base_link=False,
             flip_visual_attachments=False, init_pose_vec=(0, 0.4, 0.0))

    while not gym.query_viewer_has_closed(viewer):
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)


if __name__ == '__main__':
    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False
    show(args)
