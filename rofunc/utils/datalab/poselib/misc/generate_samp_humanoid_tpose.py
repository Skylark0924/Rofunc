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

import os
import click

from rofunc.utils.datalab.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from rofunc.utils.datalab.poselib.poselib.visualization.common import plot_skeleton_state


def get_tpose_from_fbx(fbx_file_path, save_path, verbose=False):
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file_path,
        root_joint="Hips",
        fps=60
    )

    source_tpose = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, motion.rotation[0],
                                                                    motion.root_translation[0], is_local=True)

    source_tpose.to_file(save_path)

    if verbose:
        plot_skeleton_state(source_tpose)


@click.command()
@click.argument("fbx_name")
def main(fbx_name):
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data"
    )
    os.makedirs(data_dir, exist_ok=True)
    fbx_files = [os.path.join(data_dir, f"{fbx_name}.fbx")]
    for fbx in fbx_files:
        save_path = os.path.join(data_dir, "new_tpose.npy")
        get_tpose_from_fbx(fbx, save_path, verbose=True)


if __name__ == '__main__':
    main()
