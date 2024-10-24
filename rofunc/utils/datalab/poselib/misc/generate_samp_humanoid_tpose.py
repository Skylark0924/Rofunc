#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

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
