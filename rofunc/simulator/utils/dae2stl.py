# This script converts a .dae file to a .stl file
import os
import rofunc as rf
import aspose.threed as a3d


def dae2stl(dae_files, stl_save_path):
    for dae_file in dae_files:
        scene = a3d.Scene.from_file(dae_file)
        scene.save(os.path.join(stl_save_path, os.path.basename(dae_file).replace('.dae', '.stl')))
        rf.logger.beauty_print('File {} converted to STL.'.format(os.path.basename(dae_file)), type='info')


if __name__ == '__main__':
    dae_folder = './simulator/assets/urdf/curi/meshes'
    dae_files = rf.oslab.list_absl_path(dae_folder, recursive=True, suffix='.dae')

    stl_save_path = './simulator/assets/urdf/curi/all_visual'
    dae2stl(dae_files[6:12], stl_save_path)
