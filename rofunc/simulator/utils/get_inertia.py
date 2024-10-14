import pymeshlab
from rofunc.utils.logger.beauty_logger import beauty_print
import os
def calculate_inertial_tag(file_name=None, mass=-1, pr=8, scale_factor=100):
    ms = pymeshlab.MeshSet()

    if file_name is None:
        print('Please put the input file to the same folder as this script and type in the full name of your file.')
        file_name = input()
    ms.load_new_mesh(file_name)

    if mass < 0:
        print('Please type the mass of your object in kg')
        mass = float(input())

    print('Calculating the center of mass')
    geom = ms.get_geometric_measures()
    com = geom['barycenter']

    print('Scaling the mesh')
    ms.compute_matrix_from_scaling_or_normalization(axisx=scale_factor, axisy=scale_factor, axisz=scale_factor)

    print('Generating the convex hull of the mesh')
    ms.generate_convex_hull()  # TODO only if object is not watertight

    print('Calculating intertia tensor')
    geom = ms.get_geometric_measures()
    volume = geom['mesh_volume']
    tensor = geom['inertia_tensor'] / pow(scale_factor, 2) * mass / volume

    intertial_xml = f'<inertial>\n  <origin xyz="{com[0]:.{pr}f} {com[1]:.{pr}f} {com[2]:.{pr}f}"/>\n  <mass value="{mass:.{pr}f}"/>\n  <inertia ixx="{tensor[0, 0]:.{pr}f}" ixy="{tensor[1, 0]:.{pr}f}" ixz="{tensor[2, 0]:.{pr}f}" iyy="{tensor[1, 1]:.{pr}f}" iyz="{tensor[1, 2]:.{pr}f}" izz="{tensor[2, 2]:.{pr}f}"/>\n</inertial>'
    beauty_print("Interia of {}".format(os.path.basename(file_name)))
    print(intertial_xml)


if __name__ == '__main__':
    path = "./simulator/assets/urdf/zju_humanoid/low_meshes"
    name = "WRIST_UPDOWN_R.STL"
    path = os.path.join(path, name)
    calculate_inertial_tag(path, 1)
