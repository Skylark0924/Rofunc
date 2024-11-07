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

import logging
import os
import subprocess
import traceback

import lxml.etree as et
import numpy as np
import trimesh
from tqdm import tqdm
from trimesh.decomposition import convex_decomposition
from trimesh.exchange.export import export_mesh
from trimesh.version import __version__ as trimesh_version
from rofunc.utils.oslab.path import get_rofunc_path

YCB_MASS = {
    "001_chips_can": 0.205,
    "002_master_chef_can": 0.414,
    "003_cracker_box": 0.411,
    "004_sugar_box": 0.514,
    "005_tomato_soup_can": 0.349,
    "006_mustard_bottle": 0.404,
    "007_tuna_fish_can": 0.171,
    "008_pudding_box": 0.187,
    "009_gelatin_box": 0.097,
    "010_potted_meat_can": 0.370,
    "011_banana": 0.066,
    "012_strawberry": 0.018,
    "013_apple": 0.068,
    "014_lemon": 0.029,
    "015_peach": 0.033,
    "016_pear": 0.049,
    "017_orange": 0.047,
    "018_plum": 0.025,
    "019_pitcher_base": 0.178,
    "021_bleach_cleanser": 0.479,  # check
    "022_windex_bottle": 1.022,
    "024_bowl": 0.147,
    "025_mug": 0.118,
    "026_sponge": 0.0062,
    "027_skillet": 0.950,
    "028_skillet_lid": 0.652,
    "029_plate": 0.279,
    "030_fork": 0.034,
    "031_spoon": 0.018,
    "032_knife": 0.031,
    "033_spatula": 0.0515,
    "035_power_drill": 0.895,
    "036_wood_block": 0.729,
    "037_scissors": 0.082,
    "038_padlock": 0.208,
    "040_large_marker": 0.0158,
    "041_small_marker": 0.0082,
    "042_adjustable_wrench": 0.252,
    "043_phillips_screwdriver": 0.097,
    "044_flat_screwdriver": 0.0984,
    "048_hammer": 0.665,
    "050_medium_clamp": 0.059,
    "051_large_clamp": 0.125,
    "052_extra_large_clamp": 0.202,
    "053_mini_soccer_ball": 0.123,
    "054_softball": 0.191,
    "055_baseball": 0.138,
    "056_tennis_ball": 0.057,
    "057_racquetball": 0.041,
    "058_golf_ball": 0.046,
    "059_chain": 0.1,
    "061_foam_brick": 0.028,
    "062_dice": 0.0052,
    "063-a_marbles": 0.020,
    "063-b_marbles": 0.020,
    "063-c_marbles": 0.020,
    "063-d_marbles": 0.020,
    "063-e_marbles": 0.020,
    "063-f_marbles": 0.020,
    "065-a_cups": 0.020,
    "065-b_cups": 0.020,
    "065-c_cups": 0.020,
    "065-d_cups": 0.020,
    "065-e_cups": 0.020,
    "065-f_cups": 0.020,
    "065-g_cups": 0.020,
    "065-h_cups": 0.020,
    "065-i_cups": 0.020,
    "065-j_cups": 0.020,
    "070-a_colored_wood_blocks": 0.018,
    "070-b_colored_wood_blocks": 0.018,
    "071_nine_hole_peg_test": 1.435,
    "072-a_toy_airplane": 0.020,
    "072-b_toy_airplane": 0.020,
    "072-c_toy_airplane": 0.020,
    "072-d_toy_airplane": 0.020,
    "072-e_toy_airplane": 0.020,
    "072-f_toy_airplane": 0.020,
    "072-g_toy_airplane": 0.020,
    "072-h_toy_airplane": 0.020,
    "072-i_toy_airplane": 0.020,
    "072-j_toy_airplane": 0.020,
    "072-k_toy_airplane": 0.020,
    "073-a_lego_duplo": 0.020,
    "073-b_lego_duplo": 0.020,
    "073-c_lego_duplo": 0.020,
    "073-d_lego_duplo": 0.020,
    "073-e_lego_duplo": 0.020,
    "073-f_lego_duplo": 0.020,
    "073-g_lego_duplo": 0.020,
    "073-h_lego_duplo": 0.020,
    "073-i_lego_duplo": 0.020,
    "073-j_lego_duplo": 0.020,
    "073-k_lego_duplo": 0.020,
    "073-l_lego_duplo": 0.020,
    "073-m_lego_duplo": 0.020,
    "076_timer": 0.101,
    "077_rubiks_cube": 0.094,
}


def export_urdf(mesh,
                directory,
                scale=1.0,
                color=[0.75, 0.75, 0.75],
                convex_decompose=False,
                **kwargs):
    """
    Convert a Trimesh object into a URDF package for physics simulation.
    This breaks the mesh into convex pieces and writes them to the same
    directory as the .urdf file.

    :param mesh: Trimesh object
    :param directory: str, the directory name for the URDF package
    :param scale:
    :param color:
    :param convex_decompose: bool, whether to decompose the mesh into convex pieces
    :param kwargs:
    :return: mesh: The decomposed mesh
    """

    # Extract the save directory and the file name
    fullpath = os.path.abspath(directory)
    name = os.path.basename(fullpath)
    _, ext = os.path.splitext(name)
    minimum_mass = kwargs.get('minimum_mass', 0.001)

    if ext != '':
        raise ValueError('URDF path must be a directory!')

    # Create directory if needed
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    elif not os.path.isdir(fullpath):
        raise ValueError('URDF path must be a directory!')

    # Perform a convex decomposition
    if convex_decompose:
        try:
            convex_pieces = convex_decomposition(mesh, **kwargs)
            if not isinstance(convex_pieces, list):
                convex_pieces = [convex_pieces]
        except subprocess.CalledProcessError:
            convex_pieces = [mesh]
    else:
        convex_pieces = [mesh]

    # Get the effective density of the mesh
    effective_density = mesh.volume / sum([m.volume for m in convex_pieces])

    # open an XML tree
    root = et.Element('robot', name='root')

    # Loop through all pieces, adding each as a link
    prev_link_name = None
    for i, piece in enumerate(convex_pieces):

        if convex_decompose:
            # Save each convex piece out to a file
            piece_name = '{}_convex_piece_{}'.format(name, i)
            piece_filename = '{}.obj'.format(piece_name)
            piece_filepath = os.path.join(fullpath, piece_filename)
            export_mesh(piece, piece_filepath)
            geom_name = '{}'.format(piece_filename)
            visual_name = geom_name
        else:
            piece_name = name
            visual_name = 'google_16k/textured.obj'
            geom_name = 'google_16k/nontextured.stl'

        # Set the mass properties of the piece
        piece.center_mass = mesh.center_mass
        piece.density = effective_density * mesh.density

        link_name = 'link_{}'.format(piece_name)
        I = [['{:.2E}'.format(y) for y in x] for x in piece.moment_inertia]

        # Write the link out to the XML Tree
        link = et.SubElement(root, 'link', name=link_name)
        # contact = et.SubElement(link, 'contact')
        # lateral_friction = et.SubElement(contact, 'lateral_friction', value="1.0")
        # rolling_friction = et.SubElement(contact, 'rolling_friction', value="0.0")
        # contact_cfm = et.SubElement(contact, 'contact_cfm', value="0.0")
        # contact_erp = et.SubElement(contact, 'rolling_friction', value="1.0")
        #         <contact>
        #   <lateral_friction value="1.0"/>
        #   <rolling_friction value="0.0"/>
        #   <contact_cfm value="0.0"/>
        #   <contact_erp value="1.0"/>
        # </contact>

        # Inertial information
        inertial = et.SubElement(link, 'inertial')
        et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
        if convex_decompose:
            mass = np.maximum(piece.mass.reshape(-1)[0], minimum_mass)
            et.SubElement(inertial, 'inertia', ixx=I[0][0], ixy=I[0][1], ixz=I[0][2],
                          iyy=I[1][1], iyz=I[1][2], izz=I[2][2])
        else:
            mass = np.maximum(YCB_MASS[piece_name], minimum_mass)
            et.SubElement(inertial, 'inertia', ixx='0.0001', ixy='0.0', ixz='0.0',
                          iyy='0.0001', iyz='0.0', izz='0.0001')
        et.SubElement(inertial, 'mass', value='{}'.format(mass))

        # Visual Information
        visual = et.SubElement(link, 'visual')
        et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(visual, 'geometry')
        et.SubElement(geometry, 'mesh', filename=visual_name,
                      scale="{} {} {}".format(scale, scale, scale))
        material = et.SubElement(visual, 'material', name='')
        et.SubElement(material, 'color', rgba="{:.2E} {:.2E} {:.2E} 1".format(
            color[0], color[1], color[2]))

        # Collision Information
        collision = et.SubElement(link, 'collision')
        et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(collision, 'geometry')
        et.SubElement(geometry, 'mesh', filename=geom_name,
                      scale="{} {} {}".format(scale, scale, scale))

        # Create rigid joint to previous link
        if prev_link_name is not None:
            joint_name = '{}_joint'.format(link_name)
            joint = et.SubElement(root, 'joint', name=joint_name, type='fixed')
            et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
            et.SubElement(joint, 'parent', link=prev_link_name)
            et.SubElement(joint, 'child', link=link_name)

        prev_link_name = link_name

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = '{}.urdf'.format(name)
    tree.write(os.path.join(fullpath, urdf_filename), pretty_print=True)

    # Write Gazebo config file
    root = et.Element('model')
    model = et.SubElement(root, 'name')
    model.text = name
    version = et.SubElement(root, 'version')
    version.text = '1.0'
    sdf = et.SubElement(root, 'sdf', version='1.4')
    sdf.text = '{}.urdf'.format(name)

    author = et.SubElement(root, 'author')
    et.SubElement(author, 'name').text = 'Junjia LIU'.format(trimesh_version)
    et.SubElement(author, 'email').text = 'jjliu@mae.cuhk.edu.hk'

    description = et.SubElement(root, 'description')
    description.text = name

    tree = et.ElementTree(root)
    tree.write(os.path.join(fullpath, 'model.config'))

    return np.sum(convex_pieces)


def create_urdf_file(output_directory, input_mesh):
    mesh = trimesh.load(input_mesh)

    folder_name = input_mesh.split('/')[-3].split('.')[0]

    outpath = "%s/%s" % (output_directory, folder_name)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    export_urdf(mesh, outpath)

    return outpath + '/%s%s' % (folder_name, '.urdf')


def ycb2urdf():
    ycb_output_directory = "../assets/urdf/ycb"
    urdf_output_directory = "../assets/urdf/ycb"
    # ycb_output_directory = os.path.join(get_rofunc_path(), "simulator/assets/urdf/ycb")
    # urdf_output_directory = os.path.join(get_rofunc_path(), "simulator/assets/urdf/ycb")
    objects = os.listdir(ycb_output_directory)
    with tqdm(total=len(objects)) as t:
        for obj in objects:
            if not obj[0] == "0":
                continue
            try:
                urdf_root_path = create_urdf_file(output_directory=urdf_output_directory,
                                                  input_mesh=ycb_output_directory + ('/%s/%s') % (
                                                      obj, 'google_16k/nontextured.stl'))
            except Exception as e:
                logging.error(e)
                traceback.print_exc()
                # continue
                exit(1)
            t.set_postfix_str(f"{urdf_root_path}")
            t.update(1)

        # if test_object:
        #     test_urdf(urdf_root_path=urdf_root_path)


if __name__ == '__main__':
    ycb2urdf()
