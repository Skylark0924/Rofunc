import logging
import math
import os
import subprocess
import sys

import lxml.etree as et
import numpy as np
import trimesh
from trimesh.decomposition import convex_decomposition
from trimesh.version import __version__ as trimesh_version
from trimesh.exchange.export import export_mesh
from tqdm import tqdm


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
        contact = et.SubElement(link, 'contact')
        lateral_friction = et.SubElement(contact, 'lateral_friction', value="1.0")
        rolling_friction = et.SubElement(contact, 'rolling_friction', value="0.0")
        contact_cfm = et.SubElement(contact, 'contact_cfm', value="0.0")
        contact_erp = et.SubElement(contact, 'rolling_friction', value="1.0")
        #         <contact>
        #   <lateral_friction value="1.0"/>
        #   <rolling_friction value="0.0"/>
        #   <contact_cfm value="0.0"/>
        #   <contact_erp value="1.0"/>
        # </contact>

        # Inertial information
        inertial = et.SubElement(link, 'inertial')
        et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
        mass = np.maximum(piece.mass.reshape(-1)[0], minimum_mass)
        et.SubElement(inertial, 'mass', value='{:.2E}'.format(mass))
        et.SubElement(inertial, 'inertia', ixx=I[0][0], ixy=I[0][1], ixz=I[0][2],
                      iyy=I[1][1], iyz=I[1][2], izz=I[2][2])
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
    et.SubElement(author, 'name').text = 'trimesh {}'.format(trimesh_version)
    et.SubElement(author, 'email').text = 'blank@blank.blank'

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
                continue
            t.set_postfix_str(f"{urdf_root_path}")
            t.update(1)

        # if test_object:
        #     test_urdf(urdf_root_path=urdf_root_path)


if __name__ == '__main__':
    ycb2urdf()
