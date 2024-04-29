"""
This code is from https://github.com/harvard-microrobotics/object2urdf
"""

from scipy.spatial.transform import Rotation
import numpy as np
import os
import copy
import trimesh
import xml.etree.ElementTree as ET
from tqdm import tqdm
import rofunc as rf


class ObjectUrdfBuilder:
    def __init__(self, object_folder="", log_file="vhacd_log.txt", urdf_prototype='_prototype.urdf'):
        self.object_folder = os.path.abspath(object_folder)
        self.log_file = os.path.abspath(log_file)
        self.suffix = "vhacd"

        self.urdf_base = self._read_xml(os.path.join(object_folder, urdf_prototype))

    # Recursively get all files with a specific extension, excluding a certain suffix
    def _get_files_recursively(self, start_directory, filter_extension=None, exclude_suffix=None):
        for root, dirs, files in os.walk(start_directory):
            for file in files:
                if filter_extension is None or file.lower().endswith(filter_extension):
                    if isinstance(exclude_suffix, str):
                        if not file.lower().endswith(exclude_suffix + filter_extension):
                            yield (root, file, os.path.abspath(os.path.join(root, file)))

    # Read and parse a URDF from a file
    def _read_xml(self, filename):
        root = ET.parse(filename).getroot()
        return root

    # Convert a list to a space-separated string
    def _list2str(self, in_list):
        out = ""
        for el in in_list:
            out += str(el) + " "
        return out[:-1]

    # Convert a space-separated string to a list
    def _str2list(self, in_str):
        out = in_str.split(' ')
        out = [float(el) for el in out]
        return out

    # Find the center of mass of the object
    def get_center_of_mass(self, filename):
        mesh = trimesh.load(filename)
        # print(mesh)
        if isinstance(mesh, trimesh.Scene):
            # print("Imported combined mesh: using centroid rather than center of mass")
            return mesh.centroid
        else:
            return mesh.center_mass

    # Find the geometric center of the object
    def get_geometric_center(self, filename):
        mesh = trimesh.load(filename)
        return copy.deepcopy(mesh.centroid)

    # Get the middle of a face of the bounding box
    def get_face(self, filename, edge):
        mesh = trimesh.load(filename)
        bounds = mesh.bounds
        face = copy.deepcopy(mesh.centroid)
        if edge in ['top', 'xy_pos']:
            face[2] = bounds[1][2]
        elif edge in ['bottom', 'xy_neg']:
            face[2] = bounds[0][2]
        elif edge in ['xz_pos']:
            face[1] = bounds[1][1]
        elif edge in ['xz_neg']:
            face[1] = bounds[0][1]
        elif edge in ['yz_pos']:
            face[0] = bounds[1][0]
        elif edge in ['yz_neg']:
            face[0] = bounds[0][0]

        return face

    # Do a convex decomposition
    def do_vhacd(self, filename, outfile, debug=False, **kwargs):
        try:
            mesh = trimesh.load(filename)
            # convex_list = trimesh.decomposition.convex_decomposition(mesh)
            meshes = mesh.convex_decomposition(**kwargs)

            convex = trimesh.util.concatenate(meshes)
            convex.export(outfile)
        except ValueError:
            print("No direct VHACD backend available, trying pybullet")
            pass

        try:
            import pybullet as p
            p.vhacd(filename, outfile, self.log_file, **kwargs)
        except ModuleNotFoundError:
            print(
                '\n' + "ERROR - pybullet module not found: If you want to do convex decomposisiton, make sure you install pybullet (https://pypi.org/project/pybullet) or install VHACD directly (https://github.com/mikedh/trimesh/issues/404)" + '\n')
            raise

    # Find the center of mass of the object
    def save_to_obj(self, filename):
        name, ext = os.path.splitext(filename)
        obj_filename = name + '.obj'
        mesh = trimesh.load(filename)
        mesh.export(obj_filename)
        return obj_filename

    # Replace an attribute in a feild of a URDF
    def replace_urdf_attribute(self, urdf, feild, attribute, value):
        urdf = self.replace_urdf_attributes(urdf, feild, {attribute: value})
        return urdf

    # Replace several attributes in a feild of a URDF
    def replace_urdf_attributes(self, urdf, feild, attribute_dict, sub_feild=None):

        if sub_feild is None:
            sub_feild = []

        field_obj = urdf.find(feild)

        if field_obj is not None:
            if len(sub_feild) > 0:
                for child in reversed(sub_feild):
                    field_obj = ET.SubElement(field_obj, child)
            field_obj.attrib.update(attribute_dict)
            # field_obj.attrib = attribute_dict
        else:
            feilds = feild.split("/")
            new_feild = "/".join(feilds[0:-1])
            sub_feild.append(feilds[-1])
            self.replace_urdf_attributes(urdf, new_feild, attribute_dict, sub_feild)

    # Make an updated copy of the URDF for the current object
    def update_urdf(self, object_file, object_name, collision_file=None, override=None, mass_center=None):
        # If no separate collision geometry is provided, use the object file
        if collision_file is None:
            collision_file = object_file

        # Update the filenames and object name
        new_urdf = copy.deepcopy(self.urdf_base)
        self.replace_urdf_attribute(new_urdf, './/visual/geometry/mesh', 'filename', object_file)
        self.replace_urdf_attribute(new_urdf, './/collision/geometry/mesh', 'filename', collision_file)
        new_urdf.attrib['name'] = object_name

        # Update the overrides
        if override is not None:
            for orverride_el in override:
                # Update attributes
                out_el_all = new_urdf.findall('.//' + orverride_el.tag)

                for out_el in out_el_all:

                    for key in orverride_el.attrib:
                        out_el.set(key, orverride_el.attrib[key])

                    # Remove fields that will be overwritten
                    for child in orverride_el:
                        el = out_el.find(child.tag)
                        if el is not None:
                            out_el.remove(el)
                    # Add updated feilds
                    out_el.extend(orverride_el)

        # Output the center of mass if provided
        if mass_center is not None:
            # Check if there's a geometry offset
            offset_ob = new_urdf.find('.//collision/origin')
            if offset_ob is not None:
                offset_str = offset_ob.attrib.get('xyz', '0 0 0')
                rot_str = offset_ob.attrib.get('rpy', '0 0 0')
                offset = self._str2list(offset_str)
                rpy = self._str2list(rot_str)
            else:
                offset = [0, 0, 0]
                rpy = [0, 0, 0]

            # Check if there's a scale factor and apply it
            scale_ob = new_urdf.find('.//collision/geometry/mesh')
            if scale_ob is not None:
                scale_str = scale_ob.attrib.get('scale', '1 1 1')
                scale = self._str2list(scale_str)
            else:
                scale = [1, 1, 1]

            for idx, axis in enumerate(mass_center):
                mass_center[idx] = -mass_center[idx] * scale[idx] + offset[idx]

            rot = Rotation.from_euler('xyz', rpy)
            rot_matrix = rot.as_matrix()
            mass_center = np.matmul(rot_matrix, np.vstack(np.asarray(mass_center))).squeeze()

            self.replace_urdf_attributes(new_urdf,
                                         './/visual/origin',
                                         {'xyz': self._list2str(mass_center), 'rpy': self._list2str(rpy)})
            self.replace_urdf_attributes(new_urdf,
                                         './/collision/origin',
                                         {'xyz': self._list2str(mass_center), 'rpy': self._list2str(rpy)})

        return new_urdf

    # Save a URDF to a file
    def save_urdf(self, new_urdf, filename, overwrite=False):
        out_file = os.path.join(self.object_folder, filename)

        # Do not overwrite the file unless the option is True
        if os.path.exists(out_file) and not overwrite:
            return

            # Save the file
        mydata = ET.tostring(new_urdf)
        with open(out_file, "wb") as f:
            f.write(mydata)

    # Build a URDF from an object file
    def build_urdf(self, filename, output_folder=None,
                   force_overwrite=False, decompose_concave=False, force_decompose=False,
                   center='mass', **kwargs):

        # If no output folder is specified, use the base object folder
        if output_folder is None:
            output_folder = self.object_folder

        # Generate a relative path from the output folder to the geometry files
        filename = os.path.abspath(filename)
        common = os.path.commonprefix([output_folder, filename])
        rel = os.path.join(filename.replace(common, ''))
        if rel[0] == os.path.sep:
            rel = rel[1:]
        name = rel.split(os.path.sep)[0]
        rel = rel.replace(os.path.sep, '/')

        file_name_raw, file_extension = os.path.splitext(filename)

        # If an override file exists, include its data in the URDF
        override_file = filename.replace(file_extension, '.ovr')
        if os.path.exists(override_file):
            overrides = self._read_xml(override_file)
        else:
            overrides = None

        # Calculate the center of mass
        if center == 'mass':
            mass_center = self.get_center_of_mass(filename)

        elif center == 'geometric':
            mass_center = self.get_geometric_center(filename)

        elif center in ['top', 'bottom', 'xy_pos', 'xy_neg', 'xz_pos', 'xz_neg', 'yz_pos', 'yz_neg']:
            mass_center = self.get_face(filename, center)

        else:
            mass_center = None

        # If the user wants to run convex decomposition on concave objects, do it.
        if decompose_concave:
            if file_extension == '.stl':
                obj_filename = self.save_to_obj(filename)
                visual_file = rel.replace(file_extension, '.obj')
            elif file_extension == '.obj':
                obj_filename = filename
                visual_file = rel
            else:
                raise ValueError("Your filetype needs to be an STL or OBJ to perform concave decomposition")

            outfile = obj_filename.replace('.obj', '_' + self.suffix + '.obj')
            collision_file = visual_file.replace('.obj', '_' + self.suffix + '.obj')

            # Only run a decomposition if one does not exist, or if the user forces an overwrite
            if not os.path.exists(outfile) or force_decompose:
                self.do_vhacd(obj_filename, outfile, **kwargs)

            urdf_out = self.update_urdf(visual_file, name, collision_file=collision_file, override=overrides,
                                        mass_center=mass_center)
        else:
            urdf_out = self.update_urdf(rel, name, override=overrides, mass_center=mass_center)

        self.save_urdf(urdf_out, name + '.urdf', force_overwrite)

    # Build the URDFs for all objects in your library.
    def build_library(self, **kwargs):
        rf.logger.beauty_print("\nFOLDER: %s" % self.object_folder)

        # Get all OBJ files
        obj_files = self._get_files_recursively(self.object_folder, filter_extension='.obj', exclude_suffix=self.suffix)
        stl_files = self._get_files_recursively(self.object_folder, filter_extension='.stl', exclude_suffix=self.suffix)

        # Check if the urdf is already built
        existing_urdfs = rf.oslab.list_absl_path(self.object_folder, suffix='.urdf')

        obj_folders = []
        for root, _, full_file in tqdm(obj_files, total=101):
            object_name = root.split("/")[-2]
            assert object_name[0] == '0'
            if object_name + '.urdf' in existing_urdfs:
                continue
            obj_folders.append(root)
            self.build_urdf(full_file, **kwargs)

            # common = os.path.commonprefix([self.object_folder, full_file])
            # rel = os.path.join(full_file.replace(common, ''))
            # print('\tBuilding: %s' % (rel))

        for root, _, full_file in tqdm(stl_files):
            object_name = root.split("/")[-2]
            assert object_name[0] == '0'
            if object_name + '.urdf' in existing_urdfs:
                continue
            if root not in obj_folders:
                self.build_urdf(full_file, **kwargs)

                # common = os.path.commonprefix([self.object_folder, full_file])
                # rel = os.path.join(full_file.replace(common, ''))
                # print('Building: %s' % (rel))


def ycb2urdf(force_overwrite=True, decompose_concave=True, force_decompose=False, center='top'):
    # Build entire libraries of URDFs
    rofunc_path = rf.oslab.get_rofunc_path()
    object_folder = os.path.join(rofunc_path, "simulator/assets/urdf/ycb")

    builder = ObjectUrdfBuilder(object_folder)
    builder.build_library(force_overwrite=force_overwrite,
                          decompose_concave=decompose_concave,
                          force_decompose=force_decompose,
                          center=center)


if __name__ == '__main__':
    ycb2urdf()
