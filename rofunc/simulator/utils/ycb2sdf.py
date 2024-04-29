"""
Creates Gazebo compatible SDF files from downloaded YCB data.

This looks through all the YCB objects you have downloaded in a particular
folder, and creates Gazebo compatible SDF files from a set of templates.

If the object has google_16k meshes downloaded, it will use those; else, it
will use the tsdf meshes which are of lower quality.

We recommend ensuring that you've enabled `google_16k` as one of the file
types to download in the `download_ycb_dataset.py` script.

Sebastian Castro 2020-2021
"""

import os
import trimesh
import argparse
import numpy as np

import rofunc as rf

# Define folders
default_ycb_folder = os.path.join("models", "ycb")
default_template_folder = os.path.join("templates", "ycb")

if __name__ == "__main__":

    print("Creating files to use YCB objects in Gazebo...")

    # Parse arguments
    parser = argparse.ArgumentParser(description="YCB Model Importer")
    parser.add_argument("--downsample-ratio", type=float, default=1,
                        help="Mesh vertex downsample ratio (set to 1 to leave meshes as they are)")
    parser.add_argument("--template-folder", type=str, default=default_template_folder,
                        help="Location of YCB models (defaults to ./templates/ycb)")
    parser.add_argument("--ycb-folder", type=str, default=default_ycb_folder,
                        help="Location of YCB models (defaults to ./models/ycb)")

    args = parser.parse_args()

    # Get the template files to copy over
    config_template_file = os.path.join(args.template_folder, "model.config")
    model_template_file = os.path.join(args.template_folder, "template.sdf")
    material_template_file = os.path.join(args.template_folder, "template.material")
    with open(config_template_file, "r") as f:
        config_template_text = f.read()
    with open(model_template_file, "r") as f:
        model_template_text = f.read()
    with open(material_template_file, "r") as f:
        material_template_text = f.read()

    # Now loop through all the folders
    for folder in folder_names:
        if folder != "template":
            try:
                print("Creating Gazebo files for {} ...".format(folder))

                # Extract model name and folder
                model_long = folder
                model_short = folder[4:]
                model_folder = os.path.join(args.ycb_folder, model_long)

                # Check if there are Google meshes; else use the TSDF folder
                if "google_16k" in os.listdir(model_folder):
                    mesh_type = "google_16k"
                else:
                    mesh_type = "tsdf"

                # Extract key data from the mesh
                if mesh_type == "google_16k":
                    mesh_file = os.path.join(model_folder, "google_16k", "textured.obj")
                elif mesh_type == "tsdf":
                    mesh_file = os.path.join(model_folder, "tsdf", "textured.obj")
                mesh = trimesh.load(mesh_file)
                # Mass and moments of inertia
                mass_text = str(mesh.mass)
                tf = mesh.principal_inertia_transform
                inertia = trimesh.inertia.transform_inertia(tf, mesh.moment_inertia)
                # Center of mass
                com_vec = mesh.center_mass.tolist()
                eul = trimesh.transformations.euler_from_matrix(np.linalg.inv(tf), axes="sxyz")
                com_vec.extend(list(eul))
                com_text = str(com_vec)
                com_text = com_text.replace("[", "")
                com_text = com_text.replace("]", "")
                com_text = com_text.replace(",", "")

                # Create a downsampled mesh file with a subset of vertices and faces
                if args.downsample_ratio < 1:
                    mesh_pts = mesh.vertices.shape[0]
                    num_pts = int(mesh_pts * args.downsample_ratio)
                    (_, face_idx) = mesh.sample(num_pts, True)
                    downsampled_mesh = mesh.submesh((face_idx,), append=True)
                    with open(os.path.join(model_folder, "downsampled.obj"), "w") as f:
                        downsampled_mesh.export(f, "obj")
                    collision_mesh_text = model_long + "/downsampled.obj"
                else:
                    collision_mesh_text = model_long + "/" + mesh_type + "/textured.obj"

                # Copy and modify the model configuration file template
                config_text = config_template_text.replace("$MODEL_SHORT", model_short)
                with open(os.path.join(model_folder, "model.config"), "w") as f:
                    f.write(config_text)

                # Copy and modify the model file template
                model_text = model_template_text.replace("$MODEL_SHORT", model_short)
                model_text = model_text.replace("$MODEL_LONG", model_long)
                model_text = model_text.replace("$MESH_TYPE", mesh_type)
                model_text = model_text.replace("$COLLISION_MESH", collision_mesh_text)
                model_text = model_text.replace("$MASS", mass_text)
                model_text = model_text.replace("$COM_POSE", com_text)
                model_text = model_text.replace("$IXX", str(inertia[0][0]))
                model_text = model_text.replace("$IYY", str(inertia[1][1]))
                model_text = model_text.replace("$IZZ", str(inertia[2][2]))
                model_text = model_text.replace("$IXY", str(inertia[0][1]))
                model_text = model_text.replace("$IXZ", str(inertia[0][2]))
                model_text = model_text.replace("$IYZ", str(inertia[1][2]))
                with open(os.path.join(model_folder, model_short + ".sdf"), "w") as f:
                    f.write(model_text)

                # Copy and modify the material file template
                if mesh_type == "google_16k":
                    texture_file = "texture_map.png"
                elif mesh_type == "tsdf":
                    texture_file = "textured.png"
                material_text = material_template_text.replace("$MODEL_SHORT", model_short)
                material_text = material_text.replace("$MODEL_LONG", model_long)
                material_text = material_text.replace("$MESH_TYPE", mesh_type)
                material_text = material_text.replace("$TEXTURE_FILE", texture_file)
                with open(os.path.join(model_folder, model_short + ".material"), "w") as f:
                    f.write(material_text)

            except:
                print("Error processing {}. Textured mesh likely does not exist for this object.".format(folder))

    print("Done.")
