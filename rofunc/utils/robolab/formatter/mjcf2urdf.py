import os

import pybullet_utils.bullet_client as bulllet_client
import pybullet_utils.urdfEditor as urdfEditor
import pybullet_data
import rofunc.utils.oslab as oslab


def convert_mjcf_to_urdf(mjcf_path, output_path):
    """
    Convert MuJoCo mjcf to URDF format and save.

    :param mjcf_path: the path of the input mjcf file.
    :param output_path: the path of the output urdf file.
    :return:
    """
    client = bulllet_client.BulletClient()
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    objs = client.loadMJCF(mjcf_path, flags=client.URDF_USE_IMPLICIT_CYLINDER)

    # create output directory
    oslab.create_dir(output_path)

    for obj in objs:
        print("obj=", obj, client.getBodyInfo(obj), client.getNumJoints(obj))
        humanoid = objs[obj]
        ue = urdfEditor.UrdfEditor()
        ue.initializeFromBulletBody(humanoid, client._client)
        robot_name = str(client.getBodyInfo(obj)[1], 'utf-8')
        part_name = str(client.getBodyInfo(obj)[0], 'utf-8')
        save_visuals = False
        outpath = os.path.join(output_path, "{}_{}.urdf".format(robot_name, part_name))
        ue.saveUrdf(outpath, save_visuals)


if __name__ == '__main__':
    mjcf_path = "/home/skylark/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/simulator/assets/mjcf/amp_humanoid.xml"
    output_path = "/home/skylark/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/simulator/assets/mjcf2urdf"
    convert_mjcf_to_urdf(mjcf_path, output_path)
