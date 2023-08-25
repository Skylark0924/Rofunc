"""
CURI utils
========================

Link name, joint name and actuated joint name of the CURI robot.
"""
import os

import rofunc as rf
from rofunc.utils.oslab.path import get_rofunc_path

urdf_path = os.path.join(get_rofunc_path(), 'simulator/assets/urdf/curi/urdf/curi_w_softhand.urdf')
link_name, joint_name, actuated_joint_name = rf.robolab.check_urdf(urdf_path)
print(actuated_joint_name)