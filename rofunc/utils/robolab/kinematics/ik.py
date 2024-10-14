from typing import Union, List, Tuple
import torch

from rofunc.utils.robolab.coord import convert_ori_format, convert_quat_order
from rofunc.utils.robolab.kinematics.pytorch_kinematics_utils import build_chain_from_model


def get_ik_from_chain(chain, goal_pose: Union[torch.Tensor, None, List, Tuple], device, goal_in_rob_tf: bool = True,
                      robot_pose: Union[torch.Tensor, None, List, Tuple] = None, cur_configs=None,
                      num_retries: int = 10):
    """
    Get the inverse kinematics from a serial chain
    :param chain: only the serial chain is supported
    :param goal_pose: the pose of the export ee link
    :param device: the device to run the computation
    :param goal_in_rob_tf: whether the goal pose is in the robot base frame
    :param robot_pose: the pose of the robot base frame
    :param cur_configs: let the ik solver retry from these configurations
    :param num_retries: the number of retries
    :return:
    """
    import pytorch_kinematics as pk

    goal_pos = goal_pose[:3]
    goal_rot = goal_pose[3:]
    goal_tf = pk.Transform3d(pos=goal_pos, rot=goal_rot, device=device)
    if not goal_in_rob_tf:
        assert robot_pose is not None, "The robot pose must be provided if the goal pose is not in the robot base frame"
        robot_pos = robot_pose[:3]
        robot_rot = robot_pose[3:]
        rob_tf = pk.Transform3d(pos=robot_pos, rot=robot_rot, device=device)
        goal_tf = rob_tf.inverse().compose(goal_tf)

    # get robot joint limits
    lim = torch.tensor(chain.get_joint_limits(), device=device)

    if cur_configs is not None:
        cur_configs = torch.tensor(cur_configs, device=device)

    # create the IK object
    # see the constructor for more options and their explanations, such as convergence tolerances
    ik = pk.PseudoInverseIK(chain, max_iterations=30, retry_configs=cur_configs, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False,
                            lr=0.2)

    # solve IK
    sol = ik.solve(goal_tf)
    return sol


def get_ik_from_model(model_path: str, pose: torch.Tensor, device, export_link, verbose=False):
    """
    Get the inverse kinematics from a URDF or MuJoCo XML file
    :param model_path: the path of the URDF or MuJoCo XML file
    :param pose: the pose of the end effector, 7D vector with the first 3 elements as position and the last 4 elements as rotation
    :param device: the device to run the computation
    :param export_link: the name of the end effector link
    :param verbose: whether to print the chain
    :return: the position, rotation of the end effector, and the transformation matrices of all links
    """
    import pytorch_kinematics as pk

    chain = build_chain_from_model(model_path, verbose)
    chain = pk.SerialChain(chain, export_link)

    pos = pose[:3]
    rot = convert_ori_format(pose[3:], "quat", "euler")
    sol = get_ik_from_chain(chain, pos, rot, device)
    return sol


if __name__ == '__main__':
    model_path = "./simulator/assets/urdf/franka_description/robots/franka_panda.urdf"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device)

    sol = get_ik_from_model(model_path, pose, device, export_link="panda_hand")
    print(sol.solutions)
    print("Done")

    import pybullet as p
    import pybullet_data
    import pytorch_kinematics as pk

    search_path = pybullet_data.getDataPath()

    # visualize everything
    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(search_path)

    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    m = rob_tf.get_matrix()
    pos = m[0, :3, 3]
    rot = m[0, :3, :3]
    quat = pk.matrix_to_quaternion(rot)
    pos = pos.cpu().numpy()
    rot = convert_quat_order(quat, "wxyz", "xyzw").cpu().numpy()
    armId = p.loadURDF(model_path, basePosition=pos, baseOrientation=rot, useFixedBase=True)

    visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
                                rgbaColor=[0., 1., 0., 0.5])
    # r = goal_rot[goal_num]
    # xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(r, "XYZ")))
    # goalId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId,
    #                            basePosition=goal_pos[goal_num].cpu().numpy(),
    #                            baseOrientation=xyzw.cpu().numpy())
    show_max_num_retries_per_goal = 10
    for i, q in enumerate(sol.solutions[0]):
        if i > show_max_num_retries_per_goal:
            break
        for dof in range(q.shape[0]):
            p.resetJointState(armId, dof, q[dof])
        input("Press enter to continue")

    # p.removeBody(goalId)

    while True:
        p.stepSimulation()