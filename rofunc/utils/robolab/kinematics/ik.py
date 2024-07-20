import torch

from rofunc.utils.robolab.coord import convert_ori_format, convert_quat_order
from rofunc.utils.robolab.kinematics.pytorch_kinematics_utils import build_chain_from_model


def get_ik_from_chain(chain, pos, rot, device):
    """
    Get the inverse kinematics from a serial chain

    :param chain: only the serial chain is supported
    :param pos: the position of the export_link
    :param rot: the rotation of the export_link
    :param device: the device to run the computation
    :return:
    """
    import pytorch_kinematics as pk

    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # get robot joint limits
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    cur_q = torch.rand(7, device=device) * (lim[1] - lim[0]) + lim[0]

    goal_q = cur_q.unsqueeze(0).repeat(1, 1)

    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # create the IK object
    # see the constructor for more options and their explanations, such as convergence tolerances
    # ik = PseudoInverseIK(chain, max_iterations=30, num_retries=10,
    #                      joint_limits=lim.T,
    #                      early_stopping_any_converged=True,
    #                      early_stopping_no_improvement="all",
    #                      retry_configs=cur_q.reshape(1, -1),
    #                      # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
    #                      debug=False,
    #                      lr=0.2)
    ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=10,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                            debug=False,
                            lr=0.2)

    # solve IK
    sol = ik.solve(goal_in_rob_frame_tf)
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
    model_path = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf"

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
