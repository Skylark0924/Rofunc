import pybullet as p
import pybullet_data
import time
import math

def fd(urdf, joint_torque, export_joint="EE"):
    export_joint_force = ...
    return export_joint_force


if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(1)
    planeId = p.loadURDF("plane.urdf")
    StartPos = [0, 0, 1]
    StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf", StartPos, StartOrientation)


    while True:
        time.sleep(1. / 240.)



    # urdf = ...
    # joint_torque = []
    # force = fd(urdf, joint_torque)
    # print(force)
