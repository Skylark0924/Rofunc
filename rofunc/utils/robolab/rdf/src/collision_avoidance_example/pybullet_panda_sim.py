import time
import numpy as np
from math import pi
import os
import sys

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
pandaNumDofs = 7

# restpose
rp = [0.0, 0.0, -1.57461, -1.60788, -0.785175, 1.54666, -0.882595, 0.02, 0.02]


class PandaSim():
    def __init__(self, bullet_client, base_pos, base_rot):
        self.bullet_client = bullet_client
        self.bullet_client.setAdditionalSearchPath('content/urdfs')
        self.base_pos = np.array(base_pos)
        self.base_rot = np.array(base_rot)
        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        table_rot = self.bullet_client.getQuaternionFromEuler([pi / 2, 0, pi])
        self.rp = rp
        self.panda = self.bullet_client.loadURDF(os.path.join(CUR_PATH, "panda_urdf/panda.urdf"), self.base_pos,
                                                 self.base_rot, useFixedBase=True, flags=flags)
        self.reset()
        self.t = 0.
        # self.set_joint_positions(rp)

    def reset(self):
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, self.rp[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, self.rp[index])
                index = index + 1

    def set_joint_positions(self, joint_positions):
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     joint_positions[i], force=240.)
        self.set_finger_positions(0.04)

    def set_finger_positions(self, gripper_opening):
        self.bullet_client.setJointMotorControl2(self.panda, 9, self.bullet_client.POSITION_CONTROL,
                                                 gripper_opening / 2, force=5 * 240.)
        self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL,
                                                 -gripper_opening / 2, force=5 * 240.)

    def get_joint_positions(self):
        joint_state = []
        for i in range(pandaNumDofs):
            joint_state.append(self.bullet_client.getJointState(self.panda, i)[0])
        return joint_state


class SphereManager:
    def __init__(self, pybullet_client):
        self.pb = pybullet_client
        self.spheres = []
        self.color = [.7, .1, .1, 1]
        self.color = [.63, .07, .185, 1]
        # self.color = [0.8500, 0.3250, 0.0980, 1]

    def create_sphere(self, position, radius, color):
        sphere = self.pb.createVisualShape(self.pb.GEOM_SPHERE,
                                           radius=radius,
                                           rgbaColor=color, specularColor=[0, 0, 0, 1])
        sphere = self.pb.createCollisionShape(self.pb.GEOM_SPHERE,
                                              radius=radius)

        sphere = self.pb.createMultiBody(baseVisualShapeIndex=sphere,
                                         basePosition=position)
        self.spheres.append(sphere)

    def initialize_spheres(self, obstacle_array):
        for obstacle in obstacle_array:
            self.create_sphere(obstacle[0:3], obstacle[3], self.color)

    def delete_spheres(self):
        for sphere in self.spheres:
            self.pb.removeBody(sphere)
        self.spheres = []

    def update_spheres(self, obstacle_array):
        if (obstacle_array is not None) and (len(self.spheres) == len(obstacle_array)):
            for i, sphere in enumerate(self.spheres):
                self.pb.resetBasePositionAndOrientation(sphere,
                                                        obstacle_array[i, 0:3],
                                                        [1, 0, 0, 0])
        else:
            print("Number of spheres and obstacles do not match")
            self.delete_spheres()
            self.initialize_spheres(obstacle_array)
