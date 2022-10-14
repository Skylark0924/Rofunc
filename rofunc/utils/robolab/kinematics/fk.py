def fk(urdf, joint_state, export_joint="EE"):
    export_joint_pose = ...
    return export_joint_pose


if __name__ == '__main__':
    urdf = ...
    joint_state = []
    pose = fk(urdf, joint_state)
    print(pose)
