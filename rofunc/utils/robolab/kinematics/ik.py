def ik(urdf, pose, export_joint="EE"):
    export_joint_state = ...
    return export_joint_state


if __name__ == '__main__':
    urdf = ...
    pose = []
    joint_state = ik(urdf, pose)
    print(joint_state)
