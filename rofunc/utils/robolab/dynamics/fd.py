def fd(urdf, joint_torque, export_joint="EE"):
    export_joint_force = ...
    return export_joint_force


if __name__ == '__main__':
    urdf = ...
    joint_torque = []
    force = fd(urdf, joint_torque)
    print(force)
