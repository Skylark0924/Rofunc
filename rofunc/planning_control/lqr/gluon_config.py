import numpy as np
from rofunc.utils.robolab.kinematics import get_fk_from_model, get_jacobian_from_model
from rofunc.utils.oslab.path import get_rofunc_path

class robot_config:

    def __int__(self):
        pass

    def fkine(self, x):
        """
        Forward kinematics for end-effector (in robot coordinate system)
        """
        robot, f, cfg = get_fk_from_model(get_rofunc_path() + "/simulator/assets/urdf/gluon/gluon.urdf", ["axis_joint_1", "axis_joint_2", "axis_joint_3", "axis_joint_4", "axis_joint_5", "axis_joint_6"], x, export_link='6_Link')
        return f

    def jacob(self, x):
        """
                Jacobian with analytical computation (for single time step)
                $J(x_t)= \dfrac{\partial{f(x_t)}}{\partial{x_t}}$
                """

        J = get_jacobian_from_model(get_rofunc_path() + "/simulator/assets/urdf/gluon/gluon.urdf", end_link_name='6_Link', joint_values=x)

        return J

    def fkinall(self, x):
        link_names = ['1_Link', '2_Link', '3_Link', '4_Link', '5_Link', '6_Link']

        f_all = []
        for i in range(len(link_names)):
            robot, f, cfg = get_fk_from_model(get_rofunc_path() + "/simulator/assets/urdf/gluon/gluon.urdf", ["axis_joint_1", "axis_joint_2", "axis_joint_3", "axis_joint_4", "axis_joint_5", "axis_joint_6"], x, link_names[i])

            f_all.append(f)

        f_all = np.array(f_all)

        return f_all


    def error(self, f, f0):

        """
        Input Parameters:
        f = end_effector transformation matrix
        f0 = desired end_effector pose

        Output Parameters:
        error = position and orientation error
        """

        from scipy.spatial.transform import Rotation as R

        position_error = f[:3, 3] - f0[:3]

        quat = R.from_quat(f0[3:])
        rot = R.as_matrix(quat)

        Transform = f[:3, :3]

        # f_rot = R.from_matrix(f[:3, :3])
        # f_quat = f_rot.as_quat()

        # n_e = f_quat[-1]
        # e_e = f_quat[:3]

        # Calculate the skew-symmetric of e_d
        # skew_ed = np.array([[0, -f0[5], f0[4]],
        #                     [f0[5], 0, -f0[3]],
        #                     [-f0[4], f0[3], 0]])
        #
        # n_d = f0[-1]
        # e_d = f0[3:6]

        # orientation_error = [0, 0, 0, 0]
        # Calculate the orientation error
        # orientation_error = n_e * e_d - n_d * e_e - skew_ed @ e_e

        # orientation_error[0] = (n_d*n_e) + np.dot(e_d, e_e)
        # orientation_error[1] = n_e*e_d[0] - n_d*e_e[0] + e_e[1]*e_d[2] - e_e[2]*e_d[1]
        # orientation_error[2] = n_e*e_d[1] - n_d*e_e[1] - e_e[0]*e_d[2] + e_e[2]*e_d[0]
        # orientation_error[3] = n_e*e_d[2] - n_d*e_e[2] + e_e[0]*e_d[1] - e_e[1]*e_d[0]

        orientation_error = 1 / 2 * (np.cross(rot[0:3, 0], Transform[0:3, 0]) + np.cross(rot[0:3, 1], Transform[0:3, 1]) + np.cross(rot[0:3, 2], Transform[0:3, 2]))


        error = np.hstack([position_error, orientation_error])

        return error

if __name__ == '__main__':
    import numpy as np
    kin = robot_config()

    joint_value = np.array([-3.97674561, 50.89657676, 120.01220703, 87.05993652, -7.04406738, 86.52160645])
    F = kin.fkinall(joint_value)
    F = np.asarray(F)
    print(F)

