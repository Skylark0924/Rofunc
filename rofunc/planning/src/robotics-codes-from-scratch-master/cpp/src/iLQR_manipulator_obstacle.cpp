/*
    iLQR with obstacle avoidance applied on a 2D Manipulator

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Tobias Löw <tobias.low@idiap.ch>
    Sylvain Calinon <https://calinon.ch>

    This file is part of RCFS.

    RCFS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    RCFS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with RCFS. If not, see <http://www.gnu.org/licenses/>.
*/

#include <GL/glut.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <iostream>
#include <vector>

namespace
{
    // define the parameters that influence the behaviour of the algorithm
    struct Parameters
    {
        int num_iterations = 50;  // maximum umber of iterations for iLQR

        double dt = 1e-2;        // time step size
        int num_timesteps = 50;  // number of datapoints

        // definition of the viapoints, size <= num_timesteps
        std::vector<Eigen::Vector3d> viapoints = {
            { 3.0, -1.0, 0.0 }  //
        };

        // robot link lengths
        std::vector<double> links = { 2.0,  //
                                      2.0,  //
                                      1.0 };

        // initial configuration of the robot
        // !! must be the same length as links, otherwise it will result in a segmentation fault
        std::vector<double> initial_joint_angles = { 3.0 * M_PI / 4.0,  //
                                                     -M_PI / 2.0,       //
                                                     -M_PI / 4.0 };

        // definition of the obstacles
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector2d>> obstacles = {
            { { 2.8, 2.0, M_PI / 4.0 }, { 0.5, 0.8 } },  //
            { { 3.5, 0.5, -M_PI / 6.0 }, { 0.5, 0.8 } }  //
        };

        double tracking_weight = 1e2;  // tracking weight term
        double obstacle_weight = 1e0;  // obstacle weight term
        double control_weight = 1e-3;  // control weight term
    };

    struct Model
    {
      public:
        /**
         * initialize the model with the given parameter
         * this calculates the matrices Su and Sx
         */
        Model(const Parameters &parameters);

        /**
         * implementation of the iLQR algorithm
         * this function calls the other functions as needed
         */
        Eigen::MatrixXd ilqr() const;

        /**
         * perform a trajectory rollout for the given initial joint angles and control commands
         * return a joint angle trajectory
         */
        Eigen::MatrixXd rollout(const Eigen::VectorXd &initial_joint_angles, const Eigen::VectorXd &control_commands) const;

        /**
         * reaching function, called at each iLQR iteration
         * calculates the error to each viapoint as well as the jacobians
         */
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> reach(const Eigen::MatrixXd &joint_angles) const;

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>> avoid(const Eigen::MatrixXd &joint_angles) const;

        /**
         * convenience function to extract the viapoints from given joint angle trajectory
         * also reshapes the viapoints to the expected matrix dimensions
         */
        Eigen::MatrixXd viapoints(const Eigen::MatrixXd &joint_angle_trajectory) const;

        /**
         * calculates the end effector position for each state in the given joint angle trajectory
         */
        Eigen::MatrixXd forwardKinematics(const Eigen::MatrixXd &joint_angle_trajectory) const;

        /**
         * does essentially the same as forwardKinematics, but also calculates the position of joint for each state in the given joint angle trajectory
         * this function is only used for plotting here
         */
        Eigen::MatrixXd jointPositions(const Eigen::MatrixXd &joint_angles) const;

        Eigen::MatrixXd jointPose(const Eigen::MatrixXd &joint_angles) const;

        /**
         * compute the Jacobian matrix for the given joint configuration
         */
        Eigen::MatrixXd computeJacobian(const Eigen::VectorXd &joint_angles) const;

        /**
         * this function computes the error between the given joint configuration and the reference
         * notice the name logmap is shorthand for 'logarithmic map', referring to the special treatment of the orientation
         */
        Eigen::MatrixXd logmap(const Eigen::MatrixXd &joint_angles, const Eigen::MatrixXd &ref_joint_angles) const;

        /**
         * return the viapoint_timesteps_
         * only used for plotting
         */
        const Eigen::VectorXi &viatimes() const;

        const std::vector<std::pair<Eigen::Vector2d, Eigen::Matrix2d>> &getObstacles() const;

      private:
        /// parameters defined at runtime
        Eigen::MatrixXd viapoints_;                     // internal viapoint representation
        Eigen::VectorXd links_;                         // robot link lengths
        Eigen::VectorXi viapoint_timesteps_;            // discrete timesteps of the viapoints, uniformly spread
        std::vector<Eigen::Matrix2d> rotation_matrix_;  // used for transformation into the viapoint reference frames

        double tracking_weight_;
        double obstacle_weight_;
        double r_;  // control weight parameter

        int num_links_;       // State space dimension (q1,q2,q3)
        int num_timesteps_;   // copy the number of timesteps for internal use from the Parameters
        int num_iterations_;  // maximum number of iterations for the optimization

        Eigen::VectorXd initial_joint_angles_;  // internal representation of the initial joint angles
        Eigen::MatrixXd control_weight_;        // R matrix, penalizes the control commands
        Eigen::MatrixXd precision_matrix_;      // Q matrix, penalizes the state error
        Eigen::MatrixXd mat_Su0_;               // matrix for propagating the control commands for a rollout
        Eigen::MatrixXd mat_Sx0_;               // matrix for propagating the initial joint angles for a rollout
        Eigen::MatrixXd mat_Su_;                // matrix for propagating the control commands for a rollout at the viapoints
        Eigen::MatrixXd mat_T_;                 // lower triangular matrix filled with ones for forming the translation according to the link lengths

        std::vector<std::pair<Eigen::Vector2d, Eigen::Matrix2d>> obstacles_;  // position and precision
    };

    ////////////////////////////////////////////////////////////////////////////////

    // forward declaration of the function that renders the result
    void render(const Eigen::MatrixXd &joint_angle_trajectory, const Model &model);

    // actual main function, used as a callback for the glut (drawing) cycle
    void callback()
    {
        Parameters parameters;
        Model model(parameters);  // initialize the model with the parameters that were defined at the top

        Eigen::MatrixXd joint_angle_trajectory = model.ilqr();  // calculate the iLQR solution

        render(joint_angle_trajectory, model);  // display the iLQR solution
    }

    ////////////////////////////////////////////////////////////////////////////////
    // implementation of the iLQR algorithm

    Eigen::MatrixXd Model::ilqr() const
    {
        // initial commands, currently all zero
        // can be modified if a better guess is available
        Eigen::VectorXd control_commands = Eigen::VectorXd::Zero(num_links_ * (num_timesteps_ - 1), 1);

        int iter = 1;
        for (; iter <= num_iterations_; ++iter)  // run the optimization for the maximum number of iterations
        {
            std::cout << "iteration " << iter << ":" << std::endl;

            // trajectory rollout, i.e. compute the joint angle trajectory for the given control commands starting from the initial joint angles
            // amounts to Sx * x0 + Su * u
            Eigen::MatrixXd joint_angle_trajectory = rollout(initial_joint_angles_, control_commands);

            // try reaching the viapoints with the current joint angle trajectory and calcualte the error and Jacobian matrix
            auto [state_error, jacobian_matrix] = reach(viapoints(joint_angle_trajectory));

            auto [obstacle_error, obstacle_jacobian, idx] = avoid(joint_angle_trajectory);

            Eigen::MatrixXd mat_Su2 = Eigen::MatrixXd::Zero(static_cast<int>(idx.size()), mat_Su0_.cols());
            for (unsigned i = 0; i < idx.size(); ++i)
            {
                mat_Su2.row(i) = mat_Su0_.row(idx[i]);
            }

            // find the control gradient with the least squares solution
            // (Su^T * J^T * Q * J * Su + R)^-1 * (-Su^T * J^T * Q * e - r * u)
            Eigen::MatrixXd control_gradient = (mat_Su_.transpose() * jacobian_matrix.transpose() * jacobian_matrix * mat_Su_ * tracking_weight_        //
                                                + mat_Su2.transpose() * obstacle_jacobian.transpose() * obstacle_jacobian * mat_Su2 * obstacle_weight_  //
                                                + control_weight_)                                                                                      //
                                                 .inverse()                                                                                             //
                                               * (-mat_Su_.transpose() * jacobian_matrix.transpose() * state_error * tracking_weight_                   //
                                                  - mat_Su2.transpose() * obstacle_jacobian.transpose() * obstacle_error * obstacle_weight_             //
                                                  - r_ * control_commands);

            // calculate the cost of the current joint angle trajectory
            double current_cost = state_error.squaredNorm() * tracking_weight_ + obstacle_error.squaredNorm() * obstacle_weight_ + r_ * control_commands.squaredNorm();

            std::cout << "\t cost = " << current_cost << std::endl;

            // initial step size for the line search
            double step_size = 1.0;
            // line search, i.e. find the best step size for updating the control commands with the gradient
            while (true)
            {
                //  calculate the new control commands for the current step size
                Eigen::MatrixXd tmp_control_commands = control_commands + control_gradient * step_size;

                // calculate a trajectory rollout for the current step size
                Eigen::MatrixXd tmp_joint_angle_trajectory = rollout(initial_joint_angles_, tmp_control_commands);

                // try reaching the viapoints with the current joint angle trajectory
                // we only need the state error here and can disregard the Jacobian, because we are only interested in the cost of the trajectory
                Eigen::MatrixXd tmp_state_error = reach(viapoints(tmp_joint_angle_trajectory)).first;

                Eigen::MatrixXd tmp_obstacle_error = std::get<0>(avoid(tmp_joint_angle_trajectory));

                // resulting cost when updating the control commands with the current step size
                double cost =
                  tmp_state_error.squaredNorm() * tracking_weight_ + tmp_obstacle_error.squaredNorm() * obstacle_weight_ + r_ * tmp_control_commands.squaredNorm();

                // end the line search if the current steps size reduces the cost or becomes too small
                if (cost < current_cost || step_size < 1e-3)
                {
                    control_commands = tmp_control_commands;

                    break;
                }

                // reduce the step size for the next iteration
                step_size *= 0.5;
            }

            std::cout << "\t step_size = " << step_size << std::endl;

            // stop optimizing if the gradient update becomes too small
            if ((control_gradient * step_size).norm() < 1e-2)
            {
                break;
            }
        }

        std::cout << "iLQR converged in " << iter << " iterations" << std::endl;

        return rollout(initial_joint_angles_, control_commands);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // implementation of all functions used in the iLQR algorithm

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Model::reach(const Eigen::MatrixXd &joint_angles) const
    {
        Eigen::MatrixXd state_error = logmap(forwardKinematics(joint_angles), viapoints_);

        Eigen::MatrixXd jacobian_matrix = Eigen::MatrixXd::Zero(3 * joint_angles.cols(), num_links_ * joint_angles.cols());

        for (unsigned t = 0; t < joint_angles.cols(); ++t)
        {
            state_error.col(t).topRows(2) = rotation_matrix_[t].transpose() * state_error.col(t).topRows(2);

            Eigen::MatrixXd j_t = computeJacobian(joint_angles.col(t));
            j_t.topRows(2) = rotation_matrix_[t].transpose() * j_t.topRows(2);

            jacobian_matrix.block(3 * t, t * static_cast<unsigned>(num_links_), 3, num_links_) = j_t;
        }

        state_error = Eigen::Map<Eigen::MatrixXd>(state_error.data(), state_error.size(), 1);

        return std::make_pair(state_error, jacobian_matrix);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>> Model::avoid(const Eigen::MatrixXd &joint_angles) const
    {
        std::vector<double> fs;
        std::vector<Eigen::MatrixXd> js;
        std::vector<int> idx;

        int rows = 0;
        int cols = 0;

        for (const auto &[position, precision] : obstacles_)
        {
            for (unsigned i = 0; i < joint_angles.rows(); ++i)
            {
                for (unsigned t = 0; t < joint_angles.cols(); ++t)
                {
                    Eigen::MatrixXd end_effector = jointPose(joint_angles.col(t).topRows(i + 1)).topRows(2);

                    Eigen::MatrixXd avoidance = precision.transpose() * (end_effector - position);

                    double f = 1.0 - avoidance.squaredNorm();

                    if (f > 0.0)
                    {
                        Eigen::MatrixXd jrob = Eigen::MatrixXd::Zero(joint_angles.rows(), joint_angles.rows());
                        Eigen::MatrixXd jj = computeJacobian(joint_angles.topRows(i + 1).col(t));
                        jrob.block(0, 0, jj.rows(), jj.cols()) = jj;

                        Eigen::MatrixXd j = -avoidance.transpose() * precision.transpose() * jrob.topRows(2);

                        rows += static_cast<int>(j.rows());
                        cols += static_cast<int>(j.cols());

                        js.push_back(j);
                        fs.push_back(f);

                        for (unsigned j = 0; j < joint_angles.rows(); ++j)
                        {
                            idx.push_back(static_cast<int>(t * joint_angles.rows() + j));
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd f = Eigen::Map<Eigen::MatrixXd>(fs.data(), static_cast<int>(fs.size()), 1);
        Eigen::MatrixXd j = Eigen::MatrixXd::Zero(rows, cols);

        int r = 0, c = 0;
        for (const Eigen::MatrixXd &ji : js)
        {
            j.block(r, c, ji.rows(), ji.cols()) = ji;
            r += static_cast<int>(ji.rows());
            c += static_cast<int>(ji.cols());
        }

        return std::make_tuple(f, j, idx);
    }

    Eigen::MatrixXd Model::rollout(const Eigen::VectorXd &initial_joint_angles, const Eigen::VectorXd &control_commands) const
    {
        Eigen::MatrixXd joint_angle_trajectory = mat_Su0_ * control_commands + mat_Sx0_ * initial_joint_angles;
        joint_angle_trajectory = Eigen::MatrixXd(Eigen::Map<Eigen::MatrixXd>(joint_angle_trajectory.data(), num_links_, num_timesteps_));

        return joint_angle_trajectory;
    }

    Eigen::MatrixXd Model::viapoints(const Eigen::MatrixXd &joint_angle_trajectory) const
    {
        Eigen::MatrixXd via_joint_angles = Eigen::MatrixXd::Zero(joint_angle_trajectory.rows(), viapoint_timesteps_.size());

        for (unsigned t = 0; t < viapoint_timesteps_.size(); ++t)
        {
            via_joint_angles.col(t) = joint_angle_trajectory.col(viapoint_timesteps_(t));
        }

        return via_joint_angles;
    }

    Eigen::MatrixXd Model::computeJacobian(const Eigen::VectorXd &joint_angles) const
    {
        Eigen::MatrixXd mat_T = Eigen::MatrixXd::Ones(joint_angles.rows(), joint_angles.rows()).triangularView<Eigen::Lower>();

        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Ones(3, joint_angles.rows());

        jacobian.row(0) = -(mat_T * joint_angles).array().sin().matrix().transpose() * links_.topRows(joint_angles.rows()).asDiagonal() * mat_T;
        jacobian.row(1) = (mat_T * joint_angles).array().cos().matrix().transpose() * links_.topRows(joint_angles.rows()).asDiagonal() * mat_T;

        return jacobian;
    }

    Eigen::MatrixXd Model::logmap(const Eigen::MatrixXd &joint_angles, const Eigen::MatrixXd &ref_joint_angles) const
    {
        Eigen::MatrixXd error = Eigen::MatrixXd::Zero(3, joint_angles.cols());

        for (unsigned t = 0; t < joint_angles.cols(); ++t)
        {
            error.col(t).topRows(2) = joint_angles.col(t).topRows(2) - ref_joint_angles.col(t).topRows(2);
            error.col(t)(2) = std::log(std::conj(std::exp(std::complex<double>(0.0, ref_joint_angles.col(t)(2)))) *  //
                                       std::exp(std::complex<double>(0.0, joint_angles.col(t)(2))))
                                .imag();
        }

        return error;
    }

    Eigen::MatrixXd Model::forwardKinematics(const Eigen::MatrixXd &joint_angle_trajectory) const
    {
        Eigen::MatrixXd states(num_links_, joint_angle_trajectory.cols());

        states.row(0) = links_.transpose() * Eigen::MatrixXd((mat_T_ * joint_angle_trajectory).array().cos());
        states.row(1) = links_.transpose() * Eigen::MatrixXd((mat_T_ * joint_angle_trajectory).array().sin());

        for (unsigned t = 0; t < joint_angle_trajectory.cols(); ++t)
        {
            states(2, t) = std::fmod(joint_angle_trajectory.col(t).sum() + M_PI, 2.0 * M_PI) - M_PI;
        }

        return states;
    }

    Eigen::MatrixXd Model::jointPose(const Eigen::MatrixXd &joint_angles) const
    {
        Eigen::MatrixXd states(joint_angles.rows(), 1);
        Eigen::MatrixXd mat_T = Eigen::MatrixXd::Ones(joint_angles.rows(), joint_angles.rows()).triangularView<Eigen::Lower>();

        Eigen::VectorXd state = Eigen::VectorXd::Zero(3);
        state.row(0) = links_.topRows(joint_angles.rows()).transpose() * Eigen::MatrixXd((mat_T * joint_angles).array().cos());
        state.row(1) = links_.topRows(joint_angles.rows()).transpose() * Eigen::MatrixXd((mat_T * joint_angles).array().sin());
        state(2) = std::fmod(joint_angles.sum() + M_PI, 2.0 * M_PI) - M_PI;

        return state;
    }

    Eigen::MatrixXd Model::jointPositions(const Eigen::MatrixXd &joint_angles) const
    {
        Eigen::MatrixXd mat_T2 = links_.transpose().replicate(num_links_, 1).triangularView<Eigen::Lower>();

        Eigen::MatrixXd joint_positions = Eigen::MatrixXd::Zero(3, num_links_);

        joint_positions.row(0) = (mat_T2 * Eigen::MatrixXd((mat_T_ * joint_angles).array().cos())).transpose();
        joint_positions.row(1) = (mat_T2 * Eigen::MatrixXd((mat_T_ * joint_angles).array().sin())).transpose();

        return joint_positions;
    }

    const Eigen::VectorXi &Model::viatimes() const
    {
        return viapoint_timesteps_;
    }

    const std::vector<std::pair<Eigen::Vector2d, Eigen::Matrix2d>> &Model::getObstacles() const
    {
        return obstacles_;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // precalculate matrices used in the iLQR algorithm

    Model::Model(const Parameters &parameters)
    {
        int num_viapoints = static_cast<int>(parameters.viapoints.size());
        num_links_ = static_cast<int>(parameters.links.size());

        r_ = parameters.control_weight;
        tracking_weight_ = parameters.tracking_weight;
        obstacle_weight_ = parameters.obstacle_weight;

        num_timesteps_ = parameters.num_timesteps;
        num_iterations_ = parameters.num_iterations;

        viapoints_ = Eigen::MatrixXd::Zero(3, num_viapoints);
        for (unsigned t = 0; t < parameters.viapoints.size(); ++t)
        {
            viapoints_.col(t) = parameters.viapoints[t];

            Eigen::Matrix2d a;
            a << std::cos(parameters.viapoints[t](2)), -std::sin(parameters.viapoints[t](2)), std::sin(parameters.viapoints[t](2)),
              std::cos(parameters.viapoints[t](2));

            rotation_matrix_.push_back(a);
        }

        links_ = Eigen::VectorXd::Zero(num_links_);
        initial_joint_angles_ = Eigen::VectorXd::Zero(num_links_);
        for (unsigned i = 0; i < parameters.links.size(); ++i)
        {
            links_(i) = parameters.links[i];
            initial_joint_angles_(i) = parameters.initial_joint_angles[i];
        }

        viapoint_timesteps_ = Eigen::VectorXd::LinSpaced(num_viapoints + 1, 0, num_timesteps_ - 1).bottomRows(num_viapoints).array().round().cast<int>();

        control_weight_ = r_ * Eigen::MatrixXd::Identity((num_timesteps_ - 1) * num_links_, (num_timesteps_ - 1) * num_links_);
        precision_matrix_ = 1e0 * Eigen::MatrixXd::Identity(3 * num_viapoints, 3 * num_viapoints);

        Eigen::MatrixXi idx = Eigen::VectorXi::LinSpaced(num_links_, 0, num_links_ - 1).replicate(1, num_viapoints);

        for (unsigned i = 0; i < idx.rows(); ++i)
        {
            idx.row(i) += Eigen::VectorXi((viapoint_timesteps_.array()) * num_links_).transpose();
        }

        mat_Su0_ = Eigen::MatrixXd::Zero(num_links_ * (num_timesteps_), num_links_ * (num_timesteps_ - 1));
        mat_Su0_.bottomRows(num_links_ * (num_timesteps_ - 1)) = kroneckerProduct(Eigen::MatrixXd::Ones(num_timesteps_ - 1, num_timesteps_ - 1),  //
                                                                                  parameters.dt * Eigen::MatrixXd::Identity(num_links_, num_links_))
                                                                   .eval()
                                                                   .triangularView<Eigen::Lower>();
        mat_Sx0_ = kroneckerProduct(Eigen::MatrixXd::Ones(num_timesteps_, 1),  //
                                    Eigen::MatrixXd::Identity(num_links_, num_links_))
                     .eval();

        mat_Su_ = Eigen::MatrixXd::Zero(idx.size(), num_links_ * (num_timesteps_ - 1));
        for (unsigned i = 0; i < idx.size(); ++i)
        {
            mat_Su_.row(i) = mat_Su0_.row(idx(i));
        }

        mat_T_ = Eigen::MatrixXd::Ones(num_links_, num_links_).triangularView<Eigen::Lower>();

        for (const auto &[pose, size] : parameters.obstacles)
        {
            Eigen::Matrix2d obstacle_precision;
            obstacle_precision << std::cos(pose(2)), -std::sin(pose(2)), std::sin(pose(2)), std::cos(pose(2));

            obstacle_precision = obstacle_precision * size.asDiagonal().inverse();

            obstacles_.push_back(std::make_pair(pose.topRows(2), obstacle_precision));
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // implementation of the rendering function

    void render(const Eigen::MatrixXd &joint_angle_trajectory, const Model &model)
    {
        Eigen::MatrixXd state_trajectory = model.forwardKinematics(joint_angle_trajectory);

        glLoadIdentity();
        glOrtho(-1.5, 3.5, 0.0, 3.0, -1.0, 1.0);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glutReshapeWindow(1200, 600);

        static auto drawManipulator = [&](const int &t) {
            glColor3d(1.0, 0, 0);
            glPointSize(12.0);
            glBegin(GL_POINTS);
            glVertex2d(state_trajectory(0, t), state_trajectory(1, t));
            glEnd();

            Eigen::MatrixXd joint_positions = model.jointPositions(joint_angle_trajectory.col(t));

            glColor3d(0.1, 0.1, 0.1);
            glLineWidth(4.0);
            glBegin(GL_LINE_STRIP);
            glVertex2d(0, 0);
            for (int i = 0; i < joint_positions.cols(); i++)
            {
                glVertex2d(joint_positions(0, i), joint_positions(1, i));
            }
            glEnd();
        };

        glColor3d(0.9, 0.9, 0.9);
        glLineWidth(4.0);
        glBegin(GL_LINE_STRIP);
        for (unsigned t = 0; t < state_trajectory.cols(); ++t)
        {
            glVertex2d(state_trajectory(0, t), state_trajectory(1, t));
        }
        glEnd();

        drawManipulator(0);
        for (unsigned j = 0; j < model.viatimes().rows(); ++j)
        {
            drawManipulator(model.viatimes()(j));
        }

        for (const auto &[position, precision] : model.getObstacles())
        {
            glColor3d(1.0, 0, 0);
            glPointSize(12.0);
            glBegin(GL_POINTS);
            glVertex2d(position(0), position(1));
            glEnd();
        }

        glutSwapBuffers();
    }
}  // namespace

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(20, 20);
    glutInitWindowSize(1200, 600);
    glutCreateWindow("iLQR_manipulator");
    glutDisplayFunc(callback);
    glutMainLoop();

    return 0;
}
