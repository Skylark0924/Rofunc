/*
	  iLQR applied on a bicopter problem

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Julius Jankowski <julius.jankowski@idiap.ch>,
    Jérémy Maceiras <jeremy.maceiras@idiap.ch>,Sylvain Calinon <https://calinon.ch>

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

#include <iostream>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <GL/glut.h>
#include <math.h>

using namespace Eigen;

// Parameters
// ===============================
struct Param {
  double dt, T, l, m, g;
  VectorXd X_d;
  MatrixXd Q, R;
  unsigned int nbIter, nbDoF;
  Param() {
    nbDoF = 6;
    dt = 1e-2;
    T = 1.0;
    l = 0.2;
    m = 1.0;
    g = 9.81;
    VectorXd x_d = VectorXd::Zero(nbDoF);
    x_d.head(3) << -2., 1., 0.*M_PI/4.;
    double Q_t = 0.0;
    double Q_T = 1e3;
    double R_t = 1e-3;
    unsigned int nbSteps = (unsigned int)(T / dt);
    X_d = VectorXd::Zero(nbSteps*nbDoF);
    X_d.tail(nbDoF) = x_d;
    Q = Q_t*MatrixXd::Identity(nbSteps*nbDoF, nbSteps*nbDoF);
    Q.bottomRightCorner(nbDoF,nbDoF) = Q_T*MatrixXd::Identity(nbDoF, nbDoF);
    R = R_t*MatrixXd::Identity((nbSteps-1)*2, (nbSteps-1)*2);
    nbIter = 20;
  }
};

// Helper function
// ===============================
VectorXd flatten(const MatrixXd& M)
{
  MatrixXd M_T = M.transpose();
  return Map<VectorXd>(M_T.data(), M_T.size());
}
MatrixXd reshape(VectorXd& v, unsigned int cols, unsigned int rows)
{
  return Map<MatrixXd>(v.data(), rows, cols).transpose();
}

// Dynamics functions
// x = [x_g, y_g, theta, dx_g, dy_g, dtheta]
// u = [f_l, f_r]
// ===============================
VectorXd step(const VectorXd& x, const VectorXd& u)
{
  Param param;
  VectorXd x_(param.nbDoF);
  x_(3) = x(3) - param.dt * sin(x(2))*(u(0)+u(1)) / param.m;
  x_(4) = x(4) + param.dt * (cos(x(2)) * (u(0)+u(1)) / param.m - param.g);
  x_(5) = x(5) + param.dt * 6. * (u(1)-u(0)) / (param.m*param.l); // I = m*l^2/12
  x_.head(3) = x.head(3) + param.dt * 0.5 * (x.tail(3) + x_.tail(3));
  return x_;
}

MatrixXd rollout(const VectorXd& x_init, const MatrixXd& U)
{
  Param param;
  unsigned int nbSteps = U.rows()+1;
  MatrixXd X = MatrixXd::Zero(nbSteps, param.nbDoF);
  X.row(0) = x_init;
  for(unsigned int i = 0; i < nbSteps-1; i++) {
    X.row(i+1) = step(X.row(i), U.row(i));
  }
  return X;
}

MatrixXd get_A(const VectorXd& x, const VectorXd& u)
{
  Param param;
  MatrixXd A = MatrixXd::Identity(param.nbDoF, param.nbDoF);
  A.topRightCorner(3, 3) = param.dt * MatrixXd::Identity(3, 3);
  A(0,2) = -0.5 * param.dt * param.dt * (u(0)+u(1)) * cos(x(2)) / param.m;
  A(1,2) = -0.5 * param.dt * param.dt * (u(0)+u(1)) * sin(x(2)) / param.m;
  A(3,2) = -param.dt * (u(0)+u(1)) * cos(x(2)) / param.m;
  A(4,2) = -param.dt * (u(0)+u(1)) * sin(x(2)) / param.m;
  return A;
}

MatrixXd get_B(const VectorXd& x)
{
  Param param;
  MatrixXd B = MatrixXd::Zero(param.nbDoF, 2);
  MatrixXd M_inv = 1./param.m * MatrixXd::Identity(3, 3);
  M_inv(2,2) = 12. / (param.m * param.l * param.l);
  MatrixXd G(3, 2);
  G << -sin(x(2)), -sin(x(2)), cos(x(2)), cos(x(2)), -0.5*param.l, 0.5*param.l;
  B.topRows(3) = 0.5 * param.dt * param.dt * M_inv * G;
  B.bottomRows(3) = param.dt * M_inv * G;
  return B;
}

MatrixXd get_Su(const MatrixXd& X, const MatrixXd& U)
{
  Param param;
  unsigned int nbSteps = X.rows();
  MatrixXd Su = MatrixXd::Zero(param.nbDoF*nbSteps, 2*(nbSteps-1));
  for(unsigned int j = 0; j < nbSteps-1; j++) {
    Su.block((j+1)*param.nbDoF, j*2, param.nbDoF, 2) = get_B(X.row(j));
    for(unsigned int i = 0; i < nbSteps-2-j; i++) {
      Su.block((j+2+i)*param.nbDoF, j*2, param.nbDoF, 2) = get_A(X.row(i+j+1), U.row(i+j+1)) * Su.block((j+1+i)*param.nbDoF, j*2, param.nbDoF, 2);
    }
  }
  return Su;
}

// Cost function
// ===============================
double cost(const MatrixXd& X, const MatrixXd& U)
{
  Param param;
  return (flatten(X) - param.X_d).dot(param.Q * (flatten(X) - param.X_d)) + flatten(U).dot(param.R * flatten(U));
}

// Optimal Control
// ===============================
std::vector<VectorXd> iLQR(const VectorXd& x_init)
{
  Param param;
  unsigned int nbSteps = (unsigned int)(param.T / param.dt);
  MatrixXd U = 0.5 * param.m * param.g * MatrixXd::Ones(nbSteps-1, 2);
  for(unsigned int k = 0; k < param.nbIter; k++) {
    MatrixXd X = rollout(x_init, U);
    double current_cost = cost(X, U);
    std::cout << "Iteration: " << k+1 << " Cost: " << current_cost << std::endl;
    MatrixXd Su = get_Su(X, U);
    VectorXd delta_u = (Su.transpose() * param.Q * Su + param.R).llt().solve(Su.transpose() * param.Q * (param.X_d - flatten(X)) - param.R * flatten(U));
    // Line search
    double alpha = 1.0;
    double best_cost = current_cost;
    MatrixXd U_best = U;
    for(unsigned int i = 0; i < 20; i++) {
      VectorXd u_tmp = flatten(U) + alpha * delta_u;
      MatrixXd U_tmp = reshape(u_tmp, nbSteps-1, 2);
      X = rollout(x_init, U_tmp);
      double cost_tmp = cost(X, U_tmp);
      if(cost_tmp < best_cost) {
        best_cost = cost_tmp;
        U_best = U_tmp;
      }
      alpha = alpha / 2.;
    }
    U = U_best;
  }
  MatrixXd X = rollout(x_init, U);
  std::vector<VectorXd> X_vec(nbSteps);
  for(unsigned int i = 0; i < nbSteps; i++) {
    X_vec[i] = X.row(i);
  }
  return X_vec;
}

// Plotting
// ===============================
void plot_robot(const VectorXd& x, const double c=0.0)
{
  Param param;
  glColor3d(c, c, c);
  glLineWidth(5.0);
	glBegin(GL_LINE_STRIP);
	glVertex2d(x(0) - 0.5 * param.l * cos(x(2)), x(1) - 0.5 * param.l * sin(x(2)));
	glVertex2d(x(0) + 0.5 * param.l * cos(x(2)), x(1) + 0.5 * param.l * sin(x(2)));
	glEnd();
}

void render(){
  Param param;
  VectorXd x_init = VectorXd::Zero(param.nbDoF);
  
  auto X = iLQR(x_init);

	glLoadIdentity();
	double windowWidth = 5.0;
	double windowHeight = 2.5;
	glOrtho(-windowWidth/2., windowWidth/2., -windowHeight/2., windowHeight/2., 0.0f, 1.0f);
	
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	
	unsigned int num_plots = 12;
	for(unsigned int i = 0; i < X.size(); i+=(unsigned int)(X.size()/(double)num_plots)) {
	  plot_robot(X[i]);
	}
	plot_robot(X.back());
	
	// Plot target
	VectorXd x_d = param.X_d.tail(param.nbDoF);
	glColor3d(1.0, 0, 0);
	glPointSize(12.0);
	glBegin(GL_POINTS);
	glVertex2d(x_d(0), x_d(1));
	glEnd();

	//Render
	glutSwapBuffers();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(20,20);
	glutInitWindowSize(1200,600);
	glutCreateWindow("iLQR_bicopter");
	glutDisplayFunc(render);
	glutMainLoop();
}
