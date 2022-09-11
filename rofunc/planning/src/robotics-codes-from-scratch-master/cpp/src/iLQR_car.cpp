/*
	  iLQR applied on a car parking problem

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Julius Jankowski <julius.jankowski@idiap.ch>,
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

#include <iostream>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <GL/glut.h>
#include <math.h>

using namespace Eigen;

#define DoF 5 // set 4 for velocity control, 5 for acceleration control

// Parameters
// ===============================
struct Param {
  double dt, T, l, m, g;
  VectorXd X_d;
  MatrixXd Q, R;
  unsigned int nbIter, nbDoF;
  Param() {
    nbDoF = DoF;
    dt = 1e-2;
    T = 1.0;
    l = 0.2; // car length
    VectorXd x_d = VectorXd::Zero(nbDoF);
    x_d.head(3) << 2., 1., 4.*M_PI/4.;
    double Q_t = 0.0;
    double Q_T = 1e3;
    double R_t = 1e-3;
    unsigned int nbSteps = (unsigned int)(T / dt);
    X_d = VectorXd::Zero(nbSteps*nbDoF);
    X_d.tail(nbDoF) = x_d;
    Q = Q_t*MatrixXd::Identity(nbSteps*nbDoF, nbSteps*nbDoF);
    Q.bottomRightCorner(nbDoF,nbDoF) = Q_T*MatrixXd::Identity(nbDoF, nbDoF);
    R = R_t*MatrixXd::Identity((nbSteps-1)*2, (nbSteps-1)*2);
    nbIter = 30;
  }
};

Param param;

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

#if DoF == 4

// Dynamics functions (velocity control)
// x = [x_g, y_g, theta, steering_angle]
// u = [v, dsteering_angle]
// ===============================
VectorXd step(const VectorXd& x, const VectorXd& u)
{
  VectorXd x_(param.nbDoF);
  x_(0) = x(0) + param.dt * u(0) * cos(x(2));
  x_(1) = x(1) + param.dt * u(0) * sin(x(2));
  x_(2) = x(2) + param.dt * u(0) * tan(x(3)) / param.l;
  x_(3) = x(3) + param.dt * u(1);
  return x_;
}

MatrixXd get_A(const VectorXd& x, const VectorXd& u)
{
  MatrixXd A = MatrixXd::Identity(param.nbDoF, param.nbDoF);
  A(0,2) = - param.dt * u(0) * sin(x(2));
  A(1,2) =   param.dt * u(0) * cos(x(2));
  A(2,3) = param.dt * u(0) * (1+tan(x(3))*tan(x(3))) / param.l;
  
  return A;
}

MatrixXd get_B(const VectorXd& x, const VectorXd& u)
{
  MatrixXd B = MatrixXd::Zero(param.nbDoF, 2);
  B(0,0) = param.dt * cos(x(2));
  B(1,0) = param.dt * sin(x(2));
  B(2,0) = param.dt * tan(x(3)) / param.l;
  B(3,1) = param.dt;
  return B;
}

#elif DoF == 5

// Dynamics functions (acceleration control)
// x = [x_g, y_g, theta, v, steering_angle]
// u = [dv, dsteering_angle]
// ===============================
VectorXd step(const VectorXd& x, const VectorXd& u)
{
  VectorXd x_(param.nbDoF);
  x_(0) = x(0) + param.dt * x(3) * cos(x(2));
  x_(1) = x(1) + param.dt * x(3) * sin(x(2));
  x_(2) = x(2) + param.dt * x(3) * tan(x(4)) / param.l;
  x_.tail(2) = x.tail(2) + param.dt * u;
  return x_;
}

MatrixXd get_A(const VectorXd& x, const VectorXd& u)
{
  MatrixXd A = MatrixXd::Identity(param.nbDoF, param.nbDoF);
  A(0,2) = - param.dt * x(3) * sin(x(2));
  A(1,2) =   param.dt * x(3) * cos(x(2));
  A(0,3) = param.dt * cos(x(2));
  A(1,3) = param.dt * sin(x(2));
  A(2,3) = param.dt * tan(x(4)) / param.l;
  A(2,4) = param.dt * x(3) * (1+tan(x(4))*tan(x(4))) / param.l;
  return A;
}

MatrixXd get_B(const VectorXd& x, const VectorXd& u)
{
  MatrixXd B = MatrixXd::Zero(param.nbDoF, 2);
  B.bottomRows(2) = param.dt * MatrixXd::Identity(2,2);
  return B;
}

#endif

MatrixXd rollout(const VectorXd& x_init, const MatrixXd& U)
{
  unsigned int nbSteps = U.rows()+1;
  MatrixXd X = MatrixXd::Zero(nbSteps, param.nbDoF);
  X.row(0) = x_init;
  for(unsigned int i = 0; i < nbSteps-1; i++) {
    X.row(i+1) = step(X.row(i), U.row(i));
  }
  return X;
}

MatrixXd get_Su(const MatrixXd& X, const MatrixXd& U)
{
  unsigned int nbSteps = X.rows();
  MatrixXd Su = MatrixXd::Zero(param.nbDoF*nbSteps, 2*(nbSteps-1));
  for(unsigned int j = 0; j < nbSteps-1; j++) {
    Su.block((j+1)*param.nbDoF, j*2, param.nbDoF, 2) = get_B(X.row(j), U.row(j));
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
  return (flatten(X) - param.X_d).dot(param.Q * (flatten(X) - param.X_d)) + flatten(U).dot(param.R * flatten(U));
}

// Optimal Control
// ===============================
std::vector<VectorXd> iLQR(const VectorXd& x_init)
{
  unsigned int nbSteps = (unsigned int)(param.T / param.dt);
  MatrixXd U = MatrixXd::Zero(nbSteps-1, 2);
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
    for(unsigned int i = 0; i < 10; i++) {
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
    if((flatten(U) - flatten(U_best)).squaredNorm() < 1e-2) {
      U = U_best;
      break;
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
  glColor3d(c, c, c);
  glLineWidth(4.0);
	glBegin(GL_LINE_STRIP);
	double w = 0.5 * param.l;
	glVertex2d(x(0) - 0.5 * w * sin(x(2)), x(1) + 0.5 * w * cos(x(2)));
	glVertex2d(x(0) + 0.5 * w * sin(x(2)), x(1) - 0.5 * w * cos(x(2)));
	glVertex2d(x(0) + 0.5 * w * sin(x(2)) + param.l * cos(x(2)), x(1) - 0.5 * w * cos(x(2)) + param.l * sin(x(2)));
	glVertex2d(x(0) - 0.5 * w * sin(x(2)) + param.l * cos(x(2)), x(1) + 0.5 * w * cos(x(2)) + param.l * sin(x(2)));
	glVertex2d(x(0) - 0.5 * w * sin(x(2)), x(1) + 0.5 * w * cos(x(2)));
	glEnd();
}

void render(){
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
	glutCreateWindow("iLQR_car");
	glutDisplayFunc(render);
	glutMainLoop();
}
