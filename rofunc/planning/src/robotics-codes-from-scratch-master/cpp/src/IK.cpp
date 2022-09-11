/*
    Inverse kinematics example on 2D manipulator.

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Sylvain Calinon <https://calinon.ch>

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
#include <GL/glut.h>
#include <math.h>

using namespace Eigen;

// Parameters
// ===============================
struct Param {
  unsigned int nbSteps;
  double dt;
  VectorXd linkLengths;
  Vector3d x_d;
  Param() {
    nbSteps = 500;
    dt = 1e-2;
    linkLengths = VectorXd(4);
    linkLengths << 1.0, 1.0, 0.5, 0.5;
    x_d << -2, 1, 4*M_PI/4.;
  }
};

// Kinematics functions
// ===============================
MatrixXd fkin(VectorXd q)
{
  Param param;
  long int D = param.linkLengths.size();
  MatrixXd G = MatrixXd::Ones(D, D).triangularView<UnitLower>();
  MatrixXd x = MatrixXd::Zero(3, D);
  x.row(0) = G * param.linkLengths.asDiagonal() * cos((G * q).array()).matrix();
  x.row(1) = G * param.linkLengths.asDiagonal() * sin((G * q).array()).matrix();
  x.row(2) = G * q;
  
  return x;
}

MatrixXd jacobian(VectorXd q)
{
  Param param;
  long int D = param.linkLengths.size();
  MatrixXd G = MatrixXd::Ones(D, D).triangularView<UnitLower>();
  MatrixXd J = MatrixXd::Zero(3, D);
  J.row(0) = - G.transpose() * param.linkLengths.asDiagonal() * sin((G * q).array()).matrix();
  J.row(1) =   G.transpose() * param.linkLengths.asDiagonal() * cos((G * q).array()).matrix();
  J.row(2) = VectorXd::Ones(D);
  
  return J;
}

// Iterative Inverse Kinematics
// ===============================
std::vector<MatrixXd> IK(VectorXd& q)
{
  Param param;
  std::vector<MatrixXd> x;
  
  for(unsigned int i = 0; i < param.nbSteps; i++) {
    x.push_back(fkin(q));
    auto J = jacobian(q);
    auto dx_d = - (x.back().col(param.linkLengths.size()-1) - param.x_d);
  	VectorXd dq_d = J.completeOrthogonalDecomposition().solve(dx_d);
  	q = q + param.dt * dq_d;
  }
  
  return x;
}

// Plotting
// ===============================

void plot_robot(const MatrixXd& x, const double c=0.0)
{
  Param param;
  glColor3d(c, c, c);
  glLineWidth(8.0);
	glBegin(GL_LINE_STRIP);
	glVertex2d(0, 0);
	for(unsigned int j = 0; j < param.linkLengths.size(); j++){
		glVertex2d(x(0,j), x(1,j));
	}
	glEnd();
}

void render(){
  Param param;
  VectorXd q = VectorXd::Ones(param.linkLengths.size()) * M_PI / (double)param.linkLengths.size();
  auto x = IK(q);

	glLoadIdentity();
	double d = (double)param.linkLengths.size() * .7;
	glOrtho(-d, d, -.1, d-.1, -1.0, 1.0);
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

  // Plot start and end configuration
  plot_robot(x.front(), 0.8);
  plot_robot(x.back(), 0.4);
	
	// Plot end-effector trajectory
	double c = 0.0;
  glColor3d(c, c, c);
  glLineWidth(4.0);
	glBegin(GL_LINE_STRIP);
	for(auto& x_ : x) {
	  glVertex2d(x_(0,param.linkLengths.size()-1), x_(1,param.linkLengths.size()-1));
	}
	glEnd();
	
	// Plot target
	glColor3d(1.0, 0, 0);
	glPointSize(12.0);
	glBegin(GL_POINTS);
	glVertex2d(param.x_d(0), param.x_d(1));
	glEnd();

	//Render
	glutSwapBuffers();
}

int main(int argc, char** argv){
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(20,20);
	glutInitWindowSize(1200,600);
	glutCreateWindow("IK");
	glutDisplayFunc(render);
	glutMainLoop();
}
