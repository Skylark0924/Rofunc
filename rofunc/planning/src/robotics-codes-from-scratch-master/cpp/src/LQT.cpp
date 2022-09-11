// LQT.cpp
// Sylvain Calinon, 2021

#include <iostream>
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
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
  double Q_t, Q_T, R_t;
  double T;
  Param() {
    nbSteps = 500;
    dt = 1e-2;
    linkLengths = VectorXd(4);
    linkLengths << 1.0, 1.0, 0.5, 0.5;
    x_d << -2, 1, 4*M_PI/4.;
    Q_t = 0.0;
    Q_T = 100.0;
    R_t = 1e-2;
    T = 1.0;
  }
};

// Kinematics functions
// ===============================
MatrixXd fkin(VectorXd q)
{
  Param param;
  long int nbDoF = param.linkLengths.size();
  MatrixXd G = MatrixXd::Ones(nbDoF, nbDoF).triangularView<UnitLower>();
  MatrixXd x = MatrixXd::Zero(3, nbDoF);
  x.row(0) = G * param.linkLengths.asDiagonal() * cos((G * q).array()).matrix();
  x.row(1) = G * param.linkLengths.asDiagonal() * sin((G * q).array()).matrix();
  x.row(2) = G * q;
  
  return x;
}

MatrixXd jacobian(VectorXd q)
{
  Param param;
  long int nbDoF = param.linkLengths.size();
  MatrixXd G = MatrixXd::Ones(nbDoF, nbDoF).triangularView<UnitLower>();
  MatrixXd J = MatrixXd::Zero(3, nbDoF);
  J.row(0) = - G.transpose() * param.linkLengths.asDiagonal() * sin((G * q).array()).matrix();
  J.row(1) =   G.transpose() * param.linkLengths.asDiagonal() * cos((G * q).array()).matrix();
  J.row(2) = VectorXd::Ones(nbDoF);
  
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

// Trajectory Optimization
// ===============================
std::vector<VectorXd> LQT(const VectorXd& q_init, const VectorXd& q_goal, const double T=1.0)
{
  Param param;
  long int nbDoF = param.linkLengths.size();
  unsigned int nbSteps = (unsigned int)(T / param.dt);
  MatrixXd Sq = kroneckerProduct(MatrixXd::Ones(nbSteps, 1), MatrixXd::Identity(nbDoF, nbDoF));
  MatrixXd Su = MatrixXd::Zero(nbDoF*nbSteps, (nbSteps-1)*nbDoF);
  Su.bottomRows((nbSteps-1)*nbDoF) = 
    kroneckerProduct(MatrixXd::Ones(nbSteps-1, nbSteps-1), param.dt * MatrixXd::Identity(nbDoF, nbDoF)).eval().triangularView<Eigen::Lower>();
  VectorXd q_d = VectorXd::Zero(nbDoF*nbSteps);
  q_d.tail(nbDoF) = q_goal;
  MatrixXd Q = param.Q_t * MatrixXd::Identity(nbDoF*nbSteps, nbDoF*nbSteps);
  Q.bottomRightCorner(nbDoF, nbDoF) = param.Q_T * MatrixXd::Identity(nbDoF, nbDoF);
  MatrixXd R = param.R_t * MatrixXd::Identity(nbDoF*(nbSteps-1), nbDoF*(nbSteps-1));
  VectorXd q_opt = Sq * q_init + Su * (Su.transpose() * Q * Su + R).ldlt().solve(Su.transpose() * Q * (q_d - Sq * q_init));
  std::vector<VectorXd> q(nbSteps);
  for(unsigned int i = 0; i < nbSteps; i++) {
    q[i] = q_opt.segment(i*nbDoF, nbDoF);
  }
  return q;
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
  VectorXd q_init = VectorXd::Ones(param.linkLengths.size()) * M_PI / (double)param.linkLengths.size();
  VectorXd q_goal = q_init;
  auto x_IK = IK(q_goal);
  
  auto q = LQT(q_init, q_goal, param.T);
  
  std::vector<MatrixXd> x;
  for(auto& q_ : q) {
    x.push_back(fkin(q_));
  }

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
