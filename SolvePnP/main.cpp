#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen\Eigen>

#include "App3_fgl.h"


Eigen::Matrix3f rotMat_aa(float ang_deg, Eigen::Vector3f axis) {
	float ang_rad = float(ang_deg * M_PI / 180.0);
	float axisNorm = axis.norm();
	if (axisNorm == 0.0f)
		return Eigen::Matrix3f::Identity();

	Eigen::Vector3f n = axis / axisNorm;
	Eigen::Matrix3f Nx;
	Nx << 0.0f, -n(2), n(1),
		n(2), 0.0f, -n(0),
		-n(1), n(0), 0.0f;

	Eigen::Matrix3f R = Eigen::Matrix3f::Identity() + sinf(ang_rad) * Nx + (1.0f - cosf(ang_rad)) * Nx * Nx;
	return R;
}

class Win : public WinBase {
	Eigen::MatrixXf Xw;
	Eigen::MatrixXf Xc0;

	Eigen::MatrixXf Mcw;

	void display() {
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60.0, (double)winW / winH, 0.1, 100.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		lookAt();

		Draw::GroundAxes(2.0f, 5);


		glPointSize(2.0f);
		glBegin(GL_POINTS);
		glColor3d(1, 0, 0);
		for (int i = 0; i < Xw.cols(); i++) {
			glVertex3fv(&Xw(0, i));
		}
		glColor3d(0, 0, 1);
		for (int i = 0; i < Xc0.cols(); i++) {
			glVertex3fv(&Xc0(0, i));
		}
		glEnd();

		glMultMatrixf(Mcw.data());
		Draw::Axes(1.0f);

		glutSwapBuffers();

	}

public:
	Win(int winW, int winH, std::string title) : WinBase(winW, winH, title) {
		int numX = 100;

		Xw = Eigen::MatrixXf::Random(3, numX);

		Eigen::Matrix3f R0 = Eigen::Matrix3f::Identity();
		Eigen::Vector3f t0(-1.0f, 3.0f, 5.0f);

		Mcw = Eigen::MatrixXf::Identity(4, 4);
		Mcw.block(0, 0, 3, 3) = R0;
		Mcw.block(0, 3, 3, 1) = t0;

		Xc0 = (R0 * Xw).colwise() + t0;

		Eigen::Vector3f t = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

		int eNum = Xw.rows() * Xw.cols();

		int iteMax = 10;
		for (int ite = 0; ite < iteMax; ite++) {
			Eigen::VectorXf e = Eigen::VectorXf::Zero(eNum);
			Eigen::MatrixXf J = Eigen::MatrixXf::Zero(eNum, 3);

			for (int i = 0; i < Xw.cols(); i++) {
				Eigen::Vector3f xc = Xw.col(i) + t;
				Eigen::Vector3f ei = xc - Xc0.col(i);
				Eigen::Matrix3f Ji = Eigen::Matrix3f::Identity();

				e.block(3 * i, 0, 3, 1) = ei;
				J.block(3 * i, 0, 3, 3) = Ji;
			}

			float E = 0.5 * e.dot(e);

			std::cout << ite << " : " << E << std::endl;

			Eigen::MatrixXf JTJ = J.transpose() * J;
			Eigen::VectorXf JTe = J.transpose() * e;

			Eigen::FullPivLU<Eigen::MatrixXf> solver(JTJ);
			Eigen::VectorXf dt = solver.solve(JTe);

			t = t - dt;
		}


		//int numX = 100;

		//Xw = Eigen::MatrixXf::Random(3, numX);
		////X.colwise() += Eigen::Vector3f(0.0f, 0.0f, -2.0f);

		////Eigen::Matrix3f R = rotMat_aa(60.0f, Eigen::Vector3f(1.0f, 0.0f, 0.0f));
		////Eigen::Vector3f t = Eigen::Vector3f(0.0f, 0.0f, -2.0f);
		//Eigen::Matrix3f R0 = Eigen::Matrix3f::Identity();
		//Eigen::Vector3f t0 = Eigen::Vector3f::Random();

		//Mcw = Eigen::MatrixXf::Identity(4, 4);
		//Mcw.block(0, 0, 3, 3) = R0;
		//Mcw.block(0, 3, 3, 1) = t0;

		//Xc0 = (R0 * Xw).colwise() + t0;


		//Eigen::Vector3f t = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

		//int eNum = Xw.rows()*Xw.cols();

		//int iteMax = 10;
		//for (int ite = 0; ite < iteMax; ite++) {
		//	Eigen::VectorXf e = Eigen::VectorXf::Zero(eNum);
		//	Eigen::MatrixXf J = Eigen::MatrixXf::Zero(eNum, 3);

		//	for (int i = 0; i < Xw.cols(); i++) {
		//		Eigen::Vector3f xc = Xw.col(i) + t;
		//		Eigen::Vector3f ei = xc - Xc0.col(i);
		//		Eigen::Matrix3f Ji = Eigen::Matrix3f::Identity();

		//		e.block(3 * i, 0, 3, 1) = ei;
		//		J.block(3 * i, 0, 3, 3) = Ji;
		//	}

		//	float E = 0.5 * e.dot(e);

		//	std::cout << ite << " : " << E << std::endl;

		//	Eigen::MatrixXf JTJ = J.transpose() * J;
		//	Eigen::VectorXf JTe = J.transpose() * e;

		//	// A * x = b
		//	Eigen::FullPivLU<Eigen::MatrixXf> solver(JTJ);	// solver(A)
		//	Eigen::VectorXf dt = solver.solve(JTe);			// solver.solve(b), return x

		//	t = t - dt;
		//}


		std::cout << "t0 = " << t0.transpose() << std::endl;
		std::cout << "t  = " << t.transpose() << std::endl;

	}
};


int main(int argc, char *argv[]) {
	glutInit(&argc, argv);

	Win win(640, 480, "solve PnP");

	win.run();

	return 0;
}