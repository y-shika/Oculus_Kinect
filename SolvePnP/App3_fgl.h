#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>

#include <string>
#include <gl\freeglut.h>

#include <iostream>

class Draw {
public:
	static void Axes(float len) {
		float color[4];
		glGetFloatv(GL_COLOR_ARRAY, color);

		// x axis
		glColor3d(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		glVertex3d(0.0, 0.0, 0.0);
		glVertex3d(len, 0.0, 0.0);
		glEnd();
		// y axis
		glColor3d(0.0, 1.0, 0.0);
		glBegin(GL_LINES);
		glVertex3d(0.0, 0.0, 0.0);
		glVertex3d(0.0, len, 0.0);
		glEnd();
		// z axis
		glColor3d(0.0, 0.0, 1.0);
		glBegin(GL_LINES);
		glVertex3d(0.0, 0.0, 0.0);
		glVertex3d(0.0, 0.0, len);
		glEnd();

		glColor4fv(color);

	}
	// planeType: 0 = XZ, 1 = XY, 2 = YZ 
	static void GroundAxes(double gridLen, int halfGridNum, int planeType = 0) {
		double start = -halfGridNum * gridLen;
		double end = halfGridNum * gridLen;

		float color[4];
		glGetFloatv(GL_COLOR_ARRAY, color);

		glColor3d(0.5, 0.5, 0.5);
		glBegin(GL_LINES);
		if (planeType == 0) {	// XZ
			for (int i = 0; i <= halfGridNum * 2; i++) {
				// x
				glVertex3d(start + i * gridLen, 0.0, start);
				if (i == halfGridNum)
					glVertex3d(start + i * gridLen, 0.0, 0.0);
				else
					glVertex3d(start + i * gridLen, 0.0, end);

				// z
				glVertex3d(start, 0.0, start + i * gridLen);
				if (i == halfGridNum)
					glVertex3d(0.0, 0.0, start + i * gridLen);
				else
					glVertex3d(end, 0.0, start + i * gridLen);
			}
		}
		else if (planeType == 1) {	// XY
			for (int i = 0; i <= halfGridNum * 2; i++) {
				// x
				glVertex3d(start + i * gridLen, start, 0.0);
				if (i == halfGridNum)
					glVertex3d(start + i * gridLen, 0.0, 0.0);
				else
					glVertex3d(start + i * gridLen, end, 0.0);

				// y
				glVertex3d(start, start + i * gridLen, 0.0);
				if (i == halfGridNum)
					glVertex3d(0.0, start + i * gridLen, 0.0);
				else
					glVertex3d(end, start + i * gridLen, 0.0);
			}
		}
		else if (planeType == 2) {	// YZ
			for (int i = 0; i <= halfGridNum * 2; i++) {
				// y
				glVertex3d(0.0, start + i * gridLen, start);
				if (i == halfGridNum)
					glVertex3d(0.0, start + i * gridLen, 0.0);
				else
					glVertex3d(0.0, start + i * gridLen, end);

				// z
				glVertex3d(0.0, start, start + i * gridLen);
				if (i == halfGridNum)
					glVertex3d(0.0, 0.0, start + i * gridLen);
				else
					glVertex3d(0.0, end, start + i * gridLen);
			}
		}


		// x axis
		glColor3d(1.0, 0.0, 0.0);
		glVertex3d(0.0, 0.0, 0.0);
		glVertex3d(end, 0.0, 0.0);
		// y axis
		glColor3d(0.0, 1.0, 0.0);
		glVertex3d(0.0, 0.0, 0.0);
		glVertex3d(0.0, end, 0.0);
		// z axis
		glColor3d(0.0, 0.0, 1.0);
		glVertex3d(0.0, 0.0, 0.0);
		glVertex3d(0.0, 0.0, end);

		glEnd();

		glColor4fv(color);

	}
};


inline void normalize(double &x, double &y, double &z) {
	double _a = 1.0 / sqrt(x*x + y*y + z*z);
	x *= _a;
	y *= _a;
	z *= _a;
}

class WinBase {
protected:
	static WinBase* winPtr(WinBase* p = nullptr) {
		static WinBase* ptr = nullptr;
		if (p != nullptr)
			ptr = p;
		return ptr;
	}

	int winW, winH, winID;
	double eyeR, eyeTh, eyePh, objX, objY, objZ;
	int upY, oldX, oldY, oldUpY;

	bool dragFlags[3];
	bool spKeyFlags[256];
public:
	WinBase(int winW, int winH, std::string title)
		: winW(winW), winH(winH),
		eyeR(5.0f), eyeTh(0.0f), eyePh(0.0f), objX(0.0f), objY(0.0f), objZ(0.0f), upY(1)
	{
		dragFlags[GLUT_LEFT_BUTTON] = dragFlags[GLUT_MIDDLE_BUTTON] = dragFlags[GLUT_RIGHT_BUTTON] = false;
		memset(spKeyFlags, 0, sizeof(spKeyFlags));

		winPtr(this);

		glutInitWindowSize(winW, winH);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
		winID = glutCreateWindow(title.c_str());

		glutDisplayFunc(displayCB);
		glutReshapeFunc(reshapeCB);
		glutIdleFunc(idleCB);
		glutKeyboardFunc(keyboardCB);
		glutSpecialFunc(specialKeyCB);
		glutSpecialUpFunc(specialKeyUpCB);

		glutMouseFunc(mouseCB);
		glutMotionFunc(motionCB);
		glutMouseWheelFunc(mouseWheelCB);
	}

	void draw() {
		glutSetWindow(winID);
		display();
	}
	void lookAt() {
		double th = eyeTh, ph = eyePh, uy = 1.0;
		if (ph > M_PI_2 || ph < -M_PI_2) {
			ph = M_PI - ph;
			th += M_PI;
			uy = -1.0;
		}

		double x = eyeR * cos(ph) * sin(th);
		double y = eyeR * sin(ph);
		double z = eyeR * cos(ph) * cos(th);
		gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, uy, 0.0);
	}
	void loadObjMat() {
		glTranslated(-objX, -objY, -objZ);
	}

	void run() {
		glutMainLoop();
	}

	//----------------------------------------
	// callback function
	virtual void display() {
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60.0, (double)winW / winH, 0.1, 100.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		lookAt();

		Draw::GroundAxes(2.0f, 5);

		static int deg = 0;
		glRotated(deg, 0, 1, 0);
		if (++deg >= 360)
			deg = 0;

		glColor3d(0, 0, 0);
		glutWireTeapot(1.0);

		glutSwapBuffers();
	}
	virtual void reshape(int w, int h) {
		winW = w;
		winH = h;
	}
	virtual void myKeyboard(unsigned char key, int x, int y) {}

	virtual void keyboard(unsigned char key, int x, int y) {
		switch (key) {
		case '\x1b':
			exit(0);
			break;
		case '0':
			eyeTh = eyePh = 0.0f;
			objX = objY = objZ = 0.0f;
			break;
		}
		myKeyboard(key, x, y);
	}
	virtual void specialKey(int key, int x, int y) {
		spKeyFlags[key] = true;
	}
	virtual void specialKeyUp(int key, int x, int y) {
		spKeyFlags[key] = false;
	}

	virtual void mouse(int button, int state, int x, int y) {
		if (state == GLUT_DOWN) {
			oldX = x; oldY = y;
			dragFlags[button] = true;
			if (eyePh > M_PI_2 || eyePh < -M_PI_2)
				oldUpY = -1;
			else
				oldUpY = 1;
		}
		else if (state == GLUT_UP) {
			dragFlags[button] = false;
		}
	}
	virtual void motion(int x, int y) {
		if (dragFlags[GLUT_LEFT_BUTTON]) {
			double s = 0.5;

			if (spKeyFlags[GLUT_KEY_SHIFT_L] || spKeyFlags[GLUT_KEY_SHIFT_R])
				s = 0.1;

			double ph = eyePh + (y - oldY) * s * M_PI / 180.0;
			if (oldUpY > 0)
				s *= -1;
			double th = eyeTh + (x - oldX) * s * M_PI / 180.0;
			if (th > M_PI)			th -= 2 * M_PI;
			else if (th <= -M_PI)	th += 2 * M_PI;
			if (ph > M_PI)			ph -= 2 * M_PI;
			else if (ph <= -M_PI)	ph += 2 * M_PI;

			eyeTh = th;
			eyePh = ph;

			oldX = x;
			oldY = y;
		}
		if (dragFlags[GLUT_RIGHT_BUTTON]) {
			double th = eyeTh, ph = eyePh, uy = 1.0;
			if (ph > M_PI_2 || ph < -M_PI_2) {
				ph = M_PI - ph;
				th += M_PI;
				uy = -1.0;
			}
			double sTh = sin(th), cTh = cos(th), sPh = sin(ph), cPh = cos(ph);
			double ix, iy, iz, jx, jy, jz, kx, ky, kz;

			kx = cPh * sTh;
			ky = sPh;
			kz = cPh * cTh;
			if (ph == M_PI_2 || ph == -M_PI_2) {
				ix = cTh;
				iy = 0.0;
				iz = sTh;
				jx = sTh;
				jy = 0.0;
				jz = -cTh;
			}
			else {
				ix = uy * kz;
				iy = 0.0;
				iz = -uy * kx;
				normalize(ix, iy, iz);

				jx = iz * ky;
				jy = -iz * kx + ix * kz;
				jz = -ix * ky;
				normalize(jx, jy, jz);
			}

			double s = 0.01;
			double di = -(x - oldX) * s;
			double dj = (y - oldY) * s;


			objX += di * ix + dj * jx;
			objY += di * iy + dj * jy;
			objZ += di * iz + dj * jz;

			oldX = x;
			oldY = y;
		}
	}
	virtual void mouseWheel(int wheelNo, int dir, int x, int y) {
		static double ks = 0.2;
		if (!(spKeyFlags[GLUT_KEY_SHIFT_L] || spKeyFlags[GLUT_KEY_SHIFT_R] ||
			spKeyFlags[GLUT_KEY_CTRL_L] || spKeyFlags[GLUT_KEY_CTRL_R])) {
			double tr = eyeR - ks * dir;
			if (tr > 0.0)
				eyeR = tr;
		}

	}

	//----------------------------------------
	// callback launcher
	static void displayCB() {
		winPtr()->display();
	}
	static void idleCB() {
		glutPostRedisplay();
	}
	static void reshapeCB(int w, int h) {
		winPtr()->reshape(w, h);
	}
	static void keyboardCB(unsigned char key, int x, int y) {
		winPtr()->keyboard(key, x, y);
	}
	static void specialKeyCB(int key, int x, int y) {
		winPtr()->specialKey(key, x, y);
	}
	static void specialKeyUpCB(int key, int x, int y) {
		winPtr()->specialKeyUp(key, x, y);
	}
	static void mouseCB(int button, int state, int x, int y) {
		winPtr()->mouse(button, state, x, y);
	}
	static void motionCB(int x, int y) {
		winPtr()->motion(x, y);
	}
	static void mouseWheelCB(int wheelNo, int dir, int x, int y) {
		winPtr()->mouseWheel(wheelNo, dir, x, y);
	}
};
