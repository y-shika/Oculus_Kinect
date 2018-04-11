/************************************************************************************
Filename    :   main.cpp
Content     :   First-person view test application for Oculus Rift
Created     :   11th May 2015
Authors     :   Tom Heath
Copyright   :   Copyright 2015 Oculus, Inc. All Rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*************************************************************************************/
/// This is an entry-level sample, showing a minimal VR sample, 
/// in a simple environment.  Use WASD keys to move around, and cursor keys.
/// Dismiss the health and safety warning by tapping the headset, 
/// or pressing any key. 
/// It runs with DirectX11.
// Editor : Wizapply : Ovrvision Team

// エラーを避けるためのおまじない(include順序)
#include <Kinect.h>

#include "Win32_GLAppUtil.h"
#include "OVR_CAPI_GL.h"
#include "ComPtr.h"
#include "Kernel/OVR_System.h"

#include <ovrvision_pro.h>

#include <opencv2/opencv.hpp>

// 特徴点マッチング
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include <GL/glut.h>

#include <math.h>
#include <iostream>
#include <vector>

// ReprojectionError
//#include "ceres/ceres.h"
//#include "glog/logging.h"

// Bundle Adjustmentの際に使うデータの取得.
// 改めてデータを取得する場合には"Make Dataset"と検索し, コメントアウトをすべて外す.
// Make Dataset
//#include <fstream>

#define ERROR_CHECK( ret )  \
    if ( (ret) != S_OK ) {    \
        std::stringstream ss;	\
        ss << "failed " #ret " " << std::hex << ret << std::endl;			\
        throw std::runtime_error( ss.str().c_str() );			\
    }

// 四角錘
GLfloat vertex[][3] = {
	{ 0.0f, 0.0f, 0.0f },
	{ 0.2f, 0.0f, 0.0f },
	{ 0.1f, 0.2f, 0.1f },
	{ 0.0f, 0.0f, 0.2f },
	{ 0.2f, 0.0f, 0.2f },
};
int edge[][2] = {
	{ 0, 1 },
	{ 1, 2 },
	{ 2, 0 },
	{ 3, 4 },
	{ 4, 2 },
	{ 2, 3 },
	{ 0, 3 },
	{ 1, 4 }
};

// AR-Ghost
boolean record_mode;
boolean play_mode;

// X-ray Glass
boolean Xray_mode;

// Bundle Adjustment
//boolean BundleAdjustment_mode;
//int bundle_count;

// Make Dataset
//int filewrite_count;

class CameraPlane {
	GLuint CreateShader(GLenum type, const GLchar* src) {
		GLuint shader = glCreateShader(type);

		glShaderSource(shader, 1, &src, NULL);
		glCompileShader(shader);

		GLint r;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &r);
		if (!r)
		{
			GLchar msg[1024];
			glGetShaderInfoLog(shader, sizeof(msg), 0, msg);
			if (msg[0]) {
				OVR_DEBUG_LOG(("Compiling shader failed: %s\n", msg));
			}
			return 0;
		}

		return shader;
	}
	GLuint LinkProgram(GLuint vertexShader, GLuint pixelShader) {
		GLuint program = glCreateProgram();

		glAttachShader(program, vertexShader);
		glAttachShader(program, pixelShader);

		glLinkProgram(program);

		glDetachShader(program, vertexShader);
		glDetachShader(program, pixelShader);

		GLint r;
		glGetProgramiv(program, GL_LINK_STATUS, &r);
		if (!r)
		{
			GLchar msg[1024];
			glGetProgramInfoLog(program, sizeof(msg), 0, msg);
			OVR_DEBUG_LOG(("Linking shaders failed: %s\n", msg));
		}

		return program;
	}

public:
	struct Vertex {
		float x, y, z;
		float u, v;
	};

	GLuint prog;
	GLuint tex;
	GLuint vbo, ibo;

	CameraPlane() : prog(0), tex(0), vbo(0), ibo(0) {}
	~CameraPlane() {
		if (prog) glDeleteProgram(prog);
		if (tex) glDeleteTextures(1, &tex);
		if (vbo) glDeleteBuffers(1, &vbo);
		if (ibo) glDeleteBuffers(1, &ibo);
	}
	void init(int texW, int texH, int channels) {
		static const GLchar* VertexShaderSrc =
			"#version 150\n"
			"in      vec4 Position;\n"
			"in      vec2 TexCoord;\n"
			"out     vec2 oTexCoord;\n"
			"void main()\n"
			"{\n"
			"   gl_Position = Position;\n"
			"   oTexCoord   = TexCoord;\n"
			"}\n";

		static const char* FragmentShaderSrc =
			"#version 150\n"
			"uniform sampler2D Texture0;\n"
			"in      vec2      oTexCoord;\n"
			"out     vec4      FragColor;\n"
			"void main()\n"
			"{\n"
			"   FragColor = pow(texture2D(Texture0, oTexCoord), vec4(2.2));\n"
			"}\n";

		GLuint vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
		GLuint fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);

		prog = LinkProgram(vshader, fshader);

		glDeleteShader(vshader);
		glDeleteShader(fshader);

		glBindFragDataLocation(prog, 0, "FragColor");

		glUniform1i(glGetUniformLocation(prog, "Texture0"), 0);


		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ibo);

	}

	void draw(unsigned char *camImage, int width, int height) {
		float aspect = (float)height / (float)width * 0.82f;

		Vertex V[] = {
			{ -1.0f,  aspect, 1.0f, 0.0f, 0.0f },
			{ -1.0f, -aspect, 1.0f, 0.0f, 1.0f },
			{ 1.0f, -aspect, 1.0f, 1.0f, 1.0f },
			{ 1.0f,  aspect, 1.0f, 1.0f, 0.0f }
		};
		unsigned short F[] = {
			0, 1, 2,
			2, 3, 0
		};

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(F), F, GL_STATIC_DRAW);


		glUseProgram(prog);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, camImage);
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

		GLuint posLoc = glGetAttribLocation(prog, "Position");
		GLuint uvLoc = glGetAttribLocation(prog, "TexCoord");

		glEnableVertexAttribArray(posLoc);
		glEnableVertexAttribArray(uvLoc);

		glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)OVR_OFFSETOF(Vertex, x));
		glVertexAttribPointer(uvLoc, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)OVR_OFFSETOF(Vertex, u));

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);

		glDisableVertexAttribArray(posLoc);
		glDisableVertexAttribArray(uvLoc);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	}
};

class Kinect {
private:
	// Kinect
	IKinectSensor* kinect;
	ICoordinateMapper *coordinateMapper;

	// Color
	IColorFrameReader* colorFrameReader;
	std::vector<BYTE> colorBuffer;
	int colorWidth;
	int colorHeight;
	unsigned int colorBytesPerPixel;

	// Depth
	IDepthFrameReader* depthFrameReader;
	std::vector<UINT16> depthBuffer;
	int depthWidth;
	int depthHeight;

	GLuint texture[1];

public:
	Kinect() :  kinect(nullptr), colorFrameReader(nullptr), depthFrameReader(nullptr) {}
	~Kinect() {}

	void initialize() {
		ERROR_CHECK(::GetDefaultKinectSensor(&kinect));

		ERROR_CHECK(kinect->Open());

		BOOLEAN isOpen = false;
		ERROR_CHECK(kinect->get_IsOpen(&isOpen));
		if (!isOpen) {
			throw std::runtime_error("Kinect Unloaded");
		}

		kinect->get_CoordinateMapper(&coordinateMapper);

		initializeColorFrame();
		initializeDepthFrame();
	}

	void update() {
		updateColorFrame();
		updateDepthFrame();
	}

	cv::Mat getColorImage() {
		cv::Mat colorImage(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0]);
		return colorImage;
	}

	// 三次元点群データが入っているか否かを返す関数(三次元点群をメッシュにする際に使用)
	// 三次元点群がspaeseであることから, 予めこの関数によって描画すべきか判定する.
	std::vector<boolean> isPointData(std::vector<CameraSpacePoint> cameraSpace) {
		std::vector<boolean> isData(cameraSpace.size());

		for (int i = 0; i < cameraSpace.size(); i++) {
			if (cameraSpace[i].X > -100 && cameraSpace[i].X < 100) isData[i] = true;
			else isData[i] = false;
		}
		return isData;
	}

	// 三次元点群の描画
	void draw_kinect()
	{
		// CameraSpacePoint -> 三次元点群
		std::vector<CameraSpacePoint> cameraSpace(depthWidth * depthHeight); 
		coordinateMapper->MapDepthFrameToCameraSpace(depthBuffer.size(), &depthBuffer[0], cameraSpace.size(), &cameraSpace[0]);

		cv::Mat colorImage = getColorImage();

		// 三次元点群のカラー情報にアクセス
		std::vector<ColorSpacePoint> colorPoints(cameraSpace.size());
		coordinateMapper->MapCameraPointsToColorSpace(cameraSpace.size(), &cameraSpace[0], colorPoints.size(), &colorPoints[0]);

		// Make Dataset
		/*if (filewrite_count == 0) {
			cv::imwrite("dataset\\kinect_color.bmp", colorImage);

			std::ofstream pointcloud_coordinate("dataset\\PointCloud_Coordinate.txt");
			pointcloud_coordinate << "(cameraSpace.X, cameraSpace.Y, cameraSpace.Z) = " << std::endl;
			for (int cam_i = 0; cam_i < cameraSpace.size(); cam_i++) {
				pointcloud_coordinate << cameraSpace[cam_i].X << " " << cameraSpace[cam_i].Y << " " << cameraSpace[cam_i].Z << std::endl;
			}
			pointcloud_coordinate.close();

			std::ofstream pointcloud_colorcoordinate("dataset\\PointCloud_ColorCoordinate.txt");
			pointcloud_colorcoordinate << "(colorpoint.X, colorpoint.Y) = " << std::endl;
			for (int color_i = 0; color_i < colorPoints.size(); color_i++) {
				pointcloud_colorcoordinate << colorPoints[color_i].X << " " << colorPoints[color_i].Y << std::endl;
			}
			pointcloud_colorcoordinate.close();

			filewrite_count++; // End FileWrite
		}*/

		double R = 0; double G = 0; double B = 0;

		// 描画実行
		glPointSize(0.000001);
		glBegin(GL_POINTS);
		for (int i = 0; i < colorPoints.size(); i++) {
			if (colorPoints[i].X < 0 || colorPoints[i].Y < 0 || colorPoints[i].X > colorWidth || colorPoints[i].Y > colorHeight) {
				R = 0; G = 0; B = 0;
			}
			else {
				B = colorImage.at<cv::Vec4b>((int)colorPoints[i].Y, (int)colorPoints[i].X)[0] / 255.f;
				G = colorImage.at<cv::Vec4b>((int)colorPoints[i].Y, (int)colorPoints[i].X)[1] / 255.f;
				R = colorImage.at<cv::Vec4b>((int)colorPoints[i].Y, (int)colorPoints[i].X)[2] / 255.f;
			}
			glColor3d(R, G, B);
			glVertex3f(cameraSpace[i].X, cameraSpace[i].Y, cameraSpace[i].Z);
		}
		glEnd();

	}

	// TODO
	// メッシュとテクスチャマッピングは出来てはいるが, ノイズが気になるため改善する.

	// 三次元点群をメッシュにして描画
	void draw_mesh_kinect() {
		std::vector<CameraSpacePoint> cameraSpace(depthWidth * depthHeight);
		coordinateMapper->MapDepthFrameToCameraSpace(depthBuffer.size(), &depthBuffer[0], cameraSpace.size(), &cameraSpace[0]);

		cv::Mat colorImage = getColorImage();

		std::vector<ColorSpacePoint> colorPoints(cameraSpace.size());
		coordinateMapper->MapCameraPointsToColorSpace(cameraSpace.size(), &cameraSpace[0], colorPoints.size(), &colorPoints[0]);

		std::vector<boolean> flagPoints = isPointData(cameraSpace);

		// テクスチャバッファに格納される際の配列の並び設定
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		// テクスチャの生成
		glGenTextures(1, &texture[0]);

		// テクスチャと変数とを関連付ける
		glBindTexture(GL_TEXTURE_2D, texture[0]);

		// テクスチャバッファを利用する際の設定(反転, クランプ, 拡大縮小など)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		// 配列の内容をテクスチャバッファに格納する実質的なコマンド. 第７引数(描画のフォーマット)をBGRAにしているところが肝.
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorImage.cols, colorImage.rows, 0, GL_BGRA, GL_UNSIGNED_BYTE, colorImage.data);

		glEnable(GL_TEXTURE_2D);

		// 下地の色とテクスチャの色の組み合わせの設定
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		// 指定した４頂点で作られた四角に, １本対角線を引くような描画方法(参考 : 床井研究室 メッシュを使った描画 http://marina.sys.wakayama-u.ac.jp/~tokoi/?date=20151125 )
		glBegin(GL_TRIANGLE_STRIP);

		for (int base = 0; base < cameraSpace.size(); base++) {
			int right = base + 1;
			int bottom = base + 512;
			int bottom_right = bottom + 1;

			if (base < cameraSpace.size() - 512) {
				// そのままメッシュにするとノイズが激しいため, 良いデータのみをflagによって選別する
				// またmapperを介して, Camera -> Colorに変換したとき座標値が負になることがあるため, それを取り除く.
				if (flagPoints[base] && flagPoints[right] && flagPoints[bottom] && flagPoints[bottom_right]
					&& (colorPoints[base].X > 0 && colorPoints[base].Y > 0) && (colorPoints[right].X > 0 && colorPoints[right].Y > 0) && (colorPoints[bottom].X > 0 && colorPoints[bottom].Y > 0) && (colorPoints[bottom_right].X > 0 && colorPoints[bottom_right].Y > 0)) {

					// glTexCoordで扱う座標は[0.0, 1.0]の範囲にしないと反復などしてしまうため, 座標を正規化.
					glTexCoord2f((colorPoints[base].X + 0.5f) / 1920, (colorPoints[base].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[base].X, cameraSpace[base].Y, cameraSpace[base].Z);
					glTexCoord2f((colorPoints[right].X + 0.5f) / 1920, (colorPoints[right].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[right].X, cameraSpace[right].Y, cameraSpace[right].Z);
					glTexCoord2f((colorPoints[bottom].X + 0.5f) / 1920, (colorPoints[bottom].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[bottom].X, cameraSpace[bottom].Y, cameraSpace[bottom].Z);
					glTexCoord2f((colorPoints[bottom_right].X + 0.5f) / 1920, (colorPoints[bottom_right].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[bottom_right].X, cameraSpace[bottom_right].Y, cameraSpace[bottom_right].Z);

				}
			}
		}

		// 三角形を２つ用いてする描画方法.
		// GL_TRIANGLE_STRIPと比べて, sparseになってしまったため, 不採用. 比較対象として残しておく.
		/*glBegin(GL_TRIANGLES);

		for (int base = 0; base < cameraSpace.size(); base++) {
			int right = base + 1;
			int bottom = base + 512;
			int bottom_right = bottom + 1;

			if (base < cameraSpace.size() - 512) {

				if (flagPoints[base] && flagPoints[right] && flagPoints[bottom]
					&& (colorPoints[base].X > 0 && colorPoints[base].Y > 0) && (colorPoints[right].X > 0 && colorPoints[right].Y > 0) && (colorPoints[bottom].X > 0 && colorPoints[bottom].Y > 0)) {

					glTexCoord2f((colorPoints[base].X + 0.5f) / 1920, (colorPoints[base].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[base].X, cameraSpace[base].Y, cameraSpace[base].Z);
					glTexCoord2f((colorPoints[right].X + 0.5f) / 1920, (colorPoints[right].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[right].X, cameraSpace[right].Y, cameraSpace[right].Z);
					glTexCoord2f((colorPoints[bottom].X + 0.5f) / 1920, (colorPoints[bottom].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[bottom].X, cameraSpace[bottom].Y, cameraSpace[bottom].Z);
				}

				if (flagPoints[base] && flagPoints[bottom] && flagPoints[bottom_right]
					&& (colorPoints[base].X > 0 && colorPoints[base].Y > 0) && (colorPoints[bottom].X > 0 && colorPoints[bottom].Y > 0) && (colorPoints[bottom_right].X > 0 && colorPoints[bottom_right].Y > 0)) {
					glTexCoord2f((colorPoints[base].X + 0.5f) / 1920, (colorPoints[base].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[right].X, cameraSpace[right].Y, cameraSpace[right].Z);
					glTexCoord2f((colorPoints[bottom].X + 0.5f) / 1920, (colorPoints[bottom].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[bottom].X, cameraSpace[bottom].Y, cameraSpace[bottom].Z);
					glTexCoord2f((colorPoints[bottom_right].X + 0.5f) / 1920, (colorPoints[bottom_right].Y + 0.5f) / 1080);
					glVertex3f(cameraSpace[bottom_right].X, cameraSpace[bottom_right].Y, cameraSpace[bottom_right].Z);
				}
			}
		}*/

		glEnd();

		glDisable(GL_TEXTURE_2D);
	}

private:
	void initializeColorFrame()
	{
		ComPtr<IColorFrameSource> colorFrameSource;
		ERROR_CHECK(kinect->get_ColorFrameSource(&colorFrameSource));
		ERROR_CHECK(colorFrameSource->OpenReader(&colorFrameReader));

		ComPtr<IFrameDescription> colorFrameDescription;
		ERROR_CHECK(colorFrameSource->CreateFrameDescription(
			ColorImageFormat::ColorImageFormat_Bgra, &colorFrameDescription));
		ERROR_CHECK(colorFrameDescription->get_Width(&colorWidth));
		ERROR_CHECK(colorFrameDescription->get_Height(&colorHeight));
		ERROR_CHECK(colorFrameDescription->get_BytesPerPixel(&colorBytesPerPixel));

		colorBuffer.resize(colorWidth * colorHeight * colorBytesPerPixel);
	}

	void initializeDepthFrame()
	{
		ComPtr<IDepthFrameSource> depthFrameSource;
		ERROR_CHECK(kinect->get_DepthFrameSource(&depthFrameSource));
		ERROR_CHECK(depthFrameSource->OpenReader(&depthFrameReader));

		ComPtr<IFrameDescription> depthFrameDescription;
		ERROR_CHECK(depthFrameSource->get_FrameDescription(&depthFrameDescription));
		ERROR_CHECK(depthFrameDescription->get_Width(&depthWidth));
		ERROR_CHECK(depthFrameDescription->get_Height(&depthHeight));

		depthBuffer.resize(depthWidth * depthHeight);
	}

	void updateColorFrame()
	{
		ComPtr<IColorFrame> colorFrame;
		auto ret = colorFrameReader->AcquireLatestFrame(&colorFrame);

		if (ret != S_OK) return;

		ERROR_CHECK(colorFrame->CopyConvertedFrameDataToArray(colorBuffer.size(), &colorBuffer[0], ColorImageFormat::ColorImageFormat_Bgra));
	}

	void updateDepthFrame() {
		ComPtr<IDepthFrame> depthFrame;
		auto ret = depthFrameReader->AcquireLatestFrame(&depthFrame);

		if (ret != S_OK) return;

		ERROR_CHECK(depthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0]));
	}
};

class MatchPoint {
public:
	cv::Mat pic1, pic2;
	std::vector<cv::KeyPoint> keypoint_pic1, keypoint_pic2; // 特徴点
	std::vector<cv::DMatch> match;

private:
	cv::Ptr<cv::AKAZE> algorithm;

	cv::Mat descriptor_pic1, descriptor_pic2;

	cv::Ptr<cv::DescriptorMatcher> matcher;
	std::vector<cv::DMatch> match_pic1to2, match_pic2to1;
	cv::Mat match_dest;

public:
	MatchPoint() {
		// 特徴点抽出に用いるアルゴリズムの選択
		algorithm = cv::AKAZE::create(); 

		matcher = cv::DescriptorMatcher::create("BruteForce");
	}
	~MatchPoint() {}

	void match_feature_point(cv::Mat _pic1, cv::Mat _pic2) {

		// Kinect-Color画像はovrvision画像に対して, 左右反転しているため,
		// Kinect-Color画像を反転させて, マッチング精度を向上させる.
		cv::flip(_pic1, pic1, 1);

		pic2 = _pic2;

		// 特徴点抽出
		algorithm->detect(pic1, keypoint_pic1);
		algorithm->detect(pic2, keypoint_pic2);

		// 特徴記述
		algorithm->compute(pic1, keypoint_pic1, descriptor_pic1);
		algorithm->compute(pic2, keypoint_pic2, descriptor_pic2);

		// マッチング
		matcher->match(descriptor_pic1, descriptor_pic2, match_pic1to2);
		matcher->match(descriptor_pic2, descriptor_pic1, match_pic2to1);

		// クロスチェック(1->2, 2->1の両方でマッチしたものだけを残して, 精度を向上させる.)
		for (size_t match_i = 0; match_i < match_pic1to2.size(); match_i++) {
			cv::DMatch forward = match_pic1to2[match_i];
			cv::DMatch backward = match_pic2to1[forward.trainIdx];

			// forward.distanceの不等式は, マッチの正確さを高めるための閾値. (精度をより高めるには値を大きく, 下げるには値を小さくする.)
			if (backward.trainIdx == forward.queryIdx && forward.distance < 300) {
				match.push_back(forward);
			}
		}
	}

	// デバッグ用
	// 比較した２枚の画像における対応点の描画.
	void show_match() {
		cv::drawMatches(pic1, keypoint_pic1, pic2, keypoint_pic2, match, match_dest);
		imshow("match_dest", match_dest);
		cv::waitKey(0);
	}
};

// TODO
// 三次元点群のメッシュとovrvisionのColor画像の間でBundle Adjustmentを実装する.

// 三次元点群とovrvision画像を合わせるための最適化
// ここではKinect, ovrvision同士のColor画像をaspect比, 平行移動量を変数として最適化しているが,
// それでは解決できないことが判明.(Color -> Cameraのポイント単位のMapperが無いことなどから)
// ceres solverの使い方含め, 中々わかりづらいため, プログラム中では使っていないがここに残しておく.

// Ovrvision, KinectのColor画像での誤差定義クラス(aspect, translation)
class ReprojectionError {
private:
	const double kinect_px;
	const double kinect_py;
	const double kinect_qx;
	const double kinect_qy;
	const double ovr_px;
	const double ovr_py;
	const double ovr_qx;
	const double ovr_qy;

public:
	ReprojectionError(double kinect_px, double kinect_py, double kinect_qx, double kinect_qy, double ovr_px, double ovr_py, double ovr_qx, double ovr_qy) :
		kinect_px(kinect_px), kinect_py(kinect_py), kinect_qx(kinect_qx), kinect_qy(kinect_qy), ovr_px(ovr_px), ovr_py(ovr_py), ovr_qx(ovr_qx), ovr_qy(ovr_qy) {}

	template <typename Type>
	bool operator()(const Type* const asp, const Type* const trans_x, const Type* const trans_y, Type* residual) const {
		// (kinect.p * asp - ovr.p) + t = kinect.q * asp - ovr.q
		residual[0] = trans_x[0] + (kinect_px * asp[0] - ovr_px) - (kinect_qx * asp[0] - ovr_qx);
		residual[1] = trans_y[0] + (kinect_py * asp[0] - ovr_py) - (kinect_qy * asp[0] - ovr_qy);
		return true;
	}
};

// 上のReprojectionErrorを最適化するためのceres solverクラス
// インクルードなどの都合上, コメントアウトしてある.
// 使う場合にはインクルード共々, コメントアウトを外す.
/*class SolveError {
private:
	double asp;
	double trans_x;
	double trans_y;

public:
	SolveError() : asp(1.0), trans_x(0.0), trans_y(0.0) {}
	~SolveError() {}

	void SolveReprojectionError(MatchPoint matchpoint) {
		ceres::Problem problem;

		for (int i = 0; i < matchpoint.match.size(); i++) {
			for (int j = i + 1; j < matchpoint.match.size(); j++) {
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1, 1, 1> // <CostFunc, residual_num, asp_num, trans_x_num, trans_y_num>
					(new ReprojectionError(matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.y,
						matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.y,
						matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.y,
						matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.y)),
					NULL, &asp, &trans_x, &trans_y);
			}
		}

		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

		//options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		Solve(options, &problem, &summary);
	}

	double getAsp() {
		return asp;
	}

	double getTrans_x() {
		return trans_x;
	}

	double getTrans_y() {
		return trans_y;
	}
};*/

// return true to retry later (e.g. after display lost)
static bool MainLoop(bool retryCreate) {
	TextureBuffer * eyeRenderTexture[2] = { nullptr, nullptr };
	DepthBuffer   * eyeDepthBuffer[2] = { nullptr, nullptr };
	ovrGLTexture  * mirrorTexture = nullptr;
	GLuint          mirrorFBO = 0;
	Scene         * roomScene = nullptr;

	OVR::OvrvisionPro ovrvision;
	int width = 0, height = 0, pixelsize = 4;
	CameraPlane camPlane;
	cv::Mat imgL, imgR, kinectColorImage;

	ovrHmd HMD;
	ovrGraphicsLuid luid;
	ovrResult result = ovr_Create(&HMD, &luid);
	if (!OVR_SUCCESS(result))
		return retryCreate;

	ovrHmdDesc hmdDesc = ovr_GetHmdDesc(HMD);

	Kinect kinect;
	kinect.initialize();

	std::vector<Vector3f> head_pos;
	std::vector<Quatf> head_ori;

	// AR-Ghost
	int rec_i = 0;
	int play_i = 0;

	// Setup Window and Graphics
	// Note: the mirror window can be any size, for this sample we use 1/2 the HMD resolution
	ovrSizei windowSize = { hmdDesc.Resolution.w / 2, hmdDesc.Resolution.h / 2 };
	if (!Platform.InitDevice(windowSize.w, windowSize.h, reinterpret_cast<LUID*>(&luid)))
		goto Done;

	// Make eye render buffers
	for (int eye = 0; eye < 2; ++eye)
	{
		ovrSizei idealTextureSize = ovr_GetFovTextureSize(HMD, ovrEyeType(eye), hmdDesc.DefaultEyeFov[eye], 1);
		eyeRenderTexture[eye] = new TextureBuffer(HMD, true, true, idealTextureSize, 1, NULL, 1);
		eyeDepthBuffer[eye] = new DepthBuffer(eyeRenderTexture[eye]->GetSize(), 0);

		if (!eyeRenderTexture[eye]->TextureSet)
		{
			if (retryCreate) goto Done;
			VALIDATE(false, "Failed to create texture.");
		}
	}

	// Create mirror texture and an FBO used to copy mirror texture to back buffer
	result = ovr_CreateMirrorTextureGL(HMD, GL_SRGB8_ALPHA8, windowSize.w, windowSize.h, reinterpret_cast<ovrTexture**>(&mirrorTexture));
	if (!OVR_SUCCESS(result))
	{
		if (retryCreate) goto Done;
		VALIDATE(false, "Failed to create mirror texture.");
	}

	// Configure the mirror read buffer
	glGenFramebuffers(1, &mirrorFBO);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTexture->OGL.TexId, 0);
	glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	ovrEyeRenderDesc EyeRenderDesc[2];
	EyeRenderDesc[0] = ovr_GetRenderDesc(HMD, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
	EyeRenderDesc[1] = ovr_GetRenderDesc(HMD, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

	// Turn off vsync to let the compositor do its magic
	wglSwapIntervalEXT(0);

	// Make scene - can simplify further if needed
	roomScene = new Scene(false);

	bool isVisible = true;

	int locationID = 0;
	OVR::Camprop cameraMode = OVR::OV_CAMVR_FULL;
	if (__argc > 2) {
		printf("Ovrvisin Pro mode changed.");
		//__argv[0]; ApplicationPath
		locationID = atoi(__argv[1]);
		cameraMode = (OVR::Camprop)atoi(__argv[2]);
	}

	if (ovrvision.Open(locationID, cameraMode)) {
		width = ovrvision.GetCamWidth();
		height = ovrvision.GetCamHeight();
		pixelsize = ovrvision.GetCamPixelsize();

		ovrvision.SetCameraSyncMode(false);
	}

	camPlane.init(width, height, pixelsize);

	// Oculus左右目のColor画像
	imgL = cv::Mat(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
	imgR = cv::Mat(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

	// Main loop
	while (Platform.HandleMessages())
	{
		// SDKに含まれていたコード 
		// Keyboard inputs to adjust player position
		//static Vector3f Pos2(0.0f, 1.6f, -5.0f);
		//if (Platform.Key['W'] || Platform.Key[VK_UP])     Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(0, 0, -0.05f));
		//if (Platform.Key['S'] || Platform.Key[VK_DOWN])   Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(0, 0, +0.05f));
		//if (Platform.Key['D'])                          Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(+0.05f, 0, 0));
		//if (Platform.Key['A'])                          Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(-0.05f, 0, 0));
		//Pos2.y = ovr_GetFloat(HMD, OVR_KEY_EYE_HEIGHT, Pos2.y);

		// Get eye poses, feeding in correct IPD offset

		// AR-Ghost, X-ray Glass等のモード選択
		if (Platform.Key['1']) {
			record_mode = true;
		}
		else if (Platform.Key['2']) {
			record_mode = false;
		}
		else if (Platform.Key['3']) {
			play_mode = true;
		}
		else if (Platform.Key['4']) {
			Xray_mode = true;
		}
		else if (Platform.Key['5']) {
			Xray_mode = false;
		}

		// 今回, Bundle Adjustmentは1フレームに対してのみ, 行うことを想定しているため, モードによってそのように制御する.
		/*else if (Platform.Key['6']) {
			BundleAdjustment_mode = true;
		}*/

		ovrVector3f ViewOffset[2] = { EyeRenderDesc[0].HmdToEyeViewOffset, EyeRenderDesc[1].HmdToEyeViewOffset };
		ovrPosef    EyeRenderPose[2];

		double           ftiming = ovr_GetPredictedDisplayTime(HMD, 0);
		// Keeping sensorSampleTime as close to ovr_GetTrackingState as possible - fed into the layer
		double           sensorSampleTime = ovr_GetTimeInSeconds();
		ovrTrackingState hmdState = ovr_GetTrackingState(HMD, ftiming, ovrTrue);
		ovr_CalcEyePoses(hmdState.HeadPose.ThePose, ViewOffset, EyeRenderPose);

		ovrvision.PreStoreCamData(OVR::Camqt::OV_CAMQT_DMSRMP);

		// AR-Ghost
		// 頭の位置, 傾き情報の記録
		if (record_mode) {
			head_pos.push_back(EyeRenderPose[0].Position);
			head_ori.push_back(EyeRenderPose[0].Orientation);
			rec_i++;
		}

		// Make Dataset
		/*if (perspective_mode && filewrite_count == 0) {
			std::ofstream oculus_pose_L("dataset\\Oculus_Pose_Left.txt");
			oculus_pose_L << "(pose.X, pose.Y, pose.Z) =" << std::endl;
			oculus_pose_L << EyeRenderPose[0].Position.x << " " << EyeRenderPose[0].Position.y << " " << EyeRenderPose[0].Position.z << std::endl;
			oculus_pose_L.close();

			std::ofstream oculus_pose_R("dataset\\Oculus_Pose_Right.txt");
			oculus_pose_R << "(pose.X, pose.Y, pose.Z) =" << std::endl;
			oculus_pose_R << EyeRenderPose[1].Position.x << " " << EyeRenderPose[1].Position.y << " " << EyeRenderPose[1].Position.z << std::endl;
			oculus_pose_R.close();

			std::ofstream oculus_ori_L("dataset\\Oculus_Ori_Left.txt");
			oculus_ori_L << "(ori.w, ori.x, ori.y, ori.z) =" << std::endl;
			oculus_ori_L << EyeRenderPose[0].Orientation.w << " " << EyeRenderPose[0].Orientation.x << " " << EyeRenderPose[0].Orientation.y << " " << EyeRenderPose[0].Orientation.z << std::endl;
			oculus_ori_L.close();

			std::ofstream oculus_ori_R("dataset\\Oculus_Ori_Right.txt");
			oculus_ori_R << "(ori.w, ori.x, ori.y, ori.z) =" << std::endl;
			oculus_ori_R << EyeRenderPose[1].Orientation.w << " " << EyeRenderPose[1].Orientation.x << " " << EyeRenderPose[1].Orientation.y << " " << EyeRenderPose[1].Orientation.z << std::endl;
			oculus_ori_R.close();
		}*/

		// X-ray Glass
		if(Xray_mode) kinect.update();

		if (isVisible)
		{
			for (int eye = 0; eye < 2; ++eye)
			{
				// Increment to use next texture, just before writing
				eyeRenderTexture[eye]->TextureSet->CurrentIndex = (eyeRenderTexture[eye]->TextureSet->CurrentIndex + 1) % eyeRenderTexture[eye]->TextureSet->TextureCount;

				// Switch to eye render target
				eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

				//Get the pose information in XM format
				ovrPosef pose = EyeRenderPose[eye];
				Matrix3f R = Matrix3f(pose.Orientation).Transposed();
				Vector3f c = pose.Position;
				Vector3f t = (R * pose.Position) * -1.0f;
				Matrix4f view = Matrix4f(
					R.M[0][0], R.M[0][1], R.M[0][2], t[0],
					R.M[1][0], R.M[1][1], R.M[1][2], t[1],
					R.M[2][0], R.M[2][1], R.M[2][2], t[2],
					0.0f, 0.0f, 0.0f, 1.0f
				);

				Matrix4f proj = ovrMatrix4f_Projection(hmdDesc.DefaultEyeFov[eye], 0.2f, 1000.0f, ovrProjection_RightHanded); // (angle, znear, zfar, 0?)

				glFrontFace(GL_CCW); // Set front

				glClearColor(0, 0, 0, 1);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				glDisable(GL_DEPTH_TEST);

				// Make Dataset
				/*if (perspective_mode && filewrite_count <= 1) {
					if (eye == 0) {
						memcpy_s(imgL.data, width * height * 4, ovrvision.GetCamImageBGRA(OVR::Cameye::OV_CAMEYE_LEFT), width * height * 4);
						//cv::imshow("imgL", imgL);
						cv::imwrite("dataset\\ovrvision_L.bmp", imgL);
					}
					else {
						memcpy_s(imgR.data, width * height * 4, ovrvision.GetCamImageBGRA(OVR::Cameye::OV_CAMEYE_RIGHT), width * height * 4);
						//cv::imshow("imgR", imgR);
						cv::imwrite("dataset\\ovrvision_R.bmp", imgR);
					}
					cv::waitKey(1);
					//cv::cvtColor(imgL, imgL, cv::COLOR_BGRA2RGB);
				}*/

				// Bundle Adjustment
				// 右目, 左目の両方で調整(bundle_countによって制御)
				/*if (BundleAdjustment_mode && bundle_count <= 1) {
					kinectColorImage = kinect.getColorImage();

					if (eye == 0) {
						memcpy_s(imgL.data, width * height * 4, ovrvision.GetCamImageBGRA(OVR::Cameye::OV_CAMEYE_LEFT), width * height * 4);
						MatchPoint matchpointL;
						matchpointL.match_feature_point(kinectColorImage, imgL);

						SolveError solve_imgL_kinect;
						solve_imgL_kinect.SolveReprojectionError(matchpointL);

						double L_optimized_asp = solve_imgL_kinect.getAsp();
						double L_optimized_transx = solve_imgL_kinect.getTrans_x();
						double L_optimized_transy = solve_imgL_kinect.getTrans_y();
					}

					else {
						memcpy_s(imgR.data, width * height * 4, ovrvision.GetCamImageBGRA(OVR::Cameye::OV_CAMEYE_RIGHT), width * height * 4);
						MatchPoint matchpointR;
						matchpointR.match_feature_point(kinectColorImage, imgR);
						
						SolveError solve_imgR_kinect;
						solve_imgR_kinect.SolveReprojectionError(matchpointR);

						double R_optimized_asp = solve_imgR_kinect.getAsp();
						double R_optimized_transx = solve_imgR_kinect.getTrans_x();
						double R_optimized_transy = solve_imgR_kinect.getTrans_y();
					}

					bundle_count++;
				}*/

				if (eye == 0)
					camPlane.draw(ovrvision.GetCamImageBGRA(OVR::Cameye::OV_CAMEYE_LEFT), width, height);
				else
					camPlane.draw(ovrvision.GetCamImageBGRA(OVR::Cameye::OV_CAMEYE_RIGHT), width, height);

				glUseProgram(0);

				glEnable(GL_DEPTH_TEST);

				glMatrixMode(GL_PROJECTION); // camera → projection
				glLoadIdentity();
				glMultTransposeMatrixf(&proj.M[0][0]);

				glMatrixMode(GL_MODELVIEW); // model → world → camera
				glLoadIdentity();
				glMultTransposeMatrixf(&view.M[0][0]);

				glPushMatrix();

				// Oculus - Kinect間での座標系変換
				glTranslated(0.05, -0.03, -1.15);
				
				// Oculus Sensor上への座標軸の描画(確認用)
				float len = 0.2f;
				glBegin(GL_LINES);
				glColor3d(1, 0, 0);
				glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(len, 0.0f, 0.0f);
				glColor3d(0, 1, 0);
				glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.0f, len, 0.0f);
				glColor3d(0, 0, 1);
				glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.0f, 0.0f, len);
				glEnd();

				glPushMatrix();

				// 三次元点群と実際の位置を調整するために, 180度回転させる.
				glRotated(180, 0, 1, 0);

				// 三次元点群の描画
				// メッシュにする場合には, kinect_draw_mesh_kinectとする.
				if(Xray_mode) kinect.draw_kinect();

				glPopMatrix();

				glPopMatrix();

				// AR-Ghost
				if (play_mode) {
					if (!head_pos.empty() && play_i <= rec_i) {
						glPushMatrix();
						glTranslated(head_pos[play_i].x, head_pos[play_i].y, head_pos[play_i].z);
						play_i++;

						// 四角錘の描画
						glRotated(-90, 1.0, 0.0, 0.0); // 四角錘の形調整(目線方向が頂点となるように)
						glBegin(GL_LINES);
						for (int i = 0; i < sizeof(edge) / (sizeof(int) * 2); ++i) {
							glColor3d(1.0, 0.0, 0.0);
							glVertex3fv(vertex[edge[i][0]]); glVertex3fv(vertex[edge[i][1]]);
						}
						glEnd();
						glPopMatrix();
					}
					else {
						play_mode = false;
						play_i = 0;
					}
				}

				// Avoids an error when calling SetAndClearRenderSurface during next iteration.
				// Without this, during the next while loop iteration SetAndClearRenderSurface
				// would bind a framebuffer with an invalid COLOR_ATTACHMENT0 because the texture ID
				// associated with COLOR_ATTACHMENT0 had been unlocked by calling wglDXUnlockObjectsNV.
				
				eyeRenderTexture[eye]->UnsetRenderSurface();
			}
		}

		// Do distortion rendering, Present and flush/sync

		// Set up positional data.
		ovrViewScaleDesc viewScaleDesc;
		viewScaleDesc.HmdSpaceToWorldScaleInMeters = 1.0f;
		viewScaleDesc.HmdToEyeViewOffset[0] = ViewOffset[0];
		viewScaleDesc.HmdToEyeViewOffset[1] = ViewOffset[1];

		ovrLayerEyeFov ld;
		ld.Header.Type = ovrLayerType_EyeFov;
		ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft; // Because OpenGL

		for (int eye = 0; eye < 2; ++eye)
		{
			ld.ColorTexture[eye] = eyeRenderTexture[eye]->TextureSet;
			ld.Viewport[eye] = Recti(eyeRenderTexture[eye]->GetSize());
			ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
			ld.RenderPose[eye] = EyeRenderPose[eye];
			ld.SensorSampleTime = sensorSampleTime;
		}

		ovrLayerHeader* layers = &ld.Header;
		ovrResult result = ovr_SubmitFrame(HMD, 0, &viewScaleDesc, &layers, 1);
		// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost
		if (!OVR_SUCCESS(result))
			goto Done;

		isVisible = (result == ovrSuccess);

		// Blit mirror texture to back buffer
		glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		GLint w = mirrorTexture->OGL.Header.TextureSize.w;
		GLint h = mirrorTexture->OGL.Header.TextureSize.h;
		glBlitFramebuffer(0, h, w, 0,
			0, 0, w, h,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		SwapBuffers(Platform.hDC);
	}

Done:
	delete roomScene;
	if (mirrorFBO) glDeleteFramebuffers(1, &mirrorFBO);
	if (mirrorTexture) ovr_DestroyMirrorTexture(HMD, reinterpret_cast<ovrTexture*>(mirrorTexture));
	for (int eye = 0; eye < 2; ++eye)
	{
		delete eyeRenderTexture[eye];
		delete eyeDepthBuffer[eye];
	}
	Platform.ReleaseDevice();
	ovr_Destroy(HMD);

	ovrvision.Close();
	FreeConsole(); // AllocConsole

	// Retry on ovrError_DisplayLost
	return retryCreate || OVR_SUCCESS(result) || (result == ovrError_DisplayLost);
}

// Solve ReprojectionError
/*void SolveReprojectionError(MatchPoint matchpoint, ceres::Solver::Summary summary) {
	double asp = 1.0;
	double trans_x = 0.0;
	double trans_y = 0.0;

	ceres::Problem problem;

	for (int i = 0; i < matchpoint.match.size(); i++) {
		for (int j = i + 1; j < matchpoint.match.size(); j++) {
			problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1, 1, 1> // <CostFunc, residual_num, asp_num, trans_x_num, trans_y_num>
				(new ReprojectionError(matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.y,
					matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.y,
					matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.y,
					matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.y)),
				NULL, &asp, &trans_x, &trans_y);
		}
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

	//options.minimizer_progress_to_stdout = true;
	Solve(options, &problem, &summary);
}*/

void init() {
	// AR-Ghost
	record_mode = false;
	play_mode = false;

	// X-ray Glass
	Xray_mode = false;

	// Bundle Adjustment
	//BundleAdjustment_mode = false;
	//bundle_count = 0;

	// Make Dataset
	//filewrite_count = 0;
}

// Display console
void DispConsole()
{
	AllocConsole();
	FILE *fp = NULL;

	freopen_s(&fp, "CONOUT$", "w", stdout);
	freopen_s(&fp, "COIN$", "r", stdin);
}

//-------------------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hinst, HINSTANCE, LPSTR, int)
{
	OVR::System::Init();
	init();

	// Initializes LibOVR, and the Rift
	ovrResult result = ovr_Initialize(nullptr);
	VALIDATE(OVR_SUCCESS(result), "Failed to initialize libOVR.");
	VALIDATE(Platform.InitWindow(hinst, L"Ovrvision Pro for OculusSDK"), "Failed to open window.");

	DispConsole();

	Platform.Run(MainLoop);

	ovr_Shutdown();

	OVR::System::Destroy();

	return(0);
}