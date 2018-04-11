#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include <iostream>
#include <windows.h>

class MatchPoint {
public:
	cv::Mat pic1, pic2;
	std::vector<cv::KeyPoint> keypoint_pic1, keypoint_pic2;
	std::vector<cv::DMatch> match;

private:
	cv::Ptr<cv::AKAZE> algorithm;

	cv::Mat descriptor_pic1, descriptor_pic2;

	cv::Ptr<cv::DescriptorMatcher> matcher;
	std::vector<cv::DMatch> match_pic1to2, match_pic2to1;
	cv::Mat match_dest;

public:
	MatchPoint() {
		algorithm = cv::AKAZE::create();
		matcher = cv::DescriptorMatcher::create("BruteForce");
	}
	~MatchPoint() {}

	void match_feature_point(cv::String pic1_path, cv::String pic2_path) {
		cv::Mat _pic1 = cv::imread(pic1_path, cv::IMREAD_COLOR); // Flip horizontal
		cv::flip(_pic1, pic1, 1);
		pic2 = cv::imread(pic2_path, cv::IMREAD_COLOR);

		algorithm->detect(pic1, keypoint_pic1);
		algorithm->detect(pic2, keypoint_pic2);

		algorithm->compute(pic1, keypoint_pic1, descriptor_pic1);
		algorithm->compute(pic2, keypoint_pic2, descriptor_pic2);

		matcher->match(descriptor_pic1, descriptor_pic2, match_pic1to2);
		matcher->match(descriptor_pic2, descriptor_pic1, match_pic2to1);

		for (size_t match_i = 0; match_i < match_pic1to2.size(); match_i++) {
			cv::DMatch forward = match_pic1to2[match_i];
			cv::DMatch backward = match_pic2to1[forward.trainIdx];

			// forward.distance : matching accuracy
			if (backward.trainIdx == forward.queryIdx && forward.distance < 300) {
				match.push_back(forward);
			}
		}
	}

	void draw_show_match(cv::String match_dest_path) {
		cv::drawMatches(pic1, keypoint_pic1, pic2, keypoint_pic2, match, match_dest);

		cv::imwrite(match_dest_path, match_dest);
		cv::imshow("match_dest", match_dest);
		cv::waitKey(0);
	}
};

//class ErrorFunction_asp {
//private:
//	const double p1_x;
//	const double p1_y;
//	const double p2_x;
//	const double p2_y;
//	const double q1_x;
//	const double q1_y;
//	const double q2_x;
//	const double q2_y;
//
//public:
//	ErrorFunction_asp(double p1_x, double p1_y, double p2_x, double p2_y, double q1_x, double q1_y, double q2_x, double q2_y) : p1_x(p1_x), p1_y(p1_y), p2_x(p2_x), p2_y(p2_y), q1_x(q1_x), q1_y(q1_y), q2_x(q2_x), q2_y(q2_y) {}
//
//	template <typename Type>
//	bool operator()(const Type* const asp, Type* residual) const {
//		Type p_dx = Type(p1_x) - Type(p2_x);
//		Type p_dy = Type(p1_y) - Type(p2_y);
//		Type q_dx = Type(q1_x) - Type(q2_x);
//		Type q_dy = Type(q1_y) - Type(q2_y);
//		residual[0] = asp[0] * sqrt(p_dx * p_dx + p_dy * p_dy) - sqrt(q_dx * q_dx + q_dy * q_dy);
//		return true;
//	}
//};

class ErrorFunction {
private:
	const double p1_x;
	const double p1_y;
	const double p2_x;
	const double p2_y;
	const double q1_x;
	const double q1_y;
	const double q2_x;
	const double q2_y;

public:
	ErrorFunction(double p1_x, double p1_y, double p2_x, double p2_y, double q1_x, double q1_y, double q2_x, double q2_y) : p1_x(p1_x), p1_y(p1_y), p2_x(p2_x), p2_y(p2_y), q1_x(q1_x), q1_y(q1_y), q2_x(q2_x), q2_y(q2_y) {}

	template <typename Type>
	bool operator()(const Type* const asp, const Type* const t_x, const Type* const t_y, Type* residual) const {
		residual[0] = t_x[0] + (p1_x * asp[0] - q1_x) - (p2_x * asp[0] - q2_x);
		residual[1] = t_y[0] + (p1_y * asp[0] - q1_y) - (p2_y * asp[0] - q2_y);
		return true;
	}
};

cv::String kinect_color_path = "..\\Oculus_Kinect\\dataset\\kinect_color.bmp";
cv::String ovrvision_L_path = "..\\Oculus_Kinect\\dataset\\ovrvision_L.bmp";
cv::String ovrvision_R_path = "..\\Oculus_Kinect\\dataset\\ovrvision_R.bmp";

cv::String match_kinect_ovrvisionL_path = "output\\match_kinect_ovrvisionL.bmp";
cv::String match_kinect_ovrvisionR_path = "output\\match_kinect_ovrvisionR.bmp";

void DispConsole() {
	AllocConsole();
	FILE *fp = NULL;

	freopen_s(&fp, "CONOUT$", "w", stdout);
	freopen_s(&fp, "CONIN$", "r", stdin);
}

void SolveResidual(double initial_asp, double initial_t_x, double initial_t_y, MatchPoint matchpoint) {
	double asp = initial_asp;
	double t_x = initial_t_x;
	double t_y = initial_t_y;

	ceres::Problem problem;

	for (int i = 0; i < matchpoint.match.size(); i++) {
		for (int j = i + 1; j < matchpoint.match.size(); j++) {
			problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ErrorFunction, 2, 1, 1, 1> // <CostFunc, residual_num, asp_num, t_x_num, t_y_num>
				(new ErrorFunction(matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.y,
					matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.y,
					matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.y,
					matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.y)),
				NULL, &asp, &t_x, &t_y);
		}
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	std::cout << "asp:" << initial_asp << "->" << asp << std::endl;
	std::cout << "t_x:" << initial_t_x << "->" << t_x << std::endl;
	std::cout << "t_y:" << initial_t_y << "->" << t_y << std::endl;
}

//double SolveResidual_asp(double initial_asp, MatchPoint matchpoint) {
//	double asp = initial_asp;
//	
//	ceres::Problem problem;
//
//	for (int i = 0; i < matchpoint.match.size(); i++) {
//		for (int j = i + 1; j < matchpoint.match.size(); j++) {
//			problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ErrorFunction_asp, 1, 1>
//				(new ErrorFunction_asp(matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[i].queryIdx].pt.y,
//					matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.x, matchpoint.keypoint_pic1[matchpoint.match[j].queryIdx].pt.y,
//					matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[i].trainIdx].pt.y,
//					matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.x, matchpoint.keypoint_pic2[matchpoint.match[j].trainIdx].pt.y)),
//				NULL, &asp);
//		}
//	}
//
//	ceres::Solver::Options options;
//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//
//	options.minimizer_progress_to_stdout = true;
//	ceres::Solver::Summary summary;
//	Solve(options, &problem, &summary);
//
//	std::cout << summary.BriefReport() << std::endl;
//	std::cout << "asp:" << initial_asp << "->" << asp << std::endl;
//	
//	return asp;
//}

int main(int argc, char *argv[]) {
	DispConsole();

	MatchPoint matchpointL;
	//MatchPoint matchpointR;

	matchpointL.match_feature_point(kinect_color_path, ovrvision_L_path);
	//matchpointR.match_feature_point(kinect_color_path, ovrvision_R_path);

	//matchpointL.draw_show_match(match_kinect_ovrvisionL_path);
	//matchpointR.draw_show_match(match_kinect_ovrvisionR_path);

	google::InitGoogleLogging(argv[0]);

	double initial_asp = 1.0;
	double initial_t_x = 0.0;
	double initial_t_y = 0.0;
	//double asp = SolveResidual_asp(initial_asp, matchpointL);
	SolveResidual(initial_asp, initial_t_x, initial_t_y, matchpointL);

	while (1)
		Sleep(1);

	FreeConsole();

	return 0;
}