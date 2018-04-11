#include "ceres/ceres.h"
#include "glog/logging.h"

#include <windows.h>

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
	ErrorFunction() : p1_x(1.0), p1_y(1.0), p2_x(4.0), p2_y(5.0), q1_x(2.0), q1_y(1.0), q2_x(10.0), q2_y(7.0) {}

	template <typename Type>
	bool operator()(const Type* const asp, Type* residual) const {
		Type p_dx = Type(p1_x) - Type(p2_x); 
		Type p_dy = Type(p1_y) - Type(p2_y);
		Type q_dx = Type(q1_x) - Type(q2_x);
		Type q_dy = Type(q1_y) - Type(q2_y);
		residual[0] = asp[0] * sqrt(p_dx * p_dx + p_dy * p_dy) - sqrt(q_dx * q_dx + q_dy * q_dy);
		return true;
	}
};

void DispConsole() {
	AllocConsole();
	FILE *fp = NULL;

	freopen_s(&fp, "CONOUT$", "w", stdout);
	freopen_s(&fp, "CONIN$", "r", stdin);
}

double SolveResidual(double initial_asp) {
	double asp = initial_asp;

	ceres::Problem problem;

	ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ErrorFunction, 1, 1>(new ErrorFunction);
	problem.AddResidualBlock(cost_function, NULL, &asp);

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	std::cout << "asp:" << initial_asp << "->" << asp << std::endl;

	return asp;
}

int main(int argc, char *argv[]) {
	DispConsole();

	google::InitGoogleLogging(argv[0]);

	double initial_asp = 1.0;
	double asp = SolveResidual(initial_asp);

	while (1)
		Sleep(1);

	FreeConsole();

	return 0;
}