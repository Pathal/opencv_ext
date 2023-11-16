#include "imtrans.hpp"

void cvx::imsharpen(const cv::Mat& inp, cv::Mat& dst, double ratio, cv::Size kSize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT, cv::Mat swap = cv::Mat()) {
	cv::GaussianBlur(inp, swap, kSize, sigmaX, sigmaY, borderType);
	cv::addWeighted(inp, 1.0+ratio, swap, -ratio, 0, dst);
}

void cvx::gradiant(const cv::Mat& inp, cv::Mat& dst) {
	if (dst.empty()) {
		dst = cv::Mat(inp.size(), inp.type());
	}

	cv::Mat_<float> kernel(3, 1, inp.type()); kernel << -0.5, 0.0, 0.5;
	cv::filter2D(inp(cv::Rect{ 1,1, inp.size().width - 2, inp.size().height - 2 }),
				dst(cv::Rect{ 1,1, inp.size().width - 2, inp.size().height - 2 }),
				inp.depth(),
				kernel,
				cv::Point(-1, -1),
				0,
				cv::BORDER_DEFAULT);
	// now do the borders
}
