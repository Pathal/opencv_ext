#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx {
	void imsharpen(const cv::Mat& inp, cv::Mat& dst, double ratio, cv::Size kSize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT, cv::Mat swap = cv::Mat());

	void gradiant(const cv::Mat& inp, cv::Mat& dst);
};
