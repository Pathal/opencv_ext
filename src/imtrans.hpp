#pragma once

#include <array>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx {
	namespace matlab {
		void imsharpen(const cv::Mat& inp, cv::Mat& dst, double ratio, cv::Size kSize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT, cv::Mat swap = cv::Mat());
		cv::Mat imsharpen(const cv::Mat& inp, double ratio, cv::Size kSize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT, cv::Mat swap = cv::Mat());

	}
};
