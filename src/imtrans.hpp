#pragma once

#include <array>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx {
	void imsharpen(const cv::Mat& inp, cv::Mat& dst, double ratio, cv::Size kSize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT, cv::Mat swap = cv::Mat());
	cv::Mat imsharpen(const cv::Mat& inp, double ratio, cv::Size kSize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT, cv::Mat swap = cv::Mat());

	void gradiant1D(const cv::Mat& inp, cv::Mat& dst);
	cv::Mat gradiant1D(const cv::Mat& inp);

	void gradiant2D(const cv::Mat& inp, cv::Mat& gx, cv::Mat& gy);
	std::array<cv::Mat, 2> gradiant2D(const cv::Mat& inp);

	float getPixelAsFloat(const cv::Mat& inp, int x, int y);
};
