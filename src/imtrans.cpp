#include "imtrans.hpp"

#include <iostream>

#include "core.hpp"

void cvx::matlab::imsharpen(const cv::Mat& inp, cv::Mat& dst, double ratio, cv::Size kSize, double sigmaX, double sigmaY, int borderType, cv::Mat swap) {
	cv::GaussianBlur(inp, swap, kSize, sigmaX, sigmaY, borderType);
	cv::addWeighted(inp, 1.0+ratio, swap, -ratio, 0, dst);
}
cv::Mat cvx::matlab::imsharpen(const cv::Mat& inp, double ratio, cv::Size kSize, double sigmaX, double sigmaY, int borderType, cv::Mat swap) {
	cv::Mat dst;
	cvx::matlab::imsharpen(inp, dst, ratio, kSize, sigmaX, sigmaY, borderType, swap);
	return dst;
}
