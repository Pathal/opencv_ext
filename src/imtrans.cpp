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

cv::Mat cvx::matlab::imerode(cv::Mat& inp, cv::Mat& strel) {
	cv::Mat erosion_dst;
	cv::erode(inp, erosion_dst, strel);
	return erosion_dst;
}
cv::Mat cvx::matlab::imdilate(cv::Mat& inp, cv::Mat& strel) {
	cv::Mat dilation_dst;
	cv::dilate(inp, dilation_dst, strel);
	return dilation_dst;
}
cv::Mat cvx::matlab::imopen(cv::Mat& inp, cv::Mat& strel) {
	cv::Mat t = imerode(inp, strel);
	return imdilate(t, strel);
}
cv::Mat cvx::matlab::imclose(cv::Mat& inp, cv::Mat& strel) {
	cv::Mat t = imdilate(inp, strel);
	return imerode(t, strel);
}

// Common Section

void cvx::common::clamp(cv::Mat& inp, cv::Scalar minVal, cv::Scalar maxVal) {
	cv::min(inp, maxVal, inp);
	cv::max(inp, minVal, inp);
}

void cvx::common::clamp(cv::Mat& inp, cv::Mat& dest, cv::Scalar minVal, cv::Scalar maxVal) {
	cv::min(inp, maxVal, dest);
	cv::max(dest, minVal, dest);
}
